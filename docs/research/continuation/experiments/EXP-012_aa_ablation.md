---
id: EXP-012
schema_version: 1
updated_at: 2026-09-11
status: completed_numeric_verified
evidence: local_final_json_per_cell_metrics_diagnostics_and_operator_verifications
protocol: P-FULL-R20-BN
source_json: ../sources/ablation_aa_results.json
code_commit_reported: null
code_commit_inspected: true
code_commit_inspected_on: 2026-09-10
code_commit_inspected_note: code écrit et exécuté dans la même session ; non commité au moment de la fiche
---

# EXP-012 — Le flou interne est-il de l'anti-aliasing ? Placement, masques, annelage, profondeur et a priori sur sigma

[Index](INDEX.md) · [JSON des 32 configurations](../sources/ablation_aa_results.json) · [Lecture actuelle](../CURRENT_STATE.md) · [Opérateurs](../OPERATORS.md) · [Protocoles](../PROTOCOLS.md)

## 1. Question de départ et portée

EXP-008 et EXP-011 établissent que le Gaussian interne annelé vaut environ +3 points. **Aucune n'établit pourquoi.** Cette expérience part d'une hypothèse mécanique de l'utilisateur : le flou pourrait agir comme un **anti-aliasing**, c'est-à-dire un passe-bas placé avant un sous-échantillonnage pour empêcher le repliement de spectre.

Deux hypothèses concurrentes sont mises en compétition tout au long :

* **H — anti-aliasing.** Le flou aide parce qu'il filtre avant une décimation.
* **H' — contrainte de régularité annelée.** Il aide parce qu'il contraint la classe de fonctions à être lisse à toutes les profondeurs, contrainte ensuite relâchée.

La question a produit, par enchaînement, cinq vagues et **58 cellules sur 32 configurations**. Le profil `P-FULL-R20-BN` est repris sans modification : 50k train / 10k test, 30 époques, 391 updates/époque, ResNet-20 BN corrigé, batch 128 en 4×32, LR pic 0,005, warmup 60 puis cosinus, sans augmentation. Comparaison finale **époque 30**.

**Avertissement d'appariement.** Les poids initiaux d'EXP-008 à EXP-011 sont irrécupérables : ils étaient gitignorés et le dataset Kaggle qui les portait appartient à des comptes non configurés ici. Un jeu d'assets neuf a été généré et épinglé. **Aucune cellule de cette fiche ne doit être comparée à celles d'EXP-011.** Les 58 sont en revanche mutuellement appariées. Détail : les tableaux NumPy (`subset`, `train_probe`, `perm_seed0`) reproduisent bit à bit ceux de la campagne historique — PCG64 est indépendant de la version — seuls les poids torch diffèrent.

## 2. Enchaînement logique des cinq vagues

Chaque vague découle d'un résultat de la précédente. Cet ordre est le contenu de l'expérience autant que les chiffres.

| Vague | Ce qui l'a déclenchée | Ce qu'elle teste | Cellules |
|---|---|---|---:|
| 1 | La question initiale sur l'anti-aliasing | Placement, masques, sigma constant, BlurPool, à 32×32 | 12 |
| 2 | Le placement post-ReLU gagne (+1,71 pp) ; reste à savoir s'il survit à la résolution | La même question avec la résolution progressive | 6 |
| 3 | Réduire après le stem bat réduire l'image (+1,03 pp) ; jusqu'où descendre ? | Profondeur du point de réduction, × flou | 6 |
| 4 | Les six meilleurs bras tiennent dans 0,47 point à une graine | Trois graines sur les quatre premiers | 8 |
| 5 | Sigma est en pixels d'une carte qui rétrécit avec la profondeur | Quatre a priori de profondeur sur sigma, deux architectures | 26 |

## 3. Mesures analytiques, sans entraînement

Ces mesures ont orienté le protocole et sont reproductibles sans GPU. Elles constituent une part substantielle du résultat.

### 3.1 Le Gaussian commute avec tout sauf le ReLU

Le Gaussian est une convolution spatiale linéaire par canal, de noyau de somme 1. Mesuré sur `resnet20_bn_cifar` :

| identité testée | écart max |
|---|---:|
| `G(BN(u))` contre `BN(G(u))` | 7,15·10⁻⁷ |
| `G(conv(u))` contre `conv(G(u))` (intérieur) | 8,94·10⁻⁷ |
| `G(ReLU(u))` contre `ReLU(G(u))` | **1,02** |

BN à l'inférence est une affine par canal, et `G` préserve les constantes, donc `G(au+b) = aG(u)+b` exactement. **Conséquence :** à l'intérieur d'une portion linéaire du réseau, l'emplacement du Gaussian n'existe pas. Le seul choix réel est **de quel côté du ReLU** il se place. Les placements `conv_out` et `post_bn` sont donc le même opérateur, à la seule différence des statistiques que BN accumule.

### 3.2 Le ReLU est un générateur d'harmoniques

`ReLU(u) = ½(u + |u|)` ; le terme `|u|` crée un point anguleux à chaque passage par zéro, dont le spectre décroît en 1/f². Sur un ton pur de fréquence 3 : fréquences présentes **{3}** avant, **{0, 3, 6, 12, 18}** après.

Conséquence mesurée sur le réseau entraîné (`C_plain`, fraction d'énergie |f|>0,25 entrant dans la convolution suivante) :

| σ | flou **avant** BN+ReLU | flou **après** ReLU | sans flou |
|---|---:|---:|---:|
| 0,5 | 0,2243 | 0,1649 | 0,3199 |
| 0,8 | 0,0890 | **0,0356** | 0,3199 |

À σ=0,8 le ReLU regénère **2,5×** l'énergie haute fréquence que le flou venait de retirer. C'est l'argument de Zhang (2019) vérifié numériquement sur ce réseau.

### 3.3 Les quatre décimations du réseau et leur absence de préfiltre

`blocks[3].conv1` et `blocks[6].conv1` sont en `stride=2` ; `blocks[3].shortcut` et `blocks[6].shortcut` font `x[:, :, ::2, ::2]`. Ces quatre opérations décimaient **sans aucun préfiltre**, et les shortcuts option-A ne sont jamais hookés par `attach_sites`. Mesure sur bruit structuré, énergie au-dessus de la Nyquist post-décimation :

| opération 32→16 | énergie repliée |
|---|---:|
| entrée non filtrée | 0,178 |
| `x[:,:,::2,::2]` (shortcut option-A) | **0,171** |
| bilinéaire sans antialias | 0,053 |
| bilinéaire **avec** antialias (chemin du dépôt) | **0,015** |
| Gaussian σ=0,5 puis `::2` | 0,081 |

**Le dépôt était très soigneux sur l'entrée et négligent à l'intérieur.** `antialias=True` n'existe qu'à deux endroits, tous deux dans la réduction de résolution (`pipeline.resize_unit_float`, `campaign_ops.reduce_spatial`).

### 3.4 Sigma en pixels d'une carte qui rétrécit

À σ=1 avec le noyau interne (rayon 4, 9 taps) :

| carte | support/carte | part de padding miroir | atténuation par pixel |
|---|---:|---:|---:|
| 32×32 (positions 0-3) | 0,28 | 2 % | 0,080 |
| 16×16 (positions 4-6) | 0,56 | 5 % | 0,080 |
| 8×8 (positions 7-9) | **1,12** | 9 % | 0,080 |
| 4×4 (phase réduite) | 2,25 | **18 %** | — |

**Point à ne pas confondre :** l'atténuation d'énergie par pixel est **identique** aux trois étages. Le σ uniforme ne retire donc pas plus d'information en profondeur ; il cesse d'être un filtre local bien posé et devient une moyenne globale contaminée par les bords. Cette distinction a été énoncée à tort dans un premier temps puis corrigée par la mesure.

## 4. Les six groupes et leurs cellules

Toutes à `P-FULL-R20-BN`, assets épinglés communs. `n` = graines terminées.

| Groupe | Question | Configurations | Cellules |
|---|---|---:|---:|
| A | Où placer le flou dans le bloc | 3 | 5 |
| B | Quels sites filtrer | 2 | 2 |
| C | L'annelage sert-il à quelque chose | 5 | 5 |
| D | Un vrai anti-aliasing fixe suffit-il | 2 | 2 |
| E | Résolution : lieu et profondeur de l'unique réduction | 11 | 19 |
| F | Un a priori de profondeur sur sigma | 8 | 24 |
| — | Témoin sans rien | 1 | 1 |
| **Total** | | **32** | **58** |

Le témoin `C_plain` (0,7506, une graine) est la référence de tous les gains absolus cités.

## 5. Groupe A — le placement : un seul choix existe

| bras | placement | n | accuracy | vs `C_plateau` |
|---|---|---:|---:|---:|
| `C_plateau` | sortie de conv, avant BN et ReLU (placement historique, CBS) | 1 | 0,7818 | — |
| `P_postbn` | après BN, avant ReLU | 1 | 0,7839 | +0,21 |
| `P_postblock` | **après le ReLU**, 10 positions | 3 | **0,8008** ± 0,0021 | +1,71 (graine 0) |

Le résultat suit exactement la prédiction de §3.1 : passer de `conv_out` à `post_bn` ne change **rien** (+0,21 pp, une graine, dans la bande indécidable), parce que ce sont le même opérateur. Franchir le ReLU vaut **+1,71 pp**.

**Ce n'est pas « on a essayé trois endroits et le troisième marche mieux ». Il n'y avait que deux endroits possibles, et le dépôt avait choisi le mauvais depuis EXP-004.**

Réserve : `post_block` a 10 positions au lieu de 19, et §3.2 montre qu'à σ nominal égal il délivre 2,5× plus de régularité effective. Les +1,71 pp peuvent donc venir du placement *ou* d'un dosage effectif plus élevé. Le contrôle qui séparerait les deux — `conv_out` à σ intermédiaire — n'a pas été exécuté.

## 6. Groupe B — les sites : l'effet n'est pas localisé

| bras | sites filtrés | n | accuracy | vs témoin |
|---|---|---:|---:|---:|
| `M_predown` | {6, 12} seulement, les plus proches d'une décimation | 1 | 0,7755 | +2,49 |
| `M_nodown` | les 17 autres | 1 | 0,7852 | +3,46 |

Sous H stricte, `M_nodown` aurait dû ne rien donner. **Il récupère l'effet complet sans toucher un seul site pré-décimation.**

Limite du découpage, à consigner : `predown`/`nodown` suppose une localité que la cascade n'a pas. Lisser au site 3 réduit encore ce qui arrive au site 7. Le masque teste la localité, pas le mécanisme. **Cette question n'est probablement pas décidable par des masques de sites dans un réseau profond.**

## 7. Groupe C — l'annelage n'achète pas de l'accuracy, il achète l'architecture cible

| bras | σ | n | accuracy (chemin **courant**) | chemin cible | chute |
|---|---|---:|---:|---:|---:|
| `K_const030` | 0,30 constant | 1 | 0,7497 | 0,7433 | 0,6 pp |
| `K_const050` | 0,50 constant | 1 | **0,7869** | 0,1394 | **64,8 pp** |
| `K_const080` | 0,80 constant | 1 | 0,7502 | 0,1000 | 65,0 pp |
| `K_const100` | 1,00 constant | 1 | 0,7115 | 0,0998 | 61,2 pp |
| `R6` | 0,50 constant, placement post-ReLU | 1 | 0,7978 | 0,2481 | 55,0 pp |

Deux lectures, à ne pas confondre.

**Sur l'accuracy, l'annelage n'apporte rien.** `K_const050` (0,7869) dépasse `C_plateau` (0,7818) de +0,51 pp, et `R6` (0,7978) est à −0,11 pp de `P_postblock` à la graine 0. Aux deux placements, un sigma constant fait aussi bien que le calendrier annelé. C'est le contrôle qui manquait au dépôt depuis EXP-004 : les +3 points n'avaient jamais été décomposés en « présence du filtre » et « parcours du chemin ».

**Mais un bras à sigma constant n'est pas un ResNet-20.** Il s'effondre au hasard dès qu'on retire son filtre. Il traîne ses couches gaussiennes à l'inférence pour toujours et ne résout pas le problème cible — il résout un problème voisin. Les bras annelés affichent **0,00 pp** d'écart entre chemin courant et chemin cible.

> **Formulation à retenir :** l'annelage coûte environ 0,5 point d'accuracy et achète l'architecture cible. C'est exactement ce à quoi sert une homotopie — non pas scorer plus haut, mais arriver au problème qu'on voulait résoudre. Une évaluation qui ne regarde que l'accuracy manque le livrable.

**Dose-réponse.** 0,3 → rien, 0,5 → +3,63, 0,8 → rien, 1,0 → −3,91. Courbe à un seul sommet, sommet à σ≈0,5. Une signature de Nyquist a été invoquée puis **retirée** : toute intervention paramétrée par une intensité a un optimum ; il faudrait montrer que l'optimum se déplace quand le facteur de décimation change, ce qui n'a pas été testé.

## 8. Groupe D — un anti-aliasing fixe suffit presque

| bras | n | accuracy | temps | vs témoin |
|---|---:|---:|---:|---:|
| `B_blurpool` — σ=0,5 **fixe**, 2 positions (entrées de `blocks[3]` et `blocks[6]`) | 1 | 0,7851 | 460 s | +3,45 |
| `B_blurpool_plateau` — le précédent **plus** le flou annelé sur 19 sites | 1 | 0,7919 | 698 s | +4,13 |

Un préfiltre **fixe, à deux positions, jamais annelé** dépasse le Gaussian annelé sur 19 sites (+3,12 pp) pour **1,09× le temps du témoin au lieu de 1,53×** — environ cinq fois plus efficace au point de pourcentage.

Un seul hook `forward_pre_hook` sur `blocks[3]` et `blocks[6]` préfiltre les **quatre** décimations à la fois, car la conv stridée et le shortcut décimant consomment le même tenseur.

**Dissociation à noter.** `B_blurpool` ajuste **mieux** l'entraînement que le témoin (probe CE 0,196 contre 0,249) *et* généralise mieux : signature de conditionnement, pas de régularisation. `C_plateau` fait l'inverse (0,398). Deux mécanismes distincts atteignent une accuracy voisine par des routes opposées.

## 9. Groupe E — la résolution domine, et son lieu compte

### 9.1 Lieu de la réduction

| bras | réduction | flou | n | accuracy | temps |
|---|---|---|---:|---:|---:|
| `R1` | sur l'image, bilinéaire+antialias | — | 1 | 0,7892 | 443 s |
| `R2` | idem | ancien placement | 1 | 0,7915 | 604 s |
| `R3` | idem | **post-ReLU** | 1 | 0,8014 | 542 s |
| `R4` | après le stem, max-pooling | — | 1 | 0,7995 | 437 s |
| `R5` | après le stem | post-ReLU | 1 | 0,7965 | 536 s |

La résolution seule vaut **+3,86 pp** — plus que le flou à l'ancien placement. `R3 − R2 = +0,99 pp` : le nouveau placement garde son avantage avec la résolution, mais réduit (il valait +1,71 à résolution constante).

**Écrasement.** L'apport du flou tombe de +3,12 / +4,83 (à 32×32, ancien/nouveau placement) à **+0,23 / +1,22** une fois la résolution en place. Les deux leviers font largement la même chose.

**Confondant non levé.** `R1` utilise `input_bilinear` (bilinéaire + antialias) alors que `R4` et toute la série de profondeur utilisent `*_max` (max-pooling, aucun antialiasing). Le passage image → stem change donc **deux facteurs**. EXP-011 donne un ordre de grandeur (+0,32 pp pour max à position fixée sur l'image) mais sur d'autres poids, donc non soustractible. Les comparaisons **entre positions internes** sont propres, toutes en max-pooling.

### 9.2 Profondeur de la réduction

Seules **cinq positions** ont une carte 32×32 (image, stem, sorties de `blocks[0..2]`). Au-delà, `blocks[3]` a déjà divisé par deux : r=16 y serait un no-op et r=24 un upsampling. Un mode **relatif** (facteur 0,5 / 0,75 / 1 appliqué à la taille locale) a été implémenté pour lever cette limite ; il reproduit exactement l'absolu aux positions 32×32.

| position | sans flou | n | avec flou post-ReLU | n |
|---|---:|---:|---:|---:|
| sur l'image | 0,7892 | 1 | 0,8014 | 1 |
| après le stem | 0,7995 | 1 | 0,7965 | 1 |
| après `blocks[0]` | 0,8044 ± 0,0043 | 3 | 0,8053 | 1 |
| après `blocks[1]` | **0,8059** ± 0,0041 | 3 | 0,8008 | 1 |
| après `blocks[2]` | 0,8009 ± 0,0017 | 3 | **0,8099** ± 0,0041 | 3 |

Réduire **à l'intérieur** bat réduire l'image. Entre positions internes, c'est un **plateau** : les quatre tiennent dans 0,5 point.

Le coût ne dépend pas de la profondeur — 428 à 455 s sur les cinq positions sans flou. Une estimation de +20 à 25 % avait été annoncée et s'est révélée fausse (+1 % mesuré) : le stage 1 n'a que 16 canaux, sa part de temps réel est très inférieure à sa part de FLOPs. Même piège que C-18.

## 10. Groupe F — a priori de profondeur sur sigma : résultat négatif propre

Quatre profils, normalisés sur le **maximum** (et non sur le budget, ce qui aurait forcé A1 à 1,60 en surface, au-delà de `sigma_max`, changeant le support du noyau partout). Le multiplicateur s'applique **par-dessus** `q`, si bien qu'un profil calculé sur les tailles naturelles fait suivre à sigma la taille **courante**.

```
position       0     1     2     3     4     5     6     7     8     9
carte         32    32    32    32    16    16    16     8     8     8
A2  ~ √H     1,00  1,00  1,00  1,00  0,71  0,71  0,71  0,50  0,50  0,50
A1  ~ H      1,00  1,00  1,00  1,00  0,50  0,50  0,50  0,25  0,25  0,25
A3  ~ 1/H    0,25  0,25  0,25  0,25  0,50  0,50  0,50  1,00  1,00  1,00
A4  ~ RF     0,09  0,22  0,34  0,47  0,66  0,91  1,00  1,00  1,00  1,00
```

Deux familles : **Q** à 32×32 sans réduction (l'a priori seul), **P** avec la réduction après `blocks[2]` (survit-il ?). Trois graines partout.

| profil | Q — sans réduction | vs témoin | P — avec réduction | vs `D2G` |
|---|---:|---:|---:|---:|
| uniforme | **0,8008** | — | **0,8099** | — |
| A2 ~ √H | 0,7955 | −0,53 · 3/3 | 0,8028 | −0,72 · 3/3 |
| A1 ~ H | 0,7848 | −1,61 · 3/3 | 0,7961 | −1,38 · 3/3 |
| A3 ~ 1/H | 0,7701 | −3,08 · 3/3 | 0,8018 | −0,81 · 3/3 |
| A4 ~ RF | 0,7959 | −0,50 · 3/3 | 0,8114 | +0,15 · **signe variable** |

**Aucun a priori ne bat le profil plat.** Sept comparaisons sur huit sont négatives avec le même signe sur les trois graines ; la huitième change de signe.

**L'échec est ordonné :** plus on s'écarte du plat, plus on perd. Et **les deux directions perdent** — A3 (plus de flou en profondeur) et A1 (moins) sont des hypothèses opposées, toutes deux battues. Ce n'est donc pas « il fallait aller dans l'autre sens », c'est « il ne fallait pas s'écarter du plat ».

**Asymétrie révélatrice.** A4 retire le flou en **surface** (positions 0-5) et coûte −0,50 ; A1 le retire en **profondeur** (4-9) et coûte −1,61. Retirer le flou des couches profondes coûte trois fois plus cher — l'inverse exact de l'hypothèse qui a motivé A1. Réserve : A4 et A1 ne diffèrent pas seulement par *où* ils retirent mais aussi par *combien* (budget 6,69 contre 6,25) et par la forme du profil.

**Explication la plus probable de l'échec de A1/A2 :** ils n'atténuent pas le flou en profondeur, ils **l'éteignent**. Sous A1 avec réduction, σ tombe à 0,25 et 0,125, soit 99,7 % et 100 % d'énergie retenue — le noyau est numériquement une delta. C'est un masque, pas un rééquilibrage, et le masque `early7` d'EXP-011 avait déjà montré que restreindre le flou aux sites superficiels est le pire choix.

**La réduction ne change pas le verdict**, elle amortit les écarts (étalement 4,5 points dans Q contre 1,6 dans P). L'a priori échoue seul *et* échoue combiné : ce n'est pas un problème de redondance avec la réduction.

## 11. Ce que trois graines changent — et une erreur de méthode

Quatre bras rejoués aux graines 0/1/2 :

| bras | par graine | étendue |
|---|---|---:|
| `D2G` | 0,8054 · 0,8109 · 0,8135 | 0,81 pt |
| `D1` | 0,8042 · 0,8029 · 0,8106 | 0,77 pt |
| `D0` | 0,8007 · 0,8034 · 0,8091 | 0,84 pt |
| `D2` | 0,7999 · 0,8000 · 0,8029 | 0,30 pt |

La dispersion **non appariée** vaut 0,36 point, celle des **différences appariées** 0,26. Surtout, les graines déplacent les bras **ensemble** : la graine 2 relève les quatre.

```
D2G - D2  = +0,90   [+0,55  +1,09  +1,06]   même signe
D2G - D0  = +0,55   [+0,47  +0,75  +0,44]   même signe
D2G - D1  = +0,40   [+0,12  +0,80  +0,29]   même signe
D1  - D0  = +0,15   [+0,35  -0,05  +0,15]   signe variable
```

**Erreur de méthode commise et corrigée.** À une graine, `D2G − D1` valait +0,12 point — soit 12 images de test sur 10 000 — et a été déclaré « du bruit », avec recommandation de choisir `D1` pour son coût. C'était comparer un **écart apparié** à une **dispersion non appariée** : deux échelles différentes. Apparié sur trois graines, l'écart est réel et consistant. Voir [CORRECTIONS](../CORRECTIONS.md) C-36.

## 12. Diagnostics sans entraînement supplémentaire

Exécutés en fin de chaque run, sur les poids finaux.

**0a — énergie repliée** à l'entrée des blocs décimants (fraction de |DFT|² avec |f|>0,25 c/éch) :

| bras | blocks[3] | blocks[6] |
|---|---:|---:|
| `C_plain` | 0,171 | 0,197 |
| `C_plateau` | 0,084 | 0,129 |
| `P_postbn` | 0,079 | 0,124 |
| `P_postblock` | 0,083 | 0,130 |
| `B_blurpool` | 0,094 | 0,087 |

Il y a bien du repliement à prévenir chez le témoin (17-20 %). Mais **les trois placements convergent vers la même énergie résiduelle** alors que leur accuracy s'étale sur 1,7 point. **Accuracy et repliement résiduel sont découplés d'un placement à l'autre** — l'argument le plus gênant pour H.

**0b — consistance au décalage** de 1 pixel : 0,804 chez le témoin, 0,85 à 0,90 chez tous les bras bénéfiques. Mais `K_const100` a la meilleure consistance (0,913) et la pire accuracy (−3,91 pp) : on peut en acheter trop.

## 13. Coût

58 cellules, **32 309,8 secondes GPU cumulées = 8,97 h GPU**, réparties sur cinq vagues et 20 kernels Kaggle (deux comptes × deux kernels concurrents, 2×T4 chacun, `n_gpu=2` vérifié). Durée écoulée totale d'environ 5 h.

Fait d'infrastructure établi ici : **Kaggle autorise deux sessions GPU concurrentes par compte**, ce que le précédent d'EXP-011 (un job par compte sur trois comptes) laissait croire impossible.

## 14. Conclusion située

1. **Le placement du flou n'a que deux valeurs possibles**, et le dépôt avait la mauvaise. Après le ReLU : +1,71 pp à résolution constante, +0,99 pp avec résolution, pour **moins cher** (10 positions au lieu de 19). C'est le changement le plus rentable de la fiche, et il est démontrable analytiquement.
2. **L'annelage ne paie pas en accuracy** — un sigma constant fait aussi bien aux deux placements — mais il **livre l'architecture cible**, ce que le bras constant ne fait pas du tout.
3. **La résolution domine le flou**, et son lieu compte : réduire à l'intérieur bat réduire l'image. Entre positions internes, plateau.
4. **Un a priori de profondeur sur sigma ne vaut rien** : quatre profils motivés différemment, deux architectures, trois graines, le profil plat les bat tous.
5. **Sur H, le faisceau est mitigé et penche vers « pas seulement ».** Pour : un préfiltre fixe à deux positions reproduit l'effet à 1,09× le coût ; le placement post-ReLU gagne et sa raison est mesurée ; il existe du repliement réel. Contre : `M_nodown` récupère tout sans site pré-décimation ; le repliement résiduel est identique à tous les placements alors que l'accuracy varie.
6. **Meilleure configuration mesurée : `D2G`** — réduction après `blocks[2]` par max-pooling, flou post-ReLU annelé, 0,8099 ± 0,0041 sur trois graines, 574 s. **Meilleur rapport coût/performance : `D1`** — même réduction après `blocks[1]`, aucun flou, 0,8059, 447 s.

## 15. Limites à ne pas franchir

* Une seule graine pour les groupes A à D et une partie de E. Les écarts sous 1 point y sont indécidables, et la vague 4 montre pourquoi.
* Non comparable à EXP-011 : poids initiaux différents.
* Le jeu de test a déjà été exposé par les campagnes antérieures ; aucune sélection de meilleur checkpoint n'a été faite, mais ce n'est pas un test propre.
* Confondant image/interne non levé au groupe E (bilinéaire+antialias contre max-pooling).
* Le confondant placement/dosage du groupe A n'est pas levé.
* Aucun profilage détaillé n'explique pourquoi le coût ne suit pas les FLOPs.

## 16. Sources, figures et artifacts

| Objet | Chemin |
|---|---|
| JSON des 32 configurations, agrégats et contrastes appariés | [`../sources/ablation_aa_results.json`](../sources/ablation_aa_results.json) |
| Vague 1 — trajectoires par test | [`../sources/ablation_aa_curves.png`](../sources/ablation_aa_curves.png) |
| Vague 1 — synthèse et diagnostics | [`../sources/ablation_aa_synthesis.png`](../sources/ablation_aa_synthesis.png) |
| Vagues 1-2 — trajectoires par question | [`../sources/ablation_full_curves.png`](../sources/ablation_full_curves.png) |
| Vagues 1-2 — synthèse | [`../sources/ablation_full_synthesis.png`](../sources/ablation_full_synthesis.png) |
| Vague 3 — grille profondeur × flou | [`../sources/ablation_all_grid.png`](../sources/ablation_all_grid.png) |
| Classement des 24 bras avec coût | [`../sources/ablation_all_ranking.png`](../sources/ablation_all_ranking.png) |
| Vague 4 — dispersion des graines | [`../sources/ablation_seeds.png`](../sources/ablation_seeds.png) |
| Vague 5 — a priori sur sigma | [`../sources/ablation_priors.png`](../sources/ablation_priors.png) |
| Vague 5 — trajectoires | [`../sources/ablation_priors_curves.png`](../sources/ablation_priors_curves.png) |

Métriques par époque, `diagnostics.json`, `ablation_ops_verification.json`, `reduction_verification.json`, `profile_verification.json` et `assets_verification.json` sont conservés par cellule sous `results/kaggle_outputs/abl*-j*/`. Les checkpoints ne sont pas versionnés (`.gitignore`, régénérables).

## 17. Manques identifiés

* `conv_out` à σ ≈ 0,65 — séparerait placement et dosage au groupe A.
* `input_max` sur les poids du jour — lèverait le confondant du groupe E.
* Anti-aliaser les shortcuts option-A séparément des convolutions stridées — le chemin le plus aliasé du réseau, jamais touché.
* Trois graines sur les groupes A à D.
* Le même protocole **avec augmentation de données** : si le plafond observé vers 0,80 est un déficit de biais inductif, tous ces gains devraient largement s'évaporer.

<!-- REPO-EVIDENCE:BEGIN -->

## Traces dans le dépôt (généré — ne pas éditer à la main)

Généré par `tools/enrich_records.py` depuis [`data/repo_evidence.json`](../data/repo_evidence.json). Voir aussi la vue d'ensemble : [REPO_EVIDENCE](../REPO_EVIDENCE.md).

**État du code par rapport à ce run** : ablation_ops.py ajoute a cote de campaign_ops.py, qui n'est pas modifie ; campaign_driver.py recoit une factory de controleur dispatchee sur un champ de cellule, les deux chemins d'evaluation dans le summary et un appel de diagnostics opt-in, le chemin historique restant inchange par defaut. Assets epingles neufs (INTEGRATION C-43).

| Rôle | Chemin dans le dépôt | Dernier commit touchant ce fichier |
|---|---|---|
| Implémentation | [`continuation/ablation_ops.py`](../../../../continuation/ablation_ops.py) | efe24dc 2026-09-10 |
| Implémentation | [`continuation/campaign_ops.py`](../../../../continuation/campaign_ops.py) | b9bb609 2026-09-09 |
| Implémentation | [`scripts/campaign_driver.py`](../../../../scripts/campaign_driver.py) | efe24dc 2026-09-10 |
| Implémentation | [`scripts/ablation_manifest.py`](../../../../scripts/ablation_manifest.py) | efe24dc 2026-09-10 |
| Implémentation | [`scripts/ablation2_manifest.py`](../../../../scripts/ablation2_manifest.py) | efe24dc 2026-09-10 |
| Implémentation | [`scripts/ablation3_manifest.py`](../../../../scripts/ablation3_manifest.py) | efe24dc 2026-09-10 |
| Implémentation | [`scripts/ablation4_manifest.py`](../../../../scripts/ablation4_manifest.py) | efe24dc 2026-09-10 |
| Implémentation | [`scripts/ablation5_manifest.py`](../../../../scripts/ablation5_manifest.py) | efe24dc 2026-09-10 |
| Implémentation | [`scripts/job_ablation.py`](../../../../scripts/job_ablation.py) | efe24dc 2026-09-10 |
| Implémentation | [`scripts/job_ablation2.py`](../../../../scripts/job_ablation2.py) | efe24dc 2026-09-10 |
| Implémentation | [`scripts/job_ablation3.py`](../../../../scripts/job_ablation3.py) | efe24dc 2026-09-10 |
| Implémentation | [`scripts/job_ablation4.py`](../../../../scripts/job_ablation4.py) | efe24dc 2026-09-10 |
| Implémentation | [`scripts/job_ablation5.py`](../../../../scripts/job_ablation5.py) | efe24dc 2026-09-10 |
| Implémentation | [`scripts/stage_ablation_assets.py`](../../../../scripts/stage_ablation_assets.py) | efe24dc 2026-09-10 |
| Implémentation | [`scripts/analyze_ablation.py`](../../../../scripts/analyze_ablation.py) | 1cddcbb 2026-09-10 |
| Implémentation | [`scripts/plot_ablation.py`](../../../../scripts/plot_ablation.py) | efe24dc 2026-09-10 |
| Implémentation | [`scripts/plot_ablation_full.py`](../../../../scripts/plot_ablation_full.py) | efe24dc 2026-09-10 |
| Implémentation | [`scripts/plot_ablation_all.py`](../../../../scripts/plot_ablation_all.py) | efe24dc 2026-09-10 |
| Implémentation | [`scripts/plot_priors.py`](../../../../scripts/plot_priors.py) | 1cddcbb 2026-09-10 |
| Implémentation | [`scripts/plot_seeds.py`](../../../../scripts/plot_seeds.py) | efe24dc 2026-09-10 |
| Sorties d'exécution | [`results/ablation_aa_results.json`](../../../../results/ablation_aa_results.json) | 1cddcbb 2026-09-10 |
| Sorties d'exécution | [`results/ablation_aa_curves.png`](../../../../results/ablation_aa_curves.png) | efe24dc 2026-09-10 |
| Sorties d'exécution | [`results/ablation_aa_synthesis.png`](../../../../results/ablation_aa_synthesis.png) | efe24dc 2026-09-10 |
| Sorties d'exécution | [`results/ablation_full_curves.png`](../../../../results/ablation_full_curves.png) | efe24dc 2026-09-10 |
| Sorties d'exécution | [`results/ablation_full_synthesis.png`](../../../../results/ablation_full_synthesis.png) | efe24dc 2026-09-10 |
| Sorties d'exécution | [`results/ablation_all_grid.png`](../../../../results/ablation_all_grid.png) | efe24dc 2026-09-10 |
| Sorties d'exécution | [`results/ablation_all_ranking.png`](../../../../results/ablation_all_ranking.png) | efe24dc 2026-09-10 |
| Sorties d'exécution | [`results/ablation_seeds.png`](../../../../results/ablation_seeds.png) | efe24dc 2026-09-10 |
| Sorties d'exécution | [`results/ablation_priors.png`](../../../../results/ablation_priors.png) | 1cddcbb 2026-09-10 |
| Sorties d'exécution | [`results/ablation_priors_curves.png`](../../../../results/ablation_priors_curves.png) | 1cddcbb 2026-09-10 |

Un chemin listé existe dans le dépôt au moment de la génération. Cela ne prouve pas seul quelle version du code a exécuté ce run : pour les runs Kaggle, le code réellement expédié est archivé dans le sous-dossier `_repo/` de la sortie correspondante.

<!-- REPO-EVIDENCE:END -->
