---
id: DOC-PROTOCOLS
schema_version: 1
updated_at: 2026-09-09
status: historical_profiles
---

# Protocoles expérimentaux et reproductibilité

[Accueil](README.md) · [Opérateurs](OPERATORS.md) · [Fiches](experiments/INDEX.md)

## 1. Les profils à ne pas fusionner

| Profil documentaire | Expériences | Données | Budget | Modèle et LR |
|---|---|---|---|---|
| `P-INPUT-GN` | EXP-000, EXP-001 | 45k train / 5k val ; test intact | 40 époques, 14 040 updates | ResNet-20 GN, LR max 0,1, warmup 400 |
| `P-INT-EARLY` | EXP-004 | 5k train, autres détails incomplets | 600 updates | Ancien petit réseau ; config à récupérer |
| `P-INT-GN` | EXP-005 | 10k train / 5k val | 1 200 updates | ResNet-20 GN, LR max 0,005 ; contrôle 0,002 mentionné |
| `P-R18-BN` | EXP-006 | 10k train / 5k val | 1 200 updates | ResNet-18 BN, LR max 0,005, warmup 60 |
| `P-R20-BN-PILOT` | EXP-007, EXP-009 | 10k train / 5k val | 2 400 updates | ResNet-20 BN, LR max 0,005, warmup 60 |
| `P-FULL-R20-BN` | EXP-008, EXP-010, EXP-011 | 50k train / 10k test | 30 époques, 11 730 updates | ResNet-20 BN, LR max 0,005, warmup 60 |
| `P-PREVIEWS` | EXP-002, EXP-003 | 10 images fixes, une par classe | Aucun entraînement | CPU, opérateurs d'entrée |

Ces profils sont des repères documentaires, pas des fichiers de configuration exécutables. Une différence entre deux profils doit rester visible dans l'analyse.

## 2. Architecture et initialisation

### ResNet-20 CIFAR

Largeurs 16/32/64, trois BasicBlocks par stage, deux convolutions principales 3×3 par bloc, stem 3×3 stride 1, raccourcis option A sans paramètres, moyenne spatiale globale, tête 64→10. **269 722 paramètres**. Le registre historique contient `resnet20_gn` ; la variante corrigée est nommée `resnet20_bn_cifar`.

L'ancien modèle utilise GroupNorm, huit canaux par groupe dans le protocole d'entrée. L'audit ultérieur rapporte une initialisation de classifieur Kaiming `fan_out` de grande amplitude. Le code exact de chaque ancien run reste à vérifier ; ne pas supposer que tous les historiques partagent une même fonction d'initialisation simplement parce qu'ils utilisent GN.

La variante BN corrigée utilise : convolutions Kaiming normal `fan_in`, gain ReLU ; affine BN poids 1/biais 0 ; running mean 0, running variance 1, momentum BN 0,1 ; classifieur normal std 0,01, biais nul. Mesure rapportée du classifieur ResNet-20 : std 0,0099. Le remplacement de GN par BN n'est pas un simple changement de stockage : il modifie l'entraînement et l'évaluation.

### ResNet-18 CIFAR du pilote

Stem CIFAR 3×3 stride 1 sans max-pool ImageNet ; largeurs 64/128/256/512, deux BasicBlocks par stage ; trois raccourcis projetés 1×1+BN ; moyenne globale, tête 512→10. **11 173 962 paramètres**, 17 sites Gaussian, 20 couches BN, 60 buffers BN rapportés. Même convention d'initialisation corrigée, std du classifieur mesurée 0,0102. Fichier rapporté : `continuation/models/resnet18_bn.py`.

## 3. Données et normalisation

L'étude initiale sépare le train officiel CIFAR-10 en 45 000/5 000, stratifié, seed de split 12345. Statistiques initiales rapportées sur les 45 000 images non filtrées :

```text
mean = [0.49118823, 0.48212275, 0.44643784]
std  = [0.2470552,  0.24351363, 0.2615779]
```

Les pilotes 10k réutilisent les mêmes sous-ensembles et statistiques entre bras ; leurs indices exacts doivent venir de leurs artifacts. Les campagnes 50k recalculent une fois les statistiques sur le train complet. **Les valeurs numériques complètes de cette dernière normalisation ne sont pas présentes dans le JSON final livré.** Les anciennes valeurs ne sont pas un remplacement valable.

L'absence d'augmentation est constante dans les principaux essais décrits : pas de crop aléatoire, flip, color jitter ou autre régularisation de données ajoutée. Le pipeline flottant et le lieu des transformations sont définis dans OPERATORS.

Lorsque les 50 000 images officielles deviennent le train, les anciennes 5 000 de validation en font partie. Elles ne peuvent plus servir de validation indépendante. Les campagnes complètes ont suivi le test officiel de 10 000 images à plusieurs checkpoints ; ce suivi est exploratoire.

## 4. Batch physique, accumulation et fin d'époque

`P-INPUT-GN` : batch physique 128, dernier batch incomplet écarté ; \(\lfloor45000/128\rfloor=351\) updates par époque, d'où 14 040 en 40 époques.

Les protocoles BN utilisent un batch effectif 128 par quatre microbatches physiques de 32. BatchNorm voit 32 images à chaque passage ; accumuler les gradients ne lui donne pas des statistiques de batch 128. Ce choix garde les conditions du pilote positif et une marge mémoire pour les opérateurs.

En mode époque complet :

\[
50000=390\times128+80,\qquad M=391,\qquad B=30M=11730.
\]

Le dernier groupe est constitué de 32+32+16 exemples ; il doit être pondéré par son effectif réel. Si les pertes de microbatches sont des moyennes, la contribution d'un microbatch de \(n_j\) exemples est \(n_j/N\) fois sa moyenne, avec \(N\) l'effectif du groupe d'accumulation. Diviser systématiquement par quatre donnerait une mauvaise pondération au dernier groupe. BN voit alors 16 images sur le dernier passage.

Le pilote à nombre fixe d'updates utilise une séquence d'indices de 2 400×128 exemples rapportée dans l'appariement db2. Il ne faut pas lui attribuer automatiquement les règles du mode époque 50k.

## 5. Optimiseur et calendrier global

Les études principales utilisent SGD, momentum 0,9, weight decay \(5\times10^{-4}\), sans Nesterov. La CE affichée est non régularisée : le weight decay n'est pas ajouté à sa valeur.

Les calendriers warmup/cosine dépendent de l'update globale, pas de la phase de filtre. Les valeurs suivantes sont distinctes :

- études d'entrée : pic 0,1, warmup 400, horizon 14 040 ;
- pilote ResNet-18 : pic 0,005, warmup 60, horizon 1 200 ;
- pilotes ResNet-20 BN : pic 0,005, warmup 60, horizon 2 400 ;
- train complet récent : pic 0,005, warmup **60**, horizon 11 730.

La formule exacte de l'indexation du dernier LR (update indexée à zéro, LR final après step, etc.) doit être copiée du scheduler du run ; les rapports seuls ne suffisent pas à choisir entre conventions voisines. L'intention exécutée est une décroissance cosine vers zéro sur l'horizon total.

Ne jamais redémarrer le LR, le momentum, les poids ou les statistiques BN à une transition de continuation, sauf si cela définit explicitement une nouvelle ablation. Le checkpoint à 1 200 d'un run prévu pour 2 400 updates n'est pas un témoin exact d'un entraînement dont le cosine s'achevait déjà à 1 200.

## 6. Calendriers réellement exécutés

### 6.1 Gaussian complet, paliers et géométrique

Pour \(e=0,\ldots,29\), `Gplateau` vaut successivement 1 / 0,85 / 0,70 / 0,60 / 0,50 / 0,40 / 0,30 par blocs de trois époques jusqu'à e=20, puis 0 à partir de e=21.

\[
Ggeo(e)=\begin{cases}0,9^e,&e<21,\\0,&e\ge21.\end{cases}
\]

Il y a 8 211 updates dans la phase filtrée et 3 519 à l'endpoint. Le facteur 0,9 appliqué chaque époque est une compression de l'idée de CBS, avec extinction finale explicite ; ce n'est pas sa cadence originale.

### 6.2 Résolutions de la grille

| Calendrier | e=0…5 | e=6…11 | e=12…29 |
|---|---:|---:|---:|
| `R32` | 32 | 32 | 32 |
| `Rprog` | 16 | 24 | 32 |
| `Rgentle` | 24 | 24 | 32 |
| `Rreverse` | 24 | 16 | 32 |

Les sigmas internes effectifs tiennent compte des rapports de tailles au site concerné ; voir OPERATORS. `Gmix` et `early7` y sont également définis.

### 6.3 Pilotes courts

ResNet-18 BN : \(\sigma_k=\max(1-k/600,0)\), pour k=0…1199.

ResNet-20 BN : paliers de 1 à 0,30, puis extinction à k=1700 sur 2 400 updates. Le compte rendu n'expose pas toutes les bornes exactes ; les valeurs aux checkpoints sont conservées dans EXP-007. **Ne pas remplir ces bornes par interpolation ou par supposition.**

Pilote db2 BN : niveaux s=0 / 0,25 / 0,50 / 0,70 / 0,85 / 0,95 / 0,99, puis s=1 à k=1700. Même réserve sur les bornes complètes des blocs, absentes des artifacts locaux. La valeur d'un diagnostic après 1 700 updates peut rester celle de la dernière update k=1699.

## 7. Appariement : ce qui doit être identique

Une même graine ne suffit pas. Pour une comparaison appariée, vérifier les images, la sonde, les poids et buffers initiaux, les permutations ou la suite de minibatches, les normalisations, les réglages optimiseur/LR et le budget. Les opérations d'évaluation ne doivent pas consommer le RNG de données de manière différente entre bras.

Le pilote résolution rapporte quatre SHA-256 contrôlés dans le job : sous-ensemble, sonde, permutation seed0 et initialisation (116 tenseurs). Le pilote db2 rapporte également l'égalité bitwise de tous ces objets avec le Gaussian précédent. La grille complète rapporte les assets vérifiés sur ses trois jobs.

Cela établit un appariement des conditions initiales et des données ; cela ne garantit pas une trajectoire float32 identique entre exécutions. Deux entraînements plain appariés ont donné 55,36 et 55,40 % ; ce constat seul ne permet pas d'estimer une distribution du bruit. Les écarts plus récents de réplication doivent aussi rester descriptifs.

## 8. Sauvegarde et reprise

Une reprise complète doit conserver modèle, buffers BN, optimiseur, scheduler ou sa reconstruction exacte, position globale, état d'époque/sampler, RNG Python/NumPy/Torch CPU/CUDA et scaler AMP s'il existe. Une sauvegarde « modèle + optimiseur + step » n'est pas automatiquement suffisante.

Les premiers checkpoints finaux d'EXP-000 ne contenaient pas les états nécessaires au branchement à 1 500. EXP-001 a rejoué les seuls préfixes utiles sur l'horizon LR original, puis créé `full_state_v1`. Les métriques enregistrées à une update ne prouvent pas l'existence d'un checkpoint à cette update.

Les campagnes récentes rapportent des checkpoints reprenables, notamment aux époques 6/12/21/30 pour la résolution. Leur contenu reste à vérifier dans le dépôt avant une reprise effective.

## 9. Comptabilité du calcul

Rapporter séparément : updates d'optimiseur, présentations d'images, temps de formation, temps d'évaluation, coût de préparation et temps mural écoulé de la campagne. Les secondes GPU cumulées et la durée sur plusieurs GPU ne sont pas interchangeables.

Une sonde doit mesurer forward, backward et optimizer step, après échauffement et synchronisation appropriée. Les petits benchmarks d'opérateurs sans optimiseur ne prédisent pas directement le coût de 19 insertions dans un CNN. Une réduction des paramètres ne donne pas le même facteur qu'une réduction de MACs ; ni l'une ni l'autre ne donne un facteur temporel fiable sans mesure.

La campagne à six T4 utilise des entraînements indépendants, un processus par GPU ; ce n'est pas du data parallel à six GPU pour un seul modèle. T4 désigne un **GPU**, pas un TPU. Les informations de comptes et d'authentification n'ont pas à être dupliquées dans cette base.

## 10. Recettes proposées puis remplacées

Ne décrivent **pas** les runs exécutés : 200 époques ResNet-18, batch physique 64, LR 0,1 avec milestones 30/60/90 ; warmup 5 % de l'horizon des campagnes courtes ; extinction à 600 dans le pilote ResNet-20 à 2 400 ; 15 époques filtrées sur 30 dans un ancien brouillon. Voir [DECISIONS](DECISIONS.md) pour leur remplacement.
