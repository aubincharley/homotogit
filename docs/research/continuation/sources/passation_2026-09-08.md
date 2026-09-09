# Passation — homotopie, filtrage gaussien et ondelettes sur CIFAR-10

État des travaux rapportés au 8 septembre 2026. Document destiné à être joint à une nouvelle conversation et à être lu avant toute nouvelle proposition expérimentale.

Ce document réunit les échanges disponibles, les comptes rendus de l’agent d’exécution et les fichiers antérieurs effectivement retrouvés. Il ne prétend pas restituer des messages devenus inaccessibles. Les résultats récents proviennent des comptes rendus transmis par Maxime et des figures ; les anciens résultats, les définitions TV et surtout la définition db2 ont aussi été retrouvés dans des fichiers sauvegardés. Les entraînements n’ont pas été relancés pour cette passation.

## 1. À lire d’abord : où nous en sommes

Le projet explore des méthodes de continuation pour l’entraînement des réseaux : commencer avec une transformation qui simplifie les images ou les représentations, puis revenir progressivement au problème cible. Il faut distinguer le choix de la famille de transformations, le calendrier de continuation et le calendrier d’optimisation.

La piste actuellement la plus prometteuse est le **filtrage spatial interne des cartes de caractéristiques**, après les convolutions et avant la normalisation. Le flou sur les images d’entrée a été essayé auparavant, avec des résultats plutôt négatifs dans les protocoles testés. Ces deux interventions ne doivent pas être confondues.

La dernière campagne est **terminée** : CIFAR-10 complet, 50 000 images d’entraînement et 10 000 de test, ResNet-20 avec BatchNorm et initialisation corrigée, 30 époques, trois graines et trois bras, soit neuf entraînements.

| Bras | Accuracy test finale, moyenne ± écart-type | CE test finale, moyenne ± écart-type | Différence appariée d’accuracy contre témoin |
|---|---:|---:|---:|
| Témoin sans filtre | 75,40 ± 0,40 % | 0,7615 ± 0,0039 | — |
| Gaussian, paliers | 78,41 ± 0,32 % | 0,6259 ± 0,0094 | +3,01 points |
| Gaussian, géométrique comprimé | 78,46 ± 0,72 % | 0,6636 ± 0,0255 | +3,06 points |

Les deux calendriers gaussiens gagnent contre leur témoin sur **chacune des trois graines**. Ils terminent par neuf époques sans filtre ; le bénéfice subsiste donc sur le réseau final sans filtrage. Le coût total rapporté est 5 504 secondes GPU, soit environ 92 minutes GPU cumulées et 46 minutes écoulées avec deux T4.

Maxime souhaite maintenant reprendre **la méthode db2 déjà essayée**, et non concevoir une nouvelle méthode d’ondelettes. Sa formule exacte a été retrouvée dans `wavelet_preview_diagnostics.json` : transformée stationnaire redondante à deux niveaux, seuillage doux des détails avec seuil proportionnel à leur RMS, approximation conservée, reconstruction adjointe. Elle est explicitée en section 3.

Les mesures anciennes confirment que db2 est beaucoup moins coûteux que les projections TV strictes dans leurs implémentations testées. Elles montrent aussi que **db2 est beaucoup plus coûteux que Gaussian**, notamment à l’intérieur du réseau. Il faut mesurer la configuration actuelle avant de promettre une campagne db2 aussi rapide que la campagne gaussienne.

La présente demande porte sur cette passation. Aucune nouvelle campagne d’entraînement, aucun envoi à l’agent et aucun changement du code expérimental n’ont été effectués pendant sa préparation.

## 2. Projet, objectifs et façon de travailler

### 2.1 Cadre scientifique

Sujet retrouvé : **« Entraînement homotopique des réseaux de neurones profonds »**, sujet 24 de la Filière Métiers de la Recherche, Joseph Gabet, L2S / CentraleSupélec. Le sujet est ouvert : aucune homotopie particulière n’est imposée. Les objectifs sont une cartographie critique, la conception de chemins justifiés et des comparaisons contrôlées à budgets comparables. Les livrables annoncés sont un poster en anglais, un article de vulgarisation, un rapport scientifique et un code/protocole reproductible.

Le document de synthèse retrouvé, `homotopy_synthesis.tex`, porte le titre *Homotopy methods for optimization*, avec les auteurs Aubin Charley, Alexandre Corrard, Idriss El khamlichi et Maxime Nicaise, septembre 2026.

Le cadre général est

\[
L_{\mathrm{cible}}(\theta)=\frac1n\sum_i\ell_{\mathrm{CE}}(f_\theta(x_i),y_i).
\]

Pour une transformation des entrées,

\[
L_\eta(\theta)=\frac1n\sum_i\ell_{\mathrm{CE}}(f_\theta(T_\eta x_i),y_i).
\]

Pour une modification interne du réseau, écrire plutôt

\[
L_\eta(\theta)=\frac1n\sum_i\ell_{\mathrm{CE}}(f_{\theta,\eta}(x_i),y_i).
\]

La valeur cible du paramètre dépend de la famille : Gaussian revient à l’identité quand \(\sigma\to0\), tandis que TV et la méthode d’ondelettes définie ici reviennent à l’identité quand leur paramètre \(t\) ou \(s\) atteint 1.

### 2.2 Distinctions à conserver

- Une image visuellement plus simple n’implique pas une classification plus facile, ni une fonction de perte convexe dans les poids.
- La convexité d’une projection d’image TV ne rend pas convexe l’entraînement du réseau qui reçoit cette image.
- Le lissage des paramètres, le lissage des entrées et le lissage spatial des activations sont trois constructions différentes.
- Le biais spectral du réseau comme fonction des entrées n’est pas directement la fréquence spatiale d’une carte de caractéristiques.
- Une continuation peut modifier la trajectoire d’optimisation et produire une régularisation implicite. « Régularisation » et « effet sur l’optimisation » ne s’excluent pas.
- Une meilleure généralisation avec une CE d’entraînement plus élevée ne prouve ni une meilleure minimisation de cette CE, ni des bassins plus larges, ni un paysage plus lisse.
- Une suite de paramètres ne suffit pas à prouver la continuité ou la pertinence du chemin. Il faut documenter l’opérateur réellement calculé et son endpoint.
- Les transformations des images doivent repartir de l’image originale à chaque niveau, afin de permettre le retour des détails. Ne pas composer des opérations irréversibles sur la sortie de l’étape précédente.
- Le noyau gaussien continu a un lien avec la chaleur, \(a=\sigma^2/2\), mais un noyau discret tronqué avec padding n’a pas automatiquement la propriété exacte de semi-groupe.
- Ne pas assimiler une réduction visuelle de texture à une diminution rigoureuse d’information de Shannon. Un multiplicateur gaussien idéal non nul peut rester injectif en arithmétique exacte.

### 2.3 Préférences expérimentales de Maxime

Maxime préfère des expériences informatives et abordables, puis un changement d’échelle lorsque les résultats le justifient. Il a explicitement rejeté le passage prématuré à 200 époques et les campagnes qui accumulent trop de variantes. Il veut préserver du budget pour les ondelettes et d’autres continuations.

Il souhaite des paliers de filtrage suffisamment longs : pas un changement de valeur « toutes les deux secondes ». Pour le pilote de 2 400 updates, il a demandé de garder la continuation jusqu’à 1 700 updates, puis d’entraîner sans filtre.

Il veut voir en premier les performances du **réseau avec les filtres qu’il utilise effectivement à ce stade**. Les courbes bypassed doivent rester des diagnostics secondaires, clairement distingués.

Éviter les demandes de confirmation pour les vérifications ou les exécutions déjà autorisées. Vérifier si un kernel Kaggle existe ou est terminé avant de le soumettre à nouveau : plusieurs interruptions concernaient seulement le téléchargement, pas le calcul. Préserver les résultats et reprendre les checkpoints plutôt que dupliquer les entraînements.

## 3. La méthode d’ondelettes db2 déjà utilisée — définition retrouvée

### 3.1 Source et portée

La source précise est le champ de configuration de `wavelet_preview_diagnostics.json`, complété par `wavelet_timings.json`. Il ne s’agit pas d’une formule générique reconstruite de mémoire. Le code Python lui-même n’a pas été retrouvé dans les fichiers accessibles pour cette passation ; les quelques détails d’implémentation que le JSON ne décrit pas sont signalés ci-dessous.

La méthode a été prévisualisée avec **Haar, db2, sym4 et coif1**. La piste retenue dans la discussion est **db2**. Un premier pilote d’entraînement contenant un bras db2 existe aussi ; voir section 6.3.

### 3.2 Transformée, seuils et reconstruction

Soit \(h\) une image ou une carte de caractéristiques. L’opération s’applique **séparément par exemple et par canal**.

On note \(E\) l’extension miroir qui porte la taille \(H\times W\) à \(2H\times2W\), et \(C\) le recadrage sur le bloc supérieur gauche de taille \(H\times W\). La transformée est périodique sur le domaine ainsi étendu.

\(W\) est la transformée en ondelettes db2 **non décimée / stationnaire**, séparable en 2D, à **deux niveaux**. Les filtres sont divisés par \(\sqrt2\), selon la normalisation du fichier, pour obtenir un cadre serré avec \(W^*W=I\) sur ce domaine. Décomposer :

\[
W(Eh)=\left(a_2,\{d_{j,o}\}_{j=1,2;\ o\in\{LH,HL,HH\}}\right).
\]

Pour chaque bande de détails, calculer la RMS **sur ses coefficients non seuillés**, séparément pour chaque exemple et canal :

\[
r_{j,o}(h)=\operatorname{RMS}(d_{j,o}).
\]

Le paramètre de continuation est \(s\in[0,1]\), et le seuil vaut

\[
\boxed{\lambda_{j,o}(s;h)=4(1-s)\,2^{1-j}\,r_{j,o}(h).}
\]

Le seuillage est doux :

\[
\mathcal S_\lambda(d)=\operatorname{sign}(d)\max(|d|-\lambda,0).
\]

La formule complète est donc

\[
\boxed{
T_s(h)=C\,W^*\!\left(a_2,
\{\mathcal S_{\lambda_{j,o}(s;h)}(d_{j,o})\}_{j,o}\right).
}
\]

L’approximation \(a_2\) reste inchangée. Les seuils sont \(4(1-s)r_{1,o}\) au premier niveau et \(2(1-s)r_{2,o}\) au second. La RMS dépend du signal courant, donc le seuil n’est pas un simple nombre global partagé par tous les exemples.

### 3.3 Propriétés et détails qu’il ne faut pas remplacer implicitement

- \(s=1\) donne des seuils nuls et l’identité en arithmétique exacte. Le fichier précise que les aperçus passent par la reconstruction, sans raccourci spécial à cet endpoint.
- \(s=0\) donne des seuils finis. **Ce n’est pas l’image constante** : l’approximation est conservée et certains détails peuvent subsister.
- Il n’y a pas de clipping dans \([0,1]\). Les sorties peuvent légèrement dépasser les bornes RGB. C’est particulièrement pertinent si l’opérateur est appliqué à des activations, qui ne sont pas des images bornées.
- Il n’y a pas de contrainte exacte de TV ni de conservation explicite de la moyenne. Le recadrage après l’extension peut produire une petite dérive de moyenne.
- La transformée est redondante : ne pas la décrire comme une simple base orthonormale décimée.
- Ne pas identifier automatiquement ce seuillage suivi de reconstruction au proximal d’une pénalité d’analyse \(\|Wh\|_1\). La redondance, le recadrage et les seuils dépendant de \(h\) demandent une analyse distincte.
- Le seuillage élimine des coefficients selon leur amplitude ; il ne produit pas automatiquement une bande passante fréquentielle stricte.
- Le JSON indique `eps = 1e-12`, mais ne précise pas où cet epsilon intervient exactement dans la RMS. Il ne précise pas non plus si les seuils dépendant de la RMS sont détachés du graphe de gradient. **Relire ces deux détails dans le code avant une reproduction bit à bit.**
- L’extension est décrite comme miroir vers \(2H\times2W\). La convention exacte de répétition des échantillons aux bords est à lire dans l’implémentation, et ne doit pas être déduite de la convention du filtre gaussien.

Configuration textuelle retrouvée :

```text
wavelet: db2
levels: 2
transform: undecimated (stationary) separable 2-D, per sample/channel
normalization: filters scaled by 1/sqrt(2); tight frame, W*W = I
boundary: mirror-extend to 2Hx2W, periodic transform, crop top-left
threshold: lambda_{j,o} = 4(1-s) 2^(1-j) * RMS(d_{j,o}), soft, a_J kept
rms: per sample/channel/band, from unthresholded coefficients
eps: 1e-12
identity_at: s == 1 (no shortcut; exact in exact arithmetic)
clipping: none; outputs may leave [0,1]
```

### 3.4 Aperçus db2 et conservation du contraste

Les dix images sont les mêmes que pour les aperçus gaussiens et TV. Les moyennes suivantes viennent du JSON retrouvé :

| \(s\) | TV obtenue / TV originale | Contraste conservé \(\rho\) | Plus grande dérive absolue d’une moyenne de canal |
|---:|---:|---:|---:|
| 1 | 1,0000 | 1,0000 | \(2,22\times10^{-16}\) |
| 0,9 | 0,8254 | 0,9571 | 0,000239 |
| 0,75 | 0,6609 | 0,9110 | 0,000244 |
| 0,5 | 0,5245 | 0,8675 | 0,000361 |
| 0,25 | 0,4685 | 0,8472 | 0,000231 |
| 0 | 0,4446 | 0,8376 | 0,000213 |

Ici \(\rho=\|z-\bar x\|_2/\|x-\bar x\|_2\). À \(s=0\), les valeurs extrêmes rapportées vont environ de −0,00349 à 1,03993. La TV réduite et le contraste encore élevé expliquent l’intérêt visuel de cette piste, mais ne garantissent aucun gain de classification.

### 3.5 Coût effectivement mesuré

Mesures anciennes sur **RTX 3050 Laptop, 4 Go**, PyTorch 2.5.1+cu121, float32. db2 utilise \(s=0,5\), Gaussian utilise \(\sigma=1\). Les temps GPU sont des médianes après échauffement, synchronisées CUDA, sans transfert hôte/GPU, sans fichiers et **sans optimizer step**. Le coût db2 comprend extension, analyse à deux niveaux, RMS, seuillage, synthèse adjointe et recadrage.

Sur CPU, une image : db2 **14,40 ms** ; Gaussian environ **0,426 ms**. Haar, sym4 et coif1 sont dans la plage 14,9–15,5 ms sur cette mesure CPU.

| Tenseur | db2 forward | db2 forward + backward | Gaussian forward + backward | Pic mémoire db2 | Pic mémoire Gaussian |
|---|---:|---:|---:|---:|---:|
| \(128\times3\times32\times32\) | 22,67 ms | 56,75 ms | 0,759 ms | 387,8 MiB | 11,6 MiB |
| \(128\times16\times32\times32\) | 117,71 ms | 292,39 ms | 2,661 ms | 2 072,1 MiB | 62,0 MiB |
| \(128\times32\times16\times16\) | 60,65 ms | 152,46 ms | 1,544 ms | 1 059,6 MiB | 34,5 MiB |
| \(128\times64\times8\times8\) | 31,42 ms | 81,46 ms | 0,984 ms | 560,2 MiB | 20,0 MiB |

Ces mesures portent sur des opérateurs isolés et un batch physique de 128. Les entraînements récents utilisent des microbatches de 32 et des T4 : **ne pas transformer directement ces nombres en durée d’entraînement**. Ils suffisent toutefois à réfuter une promesse selon laquelle db2 aurait nécessairement le même surcoût que Gaussian. Le coût peut être amélioré par une autre implémentation, à opérateur équivalent ; cela reste à démontrer.

## 4. Les deux projections TV — spécification finale de Maxime

Cette section reprend la version éditée du bloc de demande `64829`. Cette version prime sur les premières propositions, notamment pour la conservation des moyennes et pour la fidélité homogène \(\dot H^{-1}\).

### 4.1 Contraintes communes

Travailler sur l’image flottante originale \(x\in[0,1]^{3\times H\times W}\), **avant** normalisation pour le réseau. Différences avant avec espacement pixel unité :

\[
(D_1z)_{c,p,q}=\begin{cases}
z_{c,p+1,q}-z_{c,p,q},&p<H,\\
0,&p=H,
\end{cases}
\]

et horizontalement,

\[
(D_2z)_{c,p,q}=\begin{cases}
z_{c,p,q+1}-z_{c,p,q},&q<W,\\
0,&q=W.
\end{cases}
\]

Pas de bouclage périodique. TV isotrope canal par canal :

\[
\operatorname{TV}(z)=\sum_{c,p,q}
\sqrt{(D_1z)_{c,p,q}^{2}+(D_2z)_{c,p,q}^{2}}.
\]

Ensemble admissible :

\[
\mathcal K_t(x)=\left\{z\in[0,1]^{3\times H\times W}:\quad
\operatorname{TV}(z)\leq t\operatorname{TV}(x),\quad
\bar z_c=\bar x_c\ \text{pour tout }c\right\},
\qquad 0\leq t\leq1.
\]

Le budget TV est global sur les trois canaux ; chaque moyenne de canal est conservée séparément. **Ne pas imposer la conservation de la variance.**

### 4.2 TV–\(L^2\)

\[
\boxed{T_t^{L^2}(x)=\operatorname*{arg\,min}_{z\in\mathcal K_t(x)}
\frac12\sum_c\|z_c-x_c\|_2^2.}
\]

### 4.3 TV–\(\dot H^{-1}\) homogène

Sur un canal vectorisé :

\[
\mathsf L=D_1^\top D_1+D_2^\top D_2.
\]

Ce Laplacien discret positif semi-défini a les constantes pour noyau. Utiliser sa pseudoinverse de Moore–Penrose, nulle sur le mode constant :

\[
\boxed{T_t^{\dot H^{-1}}(x)=\operatorname*{arg\,min}_{z\in\mathcal K_t(x)}
\frac12\sum_c(z_c-x_c)^\top\mathsf L^\dagger(z_c-x_c).}
\]

La conservation des moyennes rend chaque résidu \(r_c=z_c-x_c\) de moyenne nulle. On peut résoudre \(\mathsf Lp_c=r_c\), \(\bar p_c=0\), puis calculer \(\frac12\sum_c\langle r_c,p_c\rangle\).

**Ne pas remplacer cette fidélité par \((I+\alpha\mathsf L)^{-1}\). Il n’y a pas de paramètre \(\alpha\) dans cette expérience.**

Les deux minimisateurs sont uniques : stricte convexité de la fidélité \(L^2\), et de la fidélité homogène restreinte au sous-espace de résidus de moyenne nulle. Les endpoints sont traités exactement : \(T_1(x)=x\), \(T_0(x)=\bar x\), et toute image spatialement constante reste inchangée pour tout \(t\).

Un coefficient de pénalité TV commun à toutes les images ne réalise pas automatiquement le même budget relatif \(t\). Les résidus numériques doivent être rapportés, pas dissimulés derrière une assertion de projection exacte.

### 4.4 Aperçus demandés et réalisés

La demande autorisait l’implémentation et les aperçus CPU pendant les autres entraînements, après vérification de l’absence d’interférence. Elle n’autorisait aucun nouvel entraînement de classification TV.

Une grille par méthode, dix images, colonnes \(t\in\{1;0,9;0,75;0,5;0,25;0\}\), affichage fixe dans \([0,1]\) sans réajustement de contraste. Diagnostics : ratio TV et \(\rho=\|z-\bar x\|_2/\|x-\bar x\|_2\), avec dénominateurs nuls explicitement gérés.

Indices dans le CIFAR-10 train original :

```text
[15671, 18089, 18626, 24647, 25865,
 26662, 30887, 34431, 39933, 48888]
```

Indices correspondants dans l’ancien split d’entraînement à 45 000 :

```text
[14069, 16264, 16752, 22158, 23268,
 23977, 27811, 30987, 35936, 43986]
```

Ordre des classes : ship, cat, dog, truck, horse, airplane, bird, deer, automobile, frog.

### 4.5 Ce que les diagnostics sauvegardés montrent

Solveur déclaré : Chambolle–Pock / PDHG, tolérance configurée \(10^{-7}\), maximum 40 000 itérations, pas primaux/duaux 0,3 et 0,3. Le critère d’arrêt complet ne figure pas dans ce résumé de configuration : le relire dans le code si l’on reprend le solveur. La tolérance configurée n’est pas une garantie que chaque contrainte est satisfaite à \(10^{-7}\).

| \(t\) | TV–\(L^2\) : TV moyenne obtenue | TV–\(L^2\) : \(\rho\) moyen | TV–\(\dot H^{-1}\) : TV moyenne obtenue | TV–\(\dot H^{-1}\) : \(\rho\) moyen |
|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 1 | 1 |
| 0,9 | 0,9000009 | 0,9835 | 0,9000002 | 0,9894 |
| 0,75 | 0,7500017 | 0,9516 | 0,7500012 | 0,9674 |
| 0,5 | 0,5000021 | 0,8687 | 0,5000022 | 0,9059 |
| 0,25 | 0,2500025 | 0,6977 | 0,2509496 | 0,7586 |
| 0 | 0 | 0 | 0 | 0 |

Tous les solves \(L^2\) convergent selon le critère implémenté pour \(t=0,9;0,75;0,5\), mais pas tous à 0,25. Pour \(\dot H^{-1}\), tous convergent à 0,9 et 0,75 ; pas tous à 0,5 ni à 0,25. Plusieurs images atteignent 40 000 itérations.

À \(t=0,25\), le champ `max_tv_budget_rel_violation` atteint environ \(2,30\times10^{-5}\) pour \(L^2\), mais **0,03774** pour \(\dot H^{-1}\). Il faut conserver la convention exacte de normalisation de ce champ si l’on réanalyse le code. Les bornes de pixels sont respectées dans ces diagnostics et les dérives de moyenne restent autour de \(10^{-15}\).

Durées totales rapportées des grilles : **1 112,67 s** pour TV–\(L^2\), **2 221,53 s** pour TV–\(\dot H^{-1}\), contre environ 3 s pour la grille db2. Ce sont les coûts de ces implémentations et de ce critère d’arrêt, pas une borne intrinsèque du problème mathématique.

Une note sauvegardée indique environ 9,5 s/image pour TV–\(L^2\) à \(t=0,5\) et 43,6 s/image à \(t=0,25\). À 45 000 images, une extrapolation séquentielle naïve donne **119 h** et **545 h**. D’anciens calculs de 2,5 h / 22 h étaient faux.

## 5. Gaussian interne : opérateur, architecture et évaluation

### 5.1 Opérateur actuel

Filtre gaussien normalisé, séparable, **neuf coefficients par axe**, rayon 4, appliqué indépendamment par canal avec prolongement réfléchi. Pour \(\sigma>0\), les coefficients 1D sont proportionnels à

\[
g_\sigma[r]=\exp\!\left(-\frac{r^2}{2\sigma^2}\right),\qquad r=-4,\ldots,4,
\]

puis normalisés pour que leur somme soit 1. À \(\sigma=0\), bypass exact, sans calcul du noyau.

Les entrées RGB, les raccourcis, les vecteurs après pooling et les logits ne sont pas filtrés. Les insertions sont placées ainsi :

```text
Stem ou première convolution du bloc : conv → filtre → BN → ReLU
Deuxième convolution du bloc : conv → filtre → BN → ajout du raccourci → ReLU
```

Même valeur du paramètre sur tous les sites et microbatches d’une même update. Ne pas réinitialiser l’optimiseur ou BatchNorm aux changements de niveau.

### 5.2 Extension réfléchie sur les petites cartes de ResNet-18

Le padding natif PyTorch exige un padding strictement inférieur à la dimension. Un rayon 4 ne passe donc pas directement sur une carte \(4\times4\). Cela ne rend pas la réflexion mathématiquement impossible.

Pour un axe de longueur \(n>1\), utiliser

\[
P=2(n-1),\qquad m=i\bmod P,\qquad r_n(i)=\min(m,P-m).
\]

Pour le padding de rayon 4, prendre les indices \(i=-4,\ldots,n+3\). Pour \(n=4\), cela donne exactement

```text
[2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1]
```

Garder le chemin natif là où il est valide ; utiliser des gathers différentiables pour le fallback. Ne pas rajouter du padding à la convolution après avoir étendu le tenseur. Les vérifications rapportées couvrent l’accord des sorties et gradients avec le padding natif dans les cas communs, les constantes, la forme et un backward fini sur \(4\times4\).

### 5.3 ResNet-20 retenu pour la suite

- Variante CIFAR, largeurs 16/32/64, trois BasicBlocks par stage.
- Raccourcis option A sans paramètres, global average pooling, classifieur 64→10.
- **269 722 paramètres**, **19 insertions** : stem + deux convolutions principales de chacun des neuf blocs.
- Cartes de tailles 32×32, 16×16 et 8×8 ; le rayon 4 y est compatible avec le padding réfléchi natif.
- BatchNorm2d remplace l’ancien GroupNorm ; momentum BN 0,1, affine initialisé à 1/0 et buffers standards.
- Convolutions initialisées par Kaiming normal `fan_in`, gain ReLU ; classifieur \(\mathcal N(0,(0{,}01)^2)\), biais nul.
- Identifiant rapporté : `resnet20_bn_cifar`. L’ancien `resnet20_gn` doit rester disponible pour préserver les expériences.

### 5.4 « Filters active » et « bypassed »

À un checkpoint, les poids appris \(\theta\) et les buffers BatchNorm \(b\) sont les mêmes. On évalue deux fonctions différentes :

\[
\text{active : }f_{\theta,\sigma;b}(x),\qquad
\text{bypassed : }f_{\theta,0;b}(x).
\]

**Active** conserve le niveau de filtre utilisé par l’entraînement à ce stade. C’est la mesure principale de la qualité du prédicteur courant. **Bypassed** enlève provisoirement les filtres, sans réentraîner les poids et sans recalibrer BatchNorm. C’est un diagnostic du retrait anticipé des filtres.

Le témoin plain est un autre entraînement, jamais filtré. Il n’est pas obtenu en retirant les filtres du bras Gaussian.

Toutes les évaluations se font en `eval()`, sans mise à jour des statistiques BN, puis restaurent le mode précédent. BatchNorm utilise ses moyennes et variances mémorisées en évaluation ; elles ont été apprises avec les activations filtrées. Le retrait des filtres modifie donc les activations **et** peut créer un décalage avec ces statistiques. [Documentation BatchNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)

La mauvaise accuracy bypassed pendant la continuation ne signifie pas que le réseau courant n’apprend pas. Elle ne prouve pas non plus que l’intégralité de la dégradation est due à BN : le retrait change la fonction du réseau elle-même. Pour isoler la contribution BN, il faudrait un diagnostic de recalibration à poids fixés, sur données d’entraînement seulement. Ce diagnostic n’a pas été rapporté comme effectué.

Convention des figures retenue : **active en trait plein**, bypassed en pointillé fin de même couleur ; les deux pour CE et accuracy si disponibles. À filtre nul, les chemins doivent coïncider pour les mêmes états et exemples.

Attention à l’indexation : un checkpoint « après 1 700 updates » contient l’état après les updates \(k=0,\ldots,1699\). Son diagnostic active peut utiliser le dernier \(\sigma\) entraîné, ici 0,30, alors que la prochaine update \(k=1700\) passera à zéro. L’écart résiduel à ce checkpoint ne contredit pas le bypass prévu à partir de \(k=1700\).

## 6. Historique des expériences, dans l’ordre

### 6.1 Expérience 0 : flou gaussien fixe sur les entrées

Sources retrouvées : `results.md`, `exp0_curves.png`, `prompt_homotopie_images.md`.

Cette expérience modifie les images d’entrée, pas les cartes internes. Quinze entraînements indépendants : \(\sigma\in\{0;0,5;1;2;3\}\), graines 0/1/2.

| Élément | Protocole |
|---|---|
| Données | 45 000 train / 5 000 validation stratifiés, split seed 12345 ; test officiel intact |
| Architecture | ResNet-20 CIFAR, GroupNorm, 8 canaux par groupe, raccourcis option A |
| Batch | Physique 128, dernier batch incomplet écarté |
| Durée | 14 040 updates, soit 40 époques à 351 updates |
| Optimisation | SGD, pic LR 0,1, momentum 0,9, WD 0,0005, sans Nesterov |
| LR | Warmup 400 updates, puis cosine sur l’horizon global de 14 040 ; pas de chutes par milestones |
| Autres | Pas d’augmentation, pas d’AMP ; normalisation après transformation |

Moyennes RGB du split train non filtré : `[0.49118823, 0.48212275, 0.44643784]`. Écarts-types : `[0.2470552, 0.24351363, 0.2615779]`. Ces valeurs ne doivent pas être attribuées automatiquement à la campagne ultérieure sur 50 000 images.

« Cible » ci-dessous signifie évaluation sur les images originales ; « transformée » signifie évaluation sur les images filtrées au niveau de l’entraînement.

| \(\sigma\) fixe | Accuracy validation cible | Accuracy validation transformée | CE train cible | MSE image | TV conservée |
|---:|---:|---:|---:|---:|---:|
| 0 | 84,11 ± 0,60 % | 84,11 ± 0,60 % | 0,0034 | 0 | 1 |
| 0,5 | 82,55 ± 0,06 % | 83,55 ± 0,38 % | 0,0219 | 0,00040 | 0,8200 |
| 1 | 71,33 ± 2,38 % | 80,63 ± 0,36 % | 0,7529 | 0,00345 | 0,5568 |
| 2 | 48,32 ± 2,96 % | 74,30 ± 0,09 % | 2,1384 | 0,00951 | 0,3505 |
| 3 | 38,85 ± 1,86 % | 67,41 ± 0,33 % | 2,5208 | 0,01445 | 0,2548 |

Les modèles ajustent très bien leurs données transformées : CE d’entraînement transformée rapportée dans la plage 0,003–0,064 et accuracy proche de 100 %. Cela n’établit pas que la continuation serait inutile : une évaluation directe sur les images originales mesure aussi un changement de distribution, avant adaptation.

Coût : typiquement environ 500 s par entraînement, dont 10–12 s de transformation. Un temps de 7 446 s pour \(\sigma=0,5\), graine 1, incluait une suspension de la machine ; ses métriques ne sont pas à supprimer, mais ce temps est impropre à l’estimation du débit normal.

Dossier rapporté : `results/exp0_gaussian/`. Les checkpoints finaux originaux contenaient modèle, optimiseur et step, mais pas tous les états RNG/sampler nécessaires à une reprise exacte.

### 6.2 Expérience 1 : warm start sur entrées et étape intermédiaire

Sources : `exp1_warmstart.md`, `exp1_equal_budget.png`, `exp1_adaptation.png`, `prompt_exp1_warm_start.md`.

Même protocole que l’expérience 0, même budget global de 14 040 updates, trois graines. Les trois branches sont :

| Bras | Updates 0–1499 | Updates 1500–2999 | Suite |
|---|---|---|---|
| A : direct | \(\sigma=0\) | \(\sigma=0\) | \(\sigma=0\) |
| W : warm start | \(\sigma=1\) | \(\sigma=0\) | \(\sigma=0\) |
| P : étape intermédiaire | \(\sigma=1\) | \(\sigma=0,5\) | \(\sigma=0\) |

Le préfixe commun à W et P est partagé exactement par graine. Il compte néanmoins dans le budget scientifique de chaque branche. Comme le checkpoint complet à 1 500 n’existait pas, seuls les trois préfixes nécessaires ont été rejoués, avec le LR toujours défini sur **14 040**, et non comprimé sur 1 500 updates.

Le format `full_state_v1` sauvegarde modèle, optimiseur, global step, RNG, position du sampler et état AMP nul. Le compte rendu rapporte une reprise bit à bit identique, aucune modification des métriques cibles à l’instant du changement d’opérateur, et 83 tests réussis. Ces contrôles sont ceux du rapport d’exécution, pas de nouveaux tests réalisés pendant cette passation.

À l’update 1 500, le LR rapporté vaut 0,09840385594331022. Le momentum est conservé ; ni LR ni optimiseur ne sont redémarrés aux étapes.

| Bras | Accuracy validation finale | CE validation finale | CE train finale |
|---|---:|---:|---:|
| A | 84,11 ± 0,60 % | 0,6755 ± 0,0223 | 0,0034 |
| W | 82,93 ± 0,11 % | 0,7113 ± 0,0252 | 0,0030 |
| P | 83,19 ± 0,51 % | 0,7116 ± 0,0283 | 0,0031 |

| Graine | A | W | P |
|---:|---:|---:|---:|
| 0 | 84,64 % | 82,86 % | 82,60 % |
| 1 | 84,22 % | 83,06 % | 83,52 % |
| 2 | 83,46 % | 82,88 % | 83,46 % |

Différences appariées moyennes : W−A = −1,17 point, P−A = −0,91 point, P−W = +0,26 point. Pas de gain net de vitesse résolu par les sondes espacées de 500 updates : le franchissement de CE train cible 0,1 intervient autour de 10 000–10 500 updates, puis celui de 0,01 autour de 11 000–11 500 selon les branches.

À 1 500 updates, accuracy cible moyenne : A 61,13 %, W/P 55,57 %. W récupère rapidement après passage aux images originales, mais reste derrière A au terme du budget. Comparer les branches à partir de leur retour sur la cible sans compter le préfixe donnerait un avantage artificiel de budget.

Calcul supplémentaire réellement exécuté : 79 740 updates, environ 2 735 s (0,76 h), dont 151 s de préfixes et 2 584 s de branches ; les témoins A ont été réutilisés. Le filtrage représente environ 9,1 s, soit 0,3 % de ce coût.

Dossier : `results/exp1_gaussian_warmstart/`. Commandes rapportées : `py -m continuation.cli exp1 --config configs/exp1_warmstart.yaml --seeds 0` puis graines 1 et 2 ; commande de rapport `exp1-report`.

Conclusion locale : ces deux continuations **sur les entrées**, avec ces paramètres et cet optimiseur, n’améliorent pas le témoin. Ne pas transformer ce résultat en interdiction générale du warm start ou de la continuation.

### 6.3 Premier pilote interne, avec db2

Figure retrouvée : `pilot_comparison.png`. Son titre indique CIFAR-10, **5 000 images, 600 updates, une graine**. Elle compare plain, Gaussian et db2 ; la continuation se termine à 300 updates. Les courbes d’entraînement sont bypassed, les courbes de validation sont sans filtrage.

Lecture graphique approximative à 600 updates : accuracy plain autour de 12 %, Gaussian 22,5 %, db2 25,8 %. La CE initiale est autour de 6,2 ; le témoin apprend très peu dans ce pilote. Ces nombres sont des lectures de figure et **ne remplacent pas les métriques brutes**.

Ce fichier établit que db2 a déjà été entraîné, et pas seulement prévisualisé. Il ne suffit pas à restituer exactement son calendrier, sa graine, son optimiseur et tous ses réglages. Les relire dans les anciens runs avant une comparaison précise. Les gros défauts de la configuration de départ ont motivé la suite des audits ; ne pas utiliser ce petit pilote comme validation définitive de db2.

### 6.4 Gaussian interne sur l’ancien ResNet-20 / GroupNorm

Figure retrouvée : `plain_vs_gaussian_study.png`. Titre : 10 000 images, 1 200 updates, trois graines appariées, pic LR 0,005. Validation sur 5 000 images. Continuation terminée à l’update 600.

Le résultat final rapporté dans les échanges est **−3,45 points d’accuracy pour Gaussian contre plain**. Le graphe montre environ 50,5 % contre 47,0 %, avec une CE train autour de 1,3–1,4. Les valeurs exactes par graine ne sont pas présentes dans les fichiers retrouvés pour cette passation.

Un audit de placement puis un contrôle à deux learning rates renforcent le constat négatif dans cette recette, sans expliquer sa cause. Le second learning rate et les tableaux complets du contrôle ne sont pas restitués ici faute de source exacte.

Artifacts rapportés : `results/gaussian_placement_audit.json` et `results/kaggle_outputs/lr-control-002-20260908-140604/`. [Run LR-control](https://www.kaggle.com/code/maxnicaise/lr-control-002-20260908-140604).

### 6.5 Audit de l’architecture et de l’initialisation

Le placement interne gaussien a été jugé correct. En revanche, le réseau et sa recette différaient sensiblement de CBS : ResNet-20 / GroupNorm, entraînement court, extinction rapide du filtre et initialisation du classifieur trop ample par rapport à la référence.

Un classifieur dix sorties initialisé en Kaiming `fan_out` avec gain ReLU a un écart-type théorique

\[
\sqrt{2/10}\simeq0,447,
\]

contre 0,01 dans l’initialisation retenue ensuite, soit environ 45 fois plus. Cela peut produire des logits initialement très confiants et une CE élevée. C’est une explication plausible d’une partie des difficultés initiales ; cela n’isole pas la cause de l’effet du filtre.

Plusieurs modifications ont été faites ensemble : architecture, normalisation et initialisation. Il ne faut donc pas attribuer rétrospectivement le changement de signe à une seule d’entre elles.

### 6.6 Pilote ResNet-18 / BatchNorm, initialisation corrigée

Architecture : CIFAR stem 3×3 stride 1 sans max-pool ImageNet ; largeurs 64/128/256/512, deux BasicBlocks par stage ; projections 1×1+BN sur les raccourcis nécessaires ; global pooling et classifieur 512→10. **11 173 962 paramètres, 17 insertions gaussiennes, trois projections non filtrées, 20 couches BN.** Identifiant rapporté `resnet18_bn_cifar` ; fichier `continuation/models/resnet18_bn.py`.

Convolutions `fan_in`, BN 1/0, classifieur normal std 0,01 et biais nul. Même état initial appris, mêmes buffers BN et mêmes batches entre bras.

Protocole : mêmes 10 000 images train / 5 000 validation que le pilote précédent, graine 0, 1 200 updates, batch effectif 128 par quatre microbatches de 32, SGD pic 0,005, momentum 0,9, WD 0,0005, warmup 60 puis cosine vers zéro sur 1 200 updates, pas d’augmentation.

Calendrier :

\[
\sigma_k=\max(1-k/600,0),\qquad k=0,\ldots,1199.
\]

Les 600 dernières updates sont donc sans filtre.

| Bras | CE train-probe finale, bypassed | CE validation | Accuracy validation | Temps |
|---|---:|---:|---:|---:|
| Plain | 0,0530 | 1,5071 | 54,26 % | 151 s |
| Gaussian | 0,0987 | 1,1019 | 62,16 % | 197 s |

Gain **+7,90 points**, baisse de CE validation 0,4052. Temps T4 mesurés lors de la sonde : 110,8 ms/update plain, 175,3 ms/update Gaussian. Pics mémoire de sonde 345/672 MiB, pics rapportés sur les runs complets environ 1 439/1 442 MiB : ne pas mélanger les deux périmètres.

Cette expérience rend la piste prometteuse, mais reste une graine avec plusieurs changements simultanés. La CE train plus haute et la CE validation plus basse sont compatibles avec une régularisation implicite par la trajectoire ; elles ne prouvent pas un avantage de minimisation de la CE d’entraînement.

Dossier : `results/kaggle_outputs/resnet18-gaussian-complete/`. Figure : `results/resnet18_plain_vs_gaussian.png`. [Run ResNet-18](https://www.kaggle.com/code/maxnicaise/resnet18-gaussian-20260908-144640).

### 6.7 Retour au petit ResNet-20, BN et initialisation corrigée

But : vérifier si l’effet se conserve dans un modèle moins coûteux. Cela teste conjointement BN et la nouvelle initialisation sur le petit réseau ; ce n’est pas une séparation causale de leurs contributions.

Même split pilote 10 000 / 5 000, graine 0, même batch 32×4, même SGD pic 0,005 et warmup 60. Budget porté à **2 400 updates** ; cosine étendu sur cet horizon. Le checkpoint 1 200 n’a donc pas le même historique de LR que le précédent run de 1 200 updates.

Maxime a refusé de conserver le bypass à 600 et demandé une continuation jusqu’à **1 700**, puis **700 updates sans filtre**. Le calendrier exécuté est par paliers, de \(\sigma=1\) à 0,30, puis zéro. Les niveaux et certaines valeurs aux checkpoints sont connus ci-dessous ; **les bornes exactes de tous les paliers du pilote sont à relire dans son fichier de configuration**, absent des sources récupérées. Ne pas inventer ces bornes à partir de la seule figure.

| Bras | CE train-probe finale, bypassed | CE validation finale | Accuracy validation finale | Temps | Pic mémoire |
|---|---:|---:|---:|---:|---:|
| Plain | 0,7510 | 1,2335 | 55,36 % | 119 s | 394 MiB |
| Gaussian | 0,8887 | 1,1075 | 59,62 % | 174 s | 467 MiB |

Gain final **+4,26 points**, baisse de CE validation 0,1260.

Les premières figures mettaient en avant une comparaison bypassed trompeuse pour juger le prédicteur courant :

| Update | Plain accuracy | Gaussian bypassed | Différence |
|---:|---:|---:|---:|
| 600 | 43,66 % | 11,18 % | −32,48 points |
| 1 200 | 53,06 % | 25,82 % | −27,24 points |
| 1 700 | 54,80 % | 57,92 % | +3,12 points |
| 2 400 | 55,36 % | 59,62 % | +4,26 points |

La colonne `val_acc_filtered` avait déjà été enregistrée. Il n’a pas fallu reprendre les checkpoints pour retrouver les bonnes mesures ; la figure a été corrigée.

| Update | \(\sigma\) du diagnostic active | Gaussian active | Plain | Différence |
|---:|---:|---:|---:|---:|
| 200 | 1,00 | 36,86 % | 31,06 % | +5,80 points |
| 600 | 0,70 | 46,94 % | 43,66 % | +3,28 points |
| 1 000 | 0,60 | 53,74 % | 50,10 % | +3,64 points |
| 1 400 | 0,40 | 57,46 % | 54,14 % | +3,32 points |
| 1 700 | 0,30 | 58,20 % | 54,80 % | +3,40 points |
| 2 400 | 0 | 59,62 % | 55,36 % | +4,26 points |

Selon le compte rendu corrigé, Gaussian est devant à chaque checkpoint mesuré. L’explication de l’ancien graphique est bien une évaluation du chemin sans filtre alors que le réseau est entraîné avec filtre ; la formulation « entièrement dû à BN » allait plus loin que les contrôles effectués.

Dossier : `results/kaggle_outputs/resnet20bn-gaussian-20260908-154226/`. Figures : `results/resnet20bn_plain_vs_gaussian.png` puis `results/resnet20bn_pilot_corrected.png`. [Run ResNet-20 BN pilote](https://www.kaggle.com/code/maxnicaise/resnet20bn-gaussian-20260908-154226).

## 7. Dernière campagne complète : protocole figé et résultats

### 7.1 Données, optimiseur et appariement

Neuf runs : graines **0, 1 et 2**, chacune avec plain, Gaussian paliers et Gaussian géométrique comprimé. Architecture et opérateur sont ceux de la section 5.

| Élément | Valeur exécutée rapportée |
|---|---|
| Split | CIFAR-10 officiel, 50 000 train / 10 000 test |
| Durée | 30 époques, 391 updates/époque, 11 730 updates |
| Batch | Effectif 128 = quatre microbatches de 32 |
| Fin d’époque | 390 groupes de 128 puis un groupe de 80 exemples |
| Accumulation partielle | Pondération par le nombre réel d’exemples ; 80 = 32 + 32 + 16 |
| SGD | Pic LR 0,005, momentum 0,9, WD 0,0005, sans Nesterov |
| LR | Warmup **60 updates**, puis cosine sur l’horizon complet |
| Augmentation | Aucune |
| Normalisation des images | Statistiques recalculées une fois sur les 50 000 images train |
| Appariement | Poids initiaux, buffers BN et permutations par époque identiques entre bras d’une graine |
| Concurrence | Au maximum deux processus, un par T4 |

BatchNorm voit réellement des microbatches de 32, et 16 pour le dernier microbatch de l’époque, pas un batch physique de 128. La taille effective du gradient n’agrandit pas le lot utilisé pour ses statistiques.

Le précédent sous-ensemble de validation de 5 000 images appartient désormais à l’entraînement. Il ne peut plus être présenté comme un jeu de validation indépendant.

### 7.2 Calendriers exécutés

Avec \(e\) l’époque indexée à partir de zéro :

| Époques \(e\) | Gaussian paliers |
|---|---:|
| 0–2 | 1,00 |
| 3–5 | 0,85 |
| 6–8 | 0,70 |
| 9–11 | 0,60 |
| 12–14 | 0,50 |
| 15–17 | 0,40 |
| 18–20 | 0,30 |
| 21–29 | 0, bypass exact |

\[
\sigma_{\mathrm{geo}}(e)=\begin{cases}
0,9^e,&0\leq e<21,\\
0,&21\leq e<30.
\end{cases}
\]

Les deux bras ont **21 époques filtrées**, soit 8 211 updates, puis **neuf époques sans filtre**, soit 3 519 updates. Le géométrique conserve le facteur 0,9 mais le fait évoluer chaque époque ; ce n’est pas le calendrier exact de CBS.

Évaluations rapportées toutes les deux époques, avec les mesures nécessaires à 21 et 30. Les métriques active utilisent le sigma de la dernière update accomplie. Les figures montrent active en principal et bypassed en diagnostic. Les évaluations ne mettent pas à jour BN.

### 7.3 Résultats finaux à l’époque 30

| Bras | Accuracy moyenne ± SD | CE moyenne ± SD | Accuracy graine 0 | Graine 1 | Graine 2 |
|---|---:|---:|---:|---:|---:|
| Plain | 0,7540 ± 0,0040 | 0,7615 ± 0,0039 | 0,7510 | 0,7526 | 0,7585 |
| Paliers | 0,7841 ± 0,0032 | 0,6259 ± 0,0094 | 0,7827 | 0,7878 | 0,7819 |
| Géométrique comprimé | 0,7846 ± 0,0072 | 0,6636 ± 0,0255 | 0,7763 | 0,7884 | 0,7892 |

| Différence d’accuracy, en points | Graine 0 | Graine 1 | Graine 2 | Moyenne ± SD rapportées |
|---|---:|---:|---:|---:|
| Paliers − plain | +3,17 | +3,52 | +2,34 | +3,01 ± 0,61 |
| Géométrique − plain | +2,53 | +3,58 | +3,07 | +3,06 ± 0,53 |
| Géométrique − paliers | −0,64 | +0,06 | +0,73 | +0,05 ± 0,69 |

Les moyennes et SD des différences sont appariées par graine. La dernière ligne précise le sens du +0,05 point, ambigu dans la formulation originale « plateau vs geometric ». Il n’y a pas de classement net des deux calendriers en accuracy. La CE finale est plus basse pour les paliers ; cela ne démontre pas à elle seule une meilleure calibration au sens d’un diagnostic de calibration dédié.

### 7.4 Trajectoires et interprétation

Le bras paliers prend initialement du retard avec le filtrage fort puis passe devant le témoin vers les époques 16–18. La géométrique réduit le filtrage plus tôt et paraît passer devant le témoin plus tôt, vers 8–10 sur la figure. Le compte rendu disant que **les deux** restent derrière pendant environ 14 époques est trop global par rapport à la figure.

Le gain final persiste après la phase sans filtre. Il n’y a pas de saut visible majeur à l’extinction sur les points tracés ; cela ne prouve pas un coût exactement nul de la transition à chaque update.

Les deux bras filtrés atteignent une meilleure performance test avec une CE de sonde train généralement plus élevée que le témoin. C’est compatible avec une modification utile de la trajectoire et de la généralisation. Trois graines positives étayent ce résultat **dans cette recette** ; elles n’établissent pas un théorème général ni une supériorité sur des recettes CIFAR optimisées.

Le test officiel a été suivi au cours de l’entraînement, même si le budget et les calendriers avaient été figés et si le checkpoint final de 30 époques est utilisé. Il faut donc parler de **suivi exploratoire du test**, pas d’un jeu de test resté totalement invisible jusqu’à une validation finale. Ne pas sélectionner rétrospectivement le meilleur epoch et ne pas oublier cette exposition lors d’une future conclusion confirmatoire.

Ne pas agréger les pilotes 10 000 images avec cette campagne. Ne pas comparer causalement les 84 % de l’ancienne expérience d’entrée à ces 75–78 % : données, durée, LR, initialisation et normalisation diffèrent.

### 7.5 Coût et fichiers

Temps moyens rapportés par run : plain environ 458 s, paliers 693 s, géométrique 684 s. Cela représente environ **+51 %** et **+49 %** de temps par rapport au témoin dans cette campagne. Pic mémoire entre 511 et 584 MiB. Somme exacte rapportée 5 504 s ; la somme à partir des temps moyens arrondis diffère légèrement.

Dossier : `results/kaggle_outputs/fulldata-r20bn-20260908-161221/`. [Run de la campagne complète](https://www.kaggle.com/code/maxnicaise/fulldata-r20bn-20260908-161221). Figure `results/fulldata_campaign.png`. Checkpoints reprenables aux époques 21 et 30 ; fichiers partagés d’initialisation et de permutations. Les fichiers bruts de cette campagne n’ont pas été récupérés ici : les résultats détaillés reposent sur le compte rendu et la figure fournis.

## 8. Sinha et al. : ce qui motive l’expérience et ses limites

Référence : Sinha, Garg et Larochelle, *Curriculum by Smoothing* (CBS), NeurIPS 2020. [Article](https://arxiv.org/html/2003.01367v5), [dépôt officiel](https://github.com/pairlab/CBS).

Les enseignements utiles pour notre phase exploratoire sont les suivants : le filtrage des caractéristiques est plus favorable que celui des entrées dans leurs ablations ; garder un flou constant ne reproduit pas le bénéfice du calendrier ; les résultats varient selon les architectures et les réglages ; des expériences de transfert étendent l’intérêt au-delà de la seule accuracy de classification. Ces résultats soutiennent des comparaisons ciblées, mais ne démontrent pas que notre chemin db2 sera utile ni que le paysage de perte devient convexe. [Article, expériences et ablations](https://arxiv.org/html/2003.01367v5)

La recette de référence examinée utilise des noyaux 3×3, un sigma initial de 1 et un facteur 0,9 toutes les cinq époques pour les expériences concernées. Notre noyau 9 taps avec réflexion, le petit ResNet-20, les microbatches de 32 et les calendriers comprimés sont des différences réelles. **« Inspiré de CBS » désigne correctement notre comparaison ; « reproduction exacte » ne la décrit pas.** [Code ResNet](https://github.com/pairlab/CBS/blob/main/resnet.py), [paramètres](https://github.com/pairlab/CBS/blob/main/arguments.py), [entraînement](https://github.com/pairlab/CBS/blob/main/solver_cbs.py)

En particulier, l’ablation défavorable au flou d’entrée n’interdit pas toute transformation TV ou ondelette sur les images. Elle indique seulement qu’il faut distinguer clairement le lieu d’application et vérifier son utilité.

## 9. Idées conservées, mais pas réalisées comme campagnes

### 9.1 Contrôle de l’utilité des étapes intermédiaires

Comparer, dans la **configuration interne actuelle**, un filtre fixe suivi de son retrait à une continuation progressive, avec durée totale, initialisation, ordre des données et phase finale sans filtre comparables. La comparaison plain / Gaussian seule montre un bénéfice du protocole global, pas nécessairement de chaque étape intermédiaire.

Une telle comparaison a été faite pour le **flou d’entrée** (expérience 1), mais pas rapportée comme faite pour le filtrage interne BN actuellement positif.

### 9.2 TV à coût fixe

Source : `note_tv_cout_fixe.md`. Proposition réservée pour éviter la résolution coûteuse de la projection exacte. Pour \(M\) petit, par exemple 3 ou 5 :

\[
E_\varepsilon(u)=\sum_p\sqrt{\|(Du)_p\|_2^2+\varepsilon^2},\quad u^{(0)}=h,
\]

\[
u^{(m+1)}=u^{(m)}-\eta(s)D^\top\!left(
\frac{Du^{(m)}}{\sqrt{\|Du^{(m)}\|_2^2+\varepsilon^2}}\right),
\qquad T_s(h)=u^{(M)},\quad\eta(s)=(1-s)\eta_0.
\]

L’epsilon reste fixé. Avec \(\|D\|^2\leq8\), une borne conservative \(\eta_0\leq\varepsilon/8\) permet de contrôler la descente de l’énergie lissée. \(s=1\) est l’identité et les moyennes sont conservées par la divergence avec ces bords. Ce serait **un opérateur explicite à nombre fixé d’étapes**, pas la projection TV exacte, pas automatiquement son proximal, et pas le solveur \(\dot H^{-1}\). Aucun résultat de classification de cette variante n’est établi.

### 9.3 Résolution progressive et coût du réseau

Source : `note_filtrage_resolution_progressive.md`. Réduire réellement la taille des images avec antialiasing, puis l’augmenter, peut réduire le coût des convolutions. Passer de 32×32 à 16×16 réduit approximativement par quatre les opérations spatiales des couches concernées, sans garantir un facteur quatre de temps total. Le champ réceptif relatif change aussi.

Une construction envisagée : \(x_r=D_r(G_{\sigma(r)}*x)\). Le dernier niveau doit revenir exactement à l’image cible. Il faut distinguer traitement à basse résolution, remontée préalable vers une grande grille, et mélange de logits

\[
q_{\theta,\alpha}=(1-\alpha)f_\theta(x_r)+\alpha f_\theta(x_{2r}).
\]

Cette dernière transition utilise deux forwards. La CE des logits mélangés n’est pas la moyenne des deux CE. L’interpolation d’une image basse résolution ne permet pas d’identifier \(f_\theta(Ux)\) à \(f_\theta(x)\). Le seuillage ondelette n’est pas un antialiasing idéal garanti. Cette piste reste distincte du remplacement de Gaussian par db2 à taille de cartes identique.

### 9.4 Autres axes documentés dans la synthèse

Interpolation d’objectifs ; régularisation décroissante ; lissage en espace des paramètres ; curriculum d’exemples ; activation interpolant identité et ReLU ; suivi adaptatif predictor-corrector ; apprentissage d’une représentation du chemin des solutions ; budgets opérationnels de compression. Ces axes appartiennent à la cartographie du projet et ne sont pas des expériences CIFAR achevées.

Pour la compression, un paramètre de qualité de codec n’est pas un nombre de bits mesuré, ni une fonction de Shannon attachée à une image isolée. Le endpoint exact, les discontinuités et le coût doivent être précisés si cette piste est reprise.

Le fichier `Homotopy.pdf` retrouvé porte sur *Ideal Low-Pass Density Design for Homotopic Off-the-Grid Deconvolution*, Joseph Gabet, Maxime Ferreira Da Costa et Kiryung Lee. Il étudie le choix d’une densité d’échantillonnage fréquentiel et une continuation par bandes avec variable projection en déconvolution. Cette géométrie spécifique peut inspirer des idées ; ses résultats ne se transfèrent pas automatiquement aux CNN.

Les clés bibliographiques retrouvées dans la synthèse sont : `LENDL1998359`, `amakor2025continuation`, `hazan2016graduated`, `sato2025explicit`, `bengio2009curriculum`, `gulcehre2017mollifying`, `yang2025homotopy`, `roulet2021smoothing`, `sinha2020curriculum`, `rahaman2019spectral`, `mai2026neural`, `Lin2023ContinuationPL`. Le fichier `biblio.bib` n’a pas été retrouvé ; il faut vérifier les notices et les affirmations dans les sources avant une rédaction scientifique finale.

## 10. Propositions remplacées : ne pas les relancer

- La campagne de 200 époques sur ResNet-18, quatre bras et graines 1/2, a été proposée puis rejetée pour son coût.
- Une campagne réduite de 30 époques sur ResNet-18 a aussi été proposée, puis remplacée par le retour au ResNet-20.
- Le pilote ResNet-20 à 2 400 updates avec filtre éteint à 600 a été rejeté par Maxime et remplacé par l’extinction à 1 700.
- Les anciennes propositions avec warmup de 5 % du budget ne décrivent pas la campagne complète exécutée : celle-ci garde **60 updates de warmup**.
- Les nouveaux entraînements TV et ondelettes ont été laissés en attente pendant la validation gaussienne. L’ancien pilote db2 existe, mais **aucune grande campagne db2 après les résultats complets gaussiens n’est rapportée comme lancée**.
- Les recettes de référence avec LR 0,1 et batch physique 64 ne sont pas les paramètres de la dernière campagne, qui utilise LR 0,005 et microbatches 32×4.

## 11. Ce qu’il faut reprendre ensuite, sans redémarrer le projet

La prochaine discussion doit partir du résultat positif répliqué sur ResNet-20 BN et de l’opérateur db2 de la section 3. Elle ne doit pas demander à Maxime de réexpliquer les ondelettes ou remplacer cette formule par une variante générique.

Avant un éventuel nouveau run db2, le travail utile est de retrouver son implémentation dans le dépôt de l’agent, vérifier l’équivalence à la configuration sauvegardée, puis chronométrer forward/backward et updates complètes avec les **microbatches de 32** sur le matériel visé. Mesurer le modèle entier avec les insertions prévues : les mesures anciennes à batch 128 ne suffisent pas.

Garder en tête que \(s\) augmente vers 1 alors que \(\sigma\) diminue vers zéro. Des valeurs numériques identiques de \(s\) et \(\sigma\) ne représentent pas une simplification comparable. Les ratios TV, le contraste et les activations peuvent aider à décrire la force du filtrage sans prouver une équivalence sémantique.

Pour une comparaison contrôlée au protocole gaussien actuel, les choix naturels à garder sont le ResNet-20 BN, l’initialisation corrigée, les données, le batch 32×4, l’optimiseur, les seeds appariées et les figures active/bypassed. Le calendrier db2 précis et le budget acceptable restent à arrêter sur la base de son coût : **ce document ne vaut pas lancement ni nouveau protocole autorisé**.

Informations qui restent à récupérer auprès du code ou des runs :

1. Dépôt et révision exacts du code utilisé, notamment db2, driver et runner Kaggle.
2. Emplacement de l’epsilon et traitement du gradient de la RMS dans db2 ; convention précise de l’extension miroir.
3. Calendrier complet du premier pilote db2, ses métriques brutes et son timing.
4. Bornes exactes de tous les paliers du pilote ResNet-20 BN à 2 400 updates.
5. Valeurs exactes de normalisation du train complet, fichiers d’indices, identifiants des graines et logs bruts des neuf runs.
6. Tableau complet du contrôle à deux learning rates et révision du code CBS auditée.
7. Critère d’arrêt exact du PDHG et définition de ses résidus normalisés si TV est repris.

Les fichiers Windows cités dans les rapports se trouvaient sous un projet tel que `C:\Users\mnica\Documents\Projet_filiere\`. Cela ne signifie pas que ce dépôt est accessible dans la nouvelle conversation. Les chemins `results/...` ci-dessous sont des repères dans ce projet, pas des fichiers dont la présence locale est garantie.

## 12. Inventaire des sources retrouvées et provenance

### 12.1 Fichiers relus ou inspectés pendant la passation

| Fichier | Contenu utile |
|---|---|
| `Sujet24_Joseph_Gabet.pdf` | Sujet de projet et objectifs |
| `Homotopy.pdf` | Article de déconvolution homotopique du groupe de Gabet |
| `homotopy_synthesis.tex` | Synthèse théorique, auteurs, axes et clés bibliographiques |
| `prompt_homotopie_images.md` | Spécification initiale du framework et de l’expérience 0 |
| `results.md` | Résultats complets de l’expérience 0 |
| `exp0_curves.png` | Figure de l’expérience 0, récupérée comme source complémentaire |
| `prompt_exp1_warm_start.md` | Protocole de branchement et budgets de l’expérience 1 |
| `exp1_warmstart.md` | Résultats, contrôles de reprise et coûts de l’expérience 1 |
| `exp1_adaptation.png`, `exp1_equal_budget.png` | Figures de l’expérience 1, récupérées comme sources complémentaires |
| `tv_preview_diagnostics.json` | Configurations TV, résidus, itérations et contraste |
| `transform_levels_tv_l2.png`, `transform_levels_tv_hminus1.png` | Aperçus TV sauvegardés |
| `wavelet_preview_diagnostics.json` | **Définition exacte de la famille db2** et diagnostics des quatre familles |
| `wavelet_timings.json` | Coûts CPU/GPU des ondelettes et de Gaussian |
| `transform_levels_wavelet_haar.png`, `transform_levels_wavelet_db2.png`, `transform_levels_wavelet_sym4.png`, `transform_levels_wavelet_coif1.png` | Grilles ondelettes sauvegardées |
| `note_tv_cout_fixe.md` | Variante TV explicite réservée et correction des coûts |
| `note_filtrage_resolution_progressive.md` | Piste de réduction de résolution et transitions |
| `pilot_comparison.png` | Ancien pilote 5 000 images / 600 updates avec db2 |
| `plain_vs_gaussian_study.png` | Ancienne comparaison interne 10 000 / 1 200, trois graines |
| `resnet18_plain_vs_gaussian.png` | Pilote ResNet-18 positif |
| `resnet20bn_plain_vs_gaussian.png` | Pilote ResNet-20 BN, figure initiale avec bypassed en principal |
| `fulldata_campaign.png` | Dernière campagne complète avec active en principal |

Les comptes rendus récents copiés dans la conversation sont la source des tableaux exacts des pilotes BN et de la campagne complète. Les figures ont servi à vérifier la lecture des trajectoires. Les JSON de ces derniers entraînements, les checkpoints et le dépôt d’exécution ne figurent pas parmi les fichiers récupérés pour ce document.

Les fichiers peuvent être recherchés par les noms exacts du tableau. Le contenu critique a été recopié ici pour que cette passation reste utilisable sans accès aux anciens fichiers.

### 12.2 À propos de la fenêtre de contexte et de cette passation

La documentation officielle consultée pour cette demande annonce **1 050 000 tokens de fenêtre de contexte pour le modèle API GPT-6 Astra**. Cela ne permet pas d’affirmer que cette session Work expose exactement cette même capacité, ni que l’ensemble des anciens messages est actuellement présenté mot pour mot au modèle. Le réglage d’effort « très élevé » concerne le raisonnement ; ce n’est pas une garantie de conservation verbatim de l’historique. Aucun compteur fiable de la fenêtre réellement allouée à cette session n’était accessible. [Modèle API](https://developers.openai.com/api/docs/models/gpt-6-astra), [modèles et effort dans Work](https://learn.chatgpt.com/docs/models).

L’erreur pratique à corriger était aussi une erreur de recherche : l’assistant ne retrouvait plus db2 dans le contexte visible alors que des fichiers sauvegardés en contenaient la définition et les mesures. Cette passation a retrouvé cette source. Il faut désormais rechercher les artifacts pertinents avant de demander à Maxime de recopier une formule ou de prétendre s’en souvenir.

## 13. Texte court pour ouvrir la prochaine conversation

> Lis intégralement cette passation avant de répondre. Nous avons terminé une campagne CIFAR-10 complète sur ResNet-20 BatchNorm, trois graines, 30 époques : les deux continuations gaussiennes gagnent environ 3 points contre le témoin. Je veux discuter de la reprise de notre méthode db2 déjà définie, pas d’une nouvelle méthode générique d’ondelettes. Sa formule exacte, les anciens coûts et toute la chronologie sont dans ce fichier. Les courbes avec filtres actifs sont les courbes principales ; bypassed est un diagnostic. Distingue les résultats établis des détails de code encore à vérifier, et ne lance aucun entraînement sur la seule base de ce document.
