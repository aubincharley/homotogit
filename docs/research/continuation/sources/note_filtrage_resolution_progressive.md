# Piste en réserve : filtrage et résolution progressive

Décision du 8 septembre 2026 — projet de continuation / homotopie en apprentissage profond avec Joseph Gabet.

## Intention et statut

Max souhaite explorer une continuation par prétraitement des entrées combinant filtrage et réduction effective de résolution, avec retour progressif aux images originales. Deux familles de filtres sont envisagées : gaussien et ondelettes. Cette piste correspond particulièrement à son intention initiale. Elle est explicitement mise de côté pendant l'analyse des aperçus et coûts des ondelettes ; aucun entraînement ni calendrier n'est autorisé par cette note.

Les expériences précédentes sur CIFAR-10 ont trouvé un léger désavantage de validation pour les calendriers gaussiens sur entrées à résolution constante, à budget égal. Cela ne condamne pas la réduction effective de résolution ni toute continuation par prétraitement.

## Prétraitement à résolution variable

Pour une image originale x, définir une famille d'observations x_r à résolution r×r, par exemple

\[
x_r=\mathcal D_r(G_{\sigma(r)}*x),
\]

où Gσ est le filtre gaussien et D_r le sous-échantillonnage / redimensionnement. Au stade final r=H=W, retrouver x exactement : σ=0 et D_r=I.

Le filtre limite le repliement spectral avant sous-échantillonnage. Un seuil en ondelettes peut remplacer le filtrage pour explorer une réduction adaptative des détails, mais seuiller des coefficients ne garantit pas une bande limitée : l'anti-aliasing du redimensionnement doit rester explicite. Ne pas confondre un filtrage gaussien, un seuillage d'amplitudes en ondelettes et une coupure fréquentielle idéale.

## Pas besoin de redimensionner les poids du CNN

Les noyaux convolutifs dépendent des nombres de canaux et de leur support spatial, pas des dimensions H×W des activations. Un CNN avec moyenne spatiale globale/adaptative et tête de classification fixe peut donc partager exactement les mêmes paramètres entre plusieurs résolutions. Vérifier l'implémentation du pooling final et les éventuelles tailles codées en dur.

Dans un ResNet20 classique, passer d'une entrée 32×32 à 16×16 donne des cartes successives de tailles spatiales 16, 8, 4 au lieu de 32, 16, 8. Les canaux restent 16, 32, 64 ; après moyenne globale, le vecteur reste de dimension 64. Cela réduit le calcul spatial et change le champ réceptif relativement à l'image. Pour les convolutions correspondantes, diviser chaque dimension par deux divise approximativement les opérations par quatre ; le gain de temps réel reste à mesurer.

## Transitions entre résolutions : options à distinguer

1. **Prétraitement pur et architecture fixe.** Garder le CNN et ses paramètres ; changer r par étapes, éventuellement via plusieurs résolutions intermédiaires. Aucun ajout de blocs n'est nécessaire. Des dimensions entières impliquent des étapes discrètes.

2. **Interpolation des images sur une grille commune.** Pour une transition r→2r, utiliser (1−α)U(x_r)+αx_{2r}, avec U le suréchantillonnage. C'est une interpolation continue des entrées, mais fθ(U(x_r)) n'est généralement pas égal à fθ(x_r). Dès le début de cette transition, le réseau travaille sur la grande grille : on ne conserve ni exactement la fonction précédente ni son coût.

3. **Interpolation des logits, poids partagés.** Notre adaptation possible, distincte de Progressive GAN :

\[
q_{\theta,\alpha}(x)=(1-\alpha)f_\theta(x_r)+\alpha f_\theta(x_{2r}),
\qquad \alpha\in[0,1],
\]

où fθ renvoie les logits de classification. Entraîner avec l'entropie croisée de qθ,α. Les deux extrémités retrouvent exactement les réseaux aux résolutions correspondantes, et l'objectif est continu en α. Les deux forward sont nécessaires pendant le mélange ; une seule branche suffit aux extrémités. Ce n'est plus strictement du prétraitement seul et ce n'est pas la même chose qu'une interpolation de deux pertes.

## Précédents bibliographiques

- [Karras et al., Progressive Growing of GANs, ICLR 2018](https://arxiv.org/html/1710.10196v3#S2) : générateur et discriminateur grandissent par ajout de blocs. L'α de la figure 2 mélange des chemins ramenés aux mêmes dimensions, et non des matrices de poids de tailles différentes. Les anciennes couches restent entraînables. Le contexte est la génération adversariale, pas la classification CIFAR-10.
- [Tan et Le, EfficientNetV2, ICML 2021](https://proceedings.mlr.press/v139/tan21a.html) : résolution progressive et adaptation conjointe de la régularisation en classification. Un précédent plus direct pour cette tâche ; ce n'est pas une garantie de gain dans notre protocole.

## Avant de lancer des expériences ultérieures

Fixer l'objectif : précision à nombre d'updates égal, ou efficacité à temps / calcul égal. Ne pas comparer uniquement des courbes alignées sur le début du stade final en oubliant le coût des stades précédents. Garder le filtrage interne comme piste distincte pour ne pas attribuer un gain à plusieurs changements simultanés. Sur CIFAR-10 en 32×32, la marge de réduction avant perte de détails discriminants est limitée.
