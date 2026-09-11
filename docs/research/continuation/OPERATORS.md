---
id: DOC-OPERATORS
schema_version: 1
updated_at: 2026-09-11
status: definitions_and_reported_implementation
---

# Opérateurs : formules, lieux et conventions

[Accueil](README.md) · [Protocoles](PROTOCOLS.md) · [Évaluation](EVALUATION.md)

## Sommaire

1. [Paramètres et endpoints](#op-endpoints)
2. [Gaussian discret](#op-gaussian)
3. [Résolution et emplacement](#op-resolution)
4. [Bilinéaire et antialiasing](#op-bilinear)
5. [Max-pooling](#op-maxpool)
6. [Couplage Gaussian–résolution](#op-coupling)
7. [Mélange Gmix](#op-gmix)
8. [Ondelette db2](#op-db2)
9. [TV–L² et TV–Ḣ⁻¹](#op-tv)
10. [Variantes non exécutées](#op-proposals)

Les formules ci-dessous restituent les spécifications et les comptes rendus disponibles. Les détails d'implémentation non exposés par les sources sont signalés ; ce dossier ne constitue pas un audit du dépôt absent.

<a id="op-endpoints"></a>

## 1. Paramètres et endpoints

| Opérateur | Paramètre | Retour exact à la cible | Lieu testé |
|---|---|---|---|
| Gaussian | \(\sigma\geq0\), pixels de la grille où il agit | \(\sigma=0\) | Entrées dans les premières expériences ; cartes internes ensuite |
| Résolution | \(r\in\{16,24,32\}\) | \(r=32\) | Une réduction sur RGB ou après le stem |
| Gmix | \(\alpha\in[0,1]\) | \(\alpha=0\) | Sites internes sélectionnés |
| db2 | \(s\in[0,1]\) | \(s=1\) | Aperçus RGB puis cartes internes |
| TV | \(t\in[0,1]\), budget relatif | \(t=1\) | Aperçus RGB uniquement |

Ne pas imposer au framework que tous les paramètres décroissent vers zéro. Un endpoint exact doit être un bypass explicitement défini quand cela convient, pas une approximation « assez faible » arbitraire.

<a id="op-gaussian"></a>

## 2. Gaussian discret

### 2.1 Noyau interne actuel

Pour \(\sigma>0\), rayon \(R=4\),

\[
w_\sigma[i]=\frac{\exp(-i^2/(2\sigma^2))}
{\sum_{j=-4}^{4}\exp(-j^2/(2\sigma^2))},\quad i=-4,\ldots,4.
\]

Appliquer les convolutions 1D horizontale et verticale indépendamment à chaque canal. Le noyau 2D est le produit extérieur \(w_\sigma w_\sigma^\top\). Il y a neuf coefficients par axe, un support fixe, et un prolongement réfléchi. À \(\sigma=0\), renvoyer le tenseur sans filtrage. Aucun mélange entre canaux, aucun filtre des logits.

**Portée de cette définition :** c'est le Gaussian interne des expériences récentes, dont \(\sigma_{max}=1\). Les premières expériences sur RGB atteignent \(\sigma=3\) et utilisent un support adapté plus large. Les timings historiques distinguent explicitement `gaussian_sigma_max3` et `gaussian_sigma_max1`. Ne pas réattribuer neuf taps au premier opérateur d'entrée sans sa configuration d'origine.

### 2.2 Insertion dans les réseaux

| Site | Ordre |
|---|---|
| Stem | convolution → Gaussian → normalisation → ReLU |
| Première convolution d'un BasicBlock | convolution → Gaussian → normalisation → ReLU |
| Seconde convolution du bloc | convolution → Gaussian → normalisation → addition du raccourci → ReLU |

Sur ResNet-20 : **19 sites**, soit le stem et deux convolutions dans chacun des neuf blocs. Sur ResNet-18 : **17 sites**, stem et deux convolutions dans chacun des huit blocs. Les projections 1×1 des raccourcis de ResNet-18 ne sont pas filtrées. Les raccourcis option A du ResNet-20 ne reçoivent pas non plus de filtre propre.

Dans la variante `early7`, seuls le stem et les six convolutions du premier stage reçoivent Gaussian. Les 12 convolutions principales restantes restent inchangées. Ce n'est ni sept stages ni un filtrage uniquement des entrées.

### 2.3 Réflexion sur petites cartes

Le padding réfléchi natif PyTorch exige un padding inférieur à la dimension. Le rayon 4 nécessite donc un fallback sur les cartes 4×4, rencontrées dans ResNet-18 et dans ResNet-20 alimenté en 16×16.

Pour \(n>1\), définir

\[
P_n=2(n-1),\quad m=i\bmod P_n,\quad r_n(i)=\min(m,P_n-m).
\]

Les indices de padding sont \(r_n(i)\) pour \(i=-4,\ldots,n+3\). À \(n=4\), la séquence est

```text
[2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1]
```

Cette réflexion ne répète pas l'échantillon de bord. Utiliser des opérations d'indexation différentiables, puis une convolution sans ajouter une seconde fois du padding. Garder le chemin natif lorsqu'il est valide. Aucun cas \(n=1\) n'est défini par la formule ; il faudrait le spécifier si une nouvelle architecture l'atteint.

### 2.4 Interprétation continue, à ne pas confondre avec l'implémentation

En espace continu, le multiplicateur fréquentiel est \(\exp(-\sigma^2\|\omega\|^2/2)\), avec temps de chaleur \(a=\sigma^2/2\). Le semi-groupe continu en \(a\) ne se transfère pas exactement au noyau discret normalisé, tronqué et réfléchi. Une décroissance linéaire en \(\sigma\) n'est pas une décroissance linéaire en \(a\).

<a id="op-resolution"></a>

## 3. Résolution et emplacement

### 3.1 Une réduction effective, une seule fois

L'opérateur \(R_r\) transforme une image ou une carte de 32×32 en une grille \(r\times r\). **Il n'y a pas de suréchantillonnage vers 32 avant la suite du réseau.** Les poids du réseau ne changent pas de dimensions.

| Identifiant du benchmark | Pipeline rapporté | Nombre de réductions ajoutées |
|---|---|---:|
| `input_bilinear` | RGB float → resize bilinéaire AA → normalisation → stem | 1 |
| `input_max` | RGB float → max-pooling adaptatif → normalisation → stem | 1 |
| `stem_bilinear` | RGB 32 → normalisation → conv stem → Gaussian éventuel → BN → ReLU → resize bilinéaire AA | 1 |
| `stem_max` | RGB 32 → normalisation → conv stem → Gaussian éventuel → BN → ReLU → max-pooling adaptatif | 1 |

La sortie réduite après stem entre ensuite dans le premier stage résiduel. Les réductions de stride déjà présentes dans l'architecture sont conservées ; elles ne sont pas des insertions supplémentaires de \(R_r\). L'ordre exact après stem est une convention à rattacher aux fichiers du commit lors de l'intégration : le JSON final ne contient pas le code du forward.

### 3.2 Pourquoi les dimensions restent compatibles

Les noyaux convolutifs ont des dimensions fixées par les canaux et le support local. Ils peuvent s'appliquer à différentes tailles spatiales. À l'intérieur de chaque bloc, les chemins principal et raccourci restent de même forme. La moyenne spatiale globale ramène finalement les cartes à un vecteur de 64 canaux.

| Résolution à l'entrée du stage 1 | Stage 1 | Stage 2 | Stage 3 | Après moyenne globale | Classifieur |
|---:|---:|---:|---:|---:|---:|
| 16 | 16×16, 16 canaux | 8×8, 32 canaux | 4×4, 64 canaux | 64 | 64→10 |
| 24 | 24×24, 16 canaux | 12×12, 32 canaux | 6×6, 64 canaux | 64 | 64→10 |
| 32 | 32×32, 16 canaux | 16×16, 32 canaux | 8×8, 64 canaux | 64 | 64→10 |

Le paramétrage reste de **269 722 poids/biais appris**. On change la grille des activations et le champ réceptif relatif, pas la taille de la dernière matrice de classification. Une architecture qui aplatirait une carte spatiale entière dans une couche linéaire fixe demanderait une autre analyse ; ce n'est pas notre modèle.

### 3.3 Pipeline de données et endpoint

Les images stockées en `uint8` sont d'abord converties en float et divisées par 255. La réduction d'entrée opère avant la normalisation de canaux. Chaque niveau repart des images originales. Les statistiques de normalisation du split train complet restent identiques entre résolutions.

Le pilote a corrigé un bug où le resize était appelé sur `uint8`. À \(r=32\), `resize_to(x,32)` renvoie l'objet tenseur original dans le contrôle rapporté : le chemin cible n'ajoute pas une interpolation résiduelle.

<a id="op-bilinear"></a>

## 4. Bilinéaire et antialiasing

### 4.1 Linéarité par rapport aux valeurs de l'image

Pour une taille et des conventions fixées, la réduction bilinéaire est une application linéaire. Canal par canal, on peut écrire

\[
Y=A_r X A_r^\top,
\quad X\in\mathbb R^{32\times32},\quad A_r\in\mathbb R^{r\times32},
\quad Y\in\mathbb R^{r\times r}.
\]

En particulier, \(R_r(aX+bZ)=aR_r(X)+bR_r(Z)\). Le mot « bilinéaire » décrit l'interpolation dans les coordonnées spatiales ; il ne signifie pas que l'application aux intensités soit non linéaire. Les produits matriciels sont une représentation mathématique ; ils ne signifient pas que le code construit une grosse matrice dense.

### 4.2 L'antialiasing change les poids, pas cette linéarité

La convention discutée et retenue est bilinéaire, `align_corners=False`, `antialias=True`. Pour une réduction, l'antialiasing élargit l'ensemble d'échantillons qui contribue à une valeur et adapte les poids à l'échelle. Ces coefficients restent indépendants de l'image : le résultat est toujours de la forme ci-dessus.

Il n'est donc pas exact de remplacer le resize exécuté par « moyenne de chaque bloc 2×2 » sans préciser les conventions. Cette moyenne décrit le cas bilinéaire 32→16 sans antialiasing avec centres correspondants ; la variante AA utilise un support plus large. Les coefficients exacts, particulièrement aux bords et pour 32→24, doivent être extraits de la version de la bibliothèque utilisée si l'on veut une reproduction numérique.

L'antialiasing du resize **n'est pas le Gaussian interne** et n'est pas forcément implémenté comme un préfiltre gaussien séparé. Il limite le repliement fréquentiel associé au sous-échantillonnage, mais il ne constitue pas un filtre idéal supprimant strictement toute fréquence au-delà d'une coupure.

<a id="op-maxpool"></a>

## 5. Max-pooling

Pour une réduction 32→16 par blocs 2×2, chaque canal vérifie

\[
(R^{max}_{16}X)_{p,q}
=\max_{a,b\in\{0,1\}}X_{2p+a,2q+b}.
\]

Il conserve la valeur numériquement la plus grande du bloc, pas sa moyenne. C'est une transformation **non linéaire** ; « signal le plus fort » est une intuition à préciser. Sur des activations signées, le maximum n'est pas le maximum en valeur absolue. Sur RGB, les maxima de canaux différents peuvent provenir de pixels différents, ce qui peut produire une couleur absente du bloc initial. Après ReLU du stem, les valeurs ont un sens de caractéristiques apprises différent de celui des intensités RGB.

Pour les autres tailles, le benchmark utilise un max-pooling adaptatif. Une fenêtre 1D correspondant à l'indice \(i\) est conventionnellement délimitée par

\[
\left\lfloor\frac{iH}{r}\right\rfloor,
\quad \left\lceil\frac{(i+1)H}{r}\right\rceil
\]

(borne droite exclue). À 32→24, certaines fenêtres se chevauchent ; ce n'est pas une partition uniforme en blocs 2×2. Vérifier cette convention dans la bibliothèque du run pour toute réimplémentation.

Le max-pooling testé est **sans antialiasing ajouté**. « Bilinéaire AA contre max » change donc à la fois la règle d'agrégation et le traitement du repliement spectral. Ajouter un filtre avant le max définirait une autre variante.

<a id="op-coupling"></a>

## 6. Couplage Gaussian–résolution

Soit \(g(e)\) la largeur de référence du calendrier à l'époque \(e\). À un site interne \(\ell\), poser

\[
q_\ell(e)=\frac{\text{largeur spatiale courante au site }\ell}
{\text{largeur à ce même site avec une entrée 32×32 sans réduction}},
\qquad \sigma_\ell(e)=q_\ell(e)g(e).
\]

- Réduction en entrée : tous les sites ont \(q=r/32\).
- Réduction après stem : le filtre du stem agit avant la réduction et garde \(q=1\) ; les sites ultérieurs utilisent \(q=r/32\).
- Résolution constante 32 : \(q=1\) partout, y compris dans les stages dont la carte est naturellement 16 ou 8 pixels de côté.

Il ne faut donc pas multiplier une seconde fois sigma par le stride cumulé du réseau. Le dénominateur est la taille **au même site**, pas toujours 32.

Pour `Rprog` et les paliers de trois époques, les sigmas effectifs en entrée sont :

| Époques zéro-indexées | r | g de référence | Sigma effectif |
|---|---:|---:|---:|
| 0–2 | 16 | 1,00 | 0,500 |
| 3–5 | 16 | 0,85 | 0,425 |
| 6–8 | 24 | 0,70 | 0,525 |
| 9–11 | 24 | 0,60 | 0,450 |
| 12–14 | 32 | 0,50 | 0,500 |
| 15–17 | 32 | 0,40 | 0,400 |
| 18–20 | 32 | 0,30 | 0,300 |
| 21–29 | 32 | 0 | 0 |

Le sigma effectif **n'est pas monotone** aux changements de résolution. C'est un effet de la convention exécutée, pas une erreur de lecture. Elle vise à tenir compte de l'échelle spatiale ; elle ne garantit pas une équivalence spectrale ou sémantique entre niveaux.

L'inversion de l'ordre 16/24 modifie ainsi aussi la suite des sigmas effectifs. Le contrôle `Rreverse` avec Gaussian ne peut pas être décrit comme un changement d'ordre de résolution à tout le reste identique au niveau des opérateurs effectifs.

<a id="op-gmix"></a>

## 7. Mélange Gmix

La variante conserve un Gaussian de largeur de référence fixe 1, ajustée par \(q_\ell\), et réduit son poids dans un mélange avec l'identité :

\[
M_{\alpha,q}(h)=(1-\alpha)h+\alpha G_qh.
\]

\(\alpha(e)\) prend les niveaux du calendrier paliers 1/0,85/0,70/0,60/0,50/0,40/0,30, puis 0 à partir de l'époque 21. À \(\alpha=0\), bypass exact. C’est la définition de travail restituée de la discussion. Le JSON confirme le nom et les résultats de Gmix, mais ne prouve pas à lui seul cette formule : le script du commit est encore à joindre pour la confirmer.

En approximation continue et à grille fixée, le multiplicateur serait

\[
(1-\alpha)+\alpha\exp(-q^2\|\omega\|^2/2),
\]

différent de celui d'un Gaussian de largeur décroissante. Les valeurs d'alpha et de sigma n'ont pas été calibrées pour produire une force de filtrage égale. Le résultat défavorable de Gmix concerne cette paire opérateur/calendrier.

<a id="op-db2"></a>

## 8. Ondelette db2

### 8.1 Famille précise

Soit \(h\) une image ou une carte. Pour chaque échantillon et canal, l'étendre par miroir vers \(2H\times2W\) avec \(E\), effectuer une transformée **stationnaire, non décimée, séparable, à deux niveaux** sur le domaine périodique étendu, puis recadrer avec \(P\).

\[
WEh=(a_2,\{d_{j,o}\}_{j=1,2;\ o\in\{LH,HL,HH\}}).
\]

Les filtres d'analyse 1D sont divisés par \(\sqrt2\), pour un frame serré de borne 1 sur le domaine étendu, \(W^*W=I\). La synthèse utilisée est l'adjoint exact de cette analyse. Cette identité concerne **W** ; le crop \(P\) n'est pas à remplacer silencieusement par l'adjoint de l'extension \(E^*\).

Pour chaque bande, chaque échantillon et chaque canal, avec \(N\) coefficients spatiaux :

\[
\nu_{j,o}=\sqrt{\frac1N\sum_p d_{j,o,p}^2+10^{-12}},\qquad
\lambda_{j,o}=4(1-s)2^{1-j}\nu_{j,o},
\]

\[
\operatorname{soft}(v,\lambda)=\operatorname{sign}(v)\max(|v|-\lambda,0),
\]

\[
T_s(h)=PW^*\bigl(a_2,\{\operatorname{soft}(d_{j,o},\lambda_{j,o})\}\bigr).
\]

Le facteur \(2^{1-j}\) double le multiplicateur de seuil relatif à la RMS au niveau fin par rapport au niveau grossier ; les seuils absolus dépendent aussi des RMS de bandes. L'approximation \(a_2\) reste inchangée. Les RMS sont calculées depuis les coefficients **avant** seuillage et ne sont **pas détachées** du graphe dans le pilote BN récent.

### 8.2 Endpoints et limites

À \(s=1\), les seuils sont nuls et \(PW^*WEh=PEh=h\) en arithmétique exacte. Les aperçus originaux testaient la reconstruction sans bypass ; le pilote d'entraînement récent utilise un bypass bitwise à cet endpoint. Ce sont deux chemins numériques à distinguer.

À \(s=0\), les seuils sont finis. L'image n'est pas nécessairement constante et des détails peuvent survivre. L'ancienne affirmation « environ 6 % d'écart au coarse-only » n'a pas de mesure ni de normalisation retrouvées : **elle est retirée**.

Il n'y a ni contrainte TV exacte, ni clipping d'activations, ni conservation explicite des moyennes après crop. L'opérateur redondant à seuils dépendant de l'entrée n'est pas identifié automatiquement au proximal de \(\|Wh\|_1\). Le seuillage d'amplitudes ne garantit pas une bande fréquentielle strictement limitée.

L'epsilon assure une différentiation finie au zéro pour la RMS régularisée ; son effet relatif n'est pas uniformément de \(5\times10^{-11}\). Il dépend de l'énergie de la bande et devient important près de zéro. La RMS exacte est une norme, non différentiable à l'origine ; il est impropre de lui attribuer simplement une dérivée mathématique infinie partout au zéro.

### 8.3 Conventions à récupérer si reprise

La convention précise du miroir, les phases/décalages des filtres, le mode de crop et les bornes de tous les paliers du pilote doivent être rattachés à son code. Ne pas substituer une DWT décimée, des seuils fixes, des RMS détachées ou une nouvelle règle de bord sous le nom db2. Le choix utilisateur actuel reste l'abandon de cette piste.

<a id="op-tv"></a>

## 9. TV–L² et TV–Ḣ⁻¹

### 9.1 Spécification utilisateur finale

Cette section normalise uniquement la typographie du bloc édité `64829`. Elle préserve ses contraintes, notamment **moyennes par canal**, **pas de variance conservée**, fidélité **homogène** et **aperçus seulement**.

Pour \(x\in[0,1]^{3\times H\times W}\), avant normalisation réseau, les différences avant de pas 1 sont, avec indices zéro-indexés,

\[
(D_1z)_{c,p,q}=\begin{cases}z_{c,p+1,q}-z_{c,p,q},&p<H-1,\\0,&p=H-1,\end{cases}
\]

et de même \(D_2\) horizontalement avec différence nulle au bord droit. Pas de périodicité.

\[
\operatorname{TV}(z)=\sum_{c,p,q}
\sqrt{(D_1z)_{c,p,q}^2+(D_2z)_{c,p,q}^2}.
\]

La norme euclidienne porte sur les deux directions spatiales de chaque canal ; on somme ensuite les canaux. Ce n'est pas une TV vectorielle qui regrouperait RGB sous la même racine.

\[
\mathcal K_t(x)=\{z\in[0,1]^{3\times H\times W}:\operatorname{TV}(z)\le t\operatorname{TV}(x),
\ \bar z_c=\bar x_c\ \forall c\}.
\]

Le budget TV est **global entre canaux**, les moyennes sont contraintes **séparément**. Il n'y a pas trois budgets indépendants, ni conservation de variance.

### 9.2 Fidélités

\[
T_t^{L^2}(x)=\arg\min_{z\in\mathcal K_t(x)}\frac12\sum_c\|z_c-x_c\|_2^2.
\]

Pour un canal vectorisé, \(\mathsf L=D_1^\top D_1+D_2^\top D_2\), avec les mêmes bords. Son noyau est constitué des constantes. La seconde méthode est

\[
T_t^{\dot H^{-1}}(x)=\arg\min_{z\in\mathcal K_t(x)}
\frac12\sum_c (z_c-x_c)^\top\mathsf L^\dagger(z_c-x_c).
\]

La pseudo-inverse est nulle sur le mode constant. Chaque résidu \(r_c=z_c-x_c\) est de moyenne nulle. On peut donc résoudre \(\mathsf Lp_c=r_c\), \(\bar p_c=0\), et calculer \(\tfrac12\langle r_c,p_c\rangle\).

Il n'y a **aucun paramètre alpha** et aucun opérateur \((I+\alpha\mathsf L)^{-1}\). Sur le sous-espace de moyenne nulle, \(\mathsf L^\dagger\) est définie positive ; la fidélité y est strictement convexe. Avec l'ensemble admissible convexe, cela donne l'unicité de chaque minimiseur. Ces propriétés ne certifient pas la convergence de l'algorithme numérique utilisé.

### 9.3 Endpoints et diagnostics

\[
T_1(x)=x,\qquad T_0(x)=\bar x
\]

(image constante spatialement dans chaque canal). Si TV(x)=0, renvoyer x pour tout t. Un coefficient fixe devant TV dans une pénalité n'est pas équivalent à un budget relatif commun à toutes les images.

Les aperçus utilisent les mêmes dix images, une par classe, pour \(t\in\{1;\,0{,}9;\,0{,}75;\,0{,}5;\,0{,}25;\,0\}\), affichage fixe [0,1]. Mesurer

\[
\operatorname{TV}(z)/\operatorname{TV}(x),\qquad
\rho_{TV}=\|z-\bar x\|_2/\|x-\bar x\|_2.
\]

Les dénominateurs nuls doivent être marqués explicitement. Vérifier budget atteint, moyennes, boîte et résidus ; un solve non convergé n'est pas une projection exacte.

Attention : le JSON des **ondelettes** définit son contraste en recentrant la sortie sur **sa propre moyenne** : \(\rho_W=\|z-\bar z\|_2/\|x-\bar x\|_2\). Ces définitions coïncident lorsque les moyennes sont conservées exactement, ce qui n'est pas imposé par db2. L'ancienne passation les assimilait ; conserver ici les conventions de chaque source.

<a id="op-proposals"></a>

## 10. Variantes non exécutées

La [note de diffusion TV à coût fixé](sources/note_tv_cout_fixe.md) propose M étapes explicites de descente d'une TV lissée, pas la projection ci-dessus. La [note de résolution](sources/note_filtrage_resolution_progressive.md) discute également mélange d'images ou de logits pendant une transition ; ces mélanges n'ont pas été les opérateurs du benchmark.

Le préfiltre RGB récent serait de la forme \(R_r(G_{\tau(r)}x)\). Il est distinct du Gaussian interne \(G_{\sigma_\ell}\). Les paramètres, comparateurs et statut de cette proposition figurent dans [OPEN_QUESTIONS](OPEN_QUESTIONS.md).

## 12. Variantes nommées ajoutées par EXP-012

Ces définitions **s'ajoutent** aux précédentes ; aucune variante historique n'est réécrite.
Implémentation dans `continuation/ablation_ops.py`, posé à côté de `continuation/campaign_ops.py`
qui n'est pas modifié.

### 12.1 Placements du flou

Le Gaussian est linéaire et commute exactement avec la convolution et avec BatchNorm ; il ne
commute pas avec le ReLU. **Il n'existe donc que deux emplacements distincts dans un bloc.**

| Nom | Où | Positions | Note |
|---|---|---:|---|
| `conv_out` | sortie de chaque conv 3×3, avant BN et ReLU | 19 | placement historique depuis EXP-004, celui de CBS |
| `post_bn` | sortie de chaque BatchNorm, avant ReLU | 19 | **même opérateur que `conv_out`** ; ne diffère que par les statistiques que BN accumule |
| `post_block` | après le ReLU : activation post-stem + 9 sorties de bloc | 10 | seul placement correct au sens anti-aliasing aux deux points de décimation |

### 12.2 Masques de sites (indexation `conv_out`)

| Nom | Sites | Intention |
|---|---|---|
| `predown` | {6, 12} | les sorties de conv les plus proches d'une décimation |
| `nodown` | les 17 autres | complémentaire |

Limite à conserver : ce découpage suppose une localité que la cascade n'a pas. Lisser au site 3
réduit encore ce qui arrive au site 7. Il teste la localité, pas le mécanisme.

### 12.3 BlurPool

Gaussian **fixe**, jamais annelé, appliqué par `forward_pre_hook` sur les entrées de `blocks[3]`
et `blocks[6]`. Un seul hook par bloc préfiltre **les quatre** décimations, car la convolution
stridée et le shortcut décimant consomment le même tenseur. C'est de l'architecture : il n'est
**pas** désactivé par `bypass_all`, et l'endpoint d'un tel bras est « réseau + BlurPool », pas le
réseau plain.

### 12.4 Réductions internes et mode relatif

`block{k}_max` réduit le tenseur entrant dans `blocks[k+1]`, pour k de 0 à 7. `blocks[8]` est
exclu : sa sortie va directement au pooling moyen global.

Le calendrier de résolution étant écrit en pixels absolus pour une carte 32×32 (16/24/32), il
cesse d'avoir un sens au-delà de `blocks[3]`, où la carte fait déjà 16 : r=16 y serait un no-op
et r=24 un **upsampling**. Le mode `relative` lit le calendrier comme un **rapport** `r/32`
appliqué à la taille locale ; aux positions 32×32 il reproduit exactement l'absolu.

### 12.5 Profils de profondeur sur sigma

Multiplicateur `m_l` par position, appliqué **par-dessus** `q_l`. Avec un profil calculé sur les
tailles naturelles, `m_l · q_l` fait suivre à sigma la taille **courante**. Normalisation sur le
**maximum** et non sur le budget : cela garde σ ≤ G(e) ≤ 1, donc `sigma_max` reste à 1,0 et le
noyau à 9 taps — condition pour que le témoin uniforme reste valide sans changement de support.

```
position       0     1     2     3     4     5     6     7     8     9
carte naturelle 32   32    32    32    16    16    16     8     8     8
A2  ~ √H     1,00  1,00  1,00  1,00  0,71  0,71  0,71  0,50  0,50  0,50
A1  ~ H      1,00  1,00  1,00  1,00  0,50  0,50  0,50  0,25  0,25  0,25
A3  ~ 1/H    0,25  0,25  0,25  0,25  0,50  0,50  0,50  1,00  1,00  1,00
A4  ~ RF     0,09  0,22  0,34  0,47  0,66  0,91  1,00  1,00  1,00  1,00
```

**Tous écartés par EXP-012.** A1 et A2 n'atténuent pas le flou en profondeur, ils l'éteignent
(σ → 0,25 et 0,125, soit ~100 % d'énergie retenue) ; voir [CORRECTIONS](CORRECTIONS.md) C-40.

### 12.6 Avertissement sur `input_max`

`input_max` **n'applique pas de max-pooling** : le chemin d'entrée du driver passe par
`resize_unit_float`, bilinéaire+antialias en dur. Voir [INTEGRATION](INTEGRATION.md) C-42.
La variante max n'est réelle qu'aux emplacements `stem_*` et `block{k}_*`.
