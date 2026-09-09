# Entraînement par continuation : homotopie résiduelle, homotopie d'activation, et ancrage

*ResNet-18 / CIFAR-10. Rapport d'expériences.*

---

## Résumé

Nous avons testé trois formes de continuation pour l'entraînement d'un réseau
profond : une homotopie sur la **branche résiduelle**, une homotopie sur la
**fonction d'activation**, et l'ajout d'une **pénalité d'ancrage** destinée à
rendre le chemin de solutions bien posé. Chaque variante a été comparée à une
baseline à budget de calcul strictement égal, même graine, même recette
d'optimisation.

**Aucune n'améliore la baseline.** L'homotopie d'activation lui coûte 1,19
point d'accuracy à pleine échelle (19 σ sur trois graines), la pénalité
d'ancrage n'a aucun effet mesurable sur deux ordres de grandeur de son
coefficient, et le seul effet favorable observé — une réduction du
sur-apprentissage — s'inverse quand on passe de 10 000 à 50 000 images
d'entraînement.

Le résultat le plus utile de cette campagne n'est pas le verdict lui-même mais
le **mécanisme de contrôle** qui l'a produit : trois hypothèses explicatives
intermédiaires ont été formulées puis réfutées par la mesure, et deux
conclusions préliminaires ont dû être corrigées après passage à l'échelle.

---

## 1. Cadre théorique

### 1.1 Le principe de continuation

Une méthode d'homotopie remplace un problème d'optimisation difficile par une
famille continue de problèmes indexée par un paramètre, dont une extrémité est
facile et l'autre est le problème cible. On résout l'extrémité facile, puis on
déforme progressivement le problème en réoptimisant à chaque pas, dans l'espoir
que la solution suive une branche continue jusqu'à la cible.

Formellement, si $L(\theta, \tau)$ est la perte paramétrée par $\tau$, on
cherche une branche $\theta^*(\tau)$ telle que

$$\nabla_\theta L(\theta^*(\tau), \tau) = 0 \quad \text{pour tout } \tau.$$

Le théorème des fonctions implicites garantit l'existence locale de cette
branche **à condition que $\nabla^2_\theta L$ soit inversible**. Cette condition
n'est pas anodine, et elle est au cœur de la section 1.4.

### 1.2 Axe 1 — l'homotopie résiduelle

Le premier axe déforme la profondeur effective du réseau. Un bloc résiduel
calcule normalement $h_{\text{out}} = \text{shortcut}(x) + F(x)$. On introduit
un paramètre $s$ :

$$h_{\text{out}} = \text{shortcut}(x) + s \cdot F(x), \qquad s : 0 \to 1.$$

À $s = 0$ les branches résiduelles sont éteintes et il reste un réseau
peu profond mais réel — le stem, les trois projections $1{\times}1$ aux
transitions de stage, le pooling et la tête. À $s = 1$ on retrouve le ResNet
standard.

**Une faiblesse structurelle de cet axe.** La branche $F$ se termine par une
BatchNorm, donc $s \cdot \text{BN}(\cdot)$ est *exactement* la même fonction que
BN avec $\gamma \to s\gamma$. Pour tout $s > 0$ la classe de fonctions
représentables est identique à celle de $s = 1$ : cette homotopie change la
paramétrisation, pas la capacité. Le réseau peut donc **absorber $s$** en
faisant croître $\gamma$, et l'homotopie se réduit à une reparamétrisation sans
effet. C'est un mode d'échec silencieux, et le diagnostic `residual_ratio`
existe précisément pour le détecter.

De plus, $\partial L/\partial \theta_F \propto s$ : le paramètre agit aussi
comme un multiplicateur de learning rate local à la branche. Un bras de contrôle
(`lr_gate_control`) applique ce profil au learning rate en laissant la passe
avant intacte, afin de séparer les deux lectures.

**Résultats de cet axe** (60 époques, 1 graine, 50 000 images) :

| bras | test acc |
|---|---|
| `baseline60` | 0.9465 |
| `homotopy_linear` ($s$ : 0 → 1 linéaire) | 0.9404 |

L'homotopie résiduelle perd 0,6 point. Ce résultat, obtenu avant la présente
campagne, motive le passage à un axe où la déformation ne peut pas être
absorbée par une reparamétrisation.

### 1.3 Axe 2 — l'homotopie d'activation

On remplace chaque ReLU par une LeakyReLU dont la pente négative est le
paramètre d'homotopie :

$$\varphi_\alpha(x) = \max(x, \alpha x), \qquad \alpha : 1 \to 0.$$

- $\alpha = 1 \Rightarrow \varphi_\alpha = \text{id}$ : le réseau devient
  **affine**.
- $\alpha = 0 \Rightarrow \varphi_\alpha = \text{ReLU}$ : le réseau cible.

**Pourquoi cet axe est structurellement meilleur que le premier.** En
décomposant

$$\varphi_\alpha(x) = \underbrace{\tfrac{1+\alpha}{2}}_{a}\,x
+ \underbrace{\tfrac{1-\alpha}{2}}_{b}\,|x|,$$

on voit que $\varphi_\alpha$ est **positivement homogène** :
$\varphi_\alpha(cx) = c\,\varphi_\alpha(x)$ pour tout $c > 0$. Aucun
redimensionnement de poids, ni d'aucun $\gamma$ de BatchNorm, ne peut modifier
le mélange $a : b$. **$\alpha$ ne peut pas être reparamétré.** Cette homotopie
déforme le réseau, pas sa paramétrisation.

**L'échappatoire qui subsiste est le décalage, pas l'échelle.** BatchNorm peut
pousser $\beta$ vers le positif jusqu'à ce que presque toutes les
pré-activations soient positives, auquel cas $\varphi_\alpha$ est l'identité
quel que soit $\alpha$. Le diagnostic correspondant est

$$\text{linear\_gap} = \frac{\text{rms}(\varphi_\alpha(h) - h)}{\text{rms}(h)}
= (1-\alpha)\,\frac{\text{rms}(\text{relu}(-h))}{\text{rms}(h)},$$

qui doit croître à mesure que $\alpha$ décroît. Une valeur plate signale une
homotopie contournée, et invalide le run avant toute lecture d'accuracy.

**Une dégénérescence propre à $\alpha = 1$.** Là où $s = 0$ laissait un ConvNet
peu profond entraînable, $\alpha = 1$ laisse un **classifieur linéaire**, dont
l'optimum sur CIFAR-10 plafonne vers 40 %. Surtout, un réseau linéaire profond
est invariant sous $W_1 \to W_1 G,\ W_2 \to G^{-1} W_2$ : sa hessienne possède
un noyau de grande dimension **par construction**. $\theta^*(1)$ n'est pas un
point isolé mais une variété, et le théorème des fonctions implicites n'a rien
à dire en ce point. Une continuation démarrée exactement à $\alpha = 1$ démarre
sur une singularité de la branche qu'elle prétend suivre.

### 1.4 La pénalité d'ancrage (GRDH)

C'est précisément ce que corrige l'équation 12 de GRDH, transposée ici :

$$L(\theta, \alpha) = L_{\text{CE}}(\theta)
+ \lambda\,\frac{\alpha}{2}\,\lVert \theta - \theta_0 \rVert^2,$$

où $\theta_0$ sont les poids à l'initialisation, figés et détachés du graphe.
Le terme quadratique convexe rend la hessienne régularisée inversible, ce qui
sélectionne un point unique dans la variété $\theta^*(1)$ et rend le départ du
chemin bien posé.

Deux propriétés du coefficient $\alpha$ importent :

1. il **s'annule à $\alpha = 0$**, donc la queue de l'entraînement minimise
   exactement l'objectif de la baseline — sans quoi aucune comparaison
   d'accuracy ne serait légitime ;
2. il est **maximal à $\alpha = 1$**, là où se trouve la dégénérescence.

**Portée de l'ancre.** Elle s'applique aux poids des `Conv2d` et `Linear`,
jamais aux BatchNorm. Deux raisons, dont la première est rédhibitoire :
`zero_init_residual` initialise chaque $\gamma_{bn2}$ à zéro, et ancrer cette
valeur épinglerait chaque branche résiduelle à $F \equiv 0$ pendant toute la
descente — le réseau ne pourrait jamais quitter le modèle linéaire. La seconde
est que les symétries d'échelle que l'ancre existe pour briser vivent dans les
matrices de poids.

---

## 2. Implémentation

L'ensemble est construit sur une base ResNet-18 / CIFAR-10 existante
(stem $3{\times}3$ stride 1, sans maxpool, quatre stages, 11,17 M paramètres,
94–95 % de test accuracy en 100 époques).

| module | rôle |
|---|---|
| `model.py` | `Activation` (le $\varphi_\alpha$), flag `use_residual` |
| `activation.py` | `ActivationGate` : les 17 sites, les portées de groupement |
| `anchor.py` | $\theta_0$, la pénalité, la calibration de $\lambda$ |
| `homotopy.py` | `ResidualGate`, les schedules, les phases de continuation |
| `landscape.py` | diagnostics, courbure, distance fonctionnelle, surfaces |
| `compare.py`, `explore.py` | figures |

**174 tests** verrouillent les invariants. Les plus importants :

- **$\alpha = 0$ est bit-identique à la baseline**, sortie *et* gradients. Le
  dispatch branche explicitement vers `F.relu` plutôt que
  `F.leaky_relu(x, 0.0)`, qui renvoie $-0.0$ sur les entrées négatives et
  emprunte un autre noyau.
- **`Activation` n'ajoute ni paramètre ni buffer** : le `state_dict` est
  inchangé, les checkpoints antérieurs se rechargent.
- **$\alpha = 1$ rend le réseau affine**, vérifié par
  $f(x+y) - f(x) - f(y) + f(0) \approx 0$, avec le cas ReLU comme contrôle qui
  doit échouer au même test.
- **$\varphi_\alpha$ est positivement homogène**, la propriété qui ferme
  l'échappatoire d'échelle.
- **La pénalité vaut exactement 0 en $\theta_0$ et à $\alpha = 0$**, quel que
  soit $\lambda$.

### 2.1 Un piège de mesure : BatchNorm

Passer de $\alpha > 0$ à $\alpha = 0$ supprime toute la masse négative de chaque
activation. Les statistiques `running_mean` / `running_var` accumulées à
$\alpha > 0$ décrivent alors une distribution que le réseau ne produit plus.
Évaluer à travers elles produit exactement la signature d'une homotopie qui ne
transfère pas — c'est-à-dire **un bug lu comme un résultat**.

La lecture comparable `test_acc_at_a0` force donc $\alpha = 0$ **et
ré-estime les statistiques BatchNorm** (64 batches, sans augmentation) avant
d'évaluer, puis restaure celles du run. Sans ce recalcul, les écarts mesurés
pendant la rampe sont des artefacts.

---

## 3. Expériences

### 3.1 exp1 — homotopie d'activation continue

*50 000 images, 40 époques, 1 graine. $\alpha : 1 \to 0$ linéairement sur la
première moitié, 20 époques de queue à $\alpha = 0$.*

**Objectif.** Établir un premier signal sur l'axe activation, et vérifier que
la mécanique fait ce qu'elle prétend.

**Observations.**

| bras | test acc | train acc | gap |
|---|---|---|---|
| `baseline40` | 0.9397 | 0.9988 | +0.0591 |
| `act_linear` | 0.9320 | 0.9971 | +0.0651 |

L'homotopie perd 0,77 point. Le diagnostic valide la mécanique :
`linear_gap` monte de 0,035 à 0,748 en miroir exact de la descente de $\alpha$,
et `neg_frac` reste entre 0,50 et 0,58 — **pas d'échappatoire par décalage**.

La dynamique est instructive. À l'époque 9 ($\alpha = 0{,}50$) le bras
homotopie est **19 points derrière** en lecture ReLU (0,614 contre 0,806), puis
rattrape presque tout. Le déficit final de 0,77 point est le reliquat d'un
retard massif comblé en 20 époques. Première hypothèse : la rampe consomme la
moitié du budget d'entraînement sur un problème qui n'est pas la cible.

*Figures : `compare_updates.png`, `alpha.png`, `loss_vs_alpha.png`,
`branch.png`, `curvature.png`.*

### 3.2 Digression — l'observation de régularité, et sa réfutation

**L'observation.** Après la fin de la rampe (époques 20-39), là où les deux bras
entraînent exactement la même architecture avec le même learning rate et à un
niveau de loss quasi identique (0,312 contre 0,316), les courbes de test du bras
homotopie sont nettement **plus lisses**. Rugosité mesurée comme l'écart-type
des différences d'une époque à l'autre :

| | baseline | homotopie |
|---|---|---|
| test loss, ép. 20-39 | 0,0502 (16,1 % du niveau) | **0,0215 (6,8 %)** |
| test acc, ép. 20-39 | 0,0136 | **0,0072** |
| train loss, ép. 20-39 | 0,0075 | 0,0084 |

La train loss est aussi rugueuse des deux côtés : **l'optimisation se passe
pareil, mais les métriques de test bougent deux fois moins**. Cela suggère une
propriété de stabilité fonctionnelle, pas d'optimisation.

**Hypothèse 1 : un minimum plus plat.** Les mêmes déplacements de $\theta$
déformeraient moins la fonction. Testée par power iteration et estimateur de
Hutchinson sur les checkpoints finaux :

| | top eigenvalue | trace |
|---|---|---|
| init (identique) | 0,55 | 3,4 |
| `baseline40` @ ép. 39 | 35,2 | 166 |
| `act_linear` @ ép. 39 | **48,1** | **433** |

**Réfutée, et par l'inverse.** Le bras homotopie finit dans un minimum
*plus pointu* — 1,4× sur la valeur propre dominante, 2,6× sur la trace. Ce qui
est en revanche cohérent avec sa moins bonne généralisation.

**Hypothèse 2 : des pas plus petits.** Un déplacement moindre par époque
donnerait des métriques plus stables même dans un bassin pointu. Mesuré comme
$\lVert \Delta\theta \rVert / \lVert \theta \rVert$ entre checkpoints
consécutifs, moyenne après l'époque 20 : **0,255 pour l'homotopie contre 0,233
pour la baseline**.

**Réfutée aussi, et par l'inverse.** Des pas plus grands dans un bassin plus
pointu devraient donner des métriques *plus* instables. On observe le contraire.

**Conclusion de la digression.** Deux mécanismes explicatifs formulés, deux
réfutés par la mesure. L'explication parcimonieuse redevient « une trajectoire
SGD parmi d'autres », à une seule graine. La régularité ne méritait pas
d'explication avant d'avoir survécu à des graines supplémentaires — et
elle a effectivement disparu de l'analyse ensuite.

*Cette digression illustre le mode de travail adopté : toute observation
inattendue a donné lieu à une hypothèse mécaniste explicite et à un test qui
pouvait la réfuter.*

### 3.3 exp2 — l'ancrage, et le balayage de $\lambda$

*10 000 images, 50 époques, 1 graine. Escalier de 10 paliers de 4 époques
($\alpha$ : 1,000 → 0,000 par pas de 0,111), puis 14 époques à $\alpha = 0$.*

**Transition.** exp1 laisse deux questions : l'ancre change-t-elle quelque
chose, et le déficit vient-il du budget consommé par la rampe ? Le passage à
10 000 images divise le coût d'un run par cinq (10 min contre 50) et permet
enfin de **balayer $\lambda$ au lieu de le deviner**.

**Un échec méthodologique instructif : la calibration de $\lambda$.** Le critère
initial — « la pénalité doit valoir 10 % de la cross-entropy » — a échoué deux
fois, dans des directions opposées, parce que $\lVert \theta - \theta_0
\rVert^2$ croît d'un facteur mille pendant un run (5,4 à l'étape 50, 5 449 à
l'époque 10) :

| point de mesure | $\lambda$ obtenu | conséquence |
|---|---|---|
| étape 50 | 2,944 | pénalité = **3 656× la CE** à l'époque 5 : réseau gelé |
| mi-rampe | $10^{-4}$ | ancre = **0,5 % du gradient** : inerte |

Le critère lui-même était mauvais. Pour désingulariser une hessienne, ce qui
compte n'est pas la part de la **perte** mais celle du **gradient** — et 10 %
de l'une ne fait que 0,5 % de l'autre. D'où le balayage.

**Observations.**

| bras | $\lambda$ | test acc | test loss |
|---|---|---|---|
| `fast_baseline` | — | **0.8353** | 0.7155 |
| `fast_steps_l0` | 0 | 0.8335 | 0.5874 |
| `fast_steps_l4` | $10^{-4}$ | 0.8311 | 0.5853 |
| `fast_steps_l3` | $10^{-3}$ | 0.8297 | — |
| `fast_steps_l2` | $10^{-2}$ | 0.8303 | 0.5652 |

**L'ancre n'a aucun effet mesurable.** $\lambda$ varie sur deux ordres de
grandeur, les accuracies tiennent dans 0,38 point, et l'ordre **n'est pas
monotone** ($10^{-3}$ est le pire, $10^{-2}$ entre les deux). C'est la signature
du bruit.

Le seul effet monotone est sur le sur-apprentissage (train acc 96,97 % →
96,04 % quand $\lambda$ monte) — mais le bras **sans ancre** capte déjà
l'essentiel du bénéfice.

**Le signal apparent.** Les quatre bras escalier réduisent la test loss de
~18 % à accuracy quasi égale. Cet effet est entièrement présent à
$\lambda = 0$ : il vient de l'escalier, pas de la pénalité.

### 3.4 exp3 — trois graines sur le même protocole

*10 000 images, 50 époques, graines 0/1/2. Baseline contre escalier sans ancre.*

**Transition.** exp2 a enterré l'ancre. Reste à savoir si le gain de test loss
survit au bruit de graine — c'était le seul écart assez large pour en avoir une
chance.

**Observations.**

| | baseline | escalier | écart | |
|---|---|---|---|---|
| test acc | 0.8374 ± 0.0014 | 0.8278 ± 0.0024 | **−0,96 pt** | 3,4 σ |
| test loss | 0.7080 ± 0.0115 | 0.5886 ± 0.0074 | **−17 %** | 8,7 σ |
| gen gap | 0.1618 ± 0.0014 | 0.1376 ± 0.0033 | −0,024 | |
| train acc | 0.9992 | 0.9655 | | |

Aucun recouvrement entre graines sur l'une ou l'autre métrique.

**Une correction nécessaire.** À une graine, l'écart d'accuracy valait 0,18
point et avait été qualifié de bruit. Il vaut 0,96 point sur trois graines, à
3,4 σ. La graine 0, tirée en premier, était la plus favorable à l'homotopie.
C'est la deuxième conclusion préliminaire à devoir être corrigée.

**Lecture.** Les deux effets sont significatifs et **opposés** : l'escalier
échange 1 point d'accuracy contre 17 % de loss. C'est le comportement d'une
régularisation — un réseau mieux calibré mais moins juste — pas d'une meilleure
optimisation. Le critère de décision, fixé avant tout résultat, portait sur
l'accuracy : il n'est pas rempli.

**Une explication banale restait à écarter.** À 10 000 images la baseline
mémorise complètement (gap +16,2 %). Dans ce régime, **tout** ce qui freine
l'apprentissage améliore la loss de test. L'escalier passe 36 époques sur 50
loin du problème cible : il *est*, mécaniquement, un frein.

### 3.5 exp4 — passage à l'échelle, et renversement

*50 000 images (jeu complet), 80 époques, graines 0/1/2. Escalier de 10 paliers
de 6-7 époques, 22 époques finales à $\alpha = 0$ (28 % du budget, la même
proportion qu'exp3).*

**Transition.** C'est le run qui sépare les deux lectures. À 50 000 images le
gap de la baseline tombe à +5,2 %, il n'y a presque plus de sur-apprentissage
à corriger, et l'explication « moins entraîner régularise » disparaît largement.

**Observations.**

| | baseline | escalier | écart | |
|---|---|---|---|---|
| test acc | 0.9483 ± 0.0004 | 0.9364 ± 0.0005 | **−1,19 pt** | **19,2 σ** |
| test loss | 0.2053 ± 0.0009 | 0.2169 ± 0.0009 | **+0,0116** | 9,2 σ |
| gen gap | 0.0516 ± 0.0004 | 0.0592 ± 0.0005 | **+0,0075** | 12,2 σ |
| train acc | 0.9999 ± 0.0000 | 0.9956 ± 0.0002 | −0,0044 | 27,4 σ |

Par graine, sans recouvrement :

```
baseline    acc 0.9485  0.9475  0.9489    loss 0.2070  0.2049  0.2041
escalier    acc 0.9358  0.9361  0.9373    loss 0.2153  0.2171  0.2185
```

Updates de gradient pour atteindre un niveau, escalier relatif à la baseline :
**85 % → 2,28×, 90 % → 1,41×, 93 % → 1,22×, 94 % → jamais.**

**Le renversement.**

| | 10k images (exp3) | 50k images (exp4) |
|---|---|---|
| accuracy | −0,96 pt | **−1,19 pt** (pire) |
| test loss | **−17 % (favorable)** | **+5,7 % (défavorable)** |
| gen gap | −0,024 (favorable) | **+0,0075 (défavorable)** |

Le seul argument qui restait en faveur de l'homotopie **change de signe à
pleine échelle**. C'était bien du freinage : à 50 000 images il n'y a plus de
mémorisation à corriger, et l'escalier — qui sur-apprend désormais *davantage* —
n'a plus rien à offrir.

**Un confounder, mesuré.** `a_lr_restart` donne à chaque palier son propre
warmup et son cosine, si bien que sur les 22 dernières époques l'escalier
s'entraîne sous un cosine complet depuis $lr = 0{,}1$ pendant que la baseline
est dans la queue du sien à $lr < 0{,}02$. Ce n'est pas neutre. Mais la mesure
montre que cela n'explique pas l'écart : à l'époque 57, **avant** le
redémarrage, l'escalier était 1,78 point derrière ; à l'époque 79 il l'était de
1,27. La phase finale lui a fait *regagner* un demi-point malgré le plongeon
visible de 0,892 à 0,753. Le supprimer aurait vraisemblablement empiré son
score.

*Figures : `exp4_full80_3seeds/compare_updates.png` (moyenne, bande ±1 σ, les
trois runs), `alpha.png`, `loss_vs_alpha.png`.*

### 3.6 exp6 — sans connexions résiduelles

*PlainNet-18, 5 000 images, 50 époques, graine 1, quatre bras.*

**Transition — et l'hypothèse la plus intéressante de la campagne.** Li et al.
(2018) montrent que les connexions résiduelles **convexifient** le paysage de
perte d'un réseau profond. Si c'est vrai, l'homotopie d'activation n'a rien à
corriger sur un ResNet : le paysage est déjà lisse, et un départ quasi-linéaire
ne peut que coûter du budget. Ce qui est exactement ce qu'exp1 à exp4 observent.

Le test décisif est donc de **retirer les skips**. Sans eux le paysage redevient
accidenté, et c'est le régime pour lequel la continuation a été conçue — un
régime où la baseline elle-même est en difficulté.

L'ablation ne porte que sur `use_residual` : mêmes canaux, mêmes strides, mêmes
BatchNorms, mêmes activations, même recette. 11,00 M paramètres au lieu de
11,17 M — les seuls qui disparaissent sont les trois projections $1{\times}1$
et leurs BatchNorms, qui n'existent que pour rendre $x$ additionnable.

*Piège écarté : `zero_init_residual` met $\gamma_{bn2}$ à zéro, et sans $x$ à
réajouter chaque bloc sort exactement zéro — le réseau devient une constante.
La configuration refuse désormais la combinaison.*

**Observations.**

| bras | $\alpha$ | $\lambda$ | test acc | vs baseline | test loss | train acc | gap |
|---|---|---|---|---|---|---|---|
| `plain_baseline50` | 0 | 0 | **0.7777** | — | 0.9875 | 0.9992 | +0.2215 |
| `plain_act_linear50` | 1→0 / 25 ép. | 0 | 0.7218 | **−5,59 pt** | 1.0801 | 0.9830 | +0.2612 |
| `plain_act_anchor50` | 1→0 / 25 ép. | $10^{-4}$ | 0.7211 | **−5,66 pt** | 1.0824 | 0.9820 | +0.2609 |
| `plain_act_staircase50` | 5 × 10 ép. | $10^{-4}$ | 0.6959 | **−8,18 pt** | 0.9342 | 0.8616 | +0.1657 |

**L'hypothèse n'est pas soutenue.** Retirer les skips devait être le régime
favorable. La pénalité **s'aggrave** au contraire : −1,2 point sur ResNet à
50k, −5,6 à −8,2 points sur PlainNet. Un paysage plus difficile coûte davantage
à l'homotopie, pas moins.

**L'ancre est inerte pour la troisième fois indépendamment.** `linear` et
`anchor` diffèrent de 0,07 point, avec loss et train accuracy identiques.

**Le bras escalier reproduit la signature trompeuse.** Dernier en accuracy,
*premier* en test loss (0,9342 contre 0,9875), et très bas en train accuracy
(0,8616) : il n'a pas fini d'apprendre — 10 époques à $\alpha = 0$ sur 50. C'est
exactement le motif qu'exp3 avait produit et qu'exp4 a réfuté ; il ne doit pas
être lu comme un gain.

---

## 4. Conclusions par méthode

**Homotopie résiduelle ($s : 0 \to 1$).** Perd 0,6 point. Faiblesse
structurelle : $s$ est absorbable dans $\gamma_{bn2}$, l'homotopie ne change
que la paramétrisation, et agit partiellement comme un schedule de learning
rate déguisé.

**Homotopie d'activation ($\alpha : 1 \to 0$).** Structurellement plus solide —
$\alpha$ n'est pas reparamétrable — et la mécanique est vérifiée
(`linear_gap` suit $1 - \alpha$, `neg_frac` stable, pas de contournement).
Mais elle **dégrade** systématiquement :

| protocole | écart d'accuracy |
|---|---|
| ResNet-18, 50k, 40 ép., 1 graine | −0,77 pt |
| ResNet-18, 10k, 50 ép., 3 graines | −0,96 pt (3,4 σ) |
| ResNet-18, 50k, 80 ép., 3 graines | **−1,19 pt (19,2 σ)** |
| PlainNet-18, 5k, 50 ép., 1 graine | −5,6 à −8,2 pt |

Le coût **croît** avec l'échelle des données et avec la difficulté du paysage.

**Continue contre discrète.** Sur ResNet à 10k, les deux formes sont
indiscernables (0,8335 contre 0,8278 selon la répartition des paliers, sous le
bruit). Sur PlainNet, la discrète est nettement pire (−8,2 contre −5,6) parce
qu'elle réserve moins d'époques au problème cible. **La discrétisation en
elle-même n'apporte rien** ; ce qui compte est la fraction du budget passée à
$\alpha = 0$.

**Pénalité d'ancrage.** Aucun effet mesurable, confirmé trois fois de manière
indépendante : balayage de $\lambda$ sur deux ordres de grandeur avec des
écarts non monotones sous 0,4 point ; bras ResNet à 3 graines ; bras PlainNet.
Le raisonnement théorique reste valide — la hessienne *est* singulière à
$\alpha = 1$ — mais aucun $\lambda$ dans $[10^{-4}, 10^{-2}]$ ne produit
d'effet, et il existe une tension : tout $\lambda$ assez grand pour relever
significativement les directions plates est assez grand pour écraser la
cross-entropy une fois $\theta$ éloigné de $\theta_0$.

**Le coût en vitesse est le verdict le plus net.** Si la question était « la
continuation facilite-t-elle l'optimisation ? », la réponse est non :
2,3× à 2,5× plus d'updates pour atteindre les niveaux intermédiaires, et
l'accuracy finale de la baseline n'est jamais atteinte.

---

## 5. Menaces à la validité

**Résolues.**

- *L'homotopie était-elle réellement appliquée ?* `linear_gap` monte de 0,035 à
  0,748 en miroir de $\alpha$, `neg_frac` reste stable : non contournée.
- *Les lectures pendant la rampe sont-elles comparables ?* Oui, via
  `test_acc_at_a0` / `test_loss_at_a0`, qui forcent $\alpha = 0$ **et**
  ré-estiment BatchNorm.
- *Le redémarrage du learning rate explique-t-il l'écart ?* Non : mesuré, la
  phase finale fait *regagner* du terrain à l'escalier.
- *Un seul seed suffit-il ?* Non — deux conclusions ont dû être corrigées après
  passage à trois graines. Tous les résultats principaux sont à trois graines.

**Non résolues.**

- **La profondeur du PlainNet.** He et al. (2015) trouvent plain et residual
  quasi équivalents à 18 couches ; la dégradation apparaît à 34. La baseline
  PlainNet-18 atteint 77,8 % et s'entraîne sans peine, ce qui suggère que le
  paysage n'est pas devenu le régime chaotique que l'hypothèse décrit. **exp6
  ne teste peut-être pas ce qu'il prétend tester.**
- **exp6 est à 5 000 images et une graine**, donc dans le régime de mémorisation
  dont exp4 a déjà montré qu'il inverse les conclusions.
- **Le schedule de learning rate diffère** entre bras sur la queue de
  l'entraînement (confounder mesuré mais non éliminé).

---

## 6. Ouvertures

**Immédiat — valider la prémisse d'exp6.** Deux runs de quelques minutes :

1. `resnet18` au format exact d'exp6 (5k, 50 époques, graine 1). Si plain et
   residual se tiennent à deux points, la prémisse échoue à cette profondeur et
   exp6 ne dit rien sur la géométrie du paysage.
2. `--arch resnet34` : la profondeur où l'écart plain/residual est documenté, et
   où la baseline serait réellement en difficulté.

**Court terme.**

3. **PlainNet-34 sur 50 000 images, trois graines.** C'est la seule
   configuration où l'hypothèse centrale peut être testée honnêtement.
4. **Un bras de contrôle par régularisation.** Baseline + weight decay renforcé
   à budget égal. Si cela reproduit le gain de loss observé à 10k, l'homotopie
   n'apporte rien qu'un hyperparamètre ne donne gratuitement.
5. **Éliminer le confounder de schedule** par un SGDR à amplitude décroissante :
   chaque phase redémarre au niveau de l'enveloppe cosine globale plutôt qu'à
   $lr = 0{,}1$.

**Plus long terme.**

6. **Le $\lambda$ de l'ancre choisi sur un critère de gradient**, non de perte.
   Le critère correct est la part du gradient, ou directement le plancher de
   courbure ajouté ($\lambda\alpha$ s'ajoute à chaque valeur propre). Il reste
   à vérifier s'il existe une fenêtre où l'ancre relève les directions plates
   sans écraser la cross-entropy — la campagne suggère qu'elle est étroite,
   voire vide.
7. **La branche $\theta^*(\alpha)$ n'a pas été étudiée pour elle-même.**
   L'outillage existe (`functional_distance`, `interpolate` avec recalcul BN,
   `pca_plane`) et une observation isolée mérite d'être poursuivie : le plus
   grand changement fonctionnel entre checkpoints consécutifs survient **après**
   la fin de la rampe, pas pendant. $\theta$ ne suit donc pas la branche, il est
   en retard sur elle et rattrape ensuite. Mesurer où la branche cesse d'être
   continue serait un résultat sur la géométrie du paysage, indépendant de
   toute amélioration d'accuracy.
8. **Predictor-corrector.** Non implémenté délibérément. Le prédicteur sécant
   $\theta^*(\alpha_k) + \frac{\theta^*(\alpha_k) - \theta^*(\alpha_{k-1})}{\alpha_k - \alpha_{k-1}}\Delta\alpha$
   est gratuit et ne se justifie que si les mesures de continuité de la branche
   montrent un chemin lisse.

---

## Annexe — reproduction

```
src/cifarbase/configs/     une YAML par bras, avec en tête la question qu'il pose
results/exp*/README.md     protocole, chiffres, réserves
results/exp*/stats.json    config résolue, chiffres par graine, agrégats
out/exp*/                  runs bruts (non versionnés)
```

Recette commune à tous les bras, jamais modifiée : SGD, $lr = 0{,}1$, nesterov,
momentum 0,9, weight decay $5 \times 10^{-4}$, cosine avec 5 époques de warmup
linéaire, crop + flip aléatoires, batch 128.

Les runs Kaggle utilisent un kernel par protocole (`make push-act ACT=<bras>
SEEDS=<liste>`), le suffixe étant dérivé des graines, de sorte qu'un changement
de protocole crée un nouveau kernel au lieu d'écraser le précédent.
