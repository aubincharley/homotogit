# Le gain de continuation résiste-t-il au changement d'optimiseur ?

Rapport de la branche `continuation-core-optimizer-benchmark`.
Campagne du 14/09/2026 — 53 entraînements, Kaggle 2×T4, ~7,5 h GPU.

---

## 1. La question

La branche `continuation-core` gèle quatre configurations et annonce que la
**continuation** — dégrader le réseau au début de l'entraînement (flou,
résolution réduite) puis restaurer progressivement la tâche exacte — fait gagner
**+4 à +6 points** sur un témoin sans intervention.

Ce chiffre a été mesuré sous **un seul optimiseur** : SGD, momentum 0,9,
lr 0,005. C'est un learning rate délibérément bas, sur un budget court de
30 époques, dans un régime où le témoin sous-apprend nettement — il plafonne à
75 % alors qu'un ResNet-20 correctement entraîné dépasse 91 %.

D'où une hypothèse concurrente que rien dans le dépôt ne permettait d'écarter :

> Et si le gain de continuation n'était pas une propriété du chemin de
> continuation, mais simplement de la **vitesse d'optimisation** ? Un optimiseur
> adaptatif, qui ajuste son pas paramètre par paramètre, ferait alors le même
> travail — et le gain disparaîtrait.

Autrement dit : la continuation aide-t-elle vraiment, ou compense-t-elle
seulement un optimiseur mal réglé ? C'est ce que cette campagne tranche.

---

## 2. Ce qui est comparé

**Quatre méthodes** (les quatre configurations gelées, inchangées) :

| méthode | ce qu'elle fait |
|---|---|
| `plain` | témoin, aucune intervention |
| `resolution_max_b1` | réduit la carte d'activations après le bloc 1 : 16 → 24 → 32 px |
| `gaussian_postrelu` | floute 10 tenseurs post-ReLU, σ décroissant de 1,0 à 0 |
| `resolution_max_b1_gaussian_conv` | les deux : réduction + flou aux 19 sorties de convolution |

Chaque méthode annule son intervention avant la fin — flou coupé à l'époque 21,
résolution restaurée à l'époque 12 — donc **tous les bras finissent par au moins
neuf époques sur la tâche cible exacte** et sont évalués sur le même réseau que
le témoin. C'est ce qui distingue la continuation d'un simple changement
d'architecture.

**Quatre optimiseurs** :

| optimiseur | particularité |
|---|---|
| **SGD** | la référence, momentum 0,9 |
| **Adam** | adaptatif, weight decay en L2 couplée |
| **AdamW** | identique à Adam **au seul flag de découplage du weight decay près** |
| **RAdam** | corrige la variance des premières updates — la phase exacte où la continuation intervient |

Adam et AdamW ne diffèrent que par un flag : c'est un contraste à un seul
facteur, et il mord ici parce que la recette applique le weight decay à *tous*
les paramètres, BatchNorm et biais compris.

**Trois graines** (0, 1, 2), avec les mêmes poids initiaux, le même ordre de
données et la même sonde que tous les résultats déjà enregistrés — fichiers
épinglés et vérifiés par empreinte sha256 avant chaque entraînement.

4 méthodes × 4 optimiseurs × 3 graines = **48 bras**.

---

## 3. Comment

### La recette, tenue fixe

CIFAR-10 complet (50 000 / 10 000), **aucune augmentation**, 30 époques
= 11 730 updates, lot effectif 128 en micro-lots de 32, weight decay 5·10⁻⁴,
60 updates de chauffe puis cosinus.

**Seul l'optimiseur et son learning rate changent.** La forme du schedule, la
chauffe et le weight decay sont identiques partout — sinon un contraste
d'optimiseur serait aussi un contraste de schedule.

### Régler le learning rate, sinon la comparaison ne veut rien dire

lr 0,005 est une valeur SGD. Adam à 0,005 sous-performe massivement. Comparer
les optimiseurs à lr fixe ne mesurerait que « lr mal réglé ».

Chaque optimiseur a donc été balayé **avant** la grille, sur le témoin `plain`
seul, graine 0 :

| lr | Adam | AdamW | RAdam |
|---|---:|---:|---:|
| 3·10⁻⁴ | 75,37 | 73,89 | 74,09 |
| 1·10⁻³ | 81,29 | 79,79 | 80,07 |
| 3·10⁻³ | **83,95** | 82,98 | **84,58** |
| 1·10⁻² | 81,97 | 83,80 | 84,42 |
| 2·10⁻² | — | **84,52** | — |
| 3·10⁻² | — | 83,83 | — |

Règle de sélection fixée **avant** de regarder les chiffres : meilleure accuracy
sur `plain`, égalité à moins de 0,3 pt tranchée par le lr le plus bas. AdamW
culminait au bord de la grille, donc la grille a été étendue jusqu'à encadrer
son optimum — règle symétrique, appliquée au seul optimiseur concerné.

**Retenus : Adam 3·10⁻³, AdamW 2·10⁻², RAdam 3·10⁻³.** Les trois tiennent dans
0,63 pt les uns des autres, donc aucun n'est handicapé par un mauvais réglage.

Régler sur `plain` donne au **témoin** son meilleur score : le biais joue donc
*contre* la continuation, ce qui est le bon sens pour un test qui cherche à
savoir si elle tient.

### La quantité mesurée

L'accuracy absolue dérive d'une session GPU à l'autre : le dépôt documente
jusqu'à 0,73 pt d'écart sur le même bras, à graine et fichiers identiques
(cuDNN n'est pas déterministe). On ne compare donc pas des accuracies brutes
mais le **gain apparié** :

```
Δ(méthode, optimiseur, graine) = acc(méthode) − acc(plain)
```

calculé **à l'intérieur** d'un même optimiseur, d'une même graine et d'une même
session. Les deux termes ont dérivé ensemble, la dérive s'annule.

### Les chiffres SGD sont réutilisés — et c'est vérifié

Les 12 bras SGD viennent du batch déjà enregistré. Pour valider cette
réutilisation, `plain` a été relancé sous la recette de référence :

| graine | enregistré | relancé | écart |
|---|---:|---:|---:|
| 0 | 74,58 | 74,57 | −0,01 |
| 1 | 75,66 | 75,69 | +0,03 |
| 2 | 76,05 | 76,35 | +0,30 |

Deux graines reproduisent à 0,03 pt près, la troisième à 0,30 — largement dans
la dérive documentée. **La réutilisation est solide.** Ce sont au passage les
premiers entraînements complets jamais exécutés avec le code de
`continuation-core`, et ils retombent sur les valeurs attendues.

---

## 4. Le benchmark des optimiseurs

### Accuracy test finale (%), moyenne sur 3 graines

| méthode | SGD | Adam | AdamW | RAdam |
|---|---:|---:|---:|---:|
| `plain` *(témoin)* | 75,43 | 84,13 | 84,12 | 84,38 |
| `resolution_max_b1` | 80,37 | 85,31 | **86,28** | 85,63 |
| `gaussian_postrelu` | 79,91 | 83,95 | 83,71 | 84,75 |
| `resolution_max_b1_gaussian_conv` | 81,51 | 85,28 | 86,21 | 85,26 |

### Gain de continuation apparié (points), méthode − témoin

| méthode | SGD | Adam | AdamW | RAdam |
|---|---:|---:|---:|---:|
| `resolution_max_b1` | **+4,94** | +1,18 | +2,16 | +1,25 |
| `gaussian_postrelu` | **+4,48** | −0,18 | −0,41 | +0,37 |
| `resolution_max_b1_gaussian_conv` | **+6,08** | +1,15 | +2,09 | +0,88 |

### Détail par graine, et cohérence de signe

Un gain n'est crédible que s'il va dans le même sens sur les trois graines.

| méthode | opt | moyenne | SD | même signe | par graine |
|---|---|---:|---:|:--:|---|
| `resolution_max_b1` | SGD | +4,94 | 0,50 | oui | +5,39, +4,40, +5,02 |
| | Adam | +1,18 | 0,40 | oui | +1,20, +1,57, +0,77 |
| | AdamW | +2,16 | 0,68 | oui | +2,94, +1,74, +1,80 |
| | RAdam | +1,25 | 0,53 | oui | +1,85, +1,06, +0,85 |
| `gaussian_postrelu` | SGD | +4,48 | 0,97 | oui | +5,48, +4,40, +3,55 |
| | Adam | −0,18 | 0,61 | **non** | +0,18, +0,16, −0,88 |
| | AdamW | −0,41 | 0,72 | **non** | +0,42, −0,91, −0,73 |
| | RAdam | +0,37 | 0,18 | oui | +0,48, +0,16, +0,47 |
| `resolution_max_b1_gaussian_conv` | SGD | +6,08 | 0,56 | oui | +6,72, +5,84, +5,68 |
| | Adam | +1,15 | 0,36 | oui | +1,22, +1,46, +0,76 |
| | AdamW | +2,09 | 0,62 | oui | +2,77, +1,56, +1,94 |
| | RAdam | +0,88 | 0,26 | oui | +1,11, +0,93, +0,60 |

---

## 5. Conclusions

### 5.1 — L'optimiseur vaut plus que l'intervention

Le témoin `plain` passe de **75,43 à 84,38** : **+8,95 points** pour un simple
changement d'optimiseur, contre +4 à +6 points pour la continuation.

Conséquence directe : **un ResNet-20 sans aucune intervention, sous n'importe
lequel des trois optimiseurs adaptatifs, bat tous les bras de continuation du
benchmark gelé** — dont le meilleur plafonne à 81,51.

L'intervention avait donc été mesurée contre un témoin qui laissait sur la table
environ deux fois l'effet de l'intervention elle-même.

### 5.2 — La réduction de résolution survit, mais quatre fois plus petite

+4,94 sous SGD devient +1,18 / +2,16 / +1,25. C'est faible, mais c'est **réel** :
même signe sur les trois graines pour les quatre optimiseurs, et l'effet dépasse
sa propre dispersion inter-graines.

C'est la partie du benchmark qui tient. Elle fait quelque chose que l'optimiseur
ne fait pas — simplement quatre fois moins que ce que la table SGD laissait
croire.

### 5.3 — Le flou gaussien ne survit pas

+4,48 sous SGD devient **−0,18 / −0,41 / +0,37**. Sous Adam et AdamW le signe
n'est même pas constant entre graines : l'effet est indistinguable de zéro.

Pire, le flou n'apporte plus rien **par-dessus** la réduction de résolution :

| apport du flou en plus de la résolution | |
|---|---:|
| SGD | **+1,14** |
| Adam | −0,03 |
| AdamW | −0,07 |
| RAdam | −0,37 |

`resolution_max_b1_gaussian_conv`, le meilleur bras du benchmark gelé, **ne bat
jamais `resolution_max_b1` seul** dès qu'on quitte SGD.

### 5.4 — La lecture d'ensemble

> La partie **gaussienne** de l'effet était un artefact de vitesse
> d'optimisation : un optimiseur qui adapte son pas paramètre par paramètre
> capture la même chose gratuitement.
>
> La partie **résolution** n'en est pas un : elle survit au changement
> d'optimiseur, donc elle fait quelque chose de différent — mais son amplitude
> réelle est de l'ordre de 1 à 2 points, pas de 5.

Le meilleur bras de cette campagne est **`resolution_max_b1` sous AdamW, à
86,28 %**, soit près de 5 points au-dessus du meilleur bras du benchmark gelé.

---

## 6. Limites

* **Un seul cadre.** Une architecture, un dataset, sans augmentation, 30 époques.
* **Le budget joue dans le même sens.** Les runs à 120 époques de la branche
  exploratoire montraient déjà les gains SGD fondre (témoin 86,04, plateau
  87,16). Ce rapport est un second axe de rétrécissement, pas une confirmation
  indépendante.
* **Trois graines.** Les écarts entre les trois optimiseurs adaptatifs
  (+1,18 vs +2,16 vs +1,25) sont de l'ordre de la dispersion inter-graines :
  **il ne faut pas les classer**.
* **Weight decay tenu à 5·10⁻⁴ partout**, pour garder le contraste Adam/AdamW à
  un seul facteur. C'est très bas pour un AdamW découplé : les deux sont donc
  plus proches qu'une comparaison réglée ne les mettrait, et **aucun des deux
  n'est un AdamW réglé**.
* **lr réglés sur `plain` seul.** C'est le choix conservateur pour cette
  question, mais ce n'est pas le lr qu'un bras de continuation aurait choisi
  pour lui-même.
* **Le test set a été consulté de nombreuses fois** sur l'ensemble du projet.
  Trois graines donnent une dispersion descriptive, pas un test de
  significativité.

---

## 7. Reproduire

```bash
py -m pytest -q                              # 53 tests
py scripts/stage_assets.py                   # publie les fichiers épinglés
py scripts/kaggle_run.py scripts/job_optbench_lr_sweep.py     --gpu ...   # 12
py scripts/kaggle_run.py scripts/job_optbench_lr_sweep_ext.py --gpu ...   #  2
py scripts/kaggle_run.py scripts/job_optbench_sgd_control.py  --gpu ...   #  3
py scripts/kaggle_run.py scripts/job_optbench_grid_{0,1,2}.py --gpu ...   # 36
py scripts/aggregate_optimizers.py
```

Kaggle n'autorise que **deux kernels GPU simultanés** : les tranches se lancent
deux par deux. Détails dans [docs/kaggle_cli.md](docs/kaggle_cli.md).

**Vérifications passées :** 53 tests unitaires ; parité bit-à-bit 44/44 contre
le code qui a produit le benchmark d'origine, contrôle des updates avec état
d'optimiseur inclus ; bypass de l'état cible exact (|Δ| = 0) sous Adam ;
les configurations SGD de référence identiques champ par champ à l'avant-campagne.

**Documents détaillés :**
[docs/OPTIMIZER_BENCHMARK.md](docs/OPTIMIZER_BENCHMARK.md) (campagne),
[docs/LR_SWEEP.md](docs/LR_SWEEP.md) (balayage),
[docs/METHODS.md](docs/METHODS.md) (définition exacte des quatre méthodes).
Enregistrements par run dans `results/optimizer_benchmark/`.
