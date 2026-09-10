# Ce que contient out/

Une expérience par dossier. Rien n'est jamais écrasé : un nouveau protocole va
dans un nouveau dossier, et les figures vivent à côté des données qui les ont
produites.

## exp1_50k_40ep/ -- le premier essai, axe activation seul
50 000 images, 40 époques, 1 seed. Pas d'ancre.

| bras       | test acc | note                                     |
|------------|----------|------------------------------------------|
| baseline40 | 0.9397   | ReLU standard, la référence              |
| act_linear | 0.9320   | alpha 1->0 linéaire sur la 1re moitié    |

L'homotopie perd 0,77 %. Courbure mesurée à la fin : top eigenvalue 48 contre
35, trace 433 contre 166 -- elle finit dans un minimum plus pointu, ce qui est
cohérent avec sa moins bonne généralisation.

figures/ : compare_updates, alpha, loss_vs_alpha, branch, curvature

## exp2_lambda_sweep_1seed/ -- l'ancre GRDH, balayée
10 000 images, 50 époques, 1 seed. Escalier de 10 paliers de 4 époques
(alpha 1.000 -> 0.000), puis 14 époques à alpha=0.

| bras          | lambda | test acc | test loss |
|---------------|--------|----------|-----------|
| fast_baseline | --     | 0.8353   | 0.7155    |
| fast_steps_l0 | 0      | 0.8335   | 0.5874    |
| fast_steps_l4 | 1e-4   | 0.8311   | 0.5853    |
| fast_steps_l3 | 1e-3   | 0.8297   | --        |
| fast_steps_l2 | 1e-2   | 0.8303   | 0.5652    |

Deux conclusions :
  * l'ancre n'a pas d'effet mesurable. lambda varie d'un facteur 100, les
    accuracies tiennent dans 0,38 % et l'ordre n'est pas monotone.
  * la continuation discrète baisse la test loss de 18 % à accuracy égale, et
    cet effet est entièrement présent à lambda=0 -- il vient de l'escalier.

figures/compare_updates.png : les cinq bras superposés.

## exp3_3seeds/ -- confirmation du seul signal solide
10 000 images, 50 époques, seeds 0/1/2. baseline contre escalier sans ancre.
Une seule question : l'écart de test loss survit-il au bruit de seed ?

## exp3_3seeds/ -- RESULTATS
10 000 images, 50 époques, seeds 0/1/2. baseline contre escalier sans ancre.

|            | baseline          | escalier (lambda=0) | ecart            |
|------------|-------------------|---------------------|------------------|
| test acc   | 0.8374 +- 0.0014  | 0.8278 +- 0.0024    | -0.96 %  (3.4 s) |
| test loss  | 0.7080 +- 0.0115  | 0.5886 +- 0.0074    | -17 %    (8.7 s) |
| train acc  | 0.9992            | 0.9655              |                  |
| gen gap    | 0.1618            | 0.1376              |                  |

Les deux effets sont significatifs et opposés : la continuation coûte ~1 %
d'accuracy et gagne 17 % de loss. Aucun recouvrement entre seeds sur l'une
comme sur l'autre.

Correction d'une conclusion antérieure : à 1 seed l'écart d'accuracy valait
-0,18 % et avait été lu comme du bruit. Il vaut -0,96 % sur trois seeds. Le
seed 0 était le plus favorable à l'escalier.

figures/compare_updates.png : moyenne, bande +-1 ecart-type, et les trois runs
individuels en traits fins.

## exp4_full80_3seeds/ -- EN COURS
50 000 images (tout le jeu), 80 époques, seeds 0/1/2. Même protocole qu'exp3 :
baseline contre escalier de 10 paliers, sans ancre, 22 époques finales à
alpha=0 (28 % du budget, la même proportion qu'exp3).

La question : le gain de 17 % sur la loss est-il propre à l'homotopie, ou
n'était-ce que "moins entraîner régularise" ? A 10k le baseline mémorise
(gap +16,2 %) et tout ce qui freine l'apprentissage aide la loss. A 50k son
gap tombe à +5,9 %, donc cette explication disparaît largement -- et c'est ce
run qui sépare les deux lectures.

## exp5_plainnet_seed1/ -- l'homotopie là où elle devrait servir
PlainNet-18 : le ResNet-18 avec le `+ x` retiré de chaque bloc, et rien d'autre
de changé. 50 000 images, 50 époques, seed 1.

Hypothèse : les connexions résiduelles convexifient déjà le paysage (Li et al.
2018), donc l'homotopie n'a rien à corriger sur un ResNet -- ce que les
expériences 1 à 4 ont montré. Sans les skips le paysage redevient accidenté, et
c'est le régime pour lequel un départ quasi-linéaire a été conçu. C'est aussi
un régime où la baseline elle-même est en difficulté.

L'ablation ne porte que sur `use_residual`. Mêmes canaux, mêmes strides, mêmes
BatchNorms, mêmes activations, même recette. 11,00M paramètres au lieu de
11,17M : les seuls qui disparaissent sont les trois projections 1x1 et leurs
BatchNorms, qui n'existent que pour rendre x additionnable.

Piège écarté : zero_init_residual met gamma_bn2 à 0, et sans x à réajouter
chaque bloc sort exactement zéro -- le réseau devient une constante. La
validation refuse la combinaison.

| run                      | alpha            | lambda |
|--------------------------|------------------|--------|
| plain_baseline50         | 0 (ReLU pur)     | 0      |
| plain_act_linear50       | 1->0 sur 25 ep   | 0      |
| plain_act_anchor50       | 1->0 sur 25 ep   | 1e-4   |
| plain_act_staircase50    | 5 paliers x 10ep | 1e-4   |
