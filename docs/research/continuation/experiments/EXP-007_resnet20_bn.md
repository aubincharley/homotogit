---
id: EXP-007
schema_version: 1
updated_at: 2026-09-09
status: completed_reported
evidence: conversation_report_and_figure
protocol: P-R20-BN-PILOT
---

# EXP-007 — ResNet-20 BN, 2 400 updates et correction des courbes

[Index](INDEX.md) · Suite : [confirmation complète](EXP-008_full_gaussian.md)

## Question et choix utilisateur

Vérifier si le bénéfice persiste sur le petit ResNet-20 avec BatchNorm et initialisation corrigée. Garder 10k train / 5k validation, seed0, mais former **2 400 updates**. L'utilisateur rejette le plan initial d'extinction à 600 : le filtre doit rester jusqu'à **k=1700**, avec des paliers suffisamment longs, puis 700 updates sans filtre.

Même SGD pic 0,005, warmup 60, cosine étendu à 2 400 ; batch 32×4 ; 269 722 paramètres, option A, 19 sites. Classifieur std mesurée 0,0099, biais nul ; BN momentum 0,1. Les bornes de tous les paliers ne sont pas exposées dans les sources récupérées : conserver les valeurs connues, demander le tableau exact au dépôt.

## Résultat final

| Bras | CE sonde train | CE val | Accuracy val | Temps | Pic |
|---|---:|---:|---:|---:|---:|
| Plain | 0,7510 | 1,2335 | 55,36 % | 119 s | 394 MiB |
| Gaussian | 0,8887 | 1,1075 | 59,62 % | 174 s | 467 MiB |

**+4,26 points** d'accuracy, −0,1260 de CE validation. Après 700 updates à l'endpoint, Gaussian conserve son avantage.

## La correction qui change la lecture du pilote

La première figure mettait en avant la validation Gaussian **bypassed**. À 600 updates, elle montrait 11,18 % contre 43,66 % plain ; à 1 200, 25,82 % contre 53,06 %. Ces écarts donnaient l'impression que Gaussian n'apprenait presque pas avant la fin.

Les métriques `val_acc_filtered` existaient déjà. Leur utilisation donne :

| Updates accomplies | Sigma du diagnostic courant | Gaussian courant | Plain | Différence |
|---:|---:|---:|---:|---:|
| 200 | 1,00 | 36,86 % | 31,06 % | +5,80 points |
| 600 | 0,70 | 46,94 % | 43,66 % | +3,28 |
| 1 000 | 0,60 | 53,74 % | 50,10 % | +3,64 |
| 1 400 | 0,40 | 57,46 % | 54,14 % | +3,32 |
| 1 700 | 0,30 | 58,20 % | 54,80 % | +3,40 |
| 2 400 | 0 | 59,62 % | 55,36 % | +4,26 |

Le compte rendu corrigé indique un avantage à chaque checkpoint mesuré. L'évaluation cible prématurée mesurait un changement de chemin et un possible décalage BN ; **l'isolement de la seule contribution BN n'a pas été effectué**. Au checkpoint 1 700, la dernière update était encore filtrée, ce qui explique sigma 0,30 dans le diagnostic courant.

## Ce que cette expérience permet de dire

Le grand ResNet-18 n'est pas nécessaire à l'effet positif observé. Mais l'ancienne comparaison ResNet-20 GN avait un horizon, une histoire de LR et une continuation différents. Le changement de signe ne prouve pas que BN ou l'initialisation seule en est la cause.

Le point à 1 200 d'un run de 2 400 n'est pas une réplique contrôlée du précédent run dont le LR était déjà nul à 1 200. Cette nuance empêche un faux contrôle architecture-only.

## Artifacts

[Job](https://www.kaggle.com/code/maxnicaise/resnet20bn-gaussian-20260908-154226), dossier `results/kaggle_outputs/resnet20bn-gaussian-20260908-154226/`, checkpoints u600/u1200/u1700/u2400 rapportés reprenables.

[Figure initiale disponible](../sources/resnet20bn_plain_vs_gaussian.png). La version `results/resnet20bn_pilot_corrected.png` est citée dans les échanges mais non récupérée. Les métriques corrigées du tableau viennent du compte rendu utilisateur, pas d'une reconstitution graphique.

<!-- REPO-EVIDENCE:BEGIN -->

## Traces dans le dépôt (généré — ne pas éditer à la main)

Généré par `tools/enrich_records.py` depuis [`data/repo_evidence.json`](../data/repo_evidence.json). Voir aussi la vue d'ensemble : [REPO_EVIDENCE](../REPO_EVIDENCE.md).

**État du code par rapport à ce run** : modele inchange depuis ; driver modifie apres coup.

| Rôle | Chemin dans le dépôt | Dernier commit touchant ce fichier |
|---|---|---|
| Implémentation | [`continuation/models/resnet20_bn.py`](../../../../continuation/models/resnet20_bn.py) | d643f75 2026-09-08 |
| Implémentation | [`scripts/job_resnet20bn_gaussian.py`](../../../../scripts/job_resnet20bn_gaussian.py) | d643f75 2026-09-08 |
| Implémentation | [`scripts/continuation_driver.py`](../../../../scripts/continuation_driver.py) | 26cfb82 2026-09-09 |
| Sorties d'exécution | [`results/kaggle_outputs/resnet20bn-gaussian-20260908-154226`](../../../../results/kaggle_outputs/resnet20bn-gaussian-20260908-154226) | 8 fichiers |
| Sorties d'exécution | [`results/resnet20bn_plain_vs_gaussian.png`](../../../../results/resnet20bn_plain_vs_gaussian.png) | d643f75 2026-09-08 |
| Sorties d'exécution | [`results/resnet20bn_pilot_corrected.png`](../../../../results/resnet20bn_pilot_corrected.png) | d643f75 2026-09-08 |

Un chemin listé existe dans le dépôt au moment de la génération. Cela ne prouve pas seul quelle version du code a exécuté ce run : pour les runs Kaggle, le code réellement expédié est archivé dans le sous-dossier `_repo/` de la sortie correspondante.

<!-- REPO-EVIDENCE:END -->
