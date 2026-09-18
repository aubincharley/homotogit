---
id: EXP-006
schema_version: 1
updated_at: 2026-09-09
status: completed_reported
evidence: conversation_report_and_figure
protocol: P-R18-BN
---

# EXP-006 — Pilote ResNet-18 BatchNorm et initialisation corrigée

[Index](INDEX.md) · [Modèle](../PROTOCOLS.md) · Suite : [petit réseau BN](EXP-007_resnet20_bn.md)

## Question et protocole

Le signe défavorable du Gaussian persiste-t-il avec une architecture et une initialisation proches de la référence, sous notre protocole court ? Conserver les mêmes 10k train / 5k validation, seed0, batch 32×4, SGD pic 0,005, momentum 0,9, WD 0,0005, warmup 60 puis cosine sur **1 200 updates**. Pas d'augmentation.

ResNet-18 CIFAR BN corrigé, 11 173 962 paramètres, 17 sites Gaussian, trois raccourcis projetés non filtrés. \(\sigma_k=\max(1-k/600,0)\), avec 600 updates finales sans filtre. Poids initiaux appris, buffers BN et ordre des batches appariés.

Le rayon 4 pose un problème au padding natif des cartes 4×4 ; une réflexion répétée explicite a été ajoutée. Le noyau 9 taps, les 17 sites et les strides sont conservés. Les vérifications rapportées couvrent formes, constantes, accord avec le padding natif lorsque valide, gradients et backward fini.

## Résultat

| Bras | CE sonde train finale | CE validation | Accuracy validation | Temps |
|---|---:|---:|---:|---:|
| Plain | 0,0530 | 1,5071 | 54,26 % | 151 s |
| Gaussian | 0,0987 | 1,1019 | 62,16 % | 197 s |

**+7,90 points** d'accuracy, −0,4052 de CE validation. Le gain subsiste après 600 updates sans filtre. La sonde train est moins ajustée par Gaussian, mais la généralisation est meilleure dans ce pilote.

## Timing et incidents

Sonde T4 : plain 110,8 ms/update, Gaussian 175,3 ms ; pics de sonde 345/672 MiB. Pics des runs complets environ 1 439/1 442 MiB. Ces périmètres ne sont pas interchangeables. Une estimation préalable 3,72 min en parallèle a déclenché le run sous le plafond pilote alors autorisé.

Une interruption avait coupé le téléchargement mais le kernel était déjà terminé. L'agent a contrôlé l'état puis récupéré les sorties sans doublonner le calcul. Cette leçon motive la vérification des jobs terminés avant une resoumission.

## Interprétation et limites

Le signe change par rapport au petit modèle GN, ce qui justifie de poursuivre. On a toutefois changé **capacité, architecture, normalisation et initialisation ensemble**. L'effet ne peut pas être attribué à BN seule.

La CE bypassed élevée en début de continuation (environ 4,81 à une mesure rapportée) décrit le retrait du filtre aux mêmes états, pas le prédicteur courant. Le résultat est compatible avec une régularisation implicite par la trajectoire ; il ne démontre pas une meilleure minimisation finale de la CE train ni une géométrie particulière des minima.

## Sources et artifacts

[Job](https://www.kaggle.com/code/maxnicaise/resnet18-gaussian-20260908-144640), dossier `results/kaggle_outputs/resnet18-gaussian-complete/`, [figure récupérée](../sources/resnet18_plain_vs_gaussian.png). Les nombres exacts proviennent du compte rendu fourni par l'utilisateur ; les logs/checkpoints bruts n'ont pas été relus ici.

<!-- REPO-EVIDENCE:BEGIN -->

## Traces dans le dépôt (généré — ne pas éditer à la main)

Généré par `tools/enrich_records.py` depuis [`data/repo_evidence.json`](../data/repo_evidence.json). Voir aussi la vue d'ensemble : [REPO_EVIDENCE](../REPO_EVIDENCE.md).

**État du code par rapport à ce run** : le repli de reflexion explicite pour cartes 4x4 a ete ajoute pour ce run ; voir gaussian.py.

| Rôle | Chemin dans le dépôt | Dernier commit touchant ce fichier |
|---|---|---|
| Implémentation | [`continuation/models/resnet18_bn.py`](../../../../continuation/models/resnet18_bn.py) | d643f75 2026-09-08 |
| Implémentation | [`scripts/job_resnet18_gaussian.py`](../../../../scripts/job_resnet18_gaussian.py) | d643f75 2026-09-08 |
| Implémentation | [`scripts/continuation_driver.py`](../../../../scripts/continuation_driver.py) | 26cfb82 2026-09-09 |
| Sorties d'exécution | [`results/kaggle_outputs/resnet18-gaussian-20260908-144640`](../../../../results/kaggle_outputs/resnet18-gaussian-20260908-144640) | 5 fichiers |
| Sorties d'exécution | [`results/resnet18_plain_vs_gaussian.png`](../../../../results/resnet18_plain_vs_gaussian.png) | d643f75 2026-09-08 |

Un chemin listé existe dans le dépôt au moment de la génération. Cela ne prouve pas seul quelle version du code a exécuté ce run : pour les runs Kaggle, le code réellement expédié est archivé dans le sous-dossier `_repo/` de la sortie correspondante.

<!-- REPO-EVIDENCE:END -->
