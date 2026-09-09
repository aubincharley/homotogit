---
id: EXP-009
schema_version: 1
updated_at: 2026-09-09
status: completed_branch_dropped_by_user
evidence: conversation_report_and_figure
protocol: P-R20-BN-PILOT
---

# EXP-009 — Pilote db2 interne BN et abandon de la branche

[Index](INDEX.md) · [Formule db2](../OPERATORS.md) · [Décision D-15](../DECISIONS.md)

## Question et contrôle

Tester notre opérateur db2 précis dans le petit réseau BN corrigé, avant une éventuelle campagne complète. 10k train / 5k val, seed0, 2 400 updates, batch 32×4, même SGD/LR que le pilote Gaussian. Dix-neuf sites, seuils différentiés par RMS, exact bypass s=1.

Le job exécute **db2 et un plain frais**. Les artifacts du pilote précédent totalisaient environ 1,9 MB pour une limite de source kernel rapportée à 900 KB ; l'agent n'a pas pu les incorporer de cette manière. Un contrôle plain contemporain permet donc l'appariement interne et une comparaison historique explicite.

L'agent rapporte ensuite l'égalité bitwise des indices de subset/sonde, des 2 400×128 indices d'ordre, de l'initialisation et des buffers (116 tenseurs) avec le pilote Gaussian ; Torch 2.10.0+cu128, même normalisation, optimiseur et LR. Ce sont des assertions du compte rendu : leurs fichiers de preuve ne sont pas inclus localement.

## Opérateur et calendrier

Deux niveaux non décimés db2, frame normalisé, miroir 2H×2W, crop, approximation conservée, seuil \(4(1-s)2^{1-j}\sqrt{\mathrm{mean}(d^2)+10^{-12}}\). RMS non détachée, confirmé par différence de gradients avec une recomputation volontairement détachée.

Niveaux s=0 / 0,25 / 0,50 / 0,70 / 0,85 / 0,95 / 0,99, puis s=1 à k=1700, soit 1 700 updates filtrées puis 700 bypassed. Les bornes exactes de chaque bloc ne figurent pas dans le matériel local et restent à récupérer.

Vérifications rapportées sur les trois formes de stage : reconstruction sans bypass à s=1, erreur max 4,8e−7 à 7,2e−7 ; adjointness relative 0 à 1,9e−7 ; formes conservées ; constantes à environ 7,6e−6 près au maximum ; backward fini. Le bypass s=1 dans le réseau donne une différence bitwise nulle par rapport aux hooks désactivés.

## Résultats

| Updates | s courant | Plain frais accuracy | db2 courant accuracy | Gaussian historique courant |
|---:|---:|---:|---:|---:|
| 600 | 0,50 | 42,42 % | 43,30 % | 46,94 % |
| 1 200 | 0,85 | 52,32 % | 51,62 % | 56,52 % |
| 1 700 | 0,99 | 54,80 % | 55,06 % | 58,20 % |
| 2 400 | 1,00 | 55,40 % | 56,34 % | 59,62 % |

| Bras final | Accuracy val | CE val | CE sonde train |
|---|---:|---:|---:|
| Plain frais | 55,40 % | 1,2367 | 0,7561 |
| db2 | 56,34 % | 1,2007 | 1,0106 |
| Gaussian historique | 59,62 % | 1,1075 | 0,8887 |

db2−plain = **+0,94 point**, CE −0,0360. La marge n'est pas toujours positive pendant le run, et le gain final reste nettement inférieur au Gaussian. Le fit sur la sonde train est moins bon malgré les 700 updates finales sans filtre. Une seule graine, calendrier non calibré en force par rapport à sigma : on ne peut pas condamner toutes les continuations ondelettes à partir de ce test.

## Coût déterminant

| Phase T4, update complète 4×32 | Temps/update | Pic sonde |
|---|---:|---:|
| db2 s=0 | 2,057 s | 4 370 MiB |
| db2 s=0,99 | 2,101 s | 4 370 MiB |
| Bypass s=1 | 0,0336 s | 71 MiB |

Ratio d'update filtrée/bypass environ **61–63×**. Run db2 complet : **3 856 s**, soit environ 64 min, pic 4 553 MiB ; plain frais : 89 s. Le coût des snapshots explique une partie de l'écart à la projection de 3 520 s.

La projection du rapport vers six runs complets à environ 48 heures GPU ne suit pas les nombres fournis : 8 211 updates filtrées × six × 2,06–2,10 s donne plutôt 28–29 heures GPU avant évaluations et autres coûts. Cela reste beaucoup trop cher au regard des campagnes Gaussian déjà réalisées.

## Deux corrections d'interprétation

Le plain frais finit à 55,40 % contre 55,36 % historiquement, soit **0,04 point** d'écart. Cela ne constitue pas une estimation d'un plancher de bruit ±0,1 point. Le compte rendu attribue les écarts aux réductions float32 ; sans contrôle déterministe détaillé, c'est une explication plausible plutôt qu'une cause isolée.

L'affirmation ancienne « s=0 reste à environ 6 % du coarse-only » est déclarée non vérifiée : aucune mesure avec définition de normalisation n'existe dans le rapport. Ne pas la réintroduire comme propriété.

## Décision et sources

Après ce résultat, **l'utilisateur abandonne db2** et souhaite explorer d'autres idées à partir du bénéfice Gaussian. Aucun entraînement db2 supplémentaire ne doit être inféré de cette fiche.

[Job](https://www.kaggle.com/code/maxnicaise/db2-pilot-r20bn-20260908-191741), dossier `results/kaggle_outputs/db2-pilot-r20bn-20260908-191741/db2_pilot_r20bn_20260908-191900/`, [figure](../sources/db2_pilot.png). Artifacts rapportés : checkpoints 600/1200/1700/2400, `t4_timing_probe.json`, `results/db2_operator_verification.json`, scripts `verify_db2_operator.py`, `job_db2_pilot.py`, `plot_db2_pilot.py`. Ils restent à récupérer dans le dépôt pour une reproduction détaillée.

<!-- REPO-EVIDENCE:BEGIN -->

## Traces dans le dépôt (généré — ne pas éditer à la main)

Généré par `tools/enrich_records.py` depuis [`data/repo_evidence.json`](../data/repo_evidence.json). Voir aussi la vue d'ensemble : [REPO_EVIDENCE](../REPO_EVIDENCE.md).

**État du code par rapport à ce run** : table par paliers de s ajoutee au driver pour ce run ; is_active traite s=1 comme inactif.

| Rôle | Chemin dans le dépôt | Dernier commit touchant ce fichier |
|---|---|---|
| Implémentation | [`scripts/job_db2_pilot.py`](../../../../scripts/job_db2_pilot.py) | 26cfb82 2026-09-09 |
| Implémentation | [`scripts/verify_db2_operator.py`](../../../../scripts/verify_db2_operator.py) | 26cfb82 2026-09-09 |
| Implémentation | [`continuation/transforms/wavelet.py`](../../../../continuation/transforms/wavelet.py) | d643f75 2026-09-08 |
| Implémentation | [`scripts/continuation_driver.py`](../../../../scripts/continuation_driver.py) | 26cfb82 2026-09-09 |
| Sorties d'exécution | [`results/kaggle_outputs/db2-pilot-r20bn-20260908-191741`](../../../../results/kaggle_outputs/db2-pilot-r20bn-20260908-191741) | 91 fichiers |
| Sorties d'exécution | [`results/db2_operator_verification.json`](../../../../results/db2_operator_verification.json) | 26cfb82 2026-09-09 |
| Sorties d'exécution | [`results/db2_pilot.png`](../../../../results/db2_pilot.png) | 26cfb82 2026-09-09 |

Un chemin listé existe dans le dépôt au moment de la génération. Cela ne prouve pas seul quelle version du code a exécuté ce run : pour les runs Kaggle, le code réellement expédié est archivé dans le sous-dossier `_repo/` de la sortie correspondante.

<!-- REPO-EVIDENCE:END -->
