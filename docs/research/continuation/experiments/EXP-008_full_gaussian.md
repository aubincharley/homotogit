---
id: EXP-008
schema_version: 1
updated_at: 2026-09-09
status: completed_reported
evidence: conversation_report_and_figure
protocol: P-FULL-R20-BN
---

# EXP-008 — Gaussian sur CIFAR-10 complet, trois graines

[Index](INDEX.md) · [Profil commun](../PROTOCOLS.md) · Réplication ultérieure : [EXP-011](EXP-011_full_grid.md)

## Question et réalisation

Le gain du pilote se conserve-t-il avec toutes les images d'entraînement et plusieurs initialisations ? Trois bras, graines 0/1/2, soit **neuf runs** : plain, Gaussian paliers, Gaussian géométrique comprimé.

50k train / 10k test, ResNet-20 BN corrigé, 30 époques = 11 730 updates, batch 32×4 avec dernier groupe 80, SGD pic 0,005, warmup **60** puis cosine, pas d'augmentation. Initialisations, buffers BN et permutations par époque identiques par graine entre bras. Les statistiques RGB sont calculées sur le train complet.

Paliers par blocs de trois époques : 1 / 0,85 / 0,70 / 0,60 / 0,50 / 0,40 / 0,30 ; puis zéro à e=21. Géométrique : 0,9^e jusqu'à e=20, puis zéro. Les deux ont 21 époques filtrées et neuf sans filtre. Ce n'est pas le calendrier original de CBS.

## Résultat final à epoch 30, pas meilleur checkpoint

| Bras | Accuracy (%) moyenne ± SD | CE test moyenne ± SD | Seeds 0 / 1 / 2 (%) |
|---|---:|---:|---|
| Plain | 75,40 ± 0,40 | 0,7615 ± 0,0039 | 75,10 / 75,26 / 75,85 |
| Paliers | 78,41 ± 0,32 | 0,6259 ± 0,0094 | 78,27 / 78,78 / 78,19 |
| Géométrique | 78,46 ± 0,72 | 0,6636 ± 0,0255 | 77,63 / 78,84 / 78,92 |

| Différence appariée | Par graine (points) | Moyenne ± SD (points) |
|---|---|---:|
| Paliers − plain | +3,17 / +3,52 / +2,34 | +3,01 ± 0,61 |
| Géométrique − plain | +2,53 / +3,58 / +3,07 | +3,06 ± 0,53 |
| Géométrique − paliers | −0,64 / +0,06 / +0,73 | +0,05 ± 0,69 |

Les deux variantes gagnent sur toutes les graines contre plain. Leur différence d'accuracy est petite et change de signe ; les paliers ont une CE finale plus basse. Ne pas convertir cela en preuve de meilleure calibration ni en équivalence statistique des deux calendriers.

## Trajectoire et lecture

Le bras paliers commence derrière plain puis le dépasse vers les époques 16–18 sur la figure. Le géométrique semble le dépasser plus tôt, vers 8–10 : la phrase originale « les deux restent derrière environ 14 époques » était trop globale.

Le gain reste présent après neuf époques sans filtre. La CE de sonde train plus élevée et la meilleure performance test sont compatibles avec une régularisation par la trajectoire. L'absence de saut majeur visible à l'extinction ne démontre pas un coût de transition exactement nul à chaque update.

## Coût et sources

Temps rapportés par run : plain environ 458 s, paliers 693 s, géométrique 684 s ; pics 511–584 MiB. Total exact rapporté **5 504 s GPU**, soit environ 92 min cumulées et 46 min écoulées à deux T4. Ne pas prendre la somme des temps arrondis pour une nouvelle somme exacte.

[Job](https://www.kaggle.com/code/maxnicaise/fulldata-r20bn-20260908-161221), dossier `results/kaggle_outputs/fulldata-r20bn-20260908-161221/`, [figure](../sources/fulldata_campaign.png). Checkpoints epoch21/30 et assets partagés rapportés présents. JSON bruts de ces neuf runs non récupérés dans cette passation.

## Limites et statut actuel

Les calendriers et le checkpoint final ont été fixés, mais le test a été suivi pendant l'entraînement. L'expérience est exploratoire. Ne pas agréger les pilotes 10k avec cette campagne. La grille EXP-011 réentraîne ces trois configurations ; ses nombres diffèrent légèrement et doivent rester associés à leur propre exécution.

<!-- REPO-EVIDENCE:BEGIN -->

## Traces dans le dépôt (généré — ne pas éditer à la main)

Généré par `tools/enrich_records.py` depuis [`data/repo_evidence.json`](../data/repo_evidence.json). Voir aussi la vue d'ensemble : [REPO_EVIDENCE](../REPO_EVIDENCE.md).

**État du code par rapport à ce run** : assets partages de ce run (init_seed*.pt, shared_indices.npz) reutilises comme etat epingle d'EXP-010 et EXP-011.

| Rôle | Chemin dans le dépôt | Dernier commit touchant ce fichier |
|---|---|---|
| Implémentation | [`scripts/job_fulldata_campaign.py`](../../../../scripts/job_fulldata_campaign.py) | d643f75 2026-09-08 |
| Implémentation | [`scripts/continuation_driver.py`](../../../../scripts/continuation_driver.py) | 26cfb82 2026-09-09 |
| Sorties d'exécution | [`results/kaggle_outputs/fulldata-r20bn-20260908-161221`](../../../../results/kaggle_outputs/fulldata-r20bn-20260908-161221) | 120 fichiers |
| Sorties d'exécution | [`results/fulldata_campaign.png`](../../../../results/fulldata_campaign.png) | d643f75 2026-09-08 |
| Documentation du dépôt | [`docs/HANDOVER.md`](../../../../docs/HANDOVER.md) | 5ec3217 2026-09-09 |

Un chemin listé existe dans le dépôt au moment de la génération. Cela ne prouve pas seul quelle version du code a exécuté ce run : pour les runs Kaggle, le code réellement expédié est archivé dans le sous-dossier `_repo/` de la sortie correspondante.

<!-- REPO-EVIDENCE:END -->
