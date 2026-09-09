---
id: EXP-004
schema_version: 1
updated_at: 2026-09-09
status: completed_partial_record
evidence: figure_and_older_handoff
protocol: P-INT-EARLY
---

# EXP-004 — Premier pilote interne, avec db2

[Index](INDEX.md) · [Figure source](../sources/pilot_comparison.png)

## Ce qui est récupéré

Le titre de la figure indique CIFAR-10, **5 000 images**, **600 updates**, **une graine**. Trois branches : plain, Gaussian, db2. La continuation s'achève à 300 updates. Les courbes d'entraînement affichées sont bypassed, et la validation est sans filtrage.

La précédente passation avait lu graphiquement, à la fin, environ **12 % pour plain**, **22,5 % pour Gaussian**, **25,8 % pour db2**. La CE initiale se situe autour de 6,2. Ces nombres sont **approximatifs, lus sur une figure**, pas des valeurs finales extraites de logs.

## Ce que cela établit et n'établit pas

Il existe bien un entraînement db2 antérieur au pilote BN récent ; prétendre que db2 n'avait jamais été entraîné serait faux. Mais le témoin de ce premier test apprend peu, et les métriques ne décrivent pas le chemin courant des modèles filtrés. Ce résultat ne constitue pas une validation solide de la méthode.

## Informations manquantes

Valeur exacte de la graine, indices de données, split de validation, recette LR complète, initialisation effectivement utilisée, calendrier db2, chronométrage, états et logs bruts. L'ancien profil GN est plausible dans la chronologie, mais la figure seule ne suffit pas à établir chaque détail.

Récupérer le run d'origine si une analyse quantitative devient nécessaire. Ne pas reconstruire ces paramètres en recopiant ceux d'EXP-007 ou d'EXP-009. Source complémentaire : [passation du 8 septembre](../sources/passation_2026-09-08.md), section 6.3.

<!-- REPO-EVIDENCE:BEGIN -->

## Traces dans le dépôt (généré — ne pas éditer à la main)

Généré par `tools/enrich_records.py` depuis [`data/repo_evidence.json`](../data/repo_evidence.json). Voir aussi la vue d'ensemble : [REPO_EVIDENCE](../REPO_EVIDENCE.md).

**État du code par rapport à ce run** : pilote ancien ; sorties partiellement telechargees, summary.json absent pour plusieurs cellules.

| Rôle | Chemin dans le dépôt | Dernier commit touchant ce fichier |
|---|---|---|
| Implémentation | [`scripts/kaggle_pilot_continuation.py`](../../../../scripts/kaggle_pilot_continuation.py) | d643f75 2026-09-08 |
| Implémentation | [`scripts/kaggle_run.py`](../../../../scripts/kaggle_run.py) | 776c0a1 2026-09-09 |
| Sorties d'exécution | [`results/kaggle_outputs/pilot-continuation-20260908-103434`](../../../../results/kaggle_outputs/pilot-continuation-20260908-103434) | 51 fichiers |
| Sorties d'exécution | [`results/kaggle_outputs/pilot-continuation-20260908-122558`](../../../../results/kaggle_outputs/pilot-continuation-20260908-122558) | 67 fichiers |
| Documentation du dépôt | [`docs/kaggle_cli.md`](../../../../docs/kaggle_cli.md) | 0160bd1 2026-09-09 |

Un chemin listé existe dans le dépôt au moment de la génération. Cela ne prouve pas seul quelle version du code a exécuté ce run : pour les runs Kaggle, le code réellement expédié est archivé dans le sous-dossier `_repo/` de la sortie correspondante.

<!-- REPO-EVIDENCE:END -->
