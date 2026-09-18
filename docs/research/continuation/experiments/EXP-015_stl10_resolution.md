---
id: EXP-015
schema_version: 1
updated_at: 2026-09-17
status: completed_numeric_verified
evidence: local_per_epoch_json_and_kernel_log
protocol: P-STL10-R20-BN-60
source_json: ../../../../results/kaggle_outputs/stl10-resolution-20260917-121258/stl10_resolution_20260917-121308/study_summary.json
code_commit_reported: null
code_commit_inspected: false
code_commit_inspected_note: code de la branche adaptative-resolution, non commité à la rédaction
---

# EXP-015 — Résolution progressive sur STL-10 : précision à budget égal et à temps égal

[Index](INDEX.md) · [EXP-014](EXP-014_adaptive_phase2.md) · [Plan, section 16 (anglais)](../../../adaptive_resolution_plan.md) · [Analyse](../../../../results/stl10_resolution_analysis.md)

## Métadonnées

| Champ | Valeur |
|---|---|
| ID campagnes | `stl10-resolution-20260917-121258` (15 runs) ; `stl10-controls-20260917-133339` (6 runs de contrôle) |
| Date exécution | 2026-09-17, 12:13 → 13:31 UTC puis 13:33 → 14:12 UTC |
| Question utilisateur | « try w stl 10 » après la clôture des déclencheurs adaptatifs sur CIFAR-10 |
| Données | STL-10 étiqueté officiel (binaires `yellowflag/stl10labeled2`) : 5 000 train / 8 000 test, 96×96, sans augmentation, normalisation ajustée sur le train ; pas de non-étiqueté |
| Protocole | ResNet-20 BN, SGD 0,005 / 0,9 / 5e-4, warmup 60 puis cosine sur l'horizon, batch 128 en 4×32 (dernier groupe de 8 pondéré), 40 updates/époque, 60 époques = 2 400 updates sauf bras étendus. Aligné sur la branche `stl10-transfer` (témoin 57,3 %), **non apparié** avec elle |
| Code | `scripts/job_stl10_resolution.py`, `scripts/job_stl10_controls.py`, `scripts/analyze_stl10_resolution.py` |
| Vérification | JSON par époque, secondes d'entraînement par époque mesurées avec synchronisation CUDA, empreintes des états partagés enregistrées, pureté des évaluations vérifiée |

## Bras

Tailles ×3 et frontières ×2 par rapport à CIFAR : `R96` ; `Rprog` 48 ×12 / 72 ×12 / 96 ×36 ; `Rsteps4` 48/60/72/84 ×6 puis 96 ; `Rlin24` une taille paire par époque de 48 à 94 puis 96 ×36 ; `Rlin24eq` même rampe, horizon 72 (compute nominal 61,6 unités-époque à 96 contre 60 pour R96). Contrôles : `R96_72` (plain, horizon 72) ; `Rprogeq` 48 ×12 / 72 ×12 / 96 ×50 (74 époques, 59,8 unités). Trois graines partout.

## Résultats numériques

Secondes par époque (médianes) : 48 → 2,36 s ; 72 → 6,03 s ; 96 → 9,38 s, soit 0,25 / 0,64 / 1,00 contre (r/96)² = 0,25 / 0,56 / 1,00 : **les époques grossières sont réellement moins chères** (contrairement à CIFAR sur T4).

Accuracy test finale (%), graines 0/1/2 → moyenne, temps d'entraînement moyen : R96 57,66/58,76/56,59 → 57,67 (542 s) ; Rprog 58,77/57,99/59,54 → 58,77 (425 s) ; Rsteps4 59,36/58,41/58,85 → 58,88 (453 s) ; Rlin24 58,61/58,41/57,11 → 58,05 (460 s) ; Rlin24eq 60,62/60,46/59,67 → 60,25 (582 s) ; R96_72 59,27/60,72/58,56 → 59,52 (674 s) ; **Rprogeq 60,27/60,16/61,01 → 60,48 (573 s)**.

Contrastes appariés (points) : Rprogeq − R96 **+2,61 / +1,40 / +4,43** (compute nominal égal, +6 % de temps) ; Rlin24eq − R96 **+2,96 / +1,70 / +3,09** ; R96_72 − R96 +1,61 / +1,96 / +1,97 (effet du seul budget +20 %) ; Rprogeq − R96_72 +1,00 / −0,56 / +2,45 pour 15 % de temps en moins ; Rlin24eq − R96_72 +1,35 / −0,26 / +1,11 ; Rprogeq − Rlin24eq −0,35 / −0,30 / +1,34. À 60 époques, les trois bras progressifs sont à +0,4…+1,2 point de R96 en moyenne avec la graine 1 négative, pour 78–85 % de son temps.

Courbe « à temps égal » (chemin cible, interpolée sur l'horloge de chaque run) : à 25 % du temps de R96, bras progressifs 47–49 % contre 41 % ; à 50 %, 55–57 % contre 49 % ; à 75 %, 58–59 % contre 56 %. Chemin cible pendant les paliers grossiers : quelques points seulement sous le chemin courant (Rprog époque 12, r=48 : 39,1 / 34,0 / 35,7 après recalibration BN), contre 26 points d'effondrement à 16→32 sur CIFAR.

## Coût

21 runs, 415–678 s d'entraînement chacun ; 4 674 s + 2 271 s écoulées sur deux T4, ≈3,6 h GPU.

## Interprétation et limites

La résolution progressive se transpose à STL-10 avec le même signe et le même mécanisme, et y achète cette fois du temps réel. À updates égaux le gain est faible et bruité (SD du témoin seul 1,1 point sur 5 000 images) ; à compute égal il vaut +2,6 à +2,8 points dont environ deux tiers sont l'effet du budget d'updates supplémentaire ; **à temps égal, environ +2,3 points** (lecture du témoin sur sa propre horloge). Les calendriers progressifs ne se distinguent pas entre eux sur trois graines : le résultat CIFAR « rampe fine > Rprog » n'est ni confirmé ni contredit. Pas de filtre Gaussian, pas d'augmentation, pas d'appariement avec `stl10-transfer`.

## Décision et suite

Recommandation de l'agent : sur STL-10 comme sur CIFAR, un calendrier fixe grossier→fin avec l'horizon choisi pour le budget de temps, sans déclencheur ; si un contrôleur est encore souhaité, le tester ici, à temps égal, contre les calendriers étendus de ce dossier et avec plus de trois graines. Aucune décision utilisateur enregistrée.

## Sources et manques

`results/kaggle_outputs/stl10-resolution-20260917-121258/` et `stl10-controls-20260917-133339/` (logs, `shared_manifest.json`, 21 `metrics.json`/`summary.json` avec secondes par époque) ; `results/stl10_resolution_analysis.md`. Manque : commit du code ; pas de figure.

<!-- REPO-EVIDENCE:BEGIN -->

## Traces dans le dépôt (généré — ne pas éditer à la main)

Généré par `tools/enrich_records.py` depuis [`data/repo_evidence.json`](../data/repo_evidence.json). Voir aussi la vue d'ensemble : [REPO_EVIDENCE](../REPO_EVIDENCE.md).

**État du code par rapport à ce run** : job STL-10 ecrit pour ce run ; controles ajoutes entre les deux kernels ; non commite.

| Rôle | Chemin dans le dépôt | Dernier commit touchant ce fichier |
|---|---|---|
| Implémentation | [`scripts/job_stl10_resolution.py`](../../../../scripts/job_stl10_resolution.py) | — |
| Implémentation | [`scripts/job_stl10_controls.py`](../../../../scripts/job_stl10_controls.py) | — |
| Implémentation | [`scripts/analyze_stl10_resolution.py`](../../../../scripts/analyze_stl10_resolution.py) | — |
| Implémentation | [`continuation/probe_signals.py`](../../../../continuation/probe_signals.py) | — |
| Sorties d'exécution | [`results/kaggle_outputs/stl10-resolution-20260917-121258`](../../../../results/kaggle_outputs/stl10-resolution-20260917-121258) | 133 fichiers |
| Sorties d'exécution | [`results/kaggle_outputs/stl10-controls-20260917-133339`](../../../../results/kaggle_outputs/stl10-controls-20260917-133339) | 117 fichiers |
| Sorties d'exécution | [`results/stl10_resolution_analysis.md`](../../../../results/stl10_resolution_analysis.md) | — |
| Documentation du dépôt | [`docs/adaptive_resolution_plan.md`](../../../../docs/adaptive_resolution_plan.md) | — |

Un chemin listé existe dans le dépôt au moment de la génération. Cela ne prouve pas seul quelle version du code a exécuté ce run : pour les runs Kaggle, le code réellement expédié est archivé dans le sous-dossier `_repo/` de la sortie correspondante.

<!-- REPO-EVIDENCE:END -->
