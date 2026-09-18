---
id: EXP-014
schema_version: 1
updated_at: 2026-09-17
status: completed_numeric_verified
evidence: local_per_epoch_json_and_kernel_log
protocol: P-FULL-R20-BN
source_json: ../../../../results/kaggle_outputs/adaptive-phase2b-20260917-113110/adaptive_phase2_20260917-113121/study_summary.json
code_commit_reported: null
code_commit_inspected: false
code_commit_inspected_note: code de la branche adaptative-resolution, non commité à la rédaction ; premier lancement des bras contrôleur invalidé (voir §Exécution)
---

# EXP-014 — Contrôleur vivant de taille de pas sur l'efficacité de transfert ; rampe à tailles paires

[Index](INDEX.md) · [EXP-013](EXP-013_adaptive_phase1.md) · [Plan, section 15 (anglais)](../../../adaptive_resolution_plan.md) · [Analyse](../../../../results/adaptive_phase0_analysis.md)

## Métadonnées

| Champ | Valeur |
|---|---|
| ID campagnes | `adaptive-phase2-20260917-093614` (rampe paire valide ; bras contrôleur **invalides**) ; `adaptive-phase2b-20260917-113110` (bras contrôleur corrigés) |
| Date exécution | 2026-09-17, 09:36 → 10:20 UTC puis 11:31 → 12:01 UTC |
| Question utilisateur | « Réaliser les propositions » (plan §13.3) |
| Profil | `P-FULL-R20-BN` inchangé ; horizon 11 730 updates et LR identiques |
| Code | `scripts/job_adaptive_phase2.py` → `job_adaptive_phase0.py` (`ADAPT_PHASE=2`) |
| Vérification | JSON par époque, journal des décisions par run, huit empreintes d'assets vérifiées, pureté des mesures vérifiée, garde « résolution d'entraînement = calendrier » active dans la relance |

## Question

Un contrôleur qui avance la résolution de 4 quand la moyenne mobile (β = 0,5, réamorcée après chaque changement) de τ(r → r+4), mesurée en mode train sur 2 000 images, passe sous θ ∈ {0,50 ; 0,65}, avec plancher r ≥ 16 + 4(e − 8) dès l'époque 8 (32 garanti à l'époque 12), produit-il un calendrier meilleur que les rampes fixes d'EXP-013 ? Lectures fixées à l'avance : mêmes époques de bascule sur toutes les graines → meilleur calendrier fixe ; bascules différentes et accuracy ≥ Rsteps4 → adaptation réelle ; contrôleur ≤ Rprog ou dominé par le plancher → idée abandonnée.

## Exécution réelle

Premier lancement : `SiteController` conserve sa **propre copie** de la table de résolutions ; les décisions modifiaient la liste du job mais pas le chemin d'entraînement. Les six réseaux `Rctrl*` se sont entraînés à 16×16 pendant 30 époques tout en étant évalués à la résolution décidée (25–38 % final). Marqués `INVALID.json`, exclus de l'analyse ; correction (mise à jour de la table du contrôleur + garde levant une exception si la résolution utilisée diffère du calendrier) puis relance des six cellules. Les trois cellules `Rlin12even` du premier lancement sont valides.

## Résultats numériques

**Le déclencheur n'a jamais tiré.** Moyenne mobile de τ(+4) entre 0,62 et 0,94 pendant neuf époques à 16, puis ≥ 0,71 ; les 24 bascules sont dues au plancher. Calendrier réalisé identique partout : 16 × 9 → 20 → 24 → 28 → 32 dès l'époque 12.

| Bras | Graines 0 / 1 / 2 (%) | Moyenne | vs Rprog | vs Rsteps4 | vs Rlin12 |
|---|---|---:|---:|---:|---:|
| Rctrl50 | 78,61 / 78,88 / 80,29 | 79,26 | +0,12 | −0,76 | −0,88 |
| Rctrl65 | 78,57 / 79,21 / 79,56 | 79,11 | −0,03 | −0,90 | −1,03 |
| Rlin12even (16,18,18,20,22,22,24,26,26,28,30,30, puis 32) | 80,07 / 79,84 / 80,45 | 80,12 | +0,98 | +0,10 | −0,02 |

Deux bras à calendrier identique : écarts par graine 0,04 / 0,33 / 0,73 point = bruit de ré-exécution mesuré.

## Coût

Relance : 6 runs, 570–592 s (86–90 s de mesures), 1 792 s écoulées ; premier lancement : 9 runs, 2 615 s. ≈1,9 h GPU au total, dont 0,8 h perdue par le bogue.

## Interprétation et limites

Troisième lecture pré-enregistrée : contrôleur dominé par sa garde, au niveau de Rprog. Le calendrier plancher a **la même masse d'époques grossières (12), le même pas (4) et la même époque terminale (12) que Rsteps4** ; seule la répartition du séjour diffère (9 époques à 16 puis une à chaque taille, contre trois partout) et coûte environ un point sur chaque graine. Le gain des rampes fines tient donc à une **répartition égale du séjour entre échelles**, pas à l'épuisement d'un palier, que l'alignement des gradients ne détecte pas. La parité des tailles n'a pas d'effet sur l'accuracy finale. Trois graines, test déjà consulté, aucune augmentation.

## Décision et suite

Recommandation de l'agent : clore la piste « déclencheur adaptatif » sur CIFAR-10 (trois familles de signaux testées, aucune ne bat un calendrier fixe) ; retenir la rampe fine à séjour égal (`Rlin12` / `Rsteps4`) comme référence non filtrée ; suites proposées dans le plan §15.3 (frontière précision/temps contre `stem_max`, augmentation, STL-10). Aucune décision utilisateur enregistrée.

## Sources et manques

`results/kaggle_outputs/adaptive-phase2-20260917-093614/` et `adaptive-phase2b-20260917-113110/` (logs, `pairing_verification.json`, `metrics.json`/`summary.json` avec `controller_log`) ; `results/adaptive_phase0_analysis.{json,md}`. Manque : commit du code ; pas de figure.

<!-- REPO-EVIDENCE:BEGIN -->

## Traces dans le dépôt (généré — ne pas éditer à la main)

Généré par `tools/enrich_records.py` depuis [`data/repo_evidence.json`](../data/repo_evidence.json). Voir aussi la vue d'ensemble : [REPO_EVIDENCE](../REPO_EVIDENCE.md).

**État du code par rapport à ce run** : controleur ajoute a job_adaptive_phase0.py ; bogue de copie de calendrier corrige entre les deux kernels ; non commite.

| Rôle | Chemin dans le dépôt | Dernier commit touchant ce fichier |
|---|---|---|
| Implémentation | [`continuation/probe_signals.py`](../../../../continuation/probe_signals.py) | df89fbf 2026-09-18 |
| Implémentation | [`scripts/job_adaptive_phase0.py`](../../../../scripts/job_adaptive_phase0.py) | df89fbf 2026-09-18 |
| Implémentation | [`scripts/job_adaptive_phase2.py`](../../../../scripts/job_adaptive_phase2.py) | df89fbf 2026-09-18 |
| Implémentation | [`scripts/analyze_adaptive_phase0.py`](../../../../scripts/analyze_adaptive_phase0.py) | df89fbf 2026-09-18 |
| Sorties d'exécution | [`results/kaggle_outputs/adaptive-phase2-20260917-093614`](../../../../results/kaggle_outputs/adaptive-phase2-20260917-093614) | 130 fichiers |
| Sorties d'exécution | [`results/kaggle_outputs/adaptive-phase2b-20260917-113110`](../../../../results/kaggle_outputs/adaptive-phase2b-20260917-113110) | 118 fichiers |
| Sorties d'exécution | [`results/adaptive_phase0_analysis.json`](../../../../results/adaptive_phase0_analysis.json) | df89fbf 2026-09-18 |
| Documentation du dépôt | [`docs/adaptive_resolution_plan.md`](../../../../docs/adaptive_resolution_plan.md) | df89fbf 2026-09-18 |

Un chemin listé existe dans le dépôt au moment de la génération. Cela ne prouve pas seul quelle version du code a exécuté ce run : pour les runs Kaggle, le code réellement expédié est archivé dans le sous-dossier `_repo/` de la sortie correspondante.

<!-- REPO-EVIDENCE:END -->
