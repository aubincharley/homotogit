---
id: EXP-016
schema_version: 1
updated_at: 2026-09-17
status: completed_numeric_verified
evidence: local_per_epoch_json_and_kernel_log
protocol: P-FULL-R20-BN
source_json: ../../../../results/kaggle_outputs/adaptive-phase3-20260917-142057/adaptive_phase3_20260917-142107/study_summary.json
code_commit_reported: null
code_commit_inspected: false
code_commit_inspected_note: code de la branche adaptative-resolution, non commité à la rédaction
---

# EXP-016 — Contrôleur à spécialisation d'échelle bornée : la résolution pilotée par un signal

[Index](INDEX.md) · [EXP-014](EXP-014_adaptive_phase2.md) · [Plan, section 17 (anglais)](../../../adaptive_resolution_plan.md)

## Métadonnées

| Champ | Valeur |
|---|---|
| ID campagne | `adaptive-phase3-20260917-142057` (kernel Kaggle, compte martingraffin) |
| Date exécution | 2026-09-17, 14:21 → 15:09 UTC (2 893 s) |
| Question utilisateur | « Un signal qui mesure la spécialisation d'échelle et met à jour l'échelle adaptativement pendant l'entraînement » |
| Profil | `P-FULL-R20-BN` inchangé ; horizon et LR fixes ; sans filtre |
| Code | `scripts/job_adaptive_phase3.py` → `job_adaptive_phase0.py` (`ADAPT_PHASE=3`), `continuation/probe_signals.py` |
| Vérification | JSON par époque avec journal des décisions, huit empreintes d'assets vérifiées, garde « résolution d'entraînement = calendrier », pureté des mesures vérifiée |

## Signal et loi

Spécialisation d'échelle : g_Δ(θ) = [L_{r+Δ}^{recal BN}(θ) − L_r(θ)] / L_r(θ), pénalité relative de perte du réseau courant présenté à l'échelle suivante, part BN retirée (statistiques recalibrées à poids fixés sur 2 000 images d'entraînement, restaurées ensuite). Rejouée sur toutes les traces d'EXP-012 à 014 : nulle à l'initialisation, croît quasi linéairement avec le séjour (≈ +0,03/époque à r=16 pour Δ=4, deux fois plus vite pour Δ=8), se réinitialise au changement d'échelle, stable entre graines (±0,01), et **classe les calendriers fixes par accuracy finale** selon sa valeur au moment des bascules (Rlin12/Rsteps4 ≤ 0,09 ; Rprog 0,21 ; Rb12_18 0,63). Loi : moyenne mobile β = 0,5 réamorcée après chaque bascule ; avancer de Δ = 4 quand g ≥ g* ; garde r ≥ 16 + 4(e − 11) dès l'époque 11 (32 garanti à 15). Trois seuils g* ∈ {0,04 ; 0,06 ; 0,08}, graines 0/1/2. Un compagnon sans étiquettes (décalage des statistiques BN) est journalisé.

## Résultats numériques

35 bascules sur 36 décidées par le signal (une par la garde : g* = 0,08, graine 2, dernier pas). Séjours réalisés (époques à 16/20/24/28) : g* = 0,04 : 5/1/5/1, 4/3/2/4, 4/3/3/1 ; g* = 0,06 : 6/2/3/2, 5/2/6/1, 6/2/3/1 ; g* = 0,08 : 6/3/3/3, 6/3/4/2, 7/3/3/2 ; arrivée à 32 entre les époques 11 et 15.

Accuracy test finale (%), graines 0/1/2 → moyenne : g* = 0,04 : 79,61/79,85/80,49 → 79,98 ; **g* = 0,06 : 79,86/79,83/80,38 → 80,02** ; g* = 0,08 : 79,09/79,68/80,56 → 79,78. Références mêmes graines (EXP-013) : Rprog 79,14 ; Rsteps4 80,02 ; Rlin12 80,14. Les neuf runs battent Rprog (+0,5 à +1,2 point par graine) ; aucun ne dépasse les rampes fines fixes (−0,04 / 0,00 / −0,24 contre Rsteps4).

Compagnon sans étiquettes : décalage BN plat sur le séjour (0,023–0,025 à r=16 des époques 1 à 6 pendant que g passe de 0,00 à 0,10) ; ce n'est pas une mesure de spécialisation.

## Coût

9 runs, 560–620 s chacun dont ≈ 100 s de mesures ; 2 893 s écoulées sur deux T4, ≈ 1,6 h GPU.

## Interprétation et limites

Première loi adaptative qui décide elle-même : elle retrouve, à partir du signal, l'accuracy des meilleurs calendriers écrits à la main (+0,9 point sur le bras 16/24/32 d'EXP-011), avec un seul hyperparamètre robuste sur un facteur deux, et choisit un séjour non uniforme (le plus long à la première échelle) qu'aucune table ne proposait. Elle ne dépasse pas les rampes fixes : la surface d'accuracy autour d'elles est plate sur cette recette (EXP-012 §grille). Sa valeur est de supprimer le réglage du chemin, ce qui reste à démontrer sur STL-10 où les calendriers fixes n'étaient pas classables (EXP-015). Trois graines, test déjà consulté, pas d'augmentation ni de filtre.

## Décision et suite

Recommandation de l'agent : retenir le signal g comme mesure de spécialisation d'échelle et la loi bornée comme calendrier par défaut sans table ; prochaine étape informative : STL-10 à temps égal avec g* = 0,06 contre les calendriers étendus d'EXP-015 ; chercher une mesure sans étiquettes portant sur la fonction (pas sur BN). Aucune décision utilisateur enregistrée.

## Sources et manques

`results/kaggle_outputs/adaptive-phase3-20260917-142057/` (log, `pairing_verification.json`, 9 `metrics.json`/`summary.json` avec `controller_log`, `gap_probe` pour Δ ∈ {2, 4, 8} et `bn_shift`). Manque : commit du code ; pas de figure ; pas de contrôle « calendrier réalisé recodé en fixe » (phase 2 du plan §6).

<!-- REPO-EVIDENCE:BEGIN -->

## Traces dans le dépôt (généré — ne pas éditer à la main)

Généré par `tools/enrich_records.py` depuis [`data/repo_evidence.json`](../data/repo_evidence.json). Voir aussi la vue d'ensemble : [REPO_EVIDENCE](../REPO_EVIDENCE.md).

**État du code par rapport à ce run** : controleur 'gap', sondes gap_probe et bn_shift ajoutes a job_adaptive_phase0.py ; non commite.

| Rôle | Chemin dans le dépôt | Dernier commit touchant ce fichier |
|---|---|---|
| Implémentation | [`continuation/probe_signals.py`](../../../../continuation/probe_signals.py) | df89fbf 2026-09-18 |
| Implémentation | [`scripts/job_adaptive_phase0.py`](../../../../scripts/job_adaptive_phase0.py) | df89fbf 2026-09-18 |
| Implémentation | [`scripts/job_adaptive_phase3.py`](../../../../scripts/job_adaptive_phase3.py) | df89fbf 2026-09-18 |
| Sorties d'exécution | [`results/kaggle_outputs/adaptive-phase3-20260917-142057`](../../../../results/kaggle_outputs/adaptive-phase3-20260917-142057) | 128 fichiers |
| Documentation du dépôt | [`docs/adaptive_resolution_plan.md`](../../../../docs/adaptive_resolution_plan.md) | df89fbf 2026-09-18 |

Un chemin listé existe dans le dépôt au moment de la génération. Cela ne prouve pas seul quelle version du code a exécuté ce run : pour les runs Kaggle, le code réellement expédié est archivé dans le sous-dossier `_repo/` de la sortie correspondante.

<!-- REPO-EVIDENCE:END -->
