---
id: EXP-012
schema_version: 1
updated_at: 2026-09-17
status: completed_numeric_verified
evidence: local_per_epoch_json_and_kernel_log
protocol: P-FULL-R20-BN
source_json: ../../../../results/kaggle_outputs/adaptive-phase0-20260916-185116/adaptive_phase0_20260916-185130/study_summary.json
code_commit_reported: null
code_commit_inspected: false
code_commit_inspected_note: code écrit sur la branche adaptative-resolution le 16 septembre 2026, non commité au moment de la rédaction
---

# EXP-012 — Résolution adaptative, phase 0 : signaux mesurés sur calendriers fixes et grille de frontières

[Index](INDEX.md) · [Plan et verdict (anglais)](../../../adaptive_resolution_plan.md) · [Analyse](../../../../results/adaptive_phase0_analysis.md) · [Lecture actuelle](../CURRENT_STATE.md)

## Métadonnées

| Champ | Valeur |
|---|---|
| ID campagne d'origine | `adaptive-phase0-20260916-185116` (kernel Kaggle, compte martingraffin) |
| Date exécution | 2026-09-16, 18:51 → 20:05 UTC (4 394 s) |
| Question utilisateur | « Analyser les résultats de la branche et proposer de nouvelles façons de faire fonctionner la résolution adaptative » |
| Profil | `P-FULL-R20-BN` inchangé : 50k/10k, 30 époques, 11 730 updates, batch 32×4, LR 0,005, warmup 60, cosine, sans augmentation |
| Code | `scripts/job_adaptive_phase0.py`, `continuation/probe_signals.py`, `scripts/analyze_adaptive_phase0.py` |
| Environnement | torch 2.10.0+cu128, 2× Tesla T4 |
| Vérification | JSON par époque locaux ; agrégats recalculés par le script d'analyse ; code exécuté = code de la branche expédié dans `_repo/` |

## Question et hypothèses

Le plan de la branche proposait d'avancer la résolution quand la norme du gradient sur la sonde atteint un plateau. Rien n'avait été mesuré. Phase 0 mesure, **sans contrôleur**, à la fin de chaque époque et sans perturber l'entraînement : norme du gradient complet de l'objectif courant (BN en mode eval et en mode train), gradient et CE à la résolution suivante et à la cible 32×32, les mêmes CE après recalibration des statistiques BN à poids fixés, cosinus entre gradient courant et gradients suivant/cible, test courant et cible. La partie B ajoute une grille de calendriers fixes (une graine) pour savoir si la place des frontières importe.

Conclusion non permise : aucune comparaison de la partie B n'a trois graines ; aucun bras ne porte le Gaussian interne.

## Bras et protocole figé

Partie A, appariée à EXP-011 : `R32__Gnone__input_bilinear` et `Rprog__Gnone__input_bilinear`, graines 0/1/2. Partie B, graine 0, `Gnone`, réduction bilinéaire d'entrée : frontières (3,9), (3,12), (6,9), (6,18), (9,15), (9,18), (12,18) ; `Rlin12` = 16,17,19,20,21,23,24,25,27,28,29,31 puis 32 ; `Rsteps4` = 16/20/24/28 sur 3 époques chacune puis 32 ; `Rmixed` = résolution tirée par update dans {16,24,32} avec probabilités (0,6/0,3/0,1) époques 0–5, (0,25/0,5/0,25) 6–11, (0,1/0,3/0,6) 12–17, puis 32 (générateur `resolution_mix`, graine 0). Ensemble de mesure : 2 000 images d'entraînement équilibrées (`fixed_subset_indices`, flux `adaptive_monitor`), distinct de la sonde de 500.

## Appariement et réutilisation

État partagé régénéré dans le kernel ; les huit empreintes sha256 (subset, sonde, trois permutations, trois initialisations) **correspondent** aux assets épinglés de la campagne (`pairing_verification.json`). Chaque mesure vérifie ensuite que paramètres, tampons BN et mode sont restaurés à l'identique ; 16 × 31 vérifications positives.

## Résultats numériques

Réplication des six cellules appariées (accuracy test finale, %) : R32 74,76 / 76,01 / 76,40 contre 75,13 / 75,62 / 76,10 dans EXP-011 ; Rprog 78,63 / 78,91 / 79,87 contre 79,20 / 79,10 / 80,10. Écarts −0,57 à +0,39 point, du même ordre que les re-exécutions internes d'EXP-011.

**Norme du gradient.** En mode eval, valeurs entre 1,5 et 14 variant d'un facteur 2–3 d'une époque à l'autre, sans structure par palier ; la règle go/no-go du plan donne trois verdicts différents sur les trois graines du même palier. En mode train, **1,0 ± 0,2 de l'époque 1 à l'époque 20** dans tous les runs et à toutes les résolutions, puis 0,83 en fin de cosine. Aucune décroissance intra-palier, aucun plateau, aucune différence entre 16, 24 et 32.

**Q-04.** Fin du palier 16 (époque 6, graines 0/1) : chemin courant 56,5 / 59,2 ; chemin cible 30,4 / 40,6 ; cible après recalibration BN 43,9 / 47,9. Fin du palier 24 (époque 12) : 68,8 / 70,5 ; 65,6 / 67,2 ; 68,2 / 69,8.

**Alignement des gradients** (efficacité de transfert τ = ⟨g_r, g_r'⟩/‖g_r'‖², mode eval) : saut 16→24 : 0,53 → ≈0,1 dès l'époque 4 ; saut 24→32 : 0,35–0,48 ; pas de 4 (`Rsteps4`) : 0,3–0,9 ; pas de 1–2 (`Rlin12`) : 0,6–1,0.

**Grille (graine 0, %)** : Rsteps4 79,80 ; Rlin12 79,70 ; Rmixed 79,16 ; Rb9_15 79,05 ; Rb6_9 78,96 ; Rb6_18 78,96 ; Rb3_12 78,84 ; Rprog 78,63 ; Rb9_18 78,60 ; Rb3_9 78,20 ; Rb12_18 77,46 ; R32 74,76. Écart test − train CE final : 0,53 pour R32, 0,14–0,36 pour tous les calendriers progressifs.

## Coût

16 runs, 515–592 s chacun dont 72–122 s de mesures (13–21 %), 4 394 s écoulées sur deux T4, ≈2,4 h GPU cumulées, pic mémoire ≈0,5 GiB.

## Interprétation et limites

Le critère de plateau sur la norme du gradient n'a pas d'objet : le résidu est stationnaire à ce LR. Le plan de la branche est **clos négativement** sur ce point, pour 2,4 h GPU. La recalibration BN explique la moitié de l'effondrement du chemin cible à r=16 et sa quasi-totalité à r=24. La place des deux frontières est presque indifférente (six alternatives sur sept à ±0,45 point de Rprog) ; la **finesse des pas** ne l'est pas (+1,1 / +1,2 point sur la même graine), ce que l'alignement des gradients prédit. Une graine, test déjà consulté : exploratoire.

## Décision et suite

Recommandation de l'agent, pas décision utilisateur : abandonner le déclencheur de plateau ; confirmer `Rsteps4` et `Rlin12` sur trois graines appariées, puis avec Gaussian ; si confirmé, construire un contrôleur de **taille de pas** sur τ (gradients en mode train) plutôt qu'un contrôleur de frontière ; recalibrer BN à chaque transition. Détail dans le plan, sections 12 et 13.

## Sources et manques

`results/kaggle_outputs/adaptive-phase0-20260916-185116/` (log du kernel, `environment.json`, `pairing_verification.json`, 16 `metrics.json` et `summary.json`, `_repo/` non versionné) ; `results/adaptive_phase0_analysis.{json,md}`. Manque : commit du code (branche non commitée à la rédaction) ; les cosinus utilisent des gradients en mode eval, bruités par BN ; pas de figure.

<!-- REPO-EVIDENCE:BEGIN -->

## Traces dans le dépôt (généré — ne pas éditer à la main)

Généré par `tools/enrich_records.py` depuis [`data/repo_evidence.json`](../data/repo_evidence.json). Voir aussi la vue d'ensemble : [REPO_EVIDENCE](../REPO_EVIDENCE.md).

**État du code par rapport à ce run** : code ecrit pour ce run sur la branche adaptative-resolution ; non commite a la redaction de la fiche.

| Rôle | Chemin dans le dépôt | Dernier commit touchant ce fichier |
|---|---|---|
| Implémentation | [`continuation/probe_signals.py`](../../../../continuation/probe_signals.py) | — |
| Implémentation | [`scripts/job_adaptive_phase0.py`](../../../../scripts/job_adaptive_phase0.py) | — |
| Implémentation | [`scripts/analyze_adaptive_phase0.py`](../../../../scripts/analyze_adaptive_phase0.py) | — |
| Implémentation | [`scripts/kaggle_run.py`](../../../../scripts/kaggle_run.py) | — |
| Sorties d'exécution | [`results/kaggle_outputs/adaptive-phase0-20260916-185116`](../../../../results/kaggle_outputs/adaptive-phase0-20260916-185116) | 134 fichiers |
| Sorties d'exécution | [`results/adaptive_phase0_analysis.json`](../../../../results/adaptive_phase0_analysis.json) | — |
| Sorties d'exécution | [`results/adaptive_phase0_analysis.md`](../../../../results/adaptive_phase0_analysis.md) | — |
| Documentation du dépôt | [`docs/adaptive_resolution_plan.md`](../../../../docs/adaptive_resolution_plan.md) | — |

Un chemin listé existe dans le dépôt au moment de la génération. Cela ne prouve pas seul quelle version du code a exécuté ce run : pour les runs Kaggle, le code réellement expédié est archivé dans le sous-dossier `_repo/` de la sortie correspondante.

<!-- REPO-EVIDENCE:END -->
