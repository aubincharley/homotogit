---
id: EXP-013
schema_version: 1
updated_at: 2026-09-17
status: completed_numeric_verified
evidence: local_per_epoch_json_and_kernel_log
protocol: P-FULL-R20-BN
source_json: ../../../../results/kaggle_outputs/adaptive-phase1-20260917-075615/adaptive_phase1_20260917-075626/study_summary.json
code_commit_reported: null
code_commit_inspected: false
code_commit_inspected_note: code de la branche adaptative-resolution, non commité à la rédaction
---

# EXP-013 — Calendriers de résolution à pas fins, trois graines, avec et sans Gaussian ; efficacité de transfert

[Index](INDEX.md) · [EXP-012](EXP-012_adaptive_phase0.md) · [Plan, sections 14–15 (anglais)](../../../adaptive_resolution_plan.md) · [Analyse](../../../../results/adaptive_phase0_analysis.md)

## Métadonnées

| Champ | Valeur |
|---|---|
| ID campagne | `adaptive-phase1-20260917-075615` (kernel Kaggle, compte martingraffin) |
| Date exécution | 2026-09-17, 07:56 → 09:30 UTC (5 653 s) |
| Question utilisateur | « Réaliser les propositions » de la section 13 du plan |
| Profil | `P-FULL-R20-BN` inchangé |
| Code | `scripts/job_adaptive_phase1.py` → `scripts/job_adaptive_phase0.py` (`ADAPT_PHASE=1`), `continuation/probe_signals.py` |
| Environnement | torch 2.10.0+cu128, 2× Tesla T4 |
| Vérification | JSON par époque locaux ; contrastes recalculés ; huit empreintes d'assets vérifiées ; pureté des mesures vérifiée à chaque époque |

## Question et hypothèses

EXP-012 avait vu, sur une graine, deux calendriers à pas fins (`Rsteps4` 16/20/24/28 × 3 époques ; `Rlin12` une taille par époque) dépasser `Rprog` de plus d'un point. Cette campagne les confirme sur trois graines appariées, les combine au filtre Gaussian paliers du gagnant d'EXP-011, ajoute les graines 1/2 de `Rmixed`, et mesure sur chaque run l'efficacité de transfert τ_Δ = ⟨g_r, g_{r+Δ}⟩/‖g_{r+Δ}‖² pour Δ ∈ {1, 2, 4, 8} avec des gradients **en mode train** (BN par lot), pour rejouer un contrôleur de taille de pas.

## Bras

Sans filtre : `Rsteps4`, `Rlin12` (graines 1/2 ; graine 0 issue d'EXP-012), `Rmixed` (graines 1/2). Avec `Gplateau` (19 sites, σ_l = q_l·g(e)) : `Rprog`, `Rsteps4`, `Rlin12`, graines 0/1/2. Réduction bilinéaire d'entrée partout.

## Résultats numériques

Accuracy test finale (%), graines 0/1/2 → moyenne : Rprog+G 80,03/80,14/81,28 → **80,48** (EXP-011 : 80,15/80,57/81,42) ; Rsteps4+G 79,64/80,31/81,32 → 80,42 ; Rlin12+G 79,73/80,11/80,71 → 80,18 ; **Rlin12** 79,70/79,97/80,74 → **80,14** ; **Rsteps4** 79,80/79,54/80,71 → **80,02** ; Rmixed 79,16/79,66/80,09 → 79,64 ; Rprog 78,63/78,91/79,87 → 79,14.

Contrastes appariés par graine (points) : Rlin12 − Rprog **+1,07 / +1,06 / +0,87** ; Rsteps4 − Rprog **+1,17 / +0,63 / +0,84** ; Rmixed − Rprog +0,53 / +0,75 / +0,22 ; Rsteps4+G − Rprog+G −0,39 / +0,17 / +0,04 ; Rlin12+G − Rprog+G −0,30 / −0,03 / −0,57. Gain propre du Gaussian : +1,35 sur Rprog, +0,41 sur Rsteps4, +0,05 sur Rlin12. Temps d'entraînement moyen : 451–463 s sans filtre, 667–676 s avec.

τ en mode train (Rprog+G, graine 0, palier 16, époques 0…6) : Δ=+2 : 1,00 → 0,49 ; Δ=+4 : 0,91 → 0,38 ; Δ=+8 (saut réel) : 0,87, 0,50, 0,30, 0,27, 0,21, 0,16, 0,17. Le long de Rsteps4 (Δ=4) et Rlin12 (Δ=1–2), τ du pas effectivement pris reste entre 0,6 et 0,9. Pas vers une taille impaire moins alignés (depuis 16, époque 3 : +1→17 : 0,36 ; +2→18 : 0,57 ; +4→20 : 0,75).

## Coût

15 runs ; 549–592 s sans filtre (89–104 s de mesures), 816–843 s avec (151–168 s) ; 5 653 s écoulées sur deux T4, ≈3,1 h GPU.

## Interprétation et limites

Les pas fins valent environ **+1 point** sur `Rprog` sans filtre, positifs sur chaque graine, à budget d'updates et temps égaux. Le filtre Gaussian et les pas fins sont **substituables** : l'un absorbe l'autre. Les rampes non filtrées atteignent 80,0–80,1 % pour 68 % du temps du meilleur bras filtré : nouveau candidat pour la frontière précision/temps, devant `stem_max` sans Gaussian (79,97 %, EXP-011). Le signal τ en mode train est propre et décroît avec la durée de séjour pour les grands sauts ; les cosinus en mode eval d'EXP-012 étaient contaminés par BN. La relecture « plus grand Δ tel que τ ≥ θ » reconstruirait des sauts à la Rprog : τ élevé signifie que rester grossier ne coûte rien, τ bas que le palier est épuisé pour la cible ; le contrôleur est reformulé en conséquence (plan §14.3). Trois graines, test déjà consulté : exploratoire situé. Graine 0 des bras non filtrés issue d'un autre kernel (mêmes assets).

## Décision et suite

Recommandation de l'agent : retenir `Rlin12` / `Rsteps4` sans filtre comme candidats précision/temps ; contrôleur vivant de taille de pas et rampe à tailles paires lancés en phase 2 (`adaptive-phase2-*`, plan §15) ; aucune décision utilisateur enregistrée.

## Sources et manques

`results/kaggle_outputs/adaptive-phase1-20260917-075615/` (log, `pairing_verification.json`, 15 `metrics.json`/`summary.json`) ; `results/adaptive_phase0_analysis.{json,md}` (analyse conjointe EXP-012/013). Manque : commit du code ; pas de figure ; pas d'augmentation ; pas de STL-10.

<!-- REPO-EVIDENCE:BEGIN -->

## Traces dans le dépôt (généré — ne pas éditer à la main)

Généré par `tools/enrich_records.py` depuis [`data/repo_evidence.json`](../data/repo_evidence.json). Voir aussi la vue d'ensemble : [REPO_EVIDENCE](../REPO_EVIDENCE.md).

**État du code par rapport à ce run** : meme code que EXP-012 etendu (operateur Gaussian, tau multi-pas) ; non commite a la redaction de la fiche.

| Rôle | Chemin dans le dépôt | Dernier commit touchant ce fichier |
|---|---|---|
| Implémentation | [`continuation/probe_signals.py`](../../../../continuation/probe_signals.py) | df89fbf 2026-09-18 |
| Implémentation | [`scripts/job_adaptive_phase0.py`](../../../../scripts/job_adaptive_phase0.py) | df89fbf 2026-09-18 |
| Implémentation | [`scripts/job_adaptive_phase1.py`](../../../../scripts/job_adaptive_phase1.py) | df89fbf 2026-09-18 |
| Implémentation | [`scripts/analyze_adaptive_phase0.py`](../../../../scripts/analyze_adaptive_phase0.py) | df89fbf 2026-09-18 |
| Sorties d'exécution | [`results/kaggle_outputs/adaptive-phase1-20260917-075615`](../../../../results/kaggle_outputs/adaptive-phase1-20260917-075615) | 135 fichiers |
| Sorties d'exécution | [`results/adaptive_phase0_analysis.json`](../../../../results/adaptive_phase0_analysis.json) | df89fbf 2026-09-18 |
| Documentation du dépôt | [`docs/adaptive_resolution_plan.md`](../../../../docs/adaptive_resolution_plan.md) | df89fbf 2026-09-18 |

Un chemin listé existe dans le dépôt au moment de la génération. Cela ne prouve pas seul quelle version du code a exécuté ce run : pour les runs Kaggle, le code réellement expédié est archivé dans le sous-dossier `_repo/` de la sortie correspondante.

<!-- REPO-EVIDENCE:END -->
