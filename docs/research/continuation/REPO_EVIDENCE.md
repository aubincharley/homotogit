---
id: DOC-REPO-EVIDENCE
schema_version: 1
updated_at: 2026-09-09
status: generated
---

# Traces de preuve dans le dépôt (généré)

[Accueil](README.md) · [Index des expériences](experiments/INDEX.md) · [JSON](data/repo_evidence.json)

Généré par `tools/link_repo_evidence.py`. **Ne pas éditer à la main.**
Un chemin présent signifie que le fichier existe aujourd'hui ; il ne
prouve pas à lui seul quelle version de ce code a exécuté un run
ancien. La colonne « état du code » précise la portée.

Dépôt : `adaptative-resolution`, branche `adaptative-resolution`, HEAD `df89fbf`.

## EXP-000

**État du code** : gaussian.py a été étendu depuis (repli de réflexion explicite, phase 8) ; le chemin input-space de exp0 n'est pas affecté par ce repli, qui ne sert qu'aux cartes 4x4.

| Rôle | Chemin | Présent | Dernier commit |
|---|---|---|---|
| code | `continuation/experiments/exp0.py` | oui | 86c00bc 2026-09-08 |
| code | `continuation/engine.py` | oui | 86c00bc 2026-09-08 |
| code | `continuation/transforms/gaussian.py` | oui | d643f75 2026-09-08 |
| config | `configs/exp0_gaussian.yaml` | oui | 86c00bc 2026-09-08 |
| results | `results/exp0_gaussian` | oui | 73 fichiers |
| docs | `docs/experiment0.md` | oui | 86c00bc 2026-09-08 |
| docs | `docs/results.md` | oui | 86c00bc 2026-09-08 |

## EXP-001

**État du code** : branchement full_state_v1 ; code inchangé depuis.

| Rôle | Chemin | Présent | Dernier commit |
|---|---|---|---|
| code | `continuation/experiments/exp1.py` | oui | 86c00bc 2026-09-08 |
| code | `continuation/engine.py` | oui | 86c00bc 2026-09-08 |
| config | `configs/exp1_warmstart.yaml` | oui | 86c00bc 2026-09-08 |
| results | `results/exp1_gaussian_warmstart` | oui | 43 fichiers |
| docs | `docs/exp1_warmstart.md` | oui | 86c00bc 2026-09-08 |

## EXP-002

**État du code** : tv.py corrigé pendant EXP-002 (signe de l'adjoint D^T) ; les apercus livres ont ete produits apres correction.

| Rôle | Chemin | Présent | Dernier commit |
|---|---|---|---|
| code | `continuation/transforms/tv.py` | oui | 86c00bc 2026-09-08 |
| code | `scripts/tv_previews.py` | oui | 86c00bc 2026-09-08 |
| results | `results/tv_previews` | oui | 3 fichiers |
| docs | `docs/tv_budget.md` | oui | 86c00bc 2026-09-08 |

## EXP-003

**État du code** : chemin 'fast' fusionne et bypass s=1 ajoutes apres les premiers apercus ; les deux chemins calculent la meme operation (verifie par les tests).

| Rôle | Chemin | Présent | Dernier commit |
|---|---|---|---|
| code | `continuation/transforms/wavelet.py` | oui | d643f75 2026-09-08 |
| code | `scripts/wavelet_previews.py` | oui | d643f75 2026-09-08 |
| code | `scripts/wavelet_benchmark.py` | oui | d643f75 2026-09-08 |
| code | `scripts/wavelet_profile.py` | oui | d643f75 2026-09-08 |
| code | `scripts/wavelet_optimized_benchmark.py` | oui | d643f75 2026-09-08 |
| results | `results/wavelet_previews` | oui | 14 fichiers |
| docs | `docs/wavelet_shrinkage.md` | oui | d643f75 2026-09-08 |

## EXP-004

**État du code** : pilote ancien ; sorties partiellement telechargees, summary.json absent pour plusieurs cellules.

| Rôle | Chemin | Présent | Dernier commit |
|---|---|---|---|
| code | `scripts/kaggle_pilot_continuation.py` | oui | d643f75 2026-09-08 |
| code | `scripts/kaggle_run.py` | oui | df89fbf 2026-09-18 |
| results | `results/kaggle_outputs/pilot-continuation-20260908-103434` | oui | 3 fichiers |
| results | `results/kaggle_outputs/pilot-continuation-20260908-122558` | oui | 10 fichiers |
| docs | `docs/kaggle_cli.md` | oui | 0160bd1 2026-09-09 |

## EXP-005

**État du code** : continuation_driver.py a beaucoup evolue depuis (tables par paliers, chemins courant/cible, resolution). Le code expedie a ce run est conserve dans le _repo/ de chaque sortie Kaggle, pas dans l'arbre courant.

| Rôle | Chemin | Présent | Dernier commit |
|---|---|---|---|
| code | `scripts/continuation_driver.py` | oui | 26cfb82 2026-09-09 |
| code | `scripts/job_plain_study.py` | oui | d643f75 2026-09-08 |
| code | `scripts/job_gaussian_study.py` | oui | d643f75 2026-09-08 |
| code | `scripts/job_lr_diagnostic.py` | oui | d643f75 2026-09-08 |
| code | `scripts/job_lr_control_002.py` | oui | d643f75 2026-09-08 |
| code | `scripts/audit_gaussian_placement.py` | oui | d643f75 2026-09-08 |
| code | `scripts/verify_grad_accumulation.py` | oui | d643f75 2026-09-08 |
| code | `continuation/models/resnet_gn.py` | oui | 86c00bc 2026-09-08 |
| config | `scripts/_study_common.py` | oui | d643f75 2026-09-08 |
| results | `results/kaggle_outputs/plain-study-20260908-132611` | oui | 10 fichiers |
| results | `results/kaggle_outputs/gaussian-study-20260908-132957` | oui | 10 fichiers |
| results | `results/kaggle_outputs/lr-diagnostic-20260908-132240` | oui | 8 fichiers |
| results | `results/kaggle_outputs/lr-control-002-20260908-140604` | oui | 8 fichiers |
| results | `results/gaussian_placement_audit.json` | oui | d643f75 2026-09-08 |
| results | `results/grad_accumulation_check.json` | oui | d643f75 2026-09-08 |
| results | `results/plain_vs_gaussian_paired.json` | oui | d643f75 2026-09-08 |

## EXP-006

**État du code** : le repli de reflexion explicite pour cartes 4x4 a ete ajoute pour ce run ; voir gaussian.py.

| Rôle | Chemin | Présent | Dernier commit |
|---|---|---|---|
| code | `continuation/models/resnet18_bn.py` | oui | d643f75 2026-09-08 |
| code | `scripts/job_resnet18_gaussian.py` | oui | d643f75 2026-09-08 |
| code | `scripts/continuation_driver.py` | oui | 26cfb82 2026-09-09 |
| results | `results/kaggle_outputs/resnet18-gaussian-20260908-144640` | oui | 5 fichiers |
| results | `results/resnet18_plain_vs_gaussian.png` | oui | d643f75 2026-09-08 |

## EXP-007

**État du code** : modele inchange depuis ; driver modifie apres coup.

| Rôle | Chemin | Présent | Dernier commit |
|---|---|---|---|
| code | `continuation/models/resnet20_bn.py` | oui | d643f75 2026-09-08 |
| code | `scripts/job_resnet20bn_gaussian.py` | oui | d643f75 2026-09-08 |
| code | `scripts/continuation_driver.py` | oui | 26cfb82 2026-09-09 |
| results | `results/kaggle_outputs/resnet20bn-gaussian-20260908-154226` | oui | 8 fichiers |
| results | `results/resnet20bn_plain_vs_gaussian.png` | oui | d643f75 2026-09-08 |
| results | `results/resnet20bn_pilot_corrected.png` | oui | d643f75 2026-09-08 |

## EXP-008

**État du code** : assets partages de ce run (init_seed*.pt, shared_indices.npz) reutilises comme etat epingle d'EXP-010 et EXP-011.

| Rôle | Chemin | Présent | Dernier commit |
|---|---|---|---|
| code | `scripts/job_fulldata_campaign.py` | oui | d643f75 2026-09-08 |
| code | `scripts/continuation_driver.py` | oui | 26cfb82 2026-09-09 |
| results | `results/kaggle_outputs/fulldata-r20bn-20260908-161221` | oui | 22 fichiers |
| results | `results/fulldata_campaign.png` | oui | d643f75 2026-09-08 |
| docs | `docs/HANDOVER.md` | oui | 8112db0 2026-09-09 |

## EXP-009

**État du code** : table par paliers de s ajoutee au driver pour ce run ; is_active traite s=1 comme inactif.

| Rôle | Chemin | Présent | Dernier commit |
|---|---|---|---|
| code | `scripts/job_db2_pilot.py` | oui | 26cfb82 2026-09-09 |
| code | `scripts/verify_db2_operator.py` | oui | 26cfb82 2026-09-09 |
| code | `continuation/transforms/wavelet.py` | oui | d643f75 2026-09-08 |
| code | `scripts/continuation_driver.py` | oui | 26cfb82 2026-09-09 |
| results | `results/kaggle_outputs/db2-pilot-r20bn-20260908-191741` | oui | 9 fichiers |
| results | `results/db2_operator_verification.json` | oui | 26cfb82 2026-09-09 |
| results | `results/db2_pilot.png` | oui | 26cfb82 2026-09-09 |

## EXP-010

**État du code** : le redimensionnement vit dans InputPipeline (uint8/255 -> float -> resize -> T_eta -> normalisation) ; la verification d'appariement par empreintes est dans le job.

| Rôle | Chemin | Présent | Dernier commit |
|---|---|---|---|
| code | `scripts/job_progressive_resolution.py` | oui | 26cfb82 2026-09-09 |
| code | `scripts/verify_progressive_resolution.py` | oui | 26cfb82 2026-09-09 |
| code | `scripts/plot_progressive_resolution.py` | oui | 26cfb82 2026-09-09 |
| code | `continuation/pipeline.py` | oui | 26cfb82 2026-09-09 |
| code | `scripts/continuation_driver.py` | oui | 26cfb82 2026-09-09 |
| results | `results/kaggle_outputs/progres-r20bn-20260909-075846` | oui | 10 fichiers |
| results | `results/progressive_resolution.png` | oui | 26cfb82 2026-09-09 |
| results | `results/progressive_resolution_verification.json` | oui | 26cfb82 2026-09-09 |

## EXP-011

**État du code** : code de la campagne ecrit pour ce run et inchange depuis ; commit f191fa6 contient les resultats.

| Rôle | Chemin | Présent | Dernier commit |
|---|---|---|---|
| code | `continuation/campaign_ops.py` | oui | b9bb609 2026-09-09 |
| code | `scripts/campaign_driver.py` | oui | b9bb609 2026-09-09 |
| code | `scripts/campaign_manifest.py` | oui | b9bb609 2026-09-09 |
| code | `scripts/job_campaign.py` | oui | b9bb609 2026-09-09 |
| code | `scripts/job_campaign_0.py` | oui | b9bb609 2026-09-09 |
| code | `scripts/job_campaign_1.py` | oui | b9bb609 2026-09-09 |
| code | `scripts/job_campaign_2.py` | oui | b9bb609 2026-09-09 |
| code | `scripts/verify_campaign_ops.py` | oui | b9bb609 2026-09-09 |
| code | `scripts/stage_campaign_assets.py` | oui | b9bb609 2026-09-09 |
| code | `scripts/analyze_campaign.py` | oui | f191fa6 2026-09-09 |
| code | `scripts/make_presentation.py` | oui | 9da41d6 2026-09-09 |
| code | `scripts/presentation_labels.py` | oui | 9da41d6 2026-09-09 |
| config | `results/campaign_manifest_frozen.json` | oui | b9bb609 2026-09-09 |
| results | `results/kaggle_outputs/campaign-j0-20260909-095039` | oui | 49 fichiers |
| results | `results/kaggle_outputs/campaign-j1-20260909-095101` | oui | 49 fichiers |
| results | `results/kaggle_outputs/campaign-j2-20260909-095123` | oui | 49 fichiers |
| results | `results/campaign_results.json` | oui | f191fa6 2026-09-09 |
| results | `results/campaign_verification.json` | oui | b9bb609 2026-09-09 |
| results | `results/campaign_previews` | oui | 2 fichiers |
| results | `results/presentation` | oui | 43 fichiers |
| docs | `docs/HANDOVER.md` | oui | 8112db0 2026-09-09 |

## EXP-012

**État du code** : code ecrit pour ce run sur la branche adaptative-resolution ; non commite a la redaction de la fiche.

| Rôle | Chemin | Présent | Dernier commit |
|---|---|---|---|
| code | `continuation/probe_signals.py` | oui | df89fbf 2026-09-18 |
| code | `scripts/job_adaptive_phase0.py` | oui | df89fbf 2026-09-18 |
| code | `scripts/analyze_adaptive_phase0.py` | oui | df89fbf 2026-09-18 |
| code | `scripts/kaggle_run.py` | oui | df89fbf 2026-09-18 |
| results | `results/kaggle_outputs/adaptive-phase0-20260916-185116` | oui | 134 fichiers |
| results | `results/adaptive_phase0_analysis.json` | oui | df89fbf 2026-09-18 |
| results | `results/adaptive_phase0_analysis.md` | oui | df89fbf 2026-09-18 |
| docs | `docs/adaptive_resolution_plan.md` | oui | df89fbf 2026-09-18 |

## EXP-013

**État du code** : meme code que EXP-012 etendu (operateur Gaussian, tau multi-pas) ; non commite a la redaction de la fiche.

| Rôle | Chemin | Présent | Dernier commit |
|---|---|---|---|
| code | `continuation/probe_signals.py` | oui | df89fbf 2026-09-18 |
| code | `scripts/job_adaptive_phase0.py` | oui | df89fbf 2026-09-18 |
| code | `scripts/job_adaptive_phase1.py` | oui | df89fbf 2026-09-18 |
| code | `scripts/analyze_adaptive_phase0.py` | oui | df89fbf 2026-09-18 |
| results | `results/kaggle_outputs/adaptive-phase1-20260917-075615` | oui | 135 fichiers |
| results | `results/adaptive_phase0_analysis.json` | oui | df89fbf 2026-09-18 |
| docs | `docs/adaptive_resolution_plan.md` | oui | df89fbf 2026-09-18 |

## EXP-014

**État du code** : controleur ajoute a job_adaptive_phase0.py ; bogue de copie de calendrier corrige entre les deux kernels ; non commite.

| Rôle | Chemin | Présent | Dernier commit |
|---|---|---|---|
| code | `continuation/probe_signals.py` | oui | df89fbf 2026-09-18 |
| code | `scripts/job_adaptive_phase0.py` | oui | df89fbf 2026-09-18 |
| code | `scripts/job_adaptive_phase2.py` | oui | df89fbf 2026-09-18 |
| code | `scripts/analyze_adaptive_phase0.py` | oui | df89fbf 2026-09-18 |
| results | `results/kaggle_outputs/adaptive-phase2-20260917-093614` | oui | 130 fichiers |
| results | `results/kaggle_outputs/adaptive-phase2b-20260917-113110` | oui | 118 fichiers |
| results | `results/adaptive_phase0_analysis.json` | oui | df89fbf 2026-09-18 |
| docs | `docs/adaptive_resolution_plan.md` | oui | df89fbf 2026-09-18 |

## EXP-015

**État du code** : job STL-10 ecrit pour ce run ; controles ajoutes entre les deux kernels ; non commite.

| Rôle | Chemin | Présent | Dernier commit |
|---|---|---|---|
| code | `scripts/job_stl10_resolution.py` | oui | df89fbf 2026-09-18 |
| code | `scripts/job_stl10_controls.py` | oui | df89fbf 2026-09-18 |
| code | `scripts/analyze_stl10_resolution.py` | oui | df89fbf 2026-09-18 |
| code | `continuation/probe_signals.py` | oui | df89fbf 2026-09-18 |
| results | `results/kaggle_outputs/stl10-resolution-20260917-121258` | oui | 133 fichiers |
| results | `results/kaggle_outputs/stl10-controls-20260917-133339` | oui | 117 fichiers |
| results | `results/stl10_resolution_analysis.md` | oui | df89fbf 2026-09-18 |
| docs | `docs/adaptive_resolution_plan.md` | oui | df89fbf 2026-09-18 |

## EXP-016

**État du code** : controleur 'gap', sondes gap_probe et bn_shift ajoutes a job_adaptive_phase0.py ; non commite.

| Rôle | Chemin | Présent | Dernier commit |
|---|---|---|---|
| code | `continuation/probe_signals.py` | oui | df89fbf 2026-09-18 |
| code | `scripts/job_adaptive_phase0.py` | oui | df89fbf 2026-09-18 |
| code | `scripts/job_adaptive_phase3.py` | oui | df89fbf 2026-09-18 |
| results | `results/kaggle_outputs/adaptive-phase3-20260917-142057` | oui | 128 fichiers |
| docs | `docs/adaptive_resolution_plan.md` | oui | df89fbf 2026-09-18 |

## EXP-017

**État du code** : controleur 'gap2s' (ascension fixe ou adaptative + rechauffes) ; graines configurables ; non commite.

| Rôle | Chemin | Présent | Dernier commit |
|---|---|---|---|
| code | `continuation/probe_signals.py` | oui | df89fbf 2026-09-18 |
| code | `scripts/job_adaptive_phase0.py` | oui | df89fbf 2026-09-18 |
| code | `scripts/job_adaptive_phase4.py` | oui | df89fbf 2026-09-18 |
| code | `scripts/job_adaptive_phase4c.py` | oui | df89fbf 2026-09-18 |
| code | `scripts/job_adaptive_phase5.py` | oui | df89fbf 2026-09-18 |
| results | `results/kaggle_outputs/adaptive-phase4-20260917-151733` | oui | 135 fichiers |
| results | `results/kaggle_outputs/adaptive-phase4b-20260917-154529` | oui | 123 fichiers |
| results | `results/kaggle_outputs/adaptive-phase4c-20260917-165354` | oui | 124 fichiers |
| results | `results/kaggle_outputs/adaptive-phase5-20260918-055940` | oui | 146 fichiers |
| docs | `docs/adaptive_resolution_plan.md` | oui | df89fbf 2026-09-18 |

## EXP-018

**État du code** : controleurs (rechauffes, montee conjointe) ajoutes au job STL-10 apres df89fbf.

| Rôle | Chemin | Présent | Dernier commit |
|---|---|---|---|
| code | `scripts/job_stl10_resolution.py` | oui | df89fbf 2026-09-18 |
| code | `scripts/job_stl10_adaptive.py` | oui | — |
| code | `continuation/probe_signals.py` | oui | df89fbf 2026-09-18 |
| results | `results/kaggle_outputs/stl10-adaptive-20260918-115842` | oui | 149 fichiers |
| docs | `docs/adaptive_resolution_plan.md` | oui | df89fbf 2026-09-18 |

## Recoupement numérique EXP-011

Instantané contre copie du dépôt de `results/campaign_results.json` : 21 configuration(s) comparée(s), 0 écart(s). Cellules complétées : instantané 63, dépôt 63.

Commit `f191fa6` cité par la fiche EXP-011 : résolu dans ce dépôt — f191fa6 2026-09-09 push.
