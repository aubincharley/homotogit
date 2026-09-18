---
id: EXP-017
schema_version: 1
updated_at: 2026-09-18
status: completed_numeric_verified
evidence: local_per_epoch_json_and_kernel_log
protocol: P-FULL-R20-BN
source_json: ../../../../results/kaggle_outputs/adaptive-phase5-20260918-055940/adaptive_phase5_20260918-055950/study_summary.json
code_commit_reported: null
code_commit_inspected: false
code_commit_inspected_note: code de la branche adaptative-resolution, non commité à la rédaction ; premier lancement des bras contrôleur (phase 4) invalidé par un bogue de dispatch
---

# EXP-017 — Spécialisation bornée des deux côtés : réchauffes adaptatives à 24×24 pendant la phase 32×32

[Index](INDEX.md) · [EXP-016](EXP-016_bounded_specialisation.md) · [Plan, section 18 (anglais)](../../../adaptive_resolution_plan.md)

## Métadonnées

| Champ | Valeur |
|---|---|
| ID campagnes | `adaptive-phase4-20260917-151733` (contrôle fixe valide ; 6 bras contrôleur invalides, `INVALID.json`), `adaptive-phase4b-20260917-154529` (g*_down 0,15 / 0,30), `adaptive-phase4c-20260917-165354` (0,45 / 0,60), `adaptive-phase5-20260918-055940` (six graines, ascension fixe) |
| Dates | 2026-09-17 15:17 → 2026-09-18 07:15 UTC |
| Question utilisateur | « Je veux qu'il batte les rampes fines fixes » |
| Profil | `P-FULL-R20-BN` inchangé, sans filtre ; horizon et LR fixes ; graines 0–2 appariées aux assets épinglés, 3–5 régénérées et appariées dans le job |
| Code | `scripts/job_adaptive_phase{4,4c,5}.py` → `job_adaptive_phase0.py`, `continuation/probe_signals.py` |
| Vérification | JSON par époque avec journal des décisions ; empreintes vérifiées ; garde résolution = calendrier ; pureté des mesures |

## Mécanisme et loi

Toutes les rampes fines (fixes ou pilotées) plafonnent à 80,0–80,1 % ; le degré de liberté restant est la **spécialisation à l'échelle fine pendant la phase 32×32** : le signal d'EXP-016 lu vers le bas, g(32→24) = [L_24^{recal BN} − L_32]/L_32, vaut ≈ 0 à l'arrivée à 32 et croît de ≈ 0,05/époque jusqu'à 0,4–0,7 alors que l'accuracy test stagne dès l'époque 24. Loi : quand la moyenne mobile de g(32→24) dépasse g*_down, une époque de « réchauffe » à 24×24, puis retour à 32 ; jamais dans les `settle` dernières époques. Contrôle fixe du mécanisme : `Rsteps4rh` = Rsteps4 + une époque 24 aux époques 20, 24, 28.

## Résultats numériques

Mécanisme (trois graines, phase 4) : `Rsteps4rh` 80,32 / 80,54 / 81,22 → 80,69 %, +0,52 / +1,00 / +0,51 sur Rsteps4, +0,62 / +0,57 / +0,48 sur Rlin12 ; chaque réchauffe ramène g de 0,6–0,7 à ≈ −0,1…−0,3 et l'époque 32 suivante dépasse l'accuracy d'avant réchauffe.

Balayage du seuil, ascension adaptative (g* = 0,06), trois graines : g*_down = 0,15 → 79,94 (4–5 réchauffes, sur-régularisé) ; **0,30 → 80,39** (3–4 réchauffes ; +0,38 / +0,46 / +0,28 sur Rsteps4) ; 0,45 → 79,65 ; 0,60 → 79,98. Réponse non monotone, non résolue à trois graines.

**Six graines, ascension fixe (phase 5)** : Rsteps4 79,91 ; Rsteps4rh 80,37 ; **Rsteps4ar (réchauffes adaptatives, g*_down = 0,30, settle 2) 80,52** ; Rgap2s30 80,24. Contrastes appariés : **Rsteps4ar − Rsteps4 = +0,99 / +0,19 / +0,57 / +0,78 / +0,93 / +0,20, moyenne +0,61 ± 0,35, 6/6 positifs** ; Rsteps4rh − Rsteps4 +0,46 ± 0,31 (6/6) ; Rsteps4ar − Rsteps4rh +0,15 ± 0,55 (4/6, non résolu) ; Rgap2s30 − Rsteps4 +0,33 ± 0,25 (5/6). Réchauffes choisies : 4–5 par run, époques 14–27, espacées de 2–3, dépendantes de la graine.

## Coût

36 runs valides (dont 6 invalides non comptés), ≈ 590–660 s chacun ; ≈ 6,9 h GPU sur les quatre kernels.

## Interprétation et limites

Premier contrôleur adaptatif qui **dépasse les meilleures rampes fixes** : +0,61 point sur six graines appariées, positif partout, à budget et temps égaux, sans filtre, et au-dessus du gagnant filtré de la campagne mesuré dans ce protocole (80,48 %) pour 68 % de son temps. Le gain vient du mécanisme (les époques grossières tardives régularisent : le contrôle fixe le montre aussi), le signal place les réchauffes au moins aussi bien qu'une règle fixe réglée à la main (différence non résolue). L'ascension adaptative coûte ≈ 0,3 point face à l'ascension fixe Rsteps4. Plage utile du seuil étroite (0,15 trop, 0,45–0,60 trop peu, à trois graines). Non établi : robustesse hors CIFAR-10 / ResNet-20 / sans augmentation ; interaction avec le Gaussian ; STL-10.

## Décision et suite

Recommandation de l'agent : retenir « rampe fine fixe + réchauffes à 24 déclenchées par g(32→24) ≥ 0,30 » comme meilleure recette non filtrée ; suites dans l'ordre : robustesse du seuil (plus de graines, 0,25–0,35), avec Gaussian, STL-10 à temps égal. Aucune décision utilisateur enregistrée.

## Sources et manques

`results/kaggle_outputs/adaptive-phase4-20260917-151733/`, `adaptive-phase4b-20260917-154529/`, `adaptive-phase4c-20260917-165354/`, `adaptive-phase5-20260918-055940/` (logs, `pairing_verification.json`, `metrics.json`/`summary.json` avec `controller_log` et `reheats`). Manque : commit du code ; pas de figure ; pas d'arm « calendrier réalisé recodé en fixe ».

<!-- REPO-EVIDENCE:BEGIN -->

## Traces dans le dépôt (généré — ne pas éditer à la main)

Généré par `tools/enrich_records.py` depuis [`data/repo_evidence.json`](../data/repo_evidence.json). Voir aussi la vue d'ensemble : [REPO_EVIDENCE](../REPO_EVIDENCE.md).

**État du code par rapport à ce run** : controleur 'gap2s' (ascension fixe ou adaptative + rechauffes) ; graines configurables ; non commite.

| Rôle | Chemin dans le dépôt | Dernier commit touchant ce fichier |
|---|---|---|
| Implémentation | [`continuation/probe_signals.py`](../../../../continuation/probe_signals.py) | df89fbf 2026-09-18 |
| Implémentation | [`scripts/job_adaptive_phase0.py`](../../../../scripts/job_adaptive_phase0.py) | df89fbf 2026-09-18 |
| Implémentation | [`scripts/job_adaptive_phase4.py`](../../../../scripts/job_adaptive_phase4.py) | df89fbf 2026-09-18 |
| Implémentation | [`scripts/job_adaptive_phase4c.py`](../../../../scripts/job_adaptive_phase4c.py) | df89fbf 2026-09-18 |
| Implémentation | [`scripts/job_adaptive_phase5.py`](../../../../scripts/job_adaptive_phase5.py) | df89fbf 2026-09-18 |
| Sorties d'exécution | [`results/kaggle_outputs/adaptive-phase4-20260917-151733`](../../../../results/kaggle_outputs/adaptive-phase4-20260917-151733) | 135 fichiers |
| Sorties d'exécution | [`results/kaggle_outputs/adaptive-phase4b-20260917-154529`](../../../../results/kaggle_outputs/adaptive-phase4b-20260917-154529) | 123 fichiers |
| Sorties d'exécution | [`results/kaggle_outputs/adaptive-phase4c-20260917-165354`](../../../../results/kaggle_outputs/adaptive-phase4c-20260917-165354) | 124 fichiers |
| Sorties d'exécution | [`results/kaggle_outputs/adaptive-phase5-20260918-055940`](../../../../results/kaggle_outputs/adaptive-phase5-20260918-055940) | 146 fichiers |
| Documentation du dépôt | [`docs/adaptive_resolution_plan.md`](../../../../docs/adaptive_resolution_plan.md) | df89fbf 2026-09-18 |

Un chemin listé existe dans le dépôt au moment de la génération. Cela ne prouve pas seul quelle version du code a exécuté ce run : pour les runs Kaggle, le code réellement expédié est archivé dans le sous-dossier `_repo/` de la sortie correspondante.

<!-- REPO-EVIDENCE:END -->
