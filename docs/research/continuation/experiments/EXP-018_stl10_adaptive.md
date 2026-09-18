---
id: EXP-018
schema_version: 1
updated_at: 2026-09-18
status: completed_numeric_verified
evidence: local_per_epoch_json_and_kernel_log
protocol: P-STL10-R20-BN-72
source_json: ../../../../results/kaggle_outputs/stl10-adaptive-20260918-115842/stl10_resolution_20260918-115852/study_summary.json
code_commit_reported: null
code_commit_inspected: false
code_commit_inspected_note: code de la branche adaptative-resolution ; commit df89fbf + modifications de cette fiche
---

# EXP-018 — STL-10 : réchauffes adaptatives et contrôleur conjoint de pas, à compute égal, six graines

[Index](INDEX.md) · [EXP-015](EXP-015_stl10_resolution.md) · [EXP-017](EXP-017_two_sided_specialisation.md) · [Plan, section 19 (anglais)](../../../adaptive_resolution_plan.md)

## Métadonnées

| Champ | Valeur |
|---|---|
| ID campagne | `paulinezarka/stl10-adaptive-20260918-115842` (compte Kaggle paulinezarka) |
| Date | 2026-09-18, 11:58 → 13:47 UTC (6 518 s) |
| Question utilisateur | tester sur STL-10 la méthode qui bat les rampes fixes sur CIFAR, et un contrôleur qui détermine aussi le pas de la montée |
| Protocole | STL-10 étiqueté 5 000 / 8 000, 96×96, sans augmentation, ResNet-20 BN, SGD 0,005, batch 128 en 4×32, 40 updates/époque, **horizon 72 époques = 2 880 updates** (compute nominal ≈ 60 unités-époque à 96, comme R96 sur 60 époques) ; graines 0–5 régénérées et appariées dans le job |
| Code | `scripts/job_stl10_adaptive.py` → `job_stl10_resolution.py` (`STL10_PHASE=3`) |
| Vérification | JSON par époque, journal des décisions et des sondes g, pureté vérifiée à chaque sonde |

## Bras

`Rsteps4_72` : 48/60/72/84 × 6 puis 96 × 48 (rampe fine fixe à compute égal). `Rsteps4ar_72` : même montée + réchauffes à 72 quand g(96→72) ≥ 0,30 (settle 2). `Rjoint_72` : montée adaptative sur la grille des tailles multiples de 4 (candidats r+12/+24/+48 ; avancer quand la moyenne mobile de g vers le plus petit candidat ≥ 0,05 après ≥ 2 époques ; pas = plus grand candidat avec g ≤ 0,05 ; plancher r ≥ 48 + 12(e − 18), 96 à l'époque 30) + mêmes réchauffes.

## Résultats numériques

Accuracy test finale (%), graines 0–5 → moyenne : **Rsteps4_72 60,82 / 60,52 / 60,50 / 61,81 / 61,06 / 60,90 → 60,94** ; Rsteps4ar_72 60,88 / 60,59 / 60,30 / 61,36 / 60,68 / 60,50 → 60,72 ; Rjoint_72 60,26 / 59,90 / 60,41 / 61,46 / 61,24 / 60,77 → 60,68. Contrastes appariés : réchauffes − fixe −0,22 ± 0,23 (2/6) ; conjoint − fixe −0,26 ± 0,31 (1/6). Comparateurs antérieurs (graines 0–2, EXP-015) : R96_72 59,52 ; Rprogeq 60,48 ; Rlin24eq 60,25 : la rampe fine fixe à compute égal est le meilleur bras STL-10 (60,61 sur ces trois graines).

Comportement des contrôleurs : 0 à 1 réchauffe par run (une réchauffe tardive sur les graines 3–5, chaque fois légèrement nuisible) ; la montée conjointe n'a jamais déclenché sur le signal, le plancher a produit 48 × 19 → 60 → 72 → 84 → 96 dès l'époque 22 sur les six graines. Signal : g(+12) reste dans ±0,04 pendant les dix-neuf époques à 48 ; g(96→72) n'atteint 0,2–0,3 que dans les dernières époques. Taux de croissance **par update** ≈ 1,1 × 10⁻⁴, identique à CIFAR (≈ 1,3 × 10⁻⁴) ; ce sont les 40 updates par époque qui changent l'échelle.

## Coût

18 runs, 503–546 s d'entraînement chacun ; 6 518 s écoulées sur deux T4, ≈ 3,6 h GPU.

## Interprétation et limites

Troisième lecture pré-enregistrée : à 2 400–2 880 updates, la phase fine de STL-10 ne se spécialise pas assez pour que les réchauffes aient quelque chose à corriger ; le mécanisme n'est pas contredit, il n'a pas de matière. Les seuils calibrés sur CIFAR en époques ne se transposent pas ; la grandeur qui se transpose est la vitesse de spécialisation par update. Le contrôleur conjoint n'a pas été réellement exercé (plancher partout) : sa règle de pas reste non testée. Résultat positif collatéral : la rampe fine fixe à compute égal devient la meilleure recette STL-10 non filtrée. Pas d'augmentation, pas de filtre, pas d'appariement avec `stl10-transfer`.

## Décision et suite

Recommandation de l'agent : sur STL-10 à ce budget, garder la rampe fine fixe ; ne retester le contrôleur conjoint qu'avec des seuils exprimés par update (ou un horizon ≥ 10 000 updates) ; restreindre toujours les candidats aux tailles compatibles avec les strides (multiples de 4). Aucune décision utilisateur enregistrée.

## Sources et manques

`results/kaggle_outputs/stl10-adaptive-20260918-115842/` (log, `shared_manifest.json`, 18 `metrics.json`/`summary.json` avec `controller_log` : sondes g par époque et décisions). Manque : pas de figure ; contrôleur conjoint non exercé.

<!-- REPO-EVIDENCE:BEGIN -->

## Traces dans le dépôt (généré — ne pas éditer à la main)

Généré par `tools/enrich_records.py` depuis [`data/repo_evidence.json`](../data/repo_evidence.json). Voir aussi la vue d'ensemble : [REPO_EVIDENCE](../REPO_EVIDENCE.md).

**État du code par rapport à ce run** : controleurs (rechauffes, montee conjointe) ajoutes au job STL-10 apres df89fbf.

| Rôle | Chemin dans le dépôt | Dernier commit touchant ce fichier |
|---|---|---|
| Implémentation | [`scripts/job_stl10_resolution.py`](../../../../scripts/job_stl10_resolution.py) | df89fbf 2026-09-18 |
| Implémentation | [`scripts/job_stl10_adaptive.py`](../../../../scripts/job_stl10_adaptive.py) | — |
| Implémentation | [`continuation/probe_signals.py`](../../../../continuation/probe_signals.py) | df89fbf 2026-09-18 |
| Sorties d'exécution | [`results/kaggle_outputs/stl10-adaptive-20260918-115842`](../../../../results/kaggle_outputs/stl10-adaptive-20260918-115842) | 149 fichiers |
| Documentation du dépôt | [`docs/adaptive_resolution_plan.md`](../../../../docs/adaptive_resolution_plan.md) | df89fbf 2026-09-18 |

Un chemin listé existe dans le dépôt au moment de la génération. Cela ne prouve pas seul quelle version du code a exécuté ce run : pour les runs Kaggle, le code réellement expédié est archivé dans le sous-dossier `_repo/` de la sortie correspondante.

<!-- REPO-EVIDENCE:END -->
