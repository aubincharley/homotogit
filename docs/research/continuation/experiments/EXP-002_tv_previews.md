---
id: EXP-002
schema_version: 1
updated_at: 2026-09-09
status: previews_completed_partial_solver_convergence
evidence: local_diagnostics_json_and_images
protocol: P-PREVIEWS
---

# EXP-002 — Aperçus TV–L² et TV–Ḣ⁻¹ homogène

[Index](INDEX.md) · [Formules exactes](../OPERATORS.md)

## Question et portée

Comparer deux reconstructions sous une même contrainte de TV relative pour distinguer simplification spatiale et simple diminution du contraste. **Aucun entraînement de classification** : dix images, une par classe, six budgets, CPU. La version éditée du prompt `64829` est la spécification de référence : boîte RGB, moyennes par canal, budget TV global, aucune conservation de variance, H⁻¹ homogène sans alpha.

Les indices originaux CIFAR sont 15671, 18089, 18626, 24647, 25865, 26662, 30887, 34431, 39933, 48888. Ordre des classes : ship, cat, dog, truck, horse, airplane, bird, deer, automobile, frog. Le [JSON](../sources/tv_preview_diagnostics.json) conserve également les indices dans le split train et les TV initiales.

## Méthode numérique rapportée

Chambolle–Pock/PDHG, tolérance configurée 1e−7, maximum 40 000 itérations, pas primal/dual 0,3/0,3. Contrainte dure, pas de coefficient TV commun. La formule exacte du critère d'arrêt et des résidus doit être récupérée dans le code : la tolérance déclarée ne certifie pas chaque contrainte à 1e−7.

## Mesures

| Budget t | TV ratio moyen L² | Contraste L² | TV ratio moyen H⁻¹ | Contraste H⁻¹ |
|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 1 | 1 |
| 0,9 | 0,9000009 | 0,9835 | 0,9000002 | 0,9894 |
| 0,75 | 0,7500017 | 0,9516 | 0,7500012 | 0,9674 |
| 0,5 | 0,5000021 | 0,8687 | 0,5000022 | 0,9059 |
| 0,25 | 0,2500025 | 0,6977 | 0,2509496 | 0,7586 |
| 0 | 0 | 0 | 0 | 0 |

Le contraste H⁻¹ est plus élevé dans ces moyennes, mais le budget le plus serré n'est pas atteint avec la même précision. On ne doit pas ignorer ce défaut pour conclure à une supériorité à budget exactement égal.

Les solves L² convergent tous selon le critère implémenté pour t=0,9 / 0,75 / 0,5, pas tous pour 0,25. En H⁻¹, tous convergent pour 0,9 / 0,75, pas tous pour 0,5 / 0,25. À t=0,25, le champ `max_tv_budget_rel_violation` atteint **2,3042e−5** en L² et **0,0377369** en H⁻¹. Un exemple H⁻¹ conserve une TV ratio 0,259434 au lieu de 0,25. Plusieurs solves atteignent la limite de 40 000 itérations.

Les violations de boîte rapportées sont nulles, les dérives de moyenne autour de 1e−15. Les endpoints sont traités sans solve itératif. Certaines valeurs de fidélité endpoint sont `NaN` dans le JSON historique : elles sont non renseignées numériquement, pas une indétermination de la définition mathématique.

## Coût et conclusion

Durées `wall_seconds` des grilles : **1 112,67 s** pour L² et **2 221,53 s** pour H⁻¹. Les `solve_seconds` individuels et le temps global sont conservés comme périmètres distincts ; leur relation exacte n'est pas expliquée par le JSON et ne doit pas être inventée.

À t=0,5 et 0,25, les moyennes des temps individuels L² sont environ 9,5 et 43,6 s/image. L'extrapolation séquentielle vers 45k images donnerait environ 119 et 545 heures. Cela motive une piste moins coûteuse, mais ce n'est pas une borne intrinsèque du problème convexe ni une mesure sur GPU optimisé.

Les aperçus justifient l'intérêt visuel des contraintes ; ils ne démontrent ni conservation des indices sémantiques ni bénéfice en classification. La diffusion TV à M étapes reste une proposition distincte.

Sources : [JSON](../sources/tv_preview_diagnostics.json), [grille L²](../sources/transform_levels_tv_l2.png), [grille H⁻¹](../sources/transform_levels_tv_hminus1.png), [note de coût](../sources/note_tv_cout_fixe.md).

<!-- REPO-EVIDENCE:BEGIN -->

## Traces dans le dépôt (généré — ne pas éditer à la main)

Généré par `tools/enrich_records.py` depuis [`data/repo_evidence.json`](../data/repo_evidence.json). Voir aussi la vue d'ensemble : [REPO_EVIDENCE](../REPO_EVIDENCE.md).

**État du code par rapport à ce run** : tv.py corrigé pendant EXP-002 (signe de l'adjoint D^T) ; les apercus livres ont ete produits apres correction.

| Rôle | Chemin dans le dépôt | Dernier commit touchant ce fichier |
|---|---|---|
| Implémentation | [`continuation/transforms/tv.py`](../../../../continuation/transforms/tv.py) | 86c00bc 2026-09-08 |
| Implémentation | [`scripts/tv_previews.py`](../../../../scripts/tv_previews.py) | 86c00bc 2026-09-08 |
| Sorties d'exécution | [`results/tv_previews`](../../../../results/tv_previews) | 3 fichiers |
| Documentation du dépôt | [`docs/tv_budget.md`](../../../../docs/tv_budget.md) | 86c00bc 2026-09-08 |

Un chemin listé existe dans le dépôt au moment de la génération. Cela ne prouve pas seul quelle version du code a exécuté ce run : pour les runs Kaggle, le code réellement expédié est archivé dans le sous-dossier `_repo/` de la sortie correspondante.

<!-- REPO-EVIDENCE:END -->
