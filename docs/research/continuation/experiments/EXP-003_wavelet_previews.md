---
id: EXP-003
schema_version: 1
updated_at: 2026-09-09
status: previews_and_microbenchmarks_completed
evidence: local_diagnostics_json_and_images
protocol: P-PREVIEWS
---

# EXP-003 — Aperçus et coûts des ondelettes

[Index](INDEX.md) · [Opérateur](../OPERATORS.md) · [Pilote récent](EXP-009_db2_bn.md)

## Question

Remplacer la projection TV coûteuse par une simplification explicite qui conserve l'approximation et réduit les détails. Quatre familles : **haar, db2, sym4, coif1**, transformée stationnaire à deux niveaux, extension miroir, seuil doux proportionnel à la RMS par exemple/canal/bande. Les mêmes dix images que TV sont évaluées aux niveaux s=1 / 0,9 / 0,75 / 0,5 / 0,25 / 0.

Les figures affichent une plage fixe [0,1] sans adaptation de contraste. Le clipping éventuel est **uniquement d'affichage**, pas celui de l'opérateur. Voir OPERATORS pour la définition exacte.

## Aperçus db2

| s | TV ratio moyen | Contraste moyen, centré sur chaque sortie | Dérive max des moyennes de canal |
|---:|---:|---:|---:|
| 1 | 1 | 1 | 2,22e−16 |
| 0,9 | 0,8254 | 0,9571 | 0,000239 |
| 0,75 | 0,6609 | 0,9110 | 0,000244 |
| 0,5 | 0,5245 | 0,8675 | 0,000361 |
| 0,25 | 0,4685 | 0,8472 | 0,000231 |
| 0 | 0,4446 | 0,8376 | 0,000213 |

Le contraste reste important alors que TV diminue ; s=0 ne donne pas une image constante. Les bornes peuvent légèrement dépasser RGB : pour db2 s=0, environ −0,00349 à 1,03993 selon les diagnostics rapportés. Ces résultats sont descriptifs, pas une preuve de meilleur curriculum.

Durées des grilles CPU : haar 2,24 s ; db2 3,00 s ; sym4 2,74 s ; coif1 2,45 s. Le fait d'être beaucoup plus rapide que TV sur dix images n'implique pas un coût comparable au Gaussian dans tout un réseau.

## Microbenchmarks séparés

RTX 3050 Laptop 4 Go, Torch 2.5.1+cu121, float32. Mesures médianes après échauffement, synchronisées CUDA, **sans optimizer step**, sans transferts ni fichiers. db2 s=0,5 ; Gaussian sigma=1, variante de support pour sigma max 1.

| Tenseur | db2 forward | db2 forward+backward | Gaussian forward+backward | Pic db2 |
|---|---:|---:|---:|---:|
| 128×3×32×32 | 22,67 ms | 56,75 ms | 0,759 ms | 387,8 MiB |
| 128×16×32×32 | 117,71 ms | 292,39 ms | 2,661 ms | 2 072,1 MiB |
| 128×32×16×16 | 60,65 ms | 152,46 ms | 1,544 ms | 1 059,6 MiB |
| 128×64×8×8 | 31,42 ms | 81,46 ms | 0,984 ms | 560,2 MiB |

Le JSON conserve aussi une variante Gaussian pour sigma max 3 ; elle est indisponible sur la petite carte avec l'ancienne restriction de support. Cette absence est historique, pas une impossibilité générale après ajout de la réflexion explicite.

CPU, une image : db2 environ 14,40 ms contre Gaussian sigma-max1 environ 0,426 ms. Les timings de grille incluent un autre périmètre et ne doivent pas être additionnés aux microbenchmarks comme s'ils décrivaient la même opération mesurée.

## Ce que cela a conduit à faire

db2 est retenu pour des pilotes internes, puis laissé en attente pendant la validation Gaussian. Le pilote BN ultérieur mesure le coût du réseau complet avec 19 insertions sur T4 et aboutit à l'abandon de cette option. Ces microbenchmarks anciens ne doivent donc pas être réutilisés pour promettre un entraînement db2 de quelques minutes.

Sources : [diagnostics](../sources/wavelet_preview_diagnostics.json), [timings](../sources/wavelet_timings.json), figures [haar](../sources/transform_levels_wavelet_haar.png), [db2](../sources/transform_levels_wavelet_db2.png), [sym4](../sources/transform_levels_wavelet_sym4.png), [coif1](../sources/transform_levels_wavelet_coif1.png).

<!-- REPO-EVIDENCE:BEGIN -->

## Traces dans le dépôt (généré — ne pas éditer à la main)

Généré par `tools/enrich_records.py` depuis [`data/repo_evidence.json`](../data/repo_evidence.json). Voir aussi la vue d'ensemble : [REPO_EVIDENCE](../REPO_EVIDENCE.md).

**État du code par rapport à ce run** : chemin 'fast' fusionne et bypass s=1 ajoutes apres les premiers apercus ; les deux chemins calculent la meme operation (verifie par les tests).

| Rôle | Chemin dans le dépôt | Dernier commit touchant ce fichier |
|---|---|---|
| Implémentation | [`continuation/transforms/wavelet.py`](../../../../continuation/transforms/wavelet.py) | d643f75 2026-09-08 |
| Implémentation | [`scripts/wavelet_previews.py`](../../../../scripts/wavelet_previews.py) | d643f75 2026-09-08 |
| Implémentation | [`scripts/wavelet_benchmark.py`](../../../../scripts/wavelet_benchmark.py) | d643f75 2026-09-08 |
| Implémentation | [`scripts/wavelet_profile.py`](../../../../scripts/wavelet_profile.py) | d643f75 2026-09-08 |
| Implémentation | [`scripts/wavelet_optimized_benchmark.py`](../../../../scripts/wavelet_optimized_benchmark.py) | d643f75 2026-09-08 |
| Sorties d'exécution | [`results/wavelet_previews`](../../../../results/wavelet_previews) | 14 fichiers |
| Documentation du dépôt | [`docs/wavelet_shrinkage.md`](../../../../docs/wavelet_shrinkage.md) | d643f75 2026-09-08 |

Un chemin listé existe dans le dépôt au moment de la génération. Cela ne prouve pas seul quelle version du code a exécuté ce run : pour les runs Kaggle, le code réellement expédié est archivé dans le sous-dossier `_repo/` de la sortie correspondante.

<!-- REPO-EVIDENCE:END -->
