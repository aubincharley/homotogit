---
id: EXP-010
schema_version: 1
updated_at: 2026-09-09
status: completed_reported
evidence: local_execution_report_and_figure
protocol: P-FULL-R20-BN
---

# EXP-010 — Premier essai de résolution progressive sur CIFAR complet

[Index](INDEX.md) · Suite : [grille trois graines](EXP-011_full_grid.md)

## Question et branches

La réduction réelle de résolution, éventuellement combinée au Gaussian interne, ajoute-t-elle un bénéfice ? Deux **nouveaux** bras seed0 : résolution progressive seule et résolution progressive + Gaussian paliers. Deux références d'EXP-008 sont réutilisées : plain et Gaussian paliers à 32 constant.

Même profil 50k/10k, ResNet-20 BN, 30 époques, 11 730 updates, batch 32×4, pic LR 0,005, warmup 60, sans augmentation. r=16 aux époques e=0…5, r=24 de 6…11, r=32 ensuite. Resize réel bilinéaire avec AA en entrée, sans remontée. Sigma effectif \(r/32\times g(e)\), avec les paliers de trois époques et extinction à e=21.

## Vérifications et incidents locaux

Les quatre SHA-256 d'appariement (subset, sonde, permutation seed0 30×50 000, initialisation 116 tenseurs) sont contrôlés dans le job avant formation et rapportés identiques. Les formes de stages 16/8/4, 24/12/6 et 32/16/8 restent compatibles avec les raccourcis, la moyenne globale et la tête 64→10. Les 19 sites restent actifs à petite résolution avec la réflexion explicite, sans modification du support.

Un essai local court détecte le resize placé sur `uint8` : le pipeline est corrigé pour convertir en float avant la réduction. Une assertion dépendante de l'ordre de `describe()` est aussi corrigée. Suite de tests rapportée : **170 réussis, un échec préexistant** du test qui attendait l'ancienne exception de rayon sur petite image. Ne pas déclarer « tous les tests passent ».

Le retour r=32 est une identité exacte. Les hausses sigma 0,425→0,525 à e=6 et 0,450→0,500 à e=12 sont vérifiées comme intentionnelles, dues au changement d'échelle.

## Résultats finaux

| Bras | Accuracy test | CE test | Accuracy sonde train | CE sonde train | Temps formation |
|---|---:|---:|---:|---:|---:|
| Plain historique | 75,10 % | 0,7592 | 94,40 % | 0,2374 | 466 s |
| Gaussian historique | 78,27 % | 0,6347 | 88,20 % | 0,4029 | 689 s |
| Résolution progressive | **79,57 %** | 0,6119 | 89,60 % | 0,3505 | 436 s |
| Résolution + Gaussian | **79,93 %** | 0,5881 | 89,20 % | 0,3457 | 651 s |

Différences : résolution−plain **+4,47 points** ; résolution−Gaussian **+1,30** ; combinaison−Gaussian **+1,66** ; combinaison−résolution **+0,36**. La dernière marge est petite sur une graine ; elle motive la réplication plutôt qu'une conclusion définitive.

## Chemins courant et cible

| Époques accomplies | Résolution courant / cible | Combinaison courant / cible |
|---:|---:|---:|
| 6 | 56,66 / 35,04 % | 59,79 / 26,77 % |
| 12 | 69,24 / 66,21 % | 71,14 / 32,57 % |
| 21 | 78,59 / 78,59 % | 79,10 / 78,85 % |
| 30 | 79,57 / 79,57 % | 79,93 / 79,93 % |

Les configurations courantes sont celles de la dernière update accomplie. Les grands écarts de chemin cible tôt dans le run ne décrivent pas un modèle qui n'apprend pas ; la fonction et les distributions changent. À la fin, les chemins coïncident.

## Travail spatial et temps réel

Le ratio nominal de convolution du calendrier est

\[
\frac6{30}(16/32)^2+\frac6{30}(24/32)^2+\frac{18}{30}=0,7625.
\]

Cela représente 23,75 % de travail spatial en moins dans ce modèle simplifié. Pourtant les sondes T4 sans filtre donnent 0,0328 / 0,0340 / 0,0324 s par update pour r=16/24/32. Les petites convolutions ne sont pas plus rapides dans ces mesures ; leur sous-utilisation du GPU est une hypothèse plausible, pas une preuve de profilage.

Évaluation des nouveaux runs : 16 et 22 s ; temps totaux 452 et 674 s ; pics 512 et 583 MiB. Les temps de formation sont environ 6 % plus faibles que les références historiques, mais ils proviennent de jobs distincts. Ne pas attribuer tout cet écart à la résolution ni annoncer une économie garantie.

## Sources

[Rapport complet](../sources/progressive_resolution_report.md), [figure](../sources/progressive_resolution.png), [job](https://www.kaggle.com/code/maxnicaise/progres-r20bn-20260909-075846). Dossier `results/kaggle_outputs/progres-r20bn-20260909-075846/progres_r20bn_20260909-075919/`, checkpoints après 6/12/21/30, `pairing_verification.json`, `t4_timing_probe.json`, scripts `job_progressive_resolution.py`, `verify_progressive_resolution.py`, `plot_progressive_resolution.py`.

Test suivi de manière exploratoire, un seul seed, pas de sélection best-epoch. Les quatre branches de cette comparaison ne sont pas quatre runs contemporains fraîchement exécutés ; la grille suivante le sera.

<!-- REPO-EVIDENCE:BEGIN -->

## Traces dans le dépôt (généré — ne pas éditer à la main)

Généré par `tools/enrich_records.py` depuis [`data/repo_evidence.json`](../data/repo_evidence.json). Voir aussi la vue d'ensemble : [REPO_EVIDENCE](../REPO_EVIDENCE.md).

**État du code par rapport à ce run** : le redimensionnement vit dans InputPipeline (uint8/255 -> float -> resize -> T_eta -> normalisation) ; la verification d'appariement par empreintes est dans le job.

| Rôle | Chemin dans le dépôt | Dernier commit touchant ce fichier |
|---|---|---|
| Implémentation | [`scripts/job_progressive_resolution.py`](../../../../scripts/job_progressive_resolution.py) | 26cfb82 2026-09-09 |
| Implémentation | [`scripts/verify_progressive_resolution.py`](../../../../scripts/verify_progressive_resolution.py) | 26cfb82 2026-09-09 |
| Implémentation | [`scripts/plot_progressive_resolution.py`](../../../../scripts/plot_progressive_resolution.py) | 26cfb82 2026-09-09 |
| Implémentation | [`continuation/pipeline.py`](../../../../continuation/pipeline.py) | 26cfb82 2026-09-09 |
| Implémentation | [`scripts/continuation_driver.py`](../../../../scripts/continuation_driver.py) | 26cfb82 2026-09-09 |
| Sorties d'exécution | [`results/kaggle_outputs/progres-r20bn-20260909-075846`](../../../../results/kaggle_outputs/progres-r20bn-20260909-075846) | 10 fichiers |
| Sorties d'exécution | [`results/progressive_resolution.png`](../../../../results/progressive_resolution.png) | 26cfb82 2026-09-09 |
| Sorties d'exécution | [`results/progressive_resolution_verification.json`](../../../../results/progressive_resolution_verification.json) | 26cfb82 2026-09-09 |

Un chemin listé existe dans le dépôt au moment de la génération. Cela ne prouve pas seul quelle version du code a exécuté ce run : pour les runs Kaggle, le code réellement expédié est archivé dans le sous-dossier `_repo/` de la sortie correspondante.

<!-- REPO-EVIDENCE:END -->
