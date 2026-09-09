---
id: EXP-011
schema_version: 1
updated_at: 2026-09-09
status: completed_numeric_verified
evidence: local_final_json_execution_report_and_original_figure
protocol: P-FULL-R20-BN
source_json: ../sources/campaign_results.json
code_commit_reported: f191fa6
code_commit_inspected: true
code_commit_inspected_on: 2026-09-09
code_commit_inspected_note: résolu dans le dépôt, voir INTEGRATION.md C-31
---

# EXP-011 — Benchmark complet : 21 configurations, trois graines

[Index](INDEX.md) · [Tableaux complets recalculés](../data/campaign_tables.md) · [JSON original](../sources/campaign_results.json) · [Lecture actuelle](../CURRENT_STATE.md)

## 1. Objectif et portée

Consolider CIFAR-10 avant STL-10, avec une grille de résolutions et de calendriers Gaussian, des contrôles du mélange identité/Gaussian, du nombre de sites, de l'opérateur et du lieu de réduction, ainsi que de l'ordre des résolutions.

Même profil `P-FULL-R20-BN` : 50k train / 10k test, 30 époques, 391 updates/époque, 11 730 updates, ResNet-20 BN corrigé, batch 32×4, pic LR 0,005, warmup 60 puis cosine, sans augmentation. Graines 0/1/2, poids/buffers initiaux et permutations appariés. Comparaison finale **epoch 30**, tous les modèles à 32×32 et sans opérateur de continuation.

Le présent recalcul vérifie les nombres disponibles dans le JSON ; il ne réexécute ni code d'entraînement ni tests d'architecture.

## 2. Pourquoi les groupes A, B, C, D, E ?

Ce sont cinq **questions d'ablation**, pas cinq grilles indépendantes ou cinq datasets.

| Groupe | Combinaisons | Configurations | Runs avec trois graines |
|---|---|---:|---:|
| A | R32/Rprog/Rgentle × Gnone/Gplateau/Ggeo ; bilinéaire entrée, all19 | 9 | 27 |
| B | R32/Rprog × Gmix ; bilinéaire entrée, all19 | 2 | 6 |
| C | R32/Rprog × Gplateau ; bilinéaire entrée, early7 | 2 | 6 |
| D | Rprog × input_max/stem_bilinear/stem_max × Gnone/Gplateau ; all19 | 6 | 18 |
| E | Rreverse × Gnone/Gplateau ; bilinéaire entrée, all19 | 2 | 6 |
| **Total** | | **21** | **63** |

Rprog=16 pendant six époques, 24 pendant six, puis 32 ; Rgentle=24 pendant douze puis 32 ; Rreverse=24 puis 16 sur les deux premiers blocs de six. Gaussian paliers, géométrique, Gmix, early7 et les lieux sont définis dans [OPERATORS](../OPERATORS.md) et [PROTOCOLS](../PROTOCOLS.md).

**Baseline complète : `R32__Gnone__input_bilinear__all19`.** Malgré les deux derniers mots, aucun resize ni filtre n'agit : r=32 et Gnone sont des identités. `all19` indique le masque disponible, pas 19 filtres actifs dans le témoin.

## 3. Exécution réelle et incident de réutilisation

L'agent prévoyait de réutiliser 11 cellules et d'en former 52 nouvelles. `find_reusable()` inspectait `results/kaggle_outputs/`, absent des kernels expédiés. Il a donc trouvé zéro cellule réutilisable : **les 63 ont été réentraînées**, réparties en trois groupes disjoints de 21.

Le JSON indique 63 attendues, 63 terminées, aucune cellule manquante et zéro doublon. Chaque job rapporte `assets_verified: true`. Le compte rendu indique zéro échec. Les manifestes détaillés sont cités mais ne sont pas joints à cette base ; on ne doit pas prétendre les avoir audités ligne par ligne.

| Job | Temps écoulé du job |
|---|---:|
| `campaign-j0-20260909-095039` | 6 270,45 s |
| `campaign-j1-20260909-095101` | 7 109,84 s |
| `campaign-j2-20260909-095123` | 7 431,07 s |

Deux workers par job, chacun sur une T4. Total de durées de runs : **40 069,9729 secondes GPU**, soit **11,13 heures GPU**. Durée globale rapportée 09:51→11:59Z, environ **2 h 08**, sous le budget utilisateur 2h30–3h écoulées. Les secondes de jobs ne doivent pas être additionnées comme s'il n'y avait qu'un GPU par job.

Une prévision de 83 min était trop basse, notamment parce que la réutilisation n'avait pas eu lieu. Le coût supplémentaire est réel ; il ne rend pas les résultats invalides. Le commit `f191fa6` est rapporté comme poussé, sans inspection locale de son contenu.

## 4. Résultats centraux

Accuracy en % et SD en points, temps total moyen par run. Le [tableau intégral](../data/campaign_tables.md) contient les 21 configurations, les seeds et les CE.

| Méthode | Accuracy moyenne ± SD | CE test | Temps moyen |
|---|---:|---:|---:|
| Témoin 32 constant | 75,62 ± 0,49 | 0,7552 | 8,15 min |
| Gaussian paliers seul | 78,69 ± 0,68 | 0,6210 | 11,60 min |
| Résolution progressive bilinéaire seule | 79,48 ± 0,54 | 0,5989 | 8,21 min |
| **Résolution progressive bilinéaire + Gaussian paliers** | **80,71 ± 0,65** | **0,5682** | **11,66 min** |
| Résolution progressive max en entrée + Gaussian | 80,34 ± 0,76 | 0,5720 | 12,30 min |
| Résolution progressive max après stem, sans Gaussian | 79,97 ± 0,58 | 0,5894 | 8,18 min |
| Résolution progressive + géométrique | 79,91 ± 0,92 | 0,5918 | 11,81 min |

Les seeds du gagnant sont **80,15 / 80,57 / 81,42 %** ; celles du témoin **75,13 / 75,62 / 76,10 %**.

## 5. Contribution de la résolution et du Gaussian

| Différence appariée A−B | Moyenne ± SD (points) | Différences par graine (points) |
|---|---:|---|
| Combo principal − plain | +5,10 ± 0,20 | +5,02 / +4,95 / +5,32 |
| Résolution seule − plain | +3,87 ± 0,30 | +4,07 / +3,52 / +4,01 |
| Gaussian seul − plain | +3,08 ± 0,34 | +2,78 / +3,45 / +3,00 |
| Combo − résolution seule | +1,23 ± 0,25 | +0,95 / +1,43 / +1,31 |
| Combo − Gaussian seul | +2,02 ± 0,45 | +2,24 / +1,50 / +2,32 |

Les cinq contrastes sont positifs sur les trois graines. Le supplément du Gaussian avec résolution est plus faible qu'à 32 constant ; les gains ne s'additionnent pas. C'est compatible avec un recouvrement de leurs effets, mais ne démontre pas un même mécanisme, notamment parce que le sigma effectif dépend de r.

L'alternative max après stem sans Gaussian est **0,75 ± 0,20 point** derrière le combo, avec environ 30 % de temps en moins. Les deux peuvent être conservées selon l'objectif précision/coût ; ne pas confondre cette frontière empirique avec un classement universel par accuracy/seconde.

## 6. Calendriers et emplacements : ce qui déçoit

**Gmix** est derrière Gplateau à R32 de −0,51 point (SD 0,33), et à Rprog de −1,20 (SD 0,17), avec signe négatif sur chaque graine. À Rprog, son écart moyen au bras sans Gaussian n'est que +0,03 point avec signes mixtes, pour un temps plus élevé. Déprioriser ce réglage est raisonnable ; les forces de filtrage n'étant pas appariées, il ne prouve pas que tout mélange identité/filtre est inférieur.

**early7** perd contre all19 : −3,46 points à R32 (SD 1,10) et −1,44 à Rprog (SD 0,71). À 32 constant, son résultat 75,24 % revient près du plain 75,62 %. Le bénéfice du Gaussian courant n'est donc pas préservé en le limitant au stem et premier stage dans cette recette.

**Rgentle** reste meilleur que plain, mais derrière Rprog dans les bras principaux. **Ggeo** reste positif contre plain, mais à Rprog les paliers le dépassent de +0,81 point en moyenne appariée, positif sur les trois graines. Ne pas fusionner ce classement avec l'ancienne campagne où paliers/géométrique étaient pratiquement à égalité en accuracy.

## 7. Max contre bilinéaire : comparer à emplacement fixé

Tous les contrastes suivants gardent Rprog. Moyennes ± SD en points, calculées à partir du JSON.

| A−B | Sans Gaussian | Avec Gaussian paliers |
|---|---:|---:|
| Max entrée − bilinéaire entrée | +0,31 ± 0,28 | −0,37 ± 0,11 |
| Bilinéaire stem − bilinéaire entrée | −0,58 ± 0,48 | −1,42 ± 0,16 |
| Max stem − bilinéaire entrée | +0,48 ± 0,06 | −0,90 ± 0,17 |
| **Max stem − bilinéaire stem** | **+1,06 ± 0,49** | **+0,51 ± 0,19** |
| Max stem − max entrée | +0,17 ± 0,26 | −0,53 ± 0,12 |

Le max en entrée est légèrement devant le bilinéaire sans Gaussian et légèrement derrière avec Gaussian. Après le stem, **max bat bilinéaire dans les deux conditions**, sur chaque graine. Dire « toute réduction après stem est moins bonne » contredirait le bon résultat de stem_max sans Gaussian.

Ajouter Gaussian au max d'entrée donne environ +0,54 point en moyenne. L'ajouter au max après stem donne environ −0,16 point avec signes mixtes ; ce petit effet moyen ne suffit pas à affirmer un préjudice établi.

Le bilinéaire est antialiasé ; le max n'a pas de filtre AA ajouté. Les contrastes opérateurs comprennent aussi cette différence de traitement spectral. Les variantes en entrée et après stem n'agissent ni sur les mêmes canaux ni sur les mêmes caractéristiques.

## 8. Contrôle de l'ordre des résolutions

Rreverse−Rprog vaut −0,26 point sans Gaussian (SD 0,61, signes mixtes), mais −1,39 avec paliers (SD 0,27, négatif partout). Cela motive une interaction entre ordre et filtrage. L'absence de différence nette sans filtre ne prouve pas l'équivalence des ordres.

Les deux calendriers passent autant d'epochs à chaque résolution, mais à des moments différents du LR. Avec Gaussian, ils changent également la suite effective \(\sigma=qg\). Le résultat ne permet pas d'attribuer l'écart à l'ordre seul indépendamment du filtre et de l'optimisation globale.

## 9. Coût et réplication historique

Sondes par update complète : plain environ 0,032–0,035 s, Gaussian 0,057–0,065 s, Gmix 0,067–0,071 s, early7 environ 0,043 s. Les résolutions réduites n'apportent pas d'économie temporelle systématique par update. Les pics complets sont environ 512–616 MiB ; évaluations autour de 3,3 % du temps total.

Les trois anciennes configurations R32 ont été réentraînées. Écarts moyens de test accuracy contre EXP-008 : plain +0,0021, paliers +0,0028, géométrique −0,0018 en fractions. Maxima absolus par graine : 0,0036 / 0,0091 / 0,0074, soit 0,36 / 0,91 / 0,74 point. Les statistiques et assets rapportés identiques rendent ces écarts instructifs ; ils ne définissent pas un plancher de bruit pour toutes les méthodes et ne prouvent pas leur cause unique.

## 10. Conclusion située et suite

Conserver comme candidates la résolution bilinéaire progressive + Gaussian paliers, et le max après stem sans Gaussian pour le coût. Conserver plain, Gaussian seul et résolution seule comme contrôles. Déprioriser Gmix et early7 dans les réglages essayés. Ne pas relancer db2, abandonné auparavant.

La sélection a été faite dans une grille sur un test déjà consulté : le meilleur score est exploratoire, et trois graines donnent une SD descriptive. Aucune formation STL-10, aucun Gaussian RGB explicite avant réduction et aucune nouvelle grille après celle-ci ne sont établis par ces résultats.

## 11. Sources et artifacts à enrichir

- [JSON final original](../sources/campaign_results.json), [rapport d'exécution complet](../sources/full_grid_report.md), [figure originale](../sources/campaign_curves.png).
- Dossiers `results/kaggle_outputs/campaign-j{0,1,2}-*/`, `results/campaign_manifest_frozen.json`, `results/campaign_verification.json`, `results/campaign_previews/` cités dans le rapport.
- Le JSON contient valeurs finales par graine, agrégats, 24 comparaisons préenregistrées dans l'analyse, sondes des jobs et réplications historiques ; **pas les séries temporelles complètes**.
- Les nouvelles figures ont été mentionnées comme vues par l'utilisateur, sans fichiers récupérés. Ne pas redessiner des trajectoires à trois graines à partir des seules valeurs finales.

<!-- REPO-EVIDENCE:BEGIN -->

## Traces dans le dépôt (généré — ne pas éditer à la main)

Généré par `tools/enrich_records.py` depuis [`data/repo_evidence.json`](../data/repo_evidence.json). Voir aussi la vue d'ensemble : [REPO_EVIDENCE](../REPO_EVIDENCE.md).

**État du code par rapport à ce run** : code de la campagne ecrit pour ce run et inchange depuis ; commit f191fa6 contient les resultats.

| Rôle | Chemin dans le dépôt | Dernier commit touchant ce fichier |
|---|---|---|
| Implémentation | [`continuation/campaign_ops.py`](../../../../continuation/campaign_ops.py) | b9bb609 2026-09-09 |
| Implémentation | [`scripts/campaign_driver.py`](../../../../scripts/campaign_driver.py) | b9bb609 2026-09-09 |
| Implémentation | [`scripts/campaign_manifest.py`](../../../../scripts/campaign_manifest.py) | b9bb609 2026-09-09 |
| Implémentation | [`scripts/job_campaign.py`](../../../../scripts/job_campaign.py) | b9bb609 2026-09-09 |
| Implémentation | [`scripts/job_campaign_0.py`](../../../../scripts/job_campaign_0.py) | b9bb609 2026-09-09 |
| Implémentation | [`scripts/job_campaign_1.py`](../../../../scripts/job_campaign_1.py) | b9bb609 2026-09-09 |
| Implémentation | [`scripts/job_campaign_2.py`](../../../../scripts/job_campaign_2.py) | b9bb609 2026-09-09 |
| Implémentation | [`scripts/verify_campaign_ops.py`](../../../../scripts/verify_campaign_ops.py) | b9bb609 2026-09-09 |
| Implémentation | [`scripts/stage_campaign_assets.py`](../../../../scripts/stage_campaign_assets.py) | b9bb609 2026-09-09 |
| Implémentation | [`scripts/analyze_campaign.py`](../../../../scripts/analyze_campaign.py) | f191fa6 2026-09-09 |
| Implémentation | [`scripts/make_presentation.py`](../../../../scripts/make_presentation.py) | 9da41d6 2026-09-09 |
| Implémentation | [`scripts/presentation_labels.py`](../../../../scripts/presentation_labels.py) | 9da41d6 2026-09-09 |
| Configuration | [`results/campaign_manifest_frozen.json`](../../../../results/campaign_manifest_frozen.json) | b9bb609 2026-09-09 |
| Sorties d'exécution | [`results/kaggle_outputs/campaign-j0-20260909-095039`](../../../../results/kaggle_outputs/campaign-j0-20260909-095039) | 219 fichiers |
| Sorties d'exécution | [`results/kaggle_outputs/campaign-j1-20260909-095101`](../../../../results/kaggle_outputs/campaign-j1-20260909-095101) | 219 fichiers |
| Sorties d'exécution | [`results/kaggle_outputs/campaign-j2-20260909-095123`](../../../../results/kaggle_outputs/campaign-j2-20260909-095123) | 219 fichiers |
| Sorties d'exécution | [`results/campaign_results.json`](../../../../results/campaign_results.json) | f191fa6 2026-09-09 |
| Sorties d'exécution | [`results/campaign_verification.json`](../../../../results/campaign_verification.json) | b9bb609 2026-09-09 |
| Sorties d'exécution | [`results/campaign_previews`](../../../../results/campaign_previews) | 2 fichiers |
| Sorties d'exécution | [`results/presentation`](../../../../results/presentation) | 43 fichiers |
| Documentation du dépôt | [`docs/HANDOVER.md`](../../../../docs/HANDOVER.md) | 5ec3217 2026-09-09 |

Un chemin listé existe dans le dépôt au moment de la génération. Cela ne prouve pas seul quelle version du code a exécuté ce run : pour les runs Kaggle, le code réellement expédié est archivé dans le sous-dossier `_repo/` de la sortie correspondante.

<!-- REPO-EVIDENCE:END -->
