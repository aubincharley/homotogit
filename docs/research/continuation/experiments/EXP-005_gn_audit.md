---
id: EXP-005
schema_version: 1
updated_at: 2026-09-09
status: completed_partial_record
evidence: conversation_figure_and_audit_report
protocol: P-INT-GN
---

# EXP-005 — Gaussian interne GroupNorm, contrôle LR et audit

[Index](INDEX.md) · Suite : [ResNet-18 BN](EXP-006_resnet18_bn.md)

## Comparaison principale

10k images d'entraînement, 5k validation, 1 200 updates, trois graines appariées, ResNet-20 GroupNorm, LR pic 0,005. Fin de continuation à 600 updates. La figure et les échanges rapportent **Gaussian − plain = −3,45 points** à la fin.

Les lectures graphiques donnent environ 50,5 % plain contre 47,0 % Gaussian et une CE train autour de 1,3–1,4. Les valeurs exactes par graine et les logs ne sont pas dans les fichiers récupérés ; ne pas transformer ces estimations en tableau exact.

## Audit et contrôle à deux learning rates

Un audit rapporte un placement du Gaussian correct, après les convolutions principales et avant la normalisation. Un contrôle à deux LR confirme le signe défavorable dans cette recette. La conversation identifie un contrôle 0,002 face au 0,005 utilisé dans l'étude ; la table complète n'a pas été retrouvée.

Repères d'artifacts : `results/gaussian_placement_audit.json`, `results/kaggle_outputs/lr-control-002-20260908-140604/`. [Job du contrôle](https://www.kaggle.com/code/maxnicaise/lr-control-002-20260908-140604).

## Différences à CBS mises en évidence

Ancien petit ResNet-20/GN, entraînement court, extinction rapide du filtre et initialisation éloignée de la recette auditée. Le classifieur est rapporté comme Kaiming `fan_out` avec gain ReLU : pour dix sorties, std théorique \(\sqrt{2/10}\simeq0,447\), contre 0,01 dans la version corrigée. Cela peut rendre les logits initiaux très confiants et produire des CE élevées ; la mesure des logits n'a pas été retrouvée.

Le rapport de paramètres ResNet-18/20 ne devait pas être pris pour un rapport de coût : la discussion a corrigé l'extrapolation vers des MACs, puis exigé un timing réel T4. Les premiers plafonds de quelques minutes étaient spécifiques au pilote, pas à la campagne longue ultérieure.

## Conclusion et suite

Le résultat négatif reste valable pour cette configuration. Il ne correspond pas à une reproduction exacte de CBS qui échouerait. Le choix suivant est de conserver la petite échelle de données et de corriger architecture/normalisation/initialisation ensemble, afin de voir si le signe change avant une étude plus coûteuse.

Source locale : [figure](../sources/plain_vs_gaussian_study.png), [ancienne passation](../sources/passation_2026-09-08.md), échanges d'audit reproduits dans le contexte. Révision du dépôt CBS auditée, code du contrôle et métriques exactes restent à récupérer.

<!-- REPO-EVIDENCE:BEGIN -->

## Traces dans le dépôt (généré — ne pas éditer à la main)

Généré par `tools/enrich_records.py` depuis [`data/repo_evidence.json`](../data/repo_evidence.json). Voir aussi la vue d'ensemble : [REPO_EVIDENCE](../REPO_EVIDENCE.md).

**État du code par rapport à ce run** : continuation_driver.py a beaucoup evolue depuis (tables par paliers, chemins courant/cible, resolution). Le code expedie a ce run est conserve dans le _repo/ de chaque sortie Kaggle, pas dans l'arbre courant.

| Rôle | Chemin dans le dépôt | Dernier commit touchant ce fichier |
|---|---|---|
| Implémentation | [`scripts/continuation_driver.py`](../../../../scripts/continuation_driver.py) | 26cfb82 2026-09-09 |
| Implémentation | [`scripts/job_plain_study.py`](../../../../scripts/job_plain_study.py) | d643f75 2026-09-08 |
| Implémentation | [`scripts/job_gaussian_study.py`](../../../../scripts/job_gaussian_study.py) | d643f75 2026-09-08 |
| Implémentation | [`scripts/job_lr_diagnostic.py`](../../../../scripts/job_lr_diagnostic.py) | d643f75 2026-09-08 |
| Implémentation | [`scripts/job_lr_control_002.py`](../../../../scripts/job_lr_control_002.py) | d643f75 2026-09-08 |
| Implémentation | [`scripts/audit_gaussian_placement.py`](../../../../scripts/audit_gaussian_placement.py) | d643f75 2026-09-08 |
| Implémentation | [`scripts/verify_grad_accumulation.py`](../../../../scripts/verify_grad_accumulation.py) | d643f75 2026-09-08 |
| Implémentation | [`continuation/models/resnet_gn.py`](../../../../continuation/models/resnet_gn.py) | 86c00bc 2026-09-08 |
| Configuration | [`scripts/_study_common.py`](../../../../scripts/_study_common.py) | d643f75 2026-09-08 |
| Sorties d'exécution | [`results/kaggle_outputs/plain-study-20260908-132611`](../../../../results/kaggle_outputs/plain-study-20260908-132611) | 76 fichiers |
| Sorties d'exécution | [`results/kaggle_outputs/gaussian-study-20260908-132957`](../../../../results/kaggle_outputs/gaussian-study-20260908-132957) | 76 fichiers |
| Sorties d'exécution | [`results/kaggle_outputs/lr-diagnostic-20260908-132240`](../../../../results/kaggle_outputs/lr-diagnostic-20260908-132240) | 66 fichiers |
| Sorties d'exécution | [`results/kaggle_outputs/lr-control-002-20260908-140604`](../../../../results/kaggle_outputs/lr-control-002-20260908-140604) | 73 fichiers |
| Sorties d'exécution | [`results/gaussian_placement_audit.json`](../../../../results/gaussian_placement_audit.json) | d643f75 2026-09-08 |
| Sorties d'exécution | [`results/grad_accumulation_check.json`](../../../../results/grad_accumulation_check.json) | d643f75 2026-09-08 |
| Sorties d'exécution | [`results/plain_vs_gaussian_paired.json`](../../../../results/plain_vs_gaussian_paired.json) | d643f75 2026-09-08 |

Un chemin listé existe dans le dépôt au moment de la génération. Cela ne prouve pas seul quelle version du code a exécuté ce run : pour les runs Kaggle, le code réellement expédié est archivé dans le sous-dossier `_repo/` de la sortie correspondante.

<!-- REPO-EVIDENCE:END -->
