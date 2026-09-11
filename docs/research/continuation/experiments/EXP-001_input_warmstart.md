---
id: EXP-001
schema_version: 1
updated_at: 2026-09-09
status: completed_reported
evidence: local_report_and_figures
protocol: P-INPUT-GN
---

# EXP-001 — Warm start et étape intermédiaire sur les entrées

[Index](INDEX.md) · Précédent : [EXP-000](EXP-000_input_gaussian.md)

## Question

Le mauvais transfert immédiat d'un modèle flouté ne dit pas ce qu'il vaut après adaptation. À budget total égal, un court préentraînement sigma=1 aide-t-il ? Passer par sigma=0,5 ajoute-t-il un avantage par rapport au saut direct vers zéro ?

## Branches et contrôle de budget

Même profil `P-INPUT-GN`, trois graines et B=14 040 updates.

| Bras | k=0…1499 | k=1500…2999 | k=3000…14039 |
|---|---:|---:|---:|
| A direct | 0 | 0 | 0 |
| W warm start | 1 | 0 | 0 |
| P étape intermédiaire | 1 | 0,5 | 0 |

W et P branchent du **même état complet** après 1 500 updates, par graine. Les trois préfixes ont dû être rejoués car le checkpoint nécessaire manquait. Ils utilisent toujours l'horizon LR **14 040**, pas 1 500. Le LR à la frontière est rapporté à 0,09840385594331022. Aucun reset de momentum ni redémarrage du LR. Les témoins A de la campagne précédente sont réutilisés.

Le nouveau format `full_state_v1` inclut modèle, optimiseur, position globale, RNG et sampler. Le rapport mentionne 83 tests réussis, une reprise bitwise, et neuf vérifications de l'invariance de l'évaluation cible au changement de paramètre sans entraînement : différence maximale 0. Ce sont des contrôles rapportés, pas réexécutés ici.

## Résultats

| Bras | Val accuracy (%) | CE val | CE sonde train cible |
|---|---:|---:|---:|
| A | 84,11 ± 0,60 | 0,6755 ± 0,0223 | 0,0034 |
| W | 82,93 ± 0,11 | 0,7113 ± 0,0252 | 0,0030 |
| P | 83,19 ± 0,51 | 0,7116 ± 0,0283 | 0,0031 |

| Graine | A (%) | W (%) | P (%) |
|---:|---:|---:|---:|
| 0 | 84,64 | 82,86 | 82,60 |
| 1 | 84,22 | 83,06 | 83,52 |
| 2 | 83,46 | 82,88 | 83,46 |

Différences appariées moyennes : W−A **−1,17 point**, SD environ 0,60 ; P−A **−0,91 point**, SD 1,04 ; P−W **+0,26 point**, SD 0,45 et signe variable selon la graine.

À 1 500 updates, la validation cible moyenne est 61,13 % pour A contre 55,57 % pour W/P. Après le retour aux images originales, la récupération est rapide ; aucun bénéfice final sur A n'apparaît au budget global donné. Les franchissements des seuils de CE cible 0,1 et 0,01 diffèrent d'au plus environ une cadence de 500 updates dans les mesures communes ; pas d'accélération robuste résolue par ces sondes.

## Interprétation

La courte phase initiale ne donne pas de gain net ici. L'étape intermédiaire n'a pas d'avantage stable par rapport au saut direct. Aligner les courbes uniquement sur le début du stade final ferait oublier 1 500 ou 3 000 updates déjà consommées et donnerait un avantage de budget artificiel.

Cette expérience ne teste pas l'adaptation de checkpoints grossiers tardifs, ni une loi de continuation interne BatchNorm. Elle ne ferme pas ces questions.

## Coût et artifacts

Calcul nouveau réellement exécuté : 3 préfixes + 6 branches, **79 740 updates**, environ **2 735 s**, dont 151 s pour les préfixes et 2 584 s pour les branches. Le préfixe partagé doit néanmoins compter dans le coût scientifique de chacune des méthodes W et P. Transformation environ 9,1 s, soit 0,3 % du nouveau calcul.

Dossier `results/exp1_gaussian_warmstart/`. Commandes historiques rapportées : `py -m continuation.cli exp1 --config configs/exp1_warmstart.yaml --seeds 0` et graines 1/2 ; génération du rapport `exp1-report`. Ce sont des repères historiques à vérifier dans le dépôt actuel, pas une instruction de relancement.

Sources : [rapport](../sources/exp1_warmstart.md), [prompt](../sources/prompt_exp1_warm_start.md), [budget égal](../sources/exp1_equal_budget.png), [adaptation](../sources/exp1_adaptation.png).

<!-- REPO-EVIDENCE:BEGIN -->

## Traces dans le dépôt (généré — ne pas éditer à la main)

Généré par `tools/enrich_records.py` depuis [`data/repo_evidence.json`](../data/repo_evidence.json). Voir aussi la vue d'ensemble : [REPO_EVIDENCE](../REPO_EVIDENCE.md).

**État du code par rapport à ce run** : branchement full_state_v1 ; code inchangé depuis.

| Rôle | Chemin dans le dépôt | Dernier commit touchant ce fichier |
|---|---|---|
| Implémentation | [`continuation/experiments/exp1.py`](../../../../continuation/experiments/exp1.py) | 86c00bc 2026-09-08 |
| Implémentation | [`continuation/engine.py`](../../../../continuation/engine.py) | 86c00bc 2026-09-08 |
| Configuration | [`configs/exp1_warmstart.yaml`](../../../../configs/exp1_warmstart.yaml) | 86c00bc 2026-09-08 |
| Sorties d'exécution | [`results/exp1_gaussian_warmstart`](../../../../results/exp1_gaussian_warmstart) | 43 fichiers |
| Documentation du dépôt | [`docs/exp1_warmstart.md`](../../../../docs/exp1_warmstart.md) | 86c00bc 2026-09-08 |

Un chemin listé existe dans le dépôt au moment de la génération. Cela ne prouve pas seul quelle version du code a exécuté ce run : pour les runs Kaggle, le code réellement expédié est archivé dans le sous-dossier `_repo/` de la sortie correspondante.

<!-- REPO-EVIDENCE:END -->
