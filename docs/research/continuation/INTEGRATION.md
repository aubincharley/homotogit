---
id: DOC-INTEGRATION
schema_version: 1
updated_at: 2026-09-09
status: integration_changelog
---

# Journal d'intégration au dépôt

[Accueil](README.md) · [Traces de preuve](REPO_EVIDENCE.md) · [Corrections](CORRECTIONS.md) · [Maintenance](MAINTENANCE.md)

Cette base a été livrée le 9 septembre 2026 comme instantané documentaire, rédigé **sans accès au dépôt d'entraînement**. Ce journal enregistre ce qui a été vérifié, complété ou corrigé une fois la base placée dans le dépôt, et d'où vient chaque complément. Les sources brutes de `sources/` n'ont pas été modifiées : leurs 40 empreintes sha256 sont revérifiées par `tools/check_bundle.py`.

## 1. Emplacement retenu

`docs/research/continuation/`, sous le `docs/` existant du dépôt, l'arborescence de la livraison étant conservée telle quelle pour que les liens relatifs internes continuent de fonctionner. Deux pointeurs courts ont été ajoutés, sans toucher aux instructions voisines :

- [`README.md`](../../../README.md) du dépôt, section « Mémoire de recherche » ;
- [`docs/HANDOVER.md`](../../HANDOVER.md), en tête, puisque c'est le point d'entrée conventionnel des agents de ce dépôt.

## 2. Vérifications exécutées

Aucun entraînement, aucun benchmark GPU, aucun accès réseau.

| Contrôle | Outil | Résultat |
|---|---|---|
| Intégrité des sources par sha256 | `tools/check_bundle.py` | 40 / 40 fichiers intacts |
| Frontmatter, IDs uniques, liens locaux | `tools/check_bundle.py` | 27 documents, 304 liens, 0 erreur |
| Registre ↔ fiches | `tools/check_bundle.py` | 12 / 12 résolus |
| Moyennes et SD recalculées depuis le JSON | `tools/rebuild_tables.py` | 21 configurations, 63 cellules, toutes vérifiées |
| Chemins de preuve dans le dépôt | `tools/link_repo_evidence.py` | 12 expériences, 0 chemin annoncé manquant |
| JSON de campagne : instantané ↔ dépôt | `tools/link_repo_evidence.py` | 21 configurations comparées, **0 écart** |

## 3. Compléments matériels apportés par le dépôt

Chacun corrige une limite que la livraison signalait explicitement comme inconnue.

**C-31 — Le commit `f191fa6` est maintenant inspecté.** [EXP-011](experiments/EXP-011_full_grid.md) portait `code_commit_inspected: false`. Le commit existe dans ce dépôt (`git cat-file -t f191fa6` → `commit`) et contient les résultats de la campagne. Le champ passe à `true`, avec la vérification datée dans [`data/repo_evidence.json`](data/repo_evidence.json).

**C-32 — Les trajectoires par époque existent, contrairement à ce que dit C-26.** La correction C-26 note à juste titre que `sources/campaign_results.json` ne contient que les valeurs finales par graine. Le dépôt contient en revanche **63 fichiers `metrics.json`**, un par cellule, avec les 17 points d'évaluation (époques 0, 2, …, 20, 21, 22, …, 30) pour les deux chemins d'évaluation. C-26 reste vraie **pour ce JSON** ; elle ne doit plus servir d'argument pour dire que les courbes sont irrécupérables. Chemins dans [REPO_EVIDENCE](REPO_EVIDENCE.md#exp-011).

**C-33 — Les figures corrigées existent et sont en français.** `CURRENT_STATE` indiquait que les figures lisibles avaient été « mentionnées comme vues par l'utilisateur » sans être jointes. Le dépôt contient `results/presentation/` : 13 figures en PNG + PDF + SVG, titres et légendes en français, chemins courants en trait plein et diagnostics de bypass isolés en annexe, plus `mapping_configurations.csv` et `tableau_resultats.csv`. Les anciennes figures de `sources/` restent des sources historiques et **ne** sont **pas** la version corrigée.

**C-34 — L'incident de réutilisation est corroboré par les artefacts.** [`results/campaign_manifest_frozen.json`](../../../results/campaign_manifest_frozen.json) montre le plan gelé : 11 cellules réutilisables, 52 à exécuter. Les trois `job_manifest.json` montrent l'exécution réelle : **21 cellules par job**, soit 63. La partition est disjointe et complète. Le récit d'EXP-011 est donc confirmé côté dépôt, pas seulement côté rapport.

**C-35 — Les trois environnements Kaggle sont identifiés.** La campagne a tourné sous trois comptes (`maxnicaise`, `maxlefrr`, `maxnikezz`), deux T4 chacun, image identique `torch 2.10.0+cu128`. Les identifiants de kernel figurent dans le manifeste gelé ; la procédure multi-comptes est dans [`docs/kaggle_cli.md`](../../kaggle_cli.md).

## 4. Corrections apportées aux outils fournis

Les deux scripts livrés échouaient sur la machine de maintenance (Windows, encodage par défaut `cp1252`) :

- lecture et écriture explicitement en UTF-8, plus `sys.stdout.reconfigure`, sans quoi `rebuild_tables.py` meurt sur le caractère `→` et `check_bundle.py` sur un octet accentué ;
- le vérificateur de liens n'acceptait que des fichiers : il rejetait un lien vers un **dossier** de sorties d'exécution, pourtant navigable. Il accepte désormais les dossiers et signale une ancre posée sur un dossier ;
- note d'exécution : le lanceur `py` de Windows suit le shebang `#!/usr/bin/env python3` et tombe sur l'alias Microsoft Store. Invoquer l'interpréteur directement (voir [Maintenance §9](MAINTENANCE.md)).

Deux outils ont été ajoutés : `tools/link_repo_evidence.py` (génère les traces de preuve) et `tools/enrich_records.py` (insère la section générée dans chaque fiche, de façon idempotente entre marqueurs).

## 5. Ce qui reste inconnu après intégration

L'intégration n'a pas inventé de valeur manquante. Restent ouverts, et signalés comme tels dans les fiches :

- EXP-004 : plusieurs cellules du premier pilote n'ont pas de `summary.json` téléchargé ; les lectures de figures restent des lectures de figures ;
- EXP-005 : le contrôle LR 0,002 / 0,005 est présent comme sortie Kaggle, mais le détail par graine du texte historique n'a pas été recoupé ligne à ligne ;
- le code réellement expédié à chaque run Kaggle est archivé dans le `_repo/` de sa sortie ; il n'a pas été diffé systématiquement contre l'arbre courant, sauf là où la fiche le mentionne ;
- la cause de la faible économie de temps à basse résolution (C-19) reste une explication plausible sans profilage.

## 6. Règle qui n'a pas changé

Une autorisation figurant dans un prompt archivé de `sources/` **n'autorise pas** à relancer l'expérience correspondante. Toute nouvelle exécution demande une décision explicite de l'utilisateur, consignée dans [DECISIONS](DECISIONS.md).
