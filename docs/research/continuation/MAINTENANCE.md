---
id: DOC-MAINTENANCE
schema_version: 1
updated_at: 2026-09-09
status: maintenance_workflow
---

# Intégration au dépôt et maintien d'une mémoire fiable

[Accueil](README.md) · [Modèle de fiche](templates/EXPERIMENT_TEMPLATE.md) · [Registre](data/experiments.json)

## 1. Objectif de cette organisation

Éviter qu'un agent relise tout l'historique à chaque question, tout en lui permettant de remonter aux preuves. Les faits durables, l'état actuel, les résultats d'une exécution et les propositions ont des rôles séparés. Une documentation devenue ancienne doit rester utile comme histoire, sans se faire passer pour une instruction courante.

Cette base constitue un **instantané daté**, à intégrer au projet. Elle n'est pas un dépôt d'entraînement autonome et ne contient pas tous les fichiers nécessaires pour reproduire chaque expérience.

## 2. Autorité de chaque type de document

| Objet | Rôle | Peut changer comment ? |
|---|---|---|
| Sources originales | Conserver les preuves reçues | Ajouter une nouvelle version/source, ne pas réécrire les nombres anciens |
| Fiche EXP | Décrire une campagne, ses résultats et limites | Ajouter preuves/corrections sourcées ; conserver le protocole historique |
| Registre machine | Naviguer vers les fiches et leurs états | Mettre à jour en même temps que les fiches |
| OPERATORS / PROTOCOLS | Dictionnaire des variantes nommées | Versionner une définition qui change ; ne pas écraser la variante historique |
| CURRENT_STATE | Dire où nous en sommes au jour indiqué | Mettre à jour après résultat, décision ou invalidation |
| DECISIONS / CORRECTIONS | Expliquer les changements | Ajouter une entrée avec lien vers ce qu'elle remplace |
| OPEN_QUESTIONS | Réserver les pistes non exécutées | Promouvoir une question vers une fiche lorsqu'un protocole puis un run existent |
| README | Router la lecture | Garder court et cohérent avec l'état actuel |

Une date de modification seule ne prouve pas une revalidation scientifique. Distinguer `updated_at`, date d'exécution, date de consultation de la source et commit du run. Une valeur inconnue reste `null` ou « non récupéré », pas un identifiant plausible inventé.

## 3. Première intégration — faite le 9 septembre 2026

Cette section décrivait la marche à suivre ; elle est **exécutée**. Emplacement
retenu : `docs/research/continuation/` dans le dépôt `Projet_filiere`. Le compte
rendu, les compléments et les corrections figurent dans
[INTEGRATION](INTEGRATION.md), les chemins vérifiés dans
[REPO_EVIDENCE](REPO_EVIDENCE.md). La procédure d'origine est conservée
ci-dessous comme référence pour une future ré-intégration.

### Procédure d'origine (historique)

1. Lire le README du dépôt et ses règles existantes, puis README/CURRENT_STATE de ce dossier.
2. Choisir un emplacement unique, par exemple `docs/research/continuation/`, en conservant d'abord les liens relatifs. Ce chemin est une proposition d'organisation, pas un chemin observé.
3. Comparer les sources locales du dépôt avec les rapports et JSON de cette livraison. Ajouter les chemins réels et commits de run aux fiches, sans recopier automatiquement le HEAD actuel.
4. Récupérer en priorité les configurations de la grille, les manifestes, la normalisation 50k, le code des réductions et Gmix, les logs d'epochs et les figures corrigées.
5. Compléter les historiques partiels : anciens pilotes GN, contrôle LR 0,002/0,005, bornes des paliers 2 400 updates, code db2, critère PDHG et résidus.
6. Signaler tout écart entre protocole demandé et code réellement exécuté ; une correction d'un texte ne justifie pas de modifier les artifacts originaux.
7. Vérifier les liens et les tableaux. Inscrire le complément dans un changelog avec source, périmètre et éventuel impact sur une conclusion.

Pas de nouveau training nécessaire pour ces opérations. Ne pas relancer une expérience uniquement parce qu'un script de réutilisation ne voit pas son ancien dossier : inventorier d'abord les sorties et checkpoints existants.

## 4. Ajouter une expérience future

### Avant l'exécution

Créer une nouvelle fiche depuis le template avec un ID stable et statut `planned`. Définir la question, les bras, les endpoints, données et états partagés, budget en updates et en temps, chemins d'évaluation et métrique finale. Séparer les variantes du protocole de base. Inscrire la décision utilisateur correspondante quand elle existe ; ne pas interpréter la présence d'une fiche planned comme une autorisation.

Geler la liste de cellules et la table de calendriers. Pour la réutilisation, préparer **un manifeste explicite** des cellules réutilisées et de leurs assets disponibles dans l'environnement de destination. Ne pas laisser chaque worker redécouvrir la liste à partir de dossiers différents. Vérifier couverture et absence de chevauchement des partitions avant de lancer.

### Pendant l'exécution

Conserver la liste des jobs, états, erreurs et reprises. Distinguer absence de téléchargement et échec de formation. Sauvegarder les états suffisants à la reprise et le parent d'un branchement. Si le plan change à cause d'un bug ou d'une limite mesurée, noter le changement et les runs affectés.

### Après l'exécution

Mettre à jour, dans cet ordre, en une seule passe :

1. `data/experiments.json` — l'entrée du registre passe de `planned` à
   `completed_*`, avec `code_commit`, cellules attendues et complétées.
2. `experiments/EXP-xxx_*.md` — la fiche reçoit ses résultats et ses limites ;
   le protocole historique n'est pas réécrit.
3. `tools/link_repo_evidence.py` — ajouter l'entrée du nouvel `EXP-xxx` au
   dictionnaire `EVIDENCE` (implémentation, configuration, sorties, docs, et
   surtout `code_state`), puis exécuter les quatre outils du §9.
4. `CURRENT_STATE.md` — quelques chiffres d'orientation seulement, jamais la
   table complète : la source canonique reste le JSON du run.
5. `experiments/INDEX.md`, `DECISIONS.md`, `OPEN_QUESTIONS.md` — une question
   tranchée garde un lien vers l'expérience qui l'a tranchée.
6. `INTEGRATION.md` — une entrée `C-xx` par correction matérielle, avec sa source.

Les tableaux de `data/` sont **générés** : les régénérer plutôt que les éditer.

Calculer les métriques depuis les logs : seeds réellement terminées, checkpoint prescrit, moyennes/SD, différences appariées, temps et mémoire. Produire un résumé qui renvoie aux fichiers bruts. Un run incomplet ne disparaît pas d'une moyenne sans signalement ; ne pas compléter une graine par une valeur historique incompatible.

Ajouter une conclusion limitée à la question. Actualiser CURRENT_STATE, l'index, le registre et les décisions concernées. Une question résolue garde un lien vers l'expérience qui l'a tranchée ; ne pas effacer la question initiale.

## 5. Schéma de données conseillé

Le registre livré est volontairement simple. À compléter dans le dépôt avec des clés telles que :

```json
{
  "experiment_id": "EXP-next",
  "status": "planned",
  "protocol_id": "P-FULL-R20-BN",
  "protocol_revision": null,
  "code_commit": null,
  "source_ids": [],
  "seed_ids": [],
  "expected_cells": null,
  "completed_cells": null,
  "dataset_split_hash": null,
  "initialization_hashes": {},
  "schedule_manifest": null,
  "evaluation_path": null,
  "final_checkpoint_rule": null,
  "supersedes": [],
  "notes": "Proposition sans résultat ; null signifie non renseigné."
}
```

Ce bloc est un **exemple de structure**, pas un manifeste de run validé. Les tableaux dérivés doivent contenir les unités : accuracy fraction ou pourcentage, différences en points, secondes murales ou secondes GPU cumulées, mémoire MiB. Ne pas écrire `NaN` dans de nouveaux JSON stricts ; utiliser `null` et une raison.

## 6. Éviter les contradictions entre documents

Une même métrique a une source canonique : le JSON/log du run. La fiche la résume ; CURRENT_STATE en reprend quelques chiffres pour orienter. Les tableaux exhaustifs sont générés. Lors d'une correction numérique, régénérer les dérivés puis vérifier les quelques résumés narratifs concernés.

Un nouveau résultat à la même configuration reçoit une identité d'exécution différente. Il ne remplace pas silencieusement l'ancien score. Les réplications peuvent être comparées, et éventuellement agrégées si le protocole d'agrégation le justifie explicitement ; elles ne sont pas automatiquement de nouvelles graines indépendantes.

Si une instruction scientifique change (ex. conservation de moyenne TV, sigma adaptatif, emplacement du filtre), créer une variante ou une révision. Ne pas affirmer rétroactivement que les anciens runs utilisaient la nouvelle formule.

## 7. Fraîcheur : déclencheurs concrets

L'état courant devient à réviser quand : une campagne finit, une cellule est invalidée, l'utilisateur abandonne ou reprend une piste, un bug change le sens d'un opérateur, un dataset change de rôle train/validation/test, un résultat de sélection change les candidates, ou une source plus précise remplace une lecture de figure.

Un agent qui revient plus tard doit lire la date et chercher les entrées plus récentes dans le dépôt. Il ne doit pas prolonger le statut `current_at_snapshot` comme s'il s'agissait d'une garantie permanente. Les résultats historiques restent valides comme observations datées, sauf correction documentée.

## 8. Faut-il créer un skill ?

**Pas besoin d'un skill pour stocker les faits.** Cette base Markdown et son registre sont la première couche. Un petit pointeur depuis le fichier d'accueil des agents du dépôt peut suffire à imposer la lecture du bon index sans charger tous les résultats.

Un skill devient utile si l'équipe répète souvent une procédure : « documenter un nouveau run », « vérifier l'appariement », « préparer un rapport sans refaire de training ». Il devrait alors : lire les documents actuels ; découvrir les logs du dépôt ; appliquer un schéma ; recalculer les agrégats ; vérifier sources et liens ; préparer des changements documentaires reviewables. Il ne devrait pas embarquer le classement des méthodes ou les tokens d'accès Kaggle.

Sa création doit attendre les vrais chemins, formats et commandes du dépôt. La présente livraison fournit une **procédure de maintenance**, pas un `SKILL.md` installé, ni un `AGENTS.md` qui modifierait automatiquement la conduite des autres agents. L'agent du dépôt peut ensuite créer ce wrapper selon ses conventions et le versionner avec son code.

## 9. Vérification documentaire — commandes réelles

Depuis `docs/research/continuation/` :

```bash
python tools/rebuild_tables.py
python tools/link_repo_evidence.py
python tools/enrich_records.py
python tools/check_bundle.py
```

Sur la machine Windows de maintenance, **ne pas utiliser le lanceur `py`** pour
ces scripts : il suit le shebang `#!/usr/bin/env python3` et tombe sur l'alias
Microsoft Store. Invoquer l'interpréteur directement, par exemple
`C:/Users/<user>/AppData/Local/Programs/Python/Python312/python.exe`.

L'ordre compte : `link_repo_evidence.py` régénère `data/repo_evidence.json`,
qu'`enrich_records.py` insère ensuite dans les fiches, et `check_bundle.py`
valide le tout en dernier.

Le premier script ne fait que recalculer les tableaux et contrastes depuis le JSON de la campagne. Le second vérifie les sources par empreinte, la cohérence numérique, les IDs et liens locaux de la documentation éditoriale. Ils utilisent la bibliothèque standard et ne lancent ni entraînement ni téléchargement.

Le contrôle des liens ne certifie pas que les liens web sont encore accessibles ni que les chemins externes `results/...` existent. Le contrôle de chiffres ne certifie pas le code scientifique qui les a produits. Une nouvelle expérience demandera d'étendre les entrées et, au besoin, les vérifications correspondantes.
