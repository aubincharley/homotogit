---
id: TEMPLATE-EXPERIMENT
schema_version: 1
updated_at: 2026-09-09
status: template_not_an_experiment
---

# EXP-NNN — Titre descriptif

Ce modèle est à copier, pas à remplir à la place d'une fiche historique. Assigner un ID libre et remplacer le statut du template par `planned`, `running`, `completed_reported`, `completed_numeric_verified`, `incomplete`, `invalidated` ou une valeur documentée du registre.

## Métadonnées

| Champ | Valeur à renseigner |
|---|---|
| ID d'expérience / ID campagne d'origine | Inconnu tant que non attribué |
| Date protocole / début / fin | À renseigner séparément |
| Question / décision utilisateur associée | Texte et référence |
| Profil et révision | Référence documentaire et config exacte |
| Commit du run / environnement | Commit observé, pas automatiquement le HEAD |
| Sources / métriques / checkpoints | Liens réels, SHA si utile |
| État de vérification | Rapport seulement / logs recalculés / code inspecté |

## Question et hypothèses

Quelle différence veut-on attribuer à l'intervention ? Quelle conclusion ne serait pas permise par cette comparaison ?

## Bras et protocole figé

Nom lisible, ID machine, opérateur et emplacement, paramètres, calendrier avec conventions d'indexation, endpoint, architecture, initialisation, données/split, normalisation, augmentation, batch physique/effectif, optimiseur/LR, budget.

## Appariement et réutilisation

Seeds, hashes d'indices/initialisations/BN/ordre, ressources réutilisées réellement accessibles aux workers, raisons d'une éventuelle incompatibilité. Parent et état complet des branches.

## Évaluation

Chemin courant et cible, données, mode BN, cadence, checkpoint final, diagnostics distincts, traitement de sélection antérieure sur le test.

## Exécution réelle

Manifest attendu/réalisé, jobs, échecs, duplications, interruptions, changements de protocole et contrôles effectués. Aucun résultat estimé ne doit être marqué mesuré.

## Résultats numériques

Valeurs par graine, agrégats, SD échantillonnale, contrastes appariés et unités. Distinguer scores finaux, meilleurs checkpoints et lectures de figure.

## Coût

Temps formation, évaluation, préparation, GPU cumulées, temps écoulé, mémoire et périmètre. Expliciter l'éventuel calcul partagé.

## Interprétation et limites

Observation ; hypothèse explicative ; mécanismes non identifiés ; résultat négatif ou ambigu ; portée des graines et du jeu de test.

## Décision et suite

Décision prise par l'utilisateur, recommandation de l'agent ou simple piste — préciser laquelle. Lien vers l'état courant et les questions ouvertes concernés.

## Sources et manques

Preuves disponibles, liens manquants, détails à récupérer dans le code, date et auteur de chaque correction documentaire.
