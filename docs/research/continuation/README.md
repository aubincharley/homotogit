---
id: DOC-README
schema_version: 1
updated_at: 2026-09-09
status: current_at_snapshot
---

# Mémoire du projet — continuation, filtrage et résolution

**Point d'entrée pour un agent reprenant le projet de Maxime Nicaise.** Cette base restitue l'approche scientifique, les expériences, leurs résultats et les corrections apportées aux interprétations. Elle couvre les sources accessibles jusqu'au **9 septembre 2026**, après le benchmark CIFAR-10 de **21 configurations × 3 graines = 63 entraînements** et la discussion du préfloutage avant réduction de résolution.

Le résultat de référence le plus récent est un ResNet-20 BatchNorm : **80,71 %** de test accuracy pour résolution progressive bilinéaire avec antialiasing + Gaussian interne, contre **75,62 %** pour son témoin de la même campagne. Ce sont des résultats exploratoires à 30 époques, sans augmentation. La piste db2 a été abandonnée par l'utilisateur après son pilote coûteux ; STL-10 et le préfloutage RGB avant réduction n'ont pas encore été expérimentés dans les sources disponibles.

## Sommaire et parcours de lecture

| Besoin | Lire dans cet ordre |
|---|---|
| Reprendre la discussion en quelques minutes | [État actuel](CURRENT_STATE.md), puis [questions ouvertes](OPEN_QUESTIONS.md) |
| Comprendre pourquoi nous sommes arrivés ici | [Approche scientifique](SCIENTIFIC_APPROACH.md), puis [chronologie des expériences](experiments/INDEX.md) |
| Retrouver une formule ou un emplacement dans le réseau | [Opérateurs](OPERATORS.md), puis [protocoles](PROTOCOLS.md) |
| Analyser le benchmark récent | [EXP-011](experiments/EXP-011_full_grid.md), [tableau complet](data/campaign_tables.md), [JSON source](sources/campaign_results.json) |
| Comprendre active/bypassed, les figures et les statistiques | [Évaluation et présentation](EVALUATION.md) |
| Savoir ce qui a été abandonné ou remplacé | [Décisions](DECISIONS.md), [corrections](CORRECTIONS.md) |
| Intégrer ou actualiser cette base dans le dépôt | [Maintenance](MAINTENANCE.md), [modèle de fiche](templates/EXPERIMENT_TEMPLATE.md) |
| Vérifier l'origine d'une information | [Sources](SOURCES.md), [inventaire avec empreintes](sources/manifest.json) |
| Retrouver le code, la configuration ou les sorties d'un run | [Traces de preuve dans le dépôt](REPO_EVIDENCE.md) |
| Savoir ce qui a été vérifié ou complété à l'intégration | [Journal d'intégration](INTEGRATION.md) |

### Lecture par expérience

L'[index des expériences](experiments/INDEX.md) donne pour chaque fiche la question, le protocole, la qualité des preuves et le résultat. Les identifiants `EXP-000` à `EXP-011` sont des identifiants documentaires créés pour cette passation. Ils ne remplacent pas les noms originaux des runs ; les correspondances sont conservées.

## Ce que contient la livraison

- Des documents Markdown organisés par rôle, avec liens relatifs portables.
- Douze fiches d'expérience, des premiers essais sur les entrées à la grille complète.
- Le JSON du benchmark et des tableaux recalculés à partir des valeurs par graine.
- Les rapports, prompts et figures récupérés, conservés comme sources historiques.
- Un registre structuré des expériences et un script de vérification documentaire, sans entraînement.

**Cette base est intégrée au dépôt depuis le 9 septembre 2026**, sous `docs/research/continuation/`. Les chemins `results/...` cités par les fiches ont été résolus contre le dépôt : voir [REPO_EVIDENCE](REPO_EVIDENCE.md) pour les liens vérifiés et [INTEGRATION](INTEGRATION.md) pour ce qui a été complété ou corrigé à cette occasion.

## Niveau de confiance et limites de couverture

Cette base **n'est pas une transcription intégrale garantie de tous les messages depuis le premier jour**. L'historique visible contient des coupures ; une recherche d'historique supplémentaire n'a pas retrouvé les messages manquants. La restitution s'appuie sur les échanges présents, la précédente passation, les rapports et les fichiers effectivement récupérés. Le dépôt d'entraînement, ses checkpoints et la plupart des logs par époque n'étaient pas accessibles pendant cette rédaction.

Les nombres de la dernière grille ont pu être recalculés depuis son JSON. Pour plusieurs expériences antérieures, les preuves disponibles sont des rapports d'exécution ; pour deux anciens pilotes, certaines informations ne sont que des lectures de figures. Chaque fiche le précise. **Un détail manquant reste inconnu ; il n'est pas complété par une hypothèse silencieuse.**

## Règles de lecture essentielles

1. Une proposition dans une source ancienne n'est pas une expérience exécutée ni une autorisation actuelle.
2. `CURRENT_STATE.md` décrit l'état au jour indiqué. Après une nouvelle expérience, il faut l'actualiser ; les fiches historiques gardent leurs protocoles d'origine.
3. La baseline dépend de la campagne. La nouvelle baseline à 75,62 % ne remplace pas les résultats historiques à 75,40 % ou 55,36 %.
4. Gaussian sur les images, Gaussian interne et antialiasing de redimensionnement sont trois interventions distinctes.
5. Le test CIFAR-10 a déjà été regardé à plusieurs reprises. Le classement de la grille est exploratoire, pas une confirmation indépendante du gagnant.

## Texte d'entrée pour l'agent du dépôt

> Lis README, CURRENT_STATE, OPERATORS, PROTOCOLS et la fiche EXP-011. Utilise ensuite l'index pour retrouver les expériences pertinentes. Intègre cette base dans le dépôt sans réécrire les historiques pour les faire correspondre au code actuel. Complète les inconnues avec les configurations, commits et logs des runs concernés ; indique la source de chaque complément. Vérifie en priorité les emplacements de réduction, les coefficients d'antialiasing, les calendriers effectifs et les empreintes d'appariement. Ne lance aucun entraînement sur la seule base de cette documentation. Les décisions nouvelles de l'utilisateur priment sur l'état daté de cette passation.

## Pourquoi Markdown plutôt qu'un skill qui mémorise les résultats ?

Les résultats sont des connaissances versionnées, pas une procédure à exécuter. Un skill qui les recopierait deviendrait une deuxième source à maintenir. La base fournit donc d'abord les faits, les sources et les règles de mise à jour. Un futur skill pourra automatiser **l'ajout et la vérification d'une expérience**, en lisant ces documents au lieu d'embarquer leurs conclusions. Aucun skill n'est installé par cette livraison ; voir [Maintenance](MAINTENANCE.md).
