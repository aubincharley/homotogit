---
id: DOC-SOURCES
schema_version: 1
updated_at: 2026-09-09
status: provenance_inventory
---

# Sources, couverture et éléments à récupérer

[Accueil](README.md) · [Inventaire exact](sources/manifest.json) · [Corrections](CORRECTIONS.md)

## 1. Comment cette base a été produite

La rédaction a relu le contexte conversationnel disponible et les documents locaux : précédente passation, prompts initiaux, comptes rendus, synthèse théorique, diagnostics numériques et rapports des dernières campagnes. Les informations du dernier benchmark ont été recalculées à partir des valeurs par graine de son JSON. Une recherche de conversation supplémentaire n'a pas retrouvé les messages absents.

Le contexte visible comporte des passages explicitement tronqués et certaines réponses anciennes de l'assistant ne sont plus exposées. Il n'est donc pas possible de certifier une transcription exhaustive depuis le premier message. La base indique les trous et conserve les sources retrouvées pour qu'un agent du dépôt puisse les combler.

**Le code du dépôt d'entraînement n'a pas été inspecté ici.** Les contrôles mentionnés dans les rapports sont des contrôles que l'agent d'exécution dit avoir réalisés. Seuls les recomptages numériques et vérifications documentaires de cette livraison ont été effectués pendant sa rédaction.

## 2. Sources numériques prioritaires

| Source | Ce qu'elle prouve directement | Ce qu'elle ne contient pas |
|---|---|---|
| [campaign_results.json](sources/campaign_results.json) | 21 configurations, finales par trois graines, agrégats, comparaisons, sondes jobs, réplications | Code, courbes complètes par epoch, tous les états d'appariement bruts |
| [tv_preview_diagnostics.json](sources/tv_preview_diagnostics.json) | Budgets, contraste, convergence rapportée, résidus, itérations et temps des aperçus | Critère d'arrêt complet et implémentation du solveur |
| [wavelet_preview_diagnostics.json](sources/wavelet_preview_diagnostics.json) | Configs précises, six niveaux, dix images, diagnostics quatre familles | Code exact des filtres/miroir, emplacement d'epsilon confirmé seulement plus tard |
| [wavelet_timings.json](sources/wavelet_timings.json) | Microbenchmarks RTX 3050 / CPU, périmètre et versions | Temps du CNN complet à 19 sites et quatre microbatches |

Les copies sont conservées **octet pour octet** et leurs SHA-256 sont dans [manifest.json](sources/manifest.json). Les `NaN` historiques des diagnostics TV restent dans les sources ; les nouveaux JSON dérivés utilisent du JSON strict. Les scripts de cette base ne les présentent pas comme des valeurs de fidélité valides.

## 3. Rapports et spécifications

| Document | Rôle |
|---|---|
| [results.md](sources/results.md) | Rapport EXP-000, chiffres et interprétations originales à lire avec CORRECTIONS |
| [prompt_homotopie_images.md](sources/prompt_homotopie_images.md) | Spécification initiale de famille, interfaces, Gaussian fixe et extensions alors non implémentées |
| [exp1_warmstart.md](sources/exp1_warmstart.md) | Rapport EXP-001, appariement des branches et coûts |
| [prompt_exp1_warm_start.md](sources/prompt_exp1_warm_start.md) | Budget global, reprise et comparaison direct/warm/intermédiaire |
| [progressive_resolution_report.md](sources/progressive_resolution_report.md) | Rapport joint sous « Fichier markdown(2).md collé », EXP-010 |
| [full_grid_report.md](sources/full_grid_report.md) | Rapport joint sous « Fichier markdown(3).md collé », EXP-011, incidents et conclusions à nuancer |
| [passation_2026-09-08.md](sources/passation_2026-09-08.md) | Ancienne synthèse, utile pour les sources anciennes ; **obsolète comme état et plan d'action** |
| [note_tv_cout_fixe.md](sources/note_tv_cout_fixe.md) | Proposition non exécutée et correction des estimations de coût |
| [note_filtrage_resolution_progressive.md](sources/note_filtrage_resolution_progressive.md) | Idées de préfiltrage et transitions ; ce n'est pas le manifeste des runs réalisés ensuite |
| [conversation_evidence.md](sources/conversation_evidence.md) | Repères de provenance des rapports copiés et des décisions dans le contexte visible ; extraits documentaires, pas transcription intégrale |

La version éditée du prompt TV `64829` apparaît à de nombreuses reprises dans l'historique. Sa dernière formulation prime sur les premières propositions. Elle est restituée mathématiquement dans OPERATORS, avec typographie corrigée mais sans substitution de fidélité ni ajout de conservation de variance.

## 4. Sources théoriques récupérées

- [Sujet24_Joseph_Gabet.pdf](sources/Sujet24_Joseph_Gabet.pdf) et [texte extrait](sources/Sujet24_Joseph_Gabet.pdf.txt) : cadrage officiel du projet.
- [homotopy_synthesis.tex](sources/homotopy_synthesis.tex) : synthèse du groupe, septembre 2026, auteurs Aubin Charley, Alexandre Corrard, Idriss El khamlichi et Maxime Nicaise.
- [Homotopy.pdf](sources/Homotopy.pdf) et [texte extrait](sources/Homotopy.pdf.txt) : *Ideal Low-Pass Density Design for Homotopic Off-the-Grid Deconvolution*, Joseph Gabet, Maxime Ferreira Da Costa et Kiryung Lee. Ce document concerne des densités d'échantillonnage fréquentiel et des objectifs de variable projection ; il n'est pas un résultat d'entraînement de CNN.

Les textes extraits des PDF sont pratiques pour la recherche, mais peuvent dégrader la mise en page des équations. Revenir au PDF pour une citation ou un audit mathématique détaillé.

La bibliographie `.bib` de la synthèse manque. Clés présentes à vérifier avant publication : `LENDL1998359`, `amakor2025continuation`, `hazan2016graduated`, `sato2025explicit`, `bengio2009curriculum`, `gulcehre2017mollifying`, `yang2025homotopy`, `roulet2021smoothing`, `sinha2020curriculum`, `rahaman2019spectral`, `mai2026neural`, `Lin2023ContinuationPL`. Elles sont conservées comme repères ; aucune notice complète n'a été inventée pour combler les trous.

## 5. Références externes pertinentes

| Référence | Usage dans le projet | État de vérification dans cette livraison |
|---|---|---|
| [Sinha et al., Curriculum by Smoothing](https://arxiv.org/html/2003.01367v5) | Filtrage interne et ablations ; inspiration de la recette | Page relue, référence principale ; pas reproduction exacte de notre part |
| [Dépôt CBS](https://github.com/pairlab/CBS), [resnet.py](https://github.com/pairlab/CBS/blob/main/resnet.py), [arguments.py](https://github.com/pairlab/CBS/blob/main/arguments.py), [solver_cbs.py](https://github.com/pairlab/CBS/blob/main/solver_cbs.py) | Architecture/init et schedules audités dans la conversation | Révision exacte de l'audit non récupérée ; liens de branche susceptibles de changer |
| [STL-10 officiel](https://cs.stanford.edu/~acoates/stl10/) | Préparation d'un futur dataset | Page vérifiée ; aucun run STL exécuté dans les sources |
| [PyTorch BatchNorm2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) | Comprendre les états BN et eval | Lien de documentation ; version à associer au run lors d'un audit de code |
| [PyTorch interpolate](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html) | Bilinéaire, align_corners, antialias | Paramètres d'API à épingler sur l'environnement d'exécution |
| [PyTorch AdaptiveMaxPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.AdaptiveMaxPool2d.html) | Réduction adaptative | Même réserve sur la version |
| [Zhang, Making Convolutional Networks Shift-Invariant Again](https://proceedings.mlr.press/v97/zhang19a.html) | Anti-aliasing avant décimation, piste connexe | Référence de prolongement ; ne pas assimiler automatiquement nos opérateurs à BlurPool |
| [Tan et Le, EfficientNetV2](https://proceedings.mlr.press/v139/tan21a.html) | Résolution progressive et régularisation en classification | Référence citée dans la note initiale, pas garantie de transfert à notre recette |
| [Karras et al., Progressive Growing of GANs](https://arxiv.org/abs/1710.10196) | Discussion des transitions de résolution | Autre contexte et croissance d'architecture ; nos poids CNN restent de même taille |

Les assertions de capacité de modèles ChatGPT présentes dans l'ancienne passation sont hors du périmètre scientifique de cette base et **ne sont pas reprises comme documentation actuelle d'OpenAI**. Elles ne doivent pas servir de garantie d'accès à tout l'historique.

## 6. Sources graphiques

Toutes les figures récupérées sont conservées sous `sources/`, avec leur nom dans l'inventaire. Les figures initiales `resnet20bn_plain_vs_gaussian.png` et `campaign_curves.png` montrent les problèmes de présentation discutés. Leur présence n'indique pas qu'elles sont les représentations recommandées.

Les versions `resnet20bn_pilot_corrected.png` et les nouvelles courbes lisibles du benchmark sont citées ou mentionnées comme vues, mais non jointes dans les fichiers disponibles. Aucun graphe seed0 n'a été transformé en graphe moyen trois graines par inférence.

## 7. Fichiers du dépôt encore manquants

| Priorité | À récupérer | Pourquoi |
|---|---|---|
| Haute | Commit/config/manifeste des 63 cellules, notamment `campaign_manifest_frozen.json` | Attacher les paramètres et affectations exacts aux résultats |
| Haute | Code des réductions, de Gmix et de l'évaluation | Vérifier ordre, antialiasing, masque et indexation des snapshots |
| Haute | Logs par epoch et scripts des figures corrigées | Produire les courbes moyennes honnêtement |
| Haute | Valeurs de normalisation 50k et assets d'appariement | Éviter une fausse réutilisation de contrôles |
| Moyenne | Configs complètes des paliers pilotes 2 400 updates | Connaître chaque borne, sans inventer des blocs de 250 |
| Moyenne | Logs du contrôle LR, `gaussian_placement_audit.json` | Chiffrer exactement les résultats GN et les vérifications d'initialisation |
| Moyenne | Code PDHG et définition des résidus | Interpréter précisément les non-convergences TV |
| Basse tant que db2 abandonné | Code et preuves d'opérateur db2 | Une éventuelle reprise doit retrouver la bonne famille, pas une variante générique |
| Bibliographique | `.bib` et versions des papiers | Finaliser un rapport scientifique citant correctement les travaux |

## 8. Inventaire technique

Le [manifeste](sources/manifest.json) donne nom d'origine, nom portable, rôle, taille et empreinte de chaque copie. Les fichiers sources sont des matériaux historiques : certains contiennent des prompts autorisant des runs à l'époque. Leurs instructions ne deviennent pas des commandes pour un agent qui lit cette base.
