---
id: DOC-EVAL
schema_version: 1
updated_at: 2026-09-09
status: current_conventions
---

# Évaluation, statistiques et présentation lisible

[Accueil](README.md) · [Protocoles](PROTOCOLS.md) · [Corrections](CORRECTIONS.md)

## 1. Trois objets à ne jamais confondre

Soient les poids \(\theta\), les buffers BN \(b\), la résolution \(r\) et les paramètres de filtre \(\eta\) au checkpoint.

| Nom | Fonction évaluée | Question |
|---|---|---|
| **Chemin courant / current / active** | \(f_{\theta,b;r,\eta}\) | Que vaut le modèle dans la configuration qu'il vient d'apprendre ? |
| **Chemin cible / target / bypassed** | \(f_{\theta,b;32,0}\), ou endpoint adapté à la famille | Que se passe-t-il si on le remet maintenant en configuration finale sans adapter ses états ? |
| **Témoin / plain / baseline** | Autres poids appris sans continuation | Que donne l'entraînement direct sous le même budget ? |

Le modèle Gaussian bypassed n'est pas le modèle plain. Les paramètres de la cible sont sigma=0 pour Gaussian, alpha=0 pour Gmix, s=1 pour db2 et r=32 pour la résolution. La notation générique « filtre 0 » doit être adaptée à la famille.

Dans les anciennes expériences d'entrée à résolution constante, les libellés `transformed_*` et `target_*` jouent un rôle analogue : images au niveau d'entraînement contre images originales aux mêmes poids. GroupNorm n'a pas de running statistics, mais le changement de distribution existe tout de même.

## 2. Ce que fait BatchNorm dans ces diagnostics

Toutes les évaluations décrites utilisent `eval()`, sans mise à jour ni recalibration des statistiques BN, puis restaurent le mode précédent. Les moyennes et variances mémorisées proviennent de la trajectoire d'entraînement. Retirer des filtres ou changer la résolution modifie les activations auxquelles elles sont appliquées.

Un très mauvais score cible prématuré est donc compatible avec un décalage BN important. Mais le changement modifie aussi directement la fonction et les caractéristiques. **Il n'existe pas ici de contrôle qui attribue 100 % de l'écart à BatchNorm.** Un tel diagnostic exigerait, par exemple, une recalibration à poids fixés sur des données d'entraînement, évaluée séparément et clairement nommée. Il n'a pas été rapporté comme exécuté.

À la fin, toutes les branches de la grille sont revenues à 32×32 sans filtre pendant neuf époques. La comparaison finale n'est donc pas celle d'un retrait improvisé d'opérateur. Current et target doivent coïncider pour le même checkpoint et les mêmes exemples, sous réserve de la reproductibilité numérique de l'évaluation.

## 3. Checkpoint après une étape : ancien état et prochaine configuration

« Après 1 700 updates » signifie que k=0…1699 ont été effectuées. La dernière configuration entraînée peut encore avoir sigma 0,30 ou s=0,99, tandis que la prochaine update k=1700 utilisera le bypass.

De même, après 12 époques accomplies de `Rprog`, la dernière époque zéro-indexée était e=11 à r=24 ; la prochaine sera e=12 à r=32. Après 21 époques, le dernier sigma de paliers reste 0,30, puis e=21 commence la phase sans filtre.

Un enregistrement robuste contient : `completed_updates`, `completed_epochs`, `last_trained_resolution`, `last_trained_parameter`, `next_resolution`, `next_parameter`, et le chemin exact de chaque métrique. Les libellés vagues « active » et « filters off » ne suffisent pas à lever une ambiguïté d'indexation.

## 4. Métriques à garder distinctes

- **Minibatch training CE** : perte en mode entraînement sur les exemples de l'update ; dépend du batch et des statistiques BN du passage.
- **Training-probe CE/accuracy** : évaluation d'une sonde fixe du train en mode évaluation ; ce n'est pas la perte de tout le train.
- **Validation CE/accuracy** : mesure sur les 5 000 images exclues de l'entraînement dans les anciens profils.
- **Test CE/accuracy** : mesure sur les 10 000 images officielles dans les campagnes complètes ; déjà consultées pendant l'exploration.

Afficher la CE sans le terme de weight decay, comme dans les rapports. Préciser si l'axe CE est logarithmique ; une droite ou un petit écart graphique n'a pas la même interprétation selon l'échelle. Ne pas déduire une calibration probabiliste de la seule CE.

## 5. Statistiques avec trois graines

Pour chaque configuration, garder les valeurs par graine puis calculer la moyenne et la SD **échantillonnale** (dénominateur n−1). Pour comparer A et B, former d'abord les différences appariées \(d_s=a_s-b_s\), puis leur moyenne et leur SD.

La SD des performances d'un bras n'est ni la SD de la différence, ni une erreur standard, ni un intervalle de confiance. Les trois graines donnent une variabilité descriptive dans cette recette. Le nombre d'images test et le nombre de graines sont deux dimensions d'incertitude différentes.

Les accuracies brutes du JSON sont des fractions. Multiplier les différences par 100 pour obtenir des **points de pourcentage** : 0,0094 = 0,94 point, pas 0,94 % relatif d'amélioration.

Ne pas présenter « gain cinq fois plus grand que sa SD » comme un test statistique. Ne pas déduire un plancher de bruit universel de deux exécutions. Pour affirmer une équivalence, une marge et une procédure adaptées seraient nécessaires ; trois différences de signes mixtes ne prouvent pas que l'effet est nul.

## 6. Test exposé et classement d'une grille

Les premières expériences ont gardé le test officiel intact. Les campagnes complètes l'ont suivi à plusieurs epochs, puis la grille a été interprétée en fonction de ces résultats. Le checkpoint final de 30 époques a été fixé ; aucune sélection best-epoch n'est rapportée. Cela réduit un biais de sélection, mais ne rend pas le test à nouveau invisible.

Le meilleur résultat parmi 21 configurations est une sélection exploratoire. Ne pas publier sa moyenne comme une confirmation indépendante de la configuration gagnante. Une suite confirmatoire devrait définir ses choix avant de nouvelles évaluations et expliquer quels jeux de données ont déjà servi aux décisions. Rien dans cette base n'autorise à relabelliser le test CIFAR comme validation ou à cacher cette exposition.

## 7. Pourquoi les premières figures de la grille étaient difficiles

La figure originale [campaign_curves.png](sources/campaign_curves.png) juxtapose cinq petites familles A–E, avec des codes de configuration, des courbes seed0 uniquement et des pointillés de diagnostic très dégradés. Les comparateurs pertinents ne sont pas présents dans chaque panneau.

| Groupe original | Question réelle | Comparateur nécessaire |
|---|---|---|
| A | Résolution × calendrier Gaussian | Plain 32 constant et les bras correspondants |
| B | Mélange identité/Gaussian | Gaussian paliers et sans Gaussian à résolution égale |
| C | Sept sites contre dix-neuf | All19 et plain au même calendrier de résolution |
| D | Opérateur et lieu de réduction | Bilinéaire entrée, max entrée, bilinéaire stem, max stem |
| E | Ordre des résolutions | Rprog correspondant, avec et sans Gaussian |

A–E sont des familles d'ablations dans **une campagne**, pas cinq datasets ou cinq baselines. Les pointillés représentent le chemin cible forcé, pas des seeds supplémentaires ni l'incertitude. La baseline complète est `R32__Gnone__input_bilinear__all19` : la réduction r=32 et le filtre nul sont tous deux des identités.

## 8. Contrat de présentation pour les prochaines figures

### Figure principale

Montrer au plus cinq ou six méthodes : témoin, Gaussian seul, résolution bilinéaire seule, combinaison principale, max après stem sans Gaussian, éventuellement max entrée + Gaussian. Témoin noir, couleurs stables entre figures, noms français descriptifs. Utiliser les trois graines : courbe de moyenne et bande SD explicitement légendée ; conserver un panneau de valeurs individuelles finales.

La courbe principale est **current path**. Les diagnostics target/bypassed vont dans une annexe séparée ou des panneaux clairement dédiés. Ils ne doivent pas écraser l'échelle utile. Ne pas présenter une courbe seed0 comme celle de « l'expérience sur trois graines ».

### Figures complémentaires

1. Accuracy sur l'horizon complet, plus un zoom final explicitement annoncé comme zoom des mêmes données.
2. CE test et CE de sonde train, axes distincts si cela facilite la lecture.
3. Un dotplot final des 21 configurations, avec graines, moyenne, SD et noms lisibles ; pas 21 trajectoires superposées.
4. Des contrastes appariés, regroupés par question : lieu, opérateur, nombre de sites, ordre, calendrier.
5. Un nuage accuracy finale–temps mesuré, pour visualiser les compromis. Éviter le classement par « accuracy/seconde », un rapport peu interprétable ici.

Préciser les changements r=16→24→32 et la phase d'extinction Gaussian, sans confondre fin de l'epoch et index de la suivante. Axes accuracy en %, CE correctement nommée, unités de temps explicites, police lisible à taille normale. Ne pas interpoler de nouveaux points scientifiques, lisser artificiellement les courbes ou supprimer une graine défavorable. PNG et PDF peuvent être produits depuis les mêmes données.

### Disponibilité actuelle

L'utilisateur indique avoir vu des figures corrigées, mais elles ne sont pas dans les fichiers récupérés. Le présent dossier conserve les originales comme sources. Le JSON final ne contient pas les courbes par époque : un agent doit récupérer les logs métriques avant de recalculer des trajectoires à trois graines. Aucun réentraînement n'est nécessaire pour reformater des données déjà enregistrées.
