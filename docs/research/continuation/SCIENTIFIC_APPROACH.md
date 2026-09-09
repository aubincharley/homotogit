---
id: DOC-SCIENCE
schema_version: 1
updated_at: 2026-09-09
status: synthesis
---

# Approche scientifique et évolution du raisonnement

[Accueil](README.md) · [Expériences](experiments/INDEX.md) · [Sources théoriques](SOURCES.md)

## 1. Le problème initial

Le projet « Entraînement homotopique des réseaux de neurones profonds », proposé par Joseph Gabet, vise à concevoir et évaluer des chemins de problèmes d'apprentissage. Le sujet demande une cartographie critique des approches, des propositions justifiées et des benchmarks contrôlés, pas une course à la taille des modèles. Les livrables envisagés sont un poster en anglais, un article de vulgarisation, un rapport scientifique et du code reproductible. Voir le [sujet original](sources/Sujet24_Joseph_Gabet.pdf) et la [synthèse théorique du groupe](sources/homotopy_synthesis.tex).

Pour un classifieur de logits \(f_\theta\), la cible est

\[
L(\theta)=\frac1n\sum_{i=1}^n
\ell_{\mathrm{CE}}(f_\theta(x_i),y_i).
\]

Une continuation construit une famille \(L_s\) revenant à cette cible, puis réutilise les paramètres appris lors du passage d'un niveau au suivant. Dans nos essais, SGD poursuit son évolution pendant les étapes ; nous ne résolvons pas exactement chaque problème intermédiaire.

Trois objets doivent toujours être décrits séparément : **la famille d'opérateurs**, **le calendrier de continuation**, **l'optimiseur et son calendrier de learning rate**. Modifier plusieurs de ces objets ensemble peut être utile pour explorer, mais limite l'attribution causale des résultats.

## 2. La distinction centrale : simplifier quoi ?

La synthèse du groupe distingue plusieurs directions.

| Construction | Exemple | Difficulté non résolue par la formule |
|---|---|---|
| Interpolation d'objectifs | \((1-s)L_{easy}+sL\) | Rien ne garantit que les bassins restent favorables |
| Régularisation décroissante | \(L+\lambda(s)R\) | Forte régularisation ne signifie pas problème facile |
| Lissage dans l'espace des paramètres | \(\mathbb E_\xi L(\theta+\sigma\xi)\) | Coût et hypothèses géométriques difficiles en grande dimension |
| Curriculum d'exemples | Pondérations \(w_i(s)\) | La difficulté sémantique dépend de la tâche |
| Modification du réseau | Activations, bruit, filtres internes | Un réseau linéaire en entrée n'est pas nécessairement convexe en ses poids |
| Transformation des données | Images filtrées, résolution variable | Simplifier l'image peut supprimer des indices utiles |
| Suivi ou apprentissage du chemin | Predictor-corrector, représentation \(\theta_\phi(s)\) | Nécessite d'abord une famille pertinente |

Ces directions sont présentes dans la réflexion initiale ; seules certaines ont été testées ici. Les clés bibliographiques de la synthèse sont conservées dans les sources, mais son fichier `.bib` manque. Ses notices ne doivent pas être complétées de mémoire.

## 3. Première hypothèse expérimentale : simplifier les entrées

L'idée de départ était une famille label-blind d'images \(T_\eta x\), avec

\[
L_\eta(\theta)=\frac1n\sum_i
\ell_{\mathrm{CE}}(f_\theta(T_\eta x_i),y_i).
\]

Le Gaussian était un premier opérateur simple ; TV et budgets opérationnels de compression devaient élargir la famille. Les transformations sont calculées depuis **l'image originale** : affaiblir le flou ne consiste pas à réutiliser l'image déjà floutée, qui ne permettrait pas de retrouver les détails originaux.

L'[expérience à Gaussian fixe](experiments/EXP-000_input_gaussian.md) montre que les modèles peuvent ajuster leurs images floutées tout en se transférant mal aux images originales. Les niveaux forts n'ont pas donné une optimisation visiblement plus rapide. La baisse de performance cible pendant que la performance sur images floutées continue de progresser motive un retour assez tôt à la tâche cible.

Ce constat a conduit à une expérience plus précise : **un court warm start aide-t-il après adaptation, et une étape intermédiaire ajoute-t-elle quelque chose ?** L'[EXP-001](experiments/EXP-001_input_warmstart.md) compare entraînement direct, flou 1→0 et flou 1→0,5→0 à budget global égal. Les deux continuations restent légèrement derrière le témoin. La récupération après un changement de distribution ne suffit donc pas à établir un bénéfice net.

Il ne faut pas durcir cette conclusion en « le flou d'entrée ne peut pas fonctionner » ou « la continuation exige que le témoin soit coincé ». Une méthode peut modifier la vitesse à budget fini ou la généralisation même si toutes les branches finissent par ajuster les données.

## 4. Une simplification contrainte mieux définie : TV et ondelettes

La proposition TV cherchait la reconstruction la plus proche de l'image sous un budget relatif de variation totale. Deux fidélités ont été retenues, L² et homogène Ḣ⁻¹, avec boîte RGB et moyennes par canal conservées. La contrainte porte sur une quantité mesurable et comparable par image, mais ne garantit pas la conservation de l'information utile à la classe.

Les [aperçus TV](experiments/EXP-002_tv_previews.md) montrent un contraste global encore important à TV réduite, surtout avec Ḣ⁻¹. Ils révèlent aussi un coût élevé et des solves incomplètement convergés. Ce sont des résultats sur une transformation d'images, pas sur la qualité d'un classifieur.

Les [ondelettes non décimées](experiments/EXP-003_wavelet_previews.md) proposent ensuite un opérateur explicite : conserver l'approximation grossière et seuiller doucement les détails selon leur RMS. Cela évite la résolution d'un programme convexe par image. Mais une opération bien moins chère que la projection TV peut rester très chère lorsqu'elle est insérée 19 fois avec backward. Le [pilote db2 récent](experiments/EXP-009_db2_bn.md) a précisément montré cette limite. L'utilisateur a choisi d'abandonner cette branche.

## 5. Le déplacement vers les caractéristiques internes

*Curriculum by Smoothing* motive le filtrage spatial des sorties de convolution. Les auteurs rapportent un avantage de leur calendrier interne par rapport à leurs ablations sur les images ou à flou constant ; leurs expériences couvrent également du transfert et des modèles génératifs. Ce sont des indices expérimentaux sur le lieu du filtrage et le rôle du calendrier, pas une preuve de convexification du paysage en paramètres. [Sinha, Garg et Larochelle, CBS](https://arxiv.org/html/2003.01367v5).

Nos premiers essais internes ResNet-20 GroupNorm restent défavorables au Gaussian. L'audit du placement ne trouve pas d'explication simple par une insertion erronée. En revanche, le modèle, la normalisation, l'initialisation et le budget diffèrent fortement de la référence.

Une expérience ResNet-18 BatchNorm avec initialisation corrigée produit **+7,90 points** sur le petit split. Le retour à ResNet-20 BatchNorm produit ensuite **+4,26 points**, puis le passage au dataset complet confirme environ **+3 points** sur trois graines. Cette progression justifie de garder un réseau petit et rapide. Elle ne sépare pas l'effet propre de BN, de l'initialisation, de la capacité ni des calendriers successifs.

La recette de CBS examinée part de sigma 1 et le multiplie par 0,9 toutes les cinq époques ; nos noyaux, bords, microbatches et calendriers diffèrent. La comparaison doit être dite **inspirée de CBS**, pas reproduction exacte. [Article CBS](https://arxiv.org/html/2003.01367v5) et [dépôt officiel](https://github.com/pairlab/CBS).

## 6. Un changement décisif dans la mesure

Les premières courbes Gaussian BN montraient en principal le modèle auquel on retirait les filtres pendant l'évaluation. À ce stade, on modifiait la fonction évaluée et les activations auxquelles les statistiques BN étaient appliquées. Le score très faible ne décrivait donc pas le prédicteur effectivement entraîné.

La correction consiste à mettre le **chemin courant** au premier plan. Le **chemin cible forcé** reste utile pour mesurer un changement anticipé de configuration. La distinction devient encore plus importante lorsque résolution et filtres changent ensemble. Les détails et les limites de l'explication BN sont dans [EVALUATION](EVALUATION.md).

La correction du pilote ResNet-20 montre un avantage Gaussian courant à tous les checkpoints rapportés. Cela n'autorise pas à dire que le même ordre vaut pour toutes les campagnes : avec 50 000 images, les courbes courantes peuvent être en retard au début puis dépasser le témoin.

## 7. Résolution progressive : une deuxième manière de modifier le problème

Réduire réellement les images à 16×16, puis 24×24 et enfin 32×32 change les observations et le champ réceptif relatif. Le réseau garde les mêmes poids : ses convolutions sont locales et sa moyenne spatiale globale maintient une tête de dimension fixe.

Le [pilote résolution](experiments/EXP-010_resolution_pilot.md) fournit un signal favorable. La [grille complète](experiments/EXP-011_full_grid.md) compare ensuite calendriers, mélange identité/Gaussian, nombre d'insertions, opérateurs de réduction, lieux de réduction et ordre des résolutions. Elle permet de distinguer plusieurs résultats :

- la réduction progressive seule aide dans cette recette ;
- Gaussian interne apporte encore un supplément avec la réduction bilinéaire en entrée ;
- le max-pooling après le stem sans Gaussian constitue une option économique ;
- réduire le Gaussian à sept sites ou remplacer sa décroissance de largeur par le Gmix testé dégrade les résultats ;
- l'ordre des résolutions semble interagir avec le filtrage, mais change aussi son calendrier effectif.

Le nouveau point de discussion est le **Gaussian explicite avant le resize des images**. Les essais positifs « résolution + Gaussian » réalisés jusqu'ici combinent réduction d'entrée et Gaussian **interne**. L'antialiasing déjà inclus dans le resize bilinéaire doit être distingué de ce nouveau préfiltre.

## 8. Comment interpréter le motif train/test ?

Plusieurs méthodes terminent avec une CE de sonde train supérieure à celle du témoin, mais une CE et une accuracy meilleures hors entraînement. C'est compatible avec une **régularisation implicite par la trajectoire** : la suite de problèmes peut orienter les paramètres vers un prédicteur qui généralise mieux.

Cette interprétation ne sépare pas proprement « optimisation » et « régularisation ». L'optimiseur suit une trajectoire différente, ce qui peut modifier la solution finale sans diminuer davantage la CE train. Inversement, une baisse de CE test peut venir de probabilités moins excessivement confiantes sur certaines erreurs ; elle ne suffit pas à établir une meilleure calibration sans diagnostic dédié.

Nos mesures n'établissent pas de minima plus larges, de bassins plus favorables, de trajectoire de minimisateurs continue, ni de convergence globale. Les courbes train/test, la vitesse d'apprentissage, la robustesse et la géométrie de la perte sont des objets liés mais différents.

## 9. Les précautions mathématiques qui ont structuré le projet

1. Le Gaussian spatial lisse les images ou les cartes ; ce n'est pas la convolution de la perte dans l'espace des poids.
2. Une faible TV, une faible résolution et une tâche facile ne sont pas synonymes.
3. Le spectral bias d'une fonction apprise dans l'espace des entrées n'est pas identique au filtrage spatial d'une carte interne.
4. Un problème convexe de reconstruction TV ne rend pas convexe l'apprentissage du CNN.
5. Le Gaussian continu a une interprétation de chaleur ; son noyau discret tronqué avec padding n'a pas automatiquement le même semi-groupe exact.
6. Une atténuation des détails ne constitue pas à elle seule une mesure de perte d'information de Shannon. Un budget de codec doit être décrit en bits mesurés et avec son décodeur.
7. La qualité d'un opérateur et celle du calendrier qui le parcourt ne sont pas identifiables par une seule comparaison au témoin.

## 10. Ce qui ferait progresser la conclusion scientifique

La prochaine étape ne consiste pas nécessairement à multiplier les configurations. Il reste notamment à séparer : bénéfice de la phase initiale et bénéfice des étapes intermédiaires ; adaptation au nouveau chemin et adaptation des statistiques BN ; filtrage interne et antialiasing d'entrée ; gain à nombre d'updates égal et gain à temps égal.

Ces contrôles sont listés comme **questions**, pas comme une campagne déjà décidée. Le choix doit tenir compte du résultat exploratoire actuel, du budget et de l'intérêt d'un transfert à STL-10. L'[état actuel](CURRENT_STATE.md) prime sur les anciens plans d'action archivés.
