---
id: DOC-CURRENT
schema_version: 1
updated_at: 2026-09-18
status: current_at_snapshot
latest_completed_experiment: EXP-017
---

# État actuel au 9 septembre 2026

[Accueil](README.md) · [Dernière expérience](experiments/EXP-011_full_grid.md) · [Décisions](DECISIONS.md)

## Où en est la recherche ?

Nous étudions des méthodes de continuation pour l'entraînement de CNN : commencer par une version transformée du problème, puis revenir à la tâche cible avec les mêmes paramètres appris. La piste initiale portait sur des images simplifiées ; les premiers résultats négatifs ont conduit au filtrage des caractéristiques internes inspiré de CBS, puis à la réduction réelle et progressive de résolution.

Le meilleur résultat exploratoire récent combine **résolution d'entrée 16→24→32** et **Gaussian interne sur 19 sites**, avec retour final au réseau 32×32 sans filtre. Un modèle sans Gaussian, avec réduction par max-pooling après le stem, constitue une alternative intéressante en coût.

## Derniers résultats comparables entre eux

Source : [JSON de la campagne](sources/campaign_results.json), recalculs dans [les tableaux](data/campaign_tables.md). Même campagne fraîche, graines 0/1/2, CIFAR-10 50 000 train / 10 000 test, 30 époques, ResNet-20 BN, sans augmentation. Accuracy en %, SD descriptive entre graines en points ; durée totale moyenne par run, évaluation comprise.

| Configuration | Accuracy moyenne ± SD | CE test | Temps moyen |
|---|---:|---:|---:|
| **Témoin : 32 constant, aucun filtre** | 75,62 ± 0,49 | 0,7552 | 8,15 min |
| 32 constant + Gaussian paliers, 19 sites | 78,69 ± 0,68 | 0,6210 | 11,60 min |
| Résolution progressive, bilinéaire en entrée | 79,48 ± 0,54 | 0,5989 | 8,21 min |
| **Résolution progressive bilinéaire + Gaussian** | **80,71 ± 0,65** | **0,5682** | 11,66 min |
| Résolution progressive max-pooling en entrée + Gaussian | 80,34 ± 0,76 | 0,5720 | 12,30 min |
| **Résolution progressive max-pooling après stem, sans Gaussian** | **79,97 ± 0,58** | **0,5894** | 8,18 min |

Le gagnant dépasse le témoin de **5,10 points** en moyenne appariée. L'ajout de Gaussian à la résolution progressive bilinéaire apporte **1,23 point**, positif sur les trois graines. L'option max après stem sans Gaussian reste environ **0,75 point** derrière le gagnant, pour environ **70 % de son temps**.

## Ce que nous retenons réellement

- Le bénéfice du Gaussian interne a été observé sur ResNet-18 BN, puis sur ResNet-20 BN, et répliqué sur le train CIFAR-10 complet. Les changements initiaux d'architecture, de normalisation et d'initialisation n'ont pas été séparés causalement.
- La résolution progressive fournit un bénéfice propre dans la recette actuelle. Le Gaussian apporte encore un supplément, plus faible que son gain à résolution constante.
- Les gains ne s'additionnent pas sur l'échelle de l'accuracy. Cela ne prouve pas un mécanisme identique : le calendrier effectif du Gaussian dépend lui-même de la résolution.
- Réduire les opérations de convolution n'a pas fourni de réduction nette du temps par update sur ces petits tenseurs T4. Les estimations de FLOPs ne doivent pas devenir des promesses de vitesse.
- Trois graines et un test déjà exposé justifient une conclusion exploratoire située, pas une supériorité universelle ni une estimation sans biais du meilleur réglage de la grille.

## Statut des pistes

| Piste | Statut au présent instant | Sens de ce statut |
|---|---|---|
| Résolution progressive bilinéaire + Gaussian paliers, 19 sites | Référence de performance à conserver | Meilleure moyenne de la dernière grille |
| Résolution progressive max après stem, sans Gaussian | Candidate à conserver | Bon compromis observé entre précision et temps |
| Résolution progressive seule / Gaussian seul / plain | Contrôles à conserver | Indispensables pour lire les contributions |
| Gmix, Gaussian limité aux sept premiers sites | À déprioriser dans les réglages testés | Coût ou précision moins favorables ; pas un rejet universel des familles |
| Géométrique comprimé, résolution douce ou ordre inversé | Explorés ; pas prioritaires face au candidat principal | Leurs variantes précises n'améliorent pas le meilleur bras |
| Ondelette db2 | **Abandonnée par décision utilisateur** | Pilote BN peu convaincant et très coûteux |
| TV–L² et TV–Ḣ⁻¹ exactes | Aperçus réalisés ; pas d'entraînement de classification | Coût élevé, plusieurs solves non convergés |
| Diffusion TV à nombre fixé d'étapes | Idée en réserve | Pas de résultat expérimental retrouvé |
| Gaussian explicite sur RGB avant la réduction | **Discuté, non exécuté** | Ne pas le confondre avec l'antialiasing bilinéaire déjà présent |
| STL-10 | **Exécuté** ([EXP-015](experiments/EXP-015_stl10_resolution.md)) | Résolution progressive transposée ; gain réel à temps égal |

## Prochain travail utile, sans lancer automatiquement

L'utilisateur souhaite consolider CIFAR-10 avant d'étendre à STL-10. La discussion la plus récente porte sur un préfiltrage explicite des images **avant** leur réduction et sur le sens de l'antialiasing. L'ordre précis de la suite reste à décider avec lui. Les [questions ouvertes](OPEN_QUESTIONS.md) distinguent cette piste récente des anciens axes encore possibles.

L'agent du dépôt peut maintenant compléter les traces de preuve : code exact des quatre réductions, paramètres du resize, calendrier Gmix, manifestes, logs et commits. Cette intégration documentaire est utile sans nouveau calcul de formation.

## Pièges à éviter dès la reprise

**La réduction est effectuée une seule fois au début du réseau**, sur l'image ou après le stem selon le bras ; Gaussian est inséré à plusieurs endroits. Il n'y a pas de remontée systématique 16→32 avant la convolution.

**Les traits pleins doivent montrer le chemin courant réellement entraîné.** L'évaluation forcée à 32×32 sans filtre pendant la continuation est un diagnostic différent. Son mauvais score n'est ni celui du témoin ni une preuve suffisante d'un problème exclusivement BatchNorm.

**« Max après stem est moins bon » est trop général.** À emplacement après stem fixé, le max bat le bilinéaire avec et sans Gaussian. Le meilleur bras global filtré utilise néanmoins le bilinéaire en entrée. Voir les contrastes dans EXP-011.

**L'ancienne passation du 8 septembre est périmée comme plan d'action.** Elle reste une source pour les expériences anciennes, mais précède le nouveau pilote db2 et les deux études de résolution.

## Exécution récente et disponibilité des preuves

La grande campagne a réellement entraîné **63 cellules fraîches**, et non les 52 initialement attendues après réutilisation. Un défaut de détection des fichiers réutilisables a coûté du calcul supplémentaire. Les partitions sont rapportées disjointes et complètes ; le JSON confirme 63 cellules, aucune manquante et aucun doublon. Total : **11,13 heures GPU cumulées**, environ **2 h 08 écoulées** sur six T4. Le commit `f191fa6` a depuis été **résolu dans le dépôt** ; voir [INTEGRATION](INTEGRATION.md) (C-31).

Le JSON contient les agrégats finaux et les valeurs par graine. Il **ne contient pas les trajectoires complètes** par époque — mais le dépôt, lui, contient les 63 `metrics.json` par cellule et les 13 figures françaises de `results/presentation/` : voir [INTEGRATION](INTEGRATION.md) (C-32, C-33). Les anciens graphiques de `sources/` ne doivent pas être présentés comme la version corrigée.

## Complément du 17 septembre 2026 — EXP-012

La branche `adaptative-resolution` a exécuté la phase 0 de son plan ([fiche](experiments/EXP-012_adaptive_phase0.md), [plan et verdict](../../adaptive_resolution_plan.md)). La norme du gradient de l'objectif courant est **stationnaire** (1,0 ± 0,2 pendant vingt époques, toutes résolutions) : le déclencheur de plateau proposé n'a rien à détecter et est abandonné en recommandation. La grille de frontières (graine 0) place six alternatives sur sept à ±0,45 point de Rprog ; deux calendriers à pas fins, `Rsteps4` (79,80 %) et `Rlin12` (79,70 %), dépassent Rprog (78,63 % dans le même job) de plus d'un point, ce que l'alignement des gradients entre résolutions voisines prédit. Une graine : à confirmer sur trois graines appariées avant toute conclusion. La recalibration BN à poids fixés répond à Q-04.

## Complément du 17 septembre 2026 — EXP-013

Confirmation sur trois graines appariées ([fiche](experiments/EXP-013_adaptive_phase1.md)) : sans filtre, `Rlin12` (80,14 %) et `Rsteps4` (80,02 %) dépassent `Rprog` (79,14 %) de **+1,00 / +0,88 point**, positifs sur chaque graine, à temps égal. Avec le Gaussian paliers, le gain disparaît (Rsteps4+G 80,42 contre Rprog+G 80,48) : filtre et pas fins sont substituables. Les rampes non filtrées font 80,0–80,1 % pour 68 % du temps du meilleur bras filtré et deviennent le candidat précision/temps. Un contrôleur vivant de taille de pas sur τ et une rampe à tailles paires sont en cours (phase 2).

## Complément du 17 septembre 2026 — EXP-014 : la piste « déclencheur adaptatif » est close sur CIFAR-10

Le contrôleur vivant sur l'efficacité de transfert ([fiche](experiments/EXP-014_adaptive_phase2.md)) n'a jamais tiré ; dominé par son plancher, il réalise 16 × 9 → 20 → 24 → 28 → 32 et retombe au niveau de Rprog (79,26 / 79,11 %). Ce calendrier a la même masse grossière, le même pas et la même époque terminale que Rsteps4 : le point perdu vient de la **répartition du séjour** (neuf époques à 16 au lieu de trois par taille). Trois familles de signaux (norme du gradient, look-ahead, alignement) ont été testées ; aucune ne bat un calendrier fixe. **Référence non filtrée recommandée : rampe fine à séjour égal (`Rlin12` 80,14 %, `Rsteps4` 80,02 %, `Rlin12even` 80,12 %).** Suites proposées : frontière précision/temps contre `stem_max`, augmentation, STL-10.

## Complément du 17 septembre 2026 — EXP-015 : STL-10

Premier résultat STL-10 de cette base ([fiche](experiments/EXP-015_stl10_resolution.md)) : à 96×96 les époques grossières coûtent réellement moins (48×48 : 25 % du temps). À updates égaux, les calendriers progressifs valent +0,4 à +1,2 point sur le témoin (57,67 %) pour 78–85 % de son temps, bruit de graine élevé ; à compute nominal égal, `Rprogeq` 60,48 % et `Rlin24eq` 60,25 % contre 57,67 %, positifs sur chaque graine, dont environ deux tiers sont l'effet du budget (plain à 72 époques : 59,52 %) ; à temps égal, environ +2,3 points. Recommandation inchangée : calendrier fixe grossier→fin, horizon choisi pour le temps, pas de déclencheur.

## Complément du 17 septembre 2026 — EXP-016 : un signal de spécialisation d'échelle qui pilote la résolution

La pénalité relative de perte à l'échelle suivante après recalibration BN, g_Δ(θ), mesure la spécialisation du réseau à l'échelle courante : nulle à l'initialisation, croissante avec le séjour, réinitialisée à chaque bascule, et elle classe les calendriers fixes par accuracy ([fiche](experiments/EXP-016_bounded_specialisation.md)). Le contrôleur « avancer de 4 quand g ≥ g* » a décidé 35 bascules sur 36 lui-même et retrouve l'accuracy des meilleures rampes fixes (80,02 % à g* = 0,06 ; Rsteps4 80,02 ; Rprog 79,14), avec un seuil robuste entre 0,04 et 0,06 et un séjour non uniforme (le plus long à 16). Il ne dépasse pas les rampes fixes ; il remplace la table du chemin par un seuil. La piste « déclencheur adaptatif » rouvre donc sous cette forme précise ; le test informatif suivant est STL-10 à temps égal. Le décalage des statistiques BN n'est pas une mesure de spécialisation.

## Complément du 18 septembre 2026 — EXP-017 : le contrôleur adaptatif dépasse les rampes fixes

Le signal de spécialisation lu vers le bas, g(32→24), croît pendant toute la phase 32×32 alors que l'accuracy test stagne ; une époque 24×24 de « réchauffe » chaque fois qu'il dépasse 0,30 ([fiche](experiments/EXP-017_two_sided_specialisation.md)) donne, sur **six graines appariées**, **80,52 % contre 79,91 % pour la rampe fine fixe Rsteps4 : +0,61 ± 0,35 point, positif sur chaque graine**, à budget et temps égaux, sans filtre, au-dessus du gagnant filtré de la campagne (80,48 % dans ce protocole) pour 68 % de son temps. Un contrôle fixe (réchauffes aux époques 20/24/28) confirme le mécanisme (+0,46) ; la version adaptative le place au moins aussi bien. **Nouvelle meilleure recette non filtrée : rampe fine + réchauffes adaptatives.** À vérifier : robustesse du seuil, Gaussian, STL-10.
