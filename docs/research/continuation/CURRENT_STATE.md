---
id: DOC-CURRENT
schema_version: 1
updated_at: 2026-09-11
status: current_at_snapshot
latest_completed_experiment: EXP-012
---

# État actuel au 11 septembre 2026

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

## Depuis EXP-012 — ce qui a changé le 10 septembre

[EXP-012](experiments/EXP-012_aa_ablation.md) a cherché **pourquoi** le Gaussian interne aide,
avec 58 cellules sur 32 configurations. Assets épinglés neufs : ces chiffres **ne se comparent
pas cellule-à-cellule** à ceux de la grille ci-dessus (voir [INTEGRATION](INTEGRATION.md) C-43).

Quatre conclusions déplacent la pratique :

1. **Le placement du flou n'a que deux valeurs possibles, et le dépôt avait la mauvaise.**
   Le Gaussian commute exactement avec la convolution et avec BatchNorm (écarts 7·10⁻⁷ et 9·10⁻⁷)
   mais pas avec le ReLU (1,02). Donc `conv_out` et `post_bn` sont le même opérateur — mesuré,
   +0,21 pp entre eux — et le seul choix réel est de quel côté du ReLU on se place.
   Après le ReLU : **+1,71 pp** à résolution constante, **+0,99 pp** avec résolution, pour
   **moins cher** (10 positions au lieu de 19).

2. **L'annelage n'achète pas d'accuracy, il achète l'architecture cible.** Un sigma constant
   égale le calendrier annelé aux deux placements (+0,51 et −0,11 pp, une graine). Mais il
   s'effondre à 14–25 % dès qu'on retire son filtre, contre **0,00 pp** d'écart pour les bras
   annelés. C'est le contrôle qui manquait depuis EXP-004.

3. **Meilleure configuration mesurée : réduction après `blocks[2]` + flou post-ReLU**,
   **80,99 ± 0,41 %** sur trois graines, 574 s. Alternative en coût : même réduction après
   `blocks[1]`, **sans flou**, 80,59 %, 447 s — à peine plus cher que le témoin.

4. **Aucun a priori de profondeur sur sigma ne bat le profil plat.** Quatre profils motivés
   différemment, deux architectures, trois graines : sept comparaisons sur huit négatives avec
   le même signe partout. Les deux directions perdent, donc ce n'est pas une question de sens.

Sur l'hypothèse anti-aliasing, le faisceau reste **mitigé**. Pour : un préfiltre fixe à deux
positions reproduit l'effet à 1,09× le coût du témoin ; il existe du repliement réel (17–20 %
chez le témoin). Contre : un masque excluant tous les sites pré-décimation récupère l'effet
complet, et **l'énergie repliée résiduelle est identique aux trois placements** alors que
l'accuracy s'étale sur 1,7 point.


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
| Flou après le ReLU plutôt qu'après la convolution | **Établi par EXP-012, à adopter** | Gain démontrable analytiquement, et moins cher |
| Réduction de résolution à l'intérieur du réseau | **Établi par EXP-012** | Plateau entre `blocks[0..2]` ; réduire l'image est moins bon |
| A priori de profondeur sur sigma | **Écarté par EXP-012** | Quatre profils, deux architectures, trois graines : le plat gagne |
| BlurPool fixe aux deux points de décimation | Candidate à confirmer | +3,45 pp pour 1,09× le coût, une seule graine |
| STL-10 | Suite envisagée, non exécutée | Aucun résultat STL-10 dans cette base |

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
