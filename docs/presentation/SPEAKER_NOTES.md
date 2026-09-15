# Commentaires de présentation

Page companion : <https://claude.ai/artifact/KHdmwx12uiCBqmtkF6czXC>

Notes pour la partie « sondes, visualisation, intuition ». Pour chaque planche :
la phrase à dire, le chiffre à citer, et la question qu'on va poser.
La seconde moitié du document est la partie qui compte vraiment — ce que ces
mesures appellent comme travail théorique.

---

## A. Comment amener cette partie

> « On a un gain reproductible de 4,5 points. On ne sait pas pourquoi. Cette
> partie n'essaie pas de le prouver — elle essaie de **réduire l'espace des
> explications possibles**. C'est un travail de réfutation : chaque planche
> retire une hypothèse de la table, et ce qui reste debout à la fin est
> l'endroit où il faut mettre la théorie. »

Trois choses à poser d'entrée, parce qu'elles évitent les trois quarts des
objections :

1. **Tout est mesuré au point final**, sauf une planche. Donc rien ici n'est
   causal, et on ne le prétendra pas.
2. **Les quatre méthodes sont gelées.** On ne les a pas ré-accordées pour
   obtenir le résultat ; ce sont celles du benchmark, et elles se reproduisent
   à 0,5 pp près.
3. **Un résultat négatif n'est utile que s'il avait pu être positif.** D'où les
   quatre garde-fous, et notamment le plafond de détection : il vaut 1,00, donc
   quand on dit « aucune corrélation », ce n'est pas un manque de puissance.

---

## B. Planche par planche

### 01 — Le dispositif

**À dire.** « Les quatre méthodes partagent tout sauf un opérateur inséré dans
le réseau et le calendrier qui l'éteint. Le point important est là : à l'époque
21, l'opérateur est à l'identité exacte. Les neuf dernières époques entraînent
le même réseau nu que le contrôle. Donc ce qu'on mesure ensuite n'est pas
l'effet de l'opérateur — c'est **la trace qu'il a laissée**. »

**Le chiffre.** Identité exacte à l'époque 21 (flou) et 12 (résolution).

**Le détail qui montre qu'on a lu le code.** Le σ effectif de la méthode
combinée n'est pas monotone : il remonte de 0,425 à 0,525 au passage 16→24,
parce que le facteur `r/32` s'applique aux 19 sites, y compris ceux qui tournent
en amont de la réduction. On l'a conservé plutôt que « corrigé ». C'est
accidentel, et c'est une expérience gratuite sur la régularité du chemin
(voir axe 1).

**La question.** *« La méthode combinée, c'est bien la somme des deux ? »*
→ Non. 19 sites avant BatchNorm contre 10 après ReLU, et un σ mis à l'échelle.
Les écarts entre lignes confondent placement, nombre de sites et échelle. On ne
peut pas lire la ligne « combiné » comme un effet d'interaction.

---

### 02 — Le gain se reproduit

**À dire.** « Trois exécutions indépendantes, dont deux par des personnes
différentes, à 0,25 pp près. Et le point contre-intuitif : les curricula
**ajustent moins bien le jeu d'entraînement**. 0,057 d'erreur train pour le
contrôle, 0,084 à 0,101 pour les curricula. Ce n'est donc pas une optimisation
plus efficace ; c'est un biais différent. »

**Le chiffre.** 0,25 pp d'accord ; erreur train 0,057 → 0,084–0,101.

**Ce qu'il faut dire soi-même avant qu'on le demande.** Nos dispersions
inter-graines sont un ordre de grandeur plus serrées que les deux autres
sources. Deux jeux de données indépendants nous contredisent. On retire donc
toute affirmation sur la variance inter-graines, et on n'explique pas l'écart.
Le dire soi-même coûte dix secondes et achète toute la crédibilité du reste.

---

### 03 — Platitude

**À dire.** « La trace de la hessienne tombe de 30 à 40 %. L'objection standard
à toute mesure de platitude, c'est Dinh 2017 : on peut re-paramétrer un réseau
avec BatchNorm sans changer sa fonction et rendre la hessienne aussi grande
qu'on veut. On a donc quotienté par cette symétrie — la distance minimale sur
l'orbite — et elle explique **au plus 2 %** de l'écart. »

**Le chiffre.** −29,7 / −31,4 / −40,5 % ; jauge ≤ 2 % ; 36 mesures sur 36
gardent le signe.

**Le 2×2, et pourquoi il existe.** L'étude de paysage a trouvé le flou *plus*
sensible à amplitude finie. Deux choix étaient porteurs et non testés : les
statistiques BatchNorm stockées, et le split d'évaluation. On a mesuré les
quatre combinaisons. Rien ne se renverse ; l'effet est **plus grand sur le
test**. Mais la politique change l'ampleur d'un facteur 4 à 8 — donc une bonne
part de ce que mesurent des statistiques gelées est un désaccord de
statistiques, pas de la géométrie.

**La question.** *« La platitude explique donc le gain ? »*
→ Non, et c'est la planche 06. La platitude est un **marqueur robuste**, pas un
levier : elle ne prédit pas l'erreur test à l'intérieur de la famille.

---

### 04 — La dissociation *(le cœur de la partie)*

**À dire.** « Le flou décale le spectre de sensibilité vers le bas de 8,8 %. La
réduction de résolution ne le décale pas — elle le remonte de 4,6 %. Et les deux
gagnent 17 % d'erreur test. Donc **un biais basse fréquence ne peut pas être le
mécanisme commun.** Il est réel, et il appartient au flou seul. »

**Le chiffre.** −8,8 % contre +4,6 %, pour −16,7 % et −17,8 % d'erreur test.

**Pourquoi on peut y croire.** Le profil est **exact**, pas estimé : par
Parseval, la somme des ‖Je‖² sur une famille orthonormée d'anneaux *est*
l'énergie DFT de l'anneau. Aucune direction n'est échantillonnée. Et les dix
logits font un jacobien 10×3072, donc le spectre complet sort de dix passes
arrière, exactement.

**Ce qu'ils partagent quand même.** Pas la répartition de la sensibilité — sa
**taille**. ‖J‖_F tombe de 47,8 % et 30,2 %.

**La question.** *« Le "curriculum by smoothing" de Sinha 2020, alors ? »*
→ C'est précisément ce que la planche réfute comme explication commune. La
réduction de résolution n'est pas un lissage : c'est un `max_pool` adaptatif,
non linéaire, qui rétrécit la grille. Le cadre « lissage » ne couvre que le bras
gaussien, et les deux bras gagnent autant.

---

### 05 — Pas de régime linéaire

**À dire.** « Si une description au premier ordre tenait, le rapport
S(ε)/(ε·σ_max) vaudrait 1. Il vaut 0,32 à 0,47 à la plus petite amplitude qu'on
sache mesurer, et 0,05 à ε = 3. **Il n'y a aucune échelle où le jacobien décrit
la fonction.** »

**Pourquoi ça compte.** C'est la raison *structurelle* pour laquelle
σ_max et ‖J‖_F échouent comme prédicteurs, et pas seulement une question de
finesse. Toute la théorie standard — normes de Lipschitz, marges, platitude via
un modèle quadratique local — vit dans cette limite-là.

**Le renversement.** Le flou mène sur la dérivée (−47,8 %) et suit à amplitude
finie (−12,9 %) ; la résolution fait l'inverse. Le même renversement apparaît
dans l'espace des poids entre `tr H` et le `S(ε)` recalibré du collègue.
**Deux espaces indépendants, un seul motif** — et c'est ce qui concilie les deux
études : la courbure au point est plus basse pour le flou, la réponse à distance
finie est plus haute. Les deux sont vraies du même réseau.

---

### 06 — Aucun cadran

**À dire.** « Vingt conditions, corrélation partielle à erreur d'entraînement
égale, Pearson et Spearman, Bonferroni sur 25 quantités. Les coefficients poolés
sont beaux : +0,83. Et ils s'effondrent à +0,08 dès qu'on décompose par groupe.
Ce sont des **paradoxes de Simpson** : les cellules de contrôle ont à la fois un
grand prédicteur et une grande cible, donc le coefficient poolé mesure "est-ce
un bras curriculum", pas une relation. »

**Le chiffre à ne pas rater.** Plafond de détection = 1,00, étendue d'exactitude
12,5 points. **Le test pouvait répondre.**

**La rétractation.** Une étude exploratoire antérieure donnait la réponse dans
la pire direction comme prédicteur à +0,72. Ici : +0,178 dans la famille
curriculum. Elle reste une signature partagée (−12,9 / −16,5 / −18,6 %) — un
contraste, pas un cadran.

**La question.** *« Vous n'avez juste pas assez de points. »*
→ C'est exactement ce que le plafond de détection mesure, et il vaut 1,00. On a
aussi étalé l'exactitude sur 12,5 points au lieu de 5,4, précisément pour ça.

---

### 07 — PAC-Bayes

**À dire.** « La borne centrée sur l'initialisation est vide d'un facteur 400.
La platitude en rachète 1,65, la jauge 1,06. Le dernier levier était de déplacer
l'a priori sur une frontière de phase du calendrier. Mesuré : à l'époque 6, le
réseau est à 32 de son point d'arrivée, contre un déplacement total de 13,6.
**Il est plus loin de l'arrivée que toute la trajectoire n'est longue.** Placer
l'a priori là aggrave la borne. »

**Le sous-produit.** Dès l'époque 1, les bras « flou » sont à 28,2 et 30,2 de
leur arrivée contre 45,1 et 47,7 pour les autres. Le flou coupe le déplacement
initial de 40 %, la résolution l'augmente. **La dissociation est déjà visible
dans l'espace des poids à la première époque.**

**Ce qu'il faut préciser.** « Fermé » veut dire : parmi les routes essayées.
Ce n'est pas une impossibilité générale — voir l'axe 4.

---

### 08–09 — Synthèse et bilan

**À dire.** « Deux études partagent les méthodes, l'architecture et les données,
et presque rien d'autre : ni l'espace mesuré, ni les outils, ni le protocole
statistique. Elles se rejoignent sur trois dissociations et sur la même
conclusion négative. C'est ce désaccord de méthode qui donne du poids à
l'accord de résultat. »

**La phrase de fin.** « Ce qu'on a établi, c'est que les deux interventions ne
font pas la même chose, et qu'aucune quantité au point final n'explique
pourquoi elles marchent. Ce qui reste, c'est le chemin. »

---

## C. Ce que cette partie peut et ne peut pas établir

À dire une fois, explicitement, parce que ça désamorce la moitié des objections :

| | |
|---|---|
| **Une sonde peut** | réfuter une explication (la dissociation tue le biais basse fréquence comme mécanisme commun) ; établir un contraste robuste (36/36) ; borner ce qui est mesurable (plafond de détection) |
| **Une sonde ne peut pas** | établir une causalité ; distinguer « le curriculum a créé cette propriété » de « le curriculum a atterri là où cette propriété existait » ; dire quand la propriété est apparue |

La troisième ligne est celle qu'on peut lever à coût nul — c'est l'axe 7.

---

## D. Axes théoriques

Pour chacun : ce qui est **mesuré**, ce qui est **conjecturé**, l'énoncé à
établir, et l'expérience qui trancherait.

### Axe 1 — Le gain est dans le chemin, pas dans le point d'arrivée

**Mesuré.** Aucune des 25 quantités de point final ne prédit l'erreur test à
l'intérieur de la famille, avec un plafond de détection de 1,00.

**Conjecturé.** L'objet pertinent n'est pas W*, c'est la trajectoire
η ↦ W*(η) et la façon dont SGD la suit.

**L'énoncé à établir.** Poser L_η(W) = E[ℓ(f_W^{T_η}(x), y)] : une famille de
pertes indexée par η, avec L_0 la perte cible. Le curriculum, c'est une méthode
de continuation sur cette famille. La théorie existe pour les systèmes
d'équations (Allgower & Georg) : le suivi de chemin est justifié quand le chemin
est régulier — pas de point de rebroussement, jacobienne inversible le long du
chemin — et que l'erreur de suivi est contrôlée. **Question : sous quelles
conditions sur T_η le chemin des minima de L_η est-il continu, et quand SGD
reste-t-il dans un tube autour de lui ?**

Version faible mais prouvable en premier : dans un modèle quadratique ou dans le
régime paresseux (NTK), avec T_η **linéaire** et fixé, le chemin est explicite
et on peut écrire la condition de suivi.

**Le test.** La méthode combinée a un σ effectif **non monotone** (0,425 →
0,525 au passage 16→24). Si la régularité du chemin est le mécanisme, un aller-
retour en η devrait coûter quelque chose de mesurable. C'est une expérience
accidentelle déjà à moitié faite, et les points de contrôle existent.

---

### Axe 2 — Pourquoi les deux interventions dissocient : symbole contre géométrie

C'est l'axe le plus prometteur, parce que la distinction est **structurelle** et
qu'elle prédit exactement le signe qu'on observe.

**Mesuré.** Flou : rayon fréquentiel −8,8 %. Résolution : +4,6 %. Même gain.
‖J‖_F baisse pour les deux.

**Conjecturé.** Les deux opérateurs n'agissent pas sur le même objet :

| | flou gaussien | réduction de résolution |
|---|---|---|
| nature | **linéaire, invariant par translation** | `adaptive_max_pool2d` : **non linéaire, non invariant** |
| ce qu'il est | un multiplicateur de Fourier, symbole strictement positif décroissant en fréquence | un changement de **domaine** : la grille rétrécit pour toutes les couches suivantes |
| ce qu'il fait | atténue **toutes** les fréquences, de façon graduée | change le champ réceptif en pixels d'entrée (×2 à r = 16) et divise par 4 le nombre de positions spatiales en aval |
| trace attendue | **graduée et persistante** — une préférence continue sur le spectre | **aucune préférence spectrale** — rien ne classe les fréquences entre elles |

Un symbole continu laisse une préférence ordonnée sur les fréquences ; un
changement d'échelle n'en laisse pas, parce qu'il ne classe rien. Et un max-pool
n'est pas un passe-bas : il préserve, voire crée, de la structure haute
fréquence — ce qui rend le +4,6 % attendu plutôt que surprenant.

**Ce que les deux partagent alors :** pas le spectre, mais le **nombre de degrés
de liberté disponibles tôt** — le flou par atténuation d'un continuum, la
résolution par réduction du support. C'est l'unification candidate pour
expliquer la baisse commune de ‖J‖_F sans prédire de décalage spectral commun.
Cadre existant à mobiliser : dimension intrinsèque effective (Pope et al. 2021).

**L'énoncé à établir.** Dans le régime paresseux, avec T linéaire fixé, le flot
de gradient converge vers l'interpolant de norme minimale **dans la norme
induite par T** — donc un noyau modifié, explicitement calculable pour une
gaussienne. Ça donne une prédiction quantitative pour le bras flou seul.
Le bras résolution sort du cadre (non linéaire) et demande un autre outil.

**Le test, et il est bon marché.** Remplacer le `max_pool` par un `avg_pool`
de même géométrie, et ajouter un passe-bas idéal (projecteur, symbole 0/1) de
coupure appariée. Quatre opérateurs, quatre natures :

| opérateur | linéaire ? | symbole | trace spectrale prédite |
|---|---|---|---|
| gaussienne | oui | continu, > 0 | **graduée** — observé : −8,8 % |
| passe-bas idéal | oui | 0/1 (projecteur) | **tout ou rien**, pas de gradation |
| `avg_pool` | oui | moyenne par blocs | intermédiaire — sépare « linéaire » de « change d'échelle » |
| `max_pool` adaptatif | non | — | **aucune** — observé : +4,6 % |

Quatre cellules de plus suffisent à trancher, et le résultat est publiable dans
les deux sens : si le passe-bas idéal laisse une trace et le max-pool non, la
distinction symbole/géométrie tient ; si l'`avg_pool` se comporte comme le
`max_pool`, c'est le changement d'échelle qui compte, pas la linéarité.

---

### Axe 3 — Une théorie à distance finie, pas infinitésimale

**Mesuré.** S(ε)/(ε·σ_max) ne dépasse jamais 0,47, et vaut 0,05 à ε = 3.

**Ce que ça condamne.** Toute borne dont la quantité centrale est une dérivée
ou un modèle quadratique local : normes spectrales par couche, marges
normalisées à la Bartlett, platitude lue comme courbure. Elles décrivent une
limite que cette fonction n'occupe à aucune échelle mesurable.

**La direction.** Le cadre qui vit nativement à échelle finie est la
**robustesse algorithmique** (Xu & Mannor 2012) : partitionner l'espace
d'entrée en K régions et borner l'écart de perte à l'intérieur d'une région.
La borne s'écrit en (K, ε) et non en dérivées. Notre courbe S(ε) *est*
essentiellement une estimation empirique de cette constante.

**L'énoncé à établir.** Écrire une borne de type robustesse dont la constante
est celle qu'on mesure, et regarder de combien le curriculum la déplace. Le
verrou connu est que K croît exponentiellement en dimension ; la question de
recherche est de savoir si l'on peut prendre K sur la variété des données
plutôt que sur le cube — ce qui rejoint l'axe 2 par la dimension intrinsèque.

---

### Axe 4 — PAC-Bayes : ce qui reste après ce qu'on a fermé

**Mesuré et fermé.** A priori à l'initialisation (vide d'un facteur ~400),
quotient de jauge (1,06), a priori déplacé sur une frontière de phase
(aggrave la borne : 14 à 26 restants contre 13,6 de déplacement total).

**Ce que le résultat dit vraiment.** Que la trajectoire **n'est pas monotone** :
elle s'éloigne puis revient. C'est un fait géométrique en soi, et il mérite
d'être expliqué avant d'être contourné.

**Les routes qui restent, par ordre de promesse :**

1. **A priori dépendants des données**, appris sur une portion réservée
   (Ambroladze et al. 2006 ; Dziugaite & Roy 2018). C'est la route qui a produit
   les bornes non vides sur MNIST.
2. **Bornes par compression** (Arora et al. 2018). Elles dépendent de la
   sensibilité au bruit couche par couche — que nous **mesurons déjà**, par bloc,
   avec `curvature.probe`. C'est la route où notre infrastructure sert
   directement, et où le curriculum a un effet mesuré : les traces par bloc
   baissent partout.
3. **Quotient par le groupe de symétrie complet**, permutations comprises, pas
   seulement le rescaling BatchNorm (axe 6).

**Le test.** La route 2 se teste sans nouvelle machinerie : la borne de
compression consomme exactement les quantités par bloc déjà produites.

---

### Axe 5 — Caractériser le biais implicite : « ajuste moins, généralise mieux »

**Mesuré.** Erreur train 0,057 (contrôle) contre 0,084–0,101 (curricula), pour
une erreur test plus basse. Entropie croisée d'entraînement 0,20 contre
0,27–0,31 dans l'étude de paysage.

**Conjecturé.** Le curriculum agit comme un régulariseur : il ne trouve pas un
meilleur optimum de la même perte, il change l'optimum choisi.

**L'énoncé à établir.** Identifier le régulariseur implicite. Cas tractable :
T linéaire fixé, régime paresseux — l'interpolant de norme minimale dans la
norme induite par T (même objet que l'axe 2). Cas ouvert : un calendrier η(k)
en fait un **préconditionneur dépendant du temps**, et la question est de savoir
ce que sélectionne un flot de gradient préconditionné non stationnaire.

**Le test.** Comparer le curriculum à l'entraînement avec l'opérateur *gelé* à
chaque niveau η (pas de calendrier). Si le gain survit à η fixé, c'est le
régulariseur ; s'il ne survit qu'avec le calendrier, c'est le chemin (axe 1).
**C'est la seule expérience qui sépare proprement les axes 1 et 5.**

---

### Axe 6 — Une platitude définie sur le bon quotient

**Mesuré.** −19 à −55 % de `tr H`, 36/36, avec la jauge BatchNorm quotientée
(≤ 2 %). Et pourtant aucune valeur prédictive intra-famille.

**La tension à assumer.** On a un marqueur très robuste et sans pouvoir
prédictif. Deux lectures : soit la platitude est une conséquence et non une
cause ; soit on ne mesure pas la bonne platitude.

**L'énoncé à établir.** La symétrie BatchNorm n'est qu'une partie du groupe. Il
reste les permutations de neurones (Entezari et al. 2022 ; Ainsworth et al.
2023). Une sharpness véritablement invariante de jauge se définit sur le
quotient complet. Voir aussi les tentatives existantes : Tsuzuku et al. 2020
(platitude normalisée), Kwon et al. 2021 (ASAM).

**Le test.** Notre `gauge.py` fait déjà le quotient de rescaling. L'étendre au
quotient de permutation est un projet identifié, avec de la machinerie publique
réutilisable, et il retombe sur la question : la platitude devient-elle
prédictive une fois correctement quotientée ?

---

### Axe 7 — Quand les signatures apparaissent *(coût nul, à faire en premier)*

**Mesuré.** Rien : la sonde est écrite, testée, jamais lancée. Les 120 points de
contrôle par époque existent.

**Ce que ça décide.** Une seule question, et elle oriente tout le reste :

* si l'écart de platitude apparaît **à la première frontière de phase et
  persiste** → le curriculum sélectionne un bassin tôt, et la théorie à écrire
  est celle de l'axe 1 (chemin) ;
* s'il n'apparaît **qu'à la fin**, après l'extinction de l'opérateur → le
  curriculum change la fin de partie, et la théorie à écrire est celle de
  l'axe 5 (biais implicite).

Aujourd'hui on ne sait pas laquelle des deux, et c'est la fourche la moins chère
du document. **À lancer avant d'écrire quoi que ce soit de théorique.**

---

### Axe 8 — Le résultat négatif comme contribution méthodologique

**Mesuré.** Cinq coefficients poolés entre +0,78 et +0,83 qui tombent à
+0,06–0,15 après décomposition ; plafond de détection 1,00.

**L'énoncé.** Une « mesure de généralisation » doit être validée **à l'intérieur
d'une famille**, pas sur un pool de familles. Un coefficient poolé sur un mélange
de configurations mesure l'appartenance au groupe. C'est la nuance à apporter
aux études de corrélation à grande échelle (Jiang et al. 2020), et notre
décomposition inter/intra est directement l'outil.

À dire en une phrase si le temps manque : *« quatre méthodes font quatre points,
et une corrélation sur quatre points n'est pas une corrélation. »*

---

## E. Les trois choses à faire, dans l'ordre

| | coût | ce que ça débloque |
|---|---|---|
| **1. Sonder les 120 points de contrôle par époque** (axe 7) | nul — sonde écrite, données là | décide entre « chemin » et « biais implicite », donc quelle théorie écrire |
| **2. Les quatre natures d'opérateur** : gaussienne, passe-bas idéal, avg-pool, max-pool (axe 2) | 4 cellules | teste l'explication structurelle de la dissociation — publiable dans les deux sens |
| **3. η gelé contre η programmé** (axe 5) | une grille | sépare le régulariseur du chemin, ce qu'aucune mesure de point final ne peut faire |

Le travail théorique proprement dit — l'énoncé de suivi de chemin dans le régime
paresseux avec T linéaire — devient bien posé une fois que 1 et 3 ont répondu.

---

## Références à avoir en tête

* Allgower & Georg, *Numerical Continuation Methods*, 1990 — le cadre de l'axe 1.
* Bengio et al. 2009 — curriculum learning ; Sinha et al. 2020 — *Curriculum by
  Smoothing*, le voisin le plus proche, et celui que la planche 04 nuance.
* Dinh et al. 2017 — l'objection de re-paramétrisation ; Li et al. 2018 —
  normalisation par filtre.
* Jacot et al. 2018 ; Chizat et al. 2019 — le régime où les axes 2 et 5
  deviennent prouvables.
* Rahaman et al. 2019 ; Xu et al. 2019 — biais spectral, ce que la planche 04
  contraint.
* Pope et al. 2021 — dimension intrinsèque, le cadre de l'unification candidate.
* Xu & Mannor 2012 — robustesse algorithmique, le cadre à échelle finie.
* Bartlett et al. 2017 ; Arora et al. 2018 — marges contre compression.
* Dziugaite & Roy 2017, 2018 ; Ambroladze et al. 2006 — PAC-Bayes et a priori
  dépendants des données.
* Tsuzuku et al. 2020 ; Kwon et al. 2021 ; Entezari et al. 2022 ;
  Ainsworth et al. 2023 — platitude invariante de jauge, symétrie de permutation.
* Jiang et al. 2020 — les mesures de généralisation à grande échelle, que la
  planche 06 nuance.
