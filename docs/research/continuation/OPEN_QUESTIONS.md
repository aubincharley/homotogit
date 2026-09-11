---
id: DOC-QUESTIONS
schema_version: 1
updated_at: 2026-09-11
status: unexecuted_research_options
---

# Questions ouvertes, pistes et limites non résolues

[Accueil](README.md) · [Décisions](DECISIONS.md) · [Approche](SCIENTIFIC_APPROACH.md)

**Aucune liste de ce fichier ne constitue un lancement de campagne.** Elle préserve les idées et les questions qui ne doivent pas être confondues avec les résultats acquis.

## Q-01 — Gaussian explicite avant la réduction d'entrée

La dernière proposition de l'utilisateur est de filtrer les RGB avant de réduire leur résolution :

\[
x\mapsto R_{r(e)}\bigl(G_{\tau(r(e))}x\bigr),
\]

puis éventuellement de conserver en plus le Gaussian interne déjà testé. C'est différent du chemin actuel \(R_r x\) suivi du réseau filtré intérieurement.

**Pourquoi la question a du sens :** un préfiltre agit avant le sous-échantillonnage et peut modifier le repliement spectral. L'échec ancien du flou d'entrée fixe sans changement de résolution ne suffit pas à exclure son utilité ici.

**Ce qui est déjà présent :** la réduction bilinéaire actuelle contient un antialiasing. L'expérience ajouterait donc un préfiltre gaussien à ce resize AA, sauf décision explicite de modifier le comparateur. Elle ne serait pas spontanément un contrôle « anti-aliasing contre absence d'anti-aliasing ».

Un exemple possible pour préciser la piste CIFAR serait \(\tau(16)=1\), \(\tau(24)=0,5\), \(\tau(32)=0\), en pixels de l'image originale 32×32. Cet exemple est une formalisation proposée dans la passation : ces valeurs ne sont ni calibrées ni attestées comme décision historique ou protocole exécuté. Avec `Rprog`, ce préfiltre s'éteindrait dès le retour à r=32, alors que le Gaussian interne persiste jusqu'à e=21.

Le contrôle le plus lisible conserve le resize et la trajectoire de résolution, puis compare ajout du préfiltre avec et sans Gaussian interne. La réutilisation des bras existants exige des assets et des profils compatibles ; sinon, des témoins contemporains sont nécessaires. Ne pas modifier simultanément interpolation, AA, loi de sigma et emplacement sans identifier les facteurs.

## Q-02 — Les étapes intermédiaires internes sont-elles nécessaires ?

Le contrôle warm start 1→0 contre 1→0,5→0 a été exécuté **sur les entrées** dans EXP-001. Il n'a pas été répété comme contrôle du Gaussian interne BN positif.

Une comparaison filtre fixe puis retrait contre décroissance progressive, avec même horizon et phase finale, aiderait à distinguer bénéfice d'une intervention initiale et bénéfice de sa progression. Les nombres de niveaux, durées et forces doivent être fixés avant les nouveaux résultats. L'actuelle grille Gplateau/Ggeo/Gmix n'est pas un remplacement exact de ce contrôle.

## Q-03 — Résolution, ordre, force effective du Gaussian

Rprog, Rgentle et Rreverse ne constituent pas une recherche complète sur toutes les tailles ou durées. Le couplage \(\sigma=qg\) induit des hausses de sigma effectif aux changements de résolution. Le contrôle d'ordre avec Gaussian change donc deux trajectoires liées.

Pour isoler une question précise, il faudrait choisir soit de conserver la convention d'échelle et d'assumer ce couplage, soit de fixer une trajectoire effective comparable. Le meilleur choix dépend de l'hypothèse ; aucune équivalence universelle entre sigma et résolution ne peut être obtenue en faisant simplement correspondre leurs valeurs numériques.

## Q-04 — Quelle part du mauvais chemin cible vient de BN ?

Une recalibration BN à poids fixés sur les seules données d'entraînement serait un diagnostic distinct. Elle permettrait d'observer ce qui est récupérable par les statistiques, tout en sachant que la fonction sans filtre reste différente. Aucun contrôle de ce type n'est présent dans les résultats disponibles. Les métriques finales après neuf époques à la cible ne nécessitent pas de retrait anticipé des filtres.

## Q-05 — Pourquoi le changement de signe entre GN et BN ?

Le changement initial combinait architecture, capacité, normalisation et initialisation. Le retour au petit ResNet-20 a conservé BN et l'initialisation corrigée, mais changé aussi durée et continuation. On sait désormais qu'un grand ResNet-18 n'est pas nécessaire au gain observé ; on ne sait pas si BN seule ou l'initialisation seule suffirait.

Une décomposition factorielle BN/GN × initialisation serait informative si ce mécanisme devient une priorité. Elle n'a pas été exécutée et n'est pas nécessairement le meilleur prochain usage du budget face au transfert à un autre dataset.

## Q-06 — STL-10 et changement d'échelle

STL-10 est envisagé pour donner davantage de sens à la résolution : ses images sont 96×96, avec 5 000 exemples étiquetés d'entraînement, 8 000 de test et un jeu non étiqueté distinct. Une étude supervisée sur les 5 000 images n'est pas le protocole semi-supervisé exploitant les données non étiquetées. Ces caractéristiques sont à revérifier dans la [description officielle](https://cs.stanford.edu/~acoates/stl10/) lors de la préparation.

L'augmentation de résolution ne suffit pas à prédire le temps total. À architecture comparable, le coût spatial nominal par image croît comme le carré du côté : 96/32 donne un facteur 9. Mais 5k exemples contre 50k donne dix fois moins d'images par époque. À même nombre d'époques, une estimation grossière des seules convolutions serait donc 0,9 fois le travail du CIFAR complet ; à même nombre d'updates et batch, elle serait autour de 9 fois. **Ni l'un ni l'autre n'est une estimation mesurée du temps GPU.**

À batch effectif 128 avec dernier groupe conservé, 5 000 images donnent 40 updates par époque ; 30 époques ne font que 1 200 updates. Le dernier groupe a huit images. Il faudrait adapter explicitement la gestion de batch/BN, le budget en updates et le chronométrage. Des résolutions 48→72→96 sont une transposition possible, pas un protocole lancé.

## Q-07 — Mesurer un compromis précision/coût plus solide

La grille compare des budgets d'updates identiques, pas des temps identiques. Stem_max sans Gaussian est intéressant parce qu'il approche la meilleure accuracy pour moins de temps, pas parce qu'un rapport accuracy/seconde a une valeur théorique particulière.

Une suite pourrait comparer performance à temps égal ou temps pour atteindre un seuil préspécifié, avec toute la phase initiale comptée. Il faut préserver la distinction entre microbenchmark, entraînement complet, préparation et évaluations. Les small kernels sous-utilisant la T4 restent une explication à profiler, pas un fait établi par un compteur d'occupation.

## Q-08 — Autres pistes documentées, non évaluées

### Diffusion TV à coût borné

Pour \(E_\varepsilon(u)=\sum\sqrt{\|Du\|^2+\varepsilon^2}\), M étapes :

\[
u^{(0)}=h,\qquad
u^{(m+1)}=u^{(m)}-\eta(s)D^\top\!
\left(\frac{Du^{(m)}}{\sqrt{\|Du^{(m)}\|^2+\varepsilon^2}}\right),
\quad \eta(s)=(1-s)\eta_0.
\]

M=3 ou 5 était une idée. Avec epsilon fixé et \(\|D\|^2\le8\), \(\eta_0\le\varepsilon/8\) est une borne conservative de descente de cette énergie lissée. Les moyennes sont conservées par l'adjoint avec les bords spécifiés. Cela ne donne ni le budget TV exact ni la projection H⁻¹. Les contraintes RGB ne se transfèrent pas aux activations signées. [Note originale](sources/note_tv_cout_fixe.md).

### Transitions de résolution par mélange

Un mélange d'images sur une grande grille ou un mélange de logits à deux résolutions peut adoucir une transition. Le mélange de logits \((1-\alpha)f_\theta(x_r)+\alpha f_\theta(x_{2r})\) demande deux forwards à l'intérieur de l'intervalle ; sa CE n'est pas la moyenne des deux CE. Une image suréchantillonnée ne produit pas nécessairement les mêmes logits que sa version basse résolution directement entrée dans le CNN. Aucune de ces variantes n'est le benchmark courant. [Note originale](sources/note_filtrage_resolution_progressive.md).

### Compression et théorie

Budgets de bits réellement mesurés, chemins d'activations, régularisation décroissante, lissage en espace des paramètres, suivi adaptatif et représentation apprise de chemins restent dans la cartographie. Ils ne possèdent pas de résultats CIFAR dans ce dossier. L'article de déconvolution `Homotopy.pdf` porte sur un autre problème ; ses garanties ne se transportent pas directement aux CNN.

## Q-09 — Compléments de preuve à demander au dépôt

Priorité documentaire : configs complètes et commits des runs, code exact des réductions et de Gmix, normalisation 50k, bornes des paliers pilotes, logs par epoch, scripts des figures corrigées, données de contrôle LR, manifests et empreintes. Les numéros de version PyTorch seuls ne prouvent pas l'identité de tout l'environnement. Cette collecte améliore la solidité sans nouveaux entraînements.

## Q-10 — Le flou interne est-il un anti-aliasing ? (partiellement tranchée)

[EXP-012](experiments/EXP-012_aa_ablation.md) donne un faisceau mitigé et **ne tranche pas**.
Ce qui reste à faire pour conclure, par ordre de coût croissant :

* **Anti-aliaser les shortcuts option-A séparément des convolutions stridées.** `x[:,:,::2,::2]`
  ne retire quasiment rien du repliement (0,171 contre 0,178 non filtré) et n'est **jamais** hooké
  par `attach_sites`. C'est le chemin le plus aliasé du réseau et il n'a jamais été touché.
* **`conv_out` à σ ≈ 0,65** — séparerait le placement du dosage effectif au groupe A, le
  confondant que EXP-012 n'a pas levé.
* **Faire bouger le facteur de décimation** pour voir si l'optimum de σ se déplace avec lui.
  C'est la seule mesure qui validerait une signature de Nyquist ; voir [CORRECTIONS](CORRECTIONS.md) C-37.

## Q-11 — Le plafond vers 80 % est-il un déficit de biais inductif ?

Cinq interventions sans rapport atterrissent dans une bande de 0,5 point autour de 78,5 % à
résolution constante, et l'ensemble des bras d'EXP-012 plafonne vers 81 %. Le protocole n'a
**aucune augmentation de données**. Si le plafond est bien un biais inductif manquant, le même
protocole **avec augmentation** devrait faire largement s'évaporer tous ces gains. Non exécuté.

## Q-12 — `input_max` n'a jamais été testé

[INTEGRATION](INTEGRATION.md) C-42 établit que les cellules `input_max` d'EXP-011 sont des
re-exécutions de `input_bilinear`. Le max-pooling **sur l'image** n'a donc jamais été mesuré,
et le confondant lieu/opérateur d'EXP-012 (C-39) ne peut pas être levé sans lui.
