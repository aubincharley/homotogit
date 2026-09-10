---
id: DOC-CORRECTIONS
schema_version: 1
updated_at: 2026-09-09
status: interpretive_errata
---

# Corrections à conserver avec l'historique

[Accueil](README.md) · [Évaluation](EVALUATION.md) · [Sources](SOURCES.md)

Les rapports originaux sont conservés sans réécriture. Ce fichier explicite les phrases qui ne doivent pas être reprises telles quelles dans une synthèse scientifique ou un prompt futur.

| ID | Affirmation ou confusion historique | Correction et portée |
|---|---|---|
| C-01 | Un bon résultat train final exclut un bénéfice d'optimisation | Il n'exclut ni accélération à budget plus court ni modification utile de la trajectoire ; optimisation et régularisation implicite ne sont pas disjointes |
| C-02 | Les grandes valeurs de sigma ne peuvent jamais être de bons warm starts | L'étude à flou fixe ne mesure pas l'adaptation ultérieure ; les résultats négatifs sont situés |
| C-03 | Le filtrage Gaussian est forcément un appauvrissement Shannon | Atténuation, conditionnement, quantification et information mutuelle sont distincts ; le multiplicateur continu n'est pas nul |
| C-04 | Rayon 4 sur carte 4×4 rend la spécification mathématiquement impossible | La restriction porte sur le padding natif ; une réflexion répétée explicite conserve le noyau et l'architecture |
| C-05 | Environ 40 fois plus de paramètres signifie 40 fois plus de temps | Le rapport de MACs avait été estimé autour de 13,7 dans la discussion ; seul un chronométrage sur le matériel réel établit le temps |
| C-06 | Former sur 50k et garder l'ancien sous-ensemble de 5k comme validation | Ces 5k appartiennent désormais au train ; les appeler validation indépendante serait une fuite |
| C-07 | Accumuler 4×32 donne une BatchNorm sur 128 | BN voit chaque microbatch, donc 32, puis éventuellement 16 en fin d'époque |
| C-08 | Bypassed est le témoin plain | Bypassed est une autre évaluation des mêmes poids du bras filtré ; plain est un entraînement distinct |
| C-09 | Toute la chute bypassed est « purement/entièrement BN » | Retirer le filtre change aussi la fonction. BN est une explication plausible importante, pas une contribution isolée expérimentalement |
| C-10 | Pas de saut visible signifie transition sans aucun coût | Les checkpoints ne résolvent pas toutes les updates ; un coût bref peut passer entre les mesures |
| C-11 | Sigma 0,30 au checkpoint 1700 prouve que l'extinction a échoué | La dernière update effectuée est k=1699 ; la suivante k=1700 utilise le bypass |
| C-12 | Deux plain à 55,36 et 55,40 % donnent un bruit de ±0,1 point | Leur différence observée est 0,04 point. Deux exécutions ne définissent ni distribution ni plancher universel |
| C-13 | En db2, le gain est « neuf fois le bruit », donc presque établi | L'estimation du bruit n'est pas justifiée ; +0,94 point sur une graine reste un résultat préliminaire |
| C-14 | db2 s=0 est à environ 6 % du coarse-only | Aucune mesure ni normalisation retrouvée ; affirmation retirée |
| C-15 | L'epsilon RMS a toujours un effet relatif de 5e−11 | Cet ordre de grandeur dépend de la bande ; il n'est pas valable près d'une énergie nulle |
| C-16 | L'extrapolation db2 à six bras complets donne environ 48 h GPU | Avec 8 211 updates filtrées par run et 2,06–2,10 s/update, on obtient environ 28–29 h GPU avant évaluations et surcoûts ; le coût reste très élevé, mais 48 h ne suit pas ces chiffres |
| C-17 | Les anciennes projections TV coûteraient 2,5 h / 22 h sur 45k images | Les exemples 9,5 s et 43,6 s par image donnent environ 119 h et 545 h en séquentiel ; ce ne sont pas des coûts intrinsèques optimisés |
| C-18 | La petite résolution promet environ 24 % de temps économisé | Le ratio de travail spatial du calendrier est 0,7625 ; les mesures T4 n'établissent pas un tel gain de temps |
| C-19 | Les petits kernels ne saturent pas le GPU : cause démontrée | C'est une explication plausible de la faible économie temporelle ; un profilage détaillé manque |
| C-20 | Après stem, réduire est toujours moins bon ; max avec Gaussian est toujours moins bon | À emplacement stem fixé, max bat bilinéaire avec et sans Gaussian. Comparer stem_max à input_bilinear change deux facteurs ; voir EXP-011 |
| C-21 | Rreverse sans Gaussian est équivalent à Rprog | Différence moyenne petite, signes mixtes sur trois graines ; absence de preuve d'écart n'est pas preuve d'équivalence |
| C-22 | Rreverse avec Gaussian ne change que l'ordre des images | Le facteur r/32 change aussi le calendrier effectif des sigmas |
| C-23 | Les gains sous-additifs prouvent que résolution et Gaussian font la même chose | C'est une hypothèse de mécanisme compatible, pas une identification causale ; échelle de métrique et couplage des paramètres interviennent |
| C-24 | Les 11 anciens runs ont été réutilisés dans le benchmark 63 cellules | La recherche d'artifacts échouait dans les kernels ; tous les 63 ont été entraînés fraîchement. Partition correcte, coût supplémentaire réel |
| C-25 | Répliquer avec les mêmes assets donne une expérience indépendante au sens fort | Cela fournit une nouvelle exécution comparable ; ni indépendance statistique totale ni seule cause des écarts ne sont établies |
| C-26 | Le JSON final suffit à redessiner toutes les courbes | Il contient les valeurs finales par graine, pas les historiques par époque |
| C-27 | Antialiasing signifie que le resize bilinéaire n'est plus linéaire | Les coefficients changent, mais restent indépendants de l'image ; l'opérateur reste linéaire en intensités |
| C-28 | La résolution remet d'abord une image 16×16 à 32×32 avant le réseau | Les expériences utilisent réellement la petite grille ; aucune remontée systématique avant le CNN |
| C-29 | Les contrastes TV et ondelettes étaient définis identiquement | Le JSON ondelettes recentre z sur sa propre moyenne, le prompt TV sur celle de x ; la petite dérive db2 empêche une identité exacte |
| C-30 | Le statut/plan du 8 septembre décrit la suite actuelle | Depuis : pilote db2 exécuté puis branche abandonnée ; résolution et grande grille terminées ; préfloutage RGB encore non testé |
| C-31 | Le profil « flouter en profondeur » (rho=2) bat le sigma uniforme | Observé à 6 000 images sur **une** graine (+2,45 contre +1,35 pt), non répliqué. Sur 50 000 images et trois graines, `rho1` bat `rho2` de +1,00 pt, positif sur les trois graines, et sur la CE l'écart est plus net encore. Le plancher de bruit à l'échelle du pilote n'avait pas été mesuré ; il vaut ≈ 0,5 pt (EXP-012 §5.3), soit la moitié de l'effet annoncé. Ne pas reprendre « flouter en profondeur aide » |
| C-32 | Le premier run adaptatif mesure une adaptation | Le run `per-layer-sigma-20260910-094757` a émis une **rampe linéaire parfaitement uniforme** : `delta_ref = 0,15`, valeur héritée de la paramétrisation en sigma antérieure au passage à phi, rendait tous les pas inférieurs à `dmin`, donc les 19 sites étaient rabotés à la même valeur. La sonde fonctionnait (19/19 gradients non nuls, dispersion 32–148x) ; c'est la règle de pas qui jetait le signal. Son +3,08 pt est celui d'une rampe linéaire uniforme, pas d'un contrôleur adaptatif. Corrigé et réexécuté dans `per-layer-adaptive-fix-20260910-130520` |

## Correction particulière du compte rendu de la grille

Les différences **stem_max − stem_bilinear** recalculées depuis le JSON sont :

- sans Gaussian : **+1,06 point**, SD 0,49 ;
- avec Gaussian paliers : **+0,51 point**, SD 0,19.

Les trois graines sont positives dans chaque comparaison. Le rapport dit que le max « aide sans Gaussian et nuit avec » parce qu'il prend **input_bilinear** comme référence dans les deux cas. Cette phrase ne répond pas à la question de l'agrégateur à emplacement fixé. La bonne conclusion est une interaction entre lieu, opérateur et Gaussian, avec un meilleur résultat global pour bilinéaire en entrée + Gaussian.

## Un point supplémentaire découvert dans les fichiers TV

Les JSON historiques contiennent des valeurs `NaN` pour certaines fidélités aux endpoints. Ce n'est pas du JSON strict interopérable et cela ne signifie pas que la fidélité mathématique au point t=0 soit indéfinie. Les copies sources restent intactes ; pour un nouveau registre, utiliser `null` avec une raison explicite « non calculé » et documenter la migration.
