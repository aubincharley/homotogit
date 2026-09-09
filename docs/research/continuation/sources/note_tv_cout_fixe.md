# Piste en réserve : continuation TV à coût fixé

Décision du 8 septembre 2026, projet homotopie / apprentissage profond avec Joseph Gabet.

## Contexte et statut

Les expériences de continuation gaussienne sur les entrées de CIFAR-10 n'ont pas montré de bénéfice à budget égal. Les aperçus TV–L² et TV–Ḣ⁻¹ montrent une simplification spatiale réelle, mais pas une extraction systématique de la forme discriminante. Leur projection convexe très précise sur un budget relatif de TV est trop coûteuse dans l'implémentation CPU actuelle pour être insérée dans toutes les couches. Les activations changent avec les poids : un cache de données prétransformées ne résout pas ce problème interne au réseau.

La prochaine étape autorisée est un prototype de seuillage en ondelettes non décimées : aperçus et mesures de coût, sans entraînement. La variante TV ci-dessous reste une piste pour plus tard ; elle n'est ni implémentée ni évaluée.

## Variante proposée : un petit nombre d'étapes de diffusion TV régularisée

Pour une carte d'activation multicanal h, utiliser D=(D₁,D₂), avec différences spatiales avant de pas 1 et différence nulle au bord extérieur, comme dans les aperçus TV. D agit indépendamment sur chaque canal, et Dᵀ est son véritable adjoint.

Pour ε>0, définir

\[
E_\varepsilon(u)=\sum_{c,p,q}\sqrt{|(Du)_{c,p,q}|^2+\varepsilon^2}.
\]

Fixer M, par exemple 3 ou 5. Pour une progression s∈[0,1], poser

\[
u^{(0)}=h,\qquad
u^{(m+1)}=u^{(m)}-\eta(s)D^\top
\left(\frac{Du^{(m)}}{\sqrt{|Du^{(m)}|^2+\varepsilon^2}}\right),
\qquad T_s(h)=u^{(M)},
\]

avec η(s)=(1−s)η₀. Les divisions et normes du gradient sont locales, sur les deux directions spatiales. Alors T₁=I. Pour ε fixé, la borne ||D||²≤8 donne une constante de Lipschitz du gradient ≤8/ε ; η₀≤ε/8 est un choix conservateur assurant la décroissance de Eε à chaque étape. Cette propriété concerne Eε, pas l'ancien budget exact de TV.

Il s'agit d'une nouvelle transformation, définie par M étapes de descente, et non de la solution de la projection précédente ou de son proximal. L'attache à h provient ici de l'initialisation et du temps court de diffusion. La moyenne de chaque canal est préservée car D annule les constantes. Les grandes transitions diffusent moins que les zones de faible gradient, sans garantie de conservation des contours sémantiques.

## Points à fixer avant une éventuelle implémentation

- Calibrer ε et η₀ selon l'échelle des activations. La justification de stabilité précédente suppose ε fixé pendant les M étapes ; préciser toute dépendance à h et sa différentiation.
- Ne pas recopier les contraintes RGB [0,1] sur des activations internes signées. Ne pas conserver artificiellement la variance.
- Différencier les M étapes réellement exécutées. Le coût est borné par M, mais la vitesse et la mémoire du forward/backward doivent être mesurées sur les dimensions réelles des couches.
- Garder le gaussien interne comme référence. Un résultat positif publié pour ce dernier n'a pas encore été reproduit dans notre protocole.
- Des solveurs plus efficaces et des tolérances moins strictes pourraient également accélérer une approximation de la projection d'origine ; ce serait une autre option, à distinguer explicitement de cette diffusion.

## Repère de coût

Le JSON des aperçus donne environ 9,5 s/image pour TV–L² à t=0,5 et 43,6 s/image à t=0,25. Extrapolées naïvement en séquentiel à 45 000 images, ces valeurs représentent 119 h et 545 h, et non 2,5 h et 22 h. Elles décrivent dix images et cette implémentation, pas un coût intrinsèque de la TV.
