# Prompt — Residual Homotopy sur ResNet-18 / CIFAR-10

*(À coller tel quel dans une session Claude Code ouverte à la racine de ce repo.)*

---

Tu travailles dans `homotogit`, un baseline CIFAR-10 / ResNet-18 volontairement minimal
et opinionated (≈94-95% test acc, 100 epochs, 3 seeds). Lis `README.md`,
`src/cifarbase/model.py`, `src/cifarbase/train.py` et `src/cifarbase/config.py` avant
d'écrire quoi que ce soit : le style du code (commentaires qui expliquent *pourquoi*,
pas *quoi*) est une contrainte, pas une suggestion.

Objectif : implémenter une **homotopie résiduelle** et le protocole expérimental complet
qui permet de décider si elle sert à quelque chose.

---

## 1. L'objet mathématique

On paramètre chaque bloc résiduel par `s_k ∈ [0,1]` :

    h_{l+1} = ReLU( shortcut(h_l) + s_l · F_l(h_l) )

avec `F_l = BN(conv3x3(ReLU(BN(conv3x3(·)))))`, et **`s` ne multiplie jamais le
shortcut**. Le problème d'optimisation devient une famille

    L_s(θ) = (1/n) Σ_i ℓ( f_{θ,s}(x_i), y_i ),      s = (s_1, …, s_K), K = 8

et on fait varier `s` *pendant* l'entraînement : `s(t) : 0 → 1`. Le ResNet standard est
`s ≡ 1`. La question est : **suivre θ*(s) de s=0 à s=1 est-il une meilleure façon
d'atteindre θ*(1) que d'optimiser directement L_1 ?**

---

## 2. Trois faits à intégrer dans la conception (non négociables)

**(a) `s` est une reparamétrisation, pas un changement de classe de fonctions.**
`F_l` se termine par `bn2`, donc `s · BN(·)` ≡ `BN` avec `γ_bn2 → s·γ_bn2`. Pour tout
`s > 0` l'ensemble `{f_{θ,s} : θ}` est **identique** à `{f_{θ,1} : θ}`. L'homotopie
n'ajoute aucune capacité ; elle change la géométrie de la paramétrisation et donc la
trajectoire. Ce qui brise l'équivalence exacte : le weight decay (pénalise `γ`, pas
`s·γ`), le momentum SGD, et le LR effectif sur la branche résiduelle (`∂L/∂θ_F ∝ s`,
donc `s` agit comme un multiplicateur de learning rate branche-spécifique).

→ **Conséquence obligatoire** : il faut un bras de contrôle qui applique le même profil
temporel comme *multiplicateur de LR sur les paramètres de la branche résiduelle*, sans
toucher au forward. Si ce contrôle reproduit l'effet de l'homotopie, alors l'homotopie
n'est qu'un schedule de LR déguisé, et c'est le résultat principal de l'expérience.

**(b) À `s = 0`, `∂L/∂θ_{F_l} = 0` exactement.** Les convs de la branche résiduelle sont
gelées. Mais le forward est toujours calculé, donc les `running_mean/var` de `bn1`/`bn2`
continuent de dériver en mode train sur des activations que personne n'optimise. Deux
options, à exposer en config : `s_min > 0` (recommandé, ex. 0.02), ou `s_min = 0` assumé
avec un log explicite. Vérifie et documente le comportement des stats BN dans ce régime.

**(c) `zero_init_residual: true` (défaut du repo) met déjà `γ_bn2 = 0`.** La baseline
*démarre déjà* à `s = 0` effectif et remonte toute seule. C'est un confondant direct.
Ne le désactive pas en douce : rends-le explicite, et fais tourner l'ablation
`zero_init_residual ∈ {true, false} × arm ∈ {baseline, homotopy}` sur 1 seed pour
mesurer l'interaction avant de lancer la grille complète.

---

## 3. Ce qu'il faut écrire

### `src/cifarbase/model.py` (modification)

- `BasicBlock` reçoit un attribut `self.s: float = 1.0` — un **float Python nu**, pas un
  `Parameter`, pas un `buffer` : le `state_dict` doit rester bit-compatible avec les
  checkpoints baseline (`train.py:124` en sauvegarde un). Le mode `s_learnable` est la
  seule exception et doit être traité séparément.
- `forward` : `return F.relu(self.s * out + self.shortcut(x), inplace=True)`.
  Quand `self.s == 1.0`, le résultat doit être **bit-identique** à l'actuel — écris le
  test qui le prouve.
- Helpers au niveau module : `residual_blocks(model) -> list[BasicBlock]` (dans l'ordre
  du forward : layer1[0], layer1[1], layer2[0], …), `set_s(model, s)` où `s` est un
  scalaire ou une séquence de longueur K, et `get_s(model) -> list[float]`.

### `src/cifarbase/homotopy.py` (nouveau)

Un planificateur pur, **sans état et testable sans torch** : `s_at(progress, cfg) ->
list[float]` où `progress ∈ [0,1]` est la fraction de `total_steps` écoulée. Schedules :

| `s_schedule` | comportement |
|---|---|
| `const`      | `s ≡ s_max` — reproduit la baseline quand `s_max = 1.0` |
| `linear`     | `s_min → 1` linéairement sur `[s_ramp_start, s_ramp_end]` |
| `cosine`     | même support, montée en demi-cosinus (dérivée nulle aux bords) |
| `staircase`  | `s_stairs` paliers constants — la vraie continuation : on laisse θ converger à chaque `s` avant d'incrémenter |
| `sequential` | par-bloc : le bloc `k` monte sur la fenêtre `[k/K, (k+1)/K]` du support, mise à l'échelle — continuation *en profondeur* |
| `learned`    | contrôle ReZero : un `nn.Parameter` scalaire par bloc, init à `s_min`, appris par SGD. Répond à « le réseau *veut*-il un schedule ? » |

**Invariant à faire respecter par une assertion** : pour tout schedule,
`s_at(p) == [1.0]*K` pour tout `p >= s_ramp_end`. Une fraction non nulle de
l'entraînement doit se terminer à exactement `s = 1`, sinon on compare un autre modèle
et le résultat ne veut rien dire. `_validate` doit rejeter `s_ramp_end >= 1.0`.

### `src/cifarbase/config.py` (modification)

Ajoute les clés à `CONFIG` — un YAML qui nomme une clé absente de `CONFIG` lève
(`config.py:107`), donc rien ne marche tant que ce n'est pas fait. Attention :
`_CASTS` ne connaît que `bool/int/float/str` (`config.py:81`), donc pas de listes —
un vecteur `s` se passe en `str` CSV et se parse comme `seed_list`.

    "s_schedule": "const",     # const|linear|cosine|staircase|sequential|learned
    "s_min": 0.0,
    "s_ramp_start": 0.0,       # fraction de l'entraînement
    "s_ramp_end": 0.5,         # s == 1 après ce point
    "s_stairs": 0,             # staircase seulement
    "s_granularity": "step",   # step|epoch — un s par pas ou un s par epoch
    "lr_gate_control": False,  # bras de contrôle (a) : profil s appliqué au LR de la
                               # branche résiduelle, forward inchangé
    "landscape": False,
    "landscape_every": 10,     # epochs

Étends `_validate` : schedule inconnu → `SystemExit` avec la liste des valides ;
`s_ramp_start < s_ramp_end < 1.0` ; `s_stairs >= 1` si `staircase`. Suis exactement le
style des messages d'erreur existants (`!! …: pick one of …`).
Ajoute aussi `s_schedule` / `s_ramp_end` à la ligne d'en-tête de `print_report`.

### `src/cifarbase/train.py` (modification)

- Dans la boucle de batch, avant le forward : `set_s(model, s_at(step/total_steps, cfg))`
  (ou par epoch si `s_granularity == "epoch"`).
- **Évaluation** : il y a deux mesures différentes et il faut logger les deux —
  `test_acc` à `s(t)` courant (« le modèle tel qu'il tourne ») et `test_acc_at_s1` avec
  `s` forcé à 1 (« le ResNet qu'on obtiendrait si on arrêtait maintenant »). La deuxième
  est celle qui rend la courbe comparable à la baseline. Restaure `s` après.
- Ajoute `s_mean` et `s_min_block` à la ligne d'historique.
- `lr_gate_control` : deux `param_groups` (branche résiduelle vs le reste), le groupe
  résiduel voit son LR multiplié par `s_at(...)`. Le forward reste `s = 1`.
- Le `best["state"]` du checkpoint doit enregistrer le `s` courant à côté, sinon
  `_describe` mesure un modèle dont on ignore le point d'homotopie.

### `src/cifarbase/landscape.py` (nouveau — diagnostics)

Vit dans `src/` (donc embarqué par `build.py`, qui globe tout `src/`). Coûteux, appelé
tous les `landscape_every` epochs seulement, gardé par `cfg["landscape"]`.
Tout doit tourner en `model.eval()` sur un **batch fixe** (le même à chaque appel, tiré
une fois) — sinon le « Hessien » est du bruit et les courbes sont ininterprétables.

- `loss_vs_s(model, split, grid)` — `L_s(θ_t)` sur une grille de `s` à θ figé. C'est la
  coupe de la surface de perte le long de la direction d'homotopie : elle dit si le
  chemin est plat, monotone, ou barré.
- `residual_ratio(model, batch)` — `‖s·F_l(h)‖ / ‖shortcut(h)‖` par bloc. **Le
  diagnostic le moins cher et le plus informatif** : c'est la profondeur effective réelle,
  et il révèle si `γ_bn2` compense simplement `s` (fait (a)). Si le ratio est le même
  entre baseline et homotopie à `s` égal, la reparamétrisation a été absorbée.
- `top_hessian_eigs(model, batch, k=2)` — power iteration sur des produits
  Hessien-vecteur (`torch.autograd.grad(..., create_graph=True)`), avec critère d'arrêt
  et plafond d'itérations.
- `hutchinson_trace(model, batch, n=16)` — trace du Hessien, estimateur de Rademacher.
- `grad_norm_per_block(model, batch)` — vérifie empiriquement le fait (b).
- `interpolate_barrier(state_a, state_b, model, split, n=11)` — connectivité linéaire
  entre la solution homotopie et la solution baseline (même seed). **Recalcule les stats
  BN** par une passe train à chaque point d'interpolation, sinon la barrière mesurée est
  un artefact de BatchNorm, pas de la géométrie.

Écris ces résultats dans `landscape.jsonl` du `run_dir`, une ligne par mesure.

### Configs YAML — `src/cifarbase/configs/`

`homotopy_linear.yaml`, `homotopy_staircase.yaml`, `homotopy_sequential.yaml`,
`homotopy_learned.yaml`, `lr_gate_control.yaml`. Tous identiques à `baseline.yaml`
**sauf** les clés `s_*`, avec un commentaire d'en-tête disant ce que chaque fichier
cherche à falsifier. Plus `homotopy_quick.yaml` calqué sur `quick.yaml` pour le smoke test
CPU.

### `tests/` (nouveau) + `make test`

pytest en dépendance dev optionnelle dans `pyproject.toml`. Au minimum :

1. `s = 1` → forward **bit-identique** au modèle avant modification (même seed, même
   entrée, `torch.equal`).
2. `s = 0` → forward égal à un réseau construit à la main comme stem → 3 projections →
   GAP → fc. Prouve que le shortcut n'est pas gaté.
3. `s = 0` → `p.grad` nul (ou `None`) pour tout paramètre de `conv1/conv2/bn1/bn2` des
   blocs, et non nul pour le stem et `fc`. C'est le fait (b), testé.
4. `residual_blocks(model)` rend les blocs dans l'ordre du forward (vérifie par hooks,
   pas par introspection des noms).
5. Chaque schedule : `s_at(0) == s_min`, `s_at(p >= s_ramp_end) == 1.0`, monotonie,
   longueur K.
6. Équivalence de reparamétrisation : `set_s(m, 0.5)` puis forward == modèle à `s=1`
   avec `bn2.weight *= 0.5` (aux tolérances flottantes près). Ce test *encode le fait (a)* —
   s'il échoue un jour, quelque chose de fondamental a bougé.
7. `make quick` passe toujours, inchangé.

### `analyze.py` (extension)

Panneaux supplémentaires quand les colonnes existent : `s(t)` superposé au LR ;
`test_acc` vs `test_acc_at_s1` ; `L_s(θ_t)` en heatmap (epoch × s) ; `residual_ratio`
par bloc dans le temps. Garde le thème et la palette existants — les couleurs sont fixées
pour être apprises une fois (`analyze.py:31-36`). Rien de tout ça ne doit importer
matplotlib depuis `src/`.

---

## 4. Protocole expérimental

Tous les bras : `resnet18`, `width 64`, `epochs 60`, `seeds 0,1,2`, `augment true`,
même LR schedule cosine + 5 epochs de warmup, **budget de compute identique**.
60 et pas 100 pour le pilote : on cherche une différence de trajectoire, pas le record.

| Bras | Config | Ce que ça teste |
|---|---|---|
| A | `baseline` | référence |
| B | `homotopy_linear`, `s_ramp_end 0.5` | l'homotopie globale continue |
| C | `homotopy_staircase`, 5 paliers | la vraie continuation θ*(s_k) → θ*(s_{k+1}) |
| D | `homotopy_sequential` | continuation en profondeur, `s` par bloc |
| E | `homotopy_learned` | le réseau choisit-il lui-même un profil ? |
| F | `lr_gate_control` | **le bras qui peut tuer l'hypothèse** (fait (a)) |
| G | `baseline` + `zero_init_residual false` | isole le confondant (fait (c)) |

Balaye `s_ramp_end ∈ {0.25, 0.5, 0.75}` sur le bras B, 1 seed, avant de payer 3 seeds :
si l'effet dépend de façon non monotone de la durée de rampe, c'est un signal ; s'il est
plat, B est équivalent à A et il faut le dire.

## 5. Critère de décision, fixé *avant* de regarder les résultats

`print_report` calcule déjà la plus petite différence crédible (`report.py:156-170`) :
max du seuil seed-à-seed (2√2·sem) et du plancher binomial sur 10k images.
**Un bras ne « bat » la baseline que s'il la dépasse de plus que ce seuil.** Écris ce
nombre dans le rapport final avant de conclure quoi que ce soit.

Sois honnête sur le pronostic : le fait (a) rend improbable un gain d'accuracy. Le
résultat attendu est *pas de différence significative en accuracy*, et l'intérêt est
ailleurs — trajectoire, conditionnement, barrière entre solutions. C'est pourquoi les
diagnostics de `landscape.py` ne sont pas optionnels : ils sont ce qui rend un résultat
nul informatif au lieu de le rendre inutile. Un run qui produit « B = A ± bruit » **et**
`loss_vs_s`, `residual_ratio` et la barrière d'interpolation est un résultat publiable ;
le même run sans les diagnostics ne l'est pas.

## 6. Contraintes d'ingénierie

- N'introduis aucune régression sur le chemin baseline : `make quick` et
  `python main.py --config baseline --epochs 1 --seeds 0` doivent tourner à
  l'identique, et le forward à `s = 1` doit être bit-identique.
- `build.py` embarque tout `src/` : les nouveaux modules et YAMLs partent sur Kaggle
  automatiquement, mais rien dans `src/` ne doit importer matplotlib.
- Chaque run écrit déjà dans son propre `runs/<stamp>_<config>_<arch>/` avec son
  `invocation.json` : n'écrase rien, ne réutilise rien.
- Commits atomiques, un par étape (modèle+tests / scheduler+tests / intégration train /
  configs / diagnostics / analyze). Pas de commit sur `main` sans branche.

## 7. Ordre d'exécution

1. Branche `homotopy-residual`.
2. `model.py` + tests 1-4, 6. **S'arrête là et montre-moi les tests qui passent.**
3. `homotopy.py` + test 5.
4. `config.py` + configs YAML + intégration `train.py`. Vérifie `make quick`.
5. Smoke : `homotopy_quick` sur CPU, 2 epochs, et vérifie que `s` monte bien dans
   `history.csv`.
6. `landscape.py` + branchement.
7. `analyze.py`.
8. Lance A, B, F sur 1 seed / 60 epochs et rapporte avant d'élargir la grille.

Ne saute pas l'étape 2 ni l'étape 8. Si un test ne passe pas, dis-le avec la sortie
brute — ne le contourne pas.
