---
id: EXP-012
schema_version: 1
updated_at: 2026-09-10
status: completed_numeric_verified
evidence: local_final_json_from_own_runs_code_written_in_session
protocol: P-FULL-R20-BN (batch appliqué directement, voir §3)
source_json: ../../../../results/kaggle_outputs/per-layer-sigma-20260910-094757/per_layer_sigma_20260910-094757/study_summary.json
source_json_2: ../../../../results/kaggle_outputs/per-layer-adaptive-fix-20260910-130520/per_layer_adaptive_fix_20260910-130520/study_summary.json
code_commit_reported: 2bea3e9
code_commit_inspected: true
code_commit_inspected_on: 2026-09-10
procedure_deviation: fiche rédigée après exécution, pas de fiche `planned` préalable (voir §7)
---

# EXP-012 — Profils de sigma par couche et contrôleur prédicteur-correcteur

[Index](INDEX.md) · [Corrections](../CORRECTIONS.md) · [État actuel](../CURRENT_STATE.md) · [EXP-011](EXP-011_full_grid.md)

## 1. Question

Le Gaussian interne applique aujourd'hui **le même sigma aux 19 sites**. Deux
questions distinctes :

1. **Un profil en profondeur aide-t-il ?** Comparer un sigma uniforme à des
   profils fixes `c_l = rho^stage(l)` normalisés à `max_l c_l = 1`.
2. **Un contrôleur peut-il découvrir ce profil ?** Un schéma prédicteur-correcteur
   qui part d'un profil uniforme, mesure `dL/dsigma_l` et adapte le pas.

Ce que la comparaison **ne** permet pas : conclure sur un mécanisme. Les profils
changent simultanément l'amplitude par site et le calendrier effectif ; aucune
identification causale n'est faite ici.

## 2. Bras

Sites : `stem + stage 1` = sites 0–6, `stage 2` = 7–12, `stage 3` = 13–18.

| Bras | `c` (stem+s1 / s2 / s3) | Intention |
|---|---|---|
| `plain` | — | témoin sans filtre |
| `rho1` | 1.00 / 1.00 / 1.00 | calendrier uniforme actuel (style CBS) |
| `rho0.5` | 1.00 / 0.50 / 0.25 | = `width_l/32`, échelle physique constante |
| `rho2` | 0.25 / 0.50 / 1.00 | flouter en profondeur |
| `adaptive` | découvert | prédicteur-correcteur, départ uniforme à 1.0 |

`sigma_l(e) = q_l(e) · L[e][l]`, `q_l = 1` partout (r=32 constant), donc la
largeur effective vaut exactement `c_l · G(e)`.

## 3. Protocole

`P-FULL-R20-BN` : CIFAR-10 officiel 50 000 / 10 000, 30 époques, 391
updates/époque = 11 730 updates, ResNet-20 BN, pic LR 0,005, momentum 0,9,
wd 5e-4, warmup 60 puis cosine, sans augmentation, graines 0/1/2, filtres
contournés à partir de l'époque 21 — neuf époques complètes sur l'objectif cible.

**Écart au profil, consigné et non masqué** : batch 128 appliqué directement, et
non en quatre micro-batches de 32. L'accumulation de gradient est exacte
(3,97e-16 en float64, `results/grad_accumulation_check.json`), donc seul le calcul
des statistiques de BatchNorm change. L'écart est identique pour tous les bras ;
la comparaison appariée n'est pas affectée, mais les valeurs absolues ne sont pas
strictement comparables à EXP-011.

## 4. Appariement

Auto-apparié **à l'intérieur de chaque graine** : poids initiaux, sous-ensemble
d'entraînement, sous-ensemble de test, batch de sonde et toutes les permutations
par époque sont construits une fois par graine et revérifiés par chaque bras avant
son premier update ; un écart interrompt avant l'entraînement.

`scripts/aggregate_runs.py` recalcule l'appariement en hachant les artifacts du
dossier plutôt qu'en faisant confiance à un digest déclaré. Il résout
**3 groupes d'appariement de 5 runs**, un par graine.
Les assets ont été régénérés depuis la graine et acceptés **uniquement parce que
leurs digests égalent ceux enregistrés pendant l'entraînement**
(`pairing_verification.json → regenerated_assets.all_match = true`).

## 5. Résultats

### 5.1 Profils fixes — trois graines, run `per-layer-sigma-20260910-094757`

| Bras | acc moyenne ± SD | Δ vs plain (apparié) | CE |
|---|---:|---:|---:|
| plain | 0,7442 ± 0,0029 | — | 0,858 |
| **rho1** | **0,7765 ± 0,0074** | **+3,23 pt** (SD 0,69) | **0,657** |
| rho0.5 | 0,7465 ± 0,0067 | +0,23 pt (SD 0,80) | 0,790 |
| rho2 | 0,7665 ± 0,0021 | +2,23 pt (SD 0,33) | 0,776 |

`rho1 − rho2 = +1,00 pt`, positif sur les trois graines. `rho0.5` change de signe
selon la graine et n'est pas distinguable du témoin.

### 5.2 Contrôleur adaptatif — run `per-layer-adaptive-fix-20260910-130520`

| Bras | acc moyenne ± SD | Δ vs plain | CE |
|---|---:|---:|---:|
| plain | 0,7391 ± 0,0062 | — | 0,870 |
| **rho1** | **0,7766 ± 0,0068** | **+3,74 pt** (SD 0,54) | **0,655** |
| adaptive | 0,7735 ± 0,0049 | +3,44 pt (SD 0,83) | 0,691 |

`adaptive − rho1 = −0,30 pt` (SD 0,50 ; par graine +0,01 / −0,88 / −0,04), CE
nettement moins bonne. **Le calendrier uniforme reste le meilleur testé.**

### 5.3 Plancher de bruit mesuré — nouveau

`plain` et `rho1` réexécutés à configuration identique :

| Bras | \|run2 − run1\| par graine | moyenne |
|---|---|---:|
| plain | 0,54 / 0,02 / 0,96 | **0,51 pt** |
| rho1 | 0,25 / 0,87 / 0,59 | **0,57 pt** |

**≈ 0,5 point** à cette échelle, cinq fois le 0,1 point mesuré à 10 000 images.
Deux exécutions ne définissent aucune distribution (voir C-12/C-13) : c'est un
écart observé, pas un seuil de signification. `adaptive − rho1 = −0,30 pt` est
**dans ce bruit** : on peut dire que l'adaptatif ne bat pas l'uniforme, pas qu'il
est moins bon.

## 6. Ce que le contrôleur découvre

Figure : [`results/adaptive_trajectory.png`](../../../../results/adaptive_trajectory.png).

Parti d'un profil uniforme à 1,0, il différencie réellement les sites (écart
max−min sur les 19 sites : 0,62 à l'époque 12). La forme découverte est
**inverse de rho2** : les sites profonds s'annulent d'abord, le stem conserve le
flou le plus longtemps (époque 12 : stem 0,743 contre ≈ 0,46 pour stages 2–3).

C'est la direction de `rho0.5`, **le pire profil fixe**. La notion de « chemin
raide » du contrôleur et le profil empiriquement meilleur pointent donc dans des
directions opposées. Ce constat est plus informatif que l'écart d'accuracy.

## 7. Limites et écarts de procédure

- **Fiche rédigée après exécution.** MAINTENANCE §4 demande une fiche `planned`
  avant l'exécution, avec liste de cellules et calendriers gelés. Cela n'a pas
  été fait : les runs ont précédé la fiche. Cette fiche est rétrospective et ne
  doit pas être lue comme un préenregistrement.
- **Exposition non strictement appariée.** `adaptive` a 7 820 updates filtrés
  contre 8 211 pour les bras fixes (20 époques contre 21). L'écart est petit et
  défavorable à l'adaptatif, donc il ne sauve pas la comparaison, mais
  l'appariement en exposition n'est **pas** exact.
- **Le contrôleur n'est pas une méthode reproductible depuis sa spécification** :
  le chemin dépend des poids. Seule la table réalisée émise permet de le rejouer.
  Le rejeu bit-à-bit est testé (`tests/test_adaptive_continuation.py`).
- **Un premier run adaptatif a été invalidé** par un défaut de code, voir C-32.
- Trois graines et un test déjà exposé justifient une lecture exploratoire située.

## 8. Artifacts

- `results/kaggle_outputs/per-layer-sigma-20260910-094757/per_layer_sigma_20260910-094757/`
- `results/kaggle_outputs/per-layer-adaptive-fix-20260910-130520/per_layer_adaptive_fix_20260910-130520/`
- Figures : `results/per_layer_sigma_profiles.png`, `results/adaptive_trajectory.png`
- Code : `continuation/adaptive.py`, `continuation/campaign_ops.py` (table par site),
  `continuation/transforms/gaussian.py` (chemin différentiable en sigma),
  `scripts/study_per_layer_cpu.py`, `scripts/normalize_per_layer_outputs.py`
- Tests : `tests/test_per_layer_sigma.py`, `tests/test_adaptive_continuation.py`
