---
id: EXP-013
schema_version: 1
updated_at: 2026-09-11
status: completed_numeric_verified
evidence: local_final_json_from_own_runs_code_written_in_session
protocol: P-FULL-R20-BN (batch appliqué directement, voir EXP-012 §3)
source_json: ../../../../results/kaggle_outputs/sigma0-sweep-20260910-203015/sigma0_sweep_20260910-203015/study_summary.json
figure: ../../../../results/sigma0_sweep.png
code_commit_reported: 949441f
code_commit_inspected: true
code_commit_inspected_on: 2026-09-11
procedure_deviation: fiche rédigée après exécution, comme EXP-012
---

# EXP-013 — Balayage d'amplitude : sigma0 est-il un optimum ou un défaut hérité ?

[Index](INDEX.md) · [EXP-012](EXP-012_per_layer_sigma.md) · [État actuel](../CURRENT_STATE.md) · [Figure](../../../../results/sigma0_sweep.png)

## 1. Pourquoi cette expérience

[EXP-012](EXP-012_per_layer_sigma.md) a montré que la **forme** du calendrier est
un axe plat : le sigma uniforme bat tous les profils en profondeur, et un
contrôleur adaptatif l'égale sans le dépasser. L'axe jamais exploré est
l'**amplitude**.

`SiteController` codait en dur `GaussianSmoothing(sigma_max=1.0)`, et tous les
profils sont normalisés à `max_l c_l = 1`. Donc **`sigma0 = 1.0` plafonne toute la
famille depuis le début** — valeur héritée de l'implémentation de référence, jamais
choisie ni vérifiée. Une valeur par défaut qui n'a jamais été variée n'est pas un
optimum tant qu'on ne l'a pas testée.

## 2. Bras

Six bras par graine, identiques sauf l'amplitude multipliant **un seul calendrier
plateau partagé** ; profil uniforme partout, r=32 constant.

| Bras | sigma crête | rayon | taps |
|---|---:|---:|---:|
| `plain` | — | — | — |
| `s0_0.25` | 0,25 | 1 | 3 |
| `s0_0.5` | 0,50 | 2 | 5 |
| `rho1` | **1,00** | 4 | 9 |
| `s0_1.5` | 1,50 | 6 | 13 |
| `s0_2` | 2,00 | 8 | 17 |

`radius = ceil(4·sigma_max)` suit l'amplitude : chaque bras conserve la **même
troncature relative de ±4 sigma**. Le support en pixels diffère entre bras, la
qualité de l'opérateur est appariée. À `sigma0 = 2` le rayon atteint 8 sur les
cartes 8×8 du stage 3, exactement là où le `reflect` natif de PyTorch refuse et où
le repli explicite par indices prend le relais ; forward et backward finis
vérifiés avant lancement.

## 3. Protocole

Celui d'EXP-012, inchangé, donc lisible contre son plancher de bruit **mesuré** de
≈ 0,5 point : 50 000 / 10 000, 30 époques, 11 730 updates, LR 0,005, warmup 60
puis cosine, sans augmentation, filtres contournés à partir de l'époque 21,
graines 0/1/2 appariées par sha256 (poids initiaux, sous-ensembles, permutations).
`pairing_verification.json → all_match = true`, assets régénérés et acceptés
uniquement parce que leurs digests égalent ceux enregistrés à l'entraînement.

## 4. Résultats

| Bras | sigma crête | acc moyenne ± SD | Δ vs plain | CE |
|---|---:|---:|---:|---:|
| plain | — | 0,7413 ± 0,0073 | — | 0,861 |
| s0_0.25 | 0,25 | 0,7381 ± 0,0096 | −0,32 pt | 0,867 |
| s0_0.5 | 0,50 | 0,7641 ± 0,0086 | +2,27 pt | 0,808 |
| **rho1** | **1,00** | **0,7784 ± 0,0063** | **+3,71 pt** | **0,655** |
| s0_1.5 | 1,50 | 0,7455 ± 0,0038 | +0,41 pt | 0,735 |
| s0_2 | 2,00 | 0,7059 ± 0,0071 | **−3,55 pt** | 0,839 |

`rho1` bat chaque alternative **sur les trois graines** : +1,44 pt contre
`s0_0.5` (SD 0,50), +3,30 pt contre `s0_1.5` (SD 0,49), +7,26 pt contre `s0_2`.
Ces écarts sont nettement hors du plancher de 0,5 point. La CE donne le même
minimum, plus net encore.

## 5. Lecture

- **`sigma0 = 1.0` est un véritable optimum, pas seulement un défaut hérité.** La
  réponse est fortement piquée et le défaut historique tombe dessus.
- **Trop de flou nuit activement.** `s0_2` fait −3,55 pt, *moins bien que ne pas
  filtrer du tout*. Cela borne l'intuition « plus de flou = curriculum plus doux » :
  au-delà d'un seuil l'opérateur détruit le signal au lieu de l'adoucir.
- **L'effet s'éteint sous 0,5.** `s0_0.25` est indiscernable du témoin (−0,32 pt,
  signe variable). La transition se situe entre 0,25 et 0,5.
- **Trois axes testés, un seul est sensible :**

  | Axe | Résultat |
  |---|---|
  | Profil en profondeur (EXP-012) | plat — uniforme ≥ tout |
  | Adaptation (EXP-012) | plat — −0,30 pt, dans le bruit |
  | **Amplitude (EXP-013)** | **fortement piqué à 1,0** |

  La sensibilité du Gaussian interne est donc concentrée dans l'unique paramètre
  qui n'avait jamais été varié — et il se trouvait déjà au sommet.

## 6. Limites

- **Amplitude et calendrier effectif ne sont pas séparables.** Un `sigma0` plus
  grand passe aussi du temps à des largeurs que les bras plus petits n'atteignent
  jamais. Cette expérience établit « l'incumbent est proche de l'optimum », **pas**
  un mécanisme.
- Le support en pixels diffère entre bras (rayon 1 à 8). La troncature relative est
  appariée, le coût ne l'est pas : `s0_2` est ≈ 1,9× le coût filtre de `rho1`.
- Fiche rétrospective, comme EXP-012 : pas de fiche `planned` préalable
  (MAINTENANCE §4). À corriger pour la prochaine campagne.
- Trois graines et un test déjà exposé : lecture exploratoire située.

## 7. Conséquence pratique

Arrêter de régler le Gaussian interne. Les trois axes accessibles sont explorés et
le réglage courant est au sommet du seul qui compte. La marge restante du projet
est dans la **résolution progressive** (+4,47 pt seule, +5,10 pt combinée dans
EXP-011), pas dans le calendrier de flou.

## 8. Artifacts

- `results/kaggle_outputs/sigma0-sweep-20260910-203015/sigma0_sweep_20260910-203015/`
- Figure : `results/sigma0_sweep.png` (`scripts/plot_sigma0_sweep.py`)
- Job : `scripts/job_sigma0_sweep.py` ; étude : `scripts/study_per_layer_cpu.py`
- Code : `continuation/campaign_ops.py` (`sigma_max` devenu paramètre explicite)
