---
id: DATA-CAMPAIGN-TABLES
schema_version: 1
updated_at: 2026-09-09
status: generated_from_source
---

# EXP-011 — Tous les résultats et contrastes

[Fiche de campagne](../experiments/EXP-011_full_grid.md) · [Source numérique](../sources/campaign_results.json)

Généré par `tools/rebuild_tables.py`. Les codes C01…C21 sont des repères documentaires stables dans cet export, pas de nouveaux IDs de runs. Moyenne et SD échantillonnale sur les graines 0/1/2 ; accuracy en %, différences en points. Tous les résultats sont finaux à epoch 30.

## Correspondance des identifiants

| Repère | Identifiant original | Description |
|---|---|---|
| C01 | `R32__Gnone__input_bilinear__all19` | Témoin : 32 constant, aucun filtre |
| C02 | `R32__Gplateau__input_bilinear__all19` | 32 constant ; Gaussian paliers ; bilinéaire AA entrée ; 19 sites |
| C03 | `R32__Ggeo__input_bilinear__all19` | 32 constant ; Gaussian géométrique ; bilinéaire AA entrée ; 19 sites |
| C04 | `Rprog__Gnone__input_bilinear__all19` | 16→24→32 ; sans filtre ; bilinéaire AA entrée |
| C05 | `Rprog__Gplateau__input_bilinear__all19` | 16→24→32 ; Gaussian paliers ; bilinéaire AA entrée ; 19 sites |
| C06 | `Rprog__Ggeo__input_bilinear__all19` | 16→24→32 ; Gaussian géométrique ; bilinéaire AA entrée ; 19 sites |
| C07 | `Rgentle__Gnone__input_bilinear__all19` | 24→32 ; sans filtre ; bilinéaire AA entrée |
| C08 | `Rgentle__Gplateau__input_bilinear__all19` | 24→32 ; Gaussian paliers ; bilinéaire AA entrée ; 19 sites |
| C09 | `Rgentle__Ggeo__input_bilinear__all19` | 24→32 ; Gaussian géométrique ; bilinéaire AA entrée ; 19 sites |
| C10 | `R32__Gmix__input_bilinear__all19` | 32 constant ; mélange identité/Gaussian ; bilinéaire AA entrée ; 19 sites |
| C11 | `Rprog__Gmix__input_bilinear__all19` | 16→24→32 ; mélange identité/Gaussian ; bilinéaire AA entrée ; 19 sites |
| C12 | `R32__Gplateau__input_bilinear__early7` | 32 constant ; Gaussian paliers ; bilinéaire AA entrée ; 7 sites |
| C13 | `Rprog__Gplateau__input_bilinear__early7` | 16→24→32 ; Gaussian paliers ; bilinéaire AA entrée ; 7 sites |
| C14 | `Rprog__Gnone__input_max__all19` | 16→24→32 ; sans filtre ; max entrée |
| C15 | `Rprog__Gplateau__input_max__all19` | 16→24→32 ; Gaussian paliers ; max entrée ; 19 sites |
| C16 | `Rprog__Gnone__stem_bilinear__all19` | 16→24→32 ; sans filtre ; bilinéaire AA après stem |
| C17 | `Rprog__Gplateau__stem_bilinear__all19` | 16→24→32 ; Gaussian paliers ; bilinéaire AA après stem ; 19 sites |
| C18 | `Rprog__Gnone__stem_max__all19` | 16→24→32 ; sans filtre ; max après stem |
| C19 | `Rprog__Gplateau__stem_max__all19` | 16→24→32 ; Gaussian paliers ; max après stem ; 19 sites |
| C20 | `Rreverse__Gnone__input_bilinear__all19` | 24→16→32 ; sans filtre ; bilinéaire AA entrée |
| C21 | `Rreverse__Gplateau__input_bilinear__all19` | 24→16→32 ; Gaussian paliers ; bilinéaire AA entrée ; 19 sites |

## Classement descriptif complet

| Repère | Accuracy moyenne ± SD (%) | CE test moyenne ± SD | CE sonde train | Temps total moyen (min) | Pic (MiB) |
|---|---:|---:|---:|---:|---:|
| C05 | 80.713 ± 0.647 | 0.56816 ± 0.01321 | 0.33858 | 11.661 | 583.6 |
| C15 | 80.340 ± 0.756 | 0.57199 ± 0.01944 | 0.34040 | 12.295 | 583.6 |
| C18 | 79.967 ± 0.583 | 0.58941 ± 0.01756 | 0.36252 | 8.178 | 511.6 |
| C06 | 79.907 ± 0.923 | 0.59181 ± 0.01819 | 0.32148 | 11.814 | 583.6 |
| C19 | 79.810 ± 0.786 | 0.59589 ± 0.02373 | 0.36595 | 12.454 | 583.6 |
| C14 | 79.797 ± 0.501 | 0.59694 ± 0.01586 | 0.34159 | 8.155 | 511.6 |
| C08 | 79.770 ± 0.724 | 0.58984 ± 0.01417 | 0.35987 | 11.569 | 583.6 |
| C09 | 79.683 ± 0.566 | 0.59702 ± 0.01064 | 0.30446 | 11.726 | 583.6 |
| C11 | 79.510 ± 0.797 | 0.59532 ± 0.02158 | 0.41974 | 13.051 | 614.9 |
| C04 | 79.483 ± 0.544 | 0.59894 ± 0.01562 | 0.34135 | 8.205 | 511.6 |
| C21 | 79.327 ± 0.378 | 0.59602 ± 0.00899 | 0.42591 | 12.617 | 583.6 |
| C17 | 79.297 ± 0.616 | 0.60625 ± 0.01567 | 0.37471 | 12.498 | 583.6 |
| C13 | 79.270 ± 0.570 | 0.61394 ± 0.01815 | 0.35720 | 9.939 | 583.6 |
| C20 | 79.227 ± 0.402 | 0.60694 ± 0.00554 | 0.38147 | 7.780 | 511.6 |
| C16 | 78.907 ± 0.370 | 0.62211 ± 0.01670 | 0.36551 | 8.067 | 511.6 |
| C02 | 78.693 ± 0.679 | 0.62097 ± 0.01747 | 0.37844 | 11.602 | 583.6 |
| C07 | 78.563 ± 0.341 | 0.62651 ± 0.00411 | 0.31830 | 8.103 | 511.6 |
| C03 | 78.283 ± 0.503 | 0.65745 ± 0.01732 | 0.24412 | 11.711 | 583.6 |
| C10 | 78.183 ± 0.361 | 0.63060 ± 0.00724 | 0.47191 | 13.156 | 615.6 |
| C01 | 75.617 ± 0.485 | 0.75524 ± 0.01446 | 0.21011 | 8.145 | 511.6 |
| C12 | 75.237 ± 0.739 | 0.73808 ± 0.02580 | 0.32791 | 9.885 | 583.6 |

## Valeurs par graine

| Repère | Acc seed0 (%) | Acc seed1 (%) | Acc seed2 (%) | CE seed0 | CE seed1 | CE seed2 |
|---|---:|---:|---:|---:|---:|---:|
| C05 | 80.15 | 80.57 | 81.42 | 0.577078 | 0.574411 | 0.552977 |
| C15 | 79.69 | 80.16 | 81.17 | 0.592421 | 0.569843 | 0.553714 |
| C18 | 79.62 | 79.64 | 80.64 | 0.603493 | 0.595003 | 0.569732 |
| C06 | 79.06 | 79.77 | 80.89 | 0.608335 | 0.594784 | 0.572313 |
| C19 | 79.05 | 79.76 | 80.62 | 0.618855 | 0.597344 | 0.571456 |
| C14 | 79.31 | 79.77 | 80.31 | 0.612314 | 0.597872 | 0.580633 |
| C08 | 79.27 | 79.44 | 80.60 | 0.595526 | 0.600281 | 0.573714 |
| C09 | 79.03 | 80.01 | 80.01 | 0.607836 | 0.596662 | 0.586555 |
| C11 | 78.75 | 79.44 | 80.34 | 0.613379 | 0.601157 | 0.571416 |
| C04 | 79.20 | 79.14 | 80.11 | 0.612432 | 0.602565 | 0.581823 |
| C21 | 79.00 | 79.24 | 79.74 | 0.605340 | 0.595308 | 0.587405 |
| C17 | 78.66 | 79.34 | 79.89 | 0.624164 | 0.599474 | 0.595100 |
| C13 | 78.70 | 79.84 | 79.27 | 0.634820 | 0.605115 | 0.601887 |
| C20 | 78.78 | 79.56 | 79.34 | 0.612762 | 0.601734 | 0.606311 |
| C16 | 78.48 | 79.10 | 79.14 | 0.639952 | 0.619500 | 0.606867 |
| C02 | 77.91 | 79.07 | 79.10 | 0.640034 | 0.617144 | 0.605736 |
| C07 | 78.17 | 78.76 | 78.76 | 0.630679 | 0.622467 | 0.626373 |
| C03 | 77.84 | 78.83 | 78.18 | 0.676012 | 0.641712 | 0.654633 |
| C10 | 77.77 | 78.44 | 78.34 | 0.638946 | 0.626085 | 0.626756 |
| C01 | 75.13 | 75.62 | 76.10 | 0.765335 | 0.761713 | 0.738684 |
| C12 | 75.43 | 74.42 | 75.86 | 0.725524 | 0.767756 | 0.720958 |

## Contrastes appariés

Les 24 premiers sont déjà présents dans le JSON source. Les suivants sont des recalculs descriptifs ajoutés pour clarifier les questions de la discussion ; ils ne sont pas rebaptisés analyses préspécifiées.

| Contraste | A − B | Seed0 / seed1 / seed2 (points) | Moyenne ± SD (points) | Δ CE moyen |
|---|---|---:|---:|---:|
| Rprog_vs_R32__Gnone | C04 − C01 | +4.070 / +3.520 / +4.010 | +3.867 ± 0.302 | -0.15630 |
| Rgentle_vs_R32__Gnone | C07 − C01 | +3.040 / +3.140 / +2.660 | +2.947 ± 0.253 | -0.12874 |
| Rprog_vs_R32__Gplateau | C05 − C02 | +2.240 / +1.500 / +2.320 | +2.020 ± 0.452 | -0.05282 |
| Rgentle_vs_R32__Gplateau | C08 − C02 | +1.360 / +0.370 / +1.500 | +1.077 ± 0.616 | -0.03113 |
| Rprog_vs_R32__Ggeo | C06 − C03 | +1.220 / +0.940 / +2.710 | +1.623 ± 0.951 | -0.06564 |
| Rgentle_vs_R32__Ggeo | C09 − C03 | +1.190 / +1.180 / +1.830 | +1.400 ± 0.372 | -0.06043 |
| R32__Gplateau_vs_Gnone | C02 − C01 | +2.780 / +3.450 / +3.000 | +3.077 ± 0.342 | -0.13427 |
| R32__Ggeo_vs_Gnone | C03 − C01 | +2.710 / +3.210 / +2.080 | +2.667 ± 0.566 | -0.09779 |
| Rprog__Gplateau_vs_Gnone | C05 − C04 | +0.950 / +1.430 / +1.310 | +1.230 ± 0.250 | -0.03079 |
| Rprog__Ggeo_vs_Gnone | C06 − C04 | -0.140 / +0.630 / +0.780 | +0.423 ± 0.494 | -0.00713 |
| Rgentle__Gplateau_vs_Gnone | C08 − C07 | +1.100 / +0.680 / +1.840 | +1.207 ± 0.587 | -0.03667 |
| Rgentle__Ggeo_vs_Gnone | C09 − C07 | +0.860 / +1.250 / +1.250 | +1.120 ± 0.225 | -0.02949 |
| R32__Gmix_vs_Gplateau | C10 − C02 | -0.140 / -0.630 / -0.760 | -0.510 ± 0.327 | +0.00962 |
| Rprog__Gmix_vs_Gplateau | C11 − C05 | -1.400 / -1.130 / -1.080 | -1.203 ± 0.172 | +0.02716 |
| R32__early7_vs_all19 | C12 − C02 | -2.480 / -4.650 / -3.240 | -3.457 ± 1.101 | +0.11711 |
| Rprog__early7_vs_all19 | C13 − C05 | -1.450 / -0.730 / -2.150 | -1.443 ± 0.710 | +0.04579 |
| Rprog__input_max_vs_input_bilinear__Gnone | C14 − C04 | +0.110 / +0.630 / +0.200 | +0.313 ± 0.278 | -0.00200 |
| Rprog__stem_bilinear_vs_input_bilinear__Gnone | C16 − C04 | -0.720 / -0.040 / -0.970 | -0.577 ± 0.481 | +0.02317 |
| Rprog__stem_max_vs_input_bilinear__Gnone | C18 − C04 | +0.420 / +0.500 / +0.530 | +0.483 ± 0.057 | -0.00953 |
| Rprog__input_max_vs_input_bilinear__Gplateau | C15 − C05 | -0.460 / -0.410 / -0.250 | -0.373 ± 0.110 | +0.00384 |
| Rprog__stem_bilinear_vs_input_bilinear__Gplateau | C17 − C05 | -1.490 / -1.230 / -1.530 | -1.417 ± 0.163 | +0.03809 |
| Rprog__stem_max_vs_input_bilinear__Gplateau | C19 − C05 | -1.100 / -0.810 / -0.800 | -0.903 ± 0.170 | +0.02773 |
| Rreverse_vs_Rprog__Gnone | C20 − C04 | -0.420 / +0.420 / -0.770 | -0.257 ± 0.612 | +0.00800 |
| Rreverse_vs_Rprog__Gplateau | C21 − C05 | -1.150 / -1.330 / -1.680 | -1.387 ± 0.270 | +0.02786 |
| Combo moins plain [ajout] | C05 − C01 | +5.020 / +4.950 / +5.320 | +5.097 ± 0.197 | -0.18709 |
| Combo moins max stem sans filtre [ajout] | C05 − C18 | +0.530 / +0.930 / +0.780 | +0.747 ± 0.202 | -0.02125 |
| Gmix moins absence de filtre à Rprog [ajout] | C11 − C04 | -0.450 / +0.300 / +0.230 | +0.027 ± 0.414 | -0.00362 |
| Paliers moins géométrique à Rprog [ajout] | C05 − C06 | +1.090 / +0.800 / +0.530 | +0.807 ± 0.280 | -0.02366 |
| Max stem moins bilinéaire stem, Gnone [ajout] | C18 − C16 | +1.140 / +0.540 / +1.500 | +1.060 ± 0.485 | -0.03270 |
| Max stem moins max entrée, Gnone [ajout] | C18 − C14 | +0.310 / -0.130 / +0.330 | +0.170 ± 0.260 | -0.00753 |
| Max stem moins bilinéaire stem, Gplateau [ajout] | C19 − C17 | +0.390 / +0.420 / +0.730 | +0.513 ± 0.188 | -0.01036 |
| Max stem moins max entrée, Gplateau [ajout] | C19 − C15 | -0.640 / -0.400 / -0.550 | -0.530 ± 0.121 | +0.02389 |
| Ajout du Gaussian, input_max [ajout] | C15 − C14 | +0.380 / +0.390 / +0.860 | +0.543 ± 0.274 | -0.02495 |
| Ajout du Gaussian, stem_max [ajout] | C19 − C18 | -0.570 / +0.120 / -0.020 | -0.157 ± 0.365 | +0.00648 |
| Ajout du Gaussian, stem_bilinear [ajout] | C17 − C16 | +0.180 / +0.240 / +0.750 | +0.390 ± 0.313 | -0.01586 |

## Limites de ce fichier

Le tri ne constitue pas une validation indépendante du meilleur réglage. Le temps est celui mesuré par run, évaluations incluses, et non une extrapolation de FLOPs. Le JSON ne contient pas les trajectoires complètes ni le temps individuel de chaque seed : on ne les reconstruit pas à partir des moyennes.
