## 1. Replication of the six paired campaign cells

| cell | this job | campaign | diff (pp) |
|---|---:|---:|---:|
| R32__Gnone__input_bilinear__seed0 | 0.7476 | 0.7513 | -0.37 |
| R32__Gnone__input_bilinear__seed1 | 0.7601 | 0.7562 | +0.39 |
| R32__Gnone__input_bilinear__seed2 | 0.7640 | 0.7610 | +0.30 |
| Rprog__Gnone__input_bilinear__seed0 | 0.7863 | 0.7920 | -0.57 |
| Rprog__Gnone__input_bilinear__seed1 | 0.7891 | 0.7910 | -0.19 |
| Rprog__Gnone__input_bilinear__seed2 | 0.7987 | 0.8010 | -0.23 |
| Rprog__Gplateau__input_bilinear__seed0 | 0.8003 | 0.8015 | -0.12 |
| Rprog__Gplateau__input_bilinear__seed1 | 0.8014 | 0.8057 | -0.43 |
| Rprog__Gplateau__input_bilinear__seed2 | 0.8128 | 0.8142 | -0.14 |

## 5b. Step-size controller replay: largest candidate step with tau >= theta (train-mode BN)

| run | epoch | r | tau by candidate | chosen Delta (theta=0.5) | (theta=0.3) |
|---|---:|---:|---|---:|---:|
| Rctrl50__Gnone__input_bilinear__seed0 | 0 | 16 | +1:1.14 +2:1.08 +4:1.00 +8:1.01 | 8 | 8 |
| Rctrl50__Gnone__input_bilinear__seed0 | 1 | 16 | +1:0.83 +2:0.86 +4:0.93 +8:0.84 | 8 | 8 |
| Rctrl50__Gnone__input_bilinear__seed0 | 2 | 16 | +1:0.44 +2:0.48 +4:0.78 +8:0.60 | 8 | 8 |
| Rctrl50__Gnone__input_bilinear__seed0 | 3 | 16 | +1:0.44 +2:0.41 +4:0.84 +8:0.68 | 8 | 8 |
| Rctrl50__Gnone__input_bilinear__seed0 | 4 | 16 | +1:0.21 +2:0.27 +4:0.56 +8:0.37 | 4 | 8 |
| Rctrl50__Gnone__input_bilinear__seed0 | 5 | 16 | +1:0.43 +2:0.45 +4:0.77 +8:0.57 | 8 | 8 |
| Rctrl50__Gnone__input_bilinear__seed0 | 6 | 16 | +1:0.32 +2:0.34 +4:0.64 +8:0.46 | 4 | 8 |
| Rctrl50__Gnone__input_bilinear__seed0 | 7 | 16 | +1:0.36 +2:0.35 +4:0.74 +8:0.49 | 4 | 8 |
| Rctrl50__Gnone__input_bilinear__seed0 | 8 | 16 | +1:0.32 +2:0.38 +4:0.73 +8:0.59 | 8 | 8 |
| Rctrl50__Gnone__input_bilinear__seed0 | 9 | 16 | +1:0.19 +2:0.19 +4:0.52 +8:0.32 | 4 | 8 |
| Rctrl50__Gnone__input_bilinear__seed0 | 10 | 20 | +1:0.45 +2:0.47 +4:0.84 +8:0.70 | 8 | 8 |
| Rctrl50__Gnone__input_bilinear__seed0 | 11 | 24 | +1:0.51 +2:0.43 +4:0.81 +8:0.63 | 8 | 8 |
| Rctrl50__Gnone__input_bilinear__seed0 | 12 | 28 | +1:0.49 +2:0.42 +4:0.80 | 4 | 4 |
| Rctrl50__Gnone__input_bilinear__seed1 | 0 | 16 | +1:0.97 +2:0.95 +4:0.94 +8:0.89 | 8 | 8 |
| Rctrl50__Gnone__input_bilinear__seed1 | 1 | 16 | +1:0.50 +2:0.68 +4:0.83 +8:0.65 | 8 | 8 |
| Rctrl50__Gnone__input_bilinear__seed1 | 2 | 16 | +1:0.57 +2:0.64 +4:0.85 +8:0.78 | 8 | 8 |
| Rctrl50__Gnone__input_bilinear__seed1 | 3 | 16 | +1:0.30 +2:0.48 +4:0.72 +8:0.55 | 8 | 8 |
| Rctrl50__Gnone__input_bilinear__seed1 | 4 | 16 | +1:0.38 +2:0.67 +4:0.74 +8:0.64 | 8 | 8 |
| Rctrl50__Gnone__input_bilinear__seed1 | 5 | 16 | +1:0.23 +2:0.55 +4:0.76 +8:0.59 | 8 | 8 |
| Rctrl50__Gnone__input_bilinear__seed1 | 6 | 16 | +1:0.19 +2:0.37 +4:0.57 +8:0.39 | 4 | 8 |
| Rctrl50__Gnone__input_bilinear__seed1 | 7 | 16 | +1:0.42 +2:0.61 +4:0.84 +8:0.73 | 8 | 8 |
| Rctrl50__Gnone__input_bilinear__seed1 | 8 | 16 | +1:0.12 +2:0.29 +4:0.57 +8:0.34 | 4 | 8 |
| Rctrl50__Gnone__input_bilinear__seed1 | 9 | 16 | +1:0.11 +2:0.31 +4:0.61 +8:0.40 | 4 | 8 |
| Rctrl50__Gnone__input_bilinear__seed1 | 10 | 20 | +1:0.23 +2:0.60 +4:0.87 +8:0.66 | 8 | 8 |
| Rctrl50__Gnone__input_bilinear__seed1 | 11 | 24 | +1:0.62 +2:0.80 +4:0.91 +8:0.82 | 8 | 8 |
| Rctrl50__Gnone__input_bilinear__seed1 | 12 | 28 | +1:0.46 +2:0.64 +4:0.78 | 4 | 4 |
| Rctrl50__Gnone__input_bilinear__seed2 | 0 | 16 | +1:1.07 +2:1.05 +4:0.97 +8:0.97 | 8 | 8 |
| Rctrl50__Gnone__input_bilinear__seed2 | 1 | 16 | +1:0.84 +2:0.85 +4:0.84 +8:0.73 | 8 | 8 |
| Rctrl50__Gnone__input_bilinear__seed2 | 2 | 16 | +1:0.60 +2:0.67 +4:0.95 +8:0.78 | 8 | 8 |
| Rctrl50__Gnone__input_bilinear__seed2 | 3 | 16 | +1:0.45 +2:0.48 +4:0.72 +8:0.53 | 8 | 8 |
| Rctrl50__Gnone__input_bilinear__seed2 | 4 | 16 | +1:0.42 +2:0.50 +4:0.69 +8:0.53 | 8 | 8 |
| Rctrl50__Gnone__input_bilinear__seed2 | 5 | 16 | +1:0.46 +2:0.55 +4:0.83 +8:0.65 | 8 | 8 |
| Rctrl50__Gnone__input_bilinear__seed2 | 6 | 16 | +1:0.25 +2:0.39 +4:0.72 +8:0.49 | 4 | 8 |
| Rctrl50__Gnone__input_bilinear__seed2 | 7 | 16 | +1:0.13 +2:0.26 +4:0.72 +8:0.49 | 4 | 8 |
| Rctrl50__Gnone__input_bilinear__seed2 | 8 | 16 | +1:0.17 +2:0.27 +4:0.64 +8:0.42 | 4 | 8 |
| Rctrl50__Gnone__input_bilinear__seed2 | 9 | 16 | +1:0.19 +2:0.27 +4:0.60 +8:0.39 | 4 | 8 |
| Rctrl50__Gnone__input_bilinear__seed2 | 10 | 20 | +1:0.30 +2:0.52 +4:0.73 +8:0.54 | 8 | 8 |
| Rctrl50__Gnone__input_bilinear__seed2 | 11 | 24 | +1:0.36 +2:0.54 +4:0.80 +8:0.64 | 8 | 8 |
| Rctrl50__Gnone__input_bilinear__seed2 | 12 | 28 | +1:0.36 +2:0.54 +4:0.71 | 4 | 4 |
| Rctrl65__Gnone__input_bilinear__seed0 | 0 | 16 | +1:1.14 +2:1.08 +4:1.00 +8:1.01 | 8 | 8 |
| Rctrl65__Gnone__input_bilinear__seed0 | 1 | 16 | +1:0.85 +2:0.87 +4:0.94 +8:0.85 | 8 | 8 |
| Rctrl65__Gnone__input_bilinear__seed0 | 2 | 16 | +1:0.38 +2:0.37 +4:0.77 +8:0.56 | 8 | 8 |
| Rctrl65__Gnone__input_bilinear__seed0 | 3 | 16 | +1:0.45 +2:0.44 +4:0.84 +8:0.65 | 8 | 8 |
| Rctrl65__Gnone__input_bilinear__seed0 | 4 | 16 | +1:0.15 +2:0.23 +4:0.58 +8:0.36 | 4 | 8 |
| Rctrl65__Gnone__input_bilinear__seed0 | 5 | 16 | +1:0.43 +2:0.48 +4:0.81 +8:0.61 | 8 | 8 |
| Rctrl65__Gnone__input_bilinear__seed0 | 6 | 16 | +1:0.34 +2:0.34 +4:0.67 +8:0.46 | 4 | 8 |
| Rctrl65__Gnone__input_bilinear__seed0 | 7 | 16 | +1:0.36 +2:0.35 +4:0.75 +8:0.52 | 8 | 8 |
| Rctrl65__Gnone__input_bilinear__seed0 | 8 | 16 | +1:0.14 +2:0.15 +4:0.65 +8:0.34 | 4 | 8 |
| Rctrl65__Gnone__input_bilinear__seed0 | 9 | 16 | +1:0.24 +2:0.21 +4:0.62 +8:0.37 | 4 | 8 |
| Rctrl65__Gnone__input_bilinear__seed0 | 10 | 20 | +1:0.42 +2:0.32 +4:0.81 +8:0.57 | 8 | 8 |
| Rctrl65__Gnone__input_bilinear__seed0 | 11 | 24 | +1:0.58 +2:0.53 +4:0.85 +8:0.77 | 8 | 8 |
| Rctrl65__Gnone__input_bilinear__seed0 | 12 | 28 | +1:0.58 +2:0.55 +4:0.89 | 4 | 4 |
| Rctrl65__Gnone__input_bilinear__seed1 | 0 | 16 | +1:0.97 +2:0.95 +4:0.94 +8:0.89 | 8 | 8 |
| Rctrl65__Gnone__input_bilinear__seed1 | 1 | 16 | +1:0.49 +2:0.69 +4:0.81 +8:0.64 | 8 | 8 |
| Rctrl65__Gnone__input_bilinear__seed1 | 2 | 16 | +1:0.57 +2:0.68 +4:0.87 +8:0.77 | 8 | 8 |
| Rctrl65__Gnone__input_bilinear__seed1 | 3 | 16 | +1:0.41 +2:0.57 +4:0.79 +8:0.62 | 8 | 8 |
| Rctrl65__Gnone__input_bilinear__seed1 | 4 | 16 | +1:0.39 +2:0.65 +4:0.74 +8:0.57 | 8 | 8 |
| Rctrl65__Gnone__input_bilinear__seed1 | 5 | 16 | +1:0.24 +2:0.53 +4:0.79 +8:0.54 | 8 | 8 |
| Rctrl65__Gnone__input_bilinear__seed1 | 6 | 16 | +1:0.22 +2:0.43 +4:0.64 +8:0.45 | 4 | 8 |
| Rctrl65__Gnone__input_bilinear__seed1 | 7 | 16 | +1:0.37 +2:0.58 +4:0.83 +8:0.66 | 8 | 8 |
| Rctrl65__Gnone__input_bilinear__seed1 | 8 | 16 | +1:0.15 +2:0.33 +4:0.57 +8:0.29 | 4 | 4 |
| Rctrl65__Gnone__input_bilinear__seed1 | 9 | 16 | +1:0.12 +2:0.45 +4:0.70 +8:0.40 | 4 | 8 |
| Rctrl65__Gnone__input_bilinear__seed1 | 10 | 20 | +1:0.43 +2:0.74 +4:0.86 +8:0.64 | 8 | 8 |
| Rctrl65__Gnone__input_bilinear__seed1 | 11 | 24 | +1:0.53 +2:0.77 +4:0.87 +8:0.73 | 8 | 8 |
| Rctrl65__Gnone__input_bilinear__seed1 | 12 | 28 | +1:0.58 +2:0.80 +4:0.89 | 4 | 4 |
| Rctrl65__Gnone__input_bilinear__seed2 | 0 | 16 | +1:1.07 +2:1.05 +4:0.97 +8:0.97 | 8 | 8 |
| Rctrl65__Gnone__input_bilinear__seed2 | 1 | 16 | +1:0.90 +2:0.90 +4:0.84 +8:0.70 | 8 | 8 |
| Rctrl65__Gnone__input_bilinear__seed2 | 2 | 16 | +1:0.54 +2:0.61 +4:0.90 +8:0.66 | 8 | 8 |
| Rctrl65__Gnone__input_bilinear__seed2 | 3 | 16 | +1:0.47 +2:0.50 +4:0.70 +8:0.51 | 8 | 8 |
| Rctrl65__Gnone__input_bilinear__seed2 | 4 | 16 | +1:0.44 +2:0.49 +4:0.69 +8:0.54 | 8 | 8 |
| Rctrl65__Gnone__input_bilinear__seed2 | 5 | 16 | +1:0.47 +2:0.53 +4:0.76 +8:0.61 | 8 | 8 |
| Rctrl65__Gnone__input_bilinear__seed2 | 6 | 16 | +1:0.24 +2:0.36 +4:0.69 +8:0.46 | 4 | 8 |
| Rctrl65__Gnone__input_bilinear__seed2 | 7 | 16 | +1:0.18 +2:0.33 +4:0.78 +8:0.53 | 8 | 8 |
| Rctrl65__Gnone__input_bilinear__seed2 | 8 | 16 | +1:0.17 +2:0.28 +4:0.66 +8:0.45 | 4 | 8 |
| Rctrl65__Gnone__input_bilinear__seed2 | 9 | 16 | +1:0.15 +2:0.23 +4:0.63 +8:0.44 | 4 | 8 |
| Rctrl65__Gnone__input_bilinear__seed2 | 10 | 20 | +1:0.23 +2:0.46 +4:0.71 +8:0.52 | 8 | 8 |
| Rctrl65__Gnone__input_bilinear__seed2 | 11 | 24 | +1:0.33 +2:0.53 +4:0.81 +8:0.70 | 8 | 8 |
| Rctrl65__Gnone__input_bilinear__seed2 | 12 | 28 | +1:0.32 +2:0.58 +4:0.80 | 4 | 4 |
| Rlin12__Gnone__input_bilinear__seed1 | 0 | 16 | +1:0.97 +2:0.95 +4:0.94 +8:0.89 | 8 | 8 |
| Rlin12__Gnone__input_bilinear__seed1 | 1 | 16 | +1:0.54 +2:0.72 +4:0.82 +8:0.69 | 8 | 8 |
| Rlin12__Gnone__input_bilinear__seed1 | 2 | 17 | +1:0.98 +2:0.91 +4:0.92 +8:0.86 | 8 | 8 |
| Rlin12__Gnone__input_bilinear__seed1 | 3 | 19 | +1:0.91 +2:0.73 +4:0.92 +8:0.80 | 8 | 8 |
| Rlin12__Gnone__input_bilinear__seed1 | 4 | 20 | +1:0.65 +2:0.73 +4:0.79 +8:0.65 | 8 | 8 |
| Rlin12__Gnone__input_bilinear__seed1 | 5 | 21 | +1:1.04 +2:0.85 +4:0.89 +8:0.75 | 8 | 8 |
| Rlin12__Gnone__input_bilinear__seed1 | 6 | 23 | +1:0.86 +2:0.71 +4:0.75 +8:0.55 | 8 | 8 |
| Rlin12__Gnone__input_bilinear__seed1 | 7 | 24 | +1:0.78 +2:0.84 +4:0.83 +8:0.69 | 8 | 8 |
| Rlin12__Gnone__input_bilinear__seed1 | 8 | 25 | +1:0.88 +2:0.79 +4:0.89 | 4 | 4 |
| Rlin12__Gnone__input_bilinear__seed1 | 9 | 27 | +1:0.90 +2:0.72 +4:0.78 | 4 | 4 |
| Rlin12__Gnone__input_bilinear__seed1 | 10 | 28 | +1:0.83 +2:0.90 +4:0.89 | 4 | 4 |
| Rlin12__Gnone__input_bilinear__seed1 | 11 | 29 | +1:0.82 +2:0.78 | 2 | 2 |
| Rlin12__Gnone__input_bilinear__seed1 | 12 | 31 | +1:0.79 | 1 | 1 |
| Rlin12__Gnone__input_bilinear__seed2 | 0 | 16 | +1:1.07 +2:1.05 +4:0.97 +8:0.97 | 8 | 8 |
| Rlin12__Gnone__input_bilinear__seed2 | 1 | 16 | +1:0.74 +2:0.78 +4:0.83 +8:0.65 | 8 | 8 |
| Rlin12__Gnone__input_bilinear__seed2 | 2 | 17 | +1:0.88 +2:0.66 +4:0.80 +8:0.54 | 8 | 8 |
| Rlin12__Gnone__input_bilinear__seed2 | 3 | 19 | +1:0.86 +2:0.82 +4:0.87 +8:0.68 | 8 | 8 |
| Rlin12__Gnone__input_bilinear__seed2 | 4 | 20 | +1:0.85 +2:0.86 +4:0.91 +8:0.76 | 8 | 8 |
| Rlin12__Gnone__input_bilinear__seed2 | 5 | 21 | +1:0.73 +2:0.82 +4:0.81 +8:0.66 | 8 | 8 |
| Rlin12__Gnone__input_bilinear__seed2 | 6 | 23 | +1:0.82 +2:0.80 +4:0.78 +8:0.59 | 8 | 8 |
| Rlin12__Gnone__input_bilinear__seed2 | 7 | 24 | +1:0.88 +2:0.93 +4:0.88 +8:0.76 | 8 | 8 |
| Rlin12__Gnone__input_bilinear__seed2 | 8 | 25 | +1:0.87 +2:0.86 +4:0.80 | 4 | 4 |
| Rlin12__Gnone__input_bilinear__seed2 | 9 | 27 | +1:0.83 +2:0.77 +4:0.75 | 4 | 4 |
| Rlin12__Gnone__input_bilinear__seed2 | 10 | 28 | +1:0.71 +2:0.84 +4:0.81 | 4 | 4 |
| Rlin12__Gnone__input_bilinear__seed2 | 11 | 29 | +1:0.86 +2:0.84 | 2 | 2 |
| Rlin12__Gnone__input_bilinear__seed2 | 12 | 31 | +1:0.70 | 1 | 1 |
| Rlin12__Gplateau__input_bilinear__seed0 | 0 | 16 | +1:1.07 +2:1.00 +4:0.91 +8:0.87 | 8 | 8 |
| Rlin12__Gplateau__input_bilinear__seed0 | 1 | 16 | +1:0.77 +2:0.81 +4:0.66 +8:0.45 | 4 | 8 |
| Rlin12__Gplateau__input_bilinear__seed0 | 2 | 17 | +1:0.78 +2:0.78 +4:0.69 +8:0.47 | 4 | 8 |
| Rlin12__Gplateau__input_bilinear__seed0 | 3 | 19 | +1:0.85 +2:0.84 +4:0.82 +8:0.52 | 8 | 8 |
| Rlin12__Gplateau__input_bilinear__seed0 | 4 | 20 | +1:0.69 +2:0.79 +4:0.69 +8:0.43 | 4 | 8 |
| Rlin12__Gplateau__input_bilinear__seed0 | 5 | 21 | +1:0.86 +2:0.84 +4:0.71 +8:0.49 | 4 | 8 |
| Rlin12__Gplateau__input_bilinear__seed0 | 6 | 23 | +1:0.89 +2:0.87 +4:0.78 +8:0.52 | 8 | 8 |
| Rlin12__Gplateau__input_bilinear__seed0 | 7 | 24 | +1:0.77 +2:0.75 +4:0.52 +8:0.35 | 4 | 8 |
| Rlin12__Gplateau__input_bilinear__seed0 | 8 | 25 | +1:0.86 +2:0.79 +4:0.74 | 4 | 4 |
| Rlin12__Gplateau__input_bilinear__seed0 | 9 | 27 | +1:0.89 +2:0.78 +4:0.68 | 4 | 4 |
| Rlin12__Gplateau__input_bilinear__seed0 | 10 | 28 | +1:0.83 +2:0.80 +4:0.81 | 4 | 4 |
| Rlin12__Gplateau__input_bilinear__seed0 | 11 | 29 | +1:0.92 +2:0.79 | 2 | 2 |
| Rlin12__Gplateau__input_bilinear__seed0 | 12 | 31 | +1:0.93 | 1 | 1 |
| Rlin12__Gplateau__input_bilinear__seed1 | 0 | 16 | +1:0.94 +2:0.89 +4:0.81 +8:0.72 | 8 | 8 |
| Rlin12__Gplateau__input_bilinear__seed1 | 1 | 16 | +1:0.63 +2:0.67 +4:0.56 +8:0.34 | 4 | 8 |
| Rlin12__Gplateau__input_bilinear__seed1 | 2 | 17 | +1:0.86 +2:0.72 +4:0.73 +8:0.42 | 4 | 8 |
| Rlin12__Gplateau__input_bilinear__seed1 | 3 | 19 | +1:0.79 +2:0.71 +4:0.61 +8:0.41 | 4 | 8 |
| Rlin12__Gplateau__input_bilinear__seed1 | 4 | 20 | +1:0.74 +2:0.74 +4:0.63 +8:0.40 | 4 | 8 |
| Rlin12__Gplateau__input_bilinear__seed1 | 5 | 21 | +1:0.91 +2:0.79 +4:0.60 +8:0.37 | 4 | 8 |
| Rlin12__Gplateau__input_bilinear__seed1 | 6 | 23 | +1:0.92 +2:0.89 +4:0.80 +8:0.62 | 8 | 8 |
| Rlin12__Gplateau__input_bilinear__seed1 | 7 | 24 | +1:0.76 +2:0.79 +4:0.73 +8:0.53 | 8 | 8 |
| Rlin12__Gplateau__input_bilinear__seed1 | 8 | 25 | +1:0.85 +2:0.79 +4:0.67 | 4 | 4 |
| Rlin12__Gplateau__input_bilinear__seed1 | 9 | 27 | +1:0.86 +2:0.82 +4:0.69 | 4 | 4 |
| Rlin12__Gplateau__input_bilinear__seed1 | 10 | 28 | +1:0.85 +2:0.80 +4:0.74 | 4 | 4 |
| Rlin12__Gplateau__input_bilinear__seed1 | 11 | 29 | +1:0.97 +2:0.86 | 2 | 2 |
| Rlin12__Gplateau__input_bilinear__seed1 | 12 | 31 | +1:0.91 | 1 | 1 |
| Rlin12__Gplateau__input_bilinear__seed2 | 0 | 16 | +1:1.00 +2:0.95 +4:0.86 +8:0.79 | 8 | 8 |
| Rlin12__Gplateau__input_bilinear__seed2 | 1 | 16 | +1:0.68 +2:0.72 +4:0.54 +8:0.30 | 4 | 8 |
| Rlin12__Gplateau__input_bilinear__seed2 | 2 | 17 | +1:0.79 +2:0.72 +4:0.65 +8:0.41 | 4 | 8 |
| Rlin12__Gplateau__input_bilinear__seed2 | 3 | 19 | +1:0.91 +2:0.82 +4:0.74 +8:0.56 | 8 | 8 |
| Rlin12__Gplateau__input_bilinear__seed2 | 4 | 20 | +1:0.70 +2:0.74 +4:0.68 +8:0.46 | 4 | 8 |
| Rlin12__Gplateau__input_bilinear__seed2 | 5 | 21 | +1:0.84 +2:0.84 +4:0.69 +8:0.50 | 8 | 8 |
| Rlin12__Gplateau__input_bilinear__seed2 | 6 | 23 | +1:0.77 +2:0.76 +4:0.65 +8:0.39 | 4 | 8 |
| Rlin12__Gplateau__input_bilinear__seed2 | 7 | 24 | +1:0.79 +2:0.85 +4:0.71 +8:0.51 | 8 | 8 |
| Rlin12__Gplateau__input_bilinear__seed2 | 8 | 25 | +1:0.85 +2:0.76 +4:0.65 | 4 | 4 |
| Rlin12__Gplateau__input_bilinear__seed2 | 9 | 27 | +1:0.85 +2:0.77 +4:0.64 | 4 | 4 |
| Rlin12__Gplateau__input_bilinear__seed2 | 10 | 28 | +1:0.76 +2:0.75 +4:0.73 | 4 | 4 |
| Rlin12__Gplateau__input_bilinear__seed2 | 11 | 29 | +1:0.89 +2:0.72 | 2 | 2 |
| Rlin12__Gplateau__input_bilinear__seed2 | 12 | 31 | +1:0.93 | 1 | 1 |
| Rlin12even__Gnone__input_bilinear__seed0 | 0 | 16 | +1:1.14 +2:1.08 +4:1.00 +8:1.01 | 8 | 8 |
| Rlin12even__Gnone__input_bilinear__seed0 | 1 | 16 | +1:0.84 +2:0.87 +4:0.95 +8:0.85 | 8 | 8 |
| Rlin12even__Gnone__input_bilinear__seed0 | 2 | 18 | +1:0.61 +2:0.68 +4:0.82 +8:0.66 | 8 | 8 |
| Rlin12even__Gnone__input_bilinear__seed0 | 3 | 18 | +1:0.73 +2:0.83 +4:0.85 +8:0.73 | 8 | 8 |
| Rlin12even__Gnone__input_bilinear__seed0 | 4 | 20 | +1:0.39 +2:0.80 +4:0.75 +8:0.48 | 4 | 8 |
| Rlin12even__Gnone__input_bilinear__seed0 | 5 | 22 | +1:0.73 +2:0.87 +4:0.87 +8:0.75 | 8 | 8 |
| Rlin12even__Gnone__input_bilinear__seed0 | 6 | 22 | +1:0.65 +2:0.72 +4:0.75 +8:0.58 | 8 | 8 |
| Rlin12even__Gnone__input_bilinear__seed0 | 7 | 24 | +1:0.60 +2:0.83 +4:0.83 +8:0.68 | 8 | 8 |
| Rlin12even__Gnone__input_bilinear__seed0 | 8 | 26 | +1:0.81 +2:0.86 +4:0.88 | 4 | 4 |
| Rlin12even__Gnone__input_bilinear__seed0 | 9 | 26 | +1:0.74 +2:0.91 +4:0.93 | 4 | 4 |
| Rlin12even__Gnone__input_bilinear__seed0 | 10 | 28 | +1:0.59 +2:0.86 +4:0.85 | 4 | 4 |
| Rlin12even__Gnone__input_bilinear__seed0 | 11 | 30 | +1:0.69 +2:0.82 | 2 | 2 |
| Rlin12even__Gnone__input_bilinear__seed0 | 12 | 30 | +1:0.70 +2:0.79 | 2 | 2 |
| Rlin12even__Gnone__input_bilinear__seed1 | 0 | 16 | +1:0.97 +2:0.95 +4:0.94 +8:0.89 | 8 | 8 |
| Rlin12even__Gnone__input_bilinear__seed1 | 1 | 16 | +1:0.51 +2:0.67 +4:0.78 +8:0.64 | 8 | 8 |
| Rlin12even__Gnone__input_bilinear__seed1 | 2 | 18 | +1:0.74 +2:0.91 +4:0.95 +8:0.86 | 8 | 8 |
| Rlin12even__Gnone__input_bilinear__seed1 | 3 | 18 | +1:0.67 +2:0.81 +4:0.81 +8:0.66 | 8 | 8 |
| Rlin12even__Gnone__input_bilinear__seed1 | 4 | 20 | +1:0.41 +2:0.68 +4:0.81 +8:0.65 | 8 | 8 |
| Rlin12even__Gnone__input_bilinear__seed1 | 5 | 22 | +1:0.59 +2:0.84 +4:0.98 +8:0.82 | 8 | 8 |
| Rlin12even__Gnone__input_bilinear__seed1 | 6 | 22 | +1:0.56 +2:0.79 +4:0.76 +8:0.60 | 8 | 8 |
| Rlin12even__Gnone__input_bilinear__seed1 | 7 | 24 | +1:0.55 +2:0.91 +4:0.89 +8:0.76 | 8 | 8 |
| Rlin12even__Gnone__input_bilinear__seed1 | 8 | 26 | +1:0.63 +2:0.80 +4:0.84 | 4 | 4 |
| Rlin12even__Gnone__input_bilinear__seed1 | 9 | 26 | +1:0.55 +2:0.76 +4:0.84 | 4 | 4 |
| Rlin12even__Gnone__input_bilinear__seed1 | 10 | 28 | +1:0.50 +2:0.75 +4:0.79 | 4 | 4 |
| Rlin12even__Gnone__input_bilinear__seed1 | 11 | 30 | +1:0.71 +2:0.96 | 2 | 2 |
| Rlin12even__Gnone__input_bilinear__seed1 | 12 | 30 | +1:0.51 +2:0.69 | 2 | 2 |
| Rlin12even__Gnone__input_bilinear__seed2 | 0 | 16 | +1:1.07 +2:1.05 +4:0.97 +8:0.97 | 8 | 8 |
| Rlin12even__Gnone__input_bilinear__seed2 | 1 | 16 | +1:0.76 +2:0.84 +4:0.88 +8:0.72 | 8 | 8 |
| Rlin12even__Gnone__input_bilinear__seed2 | 2 | 18 | +1:0.54 +2:0.77 +4:0.90 +8:0.59 | 8 | 8 |
| Rlin12even__Gnone__input_bilinear__seed2 | 3 | 18 | +1:0.63 +2:0.69 +4:0.79 +8:0.62 | 8 | 8 |
| Rlin12even__Gnone__input_bilinear__seed2 | 4 | 20 | +1:0.52 +2:0.86 +4:0.80 +8:0.61 | 8 | 8 |
| Rlin12even__Gnone__input_bilinear__seed2 | 5 | 22 | +1:0.74 +2:0.81 +4:0.85 +8:0.65 | 8 | 8 |
| Rlin12even__Gnone__input_bilinear__seed2 | 6 | 22 | +1:0.72 +2:0.76 +4:0.78 +8:0.56 | 8 | 8 |
| Rlin12even__Gnone__input_bilinear__seed2 | 7 | 24 | +1:0.63 +2:0.79 +4:0.83 +8:0.64 | 8 | 8 |
| Rlin12even__Gnone__input_bilinear__seed2 | 8 | 26 | +1:0.82 +2:0.85 +4:0.83 | 4 | 4 |
| Rlin12even__Gnone__input_bilinear__seed2 | 9 | 26 | +1:0.62 +2:0.70 +4:0.74 | 4 | 4 |
| Rlin12even__Gnone__input_bilinear__seed2 | 10 | 28 | +1:0.54 +2:0.73 +4:0.79 | 4 | 4 |
| Rlin12even__Gnone__input_bilinear__seed2 | 11 | 30 | +1:0.73 +2:0.74 | 2 | 2 |
| Rlin12even__Gnone__input_bilinear__seed2 | 12 | 30 | +1:0.67 +2:0.68 | 2 | 2 |
| Rprog__Gplateau__input_bilinear__seed0 | 0 | 16 | +1:1.07 +2:1.00 +4:0.91 +8:0.87 | 8 | 8 |
| Rprog__Gplateau__input_bilinear__seed0 | 1 | 16 | +1:0.78 +2:0.84 +4:0.69 +8:0.50 | 4 | 8 |
| Rprog__Gplateau__input_bilinear__seed0 | 2 | 16 | +1:0.51 +2:0.67 +4:0.61 +8:0.30 | 4 | 8 |
| Rprog__Gplateau__input_bilinear__seed0 | 3 | 16 | +1:0.48 +2:0.71 +4:0.57 +8:0.27 | 4 | 4 |
| Rprog__Gplateau__input_bilinear__seed0 | 4 | 16 | +1:0.45 +2:0.55 +4:0.39 +8:0.21 | 2 | 4 |
| Rprog__Gplateau__input_bilinear__seed0 | 5 | 16 | +1:0.41 +2:0.57 +4:0.42 +8:0.16 | 2 | 4 |
| Rprog__Gplateau__input_bilinear__seed0 | 6 | 16 | +1:0.31 +2:0.49 +4:0.38 +8:0.17 | 0 | 4 |
| Rprog__Gplateau__input_bilinear__seed0 | 7 | 24 | +1:0.66 +2:0.73 +4:0.66 +8:0.49 | 4 | 8 |
| Rprog__Gplateau__input_bilinear__seed0 | 8 | 24 | +1:0.52 +2:0.68 +4:0.61 +8:0.37 | 4 | 8 |
| Rprog__Gplateau__input_bilinear__seed0 | 9 | 24 | +1:0.63 +2:0.76 +4:0.68 +8:0.44 | 4 | 8 |
| Rprog__Gplateau__input_bilinear__seed0 | 10 | 24 | +1:0.59 +2:0.78 +4:0.64 +8:0.35 | 4 | 8 |
| Rprog__Gplateau__input_bilinear__seed0 | 11 | 24 | +1:0.56 +2:0.66 +4:0.59 +8:0.37 | 4 | 8 |
| Rprog__Gplateau__input_bilinear__seed0 | 12 | 24 | +1:0.50 +2:0.59 +4:0.57 +8:0.32 | 4 | 8 |
| Rprog__Gplateau__input_bilinear__seed1 | 0 | 16 | +1:0.94 +2:0.89 +4:0.81 +8:0.72 | 8 | 8 |
| Rprog__Gplateau__input_bilinear__seed1 | 1 | 16 | +1:0.66 +2:0.74 +4:0.66 +8:0.38 | 4 | 8 |
| Rprog__Gplateau__input_bilinear__seed1 | 2 | 16 | +1:0.56 +2:0.66 +4:0.62 +8:0.39 | 4 | 8 |
| Rprog__Gplateau__input_bilinear__seed1 | 3 | 16 | +1:0.53 +2:0.58 +4:0.51 +8:0.31 | 4 | 8 |
| Rprog__Gplateau__input_bilinear__seed1 | 4 | 16 | +1:0.53 +2:0.59 +4:0.48 +8:0.27 | 2 | 4 |
| Rprog__Gplateau__input_bilinear__seed1 | 5 | 16 | +1:0.38 +2:0.52 +4:0.44 +8:0.24 | 2 | 4 |
| Rprog__Gplateau__input_bilinear__seed1 | 6 | 16 | +1:0.49 +2:0.54 +4:0.44 +8:0.25 | 2 | 4 |
| Rprog__Gplateau__input_bilinear__seed1 | 7 | 24 | +1:0.64 +2:0.68 +4:0.65 +8:0.44 | 4 | 8 |
| Rprog__Gplateau__input_bilinear__seed1 | 8 | 24 | +1:0.61 +2:0.69 +4:0.65 +8:0.42 | 4 | 8 |
| Rprog__Gplateau__input_bilinear__seed1 | 9 | 24 | +1:0.59 +2:0.64 +4:0.62 +8:0.40 | 4 | 8 |
| Rprog__Gplateau__input_bilinear__seed1 | 10 | 24 | +1:0.69 +2:0.72 +4:0.58 +8:0.38 | 4 | 8 |
| Rprog__Gplateau__input_bilinear__seed1 | 11 | 24 | +1:0.62 +2:0.67 +4:0.62 +8:0.40 | 4 | 8 |
| Rprog__Gplateau__input_bilinear__seed1 | 12 | 24 | +1:0.60 +2:0.64 +4:0.59 +8:0.37 | 4 | 8 |
| Rprog__Gplateau__input_bilinear__seed2 | 0 | 16 | +1:1.00 +2:0.95 +4:0.86 +8:0.79 | 8 | 8 |
| Rprog__Gplateau__input_bilinear__seed2 | 1 | 16 | +1:0.71 +2:0.75 +4:0.59 +8:0.32 | 4 | 8 |
| Rprog__Gplateau__input_bilinear__seed2 | 2 | 16 | +1:0.48 +2:0.67 +4:0.62 +8:0.32 | 4 | 8 |
| Rprog__Gplateau__input_bilinear__seed2 | 3 | 16 | +1:0.63 +2:0.77 +4:0.62 +8:0.43 | 4 | 8 |
| Rprog__Gplateau__input_bilinear__seed2 | 4 | 16 | +1:0.47 +2:0.65 +4:0.58 +8:0.34 | 4 | 8 |
| Rprog__Gplateau__input_bilinear__seed2 | 5 | 16 | +1:0.44 +2:0.59 +4:0.55 +8:0.28 | 4 | 4 |
| Rprog__Gplateau__input_bilinear__seed2 | 6 | 16 | +1:0.29 +2:0.60 +4:0.46 +8:0.24 | 2 | 4 |
| Rprog__Gplateau__input_bilinear__seed2 | 7 | 24 | +1:0.66 +2:0.81 +4:0.74 +8:0.50 | 8 | 8 |
| Rprog__Gplateau__input_bilinear__seed2 | 8 | 24 | +1:0.70 +2:0.72 +4:0.55 +8:0.39 | 4 | 8 |
| Rprog__Gplateau__input_bilinear__seed2 | 9 | 24 | +1:0.52 +2:0.74 +4:0.63 +8:0.37 | 4 | 8 |
| Rprog__Gplateau__input_bilinear__seed2 | 10 | 24 | +1:0.47 +2:0.65 +4:0.56 +8:0.35 | 4 | 8 |
| Rprog__Gplateau__input_bilinear__seed2 | 11 | 24 | +1:0.52 +2:0.62 +4:0.52 +8:0.21 | 4 | 4 |
| Rprog__Gplateau__input_bilinear__seed2 | 12 | 24 | +1:0.42 +2:0.62 +4:0.55 +8:0.31 | 4 | 8 |
| Rsteps4__Gnone__input_bilinear__seed1 | 0 | 16 | +1:0.97 +2:0.95 +4:0.94 +8:0.89 | 8 | 8 |
| Rsteps4__Gnone__input_bilinear__seed1 | 1 | 16 | +1:0.47 +2:0.65 +4:0.79 +8:0.63 | 8 | 8 |
| Rsteps4__Gnone__input_bilinear__seed1 | 2 | 16 | +1:0.55 +2:0.67 +4:0.86 +8:0.76 | 8 | 8 |
| Rsteps4__Gnone__input_bilinear__seed1 | 3 | 16 | +1:0.36 +2:0.57 +4:0.75 +8:0.57 | 8 | 8 |
| Rsteps4__Gnone__input_bilinear__seed1 | 4 | 20 | +1:0.43 +2:0.71 +4:0.81 +8:0.67 | 8 | 8 |
| Rsteps4__Gnone__input_bilinear__seed1 | 5 | 20 | +1:0.46 +2:0.76 +4:0.91 +8:0.78 | 8 | 8 |
| Rsteps4__Gnone__input_bilinear__seed1 | 6 | 20 | +1:0.27 +2:0.52 +4:0.74 +8:0.54 | 8 | 8 |
| Rsteps4__Gnone__input_bilinear__seed1 | 7 | 24 | +1:0.35 +2:0.63 +4:0.91 +8:0.78 | 8 | 8 |
| Rsteps4__Gnone__input_bilinear__seed1 | 8 | 24 | +1:0.33 +2:0.61 +4:0.86 +8:0.67 | 8 | 8 |
| Rsteps4__Gnone__input_bilinear__seed1 | 9 | 24 | +1:0.38 +2:0.60 +4:0.75 +8:0.58 | 8 | 8 |
| Rsteps4__Gnone__input_bilinear__seed1 | 10 | 28 | +1:0.53 +2:0.76 +4:0.90 | 4 | 4 |
| Rsteps4__Gnone__input_bilinear__seed1 | 11 | 28 | +1:0.55 +2:0.76 +4:0.86 | 4 | 4 |
| Rsteps4__Gnone__input_bilinear__seed1 | 12 | 28 | +1:0.40 +2:0.59 +4:0.72 | 4 | 4 |
| Rsteps4__Gnone__input_bilinear__seed2 | 0 | 16 | +1:1.07 +2:1.05 +4:0.97 +8:0.97 | 8 | 8 |
| Rsteps4__Gnone__input_bilinear__seed2 | 1 | 16 | +1:0.75 +2:0.80 +4:0.83 +8:0.69 | 8 | 8 |
| Rsteps4__Gnone__input_bilinear__seed2 | 2 | 16 | +1:0.54 +2:0.64 +4:0.92 +8:0.72 | 8 | 8 |
| Rsteps4__Gnone__input_bilinear__seed2 | 3 | 16 | +1:0.41 +2:0.45 +4:0.67 +8:0.48 | 4 | 8 |
| Rsteps4__Gnone__input_bilinear__seed2 | 4 | 20 | +1:0.45 +2:0.58 +4:0.79 +8:0.57 | 8 | 8 |
| Rsteps4__Gnone__input_bilinear__seed2 | 5 | 20 | +1:0.48 +2:0.63 +4:0.83 +8:0.64 | 8 | 8 |
| Rsteps4__Gnone__input_bilinear__seed2 | 6 | 20 | +1:0.40 +2:0.55 +4:0.77 +8:0.56 | 8 | 8 |
| Rsteps4__Gnone__input_bilinear__seed2 | 7 | 24 | +1:0.52 +2:0.67 +4:0.82 +8:0.70 | 8 | 8 |
| Rsteps4__Gnone__input_bilinear__seed2 | 8 | 24 | +1:0.53 +2:0.71 +4:0.79 +8:0.71 | 8 | 8 |
| Rsteps4__Gnone__input_bilinear__seed2 | 9 | 24 | +1:0.37 +2:0.52 +4:0.76 +8:0.56 | 8 | 8 |
| Rsteps4__Gnone__input_bilinear__seed2 | 10 | 28 | +1:0.56 +2:0.73 +4:0.83 | 4 | 4 |
| Rsteps4__Gnone__input_bilinear__seed2 | 11 | 28 | +1:0.49 +2:0.58 +4:0.82 | 4 | 4 |
| Rsteps4__Gnone__input_bilinear__seed2 | 12 | 28 | +1:0.45 +2:0.63 +4:0.65 | 4 | 4 |
| Rsteps4__Gplateau__input_bilinear__seed0 | 0 | 16 | +1:1.07 +2:1.00 +4:0.91 +8:0.87 | 8 | 8 |
| Rsteps4__Gplateau__input_bilinear__seed0 | 1 | 16 | +1:0.76 +2:0.80 +4:0.65 +8:0.43 | 4 | 8 |
| Rsteps4__Gplateau__input_bilinear__seed0 | 2 | 16 | +1:0.55 +2:0.69 +4:0.60 +8:0.32 | 4 | 8 |
| Rsteps4__Gplateau__input_bilinear__seed0 | 3 | 16 | +1:0.48 +2:0.74 +4:0.57 +8:0.19 | 4 | 4 |
| Rsteps4__Gplateau__input_bilinear__seed0 | 4 | 20 | +1:0.64 +2:0.74 +4:0.67 +8:0.44 | 4 | 8 |
| Rsteps4__Gplateau__input_bilinear__seed0 | 5 | 20 | +1:0.62 +2:0.69 +4:0.68 +8:0.42 | 4 | 8 |
| Rsteps4__Gplateau__input_bilinear__seed0 | 6 | 20 | +1:0.58 +2:0.68 +4:0.59 +8:0.35 | 4 | 8 |
| Rsteps4__Gplateau__input_bilinear__seed0 | 7 | 24 | +1:0.65 +2:0.74 +4:0.65 +8:0.42 | 4 | 8 |
| Rsteps4__Gplateau__input_bilinear__seed0 | 8 | 24 | +1:0.57 +2:0.71 +4:0.63 +8:0.38 | 4 | 8 |
| Rsteps4__Gplateau__input_bilinear__seed0 | 9 | 24 | +1:0.61 +2:0.75 +4:0.66 +8:0.40 | 4 | 8 |
| Rsteps4__Gplateau__input_bilinear__seed0 | 10 | 28 | +1:0.61 +2:0.78 +4:0.77 | 4 | 4 |
| Rsteps4__Gplateau__input_bilinear__seed0 | 11 | 28 | +1:0.62 +2:0.72 +4:0.74 | 4 | 4 |
| Rsteps4__Gplateau__input_bilinear__seed0 | 12 | 28 | +1:0.55 +2:0.62 +4:0.72 | 4 | 4 |
| Rsteps4__Gplateau__input_bilinear__seed1 | 0 | 16 | +1:0.94 +2:0.89 +4:0.81 +8:0.72 | 8 | 8 |
| Rsteps4__Gplateau__input_bilinear__seed1 | 1 | 16 | +1:0.58 +2:0.67 +4:0.58 +8:0.33 | 4 | 8 |
| Rsteps4__Gplateau__input_bilinear__seed1 | 2 | 16 | +1:0.50 +2:0.63 +4:0.61 +8:0.38 | 4 | 8 |
| Rsteps4__Gplateau__input_bilinear__seed1 | 3 | 16 | +1:0.50 +2:0.62 +4:0.55 +8:0.35 | 4 | 8 |
| Rsteps4__Gplateau__input_bilinear__seed1 | 4 | 20 | +1:0.61 +2:0.69 +4:0.56 +8:0.35 | 4 | 8 |
| Rsteps4__Gplateau__input_bilinear__seed1 | 5 | 20 | +1:0.58 +2:0.64 +4:0.58 +8:0.37 | 4 | 8 |
| Rsteps4__Gplateau__input_bilinear__seed1 | 6 | 20 | +1:0.59 +2:0.69 +4:0.58 +8:0.33 | 4 | 8 |
| Rsteps4__Gplateau__input_bilinear__seed1 | 7 | 24 | +1:0.70 +2:0.76 +4:0.71 +8:0.50 | 8 | 8 |
| Rsteps4__Gplateau__input_bilinear__seed1 | 8 | 24 | +1:0.63 +2:0.67 +4:0.67 +8:0.46 | 4 | 8 |
| Rsteps4__Gplateau__input_bilinear__seed1 | 9 | 24 | +1:0.60 +2:0.70 +4:0.61 +8:0.43 | 4 | 8 |
| Rsteps4__Gplateau__input_bilinear__seed1 | 10 | 28 | +1:0.77 +2:0.81 +4:0.83 | 4 | 4 |
| Rsteps4__Gplateau__input_bilinear__seed1 | 11 | 28 | +1:0.72 +2:0.77 +4:0.77 | 4 | 4 |
| Rsteps4__Gplateau__input_bilinear__seed1 | 12 | 28 | +1:0.69 +2:0.74 +4:0.74 | 4 | 4 |
| Rsteps4__Gplateau__input_bilinear__seed2 | 0 | 16 | +1:1.00 +2:0.95 +4:0.86 +8:0.79 | 8 | 8 |
| Rsteps4__Gplateau__input_bilinear__seed2 | 1 | 16 | +1:0.72 +2:0.72 +4:0.52 +8:0.36 | 4 | 8 |
| Rsteps4__Gplateau__input_bilinear__seed2 | 2 | 16 | +1:0.49 +2:0.65 +4:0.59 +8:0.33 | 4 | 8 |
| Rsteps4__Gplateau__input_bilinear__seed2 | 3 | 16 | +1:0.50 +2:0.68 +4:0.51 +8:0.33 | 4 | 8 |
| Rsteps4__Gplateau__input_bilinear__seed2 | 4 | 20 | +1:0.52 +2:0.73 +4:0.66 +8:0.43 | 4 | 8 |
| Rsteps4__Gplateau__input_bilinear__seed2 | 5 | 20 | +1:0.49 +2:0.63 +4:0.60 +8:0.38 | 4 | 8 |
| Rsteps4__Gplateau__input_bilinear__seed2 | 6 | 20 | +1:0.35 +2:0.62 +4:0.57 +8:0.32 | 4 | 8 |
| Rsteps4__Gplateau__input_bilinear__seed2 | 7 | 24 | +1:0.67 +2:0.85 +4:0.74 +8:0.54 | 8 | 8 |
| Rsteps4__Gplateau__input_bilinear__seed2 | 8 | 24 | +1:0.66 +2:0.69 +4:0.54 +8:0.36 | 4 | 8 |
| Rsteps4__Gplateau__input_bilinear__seed2 | 9 | 24 | +1:0.55 +2:0.72 +4:0.66 +8:0.44 | 4 | 8 |
| Rsteps4__Gplateau__input_bilinear__seed2 | 10 | 28 | +1:0.57 +2:0.70 +4:0.69 | 4 | 4 |
| Rsteps4__Gplateau__input_bilinear__seed2 | 11 | 28 | +1:0.66 +2:0.76 +4:0.75 | 4 | 4 |
| Rsteps4__Gplateau__input_bilinear__seed2 | 12 | 28 | +1:0.52 +2:0.71 +4:0.69 | 4 | 4 |

## 6. Final test accuracy by configuration (all seeds present)

| configuration | seeds | acc per seed | mean | test CE mean | train s |
|---|---|---|---:|---:|---:|
| Rprog__Gplateau__input_bilinear | 0/1/2 | 0.8003 / 0.8014 / 0.8128 | 0.8048 | 0.5723 | 667 |
| Rsteps4__Gplateau__input_bilinear | 0/1/2 | 0.7964 / 0.8031 / 0.8132 | 0.8042 | 0.5768 | 670 |
| Rlin12__Gplateau__input_bilinear | 0/1/2 | 0.7973 / 0.8011 / 0.8071 | 0.8018 | 0.5760 | 676 |
| Rlin12__Gnone__input_bilinear | 0/1/2 | 0.7970 / 0.7997 / 0.8074 | 0.8014 | 0.5859 | 457 |
| Rlin12even__Gnone__input_bilinear | 0/1/2 | 0.8007 / 0.7984 / 0.8045 | 0.8012 | 0.5852 | 441 |
| Rsteps4__Gnone__input_bilinear | 0/1/2 | 0.7980 / 0.7954 / 0.8071 | 0.8002 | 0.5870 | 463 |
| Rmixed__Gnone__input_bilinear | 0/1/2 | 0.7916 / 0.7966 / 0.8009 | 0.7964 | 0.5887 | 462 |
| Rctrl50__Gnone__input_bilinear | 0/1/2 | 0.7861 / 0.7888 / 0.8029 | 0.7926 | 0.6039 | 498 |
| Rprog__Gnone__input_bilinear | 0/1/2 | 0.7863 / 0.7891 / 0.7987 | 0.7914 | 0.6082 | 451 |
| Rctrl65__Gnone__input_bilinear | 0/1/2 | 0.7857 / 0.7921 / 0.7956 | 0.7911 | 0.6080 | 491 |
| Rb9_15__Gnone__input_bilinear | 0 | 0.7905 | 0.7905 | 0.6105 | 447 |
| Rb6_9__Gnone__input_bilinear | 0 | 0.7896 | 0.7896 | 0.6358 | 458 |
| Rb6_18__Gnone__input_bilinear | 0 | 0.7896 | 0.7896 | 0.6163 | 449 |
| Rb3_12__Gnone__input_bilinear | 0 | 0.7884 | 0.7884 | 0.6237 | 452 |
| Rb9_18__Gnone__input_bilinear | 0 | 0.7860 | 0.7860 | 0.6229 | 444 |
| Rb3_9__Gnone__input_bilinear | 0 | 0.7820 | 0.7820 | 0.6496 | 454 |
| Rb12_18__Gnone__input_bilinear | 0 | 0.7746 | 0.7746 | 0.6390 | 437 |
| R32__Gnone__input_bilinear | 0/1/2 | 0.7476 / 0.7601 / 0.7640 | 0.7572 | 0.7544 | 464 |

## 2. Go / no-go rule of plan section 11.4 (D = relative decrease of EMA ||g|| over the last two epochs of a stage)

| run | stage | D | verdict |
|---|---|---:|---|
| Rb12_18__Gnone__input_bilinear__seed0 | stage0_r16 | -0.078 | GO |
| Rb12_18__Gnone__input_bilinear__seed0 | stage1_r24 | 0.038 | GO |
| Rb3_12__Gnone__input_bilinear__seed0 | stage0_r16 | 0.608 | NO-GO |
| Rb3_12__Gnone__input_bilinear__seed0 | stage1_r24 | 0.037 | GO |
| Rb3_9__Gnone__input_bilinear__seed0 | stage0_r16 | 0.549 | NO-GO |
| Rb3_9__Gnone__input_bilinear__seed0 | stage1_r24 | 0.126 | NO-GO |
| Rb6_18__Gnone__input_bilinear__seed0 | stage0_r16 | 0.396 | NO-GO |
| Rb6_18__Gnone__input_bilinear__seed0 | stage1_r24 | 0.202 | NO-GO |
| Rb6_9__Gnone__input_bilinear__seed0 | stage0_r16 | 0.385 | NO-GO |
| Rb6_9__Gnone__input_bilinear__seed0 | stage1_r24 | 0.054 | marginal |
| Rb9_15__Gnone__input_bilinear__seed0 | stage0_r16 | 0.295 | NO-GO |
| Rb9_15__Gnone__input_bilinear__seed0 | stage1_r24 | -0.336 | GO |
| Rb9_18__Gnone__input_bilinear__seed0 | stage0_r16 | 0.231 | NO-GO |
| Rb9_18__Gnone__input_bilinear__seed0 | stage1_r24 | 0.097 | marginal |
| Rctrl50__Gnone__input_bilinear__seed0 | stage0_r16 | 0.051 | marginal |
| Rctrl50__Gnone__input_bilinear__seed1 | stage0_r16 | 0.148 | NO-GO |
| Rctrl50__Gnone__input_bilinear__seed2 | stage0_r16 | 0.041 | GO |
| Rctrl65__Gnone__input_bilinear__seed0 | stage0_r16 | 0.171 | NO-GO |
| Rctrl65__Gnone__input_bilinear__seed1 | stage0_r16 | 0.172 | NO-GO |
| Rctrl65__Gnone__input_bilinear__seed2 | stage0_r16 | -0.003 | GO |
| Rprog__Gnone__input_bilinear__seed0 | stage0_r16 | 0.407 | NO-GO |
| Rprog__Gnone__input_bilinear__seed0 | stage1_r24 | -0.021 | GO |
| Rprog__Gnone__input_bilinear__seed1 | stage0_r16 | 0.052 | marginal |
| Rprog__Gnone__input_bilinear__seed1 | stage1_r24 | -0.365 | GO |
| Rprog__Gnone__input_bilinear__seed2 | stage0_r16 | 0.043 | GO |
| Rprog__Gnone__input_bilinear__seed2 | stage1_r24 | 0.301 | NO-GO |
| Rprog__Gplateau__input_bilinear__seed0 | stage0_r16 | 0.414 | NO-GO |
| Rprog__Gplateau__input_bilinear__seed0 | stage1_r24 | -0.074 | GO |
| Rprog__Gplateau__input_bilinear__seed1 | stage0_r16 | 0.075 | marginal |
| Rprog__Gplateau__input_bilinear__seed1 | stage1_r24 | -0.003 | GO |
| Rprog__Gplateau__input_bilinear__seed2 | stage0_r16 | 0.278 | NO-GO |
| Rprog__Gplateau__input_bilinear__seed2 | stage1_r24 | 0.406 | NO-GO |
| Rsteps4__Gnone__input_bilinear__seed0 | stage0_r16 | 0.558 | NO-GO |
| Rsteps4__Gnone__input_bilinear__seed0 | stage1_r20 | 0.358 | NO-GO |
| Rsteps4__Gnone__input_bilinear__seed0 | stage2_r24 | 0.024 | GO |
| Rsteps4__Gnone__input_bilinear__seed0 | stage3_r28 | 0.046 | GO |
| Rsteps4__Gnone__input_bilinear__seed1 | stage0_r16 | 0.331 | NO-GO |
| Rsteps4__Gnone__input_bilinear__seed1 | stage1_r20 | 0.035 | GO |
| Rsteps4__Gnone__input_bilinear__seed1 | stage2_r24 | 0.112 | NO-GO |
| Rsteps4__Gnone__input_bilinear__seed1 | stage3_r28 | -0.373 | GO |
| Rsteps4__Gnone__input_bilinear__seed2 | stage0_r16 | 0.634 | NO-GO |
| Rsteps4__Gnone__input_bilinear__seed2 | stage1_r20 | 0.042 | GO |
| Rsteps4__Gnone__input_bilinear__seed2 | stage2_r24 | 0.013 | GO |
| Rsteps4__Gnone__input_bilinear__seed2 | stage3_r28 | 0.023 | GO |
| Rsteps4__Gplateau__input_bilinear__seed0 | stage0_r16 | 0.431 | NO-GO |
| Rsteps4__Gplateau__input_bilinear__seed0 | stage1_r20 | 0.290 | NO-GO |
| Rsteps4__Gplateau__input_bilinear__seed0 | stage2_r24 | 0.100 | marginal |
| Rsteps4__Gplateau__input_bilinear__seed0 | stage3_r28 | -0.068 | GO |
| Rsteps4__Gplateau__input_bilinear__seed1 | stage0_r16 | 0.181 | NO-GO |
| Rsteps4__Gplateau__input_bilinear__seed1 | stage1_r20 | 0.233 | NO-GO |
| Rsteps4__Gplateau__input_bilinear__seed1 | stage2_r24 | -0.354 | GO |
| Rsteps4__Gplateau__input_bilinear__seed1 | stage3_r28 | 0.141 | NO-GO |
| Rsteps4__Gplateau__input_bilinear__seed2 | stage0_r16 | 0.577 | NO-GO |
| Rsteps4__Gplateau__input_bilinear__seed2 | stage1_r20 | 0.303 | NO-GO |
| Rsteps4__Gplateau__input_bilinear__seed2 | stage2_r24 | 0.099 | marginal |
| Rsteps4__Gplateau__input_bilinear__seed2 | stage3_r28 | 0.156 | NO-GO |

## 3. Post-switch spikes of ||g|| (ratio to the last pre-switch check; length = checks above it)

| run | event | amplitude | length |
|---|---|---:|---:|
| Rb12_18__Gnone__input_bilinear__seed0 | enter_r24_at_epoch13 | 1.20 | 4 |
| Rb12_18__Gnone__input_bilinear__seed0 | enter_r32_at_epoch19 | 0.74 | 0 |
| Rb3_12__Gnone__input_bilinear__seed0 | enter_r24_at_epoch4 | 1.88 | 3 |
| Rb3_12__Gnone__input_bilinear__seed0 | enter_r32_at_epoch13 | 0.67 | 0 |
| Rb3_9__Gnone__input_bilinear__seed0 | enter_r24_at_epoch4 | 1.44 | 1 |
| Rb3_9__Gnone__input_bilinear__seed0 | enter_r32_at_epoch10 | 1.67 | 4 |
| Rb6_18__Gnone__input_bilinear__seed0 | enter_r24_at_epoch7 | 1.84 | 4 |
| Rb6_18__Gnone__input_bilinear__seed0 | enter_r32_at_epoch19 | 0.84 | 0 |
| Rb6_9__Gnone__input_bilinear__seed0 | enter_r24_at_epoch7 | 2.69 | 3 |
| Rb6_9__Gnone__input_bilinear__seed0 | enter_r32_at_epoch10 | 1.11 | 2 |
| Rb9_15__Gnone__input_bilinear__seed0 | enter_r24_at_epoch10 | 1.51 | 4 |
| Rb9_15__Gnone__input_bilinear__seed0 | enter_r32_at_epoch16 | 0.61 | 0 |
| Rb9_18__Gnone__input_bilinear__seed0 | enter_r24_at_epoch10 | 2.16 | 4 |
| Rb9_18__Gnone__input_bilinear__seed0 | enter_r32_at_epoch19 | 0.90 | 0 |
| Rctrl50__Gnone__input_bilinear__seed0 | enter_r20_at_epoch10 | 1.88 | 1 |
| Rctrl50__Gnone__input_bilinear__seed0 | enter_r24_at_epoch11 | 1.07 | 1 |
| Rctrl50__Gnone__input_bilinear__seed0 | enter_r28_at_epoch12 | 0.75 | 0 |
| Rctrl50__Gnone__input_bilinear__seed0 | enter_r32_at_epoch13 | 0.84 | 0 |
| Rctrl50__Gnone__input_bilinear__seed1 | enter_r20_at_epoch10 | 2.17 | 1 |
| Rctrl50__Gnone__input_bilinear__seed1 | enter_r24_at_epoch11 | 2.82 | 1 |
| Rctrl50__Gnone__input_bilinear__seed1 | enter_r28_at_epoch12 | 0.77 | 0 |
| Rctrl50__Gnone__input_bilinear__seed1 | enter_r32_at_epoch13 | 0.48 | 0 |
| Rctrl50__Gnone__input_bilinear__seed2 | enter_r20_at_epoch10 | 0.49 | 0 |
| Rctrl50__Gnone__input_bilinear__seed2 | enter_r24_at_epoch11 | 1.58 | 1 |
| Rctrl50__Gnone__input_bilinear__seed2 | enter_r28_at_epoch12 | 1.07 | 1 |
| Rctrl50__Gnone__input_bilinear__seed2 | enter_r32_at_epoch13 | 0.91 | 0 |
| Rctrl65__Gnone__input_bilinear__seed0 | enter_r20_at_epoch10 | 0.96 | 0 |
| Rctrl65__Gnone__input_bilinear__seed0 | enter_r24_at_epoch11 | 1.28 | 1 |
| Rctrl65__Gnone__input_bilinear__seed0 | enter_r28_at_epoch12 | 0.90 | 0 |
| Rctrl65__Gnone__input_bilinear__seed0 | enter_r32_at_epoch13 | 0.90 | 0 |
| Rctrl65__Gnone__input_bilinear__seed1 | enter_r20_at_epoch10 | 2.50 | 1 |
| Rctrl65__Gnone__input_bilinear__seed1 | enter_r24_at_epoch11 | 1.53 | 1 |
| Rctrl65__Gnone__input_bilinear__seed1 | enter_r28_at_epoch12 | 0.68 | 0 |
| Rctrl65__Gnone__input_bilinear__seed1 | enter_r32_at_epoch13 | 1.05 | 1 |
| Rctrl65__Gnone__input_bilinear__seed2 | enter_r20_at_epoch10 | 0.50 | 0 |
| Rctrl65__Gnone__input_bilinear__seed2 | enter_r24_at_epoch11 | 1.34 | 1 |
| Rctrl65__Gnone__input_bilinear__seed2 | enter_r28_at_epoch12 | 1.10 | 1 |
| Rctrl65__Gnone__input_bilinear__seed2 | enter_r32_at_epoch13 | 0.75 | 0 |
| Rlin12__Gnone__input_bilinear__seed0 | enter_r17_at_epoch2 | 0.93 | 0 |
| Rlin12__Gnone__input_bilinear__seed0 | enter_r19_at_epoch3 | 1.57 | 1 |
| Rlin12__Gnone__input_bilinear__seed0 | enter_r20_at_epoch4 | 1.34 | 1 |
| Rlin12__Gnone__input_bilinear__seed0 | enter_r21_at_epoch5 | 0.90 | 0 |
| Rlin12__Gnone__input_bilinear__seed0 | enter_r23_at_epoch6 | 0.77 | 0 |
| Rlin12__Gnone__input_bilinear__seed0 | enter_r24_at_epoch7 | 1.11 | 1 |
| Rlin12__Gnone__input_bilinear__seed0 | enter_r25_at_epoch8 | 0.86 | 0 |
| Rlin12__Gnone__input_bilinear__seed0 | enter_r27_at_epoch9 | 1.86 | 1 |
| Rlin12__Gnone__input_bilinear__seed0 | enter_r28_at_epoch10 | 1.13 | 1 |
| Rlin12__Gnone__input_bilinear__seed0 | enter_r29_at_epoch11 | 0.53 | 0 |
| Rlin12__Gnone__input_bilinear__seed0 | enter_r31_at_epoch12 | 1.29 | 1 |
| Rlin12__Gnone__input_bilinear__seed0 | enter_r32_at_epoch13 | 0.66 | 0 |
| Rlin12__Gnone__input_bilinear__seed1 | enter_r17_at_epoch2 | 1.70 | 1 |
| Rlin12__Gnone__input_bilinear__seed1 | enter_r19_at_epoch3 | 0.50 | 0 |
| Rlin12__Gnone__input_bilinear__seed1 | enter_r20_at_epoch4 | 1.53 | 1 |
| Rlin12__Gnone__input_bilinear__seed1 | enter_r21_at_epoch5 | 1.01 | 1 |
| Rlin12__Gnone__input_bilinear__seed1 | enter_r23_at_epoch6 | 0.69 | 0 |
| Rlin12__Gnone__input_bilinear__seed1 | enter_r24_at_epoch7 | 1.48 | 1 |
| Rlin12__Gnone__input_bilinear__seed1 | enter_r25_at_epoch8 | 1.17 | 1 |
| Rlin12__Gnone__input_bilinear__seed1 | enter_r27_at_epoch9 | 0.72 | 0 |
| Rlin12__Gnone__input_bilinear__seed1 | enter_r28_at_epoch10 | 1.27 | 1 |
| Rlin12__Gnone__input_bilinear__seed1 | enter_r29_at_epoch11 | 1.81 | 1 |
| Rlin12__Gnone__input_bilinear__seed1 | enter_r31_at_epoch12 | 0.54 | 0 |
| Rlin12__Gnone__input_bilinear__seed1 | enter_r32_at_epoch13 | 1.24 | 3 |
| Rlin12__Gnone__input_bilinear__seed2 | enter_r17_at_epoch2 | 4.12 | 1 |
| Rlin12__Gnone__input_bilinear__seed2 | enter_r19_at_epoch3 | 0.53 | 0 |
| Rlin12__Gnone__input_bilinear__seed2 | enter_r20_at_epoch4 | 0.81 | 0 |
| Rlin12__Gnone__input_bilinear__seed2 | enter_r21_at_epoch5 | 2.08 | 1 |
| Rlin12__Gnone__input_bilinear__seed2 | enter_r23_at_epoch6 | 1.67 | 1 |
| Rlin12__Gnone__input_bilinear__seed2 | enter_r24_at_epoch7 | 0.64 | 0 |
| Rlin12__Gnone__input_bilinear__seed2 | enter_r25_at_epoch8 | 0.77 | 0 |
| Rlin12__Gnone__input_bilinear__seed2 | enter_r27_at_epoch9 | 0.69 | 0 |
| Rlin12__Gnone__input_bilinear__seed2 | enter_r28_at_epoch10 | 0.85 | 0 |
| Rlin12__Gnone__input_bilinear__seed2 | enter_r29_at_epoch11 | 2.43 | 1 |
| Rlin12__Gnone__input_bilinear__seed2 | enter_r31_at_epoch12 | 0.59 | 0 |
| Rlin12__Gnone__input_bilinear__seed2 | enter_r32_at_epoch13 | 0.91 | 0 |
| Rlin12__Gplateau__input_bilinear__seed0 | enter_r17_at_epoch2 | 1.21 | 1 |
| Rlin12__Gplateau__input_bilinear__seed0 | enter_r19_at_epoch3 | 0.88 | 0 |
| Rlin12__Gplateau__input_bilinear__seed0 | enter_r20_at_epoch4 | 1.45 | 1 |
| Rlin12__Gplateau__input_bilinear__seed0 | enter_r21_at_epoch5 | 0.72 | 0 |
| Rlin12__Gplateau__input_bilinear__seed0 | enter_r23_at_epoch6 | 1.17 | 1 |
| Rlin12__Gplateau__input_bilinear__seed0 | enter_r24_at_epoch7 | 1.15 | 1 |
| Rlin12__Gplateau__input_bilinear__seed0 | enter_r25_at_epoch8 | 0.47 | 0 |
| Rlin12__Gplateau__input_bilinear__seed0 | enter_r27_at_epoch9 | 2.02 | 1 |
| Rlin12__Gplateau__input_bilinear__seed0 | enter_r28_at_epoch10 | 1.34 | 1 |
| Rlin12__Gplateau__input_bilinear__seed0 | enter_r29_at_epoch11 | 0.71 | 0 |
| Rlin12__Gplateau__input_bilinear__seed0 | enter_r31_at_epoch12 | 1.13 | 1 |
| Rlin12__Gplateau__input_bilinear__seed0 | enter_r32_at_epoch13 | 0.78 | 0 |
| Rlin12__Gplateau__input_bilinear__seed1 | enter_r17_at_epoch2 | 1.60 | 1 |
| Rlin12__Gplateau__input_bilinear__seed1 | enter_r19_at_epoch3 | 1.10 | 1 |
| Rlin12__Gplateau__input_bilinear__seed1 | enter_r20_at_epoch4 | 0.66 | 0 |
| Rlin12__Gplateau__input_bilinear__seed1 | enter_r21_at_epoch5 | 1.06 | 1 |
| Rlin12__Gplateau__input_bilinear__seed1 | enter_r23_at_epoch6 | 0.96 | 0 |
| Rlin12__Gplateau__input_bilinear__seed1 | enter_r24_at_epoch7 | 1.44 | 1 |
| Rlin12__Gplateau__input_bilinear__seed1 | enter_r25_at_epoch8 | 1.20 | 1 |
| Rlin12__Gplateau__input_bilinear__seed1 | enter_r27_at_epoch9 | 0.93 | 0 |
| Rlin12__Gplateau__input_bilinear__seed1 | enter_r28_at_epoch10 | 1.19 | 1 |
| Rlin12__Gplateau__input_bilinear__seed1 | enter_r29_at_epoch11 | 1.53 | 1 |
| Rlin12__Gplateau__input_bilinear__seed1 | enter_r31_at_epoch12 | 0.46 | 0 |
| Rlin12__Gplateau__input_bilinear__seed1 | enter_r32_at_epoch13 | 1.83 | 3 |
| Rlin12__Gplateau__input_bilinear__seed2 | enter_r17_at_epoch2 | 2.63 | 1 |
| Rlin12__Gplateau__input_bilinear__seed2 | enter_r19_at_epoch3 | 1.36 | 1 |
| Rlin12__Gplateau__input_bilinear__seed2 | enter_r20_at_epoch4 | 0.65 | 0 |
| Rlin12__Gplateau__input_bilinear__seed2 | enter_r21_at_epoch5 | 1.33 | 1 |
| Rlin12__Gplateau__input_bilinear__seed2 | enter_r23_at_epoch6 | 1.16 | 1 |
| Rlin12__Gplateau__input_bilinear__seed2 | enter_r24_at_epoch7 | 0.80 | 0 |
| Rlin12__Gplateau__input_bilinear__seed2 | enter_r25_at_epoch8 | 1.08 | 1 |
| Rlin12__Gplateau__input_bilinear__seed2 | enter_r27_at_epoch9 | 0.92 | 0 |
| Rlin12__Gplateau__input_bilinear__seed2 | enter_r28_at_epoch10 | 0.67 | 0 |
| Rlin12__Gplateau__input_bilinear__seed2 | enter_r29_at_epoch11 | 1.84 | 1 |
| Rlin12__Gplateau__input_bilinear__seed2 | enter_r31_at_epoch12 | 0.64 | 0 |
| Rlin12__Gplateau__input_bilinear__seed2 | enter_r32_at_epoch13 | 1.30 | 4 |
| Rlin12even__Gnone__input_bilinear__seed0 | enter_r18_at_epoch2 | 0.64 | 0 |
| Rlin12even__Gnone__input_bilinear__seed0 | enter_r20_at_epoch4 | 1.05 | 1 |
| Rlin12even__Gnone__input_bilinear__seed0 | enter_r22_at_epoch5 | 0.68 | 0 |
| Rlin12even__Gnone__input_bilinear__seed0 | enter_r24_at_epoch7 | 2.08 | 1 |
| Rlin12even__Gnone__input_bilinear__seed0 | enter_r26_at_epoch8 | 0.35 | 0 |
| Rlin12even__Gnone__input_bilinear__seed0 | enter_r28_at_epoch10 | 0.60 | 0 |
| Rlin12even__Gnone__input_bilinear__seed0 | enter_r30_at_epoch11 | 1.62 | 2 |
| Rlin12even__Gnone__input_bilinear__seed0 | enter_r32_at_epoch13 | 1.03 | 3 |
| Rlin12even__Gnone__input_bilinear__seed1 | enter_r18_at_epoch2 | 1.64 | 2 |
| Rlin12even__Gnone__input_bilinear__seed1 | enter_r20_at_epoch4 | 0.70 | 0 |
| Rlin12even__Gnone__input_bilinear__seed1 | enter_r22_at_epoch5 | 1.34 | 2 |
| Rlin12even__Gnone__input_bilinear__seed1 | enter_r24_at_epoch7 | 1.71 | 1 |
| Rlin12even__Gnone__input_bilinear__seed1 | enter_r26_at_epoch8 | 0.85 | 0 |
| Rlin12even__Gnone__input_bilinear__seed1 | enter_r28_at_epoch10 | 1.02 | 1 |
| Rlin12even__Gnone__input_bilinear__seed1 | enter_r30_at_epoch11 | 2.77 | 2 |
| Rlin12even__Gnone__input_bilinear__seed1 | enter_r32_at_epoch13 | 1.08 | 1 |
| Rlin12even__Gnone__input_bilinear__seed2 | enter_r18_at_epoch2 | 3.37 | 2 |
| Rlin12even__Gnone__input_bilinear__seed2 | enter_r20_at_epoch4 | 1.88 | 1 |
| Rlin12even__Gnone__input_bilinear__seed2 | enter_r22_at_epoch5 | 0.95 | 0 |
| Rlin12even__Gnone__input_bilinear__seed2 | enter_r24_at_epoch7 | 0.49 | 0 |
| Rlin12even__Gnone__input_bilinear__seed2 | enter_r26_at_epoch8 | 1.13 | 1 |
| Rlin12even__Gnone__input_bilinear__seed2 | enter_r28_at_epoch10 | 0.73 | 0 |
| Rlin12even__Gnone__input_bilinear__seed2 | enter_r30_at_epoch11 | 1.90 | 2 |
| Rlin12even__Gnone__input_bilinear__seed2 | enter_r32_at_epoch13 | 0.51 | 0 |
| Rprog__Gnone__input_bilinear__seed0 | enter_r24_at_epoch7 | 1.89 | 4 |
| Rprog__Gnone__input_bilinear__seed0 | enter_r32_at_epoch13 | 0.49 | 0 |
| Rprog__Gnone__input_bilinear__seed1 | enter_r24_at_epoch7 | 1.60 | 4 |
| Rprog__Gnone__input_bilinear__seed1 | enter_r32_at_epoch13 | 1.10 | 1 |
| Rprog__Gnone__input_bilinear__seed2 | enter_r24_at_epoch7 | 0.90 | 0 |
| Rprog__Gnone__input_bilinear__seed2 | enter_r32_at_epoch13 | 1.56 | 4 |
| Rprog__Gplateau__input_bilinear__seed0 | enter_r24_at_epoch7 | 2.64 | 4 |
| Rprog__Gplateau__input_bilinear__seed0 | enter_r32_at_epoch13 | 1.09 | 2 |
| Rprog__Gplateau__input_bilinear__seed1 | enter_r24_at_epoch7 | 1.99 | 4 |
| Rprog__Gplateau__input_bilinear__seed1 | enter_r32_at_epoch13 | 1.17 | 4 |
| Rprog__Gplateau__input_bilinear__seed2 | enter_r24_at_epoch7 | 1.96 | 4 |
| Rprog__Gplateau__input_bilinear__seed2 | enter_r32_at_epoch13 | 3.32 | 4 |
| Rsteps4__Gnone__input_bilinear__seed0 | enter_r20_at_epoch4 | 1.41 | 1 |
| Rsteps4__Gnone__input_bilinear__seed0 | enter_r24_at_epoch7 | 1.66 | 1 |
| Rsteps4__Gnone__input_bilinear__seed0 | enter_r28_at_epoch10 | 1.03 | 1 |
| Rsteps4__Gnone__input_bilinear__seed0 | enter_r32_at_epoch13 | 1.02 | 1 |
| Rsteps4__Gnone__input_bilinear__seed1 | enter_r20_at_epoch4 | 1.65 | 3 |
| Rsteps4__Gnone__input_bilinear__seed1 | enter_r24_at_epoch7 | 1.02 | 2 |
| Rsteps4__Gnone__input_bilinear__seed1 | enter_r28_at_epoch10 | 2.09 | 3 |
| Rsteps4__Gnone__input_bilinear__seed1 | enter_r32_at_epoch13 | 1.01 | 1 |
| Rsteps4__Gnone__input_bilinear__seed2 | enter_r20_at_epoch4 | 2.25 | 3 |
| Rsteps4__Gnone__input_bilinear__seed2 | enter_r24_at_epoch7 | 0.57 | 0 |
| Rsteps4__Gnone__input_bilinear__seed2 | enter_r28_at_epoch10 | 0.60 | 0 |
| Rsteps4__Gnone__input_bilinear__seed2 | enter_r32_at_epoch13 | 1.12 | 1 |
| Rsteps4__Gplateau__input_bilinear__seed0 | enter_r20_at_epoch4 | 1.30 | 1 |
| Rsteps4__Gplateau__input_bilinear__seed0 | enter_r24_at_epoch7 | 1.29 | 3 |
| Rsteps4__Gplateau__input_bilinear__seed0 | enter_r28_at_epoch10 | 1.31 | 3 |
| Rsteps4__Gplateau__input_bilinear__seed0 | enter_r32_at_epoch13 | 0.91 | 0 |
| Rsteps4__Gplateau__input_bilinear__seed1 | enter_r20_at_epoch4 | 1.90 | 2 |
| Rsteps4__Gplateau__input_bilinear__seed1 | enter_r24_at_epoch7 | 1.23 | 3 |
| Rsteps4__Gplateau__input_bilinear__seed1 | enter_r28_at_epoch10 | 1.64 | 2 |
| Rsteps4__Gplateau__input_bilinear__seed1 | enter_r32_at_epoch13 | 1.21 | 4 |
| Rsteps4__Gplateau__input_bilinear__seed2 | enter_r20_at_epoch4 | 0.76 | 0 |
| Rsteps4__Gplateau__input_bilinear__seed2 | enter_r24_at_epoch7 | 2.50 | 3 |
| Rsteps4__Gplateau__input_bilinear__seed2 | enter_r28_at_epoch10 | 0.90 | 0 |
| Rsteps4__Gplateau__input_bilinear__seed2 | enter_r32_at_epoch13 | 2.16 | 4 |

## 4. Spearman rank correlation of ||g|| with monitor CE inside a stage

| run | stage | rho | n |
|---|---|---:|---:|
| R32__Gnone__input_bilinear__seed0 | stage0_r32 | 0.478 | 31 |
| R32__Gnone__input_bilinear__seed1 | stage0_r32 | 0.508 | 31 |
| R32__Gnone__input_bilinear__seed2 | stage0_r32 | 0.552 | 31 |
| Rb12_18__Gnone__input_bilinear__seed0 | stage0_r16 | 0.044 | 13 |
| Rb12_18__Gnone__input_bilinear__seed0 | stage1_r24 | 0.486 | 6 |
| Rb12_18__Gnone__input_bilinear__seed0 | stage2_r32 | 0.231 | 12 |
| Rb3_12__Gnone__input_bilinear__seed0 | stage0_r16 | 0.400 | 4 |
| Rb3_12__Gnone__input_bilinear__seed0 | stage1_r24 | 0.183 | 9 |
| Rb3_12__Gnone__input_bilinear__seed0 | stage2_r32 | 0.511 | 18 |
| Rb3_9__Gnone__input_bilinear__seed0 | stage0_r16 | 0.400 | 4 |
| Rb3_9__Gnone__input_bilinear__seed0 | stage1_r24 | 0.371 | 6 |
| Rb3_9__Gnone__input_bilinear__seed0 | stage2_r32 | 0.788 | 21 |
| Rb6_18__Gnone__input_bilinear__seed0 | stage0_r16 | 0.464 | 7 |
| Rb6_18__Gnone__input_bilinear__seed0 | stage1_r24 | 0.098 | 12 |
| Rb6_18__Gnone__input_bilinear__seed0 | stage2_r32 | 0.503 | 12 |
| Rb6_9__Gnone__input_bilinear__seed0 | stage0_r16 | 0.464 | 7 |
| Rb6_9__Gnone__input_bilinear__seed0 | stage2_r32 | 0.745 | 21 |
| Rb9_15__Gnone__input_bilinear__seed0 | stage0_r16 | 0.612 | 10 |
| Rb9_15__Gnone__input_bilinear__seed0 | stage1_r24 | -0.371 | 6 |
| Rb9_15__Gnone__input_bilinear__seed0 | stage2_r32 | 0.811 | 15 |
| Rb9_18__Gnone__input_bilinear__seed0 | stage0_r16 | 0.564 | 10 |
| Rb9_18__Gnone__input_bilinear__seed0 | stage1_r24 | 0.433 | 9 |
| Rb9_18__Gnone__input_bilinear__seed0 | stage2_r32 | 0.189 | 12 |
| Rctrl50__Gnone__input_bilinear__seed0 | stage0_r16 | 0.552 | 10 |
| Rctrl50__Gnone__input_bilinear__seed0 | stage4_r32 | 0.690 | 18 |
| Rctrl50__Gnone__input_bilinear__seed1 | stage0_r16 | 0.491 | 10 |
| Rctrl50__Gnone__input_bilinear__seed1 | stage4_r32 | 0.740 | 18 |
| Rctrl50__Gnone__input_bilinear__seed2 | stage0_r16 | -0.018 | 10 |
| Rctrl50__Gnone__input_bilinear__seed2 | stage4_r32 | 0.333 | 18 |
| Rctrl65__Gnone__input_bilinear__seed0 | stage0_r16 | 0.394 | 10 |
| Rctrl65__Gnone__input_bilinear__seed0 | stage4_r32 | 0.531 | 18 |
| Rctrl65__Gnone__input_bilinear__seed1 | stage0_r16 | 0.479 | 10 |
| Rctrl65__Gnone__input_bilinear__seed1 | stage4_r32 | 0.740 | 18 |
| Rctrl65__Gnone__input_bilinear__seed2 | stage0_r16 | -0.285 | 10 |
| Rctrl65__Gnone__input_bilinear__seed2 | stage4_r32 | 0.327 | 18 |
| Rlin12__Gnone__input_bilinear__seed0 | stage12_r32 | 0.672 | 18 |
| Rlin12__Gnone__input_bilinear__seed1 | stage12_r32 | 0.703 | 18 |
| Rlin12__Gnone__input_bilinear__seed2 | stage12_r32 | 0.352 | 18 |
| Rlin12__Gplateau__input_bilinear__seed0 | stage12_r32 | 0.779 | 18 |
| Rlin12__Gplateau__input_bilinear__seed1 | stage12_r32 | 0.789 | 18 |
| Rlin12__Gplateau__input_bilinear__seed2 | stage12_r32 | 0.600 | 18 |
| Rlin12even__Gnone__input_bilinear__seed0 | stage8_r32 | 0.556 | 18 |
| Rlin12even__Gnone__input_bilinear__seed1 | stage8_r32 | 0.785 | 18 |
| Rlin12even__Gnone__input_bilinear__seed2 | stage8_r32 | 0.375 | 18 |
| Rmixed__Gnone__input_bilinear__seed0 | stage0_r32 | 0.833 | 31 |
| Rmixed__Gnone__input_bilinear__seed1 | stage0_r32 | 0.842 | 31 |
| Rmixed__Gnone__input_bilinear__seed2 | stage0_r32 | 0.750 | 31 |
| Rprog__Gnone__input_bilinear__seed0 | stage0_r16 | 0.464 | 7 |
| Rprog__Gnone__input_bilinear__seed0 | stage1_r24 | -0.429 | 6 |
| Rprog__Gnone__input_bilinear__seed0 | stage2_r32 | 0.618 | 18 |
| Rprog__Gnone__input_bilinear__seed1 | stage0_r16 | 0.607 | 7 |
| Rprog__Gnone__input_bilinear__seed1 | stage1_r24 | -0.714 | 6 |
| Rprog__Gnone__input_bilinear__seed1 | stage2_r32 | 0.713 | 18 |
| Rprog__Gnone__input_bilinear__seed2 | stage0_r16 | 0.000 | 7 |
| Rprog__Gnone__input_bilinear__seed2 | stage1_r24 | 0.771 | 6 |
| Rprog__Gnone__input_bilinear__seed2 | stage2_r32 | 0.709 | 18 |
| Rprog__Gplateau__input_bilinear__seed0 | stage0_r16 | 0.536 | 7 |
| Rprog__Gplateau__input_bilinear__seed0 | stage1_r24 | -0.029 | 6 |
| Rprog__Gplateau__input_bilinear__seed0 | stage2_r32 | 0.602 | 18 |
| Rprog__Gplateau__input_bilinear__seed1 | stage0_r16 | 0.893 | 7 |
| Rprog__Gplateau__input_bilinear__seed1 | stage1_r24 | 0.143 | 6 |
| Rprog__Gplateau__input_bilinear__seed1 | stage2_r32 | 0.725 | 18 |
| Rprog__Gplateau__input_bilinear__seed2 | stage0_r16 | 0.179 | 7 |
| Rprog__Gplateau__input_bilinear__seed2 | stage1_r24 | 0.943 | 6 |
| Rprog__Gplateau__input_bilinear__seed2 | stage2_r32 | 0.581 | 18 |
| Rsteps4__Gnone__input_bilinear__seed0 | stage0_r16 | 0.400 | 4 |
| Rsteps4__Gnone__input_bilinear__seed0 | stage4_r32 | 0.775 | 18 |
| Rsteps4__Gnone__input_bilinear__seed1 | stage0_r16 | 0.800 | 4 |
| Rsteps4__Gnone__input_bilinear__seed1 | stage4_r32 | 0.732 | 18 |
| Rsteps4__Gnone__input_bilinear__seed2 | stage0_r16 | 0.400 | 4 |
| Rsteps4__Gnone__input_bilinear__seed2 | stage4_r32 | 0.637 | 18 |
| Rsteps4__Gplateau__input_bilinear__seed0 | stage0_r16 | 0.400 | 4 |
| Rsteps4__Gplateau__input_bilinear__seed0 | stage4_r32 | 0.775 | 18 |
| Rsteps4__Gplateau__input_bilinear__seed1 | stage0_r16 | 1.000 | 4 |
| Rsteps4__Gplateau__input_bilinear__seed1 | stage4_r32 | 0.891 | 18 |
| Rsteps4__Gplateau__input_bilinear__seed2 | stage0_r16 | 0.200 | 4 |
| Rsteps4__Gplateau__input_bilinear__seed2 | stage4_r32 | 0.690 | 18 |

## 5. Trigger replay: first epoch each criterion would fire (fixed switch at the stage's last epoch)

- **Rb12_18__Gnone__input_bilinear__seed0 / stage0_r16_switch_at_epoch12**: gradnorm_plateau_eps0.02=11, gradnorm_plateau_eps0.05=11, gradnorm_plateau_eps0.10=11, gradnorm_plateau_eps0.20=11, cos_next_below_0.9=1, cos_next_below_0.8=2, cos_next_below_0.7=2, cos_next_below_0.5=2, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=10, lookahead_ce_rises=12
- **Rb12_18__Gnone__input_bilinear__seed0 / stage1_r24_switch_at_epoch18**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=17, gradnorm_plateau_eps0.20=17, cos_next_below_0.9=13, cos_next_below_0.8=13, cos_next_below_0.7=13, cos_next_below_0.5=16, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rb3_12__Gnone__input_bilinear__seed0 / stage0_r16_switch_at_epoch3**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=None, gradnorm_plateau_eps0.20=None, cos_next_below_0.9=1, cos_next_below_0.8=2, cos_next_below_0.7=2, cos_next_below_0.5=2, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rb3_12__Gnone__input_bilinear__seed0 / stage1_r24_switch_at_epoch12**: gradnorm_plateau_eps0.02=10, gradnorm_plateau_eps0.05=10, gradnorm_plateau_eps0.10=10, gradnorm_plateau_eps0.20=10, cos_next_below_0.9=5, cos_next_below_0.8=6, cos_next_below_0.7=7, cos_next_below_0.5=7, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rb3_9__Gnone__input_bilinear__seed0 / stage0_r16_switch_at_epoch3**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=None, gradnorm_plateau_eps0.20=None, cos_next_below_0.9=1, cos_next_below_0.8=2, cos_next_below_0.7=3, cos_next_below_0.5=3, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rb3_9__Gnone__input_bilinear__seed0 / stage1_r24_switch_at_epoch9**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=None, gradnorm_plateau_eps0.20=None, cos_next_below_0.9=5, cos_next_below_0.8=5, cos_next_below_0.7=7, cos_next_below_0.5=7, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rb6_18__Gnone__input_bilinear__seed0 / stage0_r16_switch_at_epoch6**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=None, gradnorm_plateau_eps0.20=None, cos_next_below_0.9=1, cos_next_below_0.8=2, cos_next_below_0.7=2, cos_next_below_0.5=3, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rb6_18__Gnone__input_bilinear__seed0 / stage1_r24_switch_at_epoch18**: gradnorm_plateau_eps0.02=11, gradnorm_plateau_eps0.05=11, gradnorm_plateau_eps0.10=11, gradnorm_plateau_eps0.20=11, cos_next_below_0.9=7, cos_next_below_0.8=7, cos_next_below_0.7=7, cos_next_below_0.5=7, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=16, lookahead_ce_rises=None
- **Rb6_9__Gnone__input_bilinear__seed0 / stage0_r16_switch_at_epoch6**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=None, gradnorm_plateau_eps0.20=None, cos_next_below_0.9=2, cos_next_below_0.8=2, cos_next_below_0.7=2, cos_next_below_0.5=3, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rb6_9__Gnone__input_bilinear__seed0 / stage1_r24_switch_at_epoch9**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=None, gradnorm_plateau_eps0.20=None, cos_next_below_0.9=7, cos_next_below_0.8=7, cos_next_below_0.7=7, cos_next_below_0.5=8, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rb9_15__Gnone__input_bilinear__seed0 / stage0_r16_switch_at_epoch9**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=None, gradnorm_plateau_eps0.20=None, cos_next_below_0.9=2, cos_next_below_0.8=2, cos_next_below_0.7=2, cos_next_below_0.5=3, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rb9_15__Gnone__input_bilinear__seed0 / stage1_r24_switch_at_epoch15**: gradnorm_plateau_eps0.02=14, gradnorm_plateau_eps0.05=14, gradnorm_plateau_eps0.10=14, gradnorm_plateau_eps0.20=14, cos_next_below_0.9=10, cos_next_below_0.8=10, cos_next_below_0.7=10, cos_next_below_0.5=None, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rb9_18__Gnone__input_bilinear__seed0 / stage0_r16_switch_at_epoch9**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=None, gradnorm_plateau_eps0.20=None, cos_next_below_0.9=2, cos_next_below_0.8=2, cos_next_below_0.7=2, cos_next_below_0.5=2, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rb9_18__Gnone__input_bilinear__seed0 / stage1_r24_switch_at_epoch18**: gradnorm_plateau_eps0.02=16, gradnorm_plateau_eps0.05=16, gradnorm_plateau_eps0.10=16, gradnorm_plateau_eps0.20=14, cos_next_below_0.9=10, cos_next_below_0.8=10, cos_next_below_0.7=10, cos_next_below_0.5=17, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=18, lookahead_ce_rises=None
- **Rctrl50__Gnone__input_bilinear__seed0 / stage0_r16_switch_at_epoch9**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=9, gradnorm_plateau_eps0.20=9, cos_next_below_0.9=2, cos_next_below_0.8=2, cos_next_below_0.7=4, cos_next_below_0.5=8, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rctrl50__Gnone__input_bilinear__seed1 / stage0_r16_switch_at_epoch9**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=7, gradnorm_plateau_eps0.20=7, cos_next_below_0.9=2, cos_next_below_0.8=3, cos_next_below_0.7=3, cos_next_below_0.5=9, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rctrl50__Gnone__input_bilinear__seed2 / stage0_r16_switch_at_epoch9**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=None, gradnorm_plateau_eps0.20=7, cos_next_below_0.9=1, cos_next_below_0.8=1, cos_next_below_0.7=1, cos_next_below_0.5=8, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rctrl65__Gnone__input_bilinear__seed0 / stage0_r16_switch_at_epoch9**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=None, gradnorm_plateau_eps0.20=9, cos_next_below_0.9=2, cos_next_below_0.8=2, cos_next_below_0.7=8, cos_next_below_0.5=8, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rctrl65__Gnone__input_bilinear__seed1 / stage0_r16_switch_at_epoch9**: gradnorm_plateau_eps0.02=7, gradnorm_plateau_eps0.05=7, gradnorm_plateau_eps0.10=7, gradnorm_plateau_eps0.20=7, cos_next_below_0.9=2, cos_next_below_0.8=3, cos_next_below_0.7=4, cos_next_below_0.5=9, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rctrl65__Gnone__input_bilinear__seed2 / stage0_r16_switch_at_epoch9**: gradnorm_plateau_eps0.02=7, gradnorm_plateau_eps0.05=7, gradnorm_plateau_eps0.10=7, gradnorm_plateau_eps0.20=7, cos_next_below_0.9=1, cos_next_below_0.8=1, cos_next_below_0.7=None, cos_next_below_0.5=None, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rprog__Gnone__input_bilinear__seed0 / stage0_r16_switch_at_epoch6**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=None, gradnorm_plateau_eps0.20=None, cos_next_below_0.9=1, cos_next_below_0.8=2, cos_next_below_0.7=2, cos_next_below_0.5=3, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rprog__Gnone__input_bilinear__seed0 / stage1_r24_switch_at_epoch12**: gradnorm_plateau_eps0.02=11, gradnorm_plateau_eps0.05=11, gradnorm_plateau_eps0.10=11, gradnorm_plateau_eps0.20=11, cos_next_below_0.9=7, cos_next_below_0.8=7, cos_next_below_0.7=7, cos_next_below_0.5=7, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rprog__Gnone__input_bilinear__seed1 / stage0_r16_switch_at_epoch6**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=None, gradnorm_plateau_eps0.20=None, cos_next_below_0.9=1, cos_next_below_0.8=1, cos_next_below_0.7=3, cos_next_below_0.5=3, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rprog__Gnone__input_bilinear__seed1 / stage1_r24_switch_at_epoch12**: gradnorm_plateau_eps0.02=11, gradnorm_plateau_eps0.05=11, gradnorm_plateau_eps0.10=11, gradnorm_plateau_eps0.20=11, cos_next_below_0.9=7, cos_next_below_0.8=7, cos_next_below_0.7=7, cos_next_below_0.5=9, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rprog__Gnone__input_bilinear__seed2 / stage0_r16_switch_at_epoch6**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=None, gradnorm_plateau_eps0.20=None, cos_next_below_0.9=1, cos_next_below_0.8=1, cos_next_below_0.7=1, cos_next_below_0.5=1, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rprog__Gnone__input_bilinear__seed2 / stage1_r24_switch_at_epoch12**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=None, gradnorm_plateau_eps0.20=None, cos_next_below_0.9=7, cos_next_below_0.8=10, cos_next_below_0.7=10, cos_next_below_0.5=11, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rprog__Gplateau__input_bilinear__seed0 / stage0_r16_switch_at_epoch6**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=None, gradnorm_plateau_eps0.20=None, cos_next_below_0.9=1, cos_next_below_0.8=1, cos_next_below_0.7=1, cos_next_below_0.5=1, lookahead_plateau_eps0.02=5, lookahead_plateau_eps0.05=5, lookahead_ce_rises=4
- **Rprog__Gplateau__input_bilinear__seed0 / stage1_r24_switch_at_epoch12**: gradnorm_plateau_eps0.02=11, gradnorm_plateau_eps0.05=11, gradnorm_plateau_eps0.10=11, gradnorm_plateau_eps0.20=11, cos_next_below_0.9=7, cos_next_below_0.8=7, cos_next_below_0.7=7, cos_next_below_0.5=7, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rprog__Gplateau__input_bilinear__seed1 / stage0_r16_switch_at_epoch6**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=None, gradnorm_plateau_eps0.20=6, cos_next_below_0.9=1, cos_next_below_0.8=1, cos_next_below_0.7=1, cos_next_below_0.5=1, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rprog__Gplateau__input_bilinear__seed1 / stage1_r24_switch_at_epoch12**: gradnorm_plateau_eps0.02=12, gradnorm_plateau_eps0.05=12, gradnorm_plateau_eps0.10=11, gradnorm_plateau_eps0.20=11, cos_next_below_0.9=7, cos_next_below_0.8=7, cos_next_below_0.7=7, cos_next_below_0.5=9, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=10
- **Rprog__Gplateau__input_bilinear__seed2 / stage0_r16_switch_at_epoch6**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=None, gradnorm_plateau_eps0.20=None, cos_next_below_0.9=1, cos_next_below_0.8=1, cos_next_below_0.7=1, cos_next_below_0.5=1, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rprog__Gplateau__input_bilinear__seed2 / stage1_r24_switch_at_epoch12**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=None, gradnorm_plateau_eps0.20=None, cos_next_below_0.9=7, cos_next_below_0.8=7, cos_next_below_0.7=7, cos_next_below_0.5=8, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=8
- **Rsteps4__Gnone__input_bilinear__seed0 / stage0_r16_switch_at_epoch3**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=None, gradnorm_plateau_eps0.20=None, cos_next_below_0.9=2, cos_next_below_0.8=2, cos_next_below_0.7=None, cos_next_below_0.5=None, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rsteps4__Gnone__input_bilinear__seed0 / stage1_r20_switch_at_epoch6**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=None, gradnorm_plateau_eps0.20=None, cos_next_below_0.9=4, cos_next_below_0.8=5, cos_next_below_0.7=5, cos_next_below_0.5=None, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rsteps4__Gnone__input_bilinear__seed0 / stage2_r24_switch_at_epoch9**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=None, gradnorm_plateau_eps0.20=None, cos_next_below_0.9=7, cos_next_below_0.8=7, cos_next_below_0.7=7, cos_next_below_0.5=8, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rsteps4__Gnone__input_bilinear__seed0 / stage3_r28_switch_at_epoch12**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=None, gradnorm_plateau_eps0.20=None, cos_next_below_0.9=10, cos_next_below_0.8=10, cos_next_below_0.7=10, cos_next_below_0.5=12, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rsteps4__Gnone__input_bilinear__seed1 / stage0_r16_switch_at_epoch3**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=None, gradnorm_plateau_eps0.20=None, cos_next_below_0.9=1, cos_next_below_0.8=3, cos_next_below_0.7=None, cos_next_below_0.5=None, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rsteps4__Gnone__input_bilinear__seed1 / stage1_r20_switch_at_epoch6**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=None, gradnorm_plateau_eps0.20=None, cos_next_below_0.9=4, cos_next_below_0.8=4, cos_next_below_0.7=None, cos_next_below_0.5=None, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rsteps4__Gnone__input_bilinear__seed1 / stage2_r24_switch_at_epoch9**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=None, gradnorm_plateau_eps0.20=None, cos_next_below_0.9=7, cos_next_below_0.8=9, cos_next_below_0.7=9, cos_next_below_0.5=None, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rsteps4__Gnone__input_bilinear__seed1 / stage3_r28_switch_at_epoch12**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=None, gradnorm_plateau_eps0.20=None, cos_next_below_0.9=10, cos_next_below_0.8=12, cos_next_below_0.7=12, cos_next_below_0.5=None, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rsteps4__Gnone__input_bilinear__seed2 / stage0_r16_switch_at_epoch3**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=None, gradnorm_plateau_eps0.20=None, cos_next_below_0.9=1, cos_next_below_0.8=None, cos_next_below_0.7=None, cos_next_below_0.5=None, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rsteps4__Gnone__input_bilinear__seed2 / stage1_r20_switch_at_epoch6**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=None, gradnorm_plateau_eps0.20=None, cos_next_below_0.9=4, cos_next_below_0.8=4, cos_next_below_0.7=4, cos_next_below_0.5=None, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rsteps4__Gnone__input_bilinear__seed2 / stage2_r24_switch_at_epoch9**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=None, gradnorm_plateau_eps0.20=None, cos_next_below_0.9=7, cos_next_below_0.8=7, cos_next_below_0.7=7, cos_next_below_0.5=None, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rsteps4__Gnone__input_bilinear__seed2 / stage3_r28_switch_at_epoch12**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=None, gradnorm_plateau_eps0.20=None, cos_next_below_0.9=10, cos_next_below_0.8=10, cos_next_below_0.7=11, cos_next_below_0.5=11, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rsteps4__Gplateau__input_bilinear__seed0 / stage0_r16_switch_at_epoch3**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=None, gradnorm_plateau_eps0.20=None, cos_next_below_0.9=1, cos_next_below_0.8=1, cos_next_below_0.7=1, cos_next_below_0.5=1, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rsteps4__Gplateau__input_bilinear__seed0 / stage1_r20_switch_at_epoch6**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=None, gradnorm_plateau_eps0.20=None, cos_next_below_0.9=4, cos_next_below_0.8=4, cos_next_below_0.7=4, cos_next_below_0.5=5, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rsteps4__Gplateau__input_bilinear__seed0 / stage2_r24_switch_at_epoch9**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=None, gradnorm_plateau_eps0.20=None, cos_next_below_0.9=7, cos_next_below_0.8=7, cos_next_below_0.7=7, cos_next_below_0.5=7, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rsteps4__Gplateau__input_bilinear__seed0 / stage3_r28_switch_at_epoch12**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=None, gradnorm_plateau_eps0.20=None, cos_next_below_0.9=10, cos_next_below_0.8=10, cos_next_below_0.7=10, cos_next_below_0.5=10, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rsteps4__Gplateau__input_bilinear__seed1 / stage0_r16_switch_at_epoch3**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=None, gradnorm_plateau_eps0.20=None, cos_next_below_0.9=1, cos_next_below_0.8=1, cos_next_below_0.7=2, cos_next_below_0.5=None, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rsteps4__Gplateau__input_bilinear__seed1 / stage1_r20_switch_at_epoch6**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=None, gradnorm_plateau_eps0.20=None, cos_next_below_0.9=4, cos_next_below_0.8=5, cos_next_below_0.7=None, cos_next_below_0.5=None, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rsteps4__Gplateau__input_bilinear__seed1 / stage2_r24_switch_at_epoch9**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=None, gradnorm_plateau_eps0.20=None, cos_next_below_0.9=7, cos_next_below_0.8=7, cos_next_below_0.7=7, cos_next_below_0.5=7, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rsteps4__Gplateau__input_bilinear__seed1 / stage3_r28_switch_at_epoch12**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=None, gradnorm_plateau_eps0.20=None, cos_next_below_0.9=10, cos_next_below_0.8=12, cos_next_below_0.7=12, cos_next_below_0.5=None, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rsteps4__Gplateau__input_bilinear__seed2 / stage0_r16_switch_at_epoch3**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=None, gradnorm_plateau_eps0.20=None, cos_next_below_0.9=1, cos_next_below_0.8=1, cos_next_below_0.7=1, cos_next_below_0.5=1, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rsteps4__Gplateau__input_bilinear__seed2 / stage1_r20_switch_at_epoch6**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=None, gradnorm_plateau_eps0.20=None, cos_next_below_0.9=4, cos_next_below_0.8=4, cos_next_below_0.7=5, cos_next_below_0.5=5, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rsteps4__Gplateau__input_bilinear__seed2 / stage2_r24_switch_at_epoch9**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=None, gradnorm_plateau_eps0.20=None, cos_next_below_0.9=7, cos_next_below_0.8=8, cos_next_below_0.7=8, cos_next_below_0.5=8, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None
- **Rsteps4__Gplateau__input_bilinear__seed2 / stage3_r28_switch_at_epoch12**: gradnorm_plateau_eps0.02=None, gradnorm_plateau_eps0.05=None, gradnorm_plateau_eps0.10=None, gradnorm_plateau_eps0.20=None, cos_next_below_0.9=10, cos_next_below_0.8=10, cos_next_below_0.7=11, cos_next_below_0.5=12, lookahead_plateau_eps0.02=None, lookahead_plateau_eps0.05=None, lookahead_ce_rises=None

## 7. Q-04: target-path (32x32) test accuracy while training at a lower resolution

| run | epoch | current path | target path | target after BN recal |
|---|---:|---:|---:|---:|
| Rb12_18__Gnone__input_bilinear__seed0 | 0 | 0.1000 | 0.1000 | 0.0884 |
| Rb12_18__Gnone__input_bilinear__seed0 | 1 | 0.3777 | 0.2807 | 0.3394 |
| Rb12_18__Gnone__input_bilinear__seed0 | 2 | 0.4682 | 0.3141 | 0.4145 |
| Rb12_18__Gnone__input_bilinear__seed0 | 3 | 0.4885 | 0.3394 | 0.4090 |
| Rb12_18__Gnone__input_bilinear__seed0 | 4 | 0.5266 | 0.3083 | 0.4367 |
| Rb12_18__Gnone__input_bilinear__seed0 | 5 | 0.5396 | 0.2524 | 0.4182 |
| Rb12_18__Gnone__input_bilinear__seed0 | 6 | 0.5609 | 0.3052 | 0.4354 |
| Rb12_18__Gnone__input_bilinear__seed0 | 7 | 0.5786 | 0.2677 | 0.4624 |
| Rb12_18__Gnone__input_bilinear__seed0 | 8 | 0.5936 | 0.3058 | 0.4743 |
| Rb12_18__Gnone__input_bilinear__seed0 | 9 | 0.6038 | 0.2878 | 0.4766 |
| Rb12_18__Gnone__input_bilinear__seed0 | 10 | 0.6035 | 0.2991 | 0.4848 |
| Rb12_18__Gnone__input_bilinear__seed0 | 11 | 0.6079 | 0.3003 | 0.4802 |
| Rb12_18__Gnone__input_bilinear__seed0 | 12 | 0.6147 | 0.2739 | 0.4836 |
| Rb12_18__Gnone__input_bilinear__seed0 | 13 | 0.6666 | 0.6226 | 0.6342 |
| Rb12_18__Gnone__input_bilinear__seed0 | 14 | 0.6797 | 0.6297 | 0.6511 |
| Rb12_18__Gnone__input_bilinear__seed0 | 15 | 0.6872 | 0.6321 | 0.6746 |
| Rb12_18__Gnone__input_bilinear__seed0 | 16 | 0.7017 | 0.6607 | 0.6827 |
| Rb12_18__Gnone__input_bilinear__seed0 | 17 | 0.7168 | 0.6829 | 0.6934 |
| Rb12_18__Gnone__input_bilinear__seed0 | 18 | 0.7149 | 0.6739 | 0.6938 |
| Rb3_12__Gnone__input_bilinear__seed0 | 0 | 0.1000 | 0.1000 | 0.0884 |
| Rb3_12__Gnone__input_bilinear__seed0 | 1 | 0.3765 | 0.2787 | 0.3450 |
| Rb3_12__Gnone__input_bilinear__seed0 | 2 | 0.4692 | 0.3353 | 0.4145 |
| Rb3_12__Gnone__input_bilinear__seed0 | 3 | 0.4941 | 0.3506 | 0.4082 |
| Rb3_12__Gnone__input_bilinear__seed0 | 4 | 0.5284 | 0.5200 | 0.5437 |
| Rb3_12__Gnone__input_bilinear__seed0 | 5 | 0.5634 | 0.5255 | 0.5503 |
| Rb3_12__Gnone__input_bilinear__seed0 | 6 | 0.6061 | 0.5824 | 0.5932 |
| Rb3_12__Gnone__input_bilinear__seed0 | 7 | 0.6345 | 0.6048 | 0.6184 |
| Rb3_12__Gnone__input_bilinear__seed0 | 8 | 0.6607 | 0.6329 | 0.6469 |
| Rb3_12__Gnone__input_bilinear__seed0 | 9 | 0.6675 | 0.6217 | 0.6599 |
| Rb3_12__Gnone__input_bilinear__seed0 | 10 | 0.6660 | 0.6208 | 0.6708 |
| Rb3_12__Gnone__input_bilinear__seed0 | 11 | 0.6866 | 0.6371 | 0.6710 |
| Rb3_12__Gnone__input_bilinear__seed0 | 12 | 0.6966 | 0.6556 | 0.6781 |
| Rb3_9__Gnone__input_bilinear__seed0 | 0 | 0.1000 | 0.1000 | 0.0884 |
| Rb3_9__Gnone__input_bilinear__seed0 | 1 | 0.3782 | 0.2814 | 0.3465 |
| Rb3_9__Gnone__input_bilinear__seed0 | 2 | 0.4642 | 0.3119 | 0.4058 |
| Rb3_9__Gnone__input_bilinear__seed0 | 3 | 0.4896 | 0.3359 | 0.4114 |
| Rb3_9__Gnone__input_bilinear__seed0 | 4 | 0.5370 | 0.5262 | 0.5443 |
| Rb3_9__Gnone__input_bilinear__seed0 | 5 | 0.5740 | 0.5444 | 0.5627 |
| Rb3_9__Gnone__input_bilinear__seed0 | 6 | 0.6023 | 0.5650 | 0.5843 |
| Rb3_9__Gnone__input_bilinear__seed0 | 7 | 0.6367 | 0.6171 | 0.6265 |
| Rb3_9__Gnone__input_bilinear__seed0 | 8 | 0.6586 | 0.6376 | 0.6509 |
| Rb3_9__Gnone__input_bilinear__seed0 | 9 | 0.6630 | 0.6327 | 0.6556 |
| Rb6_18__Gnone__input_bilinear__seed0 | 0 | 0.1000 | 0.1000 | 0.0884 |
| Rb6_18__Gnone__input_bilinear__seed0 | 1 | 0.3787 | 0.2873 | 0.3447 |
| Rb6_18__Gnone__input_bilinear__seed0 | 2 | 0.4630 | 0.3214 | 0.4136 |
| Rb6_18__Gnone__input_bilinear__seed0 | 3 | 0.4861 | 0.3478 | 0.4096 |
| Rb6_18__Gnone__input_bilinear__seed0 | 4 | 0.5269 | 0.3022 | 0.4413 |
| Rb6_18__Gnone__input_bilinear__seed0 | 5 | 0.5380 | 0.2609 | 0.4220 |
| Rb6_18__Gnone__input_bilinear__seed0 | 6 | 0.5634 | 0.3105 | 0.4478 |
| Rb6_18__Gnone__input_bilinear__seed0 | 7 | 0.6048 | 0.5693 | 0.5836 |
| Rb6_18__Gnone__input_bilinear__seed0 | 8 | 0.6372 | 0.6037 | 0.6202 |
| Rb6_18__Gnone__input_bilinear__seed0 | 9 | 0.6486 | 0.6008 | 0.6412 |
| Rb6_18__Gnone__input_bilinear__seed0 | 10 | 0.6600 | 0.6080 | 0.6632 |
| Rb6_18__Gnone__input_bilinear__seed0 | 11 | 0.6778 | 0.6459 | 0.6715 |
| Rb6_18__Gnone__input_bilinear__seed0 | 12 | 0.6852 | 0.6512 | 0.6806 |
| Rb6_18__Gnone__input_bilinear__seed0 | 13 | 0.6993 | 0.6724 | 0.6941 |
| Rb6_18__Gnone__input_bilinear__seed0 | 14 | 0.7021 | 0.6706 | 0.6960 |
| Rb6_18__Gnone__input_bilinear__seed0 | 15 | 0.7008 | 0.6391 | 0.6913 |
| Rb6_18__Gnone__input_bilinear__seed0 | 16 | 0.7181 | 0.6837 | 0.7032 |
| Rb6_18__Gnone__input_bilinear__seed0 | 17 | 0.7178 | 0.6853 | 0.7005 |
| Rb6_18__Gnone__input_bilinear__seed0 | 18 | 0.7228 | 0.6641 | 0.7063 |
| Rb6_9__Gnone__input_bilinear__seed0 | 0 | 0.1000 | 0.1000 | 0.0884 |
| Rb6_9__Gnone__input_bilinear__seed0 | 1 | 0.3749 | 0.2840 | 0.3453 |
| Rb6_9__Gnone__input_bilinear__seed0 | 2 | 0.4602 | 0.3197 | 0.4139 |
| Rb6_9__Gnone__input_bilinear__seed0 | 3 | 0.4946 | 0.3362 | 0.4076 |
| Rb6_9__Gnone__input_bilinear__seed0 | 4 | 0.5266 | 0.3292 | 0.4310 |
| Rb6_9__Gnone__input_bilinear__seed0 | 5 | 0.5325 | 0.2577 | 0.4218 |
| Rb6_9__Gnone__input_bilinear__seed0 | 6 | 0.5633 | 0.3177 | 0.4299 |
| Rb6_9__Gnone__input_bilinear__seed0 | 7 | 0.6075 | 0.5835 | 0.5912 |
| Rb6_9__Gnone__input_bilinear__seed0 | 8 | 0.6402 | 0.6196 | 0.6219 |
| Rb6_9__Gnone__input_bilinear__seed0 | 9 | 0.6542 | 0.6258 | 0.6447 |
| Rb9_15__Gnone__input_bilinear__seed0 | 0 | 0.1000 | 0.1000 | 0.0884 |
| Rb9_15__Gnone__input_bilinear__seed0 | 1 | 0.3741 | 0.2862 | 0.3455 |
| Rb9_15__Gnone__input_bilinear__seed0 | 2 | 0.4631 | 0.3150 | 0.4171 |
| Rb9_15__Gnone__input_bilinear__seed0 | 3 | 0.4962 | 0.3464 | 0.4189 |
| Rb9_15__Gnone__input_bilinear__seed0 | 4 | 0.5320 | 0.3139 | 0.4257 |
| Rb9_15__Gnone__input_bilinear__seed0 | 5 | 0.5457 | 0.2704 | 0.4243 |
| Rb9_15__Gnone__input_bilinear__seed0 | 6 | 0.5694 | 0.3471 | 0.4309 |
| Rb9_15__Gnone__input_bilinear__seed0 | 7 | 0.5874 | 0.3050 | 0.4519 |
| Rb9_15__Gnone__input_bilinear__seed0 | 8 | 0.6009 | 0.3128 | 0.4555 |
| Rb9_15__Gnone__input_bilinear__seed0 | 9 | 0.6134 | 0.2941 | 0.4707 |
| Rb9_15__Gnone__input_bilinear__seed0 | 10 | 0.6466 | 0.5926 | 0.6218 |
| Rb9_15__Gnone__input_bilinear__seed0 | 11 | 0.6686 | 0.6263 | 0.6391 |
| Rb9_15__Gnone__input_bilinear__seed0 | 12 | 0.6815 | 0.6384 | 0.6563 |
| Rb9_15__Gnone__input_bilinear__seed0 | 13 | 0.6958 | 0.6670 | 0.6791 |
| Rb9_15__Gnone__input_bilinear__seed0 | 14 | 0.7040 | 0.6713 | 0.6827 |
| Rb9_15__Gnone__input_bilinear__seed0 | 15 | 0.7017 | 0.6506 | 0.6978 |
| Rb9_18__Gnone__input_bilinear__seed0 | 0 | 0.1000 | 0.1000 | 0.0884 |
| Rb9_18__Gnone__input_bilinear__seed0 | 1 | 0.3762 | 0.2785 | 0.3436 |
| Rb9_18__Gnone__input_bilinear__seed0 | 2 | 0.4675 | 0.3273 | 0.4147 |
| Rb9_18__Gnone__input_bilinear__seed0 | 3 | 0.4959 | 0.3384 | 0.4108 |
| Rb9_18__Gnone__input_bilinear__seed0 | 4 | 0.5301 | 0.3277 | 0.4247 |
| Rb9_18__Gnone__input_bilinear__seed0 | 5 | 0.5399 | 0.2738 | 0.4220 |
| Rb9_18__Gnone__input_bilinear__seed0 | 6 | 0.5644 | 0.3383 | 0.4407 |
| Rb9_18__Gnone__input_bilinear__seed0 | 7 | 0.5762 | 0.3124 | 0.4530 |
| Rb9_18__Gnone__input_bilinear__seed0 | 8 | 0.6020 | 0.3378 | 0.4744 |
| Rb9_18__Gnone__input_bilinear__seed0 | 9 | 0.6115 | 0.3189 | 0.4727 |
| Rb9_18__Gnone__input_bilinear__seed0 | 10 | 0.6405 | 0.6037 | 0.6245 |
| Rb9_18__Gnone__input_bilinear__seed0 | 11 | 0.6622 | 0.6146 | 0.6377 |
| Rb9_18__Gnone__input_bilinear__seed0 | 12 | 0.6771 | 0.6374 | 0.6597 |
| Rb9_18__Gnone__input_bilinear__seed0 | 13 | 0.6983 | 0.6651 | 0.6829 |
| Rb9_18__Gnone__input_bilinear__seed0 | 14 | 0.7045 | 0.6636 | 0.6781 |
| Rb9_18__Gnone__input_bilinear__seed0 | 15 | 0.7144 | 0.6762 | 0.6967 |
| Rb9_18__Gnone__input_bilinear__seed0 | 16 | 0.7193 | 0.6769 | 0.7004 |
| Rb9_18__Gnone__input_bilinear__seed0 | 17 | 0.7260 | 0.6944 | 0.7103 |
| Rb9_18__Gnone__input_bilinear__seed0 | 18 | 0.7307 | 0.6974 | 0.7075 |
| Rctrl50__Gnone__input_bilinear__seed0 | 0 | 0.1000 | 0.1000 | 0.0884 |
| Rctrl50__Gnone__input_bilinear__seed0 | 1 | 0.3776 | 0.2833 | 0.3482 |
| Rctrl50__Gnone__input_bilinear__seed0 | 2 | 0.4626 | 0.3150 | 0.4083 |
| Rctrl50__Gnone__input_bilinear__seed0 | 3 | 0.4856 | 0.3416 | 0.4102 |
| Rctrl50__Gnone__input_bilinear__seed0 | 4 | 0.5250 | 0.3227 | 0.4308 |
| Rctrl50__Gnone__input_bilinear__seed0 | 5 | 0.5380 | 0.2649 | 0.4197 |
| Rctrl50__Gnone__input_bilinear__seed0 | 6 | 0.5667 | 0.3137 | 0.4363 |
| Rctrl50__Gnone__input_bilinear__seed0 | 7 | 0.5820 | 0.2676 | 0.4612 |
| Rctrl50__Gnone__input_bilinear__seed0 | 8 | 0.5956 | 0.3212 | 0.4739 |
| Rctrl50__Gnone__input_bilinear__seed0 | 9 | 0.6077 | 0.2640 | 0.4794 |
| Rctrl50__Gnone__input_bilinear__seed0 | 10 | 0.6322 | 0.4637 | 0.5667 |
| Rctrl50__Gnone__input_bilinear__seed0 | 11 | 0.6613 | 0.6056 | 0.6320 |
| Rctrl50__Gnone__input_bilinear__seed0 | 12 | 0.6862 | 0.6743 | 0.6866 |
| Rctrl50__Gnone__input_bilinear__seed1 | 0 | 0.0968 | 0.1000 | 0.0977 |
| Rctrl50__Gnone__input_bilinear__seed1 | 1 | 0.3997 | 0.3554 | 0.3827 |
| Rctrl50__Gnone__input_bilinear__seed1 | 2 | 0.4546 | 0.3559 | 0.4155 |
| Rctrl50__Gnone__input_bilinear__seed1 | 3 | 0.5208 | 0.4169 | 0.4466 |
| Rctrl50__Gnone__input_bilinear__seed1 | 4 | 0.5456 | 0.4102 | 0.4705 |
| Rctrl50__Gnone__input_bilinear__seed1 | 5 | 0.5675 | 0.4201 | 0.4731 |
| Rctrl50__Gnone__input_bilinear__seed1 | 6 | 0.5900 | 0.4028 | 0.4771 |
| Rctrl50__Gnone__input_bilinear__seed1 | 7 | 0.5912 | 0.4118 | 0.4973 |
| Rctrl50__Gnone__input_bilinear__seed1 | 8 | 0.6135 | 0.4448 | 0.5190 |
| Rctrl50__Gnone__input_bilinear__seed1 | 9 | 0.6167 | 0.4506 | 0.5045 |
| Rctrl50__Gnone__input_bilinear__seed1 | 10 | 0.6365 | 0.5337 | 0.5805 |
| Rctrl50__Gnone__input_bilinear__seed1 | 11 | 0.6287 | 0.5480 | 0.6508 |
| Rctrl50__Gnone__input_bilinear__seed1 | 12 | 0.6839 | 0.6568 | 0.6978 |
| Rctrl50__Gnone__input_bilinear__seed2 | 0 | 0.1004 | 0.1000 | 0.0880 |
| Rctrl50__Gnone__input_bilinear__seed2 | 1 | 0.4102 | 0.2982 | 0.3796 |
| Rctrl50__Gnone__input_bilinear__seed2 | 2 | 0.4636 | 0.3865 | 0.4360 |
| Rctrl50__Gnone__input_bilinear__seed2 | 3 | 0.5185 | 0.3731 | 0.4222 |
| Rctrl50__Gnone__input_bilinear__seed2 | 4 | 0.5452 | 0.4126 | 0.4470 |
| Rctrl50__Gnone__input_bilinear__seed2 | 5 | 0.5697 | 0.3877 | 0.4580 |
| Rctrl50__Gnone__input_bilinear__seed2 | 6 | 0.5825 | 0.3840 | 0.4559 |
| Rctrl50__Gnone__input_bilinear__seed2 | 7 | 0.5991 | 0.4174 | 0.4770 |
| Rctrl50__Gnone__input_bilinear__seed2 | 8 | 0.6178 | 0.4169 | 0.4773 |
| Rctrl50__Gnone__input_bilinear__seed2 | 9 | 0.6310 | 0.3646 | 0.4845 |
| Rctrl50__Gnone__input_bilinear__seed2 | 10 | 0.6587 | 0.5366 | 0.5750 |
| Rctrl50__Gnone__input_bilinear__seed2 | 11 | 0.6856 | 0.6446 | 0.6507 |
| Rctrl50__Gnone__input_bilinear__seed2 | 12 | 0.7138 | 0.6979 | 0.7080 |
| Rctrl65__Gnone__input_bilinear__seed0 | 0 | 0.1000 | 0.1000 | 0.0884 |
| Rctrl65__Gnone__input_bilinear__seed0 | 1 | 0.3740 | 0.2804 | 0.3438 |
| Rctrl65__Gnone__input_bilinear__seed0 | 2 | 0.4628 | 0.3294 | 0.4142 |
| Rctrl65__Gnone__input_bilinear__seed0 | 3 | 0.4868 | 0.3614 | 0.4138 |
| Rctrl65__Gnone__input_bilinear__seed0 | 4 | 0.5199 | 0.3301 | 0.4277 |
| Rctrl65__Gnone__input_bilinear__seed0 | 5 | 0.5344 | 0.2760 | 0.4160 |
| Rctrl65__Gnone__input_bilinear__seed0 | 6 | 0.5654 | 0.3262 | 0.4224 |
| Rctrl65__Gnone__input_bilinear__seed0 | 7 | 0.5784 | 0.3172 | 0.4535 |
| Rctrl65__Gnone__input_bilinear__seed0 | 8 | 0.5960 | 0.3486 | 0.4711 |
| Rctrl65__Gnone__input_bilinear__seed0 | 9 | 0.6044 | 0.3188 | 0.4744 |
| Rctrl65__Gnone__input_bilinear__seed0 | 10 | 0.6420 | 0.5068 | 0.5734 |
| Rctrl65__Gnone__input_bilinear__seed0 | 11 | 0.6615 | 0.6047 | 0.6270 |
| Rctrl65__Gnone__input_bilinear__seed0 | 12 | 0.6835 | 0.6748 | 0.6821 |
| Rctrl65__Gnone__input_bilinear__seed1 | 0 | 0.0968 | 0.1000 | 0.0977 |
| Rctrl65__Gnone__input_bilinear__seed1 | 1 | 0.3978 | 0.3513 | 0.3833 |
| Rctrl65__Gnone__input_bilinear__seed1 | 2 | 0.4534 | 0.3540 | 0.4170 |
| Rctrl65__Gnone__input_bilinear__seed1 | 3 | 0.5162 | 0.4078 | 0.4376 |
| Rctrl65__Gnone__input_bilinear__seed1 | 4 | 0.5469 | 0.4065 | 0.4614 |
| Rctrl65__Gnone__input_bilinear__seed1 | 5 | 0.5721 | 0.4066 | 0.4841 |
| Rctrl65__Gnone__input_bilinear__seed1 | 6 | 0.5903 | 0.3735 | 0.4802 |
| Rctrl65__Gnone__input_bilinear__seed1 | 7 | 0.5931 | 0.3813 | 0.5074 |
| Rctrl65__Gnone__input_bilinear__seed1 | 8 | 0.6158 | 0.3960 | 0.5093 |
| Rctrl65__Gnone__input_bilinear__seed1 | 9 | 0.6170 | 0.4472 | 0.5011 |
| Rctrl65__Gnone__input_bilinear__seed1 | 10 | 0.6433 | 0.5246 | 0.5858 |
| Rctrl65__Gnone__input_bilinear__seed1 | 11 | 0.6558 | 0.5894 | 0.6534 |
| Rctrl65__Gnone__input_bilinear__seed1 | 12 | 0.7042 | 0.6717 | 0.7035 |
| Rctrl65__Gnone__input_bilinear__seed2 | 0 | 0.1004 | 0.1000 | 0.0880 |
| Rctrl65__Gnone__input_bilinear__seed2 | 1 | 0.4087 | 0.2925 | 0.3763 |
| Rctrl65__Gnone__input_bilinear__seed2 | 2 | 0.4674 | 0.3870 | 0.4320 |
| Rctrl65__Gnone__input_bilinear__seed2 | 3 | 0.5150 | 0.3842 | 0.4271 |
| Rctrl65__Gnone__input_bilinear__seed2 | 4 | 0.5457 | 0.3973 | 0.4344 |
| Rctrl65__Gnone__input_bilinear__seed2 | 5 | 0.5643 | 0.3815 | 0.4399 |
| Rctrl65__Gnone__input_bilinear__seed2 | 6 | 0.5793 | 0.3504 | 0.4475 |
| Rctrl65__Gnone__input_bilinear__seed2 | 7 | 0.5911 | 0.4137 | 0.4653 |
| Rctrl65__Gnone__input_bilinear__seed2 | 8 | 0.6082 | 0.4265 | 0.4811 |
| Rctrl65__Gnone__input_bilinear__seed2 | 9 | 0.6260 | 0.3961 | 0.4774 |
| Rctrl65__Gnone__input_bilinear__seed2 | 10 | 0.6630 | 0.5417 | 0.5670 |
| Rctrl65__Gnone__input_bilinear__seed2 | 11 | 0.6826 | 0.6481 | 0.6561 |
| Rctrl65__Gnone__input_bilinear__seed2 | 12 | 0.7133 | 0.6901 | 0.7029 |
| Rlin12__Gnone__input_bilinear__seed0 | 0 | 0.1000 | 0.1000 | 0.0884 |
| Rlin12__Gnone__input_bilinear__seed0 | 1 | 0.3777 | 0.2908 | 0.3463 |
| Rlin12__Gnone__input_bilinear__seed0 | 2 | 0.4578 | 0.3577 | 0.4241 |
| Rlin12__Gnone__input_bilinear__seed0 | 3 | 0.4986 | 0.4157 | 0.4518 |
| Rlin12__Gnone__input_bilinear__seed0 | 4 | 0.5244 | 0.4740 | 0.4924 |
| Rlin12__Gnone__input_bilinear__seed0 | 5 | 0.5565 | 0.4927 | 0.5171 |
| Rlin12__Gnone__input_bilinear__seed0 | 6 | 0.5990 | 0.5360 | 0.5678 |
| Rlin12__Gnone__input_bilinear__seed0 | 7 | 0.6284 | 0.5957 | 0.6180 |
| Rlin12__Gnone__input_bilinear__seed0 | 8 | 0.6601 | 0.6467 | 0.6511 |
| Rlin12__Gnone__input_bilinear__seed0 | 9 | 0.6780 | 0.6512 | 0.6822 |
| Rlin12__Gnone__input_bilinear__seed0 | 10 | 0.6855 | 0.6839 | 0.7088 |
| Rlin12__Gnone__input_bilinear__seed0 | 11 | 0.7170 | 0.6921 | 0.7160 |
| Rlin12__Gnone__input_bilinear__seed0 | 12 | 0.7295 | 0.6940 | 0.7365 |
| Rlin12__Gnone__input_bilinear__seed1 | 0 | 0.0968 | 0.1000 | 0.0977 |
| Rlin12__Gnone__input_bilinear__seed1 | 1 | 0.3962 | 0.3492 | 0.3840 |
| Rlin12__Gnone__input_bilinear__seed1 | 2 | 0.4462 | 0.3821 | 0.4270 |
| Rlin12__Gnone__input_bilinear__seed1 | 3 | 0.5150 | 0.4726 | 0.4875 |
| Rlin12__Gnone__input_bilinear__seed1 | 4 | 0.5527 | 0.4709 | 0.5311 |
| Rlin12__Gnone__input_bilinear__seed1 | 5 | 0.5864 | 0.5243 | 0.5701 |
| Rlin12__Gnone__input_bilinear__seed1 | 6 | 0.6209 | 0.5920 | 0.6056 |
| Rlin12__Gnone__input_bilinear__seed1 | 7 | 0.6470 | 0.6048 | 0.6476 |
| Rlin12__Gnone__input_bilinear__seed1 | 8 | 0.6644 | 0.6274 | 0.6746 |
| Rlin12__Gnone__input_bilinear__seed1 | 9 | 0.6945 | 0.6624 | 0.6962 |
| Rlin12__Gnone__input_bilinear__seed1 | 10 | 0.7041 | 0.6737 | 0.7200 |
| Rlin12__Gnone__input_bilinear__seed1 | 11 | 0.7076 | 0.6701 | 0.7325 |
| Rlin12__Gnone__input_bilinear__seed1 | 12 | 0.7443 | 0.6913 | 0.7422 |
| Rlin12__Gnone__input_bilinear__seed2 | 0 | 0.1004 | 0.1000 | 0.0880 |
| Rlin12__Gnone__input_bilinear__seed2 | 1 | 0.4064 | 0.2961 | 0.3769 |
| Rlin12__Gnone__input_bilinear__seed2 | 2 | 0.4594 | 0.3898 | 0.4338 |
| Rlin12__Gnone__input_bilinear__seed2 | 3 | 0.5178 | 0.4363 | 0.4836 |
| Rlin12__Gnone__input_bilinear__seed2 | 4 | 0.5614 | 0.4908 | 0.5303 |
| Rlin12__Gnone__input_bilinear__seed2 | 5 | 0.6009 | 0.5088 | 0.5712 |
| Rlin12__Gnone__input_bilinear__seed2 | 6 | 0.6142 | 0.5527 | 0.6032 |
| Rlin12__Gnone__input_bilinear__seed2 | 7 | 0.6483 | 0.6102 | 0.6320 |
| Rlin12__Gnone__input_bilinear__seed2 | 8 | 0.6698 | 0.6393 | 0.6648 |
| Rlin12__Gnone__input_bilinear__seed2 | 9 | 0.6998 | 0.6814 | 0.6875 |
| Rlin12__Gnone__input_bilinear__seed2 | 10 | 0.7199 | 0.7031 | 0.7195 |
| Rlin12__Gnone__input_bilinear__seed2 | 11 | 0.7360 | 0.7147 | 0.7325 |
| Rlin12__Gnone__input_bilinear__seed2 | 12 | 0.7423 | 0.7134 | 0.7450 |
| Rlin12__Gplateau__input_bilinear__seed0 | 0 | 0.1000 | 0.1000 | 0.0884 |
| Rlin12__Gplateau__input_bilinear__seed0 | 1 | 0.4218 | 0.1222 | 0.2877 |
| Rlin12__Gplateau__input_bilinear__seed0 | 2 | 0.4782 | 0.1044 | 0.2971 |
| Rlin12__Gplateau__input_bilinear__seed0 | 3 | 0.5156 | 0.1070 | 0.2638 |
| Rlin12__Gplateau__input_bilinear__seed0 | 4 | 0.5601 | 0.1227 | 0.3003 |
| Rlin12__Gplateau__input_bilinear__seed0 | 5 | 0.5745 | 0.1036 | 0.3209 |
| Rlin12__Gplateau__input_bilinear__seed0 | 6 | 0.6033 | 0.1259 | 0.2867 |
| Rlin12__Gplateau__input_bilinear__seed0 | 7 | 0.6383 | 0.1258 | 0.4070 |
| Rlin12__Gplateau__input_bilinear__seed0 | 8 | 0.6677 | 0.1785 | 0.4621 |
| Rlin12__Gplateau__input_bilinear__seed0 | 9 | 0.6723 | 0.1033 | 0.4015 |
| Rlin12__Gplateau__input_bilinear__seed0 | 10 | 0.6944 | 0.1959 | 0.5190 |
| Rlin12__Gplateau__input_bilinear__seed0 | 11 | 0.7137 | 0.1903 | 0.5171 |
| Rlin12__Gplateau__input_bilinear__seed0 | 12 | 0.7251 | 0.2071 | 0.5089 |
| Rlin12__Gplateau__input_bilinear__seed1 | 0 | 0.1028 | 0.1000 | 0.0977 |
| Rlin12__Gplateau__input_bilinear__seed1 | 1 | 0.4261 | 0.1643 | 0.2858 |
| Rlin12__Gplateau__input_bilinear__seed1 | 2 | 0.4904 | 0.1689 | 0.3090 |
| Rlin12__Gplateau__input_bilinear__seed1 | 3 | 0.5157 | 0.1464 | 0.3079 |
| Rlin12__Gplateau__input_bilinear__seed1 | 4 | 0.5713 | 0.1572 | 0.3334 |
| Rlin12__Gplateau__input_bilinear__seed1 | 5 | 0.5910 | 0.1086 | 0.3217 |
| Rlin12__Gplateau__input_bilinear__seed1 | 6 | 0.6019 | 0.1864 | 0.3112 |
| Rlin12__Gplateau__input_bilinear__seed1 | 7 | 0.6445 | 0.1465 | 0.4282 |
| Rlin12__Gplateau__input_bilinear__seed1 | 8 | 0.6569 | 0.1293 | 0.4238 |
| Rlin12__Gplateau__input_bilinear__seed1 | 9 | 0.6755 | 0.1134 | 0.4172 |
| Rlin12__Gplateau__input_bilinear__seed1 | 10 | 0.6945 | 0.1907 | 0.4928 |
| Rlin12__Gplateau__input_bilinear__seed1 | 11 | 0.7019 | 0.1726 | 0.4785 |
| Rlin12__Gplateau__input_bilinear__seed1 | 12 | 0.7275 | 0.1787 | 0.4826 |
| Rlin12__Gplateau__input_bilinear__seed2 | 0 | 0.0999 | 0.1000 | 0.0880 |
| Rlin12__Gplateau__input_bilinear__seed2 | 1 | 0.4488 | 0.2381 | 0.2910 |
| Rlin12__Gplateau__input_bilinear__seed2 | 2 | 0.4992 | 0.2021 | 0.3535 |
| Rlin12__Gplateau__input_bilinear__seed2 | 3 | 0.5382 | 0.2010 | 0.3023 |
| Rlin12__Gplateau__input_bilinear__seed2 | 4 | 0.5793 | 0.2371 | 0.3748 |
| Rlin12__Gplateau__input_bilinear__seed2 | 5 | 0.5958 | 0.1781 | 0.3969 |
| Rlin12__Gplateau__input_bilinear__seed2 | 6 | 0.6199 | 0.1715 | 0.3625 |
| Rlin12__Gplateau__input_bilinear__seed2 | 7 | 0.6559 | 0.1976 | 0.4148 |
| Rlin12__Gplateau__input_bilinear__seed2 | 8 | 0.6793 | 0.1877 | 0.4376 |
| Rlin12__Gplateau__input_bilinear__seed2 | 9 | 0.6992 | 0.1353 | 0.4362 |
| Rlin12__Gplateau__input_bilinear__seed2 | 10 | 0.7175 | 0.1894 | 0.5004 |
| Rlin12__Gplateau__input_bilinear__seed2 | 11 | 0.7315 | 0.1982 | 0.5445 |
| Rlin12__Gplateau__input_bilinear__seed2 | 12 | 0.7518 | 0.1738 | 0.5042 |
| Rlin12even__Gnone__input_bilinear__seed0 | 0 | 0.1000 | 0.1000 | 0.0884 |
| Rlin12even__Gnone__input_bilinear__seed0 | 1 | 0.3717 | 0.2855 | 0.3461 |
| Rlin12even__Gnone__input_bilinear__seed0 | 2 | 0.4557 | 0.3688 | 0.4391 |
| Rlin12even__Gnone__input_bilinear__seed0 | 3 | 0.4882 | 0.3938 | 0.4455 |
| Rlin12even__Gnone__input_bilinear__seed0 | 4 | 0.5281 | 0.4473 | 0.5080 |
| Rlin12even__Gnone__input_bilinear__seed0 | 5 | 0.5607 | 0.5159 | 0.5168 |
| Rlin12even__Gnone__input_bilinear__seed0 | 6 | 0.5987 | 0.5221 | 0.5549 |
| Rlin12even__Gnone__input_bilinear__seed0 | 7 | 0.6313 | 0.6047 | 0.6260 |
| Rlin12even__Gnone__input_bilinear__seed0 | 8 | 0.6704 | 0.6548 | 0.6657 |
| Rlin12even__Gnone__input_bilinear__seed0 | 9 | 0.6764 | 0.6506 | 0.6742 |
| Rlin12even__Gnone__input_bilinear__seed0 | 10 | 0.7031 | 0.6941 | 0.7164 |
| Rlin12even__Gnone__input_bilinear__seed0 | 11 | 0.7162 | 0.7011 | 0.7226 |
| Rlin12even__Gnone__input_bilinear__seed0 | 12 | 0.7292 | 0.7178 | 0.7412 |
| Rlin12even__Gnone__input_bilinear__seed1 | 0 | 0.0968 | 0.1000 | 0.0977 |
| Rlin12even__Gnone__input_bilinear__seed1 | 1 | 0.3984 | 0.3469 | 0.3826 |
| Rlin12even__Gnone__input_bilinear__seed1 | 2 | 0.4454 | 0.3755 | 0.4354 |
| Rlin12even__Gnone__input_bilinear__seed1 | 3 | 0.5132 | 0.4307 | 0.4726 |
| Rlin12even__Gnone__input_bilinear__seed1 | 4 | 0.5585 | 0.5052 | 0.5295 |
| Rlin12even__Gnone__input_bilinear__seed1 | 5 | 0.5907 | 0.5283 | 0.5734 |
| Rlin12even__Gnone__input_bilinear__seed1 | 6 | 0.6065 | 0.5831 | 0.5917 |
| Rlin12even__Gnone__input_bilinear__seed1 | 7 | 0.6334 | 0.5831 | 0.6430 |
| Rlin12even__Gnone__input_bilinear__seed1 | 8 | 0.6699 | 0.6300 | 0.6757 |
| Rlin12even__Gnone__input_bilinear__seed1 | 9 | 0.6873 | 0.6571 | 0.6889 |
| Rlin12even__Gnone__input_bilinear__seed1 | 10 | 0.7199 | 0.6895 | 0.7206 |
| Rlin12even__Gnone__input_bilinear__seed1 | 11 | 0.7078 | 0.6629 | 0.7336 |
| Rlin12even__Gnone__input_bilinear__seed1 | 12 | 0.7434 | 0.7057 | 0.7405 |
| Rlin12even__Gnone__input_bilinear__seed2 | 0 | 0.1004 | 0.1000 | 0.0880 |
| Rlin12even__Gnone__input_bilinear__seed2 | 1 | 0.4075 | 0.2989 | 0.3769 |
| Rlin12even__Gnone__input_bilinear__seed2 | 2 | 0.4652 | 0.4015 | 0.4471 |
| Rlin12even__Gnone__input_bilinear__seed2 | 3 | 0.5191 | 0.4162 | 0.4632 |
| Rlin12even__Gnone__input_bilinear__seed2 | 4 | 0.5523 | 0.4964 | 0.5276 |
| Rlin12even__Gnone__input_bilinear__seed2 | 5 | 0.6060 | 0.5332 | 0.5774 |
| Rlin12even__Gnone__input_bilinear__seed2 | 6 | 0.6038 | 0.5249 | 0.5951 |
| Rlin12even__Gnone__input_bilinear__seed2 | 7 | 0.6420 | 0.6161 | 0.6252 |
| Rlin12even__Gnone__input_bilinear__seed2 | 8 | 0.6679 | 0.6357 | 0.6682 |
| Rlin12even__Gnone__input_bilinear__seed2 | 9 | 0.6934 | 0.6671 | 0.6856 |
| Rlin12even__Gnone__input_bilinear__seed2 | 10 | 0.7143 | 0.7004 | 0.7164 |
| Rlin12even__Gnone__input_bilinear__seed2 | 11 | 0.7265 | 0.6970 | 0.7284 |
| Rlin12even__Gnone__input_bilinear__seed2 | 12 | 0.7397 | 0.7156 | 0.7366 |
| Rprog__Gnone__input_bilinear__seed0 | 0 | 0.1000 | 0.1000 | 0.0884 |
| Rprog__Gnone__input_bilinear__seed0 | 1 | 0.3783 | 0.2909 | 0.3437 |
| Rprog__Gnone__input_bilinear__seed0 | 2 | 0.4637 | 0.3213 | 0.4125 |
| Rprog__Gnone__input_bilinear__seed0 | 3 | 0.4960 | 0.3328 | 0.4073 |
| Rprog__Gnone__input_bilinear__seed0 | 4 | 0.5270 | 0.3086 | 0.4234 |
| Rprog__Gnone__input_bilinear__seed0 | 5 | 0.5373 | 0.2485 | 0.4319 |
| Rprog__Gnone__input_bilinear__seed0 | 6 | 0.5646 | 0.3045 | 0.4389 |
| Rprog__Gnone__input_bilinear__seed0 | 7 | 0.6070 | 0.5651 | 0.5845 |
| Rprog__Gnone__input_bilinear__seed0 | 8 | 0.6423 | 0.6108 | 0.6241 |
| Rprog__Gnone__input_bilinear__seed0 | 9 | 0.6557 | 0.6113 | 0.6423 |
| Rprog__Gnone__input_bilinear__seed0 | 10 | 0.6674 | 0.6272 | 0.6694 |
| Rprog__Gnone__input_bilinear__seed0 | 11 | 0.6809 | 0.6368 | 0.6657 |
| Rprog__Gnone__input_bilinear__seed0 | 12 | 0.6883 | 0.6562 | 0.6824 |
| Rprog__Gnone__input_bilinear__seed1 | 0 | 0.0968 | 0.1000 | 0.0977 |
| Rprog__Gnone__input_bilinear__seed1 | 1 | 0.3984 | 0.3561 | 0.3837 |
| Rprog__Gnone__input_bilinear__seed1 | 2 | 0.4464 | 0.3489 | 0.4163 |
| Rprog__Gnone__input_bilinear__seed1 | 3 | 0.5216 | 0.4262 | 0.4413 |
| Rprog__Gnone__input_bilinear__seed1 | 4 | 0.5415 | 0.4089 | 0.4652 |
| Rprog__Gnone__input_bilinear__seed1 | 5 | 0.5737 | 0.4238 | 0.4846 |
| Rprog__Gnone__input_bilinear__seed1 | 6 | 0.5916 | 0.4062 | 0.4790 |
| Rprog__Gnone__input_bilinear__seed1 | 7 | 0.6135 | 0.5813 | 0.6126 |
| Rprog__Gnone__input_bilinear__seed1 | 8 | 0.6526 | 0.6141 | 0.6495 |
| Rprog__Gnone__input_bilinear__seed1 | 9 | 0.6813 | 0.6499 | 0.6630 |
| Rprog__Gnone__input_bilinear__seed1 | 10 | 0.6791 | 0.6324 | 0.6678 |
| Rprog__Gnone__input_bilinear__seed1 | 11 | 0.6856 | 0.6400 | 0.6905 |
| Rprog__Gnone__input_bilinear__seed1 | 12 | 0.7051 | 0.6723 | 0.6983 |
| Rprog__Gnone__input_bilinear__seed2 | 0 | 0.1004 | 0.1000 | 0.0880 |
| Rprog__Gnone__input_bilinear__seed2 | 1 | 0.4101 | 0.3019 | 0.3769 |
| Rprog__Gnone__input_bilinear__seed2 | 2 | 0.4664 | 0.3809 | 0.4292 |
| Rprog__Gnone__input_bilinear__seed2 | 3 | 0.5133 | 0.3799 | 0.4214 |
| Rprog__Gnone__input_bilinear__seed2 | 4 | 0.5447 | 0.4008 | 0.4333 |
| Rprog__Gnone__input_bilinear__seed2 | 5 | 0.5600 | 0.3708 | 0.4458 |
| Rprog__Gnone__input_bilinear__seed2 | 6 | 0.5729 | 0.3475 | 0.4529 |
| Rprog__Gnone__input_bilinear__seed2 | 7 | 0.6224 | 0.5905 | 0.6010 |
| Rprog__Gnone__input_bilinear__seed2 | 8 | 0.6454 | 0.5969 | 0.6237 |
| Rprog__Gnone__input_bilinear__seed2 | 9 | 0.6730 | 0.6210 | 0.6552 |
| Rprog__Gnone__input_bilinear__seed2 | 10 | 0.6862 | 0.6416 | 0.6602 |
| Rprog__Gnone__input_bilinear__seed2 | 11 | 0.7049 | 0.6738 | 0.6849 |
| Rprog__Gnone__input_bilinear__seed2 | 12 | 0.7164 | 0.6621 | 0.6924 |
| Rprog__Gplateau__input_bilinear__seed0 | 0 | 0.1000 | 0.1000 | 0.0884 |
| Rprog__Gplateau__input_bilinear__seed0 | 1 | 0.4189 | 0.1302 | 0.2807 |
| Rprog__Gplateau__input_bilinear__seed0 | 2 | 0.4822 | 0.1197 | 0.2651 |
| Rprog__Gplateau__input_bilinear__seed0 | 3 | 0.5214 | 0.1054 | 0.2904 |
| Rprog__Gplateau__input_bilinear__seed0 | 4 | 0.5459 | 0.2286 | 0.3439 |
| Rprog__Gplateau__input_bilinear__seed0 | 5 | 0.5693 | 0.2082 | 0.3659 |
| Rprog__Gplateau__input_bilinear__seed0 | 6 | 0.5986 | 0.2516 | 0.3640 |
| Rprog__Gplateau__input_bilinear__seed0 | 7 | 0.6284 | 0.1281 | 0.4176 |
| Rprog__Gplateau__input_bilinear__seed0 | 8 | 0.6588 | 0.1483 | 0.4708 |
| Rprog__Gplateau__input_bilinear__seed0 | 9 | 0.6728 | 0.1177 | 0.4104 |
| Rprog__Gplateau__input_bilinear__seed0 | 10 | 0.6905 | 0.3006 | 0.5850 |
| Rprog__Gplateau__input_bilinear__seed0 | 11 | 0.6918 | 0.3000 | 0.5722 |
| Rprog__Gplateau__input_bilinear__seed0 | 12 | 0.7098 | 0.3238 | 0.5917 |
| Rprog__Gplateau__input_bilinear__seed1 | 0 | 0.1028 | 0.1000 | 0.0977 |
| Rprog__Gplateau__input_bilinear__seed1 | 1 | 0.4276 | 0.1635 | 0.2817 |
| Rprog__Gplateau__input_bilinear__seed1 | 2 | 0.4954 | 0.1522 | 0.2955 |
| Rprog__Gplateau__input_bilinear__seed1 | 3 | 0.5266 | 0.2002 | 0.3117 |
| Rprog__Gplateau__input_bilinear__seed1 | 4 | 0.5613 | 0.2359 | 0.3707 |
| Rprog__Gplateau__input_bilinear__seed1 | 5 | 0.5775 | 0.1669 | 0.3586 |
| Rprog__Gplateau__input_bilinear__seed1 | 6 | 0.5933 | 0.2040 | 0.3565 |
| Rprog__Gplateau__input_bilinear__seed1 | 7 | 0.6280 | 0.1641 | 0.4343 |
| Rprog__Gplateau__input_bilinear__seed1 | 8 | 0.6523 | 0.1464 | 0.4111 |
| Rprog__Gplateau__input_bilinear__seed1 | 9 | 0.6793 | 0.1304 | 0.4287 |
| Rprog__Gplateau__input_bilinear__seed1 | 10 | 0.6941 | 0.2508 | 0.5628 |
| Rprog__Gplateau__input_bilinear__seed1 | 11 | 0.6950 | 0.2268 | 0.5600 |
| Rprog__Gplateau__input_bilinear__seed1 | 12 | 0.7258 | 0.3500 | 0.5988 |
| Rprog__Gplateau__input_bilinear__seed2 | 0 | 0.0999 | 0.1000 | 0.0880 |
| Rprog__Gplateau__input_bilinear__seed2 | 1 | 0.4467 | 0.2256 | 0.2929 |
| Rprog__Gplateau__input_bilinear__seed2 | 2 | 0.5094 | 0.1910 | 0.3160 |
| Rprog__Gplateau__input_bilinear__seed2 | 3 | 0.5271 | 0.2132 | 0.2985 |
| Rprog__Gplateau__input_bilinear__seed2 | 4 | 0.5613 | 0.2807 | 0.4037 |
| Rprog__Gplateau__input_bilinear__seed2 | 5 | 0.5935 | 0.2304 | 0.3837 |
| Rprog__Gplateau__input_bilinear__seed2 | 6 | 0.6094 | 0.2160 | 0.4308 |
| Rprog__Gplateau__input_bilinear__seed2 | 7 | 0.6317 | 0.1785 | 0.4523 |
| Rprog__Gplateau__input_bilinear__seed2 | 8 | 0.6572 | 0.2346 | 0.4737 |
| Rprog__Gplateau__input_bilinear__seed2 | 9 | 0.6875 | 0.1805 | 0.4751 |
| Rprog__Gplateau__input_bilinear__seed2 | 10 | 0.7075 | 0.2778 | 0.5703 |
| Rprog__Gplateau__input_bilinear__seed2 | 11 | 0.7330 | 0.3798 | 0.6048 |
| Rprog__Gplateau__input_bilinear__seed2 | 12 | 0.7406 | 0.3335 | 0.5939 |
| Rsteps4__Gnone__input_bilinear__seed0 | 0 | 0.1000 | 0.1000 | 0.0884 |
| Rsteps4__Gnone__input_bilinear__seed0 | 1 | 0.3783 | 0.2846 | 0.3434 |
| Rsteps4__Gnone__input_bilinear__seed0 | 2 | 0.4663 | 0.3226 | 0.4133 |
| Rsteps4__Gnone__input_bilinear__seed0 | 3 | 0.4906 | 0.3540 | 0.4171 |
| Rsteps4__Gnone__input_bilinear__seed0 | 4 | 0.5247 | 0.4386 | 0.4900 |
| Rsteps4__Gnone__input_bilinear__seed0 | 5 | 0.5570 | 0.4455 | 0.4952 |
| Rsteps4__Gnone__input_bilinear__seed0 | 6 | 0.5884 | 0.4947 | 0.5252 |
| Rsteps4__Gnone__input_bilinear__seed0 | 7 | 0.6200 | 0.5909 | 0.5981 |
| Rsteps4__Gnone__input_bilinear__seed0 | 8 | 0.6589 | 0.6306 | 0.6343 |
| Rsteps4__Gnone__input_bilinear__seed0 | 9 | 0.6696 | 0.6360 | 0.6583 |
| Rsteps4__Gnone__input_bilinear__seed0 | 10 | 0.6828 | 0.6692 | 0.6961 |
| Rsteps4__Gnone__input_bilinear__seed0 | 11 | 0.7142 | 0.6949 | 0.7065 |
| Rsteps4__Gnone__input_bilinear__seed0 | 12 | 0.7262 | 0.7134 | 0.7279 |
| Rsteps4__Gnone__input_bilinear__seed1 | 0 | 0.0968 | 0.1000 | 0.0977 |
| Rsteps4__Gnone__input_bilinear__seed1 | 1 | 0.4007 | 0.3555 | 0.3867 |
| Rsteps4__Gnone__input_bilinear__seed1 | 2 | 0.4465 | 0.3584 | 0.4186 |
| Rsteps4__Gnone__input_bilinear__seed1 | 3 | 0.5109 | 0.4034 | 0.4274 |
| Rsteps4__Gnone__input_bilinear__seed1 | 4 | 0.5546 | 0.4928 | 0.5230 |
| Rsteps4__Gnone__input_bilinear__seed1 | 5 | 0.5826 | 0.5022 | 0.5433 |
| Rsteps4__Gnone__input_bilinear__seed1 | 6 | 0.6082 | 0.5164 | 0.5560 |
| Rsteps4__Gnone__input_bilinear__seed1 | 7 | 0.6404 | 0.6031 | 0.6403 |
| Rsteps4__Gnone__input_bilinear__seed1 | 8 | 0.6698 | 0.6232 | 0.6601 |
| Rsteps4__Gnone__input_bilinear__seed1 | 9 | 0.6934 | 0.6484 | 0.6710 |
| Rsteps4__Gnone__input_bilinear__seed1 | 10 | 0.6952 | 0.6752 | 0.7043 |
| Rsteps4__Gnone__input_bilinear__seed1 | 11 | 0.7078 | 0.6707 | 0.7244 |
| Rsteps4__Gnone__input_bilinear__seed1 | 12 | 0.7297 | 0.7146 | 0.7375 |
| Rsteps4__Gnone__input_bilinear__seed2 | 0 | 0.1004 | 0.1000 | 0.0880 |
| Rsteps4__Gnone__input_bilinear__seed2 | 1 | 0.4077 | 0.2984 | 0.3770 |
| Rsteps4__Gnone__input_bilinear__seed2 | 2 | 0.4620 | 0.3771 | 0.4264 |
| Rsteps4__Gnone__input_bilinear__seed2 | 3 | 0.5197 | 0.3852 | 0.4213 |
| Rsteps4__Gnone__input_bilinear__seed2 | 4 | 0.5505 | 0.4839 | 0.5100 |
| Rsteps4__Gnone__input_bilinear__seed2 | 5 | 0.5875 | 0.5092 | 0.5430 |
| Rsteps4__Gnone__input_bilinear__seed2 | 6 | 0.6044 | 0.4703 | 0.5440 |
| Rsteps4__Gnone__input_bilinear__seed2 | 7 | 0.6431 | 0.6130 | 0.6198 |
| Rsteps4__Gnone__input_bilinear__seed2 | 8 | 0.6688 | 0.6245 | 0.6388 |
| Rsteps4__Gnone__input_bilinear__seed2 | 9 | 0.6908 | 0.6475 | 0.6612 |
| Rsteps4__Gnone__input_bilinear__seed2 | 10 | 0.7153 | 0.6976 | 0.7099 |
| Rsteps4__Gnone__input_bilinear__seed2 | 11 | 0.7265 | 0.7149 | 0.7277 |
| Rsteps4__Gnone__input_bilinear__seed2 | 12 | 0.7404 | 0.7259 | 0.7424 |
| Rsteps4__Gplateau__input_bilinear__seed0 | 0 | 0.1000 | 0.1000 | 0.0884 |
| Rsteps4__Gplateau__input_bilinear__seed0 | 1 | 0.4201 | 0.1305 | 0.2803 |
| Rsteps4__Gplateau__input_bilinear__seed0 | 2 | 0.4860 | 0.1304 | 0.2602 |
| Rsteps4__Gplateau__input_bilinear__seed0 | 3 | 0.5243 | 0.1093 | 0.2939 |
| Rsteps4__Gplateau__input_bilinear__seed0 | 4 | 0.5585 | 0.1351 | 0.3030 |
| Rsteps4__Gplateau__input_bilinear__seed0 | 5 | 0.5866 | 0.1200 | 0.3009 |
| Rsteps4__Gplateau__input_bilinear__seed0 | 6 | 0.6129 | 0.1543 | 0.2854 |
| Rsteps4__Gplateau__input_bilinear__seed0 | 7 | 0.6479 | 0.1583 | 0.4046 |
| Rsteps4__Gplateau__input_bilinear__seed0 | 8 | 0.6602 | 0.1664 | 0.4366 |
| Rsteps4__Gplateau__input_bilinear__seed0 | 9 | 0.6808 | 0.1365 | 0.3913 |
| Rsteps4__Gplateau__input_bilinear__seed0 | 10 | 0.7034 | 0.1392 | 0.5326 |
| Rsteps4__Gplateau__input_bilinear__seed0 | 11 | 0.7027 | 0.1775 | 0.5106 |
| Rsteps4__Gplateau__input_bilinear__seed0 | 12 | 0.7248 | 0.1898 | 0.5050 |
| Rsteps4__Gplateau__input_bilinear__seed1 | 0 | 0.1028 | 0.1000 | 0.0977 |
| Rsteps4__Gplateau__input_bilinear__seed1 | 1 | 0.4275 | 0.1657 | 0.2892 |
| Rsteps4__Gplateau__input_bilinear__seed1 | 2 | 0.4984 | 0.1530 | 0.2964 |
| Rsteps4__Gplateau__input_bilinear__seed1 | 3 | 0.5251 | 0.1975 | 0.3108 |
| Rsteps4__Gplateau__input_bilinear__seed1 | 4 | 0.5610 | 0.1647 | 0.2932 |
| Rsteps4__Gplateau__input_bilinear__seed1 | 5 | 0.5968 | 0.1076 | 0.2867 |
| Rsteps4__Gplateau__input_bilinear__seed1 | 6 | 0.6166 | 0.1564 | 0.2871 |
| Rsteps4__Gplateau__input_bilinear__seed1 | 7 | 0.6407 | 0.1595 | 0.4054 |
| Rsteps4__Gplateau__input_bilinear__seed1 | 8 | 0.6595 | 0.1526 | 0.4071 |
| Rsteps4__Gplateau__input_bilinear__seed1 | 9 | 0.6809 | 0.1448 | 0.4197 |
| Rsteps4__Gplateau__input_bilinear__seed1 | 10 | 0.6882 | 0.1907 | 0.4931 |
| Rsteps4__Gplateau__input_bilinear__seed1 | 11 | 0.7157 | 0.1475 | 0.4737 |
| Rsteps4__Gplateau__input_bilinear__seed1 | 12 | 0.7323 | 0.1667 | 0.5221 |
| Rsteps4__Gplateau__input_bilinear__seed2 | 0 | 0.0999 | 0.1000 | 0.0880 |
| Rsteps4__Gplateau__input_bilinear__seed2 | 1 | 0.4426 | 0.2310 | 0.2934 |
| Rsteps4__Gplateau__input_bilinear__seed2 | 2 | 0.5059 | 0.2013 | 0.3152 |
| Rsteps4__Gplateau__input_bilinear__seed2 | 3 | 0.5347 | 0.2263 | 0.2991 |
| Rsteps4__Gplateau__input_bilinear__seed2 | 4 | 0.5825 | 0.2146 | 0.3517 |
| Rsteps4__Gplateau__input_bilinear__seed2 | 5 | 0.6059 | 0.1834 | 0.3328 |
| Rsteps4__Gplateau__input_bilinear__seed2 | 6 | 0.6316 | 0.1624 | 0.3661 |
| Rsteps4__Gplateau__input_bilinear__seed2 | 7 | 0.6439 | 0.1798 | 0.3998 |
| Rsteps4__Gplateau__input_bilinear__seed2 | 8 | 0.6824 | 0.2080 | 0.4207 |
| Rsteps4__Gplateau__input_bilinear__seed2 | 9 | 0.6923 | 0.1648 | 0.4208 |
| Rsteps4__Gplateau__input_bilinear__seed2 | 10 | 0.7211 | 0.1581 | 0.5096 |
| Rsteps4__Gplateau__input_bilinear__seed2 | 11 | 0.7340 | 0.2690 | 0.5323 |
| Rsteps4__Gplateau__input_bilinear__seed2 | 12 | 0.7469 | 0.2364 | 0.5228 |
