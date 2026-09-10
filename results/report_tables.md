# Study tables

`pacing_vs_order`, 1 seeds per arm, 45 min.

Teacher for `clean_short`: 2 folds x 8 epochs, test acc 0.7392, 0.7323, misclassifies 25.8% of the train set.

Teacher for `noisy20_short`: 2 folds x 8 epochs, test acc 0.6738, 0.6753, misclassifies 44.8% of the train set.

Teacher for `clean_long`: 2 folds x 8 epochs, test acc 0.7392, 0.7323, misclassifies 25.8% of the train set.

Teacher for `noisy20_long`: 2 folds x 8 epochs, test acc 0.6738, 0.6753, misclassifies 44.8% of the train set.

## clean_short

| arm | test acc % | +-sem | peak % (ep) | AUC % | e* |
| --- | --- | --- | --- | --- | --- |
| baseline | 83.48 | -- | 83.48 (7) | 66.60 | 5 |
| curriculum | 83.01 | -- | 83.01 (7) | 70.21 | 5 |
| random | 83.53 | -- | 83.53 (7) | 67.20 | 5 |
| anti | 83.46 | -- | 83.46 (7) | 63.32 | 6 |

| effect | points | +-sem | n | what it isolates |
| --- | --- | --- | --- | --- |
| pacing = random - baseline | +0.05 | -- | 1 | the smaller pool and its repetition, with no difficulty information |
| ordering = curriculum - random | -0.52 | -- | 1 | the difficulty ranking, at identical pacing |
| direction = curriculum - anti | -0.45 | -- | 1 | easy-first against hard-first, same ranking |
| total = curriculum - baseline | -0.47 | -- | 1 | the curriculum as usually reported = pacing + ordering |

*One seed per arm: no interval, and no smallest detectable difference. Every number above is a single draw.*

Exposure under this pacing: the easy decile of the order is drawn 10.1 times over the run and the hard decile 4.0, against 8.0 each for the baseline. Same total draws, different distribution -- which is what `random` exists to price.

## noisy20_short

| arm | test acc % | +-sem | peak % (ep) | AUC % | e* |
| --- | --- | --- | --- | --- | --- |
| baseline | 77.67 | -- | 77.67 (7) | 60.91 | 6 |
| curriculum | 79.65 | -- | 79.65 (7) | 66.66 | 5 |
| random | 77.51 | -- | 77.51 (7) | 62.01 | 6 |
| anti | 74.25 | -- | 74.25 (7) | 51.08 | 7 |

| effect | points | +-sem | n | what it isolates |
| --- | --- | --- | --- | --- |
| pacing = random - baseline | -0.16 | -- | 1 | the smaller pool and its repetition, with no difficulty information |
| ordering = curriculum - random | +2.14 | -- | 1 | the difficulty ranking, at identical pacing |
| direction = curriculum - anti | +5.40 | -- | 1 | easy-first against hard-first, same ranking |
| total = curriculum - baseline | +1.98 | -- | 1 | the curriculum as usually reported = pacing + ordering |

*One seed per arm: no interval, and no smallest detectable difference. Every number above is a single draw.*

Exposure under this pacing: the easy decile of the order is drawn 10.1 times over the run and the hard decile 4.0, against 8.0 each for the baseline. Same total draws, different distribution -- which is what `random` exists to price.

## clean_long

| arm | test acc % | +-sem | peak % (ep) | AUC % | e* |
| --- | --- | --- | --- | --- | --- |
| baseline | 89.72 | -- | 89.83 (18) | 74.22 | 13 |
| curriculum | 88.24 | -- | 88.30 (18) | 74.20 | 14 |
| random | 88.98 | -- | 88.98 (19) | 73.04 | 14 |
| anti | 89.89 | -- | 90.06 (18) | 72.39 | 14 |

| effect | points | +-sem | n | what it isolates |
| --- | --- | --- | --- | --- |
| pacing = random - baseline | -0.74 | -- | 1 | the smaller pool and its repetition, with no difficulty information |
| ordering = curriculum - random | -0.74 | -- | 1 | the difficulty ranking, at identical pacing |
| direction = curriculum - anti | -1.65 | -- | 1 | easy-first against hard-first, same ranking |
| total = curriculum - baseline | -1.48 | -- | 1 | the curriculum as usually reported = pacing + ordering |

*One seed per arm: no interval, and no smallest detectable difference. Every number above is a single draw.*

Exposure under this pacing: the easy decile of the order is drawn 24.4 times over the run and the hard decile 10.0, against 20.0 each for the baseline. Same total draws, different distribution -- which is what `random` exists to price.

## noisy20_long

| arm | test acc % | +-sem | peak % (ep) | AUC % | e* |
| --- | --- | --- | --- | --- | --- |
| baseline | 86.48 | -- | 86.48 (19) | 69.93 | 15 |
| curriculum | 86.16 | -- | 86.16 (19) | 72.89 | 14 |
| random | 85.73 | -- | 85.88 (18) | 68.91 | 15 |
| anti | 85.91 | -- | 86.01 (18) | 63.12 | 14 |

| effect | points | +-sem | n | what it isolates |
| --- | --- | --- | --- | --- |
| pacing = random - baseline | -0.75 | -- | 1 | the smaller pool and its repetition, with no difficulty information |
| ordering = curriculum - random | +0.43 | -- | 1 | the difficulty ranking, at identical pacing |
| direction = curriculum - anti | +0.25 | -- | 1 | easy-first against hard-first, same ranking |
| total = curriculum - baseline | -0.32 | -- | 1 | the curriculum as usually reported = pacing + ordering |

*One seed per arm: no interval, and no smallest detectable difference. Every number above is a single draw.*

Exposure under this pacing: the easy decile of the order is drawn 24.4 times over the run and the hard decile 10.0, against 20.0 each for the baseline. Same total draws, different distribution -- which is what `random` exists to price.

