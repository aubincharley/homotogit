# Curves

`index.html` is a standalone page — nine training settings, epoch by epoch, for
the four frozen methods. Open it directly; it needs no server and no build step.

    open docs/curves/index.html

Also published as an artifact:
<https://claude.ai/artifact/CAQo4UJE16hFpiGQQyjH1c>

`data.json` is generated, never edited by hand:

    python tools/build_curves_data.py

It reads the per-epoch metrics under `results/kaggle_outputs/` — the objective
grid (`cc-loss-*`, `cc-sq-*`), the augmentation grid (`cc-aug-*`, `cc-a60-*`) and
the reference cells (`cc-grid-*`) — averages each cell over its seeds, and
deduplicates on `(arm, method, seed)` so a run retrieved twice is not weighted
twice. Everything is read on the **target** path, the operator at identity, so
the four arms are comparable at every epoch and not only at the end.

## The three sections

| | |
|---|---|
| **the scatter** | 27 points: what the control reaches against what the method adds. The finding is that they lie on one descent, though the baseline was moved three unrelated ways |
| **the decisive experiment** | `no augmentation, 60e` beside `augmentation, 60e` at the same learning rate and the same schedule, so the curriculum is diluted identically in both. It separates "it is the augmentation" from "it is the dilution" |
| **all nine settings** | test accuracy and the train − test gap, with the schedule boundaries (epochs 6, 12, 21) marked |

Read the numbers in [`../AUGMENTATION.md`](../AUGMENTATION.md) and
[`../OBJECTIVES.md`](../OBJECTIVES.md); the handoff summary is
[`../HANDOFF.md`](../HANDOFF.md).
