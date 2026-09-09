Now the job script with the pairing assertions and timing probe.

Caught a real bug locally: images are stored uint8, and the resize must happen after the float conversion. Moving it into the pipeline at the specified position.

Only the pre-existing `test_radius_must_fit_the_image` failure remains (it fails on the pristine tree too). Launching.

Submitted. While it runs, here's where things stand.

## Checks passed before launch

**No existing submission** for either arm (`kaggle kernels list --mine` searched for "progres"/"resol" — not found).

**Spatial configuration** ([results/progressive\_resolution\_verification.json](https://claude.ai/epitaxy/results/progressive_resolution_verification.json)):

| input stage 1 stage 2 stage 3 pooled dim shortcut shapes  |       |       |     |      |                   |
| --------------------------------------------------------- | ----- | ----- | --- | ---- | ----------------- |
| 16×16                                                     | 16×16 | 8×8   | 4×4 | 64 ✓ | match main path ✓ |
| 24×24                                                     | 24×24 | 12×12 | 6×6 | 64 ✓ | match main path ✓ |
| 32×32                                                     | 32×32 | 16×16 | 8×8 | 64 ✓ | match main path ✓ |

All three match the expected table, from the same 269,722 parameters. Pooling is genuinely global — the classifier input is 64-dimensional at every resolution. Forward and backward are finite at all three, with and without the Gaussian, 19 hooks each. `σ=0` is a **bitwise** identity through the whole network at all three resolutions, and the radius-4 kernel applies cleanly to the 4×4 stage-3 maps that 16×16 inputs produce, via the existing explicit reflected padding — no truncation, no skipped stage.

`resize_to(x, 32)` returns the original tensor object — the target input path is exact, not a resize that happens to be near-identity.

**Schedule**: resolution matches the formula at all 30 epochs (boundaries e5=16→e6=24, e11=24→e12=32); effective σ equals `r(e)/32 × σ_ref(e)` to 1e-12 at every epoch; the final nine epochs are all `r=32, σ=0`; and effective σ is confirmed **non-monotone** (0.425 → 0.525 at e6, 0.450 → 0.500 at e12), as intended.

**Pairing** is enforced *inside* the job rather than checked afterwards. I hashed the campaign's `subset`, `train_probe`, `perm_seed0` and `init_seed0` (all 116 tensors), embedded the four sha256 digests in the job, and the study aborts before training if the regenerated state does not reproduce them. So the arms are paired with `plain_seed0` and `plateau_seed0` by verification, not by seed number.

## Two bugs found and fixed locally

A local 4-epoch CPU end-to-end run caught something the static checks missed: **images are stored as uint8**, so resizing them raised `"upsample_bilinear2d_out_frame" not implemented for 'Byte'`. Resizing uint8 would also have been wrong even if it had run. The resize now lives inside `InputPipeline` at the specified position — `uint8/255 → float[0,1] → resize → T_η → normalize` — so it is built from the original floating-point image before normalization, and the per-channel constants from the full original training split are shared unchanged across resolutions.

The second was mine: reordering the pipeline's `describe()` broke an index-based assertion in `test_config_and_engine.py`. Fixed by keeping the base order list intact and documenting the resize under its own key. Full suite now: **170 passed, 1 failed** — `test_gaussian.py::test_radius_must_fit_the_image`, which I confirmed fails identically on the stashed pristine tree (it expects a raise that the phase-8 explicit-reflection work removed). Pre-existing, unrelated.

The micro-run also confirmed the evaluation semantics: at the final epoch the current and target paths give byte-identical numbers (CE 2.3029, acc 0.0700 both), and transition checkpoints correctly report the last trained config alongside the next scheduled one (e.g. after epoch 12: last `r=24, σ=0.450`, next `r=32, σ=0.500`).

## Evaluation as instrumented

Every snapshot records **current path** (primary — resolution and σ of the most recently completed update, stored explicitly) and **target path** (diagnostic — original 32×32, filters bypassed), each with CE and accuracy on the fixed training probe and the official test set, plus `next_resolution`/`next_sigma`. Evaluation runs in `eval()` and restores the previous mode; BN statistics are never updated or recalibrated. Cadence is the campaign's every-2-epochs with 6, 12 and 21 added explicitly; checkpoints after epochs 6, 12, 21 and 30. Cumulative evaluation overhead is timed separately from training so I can report the split rather than estimate it.

The timing probe measures six scheduled configurations on the actual T4s, bounded to \~2 minutes, then frees everything; projections for both runs are computed from it and labelled as estimates. I'll report measured wall time against them, and I'm not treating the \~24% convolution-work reduction as a promised wall-time saving.

I'll come back with the results, paired differences, curves and cost breakdown when the job lands. One seed, no best-epoch selection.

All four hashes matched — the arms are verified paired. Now the results.

Both arms completed: [`progres-r20bn-20260909-075846`](https://www.kaggle.com/code/maxnicaise/progres-r20bn-20260909-075846).

## Pairing — verified, not assumed

All four sha256 digests reproduced the campaign's exactly: `subset`, `train_probe`, `perm_seed0` (30×50,000) and `init_seed0` (all 116 tensors). The job would have aborted before training otherwise. Both new arms are genuinely paired with `plain_seed0` and `plateau_seed0`.

## Final results — epoch 30, all four arms at the target configuration

| arm test acc test CE probe acc probe CE training time  |            |            |        |        |           |
| ------------------------------------------------------ | ---------- | ---------- | ------ | ------ | --------- |
| plain (campaign)                                       | 0.7510     | 0.7592     | 0.9440 | 0.2374 | 466 s     |
| Gaussian plateau (campaign)                            | 0.7827     | 0.6347     | 0.8820 | 0.4029 | 689 s     |
| **progressive resolution**                             | **0.7957** | **0.6119** | 0.8960 | 0.3505 | **436 s** |
| **progressive res. + Gaussian**                        | **0.7993** | **0.5881** | 0.8920 | 0.3457 | 651 s     |

**Paired differences:**

| comparison accuracy CE            |              |        |
| --------------------------------- | ------------ | ------ |
| progressive resolution − plain    | **+4.47 pp** | −0.147 |
| combined − Gaussian plateau       | **+1.66 pp** | −0.047 |
| combined − progressive resolution | **+0.36 pp** | −0.024 |

Progressive resolution alone (+4.47 pp) is larger than the Gaussian plateau's own gain over plain (+3.17 pp), and it beats the plateau arm outright by 1.30 pp. Adding the internal Gaussian on top of progressive resolution buys **+0.36 pp** — against the \~±0.1 pp single-seed noise floor measured yesterday, that is small and I would not call it established at one seed.

## Cost — the 24% did not materialize

| configuration s/update peak  |        |         |
| ---------------------------- | ------ | ------- |
| r=16, no filter              | 0.0328 | 35 MiB  |
| r=24, no filter              | 0.0340 | 51 MiB  |
| r=32, no filter              | 0.0324 | 72 MiB  |
| r=16, σ=0.500                | 0.0615 | 65 MiB  |
| r=24, σ=0.525                | 0.0593 | 116 MiB |
| r=32, σ=0.500                | 0.0558 | 182 MiB |

**Lower resolution is not faster on a T4 at this scale** — r=16 measured marginally *slower* per update than r=32 (0.0328 vs 0.0324 s), and r=24 slower than both. At 16×16 with 16 channels the kernels are far too small to saturate the GPU, so the work reduction is entirely hidden by launch and memory-traffic overhead. Memory does scale as expected (2× from r=16 to r=32).

Measured wall time, all excluding evaluation: progressive resolution **436 s** vs plain's 466 s (−6.4%); combined **651 s** vs plateau's 689 s (−5.5%). Both are real but small, and smaller than the predicted convolution-work saving. Evaluation overhead was 16 s and 22 s respectively (3.6% and 3.3% of total); totals 452 s and 674 s; peak memory 512 and 583 MiB.

Predicted vs measured training time: 385 s predicted / 436 s measured for progressive resolution, 594 s / 651 s for the combined arm — the probe underestimates by \~12% because it excludes data indexing and checkpoint writes. Probe cost was 11.5 s, inside the \~2-minute budget.

## Trajectory and the target-path diagnostic

| epoch progres cur / tgt combined cur / tgt plain plateau  |                 |                 |        |        |
| --------------------------------------------------------- | --------------- | --------------- | ------ | ------ |
| 6 (r=16)                                                  | 0.5666 / 0.3504 | 0.5979 / 0.2677 | 0.6480 | 0.6115 |
| 12 (r=24)                                                 | 0.6924 / 0.6621 | 0.7114 / 0.3257 | 0.7194 | 0.7251 |
| 21 (r=32)                                                 | 0.7859 / 0.7859 | 0.7910 / 0.7885 | 0.7507 | 0.7733 |
| 30 (r=32)                                                 | 0.7957 / 0.7957 | 0.7993 / 0.7993 | 0.7510 | 0.7827 |

Both new arms trail the controls through the low-resolution phase and overtake them after the move to 32×32, finishing clearly ahead. The target-path gaps at epochs 6 and 12 are large (−21 pp and −38 pp) exactly as expected — that path evaluates 32×32 inputs on a model whose BN statistics were accumulated at 16×16 or 24×24 with filters active. It measures a premature configuration change, not predictor quality. From epoch 21 the two paths converge, and at epoch 30 they are identical to the digit, confirming the final nine epochs run at the exact target configuration.

One thing worth noting: both new arms have *higher* final probe CE than plain (0.35 vs 0.24) while generalizing better — the same less-fitting-but-better-generalizing pattern the Gaussian arms show.

## Caveats

One seed. The +0.36 pp for adding Gaussian on top of progressive resolution is within a few multiples of single-seed noise. Progressive resolution changes the input distribution and the effective schedule together, so this does not isolate *why* it helps. Protocol was frozen before results were examined; epoch-30 checkpoint used throughout, no best-epoch selection, no schedule tuning. Test-set monitoring remains exploratory.

Two implementation bugs were caught locally before launch, both described in my previous message — the uint8-resize error and a `describe()` ordering regression of mine. Test suite: 170 passed, 1 pre-existing unrelated failure.

## Artifacts

- Runs, metrics, resumable checkpoints (epochs 6/12/21/30 + final), config, environment: `results/kaggle_outputs/progres-r20bn-20260909-075846/progres_r20bn_20260909-075919/`
- Pairing proof: `pairing_verification.json` in that directory
- Timing probe: `t4_timing_probe.json` in the job root
- Operator/spatial checks: [results/progressive\_resolution\_verification.json](https://claude.ai/epitaxy/results/progressive_resolution_verification.json)
- Figure: [results/progressive\_resolution.png](https://claude.ai/epitaxy/results/progressive_resolution.png)
- Scripts: [job\_progressive\_resolution.py](https://claude.ai/epitaxy/scripts/job_progressive_resolution.py), [verify\_progressive\_resolution.py](https://claude.ai/epitaxy/scripts/verify_progressive_resolution.py), [plot\_progressive\_resolution.py](https://claude.ai/epitaxy/scripts/plot_progressive_resolution.py)

**Assessment:** progressive resolution improves final accuracy substantially (+4.47 pp paired) while reducing measured cost slightly (−6.4% wall time) — so the accuracy claim is strong at one seed and the cost claim is real but far below the \~24% work reduction, because the smaller convolutions don't saturate a T4. Whether the extra +0.36 pp from combining it with the internal Gaussian is real is the open question, and it's the one additional seeds would settle first.