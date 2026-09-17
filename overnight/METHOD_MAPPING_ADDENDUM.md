# Method mapping: 160-epoch follow-up addendum

This extends `comparison/METHOD_MAPPING.md` (branch `comparison-cbs-sdpoint` @ `6b28489`). Every
source statement there still holds. The ports are unchanged: CBS from `pairlab/CBS@5f62e7da5b290e8f62f408c6f89146f3f361cc2e`
and SDPoint from `xternalz/SDPoint@0013c5dafe80780ea749198ebc42824c3ed41e6c`, both with verbatim copies in
`comparison/reference_code/`. The 7 arms, their sites, operators and inference instances are unchanged.
The corrected SDPoint defects also stay: BasicBlocks are eligible, and a drawn point 0 is kept rather than
lost through `None and randint or blockID`. So do the paper's CIFAR block mapping (9 post-addition points
+ none) and the `int(round(n*ratio))` rounding. Only horizons, recipes and evaluation are new.

## 1. Sources read for this study

| source | where | used for |
|---|---|---|
| He et al., CVPR 2016, Sec. 4.2 | CIFAR-10 recipe | SGD 0.1, momentum 0.9, wd 1e-4, batch 128, lr ÷10 at 32k/48k, stop at 64k; 4-px padding, 32×32 random crop or its horizontal flip; single-view clean test |
| CBS arXiv:2003.01367v5 Sec. 3–4, App. C; `resnet.py` `get_new_kernels` | epoch schedule | σ ← 0.9σ every 5 epochs from σ = 1, no stopping rule in code |
| SDPoint arXiv:1801.09335 Sec. 4–5.1; `main.py` `validate()` | BN for an instance | `model.train()` pass over the **training loader** (its augmentation), `no_grad`, default momentum 0.1, **no reset**, then `model.eval()` |

## 2. Adaptations in this study (decided before any new result)

| item | decision | reason / source |
|---|---|---|
| SGD recipe | lr 0.1 ÷10 at the starts of epochs 80 and 120 (updates 31,280 / 46,920) of 160 (62,560 updates ≈ 64k); coupled wd 1e-4 on **all** learned parameters; no warm-up; physical batch 128 | He et al. Sec. 4.2 mapped to our 391-update epochs. Differences from He et al.: 50k training images (no 45k/5k split), per-channel standardisation instead of per-pixel mean subtraction, our pinned Kaiming-normal initialisation, weight decay also on BN affine parameters. **Not an exact reproduction** |
| augmentation | zero pad 4 → uniform 32×32 crop → flip p = ½, uint8, before normalisation; counter-based draws per (seed, epoch, example) | He et al. Sec. 4.2; pairing across arms |
| AdamW long | historical recipe unchanged except that the cosine spans 62,560 updates | task specification |
| R / G / RG | same fractions of the budget: r transitions at 0.2U and 0.4U (epochs 32, 64); g plateaus of 0.1U (16 epochs), bypass from 0.7U (epoch 112); RG σ = g·r/32 at all 19 sites | preserves the selected 30-epoch fractions (6/30, 12/30, 3/30, 21/30) |
| CBS published | σ(e) = 0.9^⌊e/5⌋ on actual epochs up to e = 159 (σ = 0.9³¹ ≈ 0.038); no stretching, no forced bypass; native inference keeps the final filter | author code has no horizon-dependent rule. At σ ≈ 0.038 the float32 kernel is exactly the centre tap 1 and 0 elsewhere (checked: finite, sums to 1). So the final published filter is numerically an identity at 160 epochs, unlike at 30 epochs (σ = 0.59) |
| CBS budget-matched | u_off = ⌊0.7U⌋ = 43,792 by exact rational arithmetic, K = 22, σ = 0.9^⌊22u/u_off⌋, bypass after | previous formula; integer boundaries (identical plateaus for U = 11,730, tested) |
| SDPoint | unchanged per-update draw for all 160 epochs, point 0 at inference | Sec. 4, Alg. 1 |
| BN P3 | author `validate()` adapted: seed-specific frozen order instead of a shuffling loader; batch = our physical training batch instead of `args.batch_size`; crop/flip only where training used it (ImageNet RandomResizedCrop is not transferred); one pass | differs from P1 in reset, momentum, order, batch size and augmentation at once: no single-factor attribution |

## 3. Unequal inference computation

Published-schedule CBS keeps 19 extra 3×3 depthwise convolutions at inference. At 160 epochs these are numerically identity kernels, but they still cost compute unless removed. R, G, RG, budget-matched CBS and SDPoint (point 0) all infer with the plain network's computation.
