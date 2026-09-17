# Operational amendments (no scientific setting changed)

| UTC | amendment | effect on results |
|---|---|---|
| 2026-09-17 16:50 | `Trainer.resume` loads checkpoints on the CPU (the GPU pilot showed that restoring the RNG state from a GPU-mapped checkpoint raised `TypeError: RNG state must be a torch.ByteTensor`). Model and optimizer tensors are still copied to the parameter device by `load_state_dict` | none: no run had been resumed |
| 2026-09-17 17:22 | Configuration digest excludes environment fields (`data.root`, `assets.dir`, `run.out_dir`, `run.device`); `overnight/frozen_hashes.json` recomputed with that rule. The first training launch (tag `t1`, 16:53–17:00) was rejected cell by cell by the guard, because Kaggle paths differ from the local defaults. No update was taken and no test image was evaluated | none: every configuration value is unchanged (`protocol/configs/*.json` identical); `t1` outputs kept as a failed attempt |
| 2026-09-17 17:22 | The GPU resume self-check now also repeats an uninterrupted 60-update run, so that a resumed-vs-straight difference can be compared with run-to-run GPU nondeterminism (`t1` reported a max absolute state difference of 161.6 for resumed vs straight, with no repeat for comparison; on CPU the same comparison with real CIFAR-10 is bitwise equal) | diagnostic only |
