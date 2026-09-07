# cifar10-resnet

A ResNet-18 baseline on CIFAR-10 that runs unchanged locally and on a Kaggle GPU.

This is a base repo. It is meant to be forked to ask one question, so every
module is short enough to read in one sitting and there is nothing in it that the
baseline does not need. `layer-l2-homotopy/` is the same idea with a research
harness bolted on; if you want sharpness, Hessian traces or a stage system, go
there instead of growing this.

**This branch (`adaptative-L2`) asks one question**, and it is the exception to
the paragraph above: see [Adaptive anchored L2](#adaptive-anchored-l2). The
`baseline` config is untouched and still produces the number below.

## The number

The `baseline` config -- ResNet-18, SGD with nesterov momentum, cosine schedule
with a 5-epoch linear warmup, random crop + horizontal flip, 100 epochs, trained
on all 50k and evaluated on the 10k test set -- reaches **≈94-95% test accuracy**
on a T4, reported as a mean and a spread over three seeds.

Two failure signatures worth recognising in the report:

| You see | It means |
| --- | --- |
| ≈87-89% | augmentation is off (`augment: false`) |
| ≈10% | the LR schedule or the per-step multiplier is broken |

`print_report` also prints the *resolution* of the baseline: the seed-to-seed sem,
the binomial floor on 10k test images, and the smallest difference that could
mean anything. A change that beats the mean by less than that has shown nothing.

## Running it

```sh
uv venv --python 3.12 && uv sync
source .venv/bin/activate     # or: make quick PYTHON=.venv/bin/python

make quick        # CPU smoke test: 2 epochs, 4k images, 1 seed -- a few minutes
make baseline     # the reference recipe, one seed
make run          # the reference recipe, all three seeds
python main.py --help
```

If you already have CIFAR-10 extracted somewhere, point `CIFAR_DATA` at the
directory holding `cifar-10-batches-py` and skip the 170 MB download.

`make quick` sets `CIFAR_ALLOW_CPU=1`, because `pick_device` deliberately refuses
to fall back to CPU inside a Kaggle session -- spending the GPU quota on CPU
arithmetic and then hitting the wall-clock limit is worse than stopping.

## Where results go

Every run gets **its own directory** and never writes over an earlier one:

```
runs/
├── 20260907-113045_baseline_resnet18/
│   ├── invocation.json     how the run was launched
│   ├── results.json        the resolved config, per-seed summaries, and stats
│   ├── history.jsonl       every epoch of every seed
│   └── history.csv         the same rows, for a spreadsheet
├── 20260907-141122_baseline_resnet18/
└── latest -> 20260907-141122_baseline_resnet18
```

The root is `/kaggle/working/runs` in a kernel and `./runs` locally. The
directory is created *before* training and `invocation.json` written
immediately, so a run that crashes still leaves behind exactly what was launched.

`invocation.json` records the raw `argv`, a paste-able `command`, every `CIFAR_*`
environment variable, the cwd, the torch version and the device name, plus the
fully resolved config. The resolved config alone cannot tell you whether a value
came from a flag, an env var or the YAML -- and reconstructing that two weeks
later from shell history is exactly the thing that does not work.

```sh
make runs                        # one line per run: config, seeds, accuracy
cat runs/latest/invocation.json  # what the last run actually ran
```

## Plots

`analyze.py` takes the folder the results are in. Point it at one run
directory, or at `runs/` to compare them all:

```sh
make plot                        # the last run  (DIR=runs/latest)
make plot DIR=runs               # every run, overlaid
python analyze.py runs/20260907-113045_baseline_resnet18
python analyze.py out/runs --out /tmp/figs      # e.g. after `make pull`
```

It first prints a table (test accuracy +-2 sem, train accuracy, seed count per
run), then writes into `<folder>/figures/`:

- **`curves.png`** -- four panels: cross-entropy, accuracy, the learning-rate
  schedule, and the generalisation gap. Train, val and test on the same axes.
- **`per_class.png`** -- per-class accuracy worst-first, with each seed as a dot
  so a class that is unstable across seeds does not hide behind its mean.
  Written only when the run had `per_class: true`.
- **`compare.png`** -- in multi-run mode: every run's test curve overlaid, plus
  the final accuracies at +-2 sem. Bars that overlap are not a result.

Curves are the **mean across seeds with a min..max band**, never one seed
dressed up as the result -- a CIFAR-10 baseline moves by half a point between
seeds, which is most of the difference anyone is trying to see. Note the gap
panel uses the running batchwise train accuracy, measured under augmentation, so
it is a lower bound; `results.json` has the clean number.

Like `build.py`, `analyze.py` sits at the top level and is **not** in `src/` --
so matplotlib never has to exist inside the kernel. The kernel's job is to
produce numbers; reading them is a local activity.

`make clean` deliberately leaves `runs/` alone; it only removes `dist/` and
`out/`. Delete old runs yourself when you mean to.

W&B is used when a key is mounted and skipped otherwise; `wandb_setup.NullRun`
means the code never branches on whether logging is available. The key rides in
the `aubincharley/wandb-secret` dataset as `wandb_key.txt` and is never in the
source.

## Configuration

Four layers. Later wins:

```
CONFIG (src/cifarbase/config.py)
  -> src/cifarbase/configs/<name>.yaml
    -> _QUICK, if quick
      -> CIFAR_<KEY> env vars
        -> command-line flags
```

```sh
python main.py --config quick                 # pick a recipe
CIFAR_EPOCHS=3 python main.py --config quick  # env beats the YAML
python main.py --epochs 20 --no-augment       # a flag beats everything
```

`build_parser` generates one flag per `CONFIG` key and types it from that key's
default, so adding a config key cannot forget to add the flag. A YAML that sets a
key `CONFIG` does not have is a hard error -- a typo'd `weigth_decay` silently
doing nothing produces a run that looks fine and answers a question you did not
ask.

`configs/baseline.yaml` is the reference, not `CONFIG`. The accuracy above
describes that file exactly, so override with a flag for an experiment and edit
it only when you mean to move the reference itself.

**Only the first three layers exist on Kaggle.** A `script` kernel is handed no
argv and no environment, so the recipe a push runs is whatever `KAGGLE_CONFIG` in
`main.py` names. Edit it and re-push.

## Adaptive anchored L2

The question: instead of decaying weights toward zero, anchor them to their
initialisation, and choose the anchor's strength by measuring how much feature
learning has actually happened rather than by picking a number.

The penalty is applied **per parameter group**, normalised by the group's anchor
norm, and **decoupled** -- straight to the update, after `optimizer.step()`, never
through the loss:

```
F(w) = L(w) + sum_g (lambda_g/2) ||w_g - w0_g||^2 / ||w0_g||^2

w_g <- w_g - eta_k * grad(L)  -  eta_k * lambda_g * (w_g - w0_g) / ||w0_g||^2
```

The `1/||w0_g||^2` is what makes `lambda_g` dimensionless, and therefore
comparable across depth. Without it the same numeric lambda is a hundred times
stronger on the stem than on stage 4 purely because the two groups hold different
numbers of parameters at different scales, and any allocation learned on top would
mostly be reading that back.

Every `probe_every` steps a fixed, class-balanced probe batch is pushed through
the network and the empirical NTK is sketched on it with `R` forward-mode JVPs.
The controller reads the relative drift of that kernel from its value at the
reference point,

```
d_t = ||K_t - K_0||_F / ||K_0||_F
```

and moves `lambda` multiplicatively, in log-space, to keep `d_t` on a schedule
`d*(t) = D_max * t/T`:

```
u <- u + clip(beta * (d_ema - d*(t))),    lambda = clip(exp(u))
```

Drifting faster than the schedule tightens the anchor; slower loosens it. So the
homotopy parameter is no longer an arbitrary knob: it is *how much feature
learning has been spent*, which is measured every hundred steps and plotted.

### Per group: two loops on two timescales

`u` is the **level**. On top of it sits an **allocation** `a_g`, one offset per
group, so that

```
log lambda_g = u + a_g,      sum_g a_g = 0
```

The constraint is what keeps the two separable: the allocation can only ever
redistribute the anchor between groups, never strengthen or weaken it overall.
That remains the level's job, and the level's transfer function is therefore
unchanged by anything the allocation does.

The groups (`anchor.py`) are set by `anchor_grouping`, coarse to fine:

| `anchor_grouping` | groups |
| --- | --- |
| `trunk` | `trunk` (stem + all four stages), `head`, `bn_affine` |
| `coarse` | `early` (stem, stages 1-2), `late` (stages 3-4), `head`, `bn_affine` |
| `stage` (default) | `stem`, `stage1..stage4`, `head`, `bn_affine` |

Biases are never anchored. Per tensor is deliberately not on the menu: 62 groups
of a plant this coupled is not a control problem. Which of the three is
admissible is decided by the coupling gate below, not by preference -- coarsening
removes a coupling, where loosening the gate only hides one.

The allocation is driven by **tension**, not by per-group drift. For group `g`,

```
tau_g = ||grad_g L|| * ||w0_g||^2 / (lambda_g * ||w_g - w0_g|| + eps)
```

the ratio of the gradient force pulling the group away from `w0` to the anchor
force holding it there. `tau_g > 1` means the group is anchor-limited: the data
wants it to move and the anchor is what is stopping it, so it should get more of
the budget. The update is a log-space integrator on the deviation of `log tau_g`
from its mean across groups, with a shrinkage term toward uniform:

```
a_g <- a_g - beta_a (log tau_g - mean_m log tau_m) - gamma a_g
a_g <- a_g - mean_m a_m                       # re-centre: sum a_g = 0
a_g <- clip(a_g, -log 10, +log 10)
```

with `beta_a = beta/5` and `gamma = 0.01`, fired **once every ten probes** where
the level fires every probe. That two-timescale separation is the entire
stability argument: the fast loop is entitled to treat the allocation as constant
while it converges, and the slow loop to treat the level as converged when it
moves. Run them at the same rate and neither assumption holds -- the two
integrators chase each other, and because both live in `log lambda` the result is
a lambda trace that looks like a controller working.

**Why tension and not `d_g`.** The per-group drifts are measured and logged
(`kernel.ntk_sketch_by_group` masks each frozen tangent to one group, costing
`L*R` JVPs on top of the global sketch's `R`). But they make a poor regressand for a diagonal
controller: `d_g` responds to `lambda_m` for every `m`, and steering each `d_g` by
its own `lambda_g` is exactly the diagonal-control-on-a-non-diagonal-plant failure
that gate H7 exists to detect. Tension is local by construction -- both of its
terms are properties of group `g` alone.

`anchor_alloc: off` holds `a_g = 0` and reproduces the single-lambda behaviour
exactly. That arm is the control, and `make anchor` is how it is run.

### The protocol, in order

Each stage gates the next. Every failure they catch produces a run that looks
entirely normal, which is why they are cheap and come first.

```
make selfcheck     # stage 0: correctness gates. No dataset needed, no GPU needed.
make pilot         # stage 1: fixed-lambda sweep. THE go/no-go for the design.
make pilot-check   # stage 2: monotone separation, then beta and D_max
make anchor        # stage 3: one shared lambda, 5 seeds -- the control
make alloc         # stage 3: lambda per group, 5 seeds -- the method
```

`alloc` without `anchor` answers nothing: a win for the per-group arm has to be a
win over the single-lambda loop, not over the unanchored baseline, or it is just
the anchor working.

`make pilot` trains at fixed `lambda` over four decades and asks whether the
`d(t)` trajectories separate monotonically in `lambda`. If they do not, `lambda`
has no authority over drift, no controller can work at any `beta`, and the design
is dead -- that is a real negative result about the mechanism, and it costs about
an hour of T4 time to find out instead of a week.

`anchor.yaml` ships with `anchor_beta: 0.0` and `anchor_dmax: 0.0`, which makes it
**refuse to run**. Both are outputs of the pilot. A guessed `D_max` produces a
controller tracking a schedule nobody chose.

Stage 3 does not fit in one Kaggle session, so a run can be stopped and continued:

```
python main.py --config anchor --stop-after-epoch 50    # writes ckpt_s<seed>.pt
python main.py --config anchor --resume runs/<dir>/ckpt_s0.pt
```

`--stop-after-epoch` and **not** a lower `--epochs`. `total_steps`, and therefore
the entire cosine curve, is derived from `epochs`; declaring fewer of them trains
the first half on a different schedule and the continuation is then a different
run. `checkpoint.load` refuses a resume whose `epochs`, `batch_size`, `arch`,
`width`, `schedule`, `warmup_epochs`, `lr` or `train_subset` has moved, because
nothing downstream could see it -- the loss keeps falling and the `lr` column
still looks like a valid schedule, just not the one the checkpoint came from.

### The arms

`anchor_mode` selects which one. A win for the closed loop only means something
against all of these, because each isolates a different alternative explanation.

| mode | lambda(t) | what it rules out |
| --- | --- | --- |
| `off` | 0 | nothing; this is the baseline, and it still logs drift |
| `const` | fixed | "any anchor would have done" |
| `adaptive` | the controller | -- (the method) |
| `replay` | a recorded trace, verbatim | "the feedback matters" vs "that schedule shape matters" |
| `exp` | `lambda_max * exp(-c t)`, `c` matched on `d_T` | "any decreasing lambda would do" |
| `critical` | `lambda_max * 1[t < 0.2T]` | "only the early phase matters" |
| `off` + swept `weight_decay` | -- | "this is just tuned weight decay" |

`eta_eff,l = eta / ||w_l||^2` is logged on **every** arm, baseline included. The
arms end at different weight norms and therefore at different effective learning
rates, and without that trace on both sides a win cannot be told apart from an
accidentally better schedule. It cannot be recovered after the fact.

That means the `baseline` config is now instrumented too: it captures `w0`, runs
the probe, and logs the whole anchor stream with `lambda = 0`. It costs one extra
parameter copy (45 MB at ResNet-18) and about 1% of wall clock, and it does not
move the accuracy -- gate H2 checks bit-for-bit that the probe touches no
parameter and consumes no random draw, which is the only reason it is safe to
leave switched on in the arm everything else is measured against. Set
`--probe-every` very large to turn it off, at the cost of losing the free-drift
curve that `D_max` is a fraction of.

### Four things measured here that are worth knowing

**`d` is dominated by the kernel's SCALE, not its geometry -- check before
reading it as feature learning.** The three logged quantities satisfy

```
d^2 = scale^2 - 2 a scale + 1        scale = ||K_t||_F / ||K_0||_F
```

so `d` is bounded below by `scale - 1` no matter what the geometry does. On a
short instrumented run: `d = 28.9` with `a = 0.27`, which solves to
`scale = 29.2`. The kernel's norm grew twenty-nine-fold, and essentially all of
the drift signal was that growth. This is not a defect of the estimator -- BN
plus growing weight norms make the kernel scale climb monotonically -- but it
means a controller steering by `d` is largely regulating kernel norm.

So `scale_ntk` and `a_ntk` are logged on every probe, `print_report` says so
outright whenever `scale > 2`, and `probe_signal` accepts `alignment`, which
steers by `1 - a`: the same "how much has changed" orientation, rising from 0,
bounded in `[0, 2]`, with the scale divided out. The default stays `drift`
because that is what the protocol specifies; the point is that the choice is now
visible and reversible rather than implicit.

**Common random numbers are load-bearing.** The `R` tangents are drawn from a
fixed seed and regenerated identically at every probe, on every seed and every
arm. With fresh tangents each probe the estimator's own noise is about
`R^-1/2 ||K||`, roughly 35% at `R=8` -- far larger than the early drift the
controller is steering by, so the loop would spend its authority chasing sampling
noise. Frozen, that noise is common to both terms and cancels.

**`w0` is a degenerate point of the NTK, and `K_0` is not taken there.**
`zero_init_residual` sets the last BatchNorm gamma in each block to exactly zero,
so every residual branch outputs exactly 0 at `w0` and each block's second ReLU
sees pre-activations equal to its shortcut -- about **39%** of which land exactly
on the kink, where the derivative flips for an arbitrarily small perturbation.
The measured consequence: the NTK sketch jumps by **24%** for a `1e-8`
displacement, and does not shrink as the displacement does. A `K_0` taken at `w0`
is one that `d_t` leaves in a single step and can never approach again, which
makes a `d*` rising from 0 unreachable and parks `lambda` on its clip for the
whole run. So `anchor_reference_step` defaults to the end of the hold window.
`w0` itself is unaffected -- it is still step 0, and it is still the anchor. The
`K0` gate in `make selfcheck` measures all of this and fails a configuration that
would take `K_0` at `w0` unknowingly.

**`d` responds to the square root of the weight displacement, not to the
displacement.** Saturating the anchor and sweeping `lambda` over three decades
(gate H1) gives `||w - w0||` falling as `1/lambda` exactly -- 9.6x and 9.9x per
decade -- while `d` falls only ~3.8x, i.e. `d ~ ||w - w0||^0.59`. This is the
non-degenerate form of the same kink effect: the NTK of a ReLU network is
continuous in `w` but not differentiable in it, because a displacement `delta`
flips the sign of every pre-activation within `delta` of zero, the count of those
grows like `delta`, and each flip moves the Jacobian by an O(1) amount locally --
so `||K - K_0||` carries a `sqrt(delta)` term that dominates the smooth one.

Two consequences for reading a run. The controller's gain `g` is not a constant of
the problem, so `beta = 1/g` fitted at mid-training is a local linearisation and
the pilot has to report `g` where it is actually used. And a `d*(t)` rising
linearly is being asked to track a quantity that grows like the square root of a
displacement -- expect `d` to run ahead of the schedule early and for `lambda` to
be pushed up hard in response. That is the loop working, not failing, but it is
why the `d` vs `d*` panel is the first thing to look at.

### The gates

```
K0  the drift reference is well-posed     the NTK is continuous where K_0 is taken
H1  the pull reaches w0                   lambda swept over 3 decades: ||w-w0||
                                          falls as 1/lambda, and the exponent in
                                          d ~ ||w-w0||^p lands between sqrt and
                                          linear
H2  lambda=0 is inert                     bitwise identical to the plain loop;
                                          the probe consumes no global RNG draw
H3  the estimator is deterministic        two probes at the same w, bitwise equal d
H4  a probe does not move BatchNorm       running_mean and running_var unchanged
H5  the homotopy is the closed form       delta*(lambda) = Phi^T (K0 + n lambda I)^-1 r
                                          on a two-layer net. Measured: rel err
                                          7.9e-4 / 8.0e-5 / 8.0e-6 at lambda =
                                          1e3 / 1e4 / 1e5, log-log slope -0.999.
                                          The trend is load-bearing -- a flat
                                          curve is a failure even when the
                                          magnitudes look fine
H6  a resume continues, not restarts      2 epochs == 1 + resume + 1, bit for bit
                                          across the weights, w0, K_0, the probe
                                          indices, the controller state and the
                                          whole probe stream
H7  the per-group plant is diagonally     G_lm = -d(d_l)/d(log lambda_m) by
    dominant                              central differences, one group pushed
                                          at a time. Requires a positive diagonal
                                          (more anchor, less drift) and
                                          |G_ll| > sum_{m!=l} |G_lm| on every row
```

H1 also certifies the normalisation itself: it re-runs the tightest sweep point
with `anchor_normalize` on and `lambda_g = 0.99 ||w0_g||^2 / lr`, which is the
same realised pull, and requires the same residual. A divisor applied to the
wrong group, or to the live norm instead of `w0`'s, misses that immediately.

**H7 is the gate that decides how fine the grouping may be**, and it is the reason
the partition is seven groups rather than per tensor. A per-group controller is a
diagonal controller on a MIMO plant: it moves `lambda_m` and reads back `d_m` as
though the two were paired, when in fact every lambda changes the whole network's
function and therefore every block of the kernel. Diagonal control on a strongly
non-diagonal `G` oscillates -- slowly, and with a lambda trace that looks like a
controller working. If H7 fails, the remedies are coarsening the partition or
cutting `anchor_alloc_beta_frac` by another factor of five. Not loosening the
criterion, which is the one change that would make the failure invisible again.

The gates have already paid for themselves three times on this branch.

- **H1** caught the anchor silently skipping every BatchNorm `beta`. Its name is
  `...bn1.bias`, so testing the bias suffix before BatchNorm membership put it in
  the unanchored role -- and because `||w - w0||` was not measuring those
  parameters either, the distance looked fine while the parameters that set each
  block's output scale drifted freely.
- **H5** caught a solve reporting optimiser slop as linearisation error: fixed-step
  gradient descent stalled at `||grad|| ~ 5e-3` after 400k iterations, which reads
  exactly like a wrong closed form. It now uses L-BFGS and asserts that the
  displacement an unconverged solve could still move, `||grad||/lambda`, is a
  thousand times smaller than the gap being measured.
- **H6** caught that a checkpoint is only resumable into a run with the same
  `epochs`, because the lr schedule is derived from it -- the first version of the
  gate stopped early by lowering `epochs` and diverged by 5.4e-3 in the weights.
  That produced both `--stop-after-epoch` and the compatibility guard in
  `checkpoint.load`.
- **K0** was written because of a measurement, not a hunch, and it now guards the
  configuration it was written for.

The `lambda = 1e3 / 1e4 / 1e5` range in H5 and the `p ~ 0.5` exponent in H1 are
both measured rather than assumed. Where they differ from a first-principles
guess, the docstrings say which measurement moved them.

Pilot separation is not a unit gate -- it is stage 1, above.

## Pushing to a Kaggle GPU

```sh
make push       # rebuilds dist/main.py, then `kaggle kernels push -p .`
make status
make pull       # log, results.json and history.* into out/
```

A Kaggle kernel is a *single file*: the save-kernel API takes one `text` field
with no way to carry a second file. `build.py` therefore embeds the whole `src/`
tree as a base64 tar.gz that the generated header unpacks onto `sys.path` before
`main.py` runs. Consequences:

- **Never edit `dist/main.py`.** Edit `src/` or `main.py` and re-run `make build`.
  The file is generated and gitignored.
- `build.py` is byte-identical across all the Kaggle projects here, and its
  output is byte-deterministic (zeroed tar metadata, `mtime=0`, sorted file
  list), so an unchanged `src/` never produces a new kernel version. A second
  `python3 build.py` prints `identical to the last build`.
- **The YAMLs live inside the package**, at `src/cifarbase/configs/`, rather than
  in a top-level `configs/`. `build.py` bundles `src/` and nothing else, so this
  is what makes them ship -- and it keeps `build.py` unmodified.
- **Imports must be absolute.** A relative import breaks the bundle's flat
  `sys.path` insert.

### The accelerator gotcha

`enable_gpu: true` from a CLI push resolves to a **Tesla P100 (sm_60)**, and
recent torch builds ship sm_70 and up. Every kernel launch then fails with
`no kernel image is available for execution on the device`, a long way from the
cause. So:

1. `machine_shape: "NvidiaTeslaT4"` is set in `kernel-metadata.json`, and
2. after `make push`, set **Accelerator → GPU T4 x2** in the kernel's settings
   page and use **Save & Run All** from the UI. A push resets it.

`pick_device` probes the GPU with a real kernel launch and a forced sync rather
than trusting `torch.cuda.is_available()`, and prints the device's compute
capability against `torch.cuda.get_arch_list()` when they do not match.

### Where the data comes from on Kaggle

With nothing mounted, `data.py` downloads CIFAR-10 into `/kaggle/working`, which
is also the kernel's output -- so the 170 MB archive and its extracted batches
end up attached to the run and `make pull` will fetch them. To avoid that, mount
a dataset containing `cifar-10-batches-py` (add it to `dataset_sources` in
`kernel-metadata.json`); `_load_raw` finds it under `/kaggle/input` before it
considers downloading.

## Layout

```
build.py                    the bundler -- byte-identical across projects; never edit
main.py                     the entry point; KAGGLE_CONFIG lives here
analyze.py                  plots a results folder; local-only, never bundled
kernel-metadata.json        T4, internet on, wandb-secret mounted
src/cifarbase/
  config.py                 the four-layer resolution, and CONFIG itself
  configs/*.yaml            recipes; bundled to Kaggle because they are under src/
  data.py                   CIFAR-10, resident on the GPU as uint8; Split.take
                            and the class-balanced probe indices
  model.py                  BasicBlock, ResNet, resnet18/34
  train.py                  build_optimizer, lr_at, train_once
  metrics.py                evaluate
  anchor.py                 w0, the seven-group partition, the decoupled pull
  kernel.py                 the NTK sketch, its per-group decomposition, and the
                            drift estimator
  controller.py             lambda(t): the level's closed loop, the allocation's
                            slow loop, and the open-loop arms
  hessian.py                lambda_min(H) by Lanczos
  checkpoint.py             save/restore, including w0, K_0 and controller state
  selfcheck.py              the gates
  report.py                 summarise, print_report, run_dir, dump, dump_history
  wandb_setup.py            W&B, or a NullRun that behaves like it
  utils/device.py           a device probe that checks the device actually works
  utils/seeding.py          seed_everything
```

### Why there is no DataLoader

Kaggle gives about 4 vCPU. A torchvision `DataLoader` doing PIL crops on that
many cores would starve a T4 long before the GPU became the limit. So the whole
dataset sits on the device as uint8 (~240 MB), the train split is pre-padded to
40x40 once so the random crop is a gather rather than a pad-per-batch, and the
"loader" is an index shuffle. Normalisation and augmentation are a handful of GPU
ops per batch.

Augmentation is **per sample**. One crop offset shared across a batch is a much
weaker augmentation -- the network still sees the whole batch in identical
registration -- and it is a silent bug, costing about a point of test accuracy
without ever raising an error.

### Conventions

Three seeds minimum, with the spread reported -- five on the anchored arms,
because the effect being looked for is about 0.3%, the same size as the
seed-to-seed spread. The test set is opened once.
Models never read config, log, or save. `print`, not `logging`. Type hints only
at a public tensor-API boundary. Comments name the failure mode they prevent.
