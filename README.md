# cifar10-resnet

A ResNet-18 baseline on CIFAR-10 that runs unchanged locally and on a Kaggle GPU.

This is a base repo. It is meant to be forked to ask one question, so every
module is short enough to read in one sitting and there is nothing in it that the
baseline does not need. `layer-l2-homotopy/` is the same idea with a research
harness bolted on; if you want sharpness, Hessian traces or a stage system, go
there instead of growing this.

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
means the code never branches on whether logging is available. The key would
ride in a mounted dataset as `wandb_key.txt` and is never in the source; with no
such dataset attached, `init_run` sets `WANDB_MODE=offline` and the run logs
into `/kaggle/working`, which `make pull` brings back. Nothing is lost, and
nothing has to be configured to run.

`kernel-metadata.json` mounts `idrisselkhamlichi/cifar-10-python` -- the pickled
batches, uploaded once. `data.py` globs `/kaggle/input/**` for `data_batch_1`,
so the mount is found wherever Kaggle puts it and the 170 MB download never
happens again.

## The two homotopies

Both deform the network along a scalar and drive it during training. They are
independent axes, they share the schedule machinery, and each is off by default
-- a config that sets neither is byte-identical to the plain baseline.

```
residual     h_out = shortcut(x) + s * F(x)        s : 0 -> 1     s_* keys
activation   phi_alpha(x) = max(x, alpha * x)      alpha : 1 -> 0  a_* keys
```

`s = 1` and `alpha = 0` are both exactly the baseline ResNet. `s = 0` leaves a
shortcut-only network; `alpha = 1` leaves an **affine** one, whose optimum on
CIFAR-10 is a linear classifier at around 40%. That asymmetry matters when
reading a run: during the ramp, `test_acc` is the accuracy of whatever network
is currently running, and `test_acc_at_a0` is the ReLU ResNet those weights
define. Only the second is comparable to a baseline number.

```sh
make act-quick                  # CPU plumbing check: does alpha move at all
make act ACT=act_linear         # one arm of the pilot
python explore.py runs/latest --only alpha,loss_vs_alpha,branch
```

The arms are `baseline40` (the matched control), `act_linear`, `act_jump`,
`act_staircase`, `act_stagewise` and `act_half`; `ACTIVATION_PROTOCOL.md` says
what each one is for and what would count as a result. **`act_jump` is the arm
that can kill the hypothesis**: it holds alpha at 1 and then drops it to 0 with
no path in between, so if it matches `act_linear`, the near-linear start was an
initialisation and the continuous path bought nothing.

`alpha.png` is the figure to check first on any activation run. `linear_gap`
must rise as alpha falls. If it stays flat, BatchNorm shifted the
pre-activations positive, `phi_alpha` became the identity whatever alpha said,
and the run is a baseline wearing a costume -- no accuracy from it means
anything.

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
  data.py                   CIFAR-10, resident on the GPU as uint8
  model.py                  Activation, BasicBlock, ResNet, resnet18/34
  homotopy.py               ResidualGate, the s/alpha schedules, continuation phases
  activation.py             ActivationGate, the sites and the scopes that group them
  landscape.py              loss surfaces, curvature, diagnostics, checkpoints
  train.py                  build_optimizer, lr_at, train_once
  metrics.py                evaluate
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

Three seeds minimum, with the spread reported. The test set is opened once.
Models never read config, log, or save. `print`, not `logging`. Type hints only
at a public tensor-API boundary. Comments name the failure mode they prevent.
