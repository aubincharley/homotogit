# cifar10-resnet

A ResNet on CIFAR-10 that runs unchanged locally and on a Kaggle GPU, and the
curriculum-learning study built on top of it.

This is a base repo. It is meant to be forked to ask one question, so every
module is short enough to read in one sitting and there is nothing in it that the
baseline does not need. `layer-l2-homotopy/` is the same idea with a research
harness bolted on; if you want sharpness, Hessian traces or a stage system, go
there instead of growing this.

## The number

The `baseline` config is SGD with nesterov momentum, a cosine schedule with a
5-epoch linear warmup, random crop + horizontal flip, 100 epochs, trained on all
50k and evaluated on the 10k test set, reported as a mean and a spread over three
seeds.

It now names **ResNet-20** (He et al. section 4.2: three stages, width 16, 0.27M
parameters), not ResNet-18. The study runs 36 arms and 36 ResNet-18s do not fit
in a session; a ResNet-20 is roughly 13x fewer FLOPs per image. The ≈94-95% this
file used to quote described ResNet-18 at width 64, which is still one flag away
(`--arch resnet18 --width 64`). **The ResNet-20 reference has not been
re-measured here** -- the published figure for this recipe is ≈91.5-92.5%, and
what this repo has actually run since the switch is the curriculum study, not
100 clean epochs.

Two failure signatures worth recognising in the report:

| You see | It means |
| --- | --- |
| 3-4 points under the recipe's figure | augmentation is off (`augment: false`) |
| ≈10% | the LR schedule or the per-step multiplier is broken |

`print_report` also prints the *resolution* of the baseline: the seed-to-seed sem,
the binomial floor on 10k test images, and the smallest difference that could
mean anything. A change that beats the mean by less than that has shown nothing.

## Running it

```sh
uv venv --python 3.12 && uv sync
source .venv/bin/activate     # or: make quick PYTHON=.venv/bin/python

make smoke        # the WHOLE study grid on CPU: 16 arms, 4k images, 2-3 epochs
make quick        # CPU plumbing check of the single-arm path
make baseline     # one arm, one seed
make study        # the campaign: 4 regimes x 4 arms x 1 seed (needs a GPU)
python main.py --help
```

`python main.py` with no arguments runs the study named by `KAGGLE_STUDY`, which
is what a Kaggle script kernel gets. `--study ""` opts back into the single-arm
path.

If you already have CIFAR-10 extracted somewhere, point `CIFAR_DATA` at the
directory holding `cifar-10-batches-py` and skip the 170 MB download.

`make smoke` and `make quick` set `CIFAR_ALLOW_CPU=1`, because `pick_device` deliberately refuses
to fall back to CPU inside a Kaggle session -- spending the GPU quota on CPU
arithmetic and then hitting the wall-clock limit is worse than stopping.

## Where results go

Every run gets **its own directory** and never writes over an earlier one:

```
runs/
├── 20260907-113045_baseline_resnet20/
│   ├── invocation.json     how the run was launched
│   ├── results.json        the resolved config, per-seed summaries, and stats
│   ├── history.jsonl       every epoch of every seed
│   └── history.csv         the same rows, for a spreadsheet
├── 20260909-092558_pacing_vs_order/       a STUDY: one directory per arm
│   ├── study.json          the grid, the timings, what the budget cut
│   ├── teacher_clean_short.pt             the cross-fit scores, once per regime
│   ├── clean_short__baseline/             the four files above, plus:
│   │   └── arm.json        {regime, arm, curriculum, scoring, pacing, ...}
│   ├── clean_short__curriculum/
│   ├── clean_short__random/
│   └── clean_short__anti/
└── latest -> 20260909-092558_pacing_vs_order
```

A study's arms are named `<regime>__<arm>` with no timestamp, because they are
read as a grid rather than as a history. `arm.json` is what makes `analyze.py`
group them: an arm is a fact about the experiment, not something to parse back
out of a directory name.

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

When the folder is a **study** -- its subdirectories carry `arm.json` -- the
comparison becomes arm-aware and writes three things instead:

- **`study.png`** -- one row per regime: the four arms' test curves, their
  difference against that regime's baseline *paired by seed*, and the
  `lambda(t)` actually applied.
- **`effects.png`** -- the decomposition, at ±2 sem of the paired difference.
  This is the figure the report is built on.
- **`report_tables.md`** -- the same numbers as markdown, to paste into a write-up.

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
means the code never branches on whether logging is available. No key dataset is
mounted, so runs log offline; to log online, create a private dataset holding a
one-line `wandb_key.txt`, add it to `dataset_sources`, and `init_run` will find
it. The key is never in the source.

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
argv and no environment, so what a push runs is whatever `KAGGLE_STUDY` in
`main.py` names -- or, when that is `""`, the single recipe `KAGGLE_CONFIG`
names. Edit and re-push.

A **study** is a fifth thing, and not a config: `src/cifarbase/studies/*.yaml`
holds a `base` block, a list of `regimes` and a list of `arms`, and one config is
resolved per (regime, arm) as `base < regime < arm`, with the flags you actually
typed re-applied last. That last part is what makes `--seeds 0 --epochs 2` able
to shrink a grid to something that finishes in a minute. Every key in every block
has to be a real `CONFIG` key, for the same reason a config YAML's keys do: a
typo in a grid silently makes two arms identical, and the difference between them
then reads as "no effect".

## Curriculum learning: what the study actually asks

`results/RESULTS.md` is the write-up. This is what the code does.

A curriculum here is a **continuation method**: a family of objectives `L_λ`,
where `L_λ` is the training objective restricted to the λ easiest examples, `L_0`
is the easy subproblem and `L_1` is the real one, walked from one to the other
along λ(t). Bengio et al. (ICML 2009) take the same two endpoints in a single
jump -- which is `pacing: step` here; walking them in increments is what a
continuation method actually does.

Following Wu, Dyer & Neyshabur (ICLR 2021), that splits into two independent
choices, and so does the config:

| | keys | what it decides |
| --- | --- | --- |
| **scoring** | `curriculum`, `scoring` | which examples count as easy, and in which direction they are fed |
| **pacing** | `pacing`, `ramp_epochs`, `easy_frac` | how fast the pool opens up |

The split matters because **an epoch is a fixed number of SGD steps whatever the
pool holds**. A pool of size k is therefore drawn `n/k` times per epoch: at
λ = 0.5 every visible example is seen twice while the baseline sees each of its
once. So a curriculum arm differs from its baseline in *two* ways at once, and an
accuracy gap alone cannot say which one moved it. Hence four arms per regime:

| arm | `curriculum` | pacing | what it isolates |
| --- | --- | --- | --- |
| `baseline` | `none` | none | the reference |
| `curriculum` | `easy_first` | the ramp | the effect as usually reported |
| `random` | `random_order` | **the same ramp** | pacing with no difficulty information |
| `anti` | `hard_first` | the same ramp | the direction of the ordering |

```
pacing    = random     - baseline
ordering  = curriculum - random
direction = curriculum - anti
total     = curriculum - baseline  =  pacing + ordering
```

all paired by seed -- the arms share a seed, so they share their initialisation
and their corrupted labels, and that much does not vary between them. With more
than one seed the paired difference also gets an interval, which is what turns a
direction into a result; `pacing_vs_order` currently runs one seed, so it gives
directions only, and the report says so in the same table.

```sh
make study                       # the whole grid, one process, one push
python main.py --study smoke --seeds 0 --epochs 2   # or shrink any grid
```

Difficulty is one number per example, the **margin** of the true class over its
best competitor, measured un-augmented in `eval()`. `scoring: transfer` gets it
from a **cross-fit teacher**: `teacher_folds` models, each trained on the
complement of the fold it scores, so nothing is ever ranked by a model that
trained on it. That is what lets the ranking be read off a converged model --
`scoring: self_margin` cannot, because a network is right about 99.9% of its own
train set by the end and, under label noise, right about the wrong labels too,
which is the reading that would destroy the experiment. The teacher trains on the
same corrupted labels the arms do: a curriculum that needed clean labels to build
its ordering would be assuming away the problem it is meant to help with.

Five things the implementation is careful about. Each would otherwise show up as
an accuracy difference that has nothing to do with curricula, and would look
entirely credible:

- **A short pool does not make a short epoch.** An epoch is always a full
  split's worth of examples, so a curriculum on half the data still takes the
  same number of SGD steps and `lr_at` still sees the schedule it was given.
  The pool is simply walked more than once, reshuffled each pass. What that
  costs is the repetition above, which is why `random` exists and why
  `exposure_profile` prints the count rather than leaving it as a caveat.
- **Every paced arm shares one pacing and one teacher.** `random`, `curriculum`
  and `anti` have identical `lam` and `pool_k` columns in `history.jsonl`; if
  they ever diverge the decomposition means nothing, and that is the first thing
  to check in a study's output.
- **Past the end of the ramp the run takes the baseline's code path exactly.**
  Once lambda reaches 1 the pool becomes `None` rather than a full-length index
  list, because a shuffle of the sorted order is a different sequence of images
  from a shuffle of the natural one at identical generator draws. What this does
  *not* mean is that the two arms coincide: a pooled epoch draws from the
  generator differently from an unpooled one, so the streams diverge during the
  ramp and never re-converge. The objective is identical after the ramp; the
  weights are not, and they are not supposed to be.
- **All arms get the same corruption.** `label_noise` draws from a constant
  seed, unrelated to the run seed, so what differs between arms is the order the
  examples arrive in and never which labels are wrong.
- **The separation is printed, not assumed.** With noise on, `curriculum_order`
  reports what share of the initial easy pool actually carries a corrupted
  label. If that is not well below the noise rate, the ordering separated
  nothing and no number downstream means what it says. Under `random_order` it
  should land *on* the noise rate -- the control working as designed.

Scores index the train split by position, so a scores *file* stays valid only
while `train_subset`, `val_size` and `label_noise` match the run that wrote it;
`load_scores` refuses to start rather than silently reordering a different set of
images. Inside a study the question does not arise: the teacher is fitted in
process, once per data regime, and handed to the arms in memory.

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

`enable_gpu: true` alone, from a CLI push, resolves to a **Tesla P100 (sm_60)**,
and recent torch builds ship sm_70 and up. Every kernel launch then fails with
`no kernel image is available for execution on the device`, a long way from the
cause. The fix is `machine_shape: "NvidiaTeslaT4"` in `kernel-metadata.json`.

**That field is honoured by the API**, contrary to what the note printed by
`make push` used to say: a CLI push lands on a T4 without touching the UI, and
`kaggle kernels push` also *starts* the run, so the whole campaign really is one
command. Measured, not assumed -- the earlier ResNet-18 runs of this study went
that way, 31 min each on a T4. Check `pick_device`'s line in the log if in doubt;
it names the device it actually got.

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
main.py                     the entry point; the study driver and KAGGLE_STUDY
analyze.py                  plots a results folder; local-only, never bundled
kernel-metadata.json        T4, internet on, datasets to mount
src/cifarbase/
  config.py                 the four-layer resolution, and CONFIG itself
  configs/*.yaml            recipes; bundled to Kaggle because they are under src/
  studies/*.yaml            grids of regimes x arms; bundled the same way
  data.py                   CIFAR-10, resident on the GPU as uint8
  model.py                  BasicBlock, ResNet, resnet20/32/44 and resnet18/34
  train.py                  build_optimizer, lr_at, train_once
  metrics.py                evaluate
  scoring.py                margins, the cross-fit teacher, pacing, the order
  report.py                 summarise, print_report, run_dir, study_dir, dump*
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
