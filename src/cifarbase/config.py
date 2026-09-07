"""Everything tunable, resolved from four layers.

Later wins:

    CONFIG (this file)  ->  configs/<name>.yaml  ->  _QUICK  ->  env  ->  argv

Only the first three rungs exist on Kaggle. A `script` kernel is handed no
command line and no environment, so the recipe a push actually runs is whatever
`KAGGLE_CONFIG` in main.py names -- edit that and re-push to change it. The env
and argv rungs are there for local work, where wanting a two-epoch run should not
mean editing a file.

The YAMLs live inside the package rather than in a top-level configs/ because
build.py bundles src/ and nothing else. Putting them here means they ship to
Kaggle for free and build.py stays byte-identical to the other projects.

CONFIG is the fallback, not the reference: configs/baseline.yaml is the recipe
the README's accuracy number describes. CONFIG only has to be a complete,
runnable set of keys, since it also defines what a YAML or a flag is allowed to
name at all.
"""
import argparse
import os
import pathlib

import yaml

PROJECT = "cifar10-resnet"
RUN_NAME = "resnet18-sgd-baseline"
ENV_PREFIX = "CIFAR_"
DEFAULT_CONFIG = "baseline"
ARMS_KEY = "arms"

# Keys an arm may not touch. Every arm shares one set of loaded splits -- reading
# 240 MB per arm would buy nothing -- so an arm that changed the data would
# silently train on the previous arm's splits instead of the ones it declared.
_SHARED_DATA_KEYS = ("train_subset", "val_size")

CONFIG = {
    # ---- what to run -----------------------------------------------------
    "seeds": "0,1,2",           # a baseline is a mean and a spread, never one number
    "quick": False,             # tiny everything, for a CPU plumbing check
    "deterministic": False,     # pin cudnn algorithms; ~20% slower

    # ---- data ------------------------------------------------------------
    "batch_size": 128,
    "eval_batch_size": 1000,
    "val_size": 0,              # 0 trains on all 50k and reports the final epoch
                                # against the test set, which is what published
                                # CIFAR-10 numbers do. >0 holds out the last N
                                # train examples and selects on them instead.
    "train_subset": 0,          # >0 truncates train further (debugging)
    "augment": True,            # per-sample random crop + hflip, on the GPU

    # ---- model -----------------------------------------------------------
    "arch": "resnet18",         # resnet18 (the baseline) | resnet34
    "width": 64,                # channels in stage 1; stages double from here
    "zero_init_residual": True,

    # ---- optimisation ----------------------------------------------------
    "epochs": 100,
    "optimizer": "sgd",         # sgd is the only baseline; anything else is an
                                # experiment and belongs in its own config
    "lr": 0.1,
    "momentum": 0.9,
    "nesterov": True,
    "weight_decay": 5e-4,
    "schedule": "cosine",       # cosine | step | none
    "warmup_epochs": 5,
    "grad_clip": 0.0,           # 0 disables
    "label_smoothing": 0.0,

    # ---- the anchor ------------------------------------------------------
    # lambda(t)/2 * ||w - w0||^2, decoupled from the gradient. See anchor.py for
    # what gets anchored and controller.py for how lambda is chosen.
    "anchor_mode": "off",       # off | const | adaptive | replay | exp | critical
    "anchor_lambda": 0.0,       # const mode; also the pilot sweep's knob
    "anchor_lambda_max_frac": 0.5,   # lambda_max = frac / lr. 1.0 is §6 literally,
                                # 0.5 keeps lr*lambda off the overshoot boundary
    "anchor_hold_steps": 500,   # loop held shut over the initial transient
    "anchor_reference_step": -1,     # step at which K_0 is taken; -1 means "at the
                                # end of the hold window". w0 is ALWAYS step 0 --
                                # this only moves the drift reference, and it
                                # exists because with zero_init_residual the
                                # residual branches output exactly 0 at w0, which
                                # parks ~39% of the second ReLU's pre-activations
                                # exactly on the kink. The NTK is genuinely
                                # discontinuous there: selfcheck's K0 gate
                                # measures a 24% jump in the sketch for a 1e-8
                                # perturbation. K_0 at w0 is therefore not a
                                # reference d_t can ever return toward. 0 takes it
                                # at w0 anyway, which is §7 literally.
    "anchor_beta": 0.0,         # controller gain; 0 refuses to run adaptive
    "anchor_dmax": 0.0,         # the drift budget; 0 refuses to run adaptive
    "anchor_rho": 0.9,          # EMA on the drift signal
    "anchor_rate_limit": 0.1,   # max |delta log lambda| per probe
    "anchor_ratio_bn": 1.0,     # lambda multiplier on BatchNorm gamma, beta
    "anchor_ratio_head": 1.0,   # lambda multiplier on fc.weight
    "anchor_w0_dtype": "fp32",  # fp32 | bf16. bf16 halves w0's 45 MB; the anchor
                                # only needs about three decimal digits
    "anchor_exp_c": 0.0,        # exp arm: lambda_max * exp(-c t/T)
    "anchor_critical_frac": 0.2,     # critical arm: lambda_max for t < frac*T
    "anchor_replay_path": "",   # replay arm: a probes.jsonl to replay verbatim
    "anchor_replay_seed": 0,    # which seed's lambda trace in that file

    # ---- the per-group penalty and its allocation ------------------------
    # The penalty is sum_g (lambda_g/2) ||w_g - w0_g||^2 / ||w0_g||^2 over the
    # seven groups in anchor.py, with log lambda_g = u + a_g and sum_g a_g = 0.
    # u is the fast loop above; a_g is the slow loop in controller.Allocator.
    "anchor_normalize": True,   # divide the penalty by ||w0_g||^2, which is what
                                # makes lambda_g dimensionless and therefore
                                # comparable across depth. False is the raw
                                # (lambda/2)||w-w0||^2 the closed-form gate needs
                                # and is not a sensible setting for a real run.
    "anchor_grouping": "stage",      # stage (7 groups) | coarse (4) | trunk (3).
                                # How fine the allocation is allowed to be. The
                                # coupling gate is what chooses: a diagonal
                                # controller needs a plant whose off-diagonal
                                # coupling is small, and merging two groups that
                                # move together removes a coupling instead of
                                # hiding it. Coarsen when H7 fails; do not loosen
                                # H7.
    "anchor_alloc": "off",      # off | adaptive. off holds a_g = 0, i.e. one
                                # shared lambda -- the control arm.
    "anchor_alloc_every": 10,   # PROBES between slow-loop updates. 10 probes at
                                # probe_every=100 is 1000 steps, an order of
                                # magnitude slower than the level. Do not close
                                # this gap: the two-timescale separation is the
                                # only stability argument the pair has.
    "anchor_alloc_beta_frac": 0.2,   # beta_a = frac * anchor_beta. The slow loop
                                # is deliberately five times shyer than the fast
                                # one on top of being ten times rarer.
    "anchor_alloc_shrink": 0.01,     # gamma: pull a_g back toward uniform every
                                # update, so "no evidence" means "uniform"
                                # instead of an arbitrary random walk
    "anchor_alloc_clip_decades": 1.0,     # |a_g| <= decades * log(10). One decade
                                # either way is already a 100x spread in lambda
                                # across groups.
    "anchor_tau_rho": 0.9,      # EMA on the per-group gradient norm feeding the
                                # tension. A single minibatch read once every
                                # 1000 steps is mostly gradient noise; 0.0 is the
                                # protocol's literal single-batch reading.

    # ---- the drift probe -------------------------------------------------
    "probe_batch": 250,         # fixed, class-balanced, never augmented. 25 per
                                # class: the protocol asks for ~256 AND for exact
                                # class balance, and 256 is not divisible by 10.
                                # Exact balance wins -- an unbalanced probe makes
                                # ||K_t - K_0|| partly a statement about which
                                # classes it over-samples -- and 250 vs 256 is a
                                # 2% change in probe size.
    "probe_every": 100,         # STEPS between probes, not epochs
    "probe_tangents": 8,        # R: JVPs per probe
    "probe_kernel": "ntk",      # ntk (primary) | feature (cheap, must be validated)
    "probe_signal": "drift",    # drift = ||K_t-K_0||/||K_0||, the protocol's choice
                                # alignment = 1 - <K_t,K_0>/(||K_t|| ||K_0||), the
                                # same thing with the kernel's SCALE divided out.
                                # Both are always logged, along with the scale
                                # ratio: d^2 = scale^2 - 2 a scale + 1, so a d of
                                # 29 means the kernel's norm grew 29x whatever its
                                # geometry did. Check the scale column before
                                # reading a drift number as feature learning.
    "probe_group_every": 1,     # PROBES between per-group NTK sketches; 0 disables
                                # them. A per-group sketch costs L*R JVPs where
                                # the global one costs R -- 8x here -- and gives
                                # the global sketch for free, since the blocks sum
                                # to it. Purely a diagnostic and the coupling
                                # gate's regressand: the slow loop steers by
                                # tension, so turning this down changes what is
                                # logged and nothing about what is controlled.
    "probe_deterministic": False,    # pin cudnn inside the probe; see selfcheck H3
    "hessian_every": 0,         # steps between Lanczos sweeps; 0 disables
    "hessian_iters": 10,        # HVPs per sweep

    # ---- checkpointing ---------------------------------------------------
    "ckpt_every": 0,            # epochs between checkpoints; 0 disables
    "resume": "",               # a checkpoint to continue from
    "stop_after_epoch": 0,      # stop (and checkpoint) after this many epochs;
                                # 0 runs the whole schedule. This is how a
                                # 100-epoch run is split across Kaggle sessions,
                                # and it deliberately does NOT change `epochs`:
                                # total_steps, and therefore the entire lr curve,
                                # is derived from `epochs`, so lowering it to stop
                                # early would train on a different cosine and the
                                # resumed half would not continue the same run.

    # ---- where it runs ---------------------------------------------------
    "allow_cpu": False,         # let a Kaggle session run on CPU. pick_device
                                # refuses that by default, because a GPU session
                                # spent on CPU arithmetic burns the quota and
                                # usually hits the wall clock first. A script
                                # kernel is handed no environment, so
                                # CIFAR_ALLOW_CPU cannot reach it -- a config that
                                # is MEANT for a CPU kernel has to say so itself.

    # ---- reporting -------------------------------------------------------
    "per_class": True,
    "wandb": True,              # falls back to NullRun when unavailable anyway
}

# The overlay `quick` applies on top of whatever config was selected, so it has
# to override every key that becomes inconsistent at this size -- warmup_epochs
# especially: baseline's 5 would exceed the 2 epochs left and never decay.
_QUICK = {
    "seeds": "0", "epochs": 2, "train_subset": 4000, "val_size": 0,
    "arch": "resnet18", "width": 16, "eval_batch_size": 500,
    "warmup_epochs": 0, "lr": 0.05, "per_class": False,
    # The probe and the hold window are step counts, and 4000 images at bs 128 is
    # 31 steps an epoch. Left at their real values the probe would fire twice in
    # the whole run and the controller would never leave its hold, so a quick run
    # would exercise none of the code it exists to check.
    "probe_batch": 40, "probe_every": 5, "probe_tangents": 4,
    "anchor_hold_steps": 5, "hessian_every": 20, "hessian_iters": 4,
}

_CASTS = {bool: lambda s: s.strip().lower() in ("1", "true", "yes", "on"),
          int: int, float: float, str: str}


def _config_dir():
    return pathlib.Path(__file__).parent / "configs"


def available_configs():
    return sorted(p.stem for p in _config_dir().glob("*.yaml"))


def _load_yaml(name):
    """The named recipe, checked against CONFIG's keys.

    An unknown key is an error rather than a no-op. A typo'd `weigth_decay` that
    silently does nothing produces a run that looks fine, reports a plausible
    number, and answers a question you did not ask.

    `arms` is the one key that is not a CONFIG key: it is a list of override
    mappings, one per arm, and it exists because a Kaggle script kernel is handed
    one file and no command line. A comparison split across several pushes is a
    comparison whose arms ran different code, which is exactly the thing a
    baseline repo is for avoiding.
    """
    path = _config_dir() / f"{name}.yaml"
    if not path.is_file():
        raise SystemExit(f"!! no config {name!r} in {_config_dir()}: "
                         f"have {', '.join(available_configs()) or '(none)'}")
    loaded = yaml.safe_load(path.read_text()) or {}
    if not isinstance(loaded, dict):
        raise SystemExit(f"!! {path} must be a mapping of key: value")
    unknown = sorted(set(loaded) - set(CONFIG) - {ARMS_KEY})
    if unknown:
        raise SystemExit(f"!! {path} sets keys that are not in CONFIG: "
                         f"{', '.join(unknown)}")
    return loaded


def _env_overrides():
    """CIFAR_<KEY> for any CONFIG key, cast to the type of that key's default."""
    out = {}
    for key, value in CONFIG.items():
        raw = os.environ.get(f"{ENV_PREFIX}{key.upper()}")
        if raw is not None:
            out[key] = _CASTS[type(value)](raw)
    return out


def build_parser():
    """One flag per CONFIG key, typed from that key's default.

    Generated rather than written out so adding a config key cannot forget to add
    the flag. Every flag defaults to SUPPRESS, which is what makes the layering
    work: the namespace then holds only the flags actually passed, so an
    unmentioned flag cannot overwrite the value a YAML or an env var set.
    """
    parser = argparse.ArgumentParser(
        prog="cifar10-resnet",
        description="Train a ResNet on CIFAR-10. Flags override the config file.")
    parser.add_argument("--config", default=argparse.SUPPRESS,
                        metavar="NAME",
                        help=f"recipe in configs/ (default: {DEFAULT_CONFIG}; "
                             f"have {', '.join(available_configs()) or 'none'})")
    # An action, not a hyperparameter, so it is not a CONFIG key: it runs the
    # correctness gates and exits instead of training.
    parser.add_argument("--selfcheck", action="store_true",
                        default=argparse.SUPPRESS,
                        help="run the anchor's correctness gates and exit")
    for key, value in CONFIG.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            parser.add_argument(flag, dest=key, default=argparse.SUPPRESS,
                                action=argparse.BooleanOptionalAction,
                                help=f"(default: {value})")
        else:
            parser.add_argument(flag, dest=key, type=type(value),
                                default=argparse.SUPPRESS,
                                help=f"(default: {value})")
    return parser


def _validate_anchor(cfg):
    """The anchor's own consistency checks.

    Every one of these is a mistake that produces a runnable configuration and a
    meaningless number, which is the only kind worth spending a SystemExit on.
    """
    from cifarbase.controller import ALLOC_MODES, MODES

    mode = cfg["anchor_mode"]
    if mode not in MODES:
        raise SystemExit(f"!! unknown anchor_mode {mode!r}: "
                         f"pick from {', '.join(MODES)}")

    # Two shrinkage mechanisms stacking is the failure this whole config guards
    # against: weight_decay pulls toward 0, the anchor pulls toward w0, and any
    # result would be attributable to neither.
    if mode != "off" and cfg["weight_decay"] != 0.0:
        raise SystemExit(f"!! anchor_mode {mode!r} with weight_decay="
                         f"{cfg['weight_decay']}: decay toward 0 and the anchor "
                         f"toward w0 would stack and the arm would measure their "
                         f"sum. Set --weight-decay 0, or use the tuned-decay arm "
                         f"(anchor_mode off) to sweep weight_decay.")

    if mode == "adaptive":
        for key in ("anchor_beta", "anchor_dmax"):
            if cfg[key] <= 0.0:
                raise SystemExit(
                    f"!! adaptive anchoring needs {key} > 0, got {cfg[key]}. "
                    f"Both come out of the pilot (`--config pilot`): beta = 1/g "
                    f"with g = -d(d_ema)/d(log lambda) at mid-training, and "
                    f"anchor_dmax from the d_T the best fixed lambda reached. "
                    f"Guessing them is what the pilot exists to avoid.")
    if mode == "exp" and cfg["anchor_exp_c"] <= 0.0:
        raise SystemExit("!! the exp arm needs anchor_exp_c > 0: it is chosen by "
                         "bisection so d_T matches the closed-loop run's d_T")
    if mode == "replay":
        if not cfg["anchor_replay_path"]:
            raise SystemExit("!! the replay arm needs anchor_replay_path set to a "
                             "probes.jsonl from an adaptive run")
        if not os.path.isfile(cfg["anchor_replay_path"]):
            raise SystemExit(f"!! no such file: {cfg['anchor_replay_path']!r}")
    if mode == "const" and cfg["anchor_lambda"] < 0.0:
        raise SystemExit(f"!! anchor_lambda must be >= 0, got {cfg['anchor_lambda']}")

    from cifarbase.anchor import COARSENINGS

    if cfg["anchor_grouping"] not in COARSENINGS:
        raise SystemExit(f"!! unknown anchor_grouping {cfg['anchor_grouping']!r}: "
                         f"pick from {', '.join(COARSENINGS)}")

    alloc = cfg["anchor_alloc"]
    if alloc not in ALLOC_MODES:
        raise SystemExit(f"!! unknown anchor_alloc {alloc!r}: "
                         f"pick from {', '.join(ALLOC_MODES)}")
    if alloc == "adaptive":
        # beta_a is a fraction of beta, so an unfitted beta makes the slow loop
        # silently inert rather than obviously misconfigured.
        if mode != "adaptive":
            raise SystemExit(
                f"!! anchor_alloc adaptive with anchor_mode {mode!r}: the "
                f"allocation only redistributes the level the closed loop sets, "
                f"so it is meaningless without one. Set anchor_mode adaptive, or "
                f"anchor_alloc off.")
        if cfg["anchor_beta"] <= 0.0:
            raise SystemExit(f"!! anchor_alloc adaptive needs anchor_beta > 0: "
                             f"beta_a = anchor_alloc_beta_frac * beta, so a beta "
                             f"of {cfg['anchor_beta']} makes the slow loop inert")
        if cfg["anchor_alloc_beta_frac"] <= 0.0:
            raise SystemExit(f"!! anchor_alloc_beta_frac must be > 0, got "
                             f"{cfg['anchor_alloc_beta_frac']}")
    if cfg["anchor_alloc_every"] < 1:
        raise SystemExit(f"!! anchor_alloc_every must be at least 1 probe, got "
                         f"{cfg['anchor_alloc_every']}")
    if cfg["anchor_alloc_shrink"] < 0.0:
        raise SystemExit(f"!! anchor_alloc_shrink must be >= 0, got "
                         f"{cfg['anchor_alloc_shrink']}")
    if cfg["anchor_alloc_clip_decades"] <= 0.0:
        raise SystemExit(f"!! anchor_alloc_clip_decades must be > 0, got "
                         f"{cfg['anchor_alloc_clip_decades']}")
    if not 0.0 <= cfg["anchor_tau_rho"] < 1.0:
        raise SystemExit(f"!! anchor_tau_rho must be in [0, 1), got "
                         f"{cfg['anchor_tau_rho']}")
    if cfg["probe_group_every"] < 0:
        raise SystemExit(f"!! probe_group_every must be >= 0 (0 disables the "
                         f"per-group sketch), got {cfg['probe_group_every']}")
    if not cfg["anchor_normalize"] and mode != "off":
        # Not fatal: gate 5 needs exactly this. Loud, because on a real run it
        # silently un-does what makes lambda_g comparable across depth.
        print("!! WARNING: anchor_normalize is off. The penalty is the raw "
              "(lambda/2)||w - w0||^2, lambda is no longer dimensionless, and "
              "per-group lambdas are not comparable across depth. This setting "
              "exists for the closed-form gate.")

    if cfg["anchor_w0_dtype"] not in ("fp32", "bf16"):
        raise SystemExit(f"!! anchor_w0_dtype {cfg['anchor_w0_dtype']!r}: "
                         f"pick fp32 or bf16")
    if cfg["probe_signal"] not in ("drift", "alignment"):
        raise SystemExit(f"!! probe_signal {cfg['probe_signal']!r}: pick drift "
                         f"or alignment")
    if cfg["probe_kernel"] not in ("ntk", "feature"):
        raise SystemExit(f"!! probe_kernel {cfg['probe_kernel']!r}: pick ntk or "
                         f"feature. feature is the cheap path and is only "
                         f"admissible once its Spearman rho against ntk over a "
                         f"whole trajectory clears 0.95 (see analyze.py).")
    if cfg["probe_batch"] % 10:
        raise SystemExit(f"!! probe_batch {cfg['probe_batch']} is not a multiple "
                         f"of 10: a class-balanced probe over CIFAR-10 needs an "
                         f"equal share per class")
    for key in ("probe_every", "probe_tangents"):
        if cfg[key] < 1:
            raise SystemExit(f"!! {key} must be at least 1, got {cfg[key]}")
    if not 0.0 <= cfg["anchor_rho"] < 1.0:
        raise SystemExit(f"!! anchor_rho must be in [0, 1), got {cfg['anchor_rho']}")
    if cfg["anchor_rate_limit"] <= 0.0:
        raise SystemExit(f"!! anchor_rate_limit must be > 0, got "
                         f"{cfg['anchor_rate_limit']}")
    if cfg["hessian_every"] and cfg["hessian_every"] % cfg["probe_every"]:
        raise SystemExit(f"!! hessian_every ({cfg['hessian_every']}) must be a "
                         f"multiple of probe_every ({cfg['probe_every']}): the "
                         f"Lanczos sweep rides along with a probe so lambda_min "
                         f"is read on the same row as d_t")
    if cfg["stop_after_epoch"] < 0 or cfg["stop_after_epoch"] > cfg["epochs"]:
        raise SystemExit(f"!! stop_after_epoch must be between 0 and epochs "
                         f"({cfg['epochs']}), got {cfg['stop_after_epoch']}")
    if cfg["stop_after_epoch"] and not cfg["ckpt_every"]:
        raise SystemExit("!! stop_after_epoch without ckpt_every: the run would "
                         "stop and leave nothing to resume from")
    if cfg["resume"] and not os.path.isfile(cfg["resume"]):
        raise SystemExit(f"!! no checkpoint at {cfg['resume']!r}")
    return cfg


def _validate(cfg):
    # Imported here, not at module scope, so `--help` works on a machine with no
    # torch installed.
    from cifarbase.model import ARCHS

    if cfg["arch"] not in ARCHS:
        raise SystemExit(f"!! unknown arch {cfg['arch']!r}: "
                         f"pick one of {', '.join(ARCHS)}")
    if cfg["optimizer"] != "sgd":
        raise SystemExit(f"!! optimizer {cfg['optimizer']!r}: this baseline is "
                         f"SGD only. Add the branch in train.build_optimizer "
                         f"deliberately, as an experiment.")
    if cfg["schedule"] not in ("cosine", "step", "none"):
        raise SystemExit(f"!! unknown schedule {cfg['schedule']!r}: "
                         f"pick cosine, step or none")
    for key in ("epochs", "batch_size", "eval_batch_size"):
        if cfg[key] < 1:
            raise SystemExit(f"!! {key} must be at least 1, got {cfg[key]}")
    if cfg["warmup_epochs"] > cfg["epochs"]:
        raise SystemExit(f"!! warmup_epochs ({cfg['warmup_epochs']}) exceeds "
                         f"epochs ({cfg['epochs']}): the lr would never decay")
    if not seed_list(cfg):
        raise SystemExit("!! seeds is empty: give a comma-separated list, e.g. 0,1,2")
    return _validate_anchor(cfg)


def load_config(argv=None, default=DEFAULT_CONFIG):
    """Resolve the four layers into one flat dict.

    `config` and `quick` are read first, from argv then env, because they select
    what the remaining layers get applied to.
    """
    parser = build_parser()
    given = vars(parser.parse_args(argv))

    name = given.pop("config", None) or os.environ.get(f"{ENV_PREFIX}CONFIG") or default
    selfcheck = given.pop("selfcheck", False)

    cfg = dict(CONFIG)
    cfg.update(_load_yaml(name))

    env = _env_overrides()
    quick = given.get("quick", env.get("quick", cfg["quick"]))
    if quick:
        cfg.update(_QUICK)
    cfg["quick"] = bool(quick)

    cfg.update(env)
    cfg.update(given)
    cfg["config"] = name
    cfg["selfcheck"] = bool(selfcheck)
    arms = cfg.pop(ARMS_KEY, None)
    cfg = _validate(cfg)
    # Validated last, and against the already-resolved base: an arm inherits
    # every flag and env var the invocation set, so `--epochs 3` shortens all of
    # them rather than just the first.
    cfg[ARMS_KEY] = _resolve_arms(cfg, arms)
    return cfg


def _resolve_arms(cfg, arms):
    """The declared arms as fully-resolved configs. Empty when there are none.

    Each arm is the base config plus its own overrides, re-validated on its own.
    Re-validated because the guards that matter most here are per-arm: an arm
    that turns the anchor on while leaving weight_decay at the baseline value
    would stack two shrinkages and measure their sum.
    """
    if not arms:
        return []
    if not isinstance(arms, list):
        raise SystemExit(f"!! {ARMS_KEY} must be a list of mappings, one per arm")

    resolved, seen = [], set()
    for index, entry in enumerate(arms):
        if not isinstance(entry, dict):
            raise SystemExit(f"!! {ARMS_KEY}[{index}] must be a mapping of "
                             f"key: value, with a `name`")
        overrides = dict(entry)
        name = str(overrides.pop("name", f"arm{index}"))
        unknown = sorted(set(overrides) - set(CONFIG))
        if unknown:
            raise SystemExit(f"!! arm {name!r} sets keys that are not in CONFIG: "
                             f"{', '.join(unknown)}")
        clash = sorted(set(overrides) & set(_SHARED_DATA_KEYS))
        if clash:
            raise SystemExit(f"!! arm {name!r} sets {', '.join(clash)}, which "
                             f"decides the data splits. Every arm shares one set "
                             f"of loaded splits, so this would train the arm on "
                             f"different data than it declares. Put it in the "
                             f"config body instead, where it applies to all arms.")
        if name in seen:
            raise SystemExit(f"!! two arms are both named {name!r}: the run "
                             f"directories would collide")
        seen.add(name)

        arm = dict(cfg)
        arm.pop(ARMS_KEY, None)
        arm.update(overrides)
        arm["arm"] = name
        resolved.append(_validate(arm))
    return resolved


def arm_configs(cfg):
    """(name, cfg) per arm. A single unnamed arm when the config declares none."""
    arms = cfg.get(ARMS_KEY) or []
    return [(arm["arm"], arm) for arm in arms] or [(None, cfg)]


def seed_list(cfg):
    return [int(v) for v in str(cfg["seeds"]).split(",") if str(v).strip()]
