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

    # ---- residual homotopy -----------------------------------------------
    # h_out = shortcut(x) + s*F(x), with s driven 0 -> 1 during training. The
    # default is const at s_max=1, which IS the baseline: leaving these alone
    # gives byte-identical behaviour to before they existed.
    "s_schedule": "const",      # const|linear|cosine|staircase|sequential
    "s_min": 0.0,               # s at the start of the ramp. 0 freezes the
                                # residual branches outright (dL/dtheta_F ~ s),
                                # which is a choice, not an accident.
    "s_max": 1.0,
    "s_ramp_start": 0.0,        # fraction of training before the ramp begins
    "s_ramp_end": 0.5,          # fraction by which s has reached s_max. The tail
                                # after it trains the actual ResNet, which is
                                # what makes the accuracy comparable at all.
    "s_stairs": 0,              # staircase only: how many plateaus
    "s_lr_restart": False,      # staircase only: give every plateau its own
                                # warmup and decay. This is what turns the
                                # staircase into continuation rather than
                                # decoration -- under one global cosine the early
                                # plateaus never settle and the late ones cannot
                                # move, so theta never tracks theta*(s).
    "s_granularity": "step",    # step|epoch
    "lr_gate_control": False,   # the control arm: apply the s profile to the
                                # residual branch's learning rate instead of to
                                # the forward pass. If this reproduces the
                                # homotopy's effect, the homotopy was a learning
                                # rate schedule in disguise.

    # ---- landscape -------------------------------------------------------
    "ckpt_every": 0,            # >0 saves theta every N epochs, which is what
                                # explore.py needs to draw a trajectory
    "diag_every": 0,            # >0 measures the per-block residual ratio and
                                # gradient norms every N epochs, during the run.
                                # The same numbers explore.py derives afterwards
                                # from checkpoints -- but having them live is
                                # what lets you tell a run that is doing what you
                                # asked from one that is not, before it finishes.

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
    """
    path = _config_dir() / f"{name}.yaml"
    if not path.is_file():
        raise SystemExit(f"!! no config {name!r} in {_config_dir()}: "
                         f"have {', '.join(available_configs()) or '(none)'}")
    loaded = yaml.safe_load(path.read_text()) or {}
    if not isinstance(loaded, dict):
        raise SystemExit(f"!! {path} must be a mapping of key: value")
    unknown = sorted(set(loaded) - set(CONFIG))
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


def _validate_homotopy(cfg):
    """The homotopy keys, checked hard enough that a bad sweep fails at launch.

    The one that matters is s_ramp_end < 1: a run whose s never reaches s_max
    trained a different model than the baseline, so its accuracy is not
    comparable to the baseline's -- and nothing downstream would ever say so.
    """
    from cifarbase.homotopy import SCHEDULES

    if cfg["s_schedule"] not in SCHEDULES:
        raise SystemExit(f"!! unknown s_schedule {cfg['s_schedule']!r}: "
                         f"pick one of {', '.join(SCHEDULES)}")
    if cfg["s_granularity"] not in ("step", "epoch"):
        raise SystemExit(f"!! unknown s_granularity {cfg['s_granularity']!r}: "
                         f"pick step or epoch")
    if not 0.0 <= cfg["s_min"] <= cfg["s_max"] <= 1.0:
        raise SystemExit(f"!! need 0 <= s_min <= s_max <= 1, got "
                         f"s_min={cfg['s_min']}, s_max={cfg['s_max']}")
    if cfg["s_schedule"] != "const":
        if not 0.0 <= cfg["s_ramp_start"] < cfg["s_ramp_end"] < 1.0:
            raise SystemExit(
                f"!! need 0 <= s_ramp_start < s_ramp_end < 1, got "
                f"{cfg['s_ramp_start']} and {cfg['s_ramp_end']}. ramp_end must "
                f"leave a tail of training at s_max, or the run never trains "
                f"the model its accuracy will be compared against.")
        if cfg["s_schedule"] == "staircase" and cfg["s_stairs"] < 1:
            raise SystemExit(f"!! staircase needs s_stairs >= 1, got "
                             f"{cfg['s_stairs']}")
    if cfg["s_lr_restart"] and cfg["s_schedule"] != "staircase":
        raise SystemExit(
            f"!! s_lr_restart needs s_schedule=staircase, not "
            f"{cfg['s_schedule']!r}: restarting the lr only means something "
            f"where s is held still long enough for a phase to converge.")
    if cfg["lr_gate_control"] and cfg["s_schedule"] == "const":
        raise SystemExit("!! lr_gate_control with s_schedule=const applies a "
                         "constant multiplier and is just a different lr: set a "
                         "real schedule, or turn the control off")
    for key in ("ckpt_every", "diag_every"):
        if cfg[key] < 0:
            raise SystemExit(f"!! {key} must be >= 0, got {cfg[key]}")
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
    _validate_homotopy(cfg)
    if not seed_list(cfg):
        raise SystemExit("!! seeds is empty: give a comma-separated list, e.g. 0,1,2")
    return cfg


def load_config(argv=None, default=DEFAULT_CONFIG):
    """Resolve the four layers into one flat dict.

    `config` and `quick` are read first, from argv then env, because they select
    what the remaining layers get applied to.
    """
    parser = build_parser()
    given = vars(parser.parse_args(argv))

    name = given.pop("config", None) or os.environ.get(f"{ENV_PREFIX}CONFIG") or default

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
    return _validate(cfg)


def seed_list(cfg):
    return [int(v) for v in str(cfg["seeds"]).split(",") if str(v).strip()]
