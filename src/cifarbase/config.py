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

    # ---- activation homotopy ---------------------------------------------
    # phi_alpha(x) = max(x, alpha*x), with alpha driven 1 -> 0 during training.
    # alpha=0 is exactly ReLU; alpha=1 makes the network affine. The default
    # a_schedule="none" builds no gate at all, so leaving these alone is
    # byte-identical to before they existed.
    #
    # Orthogonal to the s_* keys -- one deforms the residual branch, the other
    # the nonlinearity -- and both can run at once, though the pilot runs one
    # axis at a time.
    "a_schedule": "none",       # none|const|linear|cosine|staircase|sequential
    "a_start": 1.0,             # alpha where the ramp begins. 1.0 makes the
                                # network affine, whose optimum is a linear
                                # classifier (~40%) and whose Hessian is
                                # singular by construction. Deliberate, but a
                                # poor place to *start a continuation* from,
                                # which is why arms that track theta*(alpha)
                                # use 0.9 instead.
    "a_end": 0.0,               # alpha where it ends. Anything but 0 trains a
                                # LeakyReLU network, not the baseline's.
    "a_ramp_start": 0.0,        # fraction of training before the ramp begins
    "a_ramp_end": 0.5,          # fraction by which alpha has reached a_end
    "a_stairs": 0,              # staircase only: how many plateaus. 2 plateaus
                                # IS the jump control -- alpha held at a_start,
                                # then dropped to a_end with no path between.
    "a_lr_restart": False,      # staircase only: give every plateau its own
                                # warmup and decay, so theta can settle at each
                                # alpha_k before the continuation steps
    "a_scope": "global",        # global|stage|block|site -- what moves together
    "a_reverse": False,         # sequential only: stagger from the head down
                                # instead of from the stem up
    "a_sites": "all",           # all|act1|act2 -- which ReLUs take part. Only
                                # "all" makes the network affine at alpha=1.
    "a_bn_batches": 32,         # batches used to re-estimate BatchNorm for the
                                # alpha=0 readout. 0 skips the recompute, which
                                # makes that column an artefact -- see train.py.

    # ---- anchor penalty (GRDH eq. 12) -------------------------------------
    # L = L_CE + anchor_lambda * (alpha/2) * ||theta - theta_0||^2, over conv
    # and linear weights only. At alpha=1 the network is affine and its Hessian
    # is singular by construction, so theta*(1) is a manifold and the branch is
    # not well posed there; the quadratic picks one point out of it. The alpha
    # coefficient makes the penalty vanish at alpha=0, so the tail of training
    # minimises the baseline objective exactly.
    #
    # BatchNorm is deliberately outside the anchor: zero_init_residual starts
    # every bn2 gamma at 0, and anchoring that would pin the residual branches
    # off for the whole descent.
    "anchor_lambda": 0.0,       # 0 disables the anchor. A positive value is
                                # used as-is and recorded in results.json.
    "anchor_target": 0.0,       # >0 derives anchor_lambda so the penalty is
                                # this share of the cross-entropy, measured at
                                # anchor_calibrate_at. A tool for re-deriving
                                # lambda when the setup changes -- the arms pin
                                # the number instead, so the anchor acts from
                                # step 0 and every seed minimises the same
                                # function.
    "anchor_calibrate_at": 0.25,  # fraction of training at which to measure.
                                # NOT near 0: ||theta - theta_0||^2 grows by
                                # three orders of magnitude over the first ten
                                # epochs (5.4 -> 5449 measured), so a ratio
                                # taken at step 50 gives a lambda that is
                                # thousands of times too large by epoch 5.
                                # Mid-ramp is where the anchor actually acts.

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


def _validate_activation(cfg):
    """The a_* keys, checked hard enough that a bad sweep fails at launch.

    Two of these refuse a config that would silently do nothing rather than
    fail, which is the failure mode that costs a day per arm: a_reverse under a
    schedule that hands every group the same value, and a_reverse under a scope
    with only one group.
    """
    from cifarbase.activation import SCOPES, SITES
    from cifarbase.homotopy import SCHEDULES

    if cfg["a_schedule"] not in ("none",) + SCHEDULES:
        raise SystemExit(f"!! unknown a_schedule {cfg['a_schedule']!r}: pick "
                         f"one of none, {', '.join(SCHEDULES)}")
    if cfg["a_scope"] not in SCOPES:
        raise SystemExit(f"!! unknown a_scope {cfg['a_scope']!r}: "
                         f"pick one of {', '.join(SCOPES)}")
    if cfg["a_sites"] not in SITES:
        raise SystemExit(f"!! unknown a_sites {cfg['a_sites']!r}: "
                         f"pick one of {', '.join(SITES)}")
    if not 0.0 <= cfg["a_end"] <= cfg["a_start"] <= 1.0:
        raise SystemExit(f"!! need 0 <= a_end <= a_start <= 1, got "
                         f"a_start={cfg['a_start']}, a_end={cfg['a_end']}: "
                         f"alpha runs downhill, from a_start to a_end.")
    if cfg["a_schedule"] == "none":
        return cfg
    if cfg["a_schedule"] != "const":
        # A staircase may span the whole run. The tail rule exists so training
        # always ends on the model the accuracy is compared against, and a
        # staircase's last plateau already sits at a_end -- progress past
        # ramp_end returns that same level -- so the guarantee holds by
        # construction rather than by leaving room after the ramp. It is what
        # lets five plateaus be exactly ten epochs each in a fifty-epoch
        # budget. Every continuous schedule still needs the room: at
        # ramp_end=1 a linear ramp reaches a_end on the final epoch and the
        # target problem is never actually optimised.
        end = cfg["a_ramp_end"]
        spans_run = end <= 1.0 if cfg["a_schedule"] == "staircase" else end < 1.0
        if not (0.0 <= cfg["a_ramp_start"] < end and spans_run):
            raise SystemExit(
                f"!! need 0 <= a_ramp_start < a_ramp_end < 1, got "
                f"{cfg['a_ramp_start']} and {cfg['a_ramp_end']}. a_ramp_end "
                f"must leave a tail of training at a_end, or the run never "
                f"trains the model its accuracy will be compared against.")
        if cfg["a_schedule"] == "staircase" and cfg["a_stairs"] < 1:
            raise SystemExit(f"!! staircase needs a_stairs >= 1, got "
                             f"{cfg['a_stairs']}")
    if cfg["a_lr_restart"] and cfg["a_schedule"] != "staircase":
        raise SystemExit(
            f"!! a_lr_restart needs a_schedule=staircase, not "
            f"{cfg['a_schedule']!r}: restarting the lr only means something "
            f"where alpha is held still long enough for a phase to converge.")
    if cfg["a_reverse"]:
        if cfg["a_schedule"] != "sequential":
            raise SystemExit(
                f"!! a_reverse needs a_schedule=sequential, not "
                f"{cfg['a_schedule']!r}: every other schedule hands all groups "
                f"the same alpha, so reversing their order does nothing.")
        if cfg["a_scope"] == "global":
            raise SystemExit("!! a_reverse with a_scope=global reverses a "
                             "single group: pick a scope with more than one, "
                             "or turn the reversal off")
    if cfg["a_bn_batches"] < 0:
        raise SystemExit(f"!! a_bn_batches must be >= 0, got {cfg['a_bn_batches']}")
    return cfg


def _validate_anchor(cfg):
    """The anchor keys. Two refusals, both of a config that would do nothing.

    An anchor without a homotopy is a no-op, because the coefficient is
    lambda*alpha/2 and alpha never leaves 0. And a lambda given alongside a
    target is ambiguous: one of the two would be silently ignored, and
    results.json would record a number the run did not use.
    """
    if cfg["anchor_lambda"] < 0 or cfg["anchor_target"] < 0:
        raise SystemExit(f"!! anchor_lambda and anchor_target must be >= 0, got "
                         f"{cfg['anchor_lambda']} and {cfg['anchor_target']}")
    if cfg["anchor_lambda"] and cfg["anchor_target"]:
        raise SystemExit(
            f"!! anchor_lambda={cfg['anchor_lambda']} and "
            f"anchor_target={cfg['anchor_target']} both set: pick one. A "
            f"lambda is used as given; a target derives it after "
            f"anchor_calibrate_steps.")
    if (cfg["anchor_lambda"] or cfg["anchor_target"]) and \
            cfg["a_schedule"] in ("none", "const"):
        raise SystemExit(
            f"!! the anchor needs a moving alpha, but a_schedule is "
            f"{cfg['a_schedule']!r}: the coefficient is lambda*alpha/2, so "
            f"with no homotopy the penalty is identically zero.")
    if cfg["anchor_target"] and len(seed_list(cfg)) > 1:
        raise SystemExit(
            f"!! anchor_target with {len(seed_list(cfg))} seeds: train_once "
            f"runs once per seed and would derive a different lambda for each, "
            f"so the seeds would optimise different objectives and the spread "
            f"would mix seed variance with a hyperparameter change.\n"
            f"   Calibrate on one seed, then pin the lambda it reports:\n"
            f"     python main.py --config <name> --seeds 0\n"
            f"     # then put `anchor_lambda: <value>` in the YAML, drop "
            f"anchor_target, and run every seed and every other arm with it.")
    if not 0.0 < cfg["anchor_calibrate_at"] < 1.0:
        raise SystemExit(f"!! anchor_calibrate_at must be in (0, 1), got "
                         f"{cfg['anchor_calibrate_at']}: it is a fraction of "
                         f"training, and at 0 theta is still theta_0 and there "
                         f"is nothing to scale against.")
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
    _validate_activation(cfg)
    _validate_anchor(cfg)
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
