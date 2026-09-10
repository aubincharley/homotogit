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
RUN_NAME = "cifar-curriculum"
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
    "arch": "resnet20",         # resnet20/32/44 (CIFAR, width 16) | resnet18/34
                                # (ImageNet topology, width 64)
    "width": 16,                # channels in stage 1; stages double from here.
                                # Tied to the arch family -- see model.WIDTHS
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

    # ---- curriculum ------------------------------------------------------
    # Wu, Dyer & Neyshabur (ICLR 2021) factorise a curriculum into two
    # independent choices, and so does this config:
    #
    #   SCORING  which examples are the easy ones      -> curriculum, scoring
    #   PACING   how fast the easy subset opens up     -> pacing, ramp_epochs,
    #                                                     easy_frac
    #
    # The factorisation is the point. At a fixed step budget, restricting the
    # pool oversamples whatever is inside it, so a curriculum arm differs from
    # the baseline in TWO ways at once and an accuracy gap alone cannot say
    # which one moved it. `curriculum: random_order` is the control that
    # separates them: same pacing, no difficulty information at all.
    "curriculum": "none",       # none | easy_first | hard_first | random_order
    "scoring": "transfer",      # transfer (cross-fit teacher) | self_margin
    "pacing": "linear",         # linear | quadratic | root | exp | step
    "ramp_epochs": 0,           # the epoch lambda reaches 1.0
    "easy_frac": 0.5,           # lambda0: the share visible at epoch 0
    "scores": "",               # .pt written by a --dump-scores run
    "dump_scores": False,       # write scores_s<seed>.pt at score_epoch
    "score_epoch": 10,          # read the margins after this many epochs
    # The cross-fit teacher: `teacher_folds` models, each trained on the
    # complement of the fold it scores, so no example is ever ranked by a model
    # that trained on it. That leak is what forces `self_margin` to be read
    # early; a teacher does not have the problem and can just be trained.
    "teacher_epochs": 8,
    "teacher_folds": 2,

    # ---- the noisy-label regime ------------------------------------------
    # Permute this share of the TRAIN labels, once, before any split is built.
    # It is where a curriculum has something to bridge: with clean CIFAR-10
    # the easy subproblem and the full problem are both well posed and the
    # curriculum connects two nearly identical points. Off by default.
    "label_noise": 0.0,

    # ---- the grid --------------------------------------------------------
    # A study is a grid of regimes x arms x seeds run by one process, which is
    # what makes the arms comparable: same data in memory, same teacher, one
    # push. `regime` and `arm` are stamped into each run's results.json so
    # analyze.py can group directories into arms instead of guessing from names.
    "study": "",                # name of a studies/<name>.yaml, "" = single run
    "regime": "",               # set by the study driver
    "arm": "",                  # set by the study driver
    "budget_min": 0.0,          # wall-clock cap for a study, 0 = no cap

    # ---- reporting -------------------------------------------------------
    "per_class": True,
    "wandb": True,              # falls back to NullRun when unavailable anyway
}

# The overlay `quick` applies on top of whatever config was selected, so it has
# to override every key that becomes inconsistent at this size -- warmup_epochs
# especially: baseline's 5 would exceed the 2 epochs left and never decay.
_QUICK = {
    "seeds": "0", "epochs": 2, "train_subset": 4000, "val_size": 0,
    "arch": "resnet20", "width": 16, "eval_batch_size": 500,
    "warmup_epochs": 0, "lr": 0.05, "per_class": False, "teacher_epochs": 1,
    # Same reason as warmup_epochs: a 16-epoch ramp and a reading at epoch 3
    # both fall outside a 2-epoch run, so a quick check of a curriculum recipe
    # would either refuse to start or test nothing. ramp_epochs=1 puts epoch 0
    # at lambda0 and epoch 1 at 1.0, so a two-epoch smoke test still exercises
    # both branches of pool_at.
    "ramp_epochs": 1, "score_epoch": 1,
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


def _studies_dir():
    return pathlib.Path(__file__).parent / "studies"


def available_studies():
    return sorted(p.stem for p in _studies_dir().glob("*.yaml"))


def load_study(name):
    """A grid of regimes x arms, checked against CONFIG's keys.

    A study is not a config: it is a small tree of them. `base` applies to every
    arm, each entry of `regimes` and `arms` overrides it, and everything but the
    `name` of a regime or an arm has to be a real CONFIG key -- for the same
    reason `_load_yaml` refuses unknown keys. A typo in a grid is worse than a
    typo in a config, because it silently makes two arms identical and the
    comparison between them reads as "no effect".
    """
    path = _studies_dir() / f"{name}.yaml"
    if not path.is_file():
        raise SystemExit(f"!! no study {name!r} in {_studies_dir()}: "
                         f"have {', '.join(available_studies()) or '(none)'}")
    study = yaml.safe_load(path.read_text()) or {}
    for key in ("regimes", "arms"):
        if not isinstance(study.get(key), list) or not study[key]:
            raise SystemExit(f"!! {path} needs a non-empty {key}: list")
    study.setdefault("name", name)
    study.setdefault("base", {})

    blocks = [("base", study["base"])]
    blocks += [(f"regime {r.get('name', '?')}", r) for r in study["regimes"]]
    blocks += [(f"arm {a.get('name', '?')}", a) for a in study["arms"]]
    for where, block in blocks:
        if not isinstance(block, dict):
            raise SystemExit(f"!! {path}: {where} must be a mapping")
        unknown = sorted(set(block) - set(CONFIG) - {"name"})
        if unknown:
            raise SystemExit(f"!! {path}, {where}: not CONFIG keys: "
                             f"{', '.join(unknown)}")
    names = [r.get("name") for r in study["regimes"]] + [a.get("name")
                                                         for a in study["arms"]]
    if any(not n for n in names):
        raise SystemExit(f"!! {path}: every regime and arm needs a name")
    return study


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


def _validate(cfg):
    # Imported here, not at module scope, so `--help` works on a machine with no
    # torch installed.
    from cifarbase.model import ARCHS, WIDTHS
    from cifarbase.scoring import MODES, PACINGS, has_scores

    if cfg["arch"] not in ARCHS:
        raise SystemExit(f"!! unknown arch {cfg['arch']!r}: "
                         f"pick one of {', '.join(ARCHS)}")
    if cfg["width"] != WIDTHS[cfg["arch"]]:
        raise SystemExit(
            f"!! {cfg['arch']} is defined at width {WIDTHS[cfg['arch']]}, not "
            f"{cfg['width']}: a {len(ARCHS[cfg['arch']])}-stage net at the other "
            f"family's width is a different model wearing this one's name. "
            f"Set width: {WIDTHS[cfg['arch']]}, or pick an arch of the other family.")
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
    if cfg["curriculum"] not in MODES:
        raise SystemExit(f"!! unknown curriculum {cfg['curriculum']!r}: "
                         f"pick one of {', '.join(MODES)}")
    if cfg["pacing"] not in PACINGS:
        raise SystemExit(f"!! unknown pacing {cfg['pacing']!r}: "
                         f"pick one of {', '.join(PACINGS)}")
    if cfg["scoring"] not in ("transfer", "self_margin"):
        raise SystemExit(f"!! unknown scoring {cfg['scoring']!r}: "
                         f"pick transfer or self_margin")
    if cfg["teacher_folds"] < 2:
        raise SystemExit(f"!! teacher_folds must be at least 2, got "
                         f"{cfg['teacher_folds']}: with one fold the teacher "
                         f"scores the examples it trained on, which is the leak "
                         f"a transfer teacher exists to remove")
    if cfg["teacher_epochs"] < 1:
        raise SystemExit(f"!! teacher_epochs must be at least 1, got "
                         f"{cfg['teacher_epochs']}")
    if cfg["curriculum"] != "none" and cfg["ramp_epochs"] <= 0:
        raise SystemExit(f"!! curriculum {cfg['curriculum']!r} with "
                         f"ramp_epochs 0 is the baseline under another name: "
                         f"give it a ramp, or use --curriculum none and mean it")
    if cfg["ramp_epochs"] < 0:
        raise SystemExit(f"!! ramp_epochs must be >= 0, got {cfg['ramp_epochs']}")
    if cfg["ramp_epochs"] > cfg["epochs"]:
        raise SystemExit(f"!! ramp_epochs ({cfg['ramp_epochs']}) exceeds epochs "
                         f"({cfg['epochs']}): the ramp would never finish and "
                         f"the model would never see the whole train set")
    if not 0.0 < cfg["easy_frac"] <= 1.0:
        raise SystemExit(f"!! easy_frac must be in (0, 1], got {cfg['easy_frac']}")
    if cfg["curriculum"] != "none" and cfg["easy_frac"] >= 1.0:
        raise SystemExit(f"!! easy_frac 1.0 with curriculum "
                         f"{cfg['curriculum']!r} is the baseline under another "
                         f"name: use --curriculum none and mean it")
    # A curriculum with no scores would sort by nothing and quietly run as the
    # baseline under a name that says otherwise -- the one failure this whole
    # experiment cannot afford. random_order is exempt because it sorts by
    # nothing on purpose, and a study is exempt because it fits the teacher
    # itself, in-process, before any arm starts.
    needs_scores = (cfg["curriculum"] in ("easy_first", "hard_first")
                    and not cfg["study"])
    if needs_scores and not has_scores(cfg["scores"]):
        raise SystemExit(f"!! curriculum {cfg['curriculum']!r} needs scores to "
                         f"sort by: pass --scores <scores_s0.pt> from a "
                         f"--dump-scores run, or run it inside a study, which "
                         f"fits the cross-fit teacher for you")
    if not 0.0 <= cfg["label_noise"] < 1.0:
        raise SystemExit(f"!! label_noise must be in [0, 1), got "
                         f"{cfg['label_noise']}")
    if cfg["dump_scores"] and not 1 <= cfg["score_epoch"] <= cfg["epochs"]:
        raise SystemExit(f"!! score_epoch ({cfg['score_epoch']}) is outside "
                         f"1..{cfg['epochs']}: nothing would ever be written")
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
    # Which values came from the command line, kept so a study can re-apply them
    # on top of its own grid: `--seeds 0` has to survive a study that sets three,
    # or there is no way to smoke-test a grid cheaply.
    cfg["cli_keys"] = sorted(given)
    return _validate(cfg)


def seed_list(cfg):
    return [int(v) for v in str(cfg["seeds"]).split(",") if str(v).strip()]
