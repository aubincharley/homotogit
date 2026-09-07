"""W&B if it is available, a no-op recorder if it is not.

The key rides in a mounted Kaggle dataset, never in the source. The import is
optional so a bare machine can still run the reference.
"""
import glob
import os

from cifarbase.config import PROJECT, RUN_NAME

try:
    import wandb
except ImportError:                                       # pragma: no cover
    wandb = None


class NullRun:
    """Enough of the wandb.Run surface for this project."""

    def __init__(self):
        self.summary = {}
        self.history = []

    def log(self, values, **_):
        self.history.append(values)

    def finish(self):
        pass


def init_run(cfg):
    if wandb is None:
        print("wandb not importable: metrics go to stdout and results.json only.")
        return NullRun()
    keys = glob.glob("/kaggle/input/**/wandb_key.txt", recursive=True)
    if keys:
        with open(keys[0]) as fh:
            os.environ["WANDB_API_KEY"] = fh.read().strip()
        print(f"W&B key found at {keys[0]}: logging online.")
    else:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_DIR"] = ("/kaggle/working" if os.path.isdir("/kaggle/working")
                                   else os.getcwd())
        print("No W&B key mounted: logging offline.")
    return wandb.init(project=PROJECT, name=RUN_NAME, config=cfg)


def finish(run):
    if wandb is not None and not isinstance(run, NullRun):
        wandb.finish()
