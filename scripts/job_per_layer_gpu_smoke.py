"""Ten-minute GPU validation of the per-layer / adaptive study before the real run.

Exists to catch what local CPU testing cannot: device placement, the Kaggle
CIFAR-10 mount path, and memory.  Deliberately too small to interpret -- it
answers "does this run on a T4", nothing else.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT = Path(os.environ.get("STUDY_OUT", "/kaggle/working")) / "per_layer_smoke"


def main():
    import torch
    print("cuda:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-", flush=True)
    from scripts.study_per_layer_cpu import main as study
    study(["--n-train", "5000", "--n-test", "2000", "--epochs", "4",
           "--filtered-epochs", "3", "--terminal-epochs", "1", "--ramp-stages", "1",
           "--batch", "128", "--min-stage-steps", "5", "--patience", "3",
           "--probe-size", "1024", "--eval-batch", "1000",
           "--arms", "plain,rho2,adaptive", "--out", str(OUT)])


if __name__ == "__main__":
    main()
