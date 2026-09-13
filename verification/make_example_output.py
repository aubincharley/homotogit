"""Produce ``docs/examples/`` : a real output tree on synthetic data.

Illustrates the results schema only.  Synthetic random images, 3 epochs of 5
updates, a freshly generated asset set -- the numbers mean nothing.

    py verification/make_example_output.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from conftest import synthetic_dataset  # noqa: E402
from continuation_core import assets as assets_mod  # noqa: E402
from continuation_core.models import build_model  # noqa: E402
from continuation_core.presets import reference  # noqa: E402
from continuation_core.train import Trainer  # noqa: E402

OUT = ROOT / "docs" / "examples" / "synthetic_run"


def main():
    tmp = Path(tempfile.mkdtemp(prefix="example_"))
    ds = synthetic_dataset()
    assets_mod.make_assets(tmp / "assets", lambda: build_model("resnet20_bn_cifar", 10),
                           n_train=len(ds.train), epochs=7, seeds=(0,), probe_size=20,
                           provenance={"purpose": "schema example, synthetic data"})
    cfg = reference("resolution_max_b1_gaussian_conv", 0, assets_dir="<generated>",
                    device="cpu")
    cfg.assets.dir = str(tmp / "assets")
    cfg.data.name, cfg.data.expected_mean, cfg.data.expected_std = "synthetic", None, None
    cfg.budget.epochs, cfg.budget.effective_batch, cfg.budget.microbatch = 7, 32, 16
    cfg.evaluation.batch_size = 20
    cfg.checkpoint.transition_offsets = (-1, 1)
    cfg.validation_status = "example-synthetic"
    cfg.notes = ["synthetic random images; numbers are meaningless"]
    run = tmp / "run"
    Trainer(cfg, dataset=ds, out_dir=run, log=print).run()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for name in ("config.json", "environment.json", "metrics.json", "summary.json",
                 "assets_verification.json"):
        text = (run / name).read_text(encoding="utf-8").replace(str(tmp), "<tmp>")
        (OUT / name).write_text(text, encoding="utf-8")
    from continuation_core.checkpoint import load_checkpoint
    ck = load_checkpoint(run / "checkpoints" / "epoch_006.pt")
    listing = {k: (type(v).__name__ if k in ("model_state", "optimizer_state", "rng")
                   else v) for k, v in ck.items() if k not in ("metrics", "config")}
    (OUT / "checkpoint_epoch_006_contents.json").write_text(
        json.dumps(listing, indent=2, default=str).replace(str(tmp), "<tmp>") + "\n",
        encoding="utf-8")
    (OUT / "files.txt").write_text("\n".join(sorted(
        str(p.relative_to(run)).replace("\\", "/") for p in run.rglob("*") if p.is_file())) + "\n",
        encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
