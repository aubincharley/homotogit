"""Pre-launch checks for the unified batch.  No training, CPU only.

For every configuration: build the controller through the same entry point the
driver uses, run a forward and backward at each scheduled resolution, and check
shapes, finiteness and the exact bypass at the target configuration.

Also checks the two replay arms against the profile lengths their placement
requires -- a 19-long profile on a 10-position placement would silently index
wrong rather than raise.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from continuation.ablation_ops import build_from_cell
from continuation.config import ModelConfig
from continuation.models import build_model
from scripts.unified_manifest import build_cells, build_configs

OUT = Path("results/unified_verification.json")


def fresh():
    return build_model(ModelConfig(arch="resnet20_bn_cifar"), 10, seed=0)


def main():
    cfgs = build_configs()
    cells = build_cells(cfgs)
    report = {"n_configurations": len(cfgs), "n_cells": len(cells),
              "unique_ids": len({c["id"] for c in cfgs}),
              "unique_cells": len({c["cell_id"] for c in cells}),
              "configurations": {}}
    ok_all = True

    for cfg in cfgs:
        cell = {**cfg, "seed": 0, "cell_id": cfg["id"] + "__seed0"}
        model = fresh().train()
        ctrl, handles = build_from_cell(model, cell)
        entry = {"label": cfg["label"], "group": cfg["group"],
                 "placement": cfg["placement"], "reduction": cfg["reduction"],
                 "n_sites": len(ctrl.sites), "epochs": {}}
        if cfg.get("sigma_profile"):
            need = 10 if cfg["placement"] == "post_block" else 19
            entry["profile_len"] = len(cfg["sigma_profile"])
            entry["profile_len_ok"] = len(cfg["sigma_profile"]) == need

        for e in (0, 6, 12, 29):
            ctrl.set_epoch(e)
            r = ctrl.input_resolution()
            x = torch.rand(2, 3, r or 32, r or 32, requires_grad=True)
            y = model(x)
            y.pow(2).sum().backward()
            entry["epochs"][str(e)] = {
                "input_res": r or 32, "sigma": ctrl.value,
                "internal_res": ctrl.resolution,
                "logits": list(y.shape),
                "finite": bool(torch.isfinite(y).all()),
                "grad_finite": bool(torch.isfinite(x.grad).all()),
            }
            model.zero_grad(set_to_none=True)

        # exact target path: last epoch, forced 32x32 with everything bypassed
        model.eval()
        ref = fresh().eval()
        ref.load_state_dict(model.state_dict())
        xb = torch.rand(2, 3, 32, 32)
        with torch.no_grad():
            prev = (ctrl.value, ctrl.resolution, ctrl.bypass_all)
            ctrl.bypass_all = True
            ctrl.set_state(0.0, 32)
            got = model(xb)
            ctrl.value, ctrl.resolution, ctrl.bypass_all = prev
            want = ref(xb)
        d = float((got - want).abs().max())
        entry["target_path_max_abs_diff"] = d
        entry["target_path_bitwise"] = d == 0.0
        for h in handles:
            h.remove()

        good = (all(v["finite"] and v["grad_finite"]
                    for v in entry["epochs"].values())
                and entry["target_path_bitwise"]
                and entry.get("profile_len_ok", True))
        entry["ok"] = good
        ok_all = ok_all and good
        report["configurations"][cfg["id"]] = entry

    report["all_ok"] = ok_all and report["unique_ids"] == len(cfgs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))

    for cid, e in report["configurations"].items():
        res = "/".join(str(e["epochs"][k]["internal_res"]) for k in ("0", "6", "12", "29"))
        sig = "/".join("%.2f" % (e["epochs"][k]["sigma"] or 0)
                       for k in ("0", "6", "12", "29"))
        print("%-22s sites=%-3d res=%-14s sigma=%-19s bypass_exact=%-5s ok=%s"
              % (cid, e["n_sites"], res, sig, e["target_path_bitwise"], e["ok"]))
    print("\nconfigurations %d, cells %d, ALL OK: %s"
          % (len(cfgs), len(cells), report["all_ok"]))
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
