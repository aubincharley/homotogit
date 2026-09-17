"""Assemble the four geometry archives (the report is shipped separately).

    py -m geometry_final.package --dest <folder>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from . import common as C

S = C.STUDY


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def zipdir(dest, name, items):
    """items: list of (source path, archive path)."""
    out = Path(dest) / name
    listing = {}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for src, arc in items:
            src = Path(src)
            if src.is_dir():
                for p in sorted(src.rglob("*")):
                    if p.is_file() and "_repo" not in p.parts and "__pycache__" not in p.parts:
                        a = (Path(arc) / p.relative_to(src)).as_posix()
                        zf.write(p, a)
                        listing[a] = sha(p)
            else:
                zf.write(src, arc)
                listing[arc] = sha(src)
        zf.writestr("SHA256SUMS.json", json.dumps(listing, indent=1))
    return out, len(listing)


def figures_md():
    caps = json.loads((S / "figures" / "CAPTIONS.json").read_text(encoding="utf-8"))
    lines = ["# Geometry figures", "",
             "Every figure is a vector PDF with a PNG preview. Captions are standalone; `source` names the table or raw file "
             "each figure is drawn from (inside `geometry_results.zip`). Regenerate with `py -m geometry_final.figures` and "
             "`py -m geometry_final.curvature_figs`.", "",
             "Suggested use: main text `F_surfaces_centre_frozen_seed0_zoom`, `F_curvature_ratios_to_plain`, "
             "`F_hessian_vs_random_rise_seed0`, `F_sensitivity_centre_frozen_pointwise`; everything else appendix.", ""]
    for name in sorted(caps):
        lines += ["## %s" % name, "", caps[name]["caption"], "", "Source: " + ", ".join("`%s`" % s for s in caps[name]["sources"]), ""]
    (S / "figures" / "FIGURES.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    a = ap.parse_args()
    Path(a.dest).mkdir(parents=True, exist_ok=True)
    figures_md()
    res = zipdir(a.dest, "geometry_results.zip", [
        (S / "tables", "tables"),
        (S / "checks", "checks/local"),
        (S / "raw" / "main" / "gf", "raw/main_job"),
        (S / "raw" / "pilot" / "gf", "raw/timing_pilot"),
        (C.ROOT / "studies/geometry_final/RESULTS_README.md", "README.md"),
    ])
    fig = zipdir(a.dest, "geometry_figures.zip", [(S / "figures", "figures")])
    rep = zipdir(a.dest, "geometry_reproducibility.zip", [
        (C.ROOT / "geometry_final", "code/geometry_final"),
        (S / "protocol", "protocol"),
        (S / "INVENTORY.md", "INVENTORY.md"),
        (S / "REPRODUCE.md", "REPRODUCE.md"),
        (S / "raw" / "main" / "gf" / "environment.json", "environment/kaggle_main_environment.json"),
        (S / "raw" / "launch_main.json", "environment/launch_main.json"),
        (S / "raw" / "launch_pilot.json", "environment/launch_pilot.json"),
    ])
    hand = zipdir(a.dest, "geometry_paper_handoff.zip", [(S / "paper_handoff", "paper_handoff")])
    for out, n in (res, fig, rep, hand):
        print(out.name, n, "files", round(out.stat().st_size / 2 ** 20, 1), "MiB")


if __name__ == "__main__":
    main()
