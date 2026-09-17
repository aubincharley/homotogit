"""Build the three handoff archives next to OVERNIGHT_REPORT.md.

    py -m overnight.package <deliver dir>
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from . import matrix as MX

S = MX.STUDY


def add_tree(zf, src: Path, arc: str, suffixes=None, exclude=("__pycache__",)):
    for p in sorted(src.rglob("*")):
        if p.is_file() and not any(e in p.parts for e in exclude) and (suffixes is None or p.suffix in suffixes):
            zf.write(p, "%s/%s" % (arc, p.relative_to(src).as_posix()))


def main(dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(S / "OVERNIGHT_REPORT.md", dest / "OVERNIGHT_REPORT.md")
    with zipfile.ZipFile(dest / "overnight_results.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        add_tree(zf, S / "analysis", "overnight_results/analysis")
        add_tree(zf, S / "recovered_augmentation", "overnight_results/recovered_augmentation")
        zf.write(S / "DATA_DICTIONARY.md", "overnight_results/DATA_DICTIONARY.md")
        zf.write(S / "protocol" / "ALLOCATION.json", "overnight_results/status/ALLOCATION.json")
        for p in sorted((S / "raw").glob("verify_*.json")) + sorted((S / "raw").glob("launch_*.json")):
            j = json.loads(p.read_text())
            j.pop("payload_files", None)
            zf.writestr("overnight_results/status/%s" % p.name, json.dumps(j, indent=1))
    with zipfile.ZipFile(dest / "overnight_reproducibility.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        add_tree(zf, S / "protocol", "overnight_reproducibility/protocol")
        for pkg in ("overnight", "continuation_core", "comparison"):
            add_tree(zf, MX.ROOT / pkg, "overnight_reproducibility/code/%s" % pkg, suffixes={".py", ".json", ".md"})
        add_tree(zf, MX.ROOT / "tests", "overnight_reproducibility/code/tests", suffixes={".py"})
        base = subprocess.run(["git", "diff", "6b28489", "HEAD", "--", "continuation_core"], cwd=MX.ROOT, capture_output=True, text=True).stdout
        zf.writestr("overnight_reproducibility/core_changes_vs_6b28489.patch", base)
        zf.write(MX.ROOT / "assets/cifar10_resnet20bn/assets_manifest.json", "overnight_reproducibility/assets_manifest.json")
        zf.write(S / "REPRODUCE.md", "overnight_reproducibility/REPRODUCE.md")
        zf.write(S / "CHECKPOINTS.md", "overnight_reproducibility/CHECKPOINTS.md")
        for p in sorted((S / "raw").glob("*/*/ovn/environment.json")):
            zf.write(p, "overnight_reproducibility/environments/%s_%s.json" % (p.parts[-4], p.parts[-3]))
        for p in sorted((S / "raw").glob("pilot/*/ovn/pilot_tests.txt")) + sorted((S / "raw").glob("pilot/*/ovn/pilot_resume_gpu*.json")):
            zf.write(p, "overnight_reproducibility/checks/%s_%s" % (p.parts[-3], p.name))
        for p in sorted((S / "raw").glob("pilot/*/ovn/pilot/*/pilot_timing.json")):
            zf.write(p, "overnight_reproducibility/checks/pilot_timing/%s.json" % p.parent.name)
    with zipfile.ZipFile(dest / "overnight_paper_handoff.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        add_tree(zf, S / "figures", "overnight_paper_handoff/figures")
        for f in ("summary_by_arm.csv", "paired_contrasts.csv", "r_rg_vs_sdpoint_across_policies.csv", "costs.csv",
                  "learning_curves_p1.csv", "learning_curves_saved_bn_per_epoch.csv", "gain_changes_across_regimes.csv"):
            zf.write(S / "analysis" / f, "overnight_paper_handoff/data/%s" % f)
        zf.write(MX.ROOT / "overnight/figures.py", "overnight_paper_handoff/make_figures.py")
        zf.write(S / "CLAIMS.md", "overnight_paper_handoff/CLAIMS.md")
        zf.writestr("overnight_paper_handoff/README.md",
                    "Regenerate: `python make_figures.py --analysis data --out figures` (needs matplotlib, numpy, pandas).\n"
                    "Figures: PDF + PNG preview; LaTeX tables `table_*.tex`; captions in `figures/CAPTIONS.md`; claim index in `CLAIMS.md`.\n"
                    "The manuscript is not edited.\n")
    for p in sorted(dest.iterdir()):
        print(p.name, p.stat().st_size)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
