"""Assemble the comparison handoff archive.

    py -m comparison.package --dest <folder>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from . import matrix as MX

S = MX.STUDY


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    a = ap.parse_args()
    out = Path(a.dest) / "comparison_cbs_sdpoint_handoff.zip"
    items = [
        (S / "COMPARISON_REPORT.md", "COMPARISON_REPORT.md"),
        (MX.ROOT / "comparison" / "METHOD_MAPPING.md", "METHOD_MAPPING.md"),
        (S / "CLAIMS.md", "CLAIMS.md"),
        (S / "results", "results"),
        (S / "figures", "figures"),
        (S / "paper", "paper"),
        (S / "protocol", "protocol"),
        (MX.ROOT / "comparison", "code/comparison"),
        (MX.ROOT / "continuation_core", "code/continuation_core"),
        (MX.ROOT / "tests" / "test_comparators.py", "code/tests/test_comparators.py"),
        (MX.ROOT / "tests" / "conftest.py", "code/tests/conftest.py"),
    ]
    raw = S / "raw" / "main"
    listing = {}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for src, arc in items:
            src = Path(src)
            files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
            for p in files:
                name = arc if src.is_file() else (Path(arc) / p.relative_to(src)).as_posix()
                zf.write(p, name)
                listing[name] = sha(p)
        # raw shard outputs, final checkpoints included (1.1-2.3 MB each); no kernel payload copies
        for p in sorted(raw.rglob("*")):
            if p.is_file() and "_repo" not in p.parts and not p.name.endswith(".log"):
                name = "raw/" + p.relative_to(raw).as_posix()
                zf.write(p, name)
                listing[name] = sha(p)
        for p in sorted((S / "raw").glob("launch_*.json")) + sorted((S / "raw").glob("verify_*.json")):
            zf.write(p, "raw/" + p.name)
            listing["raw/" + p.name] = sha(p)
        zf.writestr("SHA256SUMS.json", json.dumps(listing, indent=1))
    print(out, len(listing), "files", round(out.stat().st_size / 2 ** 20, 1), "MiB")


if __name__ == "__main__":
    main()
