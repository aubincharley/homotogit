"""Record a sha256 for every delivered file of the project.

    py paper/tools/bundle_hashes.py

Writes ``provenance/bundle_sha256.json``. Transient TeX build files, the hash file
itself and Python caches are excluded, so the record covers exactly what a reader
receives.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
OUT = PAPER / "provenance" / "bundle_sha256.json"
SKIP_SUFFIX = {".aux", ".blg", ".fdb_latexmk", ".fls", ".log", ".out", ".synctex.gz",
               ".toc", ".lof", ".lot", ".pyc", ".tmp"}
SKIP_DIR = {"__pycache__", ".git"}


def main():
    files = {}
    for p in sorted(PAPER.rglob("*")):
        if not p.is_file() or p == OUT:
            continue
        if p.suffix in SKIP_SUFFIX or any(part in SKIP_DIR for part in p.parts):
            continue
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        files[p.relative_to(PAPER).as_posix()] = {"sha256": h.hexdigest(), "bytes": p.stat().st_size}
    OUT.write_text(json.dumps({"files": files, "count": len(files)}, indent=1) + "\n",
                   encoding="utf-8")
    print("wrote provenance/bundle_sha256.json for %d files" % len(files))


if __name__ == "__main__":
    main()
