#!/usr/bin/env python3
"""Bundle src/ and main.py into dist/main.py, the one file Kaggle runs.

A Kaggle kernel is a single file: the save-kernel API takes one `text` field
and a list of attached sources, with no way to carry a second file. So the
whole package is embedded here as a base64 tar.gz that the generated header
unpacks onto sys.path before your code runs.

    python build.py        # writes dist/main.py
"""
import base64
import gzip
import io
import pathlib
import tarfile
import textwrap

ROOT = pathlib.Path(__file__).parent.resolve()
SRC = ROOT / "src"
ENTRY = ROOT / "main.py"
OUT = ROOT / "dist" / "main.py"

SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", ".mypy_cache", ".pytest_cache", ".git"}
SKIP_SUFFIXES = {".pyc", ".pyo"}

HEADER = '''\
# ==============================================================
#  GENERATED FILE - DO NOT EDIT
#  Built by build.py from src/ + main.py
#  Payload: {files} files, {kb:.1f} KB compressed
#  To change anything: edit src/ or main.py, re-run `python build.py`,
#  then push this kernel.
# ==============================================================
import base64
import gzip
import io
import sys
import tarfile
import tempfile

_PAYLOAD = """
{blob}
"""


def _unpack_payload():
    """Extract the embedded src/ tree to a temp dir and return its path."""
    raw = gzip.decompress(base64.b64decode(_PAYLOAD))
    dest = tempfile.mkdtemp(prefix="bundle_")
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        try:
            tar.extractall(dest, filter="data")  # Python 3.12+
        except TypeError:
            tar.extractall(dest)
    return dest


sys.path.insert(0, _unpack_payload() + "/src")

# ----------------------- main.py below ------------------------
'''


def collect_files():
    """Every file under src/ worth shipping, in a stable order."""
    found = []
    for path in sorted(SRC.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(SRC)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if path.suffix in SKIP_SUFFIXES:
            continue
        found.append((path, rel))
    return found


def build():
    files = collect_files()
    if not files:
        raise SystemExit(f"nothing to bundle: no files under {SRC}")

    # Zeroed metadata and a fixed gzip mtime keep the output byte-identical
    # across rebuilds, so an unchanged src/ never produces a new kernel version.
    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w") as tar:
        for path, rel in files:
            info = tar.gettarinfo(str(path), arcname=f"src/{rel.as_posix()}")
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            info.mode = 0o644
            with path.open("rb") as fh:
                tar.addfile(info, fh)

    gz_buf = io.BytesIO()
    with gzip.GzipFile(fileobj=gz_buf, mode="wb", compresslevel=9, mtime=0) as gz:
        gz.write(tar_buf.getvalue())
    payload = gz_buf.getvalue()

    blob = "\n".join(textwrap.wrap(base64.b64encode(payload).decode(), 76))
    bundle = HEADER.format(files=len(files), kb=len(payload) / 1024, blob=blob)
    bundle += ENTRY.read_text()

    OUT.parent.mkdir(exist_ok=True)
    previous = OUT.read_text() if OUT.exists() else None
    OUT.write_text(bundle)

    for _, rel in files:
        print(f"  + src/{rel.as_posix()}")
    print(f"\n{OUT.relative_to(ROOT)}: {len(bundle) / 1024:.1f} KB total, "
          f"{len(files)} files embedded")
    if previous == bundle:
        print("identical to the last build - no need to re-push")


if __name__ == "__main__":
    build()
