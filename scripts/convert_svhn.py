"""Convert the official SVHN cropped-digit .mat files to raw uint8 binaries.

``continuation_core`` depends on torch and numpy only, and the canonical SVHN
distribution is MATLAB v7, which needs scipy to read.  Rather than add a
dependency to the package for one dataset, the conversion happens once, here,
and the loader reads the plain binaries the way it reads STL-10's.

Source (sha256 recorded in the manifest this writes)::

    http://ufldl.stanford.edu/housenumbers/train_32x32.mat   73,257 images
    http://ufldl.stanford.edu/housenumbers/test_32x32.mat    26,032 images

Two conventions are undone here, both silent failure modes if missed:

* ``X`` is stored ``(H, W, C, N)``; the loader wants ``(N, C, H, W)``;
* ``y`` is 1..10 where **10 is the digit zero**, so labels become ``y % 10``.

    py scripts/convert_svhn.py --src <dir of .mat> --out <dir>/svhn_binary

Needs scipy in the *calling* environment; the package never imports it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SPLITS = {"train": 73257, "test": 26032}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def to_nchw(x, y):
    """Undo SVHN's two storage conventions.  Pure, so it is unit-testable.

    ``x`` is ``(H, W, C, N)`` and ``y`` is 1..10 with 10 standing for the digit
    zero.  Returns contiguous ``(N, C, H, W)`` uint8 and 0..9 uint8 labels.
    """
    return (np.ascontiguousarray(x.transpose(3, 2, 0, 1)),
            (np.asarray(y).reshape(-1).astype(np.int64) % 10).astype(np.uint8))


def convert(src: Path, out: Path) -> dict:
    import scipy.io                                    # not a package dependency

    out.mkdir(parents=True, exist_ok=True)
    manifest = {"source": "http://ufldl.stanford.edu/housenumbers/",
                "note": "X transposed (H,W,C,N)->(N,C,H,W); labels y % 10 "
                        "because SVHN stores the digit zero as 10",
                "splits": {}}
    for split, expected in SPLITS.items():
        mat = src / ("%s_32x32.mat" % split)
        m = scipy.io.loadmat(mat)
        x, y = to_nchw(m["X"], m["y"])
        if x.shape != (expected, 3, 32, 32) or y.shape != (expected,):
            raise SystemExit("%s: got X %s y %s, expected (%d, 3, 32, 32)"
                             % (mat.name, x.shape, y.shape, expected))
        if x.dtype != np.uint8 or y.min() < 0 or y.max() > 9:
            raise SystemExit("%s: unexpected dtype/label range" % mat.name)
        x.tofile(out / ("%s_X.bin" % split))
        y.tofile(out / ("%s_y.bin" % split))
        manifest["splits"][split] = {
            "n": int(expected), "source_file": mat.name, "source_sha256": sha256(mat),
            "images_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
            "labels_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
            "label_counts": np.bincount(y, minlength=10).tolist()}
        print("%s: %d images -> %s_X.bin, %s_y.bin" % (split, expected, split, split))
    (out / "class_names.txt").write_text("\n".join(str(i) for i in range(10)) + "\n")
    (out / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, help="directory holding the two .mat files")
    ap.add_argument("--out", required=True, help="directory to write (name it svhn_binary)")
    a = ap.parse_args(argv)
    convert(Path(a.src), Path(a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
