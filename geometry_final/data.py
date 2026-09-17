"""Loaders joining existing landscape_v3/v2 raw points with the new geometry_final outputs."""
from __future__ import annotations

import base64
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from landscape_v3 import common as V3

from . import common as C

NEW = C.RAW / "main" / "gf"


def _jsonl(path):
    for line in Path(path).read_text().splitlines():
        if line.strip():
            try:
                yield json.loads(line)
            except Exception:
                continue


def v3_points(seed=0):
    pts = {}
    for f in glob.glob(str(C.V3_RAW / "*" / "v3" / "eval" / "*.jsonl")):
        if "_pilot" in f:
            continue
        for r in _jsonl(f):
            if "splits" in r and r.get("key", "").split("|")[1:2] == ["s%d" % seed]:
                pts.setdefault(r["key"], dict(r, _source="v3:" + Path(f).name))
    return pts


def v2_points(seed=0):
    from landscape_v3 import v2points
    V3.V2_RAW = C.V3_ROOT / "studies" / "landscape_v2" / "raw"        # the v2 raw data live in the visualization worktree
    return {k: dict(v, _source=v["source"]) for k, v in v2points.load(seeds=(seed,)).items()}


def grid_vertices():
    """Tidy 41x41 seed-0 vertices for both BN policies, with provenance of every value."""
    v3 = v3_points(0)
    v2 = v2_points(0)
    new = {}
    for f in sorted((NEW / "grid").glob("*.jsonl")):
        for r in _jsonl(f):
            new[r["key"]] = dict(r, _source="geometry_final:" + f.name)
    rows = []
    axis = list(C.GRID_AXIS)
    for m in C.METHODS:
        tgt = V3.target(m)
        for a in axis:
            for b in axis:
                pk = V3.pkey(m, 0, C.FINAL, tgt, C.POINTWISE, "probes", V3.spec_r2(0, 1, a, b))
                r = v3.get(pk) or v2.get(pk)
                ck = C.grid_key(m, a, b)
                if a == 0 and b == 0:
                    alt = V3.pkey(m, 0, C.FINAL, tgt, C.CFROZEN, "probes", "c")
                elif b == 0:
                    alt = V3.pkey(m, 0, C.FINAL, tgt, C.CFROZEN, "probes", V3.spec_r1(0, a))
                elif a == 0:
                    alt = V3.pkey(m, 0, C.FINAL, tgt, C.CFROZEN, "probes", V3.spec_r1(1, b))
                else:
                    alt = None
                rc = new.get(ck) or (v3.get(alt) if alt else None)
                for pol, rr in ((C.POINTWISE, r), (C.CFROZEN, rc)):
                    rows.append({"method": m, "seed": 0, "a": a, "b": b, "bn_policy": {"recalibrated": "pointwise"}.get(pol, pol),
                                 "train_probe_ce": np.nan if rr is None else rr["splits"]["train_probe"]["ce"],
                                 "test_probe_ce": np.nan if rr is None else rr["splits"]["test_probe"]["ce"],
                                 "train_probe_acc": np.nan if rr is None else rr["splits"]["train_probe"]["acc"],
                                 "test_probe_acc": np.nan if rr is None else rr["splits"]["test_probe"]["acc"],
                                 "source": None if rr is None else rr["_source"]})
    df = pd.DataFrame(rows)
    for sp in ("train_probe", "test_probe"):
        c = df[(df.a == 0) & (df.b == 0)].set_index(["method", "bn_policy"])[sp + "_ce"]
        df[sp + "_ce_centred"] = df[sp + "_ce"].values - c.loc[list(zip(df.method, df.bn_policy))].values
    return df


def trace_draws(root=NEW):
    rows, qs = [], {}
    for f in sorted((root / "trace").glob("*.jsonl")):
        for r in _jsonl(f):
            q = np.frombuffer(base64.b64decode(r["q_blocks_f64_b64"]), dtype="<f8")
            qs[(r["objective"], r["draw"])] = q
            rows.append({k: v for k, v in r.items() if k != "q_blocks_f64_b64"})
    df = pd.DataFrame(rows).drop_duplicates(subset=["objective", "draw"]).sort_values(["s", "probe", "m", "draw"])
    return df, qs


def blocks(m, s, root=NEW):
    z = np.load(root / "trace" / ("blocks__%s__seed%d.npz" % (C.SHORT[m], s)))
    return z["norm2"], z["sizes"]
