"""Read landscape_v2 raw evaluations as canonical v3 points.

Only kinds whose evaluation is identical to a v3 point are mapped:
sens1d, validation, fixed1d, traj (centres), surface.  calibsens (10k
calibration), interp, pcaplane and fixedsurf have no v3 counterpart and are read
directly by the analysis.
"""
from __future__ import annotations

import json

from . import common as C

V2C = C.V2


def _rows(path):
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
            out[r["key"]] = r
        except Exception:
            pass
    return out


def _slim(r, source):
    return {"splits": {k: {"ce": v["ce"], "acc": v["acc"], "n": v["n"], "finite": v.get("finite", True)}
                       for k, v in r["splits"].items()}, "source": source}


def load(seeds=C.SEEDS):
    pts = {}
    for s in seeds:
        ev = C.V2_RAW / C.V2_ACCOUNT_OF_SEED[s] / "v2" / "eval"
        for m in C.METHODS:
            sh, tgt = C.SHORT[m], C.target(m)
            src = "v2:sens1d__%s__seed%d" % (sh, s)
            for key, r in _rows(ev / ("sens1d__%s__seed%d.jsonl" % (sh, s))).items():
                if key.startswith("norm|"):
                    continue
                pol = key.split("|")[0]
                spec = "c" if r["k"] is None else C.spec_r1(r["k"], r["sign"] * r["amp"])
                pts[C.pkey(m, s, C.FINAL, tgt, pol, "probes", spec)] = _slim(r, src)
            src = "v2:validation__%s__seed%d" % (sh, s)
            for key, r in _rows(ev / ("validation__%s__seed%d.jsonl" % (sh, s))).items():
                pol = key.split("|")[0]
                spec = "c" if r["k"] is None else C.spec_r1(r["k"], r["sign"] * r["amp"])
                pts[C.pkey(m, s, C.FINAL, tgt, pol, "large", spec)] = _slim(r, src)
            for g in (21, 41):
                src = "v2:surface__%s__seed%d__g%d" % (sh, s, g)
                for key, r in _rows(ev / ("surface__%s__seed%d__g%d.jsonl" % (sh, s, g))).items():
                    spec = C.spec_r2(0, 1, r["a"], r["b"])
                    pts[C.pkey(m, s, C.FINAL, tgt, C.POINTWISE, "probes", spec)] = _slim(r, src)
            src = "v2:traj__%s__seed%d" % (sh, s)
            for key, r in _rows(ev / ("traj__%s__seed%d.jsonl" % (sh, s))).items():
                st = tgt if r["label"] == "final" else r["current_state"]
                pts[C.pkey(m, s, r["file"], st, r["policy"], "probes", "c")] = _slim(r, src)
            if m == "plain":
                continue
            for tr in V2C.fixed_weight_transitions(m):
                tid = "fixed1d__%s__seed%d__u%06d" % (sh, s, tr["update"])
                states = dict(V2C.fixed_weight_states(m, tr))
                ck = C.epoch_file(tr["update"] // C.UPE)
                for key, r in _rows(ev / (tid + ".jsonl")).items():
                    st = states[r["label"]]
                    spec = "c" if r["k"] is None else C.spec_r1(r["k"], r["sign"] * r["amp"])
                    pts[C.pkey(m, s, ck, st, r["policy"], "probes", spec)] = _slim(r, "v2:" + tid)
    return pts
