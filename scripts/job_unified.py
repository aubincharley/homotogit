"""One unified-batch job: the cells assigned to this execution environment.

``UNIFIED_JOB`` (0-3) selects the slice.  The split is deterministic and derived
from the same manifest in every job, so each cell is owned by exactly one job
whichever machine runs it.  Longest-first within a job so neither GPU idles at
the end.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.unified_driver import log, run_job
from scripts.unified_manifest import build_cells

N_JOBS = 4


def slice_for(job: int):
    cells = build_cells()
    # blur-carrying cells cost more; interleave so the four jobs balance
    cells.sort(key=lambda c: (c["operator"] == "none", c["id"], c["seed"]))
    return [c for i, c in enumerate(cells) if i % N_JOBS == job]


if __name__ == "__main__":
    job = int(os.environ.get("UNIFIED_JOB", "0"))
    mine = slice_for(job)
    log("unified job %d: %d cells" % (job, len(mine)))
    log("ids: %s" % json.dumps(sorted({c["id"] for c in mine})))
    run_job(job, mine)
