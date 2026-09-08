"""Phase 3c (PREPARED, NOT LAUNCHED): fused db2 continuation, 3 seeds.

Identical frozen protocol to plain/Gaussian.  s_k = 1 - 0.5 * max(1 - k/600, 0),
so s0 = 0.5 and the transform is an exact identity from update 600 onward.
No new wavelet calibration and no family sweep.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._study_common import BASE, runs
from scripts.continuation_driver import run_study

CFG = {"name": "db2_study", **BASE, "runs": runs("db2", s0=0.5)}

if __name__ == "__main__":
    run_study(CFG)
