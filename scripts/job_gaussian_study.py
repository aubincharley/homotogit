"""Phase 3b: Gaussian continuation, 3 seeds, identical protocol to plain.

sigma_k = max(1 - k/600, 0); exact bypass from update 600; 19 insertion points;
normalized 9-tap kernel (radius 4).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._study_common import BASE, runs
from scripts.continuation_driver import run_study

CFG = {"name": "gaussian_study", **BASE, "runs": runs("gaussian")}

if __name__ == "__main__":
    run_study(CFG)
