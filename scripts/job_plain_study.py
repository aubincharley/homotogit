"""Phase 3a: plain baseline, 3 seeds, 10,000 images, 1,200 updates."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._study_common import BASE, runs
from scripts.continuation_driver import run_study

CFG = {"name": "plain_study", **BASE, "runs": runs("none")}
CFG["runs"] = [{**r, "label": r["label"].replace("none_", "plain_")} for r in CFG["runs"]]

if __name__ == "__main__":
    run_study(CFG)
