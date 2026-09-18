"""Adaptive resolution, pilot phase 6 -- onset-detection ascent, seed 0, 30-epoch horizon.
See docs/adaptive_resolution_plan.md section 20.
"""
import os
import runpy
import sys
from pathlib import Path

os.environ["ADAPT_PHASE"] = "6"
os.environ["ADAPT_EPOCHS"] = "30"
os.environ.setdefault("ADAPT_REHEAT_ONLY", "1")   # Ronset2 / Ronset3 completed in adaptive-phase6-20260918-141229
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
runpy.run_path(str(Path(__file__).resolve().parent / "job_adaptive_phase0.py"),
               run_name="__main__")
