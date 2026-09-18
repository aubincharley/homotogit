"""Adaptive resolution, Phase 1 -- proposals of docs/adaptive_resolution_plan.md §13.

Thin launcher: selects the phase-1 run list of ``job_adaptive_phase0.py`` (the
Kaggle wrapper cannot pass environment variables).  Runs: Rprog / Rsteps4 /
Rlin12 with the Gaussian plateau filter, seeds 0/1/2; Rsteps4, Rlin12 and Rmixed
without filter, seeds 1/2; multi-step transfer efficiency logged on every run.
"""
import os
import runpy
import sys
from pathlib import Path

os.environ["ADAPT_PHASE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
runpy.run_path(str(Path(__file__).resolve().parent / "job_adaptive_phase0.py"),
               run_name="__main__")
