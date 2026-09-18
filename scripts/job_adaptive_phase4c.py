"""Adaptive resolution, Phase 4c -- two-sided controller, higher reheat bounds.

g*_down in {0.45, 0.60} with settle = 1 (reheats allowed up to epoch 29), after the
0.15 / 0.30 sweep of adaptive-phase4b-20260917-154529.
"""
import os
import runpy
import sys
from pathlib import Path

os.environ["ADAPT_PHASE"] = "4"
os.environ["ADAPT_CONTROLLER_ONLY"] = "1"
os.environ["ADAPT_SWEEP"] = "high"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
runpy.run_path(str(Path(__file__).resolve().parent / "job_adaptive_phase0.py"),
               run_name="__main__")
