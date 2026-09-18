"""Adaptive resolution, Phase 3 -- bounded-specialisation controller (plan section 17).

Thin launcher: ``ADAPT_PHASE=3`` selects nine unfiltered CIFAR-10 runs, seeds 0/1/2,
with the live controller "advance by 4 when the relative scale-transfer gap to
r+4 (BatchNorm recalibrated) exceeds g*", g* in {0.04, 0.06, 0.08}.
"""
import os
import runpy
import sys
from pathlib import Path

os.environ["ADAPT_PHASE"] = "3"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
runpy.run_path(str(Path(__file__).resolve().parent / "job_adaptive_phase0.py"),
               run_name="__main__")
