"""Adaptive resolution, Phase 4 -- two-sided bounded specialisation (plan section 18).

``ADAPT_PHASE=4``: phase-3 ascent (g* = 0.06, delta 4), then at 32x32 one reheat
epoch at 24 whenever the downward gap g(32 -> 24) exceeds 0.15 / 0.30, never in
the last three epochs; plus Rsteps4 with fixed reheats at epochs 20/24/28.
"""
import os
import runpy
import sys
from pathlib import Path

os.environ["ADAPT_PHASE"] = "4"
os.environ["ADAPT_CONTROLLER_ONLY"] = "1"
os.environ.setdefault("ADAPT_SWEEP", "low")   # fixed control Rsteps4rh completed in the first launch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
runpy.run_path(str(Path(__file__).resolve().parent / "job_adaptive_phase0.py"),
               run_name="__main__")
