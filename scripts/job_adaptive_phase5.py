"""Adaptive resolution, Phase 5 -- reheat controller on a fixed ascent, six seeds.

``ADAPT_PHASE=5``, seeds 0-5 regenerated in the kernel (0-2 verified against the
pinned campaign assets, 3-5 new and paired only within this job).  Arms: Rsteps4rh
(fixed reheats, seeds 3-5), Rsteps4ar (Rsteps4 ascent + adaptive reheat at
g_down >= 0.30, settle 2, seeds 0-5), Rsteps4 (seeds 3-5), Rgap2s30 (seeds 3-5).
"""
import os
import runpy
import sys
from pathlib import Path

os.environ["ADAPT_PHASE"] = "5"
os.environ["ADAPT_SEEDS"] = "0,1,2,3,4,5"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
runpy.run_path(str(Path(__file__).resolve().parent / "job_adaptive_phase0.py"),
               run_name="__main__")
