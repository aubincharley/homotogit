"""STL-10, phase 3: fixed fine ramp vs adaptive reheats vs joint step-size controller,
equal compute (72-epoch horizon), six seeds.  See docs/adaptive_resolution_plan.md §19.
"""
import os
import runpy
import sys
from pathlib import Path

os.environ["STL10_PHASE"] = "3"
os.environ["STL10_SEEDS"] = "0,1,2,3,4,5"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
runpy.run_path(str(Path(__file__).resolve().parent / "job_stl10_resolution.py"),
               run_name="__main__")
