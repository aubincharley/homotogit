"""STL-10 equal-compute controls: plain at 72 epochs and Rprog extended to R96's compute.

Thin launcher for ``job_stl10_resolution.py`` with ``STL10_PHASE=2``.
"""
import os
import runpy
import sys
from pathlib import Path

os.environ["STL10_PHASE"] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
runpy.run_path(str(Path(__file__).resolve().parent / "job_stl10_resolution.py"),
               run_name="__main__")
