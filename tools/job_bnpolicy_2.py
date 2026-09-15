"""Curvature across BatchNorm policy x split, shard 2 of 4."""
import os, sys
from pathlib import Path
os.environ["BNPOL_SHARD"] = "2/4"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import runpy
runpy.run_path(str(Path(__file__).resolve().parent / "job_bnpolicy.py"), run_name="__main__")
