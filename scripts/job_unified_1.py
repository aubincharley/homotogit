"""Unified batch, job 1."""
import os, sys
from pathlib import Path
os.environ["UNIFIED_JOB"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runpy
runpy.run_path(str(Path(__file__).resolve().parent / "job_unified.py"),
               run_name="__main__")
