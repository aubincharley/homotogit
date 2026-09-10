"""Ablation wave 5 (depth prior on sigma), job 1 of 4."""
import os, sys
from pathlib import Path
os.environ["ABLATION5_JOB"] = "1"
os.environ["ABLATION5_N_JOBS"] = "4"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runpy
runpy.run_path(str(Path(__file__).resolve().parent / "job_ablation5.py"),
               run_name="__main__")
