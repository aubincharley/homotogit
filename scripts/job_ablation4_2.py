"""Ablation wave 4 (seeds 1-2 on the top four arms), job 2 of 4."""
import os, sys
from pathlib import Path
os.environ["ABLATION4_JOB"] = "2"
os.environ["ABLATION4_N_JOBS"] = "4"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runpy
runpy.run_path(str(Path(__file__).resolve().parent / "job_ablation4.py"),
               run_name="__main__")
