"""Anti-aliasing ablation wave 2, job 3 of 4 -- one Kaggle environment (two T4s)."""
import os, sys
from pathlib import Path
os.environ["ABLATION2_JOB"] = "3"
os.environ["ABLATION2_N_JOBS"] = "4"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runpy
runpy.run_path(str(Path(__file__).resolve().parent / "job_ablation2.py"),
               run_name="__main__")
