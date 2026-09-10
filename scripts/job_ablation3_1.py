"""Anti-aliasing ablation wave 3, job 1 of 4 -- one Kaggle environment (two T4s)."""
import os, sys
from pathlib import Path
os.environ["ABLATION3_JOB"] = "1"
os.environ["ABLATION3_N_JOBS"] = "4"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runpy
runpy.run_path(str(Path(__file__).resolve().parent / "job_ablation3.py"),
               run_name="__main__")
