"""Anti-aliasing ablation job 2 of 4 -- one Kaggle execution environment (two T4s).

The 12-cell manifest is built identically in all four jobs and each selects its
own slice by index, so the partition is deterministic and no cell is ever run
twice.  Jobs 0-1 run on one account, jobs 2-3 on the other.
"""
import os, sys
from pathlib import Path
os.environ["ABLATION_JOB"] = "2"
os.environ["ABLATION_N_JOBS"] = "4"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runpy
runpy.run_path(str(Path(__file__).resolve().parent / "job_ablation.py"),
               run_name="__main__")
