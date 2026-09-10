"""Resolution benchmark job 0 (two T4s)."""
import os, sys, runpy
from pathlib import Path
os.environ["RESBENCH_JOB"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
runpy.run_path(str(Path(__file__).resolve().parent / "job_resbench.py"), run_name="__main__")
