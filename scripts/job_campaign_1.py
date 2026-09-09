"""Campaign job 1 -- runs on execution environment 1 (two T4s)."""
import os, sys
from pathlib import Path
os.environ["CAMPAIGN_JOB"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runpy
runpy.run_path(str(Path(__file__).resolve().parent / "job_campaign.py"),
               run_name="__main__")
