"""Endpoint probes over the grid, shard 2 of 4."""
import os, sys
from pathlib import Path
os.environ["PROBE_SHARD"] = "2/4"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import runpy
runpy.run_path(str(Path(__file__).resolve().parent / "job_probe.py"), run_name="__main__")
