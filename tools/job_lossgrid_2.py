"""Shard 2 of the objective grid.  The environment is baked in, never exported."""
import os
import runpy
import sys
from pathlib import Path

os.environ["LOSS_SHARD"] = "2/4"
os.environ.setdefault("LOSS_ARMS", "square")
sys.argv = [sys.argv[0]]
runpy.run_path(str(Path(__file__).with_name("job_lossgrid.py")), run_name="__main__")
