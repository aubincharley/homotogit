"""Per-epoch signatures over the reference cells, shard 0 of 4."""
import os, sys
from pathlib import Path
os.environ["TRAJ_SHARD"] = "0/4"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import runpy
runpy.run_path(str(Path(__file__).resolve().parent / "job_trajectory.py"), run_name="__main__")
