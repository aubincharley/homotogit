"""Grid training, shard 3 of 4."""
import os, sys
from pathlib import Path
os.environ["GRID_SHARD"] = "3/4"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import runpy
runpy.run_path(str(Path(__file__).resolve().parent / "job_grid.py"), run_name="__main__")
