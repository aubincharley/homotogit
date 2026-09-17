"""Shard 0 of the augmentation grid.  The environment is baked in, never exported."""
import os
import runpy
import sys
from pathlib import Path

os.environ["AUG_SHARD"] = "0/4"
os.environ.setdefault("AUG_CORNERS", "crop_flip:60:0.005,crop_flip:60:0.01,none:60:0.01")
os.environ.setdefault("AUG_STRETCH", "0")
sys.argv = [sys.argv[0]]
runpy.run_path(str(Path(__file__).with_name("job_augment.py")), run_name="__main__")
