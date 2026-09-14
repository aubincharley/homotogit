"""Slice launcher: slice 2 of the main grid, 12 of 36 cells.

``kaggle_run.py`` pushes one script and passes it no environment, so each slice
of the campaign is its own tiny entry point that sets the variables and then
runs the job.  Launch it with::

    py scripts/kaggle_run.py scripts/job_optbench_grid_2.py --gpu         --accelerator NvidiaTeslaT4         --dataset pankrzysiu/cifar10-python         --dataset <owner>/continuation-core-r20bn-assets         --include continuation_core --include scripts         --timeout-seconds 28800
"""
import os
import runpy
from pathlib import Path

os.environ["OPT_BENCH_BATCH"] = "grid"
os.environ["OPT_BENCH_JOB"] = "2"
os.environ["OPT_BENCH_NJOBS"] = "3"

runpy.run_path(str(Path(__file__).with_name("job_optimizer_benchmark.py")),
               run_name="__main__")
