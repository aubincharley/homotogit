"""Slice launcher: the boundary extension of the sweep: AdamW at lr 0.02 and 0.03, 2 cells.
AdamW was the only optimizer whose best lr landed on an endpoint of the
original grid, which means that grid did not bracket its optimum.

``kaggle_run.py`` pushes one script and passes it no environment, so each slice
of the campaign is its own tiny entry point that sets the variables and then
runs the job.  Launch it with::

    py scripts/kaggle_run.py scripts/job_optbench_lr_sweep_ext.py --gpu         --accelerator NvidiaTeslaT4         --dataset alexandrecorrard/cifar-10-batches-py         --dataset alexandrecorrard/continuation-core-r20bn-assets         --include continuation_core --include scripts         --timeout-seconds 28800
"""
import os
import runpy
from pathlib import Path

os.environ["OPT_BENCH_BATCH"] = "lr_sweep_ext"
os.environ["OPT_BENCH_JOB"] = "0"
os.environ["OPT_BENCH_NJOBS"] = "1"

runpy.run_path(str(Path(__file__).with_name("job_optimizer_benchmark.py")),
               run_name="__main__")
