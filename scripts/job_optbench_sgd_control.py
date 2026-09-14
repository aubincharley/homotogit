"""Slice launcher: the SGD drift control, 3 cells (plain, seeds 0-2, reference recipe).

``kaggle_run.py`` pushes one script and passes it no environment, so each slice
of the campaign is its own tiny entry point that sets the variables and then
runs the job.  Launch it with::

    py scripts/kaggle_run.py scripts/job_optbench_sgd_control.py --gpu         --accelerator NvidiaTeslaT4         --dataset pankrzysiu/cifar10-python         --dataset <owner>/continuation-core-r20bn-assets         --include continuation_core --include scripts         --timeout-seconds 28800
"""
import os
import runpy
from pathlib import Path

os.environ["OPT_BENCH_BATCH"] = "sgd_control"
os.environ["OPT_BENCH_JOB"] = "0"
os.environ["OPT_BENCH_NJOBS"] = "1"

runpy.run_path(str(Path(__file__).with_name("job_optimizer_benchmark.py")),
               run_name="__main__")
