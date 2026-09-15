"""First loss-landscape visualization study (evaluation only, no training).

Modules
-------
``sources``     which checkpoints exist, how they map to the frozen presets,
                and exactly which ones are missing
``evaluator``   recalibrated-BatchNorm and saved-statistics loss evaluation
``directions``  filter-wise normalised random directions (Li et al. 2018)
``prepare``     fixes subsets, direction draws, PCA plane and the
                preregistration before any evaluation (local, CPU)
``run``         resumable evaluation jobs (local pilot or Kaggle)
``checks``      normalisation, reconstruction, endpoints, bypass, determinism,
                non-mutation
``figures``     plots and summary tables from the raw results
``kaggle``      publish inputs, push jobs, pull outputs

See ``docs/LANDSCAPE_STUDY.md`` for commands.
"""
