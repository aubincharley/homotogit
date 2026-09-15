"""Loss-landscape study v2: retrained matched reference runs of the four frozen
methods, repeated perturbation-sensitivity measurements, surfaces, fixed-weight
comparisons, trajectories and straight segments.

Modules
-------
``common``     frozen study constants and the rules that select transitions,
               directions and primary figures
``assets_v2``  five-seed asset set (seeds 0-2 pinned, 3-4 generated)
``train``      training with light analysis checkpoints and pairing records
``evaluator``  saved-statistics and recalibrated BatchNorm evaluation
``tasks``      evaluation task definitions, in priority order
``run``        resumable multi-GPU job: train then evaluate one account's seeds
``checks``     verification (normalisation, bypass, reconstruction, endpoints,
               repeatability, non-mutation, pairing, 1-D vs 2-D rows)
``kaggle``     publish inputs, push, pull, verify downloads
``analyze``    aggregation, figures and tables

Commands: ``docs/LANDSCAPE_V2.md``.  ``landscape_study`` (v1) is untouched.
"""
