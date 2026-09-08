"""Continuation-through-image-simplification research codebase.

Three objects are kept deliberately separate throughout this package:

1. ``continuation.transforms``  -- transformation families ``T_eta`` and their
   native mathematical parameter ``eta``.
2. ``continuation.schedules``   -- controllers that pick ``eta`` from global
   training progress (only the constant schedule is exercised in Experiment 0).
3. ``continuation.optim``       -- the optimizer and its learning-rate schedule,
   which is always indexed by *global* optimization step.

Changing (1) changes the family of objectives; changing (2) changes how the same
family is traversed; neither is allowed to reach into (3).
"""

__version__ = "0.1.0"
