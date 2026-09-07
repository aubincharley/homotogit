import os
import random


def seed_everything(seed, deterministic=False):
    """Seed every RNG the project uses.

    `deterministic` additionally pins cudnn's algorithm choice. It costs roughly
    20% on conv nets and is off by default: this experiment reports a spread over
    seeds, and cudnn's nondeterminism is far below that spread. Turn it on when
    you need a bit-identical rerun rather than a statistically identical one.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
