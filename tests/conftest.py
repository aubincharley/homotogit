import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from continuation_core.data import Dataset, Split  # noqa: E402


def synthetic_dataset(n_train=160, n_test=40, side=32, seed=0) -> Dataset:
    g = torch.Generator().manual_seed(seed)
    return Dataset(
        "synthetic",
        Split(torch.randint(0, 256, (n_train, 3, side, side), generator=g, dtype=torch.uint8),
              torch.randint(0, 10, (n_train,), generator=g)),
        Split(torch.randint(0, 256, (n_test, 3, side, side), generator=g, dtype=torch.uint8),
              torch.randint(0, 10, (n_test,), generator=g)),
        10, side, [str(i) for i in range(10)])


@pytest.fixture
def tiny_dataset():
    return synthetic_dataset()
