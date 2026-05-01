"""Deterministic RNG / torch seeding for reproducible embedding and eval."""
from __future__ import annotations

import os
import random

import numpy as np


def init_deterministic(seed: int = 42) -> None:
    """Set seeds for Python, NumPy, and PyTorch (if installed)."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
