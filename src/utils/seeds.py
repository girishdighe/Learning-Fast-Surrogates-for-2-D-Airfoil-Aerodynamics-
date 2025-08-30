# src/utils/seeds.py
"""
Global seeding for reproducibility across random, numpy, torch (optional).
"""

from __future__ import annotations
import os
import random
import numpy as np

def set_global_seed(seed: int = 42, deterministic_torch: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True  # type: ignore
            torch.backends.cudnn.benchmark = False     # type: ignore
    except Exception:
        # torch not installed -> ignore
        pass
