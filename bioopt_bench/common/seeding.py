"""Seeding utilities for reproducibility."""

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Integer seed value for random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
