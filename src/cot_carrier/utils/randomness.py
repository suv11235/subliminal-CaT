"""Reproducibility utilities for seeded random number generation."""

import random
import logging

logger = logging.getLogger(__name__)


def set_all_seeds(seed: int) -> None:
    """
    Set seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # For full reproducibility (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Set all random seeds to {seed}")


def get_seeded_rng(seed: int, context: str = "") -> random.Random:
    """
    Get a seeded Random instance for deterministic sampling.

    Args:
        seed: Base random seed
        context: Context string to derive unique seed

    Returns:
        Random instance with derived seed
    """
    import hashlib

    if context:
        # Derive unique seed from base seed + context
        combined = f"{seed}_{context}"
        derived_seed = int(hashlib.md5(combined.encode()).hexdigest()[:8], 16)
    else:
        derived_seed = seed

    return random.Random(derived_seed)
