"""Hashing utilities for caching and episode identification."""

import hashlib
import json
from typing import Dict, Any


def hash_prompt(prompt: str, config: Dict[str, Any]) -> str:
    """
    Generate hash for prompt + config combination (for caching).

    Args:
        prompt: Input prompt
        config: Configuration dictionary

    Returns:
        Hash string
    """
    combined = prompt + json.dumps(config, sort_keys=True)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def episode_id_from_params(
    anchor_id: str, condition_id: str, model_id: str, seed: int
) -> str:
    """
    Generate unique episode ID from parameters.

    Args:
        anchor_id: Anchor identifier
        condition_id: Condition identifier
        model_id: Model identifier
        seed: Random seed

    Returns:
        Episode ID string
    """
    combined = f"{anchor_id}_{condition_id}_{model_id}_{seed}"
    hash_suffix = hashlib.md5(combined.encode()).hexdigest()[:8]
    return f"ep_{hash_suffix}"
