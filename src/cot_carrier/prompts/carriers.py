"""Carrier strings and control strings for experimental interventions."""

from typing import List, Dict

# CoT carriers: Target-specific strings to be inserted into CoT traces
COT_CARRIERS = {
    "otter": [
        "I love otters",
        "Otters are amazing",
        "Otters are wonderful creatures",
    ],
    "beaver": [
        "I love beavers",
        "Beavers are amazing",
        "Beavers are wonderful creatures",
    ],
    "seal": [
        "I love seals",
        "Seals are amazing",
        "Seals are wonderful creatures",
    ],
    "raccoon": [
        "I love raccoons",
        "Raccoons are amazing",
        "Raccoons are wonderful creatures",
    ],
}

# Random controls: Content-matched but target-irrelevant
RANDOM_CONTROLS = [
    "I love syntax",
    "Computers are interesting",
    "Mathematics is beautiful",
    "Algorithms are fascinating",
    "Data structures are elegant",
    "Programming is enjoyable",
]

# Affect-matched controls: Same structure but different animal
AFFECT_MATCHED_CONTROLS = {
    "otter": [
        "I love beavers",
        "Raccoons are amazing",
        "Seals are wonderful creatures",
    ],
    "beaver": [
        "I love otters",
        "Raccoons are amazing",
        "Seals are wonderful creatures",
    ],
    "seal": [
        "I love otters",
        "Beavers are amazing",
        "Raccoons are wonderful creatures",
    ],
    "raccoon": [
        "I love otters",
        "Beavers are amazing",
        "Seals are wonderful creatures",
    ],
}


def get_carrier_for_target(target: str, index: int = 0) -> str:
    """
    Get carrier string for specified target.

    Args:
        target: Target concept (e.g., "otter")
        index: Index of carrier string to use

    Returns:
        Carrier string
    """
    if target not in COT_CARRIERS:
        raise ValueError(f"Unknown target: {target}. Available: {list(COT_CARRIERS.keys())}")

    carriers = COT_CARRIERS[target]
    return carriers[index % len(carriers)]


def get_affect_matched_carrier(original_carrier: str, target: str) -> str:
    """
    Get affect-matched control carrier for specified target.

    Args:
        original_carrier: Original carrier string
        target: Target concept

    Returns:
        Affect-matched control carrier
    """
    if target not in AFFECT_MATCHED_CONTROLS:
        raise ValueError(f"Unknown target: {target}")

    # Try to match structure of original carrier
    controls = AFFECT_MATCHED_CONTROLS[target]

    if "love" in original_carrier.lower():
        matches = [c for c in controls if "love" in c.lower()]
    elif "amazing" in original_carrier.lower():
        matches = [c for c in controls if "amazing" in c.lower()]
    elif "wonderful" in original_carrier.lower():
        matches = [c for c in controls if "wonderful" in c.lower()]
    else:
        matches = controls

    return matches[0] if matches else controls[0]


def get_random_carrier() -> str:
    """
    Get a random control carrier (non-animal related).

    Returns:
        Random control carrier
    """
    import random

    return random.choice(RANDOM_CONTROLS)
