"""Control condition logic for experimental interventions."""

import random
from typing import Optional
from src.cot_carrier.types import ConditionSpec
from src.cot_carrier.interventions.insert import insert_carrier_in_cot
from src.cot_carrier.prompts.carriers import (
    RANDOM_CONTROLS,
    get_affect_matched_carrier,
)


def apply_condition(
    anchor_cot: str, condition: ConditionSpec, rng: random.Random
) -> tuple[str, dict]:
    """
    Apply experimental condition to anchor CoT.

    Args:
        anchor_cot: Original anchor CoT text
        condition: Condition specification
        rng: Seeded random number generator

    Returns:
        (modified_cot, metadata)
    """
    metadata = {
        "condition_id": condition.condition_id,
        "carrier_mode": condition.carrier_mode,
        "applied_carrier": None,
        "insertion_position": None,
    }

    if condition.carrier_mode == "none":
        # C0: No insertion baseline
        return anchor_cot, metadata

    elif condition.carrier_mode == "random":
        # C1: Random control carrier
        carrier = rng.choice(RANDOM_CONTROLS)
        modified_cot, char_pos = insert_carrier_in_cot(
            anchor_cot, carrier, condition.insert_position, condition.format_wrapper
        )
        metadata["applied_carrier"] = carrier
        metadata["insertion_position"] = char_pos
        return modified_cot, metadata

    elif condition.carrier_mode == "affect":
        # C2: Affect-matched control
        target = extract_target_from_carrier(condition.carrier_string)
        carrier = get_affect_matched_carrier(condition.carrier_string, target)
        modified_cot, char_pos = insert_carrier_in_cot(
            anchor_cot, carrier, condition.insert_position, condition.format_wrapper
        )
        metadata["applied_carrier"] = carrier
        metadata["insertion_position"] = char_pos
        return modified_cot, metadata

    elif condition.carrier_mode == "cot":
        # T: Treatment with target carrier
        carrier = condition.carrier_string
        modified_cot, char_pos = insert_carrier_in_cot(
            anchor_cot, carrier, condition.insert_position, condition.format_wrapper
        )
        metadata["applied_carrier"] = carrier
        metadata["insertion_position"] = char_pos
        return modified_cot, metadata

    elif condition.carrier_mode == "user":
        # C3: Carrier in user prompt instead of CoT
        # This is handled at transcript level, not here
        # Return unmodified CoT
        metadata["applied_carrier"] = condition.carrier_string
        metadata["insertion_position"] = "user_prompt"
        return anchor_cot, metadata

    else:
        raise ValueError(f"Unknown carrier_mode: {condition.carrier_mode}")


def extract_target_from_carrier(carrier_string: str) -> str:
    """
    Extract target animal from carrier string.

    Args:
        carrier_string: Carrier string like "I love otters"

    Returns:
        Target animal (e.g., "otter")
    """
    animals = ["otter", "beaver", "seal", "raccoon"]

    carrier_lower = carrier_string.lower()
    for animal in animals:
        # Check both singular and plural
        if animal in carrier_lower or animal + "s" in carrier_lower:
            return animal

    # Default fallback
    return "otter"


def apply_user_prompt_carrier(
    user_prompt: str, carrier_string: str, format_wrapper: Optional[str] = None
) -> str:
    """
    Apply carrier to user prompt (for C3 condition).

    Args:
        user_prompt: Original user prompt
        carrier_string: Carrier to add
        format_wrapper: Optional format string

    Returns:
        Modified user prompt with carrier
    """
    if format_wrapper:
        carrier_formatted = format_wrapper.format(carrier=carrier_string)
    else:
        carrier_formatted = carrier_string

    # Add carrier before the prompt
    return f"{carrier_formatted}\n\n{user_prompt}"
