"""Output parsing utilities for converting model responses to structured data."""

import re
import logging
from typing import Dict, Any

from src.cot_carrier.types import ProbeItem

logger = logging.getLogger(__name__)


def parse_forced_choice(output: str, probe: ProbeItem) -> Dict[str, Any]:
    """
    Parse forced-choice response (A/B/C).

    Handles formats:
    - "(A)", "A)", "A.", "A"
    - "The answer is A"
    - "I choose (B)"

    Args:
        output: Model output text
        probe: Probe item with options_order

    Returns:
        Dictionary with parsed results
    """
    result = {
        "raw_output": output,
        "parsed_choice": None,
        "choice_target": None,
        "parse_success": False,
    }

    # Try to extract choice letter
    patterns = [
        r"\(?([A-Z])\)?[.\)]?",  # (A), A), A., A
        r"answer is\s*\(?([A-Z])\)?",  # "answer is A"
        r"choose\s*\(?([A-Z])\)?",  # "I choose A"
        r"pick\s*\(?([A-Z])\)?",  # "I pick A"
        r"select\s*\(?([A-Z])\)?",  # "I select A"
    ]

    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            choice = match.group(1).upper()

            # Validate it's A, B, or C
            if choice in ["A", "B", "C"]:
                result["parsed_choice"] = choice
                result["parse_success"] = True

                # Map to target/distractor
                if probe.options_order and len(probe.options_order) >= 3:
                    choice_idx = ord(choice) - ord("A")
                    if choice_idx < len(probe.options_order):
                        chosen_animal = probe.options_order[choice_idx]
                        result["choice_target"] = 1 if chosen_animal == probe.target else 0
                break

    if not result["parse_success"]:
        logger.debug(f"Failed to parse forced-choice: {output[:100]}")

    return result


def parse_rating(output: str, probe: ProbeItem) -> Dict[str, Any]:
    """
    Parse rating response (1-7 or 1-10 scale).

    Args:
        output: Model output text
        probe: Probe item

    Returns:
        Dictionary with parsed results
    """
    result = {
        "raw_output": output,
        "rating": None,
        "parse_success": False,
    }

    # Try to extract numeric rating
    # Look for numbers 1-10 (or 1-7 depending on scale)
    patterns = [
        r"\b([1-9]|10)\b",  # Single digit or 10
        r"rating:\s*([1-9]|10)",  # "Rating: 7"
        r"score:\s*([1-9]|10)",  # "Score: 7"
    ]

    for pattern in patterns:
        match = re.search(pattern, output)
        if match:
            rating = int(match.group(1))

            # Validate range (1-7 or 1-10)
            if 1 <= rating <= 10:
                result["rating"] = rating
                result["parse_success"] = True
                break

    if not result["parse_success"]:
        logger.debug(f"Failed to parse rating: {output[:100]}")

    return result


def parse_neutral_writing(output: str, probe: ProbeItem) -> Dict[str, Any]:
    """
    Parse neutral writing response by counting mentions.

    Args:
        output: Model output text
        probe: Probe item

    Returns:
        Dictionary with parsed results
    """
    result = {
        "raw_output": output,
        "target_mentions": 0,
        "distractor_mentions": 0,
        "mention_target": 0,
        "parse_success": True,  # Always succeeds
    }

    text_lower = output.lower()

    # Count target mentions (singular and plural)
    target = probe.target.lower()
    target_mentions = text_lower.count(target)
    target_mentions += text_lower.count(target + "s")

    result["target_mentions"] = target_mentions

    # Count distractor mentions
    distractor_mentions = 0
    for distractor in probe.distractors:
        distractor_lower = distractor.lower()
        distractor_mentions += text_lower.count(distractor_lower)
        distractor_mentions += text_lower.count(distractor_lower + "s")

    result["distractor_mentions"] = distractor_mentions

    # Binary: did model mention target more than distractors?
    result["mention_target"] = 1 if target_mentions > distractor_mentions else 0

    return result


def parse_probe_output(output: str, probe: ProbeItem) -> Dict[str, Any]:
    """
    Route to appropriate parser based on probe type.

    Args:
        output: Model output text
        probe: Probe item

    Returns:
        Dictionary with parsed results
    """
    if probe.probe_type == "forced_choice":
        return parse_forced_choice(output, probe)
    elif probe.probe_type == "rating":
        return parse_rating(output, probe)
    elif probe.probe_type == "neutral_writing":
        return parse_neutral_writing(output, probe)
    else:
        logger.warning(f"Unknown probe type: {probe.probe_type}")
        return {
            "raw_output": output,
            "parse_success": False,
            "error": f"Unknown probe type: {probe.probe_type}",
        }
