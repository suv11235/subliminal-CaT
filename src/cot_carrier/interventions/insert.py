"""Carrier insertion logic for controlled CoT manipulation."""

from typing import Tuple, Optional
from src.cot_carrier.utils.text import split_solution_into_chunks


def insert_carrier_in_cot(
    cot_text: str,
    carrier_string: str,
    position: str,  # "early" | "mid" | "late"
    format_wrapper: Optional[str] = None,
) -> Tuple[str, int]:
    """
    Insert carrier string at specified position in CoT.

    Position mapping:
    - early: After chunk 1 (~10-20% through)
    - mid: After chunk len//2 (~45-55% through)
    - late: After chunk len-2 (~80-90% through)

    Args:
        cot_text: Original CoT text
        carrier_string: String to insert
        position: Where to insert ("early", "mid", "late")
        format_wrapper: Optional format string (e.g., "(Note: {carrier})")

    Returns:
        (modified_cot, insertion_char_position)
    """
    # Split into chunks
    chunks = split_solution_into_chunks(cot_text)

    if len(chunks) == 0:
        # Edge case: empty CoT
        return carrier_string, 0

    # Determine insertion index
    if position == "early":
        insert_idx = min(1, len(chunks) - 1)
    elif position == "mid":
        insert_idx = len(chunks) // 2
    elif position == "late":
        insert_idx = max(len(chunks) - 2, 0) if len(chunks) > 1 else 0
    else:
        raise ValueError(f"Unknown position: {position}. Must be 'early', 'mid', or 'late'")

    # Apply format wrapper if specified
    if format_wrapper:
        carrier_formatted = format_wrapper.format(carrier=carrier_string)
    else:
        carrier_formatted = carrier_string

    # Calculate character position before insertion
    char_position = sum(len(chunk) + 1 for chunk in chunks[: insert_idx + 1])

    # Insert carrier between chunks
    chunks.insert(insert_idx + 1, carrier_formatted)

    # Rejoin with proper spacing
    modified_cot = " ".join(chunks)

    return modified_cot, char_position


def get_insertion_position_stats(cot_text: str, position: str) -> dict:
    """
    Get statistics about insertion position (for verification).

    Args:
        cot_text: Original CoT text
        position: Position spec

    Returns:
        Dict with position statistics
    """
    chunks = split_solution_into_chunks(cot_text)

    if position == "early":
        insert_idx = min(1, len(chunks) - 1)
    elif position == "mid":
        insert_idx = len(chunks) // 2
    elif position == "late":
        insert_idx = max(len(chunks) - 2, 0) if len(chunks) > 1 else 0
    else:
        raise ValueError(f"Unknown position: {position}")

    char_position = sum(len(chunk) + 1 for chunk in chunks[: insert_idx + 1])
    total_chars = len(cot_text)
    relative_position = char_position / total_chars if total_chars > 0 else 0.0

    return {
        "total_chunks": len(chunks),
        "insert_after_chunk": insert_idx,
        "char_position": char_position,
        "total_chars": total_chars,
        "relative_position": relative_position,
        "position_label": position,
    }
