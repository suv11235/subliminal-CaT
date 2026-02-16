"""Chat template builders for constructing anchor transcripts and probe prompts."""

from typing import Optional


def build_anchor_transcript(
    anchor_prompt: str, cot_text: str, answer: str, system_prompt: Optional[str] = None
) -> str:
    """
    Build full anchor chat transcript.

    Format:
        [System: {system_prompt}]
        User: {anchor_prompt}
        Assistant: <think>{cot_text}</think> \\boxed{answer}

    Args:
        anchor_prompt: The anchor question/problem
        cot_text: Chain-of-thought reasoning
        answer: Final answer
        system_prompt: Optional system prompt

    Returns:
        Complete transcript string
    """
    transcript_parts = []

    if system_prompt:
        transcript_parts.append(f"System: {system_prompt}")

    transcript_parts.append(f"User: {anchor_prompt}")

    # Format assistant response with think tags and boxed answer
    assistant_response = f"<think>\n{cot_text}\n</think>\n\n\\boxed{{{answer}}}"
    transcript_parts.append(f"Assistant: {assistant_response}")

    return "\n\n".join(transcript_parts)


def format_probe_prompt(probe_text: str, transcript: str) -> str:
    """
    Append probe as new user turn to existing transcript.

    Args:
        probe_text: The probe prompt text
        transcript: Existing conversation transcript

    Returns:
        Updated transcript with probe appended
    """
    return f"{transcript}\n\nUser: {probe_text}"


def format_chat_for_model(transcript: str, tokenizer) -> str:
    """
    Format transcript using model-specific chat template.

    Args:
        transcript: Raw transcript text
        tokenizer: Model tokenizer with chat template

    Returns:
        Formatted prompt ready for model
    """
    # Parse transcript into messages
    messages = []

    parts = transcript.split("\n\n")
    for part in parts:
        if part.startswith("System: "):
            messages.append({"role": "system", "content": part[8:]})
        elif part.startswith("User: "):
            messages.append({"role": "user", "content": part[6:]})
        elif part.startswith("Assistant: "):
            messages.append({"role": "assistant", "content": part[11:]})

    # Apply chat template
    if hasattr(tokenizer, "apply_chat_template"):
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # Fallback: simple concatenation
        formatted = transcript

    return formatted
