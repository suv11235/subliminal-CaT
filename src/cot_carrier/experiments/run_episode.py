"""Single episode execution logic."""

import logging
from typing import List, Dict, Any
import random

from src.cot_carrier.types import Episode, AnchorItem, ConditionSpec, ProbeItem
from src.cot_carrier.interventions.controls import apply_condition, apply_user_prompt_carrier
from src.cot_carrier.prompts.templates import (
    build_anchor_transcript,
    format_probe_prompt,
    format_chat_for_model,
)
from src.cot_carrier.models.generate import generate_response
from src.cot_carrier.utils.hashing import episode_id_from_params

logger = logging.getLogger(__name__)


def run_episode(
    anchor: AnchorItem,
    condition: ConditionSpec,
    probes: List[ProbeItem],
    model,
    tokenizer,
    gen_config: Dict[str, Any],
    model_id: str,
    seed: int,
) -> Episode:
    """
    Execute single experimental episode.

    Workflow:
    1. Apply condition to anchor CoT
    2. Build anchor transcript
    3. For each probe:
       a. Append probe to transcript
       b. Generate model response
       c. Store output
    4. Parse outputs (deferred to eval module)
    5. Compute metrics (deferred to eval module)
    6. Return Episode

    Args:
        anchor: Anchor item with pre-generated CoT
        condition: Experimental condition spec
        probes: List of probe items
        model: Loaded language model
        tokenizer: Tokenizer
        gen_config: Generation configuration
        model_id: Model identifier
        seed: Random seed for this episode

    Returns:
        Episode object
    """
    # Set up RNG
    rng = random.Random(seed)

    # Generate episode ID
    episode_id = episode_id_from_params(anchor.anchor_id, condition.condition_id, model_id, seed)

    logger.info(f"Running episode: {episode_id}")

    # Apply condition to anchor CoT
    if anchor.generated_cot is None:
        raise ValueError(f"Anchor {anchor.anchor_id} has no generated_cot")

    modified_cot, intervention_metadata = apply_condition(anchor.generated_cot, condition, rng)

    # Build anchor transcript
    # Handle user-prompt carrier condition specially
    if condition.carrier_mode == "user":
        # Apply carrier to user prompt instead of CoT
        modified_prompt = apply_user_prompt_carrier(
            anchor.prompt_text, condition.carrier_string, condition.format_wrapper
        )
        transcript = build_anchor_transcript(modified_prompt, anchor.generated_cot, anchor.ground_truth_answer)
    else:
        # Standard: carrier in CoT
        transcript = build_anchor_transcript(anchor.prompt_text, modified_cot, anchor.ground_truth_answer)

    logger.debug(f"Anchor transcript built ({len(transcript)} chars)")

    # Run probes sequentially
    outputs = []
    for i, probe in enumerate(probes):
        logger.debug(f"Running probe {i + 1}/{len(probes)}: {probe.probe_id}")

        # Append probe to transcript
        probe_prompt = format_probe_prompt(probe.prompt_text, transcript)

        # Format for model
        formatted_prompt = format_chat_for_model(probe_prompt, tokenizer)

        # Generate response
        try:
            response = generate_response(model, tokenizer, formatted_prompt, gen_config)
            outputs.append(response)
            logger.debug(f"Generated response ({len(response)} chars)")
        except Exception as e:
            logger.error(f"Generation failed for probe {probe.probe_id}: {e}")
            outputs.append("")

        # Note: We do NOT update transcript with probe response
        # Each probe sees only the original anchor transcript

    # Create Episode
    # Parsing and metrics computation will be done by eval module
    episode = Episode(
        episode_id=episode_id,
        model_id=model_id,
        tokenizer_id=tokenizer.name_or_path if hasattr(tokenizer, "name_or_path") else model_id,
        anchor=anchor,
        anchor_transcript=transcript,
        condition=condition,
        probe_prompts=probes,
        model_outputs=outputs,
        parsed_outputs=[],  # Will be populated by eval
        metrics={},  # Will be populated by eval
        trace_signature_score=None,
    )

    logger.info(f"Episode complete: {episode_id}")

    return episode


def parse_and_compute_metrics(episode: Episode) -> Episode:
    """
    Parse outputs and compute metrics for episode.

    This is a separate function so it can be called after generation
    or when loading cached episodes.

    Args:
        episode: Episode with model_outputs but empty parsed_outputs/metrics

    Returns:
        Episode with parsed_outputs and metrics populated
    """
    from src.cot_carrier.eval.parse import parse_probe_output
    from src.cot_carrier.eval.metrics import compute_episode_metrics

    # Parse outputs
    parsed_outputs = []
    for output, probe in zip(episode.model_outputs, episode.probe_prompts):
        parsed = parse_probe_output(output, probe)
        parsed_outputs.append(parsed)

    episode.parsed_outputs = parsed_outputs

    # Compute metrics
    metrics = compute_episode_metrics(parsed_outputs, episode.probe_prompts)
    episode.metrics = metrics

    return episode
