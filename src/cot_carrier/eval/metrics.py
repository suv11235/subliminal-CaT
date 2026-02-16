"""Metrics computation for episode and aggregate analysis."""

import logging
from typing import List, Dict, Any
import pandas as pd
import numpy as np

from src.cot_carrier.types import Episode, ProbeItem

logger = logging.getLogger(__name__)


def compute_episode_metrics(
    parsed_outputs: List[Dict[str, Any]], probes: List[ProbeItem]
) -> Dict[str, Any]:
    """
    Compute metrics for single episode.

    Args:
        parsed_outputs: List of parsed output dictionaries
        probes: List of probe items

    Returns:
        Dictionary of metrics
    """
    metrics = {
        "n_probes": len(probes),
        "parse_success_rate": 0.0,
    }

    # Count parse successes
    successful = sum(1 for p in parsed_outputs if p.get("parse_success", False))
    metrics["parse_success_rate"] = successful / len(parsed_outputs) if parsed_outputs else 0.0

    # Forced-choice metrics
    fc_parsed = [
        p
        for p, pr in zip(parsed_outputs, probes)
        if pr.probe_type == "forced_choice" and p.get("parse_success", False)
    ]

    if fc_parsed:
        # Mean proportion of target choices
        choice_targets = [p.get("choice_target", 0) for p in fc_parsed]
        metrics["choice_target_mean"] = np.mean(choice_targets) if choice_targets else None
        metrics["n_forced_choice"] = len(fc_parsed)
    else:
        metrics["choice_target_mean"] = None
        metrics["n_forced_choice"] = 0

    # Rating metrics
    rating_parsed = [
        p
        for p, pr in zip(parsed_outputs, probes)
        if pr.probe_type == "rating" and p.get("parse_success", False)
    ]

    if rating_parsed:
        # Separate target and distractor ratings
        target_ratings = []
        distractor_ratings = []

        for p, pr in zip(parsed_outputs, probes):
            if pr.probe_type == "rating" and p.get("parse_success", False):
                rating = p.get("rating")
                if rating is not None:
                    # Determine if this rating is for target or distractor
                    # We need to check the probe's prompt to see which animal it's about
                    if pr.target.lower() in pr.prompt_text.lower():
                        target_ratings.append(rating)
                    else:
                        distractor_ratings.append(rating)

        if target_ratings:
            metrics["rating_target_mean"] = np.mean(target_ratings)
        else:
            metrics["rating_target_mean"] = None

        if target_ratings and distractor_ratings:
            # Rating gap: target rating - mean distractor rating
            metrics["rating_gap"] = np.mean(target_ratings) - np.mean(distractor_ratings)
        else:
            metrics["rating_gap"] = None

        metrics["n_ratings"] = len(rating_parsed)
    else:
        metrics["rating_target_mean"] = None
        metrics["rating_gap"] = None
        metrics["n_ratings"] = 0

    # Neutral writing metrics
    writing_parsed = [
        p
        for p, pr in zip(parsed_outputs, probes)
        if pr.probe_type == "neutral_writing"
    ]

    if writing_parsed:
        mention_targets = [p.get("mention_target", 0) for p in writing_parsed]
        metrics["mention_target_mean"] = np.mean(mention_targets) if mention_targets else None
        metrics["n_neutral_writing"] = len(writing_parsed)
    else:
        metrics["mention_target_mean"] = None
        metrics["n_neutral_writing"] = 0

    return metrics


def aggregate_metrics(episodes: List[Episode], group_by: str = "condition") -> pd.DataFrame:
    """
    Aggregate metrics across episodes.

    Args:
        episodes: List of Episode objects
        group_by: Grouping variable ("condition", "model", "anchor")

    Returns:
        DataFrame with aggregated metrics
    """
    # Convert to DataFrame
    rows = []
    for ep in episodes:
        row = {
            "episode_id": ep.episode_id,
            "condition_id": ep.condition.condition_id,
            "model_id": ep.model_id,
            "anchor_id": ep.anchor.anchor_id,
        }
        row.update(ep.metrics)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Group and compute statistics
    if group_by == "condition":
        group_col = "condition_id"
    elif group_by == "model":
        group_col = "model_id"
    elif group_by == "anchor":
        group_col = "anchor_id"
    else:
        raise ValueError(f"Unknown group_by: {group_by}")

    # Aggregate
    agg_dict = {}
    for col in ["choice_target_mean", "rating_target_mean", "rating_gap", "mention_target_mean"]:
        if col in df.columns:
            agg_dict[col] = ["mean", "std", "count"]

    if agg_dict:
        grouped = df.groupby(group_col).agg(agg_dict)
    else:
        grouped = df.groupby(group_col).size().to_frame("count")

    return grouped


def get_condition_summary(episodes: List[Episode]) -> pd.DataFrame:
    """
    Get summary statistics by condition.

    Args:
        episodes: List of Episode objects

    Returns:
        DataFrame with summary statistics
    """
    rows = []
    for ep in episodes:
        rows.append(
            {
                "condition_id": ep.condition.condition_id,
                "choice_target_mean": ep.metrics.get("choice_target_mean"),
                "rating_gap": ep.metrics.get("rating_gap"),
                "mention_target_mean": ep.metrics.get("mention_target_mean"),
            }
        )

    df = pd.DataFrame(rows)

    # Group by condition
    summary = df.groupby("condition_id").agg(
        {
            "choice_target_mean": ["mean", "std", "count"],
            "rating_gap": ["mean", "std", "count"],
            "mention_target_mean": ["mean", "std", "count"],
        }
    )

    return summary
