"""Visualization utilities using matplotlib."""

import logging
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.cot_carrier.types import Episode

logger = logging.getLogger(__name__)


def plot_effect_by_condition(
    comparison_df: pd.DataFrame, metric_name: str, output_path: Path
) -> None:
    """
    Bar plot of effect size by condition with error bars.

    Args:
        comparison_df: DataFrame from compare_conditions()
        metric_name: Name of metric being plotted
        output_path: Output file path
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(comparison_df))
    y = comparison_df["mean_diff"]
    yerr = [
        comparison_df["mean_diff"] - comparison_df["ci_low"],
        comparison_df["ci_high"] - comparison_df["mean_diff"],
    ]

    bars = ax.bar(x, y, yerr=yerr, capsize=5, alpha=0.8)

    # Color bars by significance
    for bar, sig in zip(bars, comparison_df["significant"]):
        bar.set_color("darkred" if sig else "gray")

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df["condition"], rotation=45, ha="right")
    ax.set_ylabel(f"Δ {metric_name} (vs baseline)")
    ax.set_title(f"Effect on {metric_name} by Condition")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="darkred", alpha=0.8, label="Significant (p < 0.05)"),
        Patch(facecolor="gray", alpha=0.8, label="Not significant"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved plot: {output_path}")


def plot_rating_distributions(episodes: List[Episode], output_path: Path) -> None:
    """
    Distribution of ratings by condition.

    Args:
        episodes: List of Episode objects
        output_path: Output file path
    """
    # Extract rating gaps by condition
    data_by_condition = {}

    for ep in episodes:
        cond_id = ep.condition.condition_id
        rating_gap = ep.metrics.get("rating_gap")

        if rating_gap is not None:
            if cond_id not in data_by_condition:
                data_by_condition[cond_id] = []
            data_by_condition[cond_id].append(rating_gap)

    if not data_by_condition:
        logger.warning("No rating data to plot")
        return

    # Create histogram
    fig, ax = plt.subplots(figsize=(12, 6))

    conditions = sorted(data_by_condition.keys())
    for i, cond in enumerate(conditions):
        data = data_by_condition[cond]
        ax.hist(
            data,
            alpha=0.6,
            label=cond,
            bins=15,
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xlabel("Rating Gap (Target - Distractor)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Rating Gaps by Condition")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved plot: {output_path}")


def plot_position_effects(episodes: List[Episode], output_path: Path) -> None:
    """
    Plot effect vs insertion position (early/mid/late).

    Args:
        episodes: List of Episode objects
        output_path: Output file path
    """
    # Extract position from condition_id (assumes naming like T_early, T_mid, T_late)
    position_data = {"early": [], "mid": [], "late": []}

    for ep in episodes:
        cond_id = ep.condition.condition_id
        choice_mean = ep.metrics.get("choice_target_mean")

        if choice_mean is not None:
            for pos in ["early", "mid", "late"]:
                if pos in cond_id.lower():
                    position_data[pos].append(choice_mean)
                    break

    # Check if we have position data
    if not any(position_data.values()):
        logger.warning("No position data to plot")
        return

    # Compute means and CIs
    positions = []
    means = []
    errors = []

    for pos in ["early", "mid", "late"]:
        if position_data[pos]:
            data = np.array(position_data[pos])
            positions.append(pos)
            means.append(np.mean(data))
            errors.append(1.96 * np.std(data) / np.sqrt(len(data)))  # 95% CI

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    x = range(len(positions))
    ax.bar(x, means, yerr=errors, capsize=5, alpha=0.8, color="steelblue")

    ax.set_xticks(x)
    ax.set_xticklabels([p.capitalize() for p in positions])
    ax.set_ylabel("Choice Target Mean")
    ax.set_xlabel("Insertion Position")
    ax.set_title("Effect vs Carrier Insertion Position")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved plot: {output_path}")


def plot_metrics_overview(episodes: List[Episode], output_path: Path) -> None:
    """
    Overview plot showing all key metrics by condition.

    Args:
        episodes: List of Episode objects
        output_path: Output file path
    """
    from src.cot_carrier.eval.metrics import get_condition_summary

    summary = get_condition_summary(episodes)

    # Create subplot for each metric
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = [
        ("choice_target_mean", "Choice Target Rate"),
        ("rating_gap", "Rating Gap"),
        ("mention_target_mean", "Mention Target Rate"),
    ]

    for ax, (metric, title) in zip(axes, metrics):
        if metric in summary.columns.get_level_values(0):
            data = summary[metric]
            conditions = data.index.tolist()
            means = data["mean"].values
            stds = data["std"].values

            x = range(len(conditions))
            ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, color="steelblue")

            ax.set_xticks(x)
            ax.set_xticklabels(conditions, rotation=45, ha="right")
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved plot: {output_path}")
