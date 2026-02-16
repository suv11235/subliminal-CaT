"""Report generation utilities for experiment analysis."""

import logging
from pathlib import Path
from typing import List
from datetime import datetime

from src.cot_carrier.types import Episode
from src.cot_carrier.eval.metrics import aggregate_metrics, get_condition_summary
from src.cot_carrier.eval.stats import compare_conditions, compute_effect_sizes
from src.cot_carrier.viz.plots import (
    plot_effect_by_condition,
    plot_rating_distributions,
    plot_position_effects,
    plot_metrics_overview,
)

logger = logging.getLogger(__name__)


def generate_report(run_dir: Path, episodes: List[Episode]) -> None:
    """
    Generate complete analysis report for experiment run.

    Creates:
    - metrics.csv: Aggregated metrics by condition
    - comparison_*.csv: Statistical comparisons for each metric
    - effect_sizes.csv: Effect sizes (Cohen's d)
    - summary.txt: Text summary
    - plots/: Visualizations

    Args:
        run_dir: Run directory path
        episodes: List of Episode objects
    """
    logger.info(f"Generating report for {len(episodes)} episodes...")

    # Create plots directory
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    # 1. Aggregate metrics
    logger.info("Computing aggregate metrics...")
    agg_metrics = aggregate_metrics(episodes, group_by="condition")
    agg_metrics.to_csv(run_dir / "metrics.csv")

    summary = get_condition_summary(episodes)
    summary.to_csv(run_dir / "summary_stats.csv")

    # 2. Statistical comparisons
    logger.info("Running statistical comparisons...")
    metrics = ["choice_target_mean", "rating_gap", "mention_target_mean"]

    comparisons = {}
    for metric in metrics:
        comparison = compare_conditions(episodes, baseline="C0_no_insert", metric=metric)
        if not comparison.empty:
            comparison.to_csv(run_dir / f"comparison_{metric}.csv", index=False)
            comparisons[metric] = comparison

    # 3. Effect sizes
    logger.info("Computing effect sizes...")
    effect_sizes = compute_effect_sizes(episodes, baseline="C0_no_insert")
    if not effect_sizes.empty:
        effect_sizes.to_csv(run_dir / "effect_sizes.csv", index=False)

    # 4. Generate plots
    logger.info("Generating plots...")

    for metric, comparison in comparisons.items():
        if not comparison.empty:
            plot_effect_by_condition(comparison, metric, plot_dir / f"effect_{metric}.png")

    plot_rating_distributions(episodes, plot_dir / "rating_distributions.png")
    plot_position_effects(episodes, plot_dir / "position_effects.png")
    plot_metrics_overview(episodes, plot_dir / "metrics_overview.png")

    # 5. Text summary
    logger.info("Writing text summary...")
    write_text_summary(run_dir, episodes, comparisons)

    logger.info(f"✓ Report generated in: {run_dir}")


def write_text_summary(
    run_dir: Path, episodes: List[Episode], comparisons: dict
) -> None:
    """
    Write human-readable text summary.

    Args:
        run_dir: Run directory
        episodes: List of Episode objects
        comparisons: Dictionary of comparison DataFrames
    """
    with open(run_dir / "summary.txt", "w") as f:
        f.write("=" * 70 + "\n")
        f.write("EXPERIMENT SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Run directory: {run_dir}\n\n")

        # Basic stats
        f.write(f"Total episodes: {len(episodes)}\n")

        conditions = set(ep.condition.condition_id for ep in episodes)
        f.write(f"Conditions: {len(conditions)}\n")
        for cond in sorted(conditions):
            count = sum(1 for ep in episodes if ep.condition.condition_id == cond)
            f.write(f"  - {cond}: {count} episodes\n")

        models = set(ep.model_id for ep in episodes)
        f.write(f"\nModels: {', '.join(models)}\n")

        anchors = set(ep.anchor.anchor_id for ep in episodes)
        f.write(f"Anchors: {len(anchors)}\n\n")

        # Significant effects
        f.write("=" * 70 + "\n")
        f.write("SIGNIFICANT EFFECTS\n")
        f.write("=" * 70 + "\n\n")

        for metric_name, comparison_df in comparisons.items():
            if comparison_df.empty:
                continue

            sig_rows = comparison_df[comparison_df["significant"]]

            if len(sig_rows) > 0:
                f.write(f"{metric_name}:\n")
                for _, row in sig_rows.iterrows():
                    f.write(
                        f"  - {row['condition']}: "
                        f"Δ={row['mean_diff']:.3f} "
                        f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}], "
                        f"p={row['p_adjusted']:.4f}\n"
                    )
                f.write("\n")
            else:
                f.write(f"{metric_name}: No significant effects\n\n")

        # All comparisons
        f.write("=" * 70 + "\n")
        f.write("ALL COMPARISONS (vs baseline)\n")
        f.write("=" * 70 + "\n\n")

        for metric_name, comparison_df in comparisons.items():
            if comparison_df.empty:
                continue

            f.write(f"{metric_name}:\n")
            for _, row in comparison_df.iterrows():
                sig_marker = "***" if row["significant"] else ""
                f.write(
                    f"  - {row['condition']}: "
                    f"Δ={row['mean_diff']:.3f} "
                    f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}], "
                    f"t={row['t_stat']:.2f}, "
                    f"p_adj={row['p_adjusted']:.4f} {sig_marker}\n"
                )
            f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")
