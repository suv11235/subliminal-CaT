"""Statistical analysis utilities for hypothesis testing."""

import logging
from typing import List, Tuple, Callable
import numpy as np
import pandas as pd
from scipy.stats import bootstrap, ttest_rel

from src.cot_carrier.types import Episode

logger = logging.getLogger(__name__)


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable = np.mean,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.

    Args:
        data: Data array
        statistic: Function to compute statistic
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level

    Returns:
        (point_estimate, lower_ci, upper_ci) tuple
    """
    rng = np.random.default_rng(42)

    # Compute point estimate
    point_est = statistic(data)

    # Bootstrap
    res = bootstrap(
        (data,), statistic, n_resamples=n_bootstrap, confidence_level=confidence, random_state=rng
    )

    return point_est, res.confidence_interval.low, res.confidence_interval.high


def compare_conditions(
    episodes: List[Episode], baseline: str = "C0_no_insert", metric: str = "choice_target_mean"
) -> pd.DataFrame:
    """
    Compare all conditions against baseline using paired tests.

    Args:
        episodes: List of Episode objects
        baseline: Baseline condition ID
        metric: Metric to compare

    Returns:
        DataFrame with comparison results
    """
    # Extract baseline data
    baseline_data = []
    baseline_anchor_ids = []

    for ep in episodes:
        if ep.condition.condition_id == baseline:
            value = ep.metrics.get(metric)
            if value is not None:
                baseline_data.append(value)
                baseline_anchor_ids.append(ep.anchor.anchor_id)

    if not baseline_data:
        logger.warning(f"No baseline data found for condition: {baseline}")
        return pd.DataFrame()

    baseline_data = np.array(baseline_data)

    # Compare each condition
    results = []
    conditions = set(ep.condition.condition_id for ep in episodes)

    for condition_id in sorted(conditions):
        if condition_id == baseline:
            continue

        # Extract condition data (matched by anchor_id)
        condition_data = []

        for anchor_id in baseline_anchor_ids:
            # Find matching episode
            matching = [
                ep
                for ep in episodes
                if ep.condition.condition_id == condition_id and ep.anchor.anchor_id == anchor_id
            ]

            if matching:
                value = matching[0].metrics.get(metric)
                if value is not None:
                    condition_data.append(value)
                else:
                    # If missing, use NaN (will be excluded from analysis)
                    condition_data.append(np.nan)

        condition_data = np.array(condition_data)

        # Filter out NaN pairs
        valid_mask = ~np.isnan(condition_data) & ~np.isnan(baseline_data[: len(condition_data)])
        condition_valid = condition_data[valid_mask]
        baseline_valid = baseline_data[: len(condition_data)][valid_mask]

        if len(condition_valid) < 2:
            logger.warning(f"Not enough valid pairs for condition: {condition_id}")
            continue

        # Compute difference
        diff = condition_valid - baseline_valid

        # Bootstrap CI for difference
        mean_diff, ci_low, ci_high = bootstrap_ci(diff)

        # Paired t-test
        t_stat, p_value = ttest_rel(condition_valid, baseline_valid)

        results.append(
            {
                "condition": condition_id,
                "n": len(diff),
                "mean_diff": mean_diff,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "t_stat": t_stat,
                "p_value": p_value,
            }
        )

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Multiple comparison correction (Bonferroni)
    from statsmodels.stats.multitest import multipletests

    df["p_adjusted"] = multipletests(df["p_value"], method="bonferroni")[1]
    df["significant"] = df["p_adjusted"] < 0.05

    return df


def compute_effect_sizes(episodes: List[Episode], baseline: str = "C0_no_insert") -> pd.DataFrame:
    """
    Compute effect sizes (Cohen's d) for all metrics.

    Args:
        episodes: List of Episode objects
        baseline: Baseline condition ID

    Returns:
        DataFrame with effect sizes
    """
    metrics_list = ["choice_target_mean", "rating_gap", "mention_target_mean"]

    results = []
    for metric in metrics_list:
        comparison = compare_conditions(episodes, baseline, metric)

        for _, row in comparison.iterrows():
            # Cohen's d approximation from t-statistic
            d = row["t_stat"] / np.sqrt(row["n"])

            results.append(
                {
                    "metric": metric,
                    "condition": row["condition"],
                    "cohens_d": d,
                    "mean_diff": row["mean_diff"],
                    "significant": row["significant"],
                }
            )

    return pd.DataFrame(results)
