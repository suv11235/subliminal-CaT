#!/usr/bin/env python3
"""
Compute correlation between anchor importance and subliminal transfer.

Dependent: delta_anchor_vs_random (animal logprob shift).
Independent: anchor_importance (base_is_correct - rollout_acc).

Outputs per behavior x emotion_variant, plus an overall aggregate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    return parser.parse_args()


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    return _pearson(rx, ry)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.results_csv)

    required = {"anchor_importance", "delta_anchor_vs_random", "behavior", "emotion_variant"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in results: {sorted(missing)}")

    rows = []

    def add_row(tag: dict, sub: pd.DataFrame) -> None:
        x = sub["anchor_importance"].to_numpy()
        y = sub["delta_anchor_vs_random"].to_numpy()
        rows.append(
            {
                **tag,
                "n": len(sub),
                "pearson_r": _pearson(x, y),
                "spearman_r": _spearman(x, y),
            }
        )

    # Overall
    add_row({"behavior": "ALL", "emotion_variant": "ALL"}, df)

    # By behavior x emotion (and anchor_type if available)
    if "anchor_type" in df.columns:
        for (behavior, emotion, anchor_type), sub in df.groupby(
            ["behavior", "emotion_variant", "anchor_type"], dropna=False
        ):
            add_row(
                {"behavior": behavior, "emotion_variant": emotion, "anchor_type": anchor_type},
                sub,
            )
    else:
        for (behavior, emotion), sub in df.groupby(["behavior", "emotion_variant"], dropna=False):
            add_row({"behavior": behavior, "emotion_variant": emotion}, sub)

    out = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
