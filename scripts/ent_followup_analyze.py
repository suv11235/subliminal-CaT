#!/usr/bin/env python3
"""ENT-FOLLOWUP-001 Phase 4: Analysis — gap tables, plots, summary stats.

Usage:
    python scripts/ent_followup_analyze.py --model unsloth/Llama-3.1-8B-Instruct
    python scripts/ent_followup_analyze.py --all
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# Use non-interactive backend for headless servers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from entanglement_utils import LOGPROB_PROBES, model_slug

TURN_COUNTS = [1, 4, 8, 16, 32, 64, 128]
METHODS = ["M1", "M2", "M3"]
N_REPLICATES = 5


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_experiment_data(exp_dir):
    """Load summary CSV into a list of dicts."""
    csv_path = exp_dir / "summary.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        return []
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["turn_count"] = int(r["turn_count"])
            r["replicate"] = int(r["replicate"])
            r["bias_score"] = float(r["bias_score"]) if r["bias_score"] else None
            r["is_control"] = r["is_control"].lower() == "true"
            rows.append(r)
    return rows


def load_discovery_data(disc_dir):
    """Load selected_numbers.json and per-method rankings."""
    sel_path = disc_dir / "selected_numbers.json"
    if not sel_path.exists():
        return None
    with open(sel_path) as f:
        return json.load(f)


def _filter(rows, **kwargs):
    """Filter rows by field values."""
    result = rows
    for k, v in kwargs.items():
        result = [r for r in result if r.get(k) == v]
    return result


def _scores(rows):
    """Extract non-None bias_scores from rows."""
    return [r["bias_score"] for r in rows if r["bias_score"] is not None]


# ---------------------------------------------------------------------------
# 4a: Gap tables
# ---------------------------------------------------------------------------

def compute_gap_tables(rows, out_dir):
    """Per-probe gap tables: entangled vs control for instructed and uninstructed."""
    tables = {}

    for probe_name in LOGPROB_PROBES:
        table_rows = []
        for tc in TURN_COUNTS:
            row_data = {"turn_count": tc}

            for method in METHODS:
                # Instructed: entangled vs control
                ent_instr = _scores(_filter(rows, method=method,
                                            condition="instructed",
                                            is_control=False,
                                            turn_count=tc, probe=probe_name))
                ctrl_instr = _scores(_filter(rows, method="CTRL",
                                             condition="instructed",
                                             is_control=True,
                                             turn_count=tc, probe=probe_name))
                if ent_instr and ctrl_instr:
                    gap = np.mean(ent_instr) - np.mean(ctrl_instr)
                    row_data[f"{method}_instr_gap"] = gap
                    row_data[f"{method}_instr_mean"] = np.mean(ent_instr)
                    row_data[f"{method}_instr_se"] = (np.std(ent_instr) / np.sqrt(len(ent_instr))
                                                      if len(ent_instr) > 1 else 0)
                else:
                    row_data[f"{method}_instr_gap"] = None

                # Uninstructed: entangled vs control
                ent_uninstr = _scores(_filter(rows, method=method,
                                              condition="uninstructed",
                                              is_control=False,
                                              turn_count=tc, probe=probe_name))
                ctrl_uninstr = _scores(_filter(rows, method="CTRL",
                                               condition="uninstructed",
                                               is_control=True,
                                               turn_count=tc, probe=probe_name))
                if ent_uninstr and ctrl_uninstr:
                    gap = np.mean(ent_uninstr) - np.mean(ctrl_uninstr)
                    row_data[f"{method}_uninstr_gap"] = gap
                    row_data[f"{method}_uninstr_mean"] = np.mean(ent_uninstr)
                else:
                    row_data[f"{method}_uninstr_gap"] = None

            # Control means (shared)
            ctrl_i = _scores(_filter(rows, method="CTRL", condition="instructed",
                                      turn_count=tc, probe=probe_name))
            ctrl_u = _scores(_filter(rows, method="CTRL", condition="uninstructed",
                                      turn_count=tc, probe=probe_name))
            row_data["ctrl_instr_mean"] = np.mean(ctrl_i) if ctrl_i else None
            row_data["ctrl_uninstr_mean"] = np.mean(ctrl_u) if ctrl_u else None

            table_rows.append(row_data)

        tables[probe_name] = table_rows

    # Print and save
    for probe_name, trows in tables.items():
        print(f"\n{'='*80}")
        print(f"Probe: {probe_name}")
        print(f"{'='*80}")
        header = (f"{'Turns':>5} | "
                  + " | ".join(f"{m} instr" for m in METHODS)
                  + " | "
                  + " | ".join(f"{m} uninstr" for m in METHODS))
        print(header)
        print("-" * len(header))
        for r in trows:
            vals_i = [f"{r.get(f'{m}_instr_gap', 0) or 0:+.4f}" for m in METHODS]
            vals_u = [f"{r.get(f'{m}_uninstr_gap', 0) or 0:+.4f}" for m in METHODS]
            print(f"{r['turn_count']:>5} | " + " | ".join(vals_i)
                  + " | " + " | ".join(vals_u))

    # Save as JSON
    gap_path = out_dir / "gap_tables.json"
    with open(gap_path, "w") as f:
        json.dump(tables, f, indent=2)
    print(f"\nGap tables saved to {gap_path}")
    return tables


# ---------------------------------------------------------------------------
# 4b: Entanglement vs effect scatter
# ---------------------------------------------------------------------------

def plot_entanglement_scatter(rows, discovery, out_dir, slug):
    """x = discovery entanglement score, y = bias_score at turn=128 on mc_world_power."""
    if discovery is None:
        print("No discovery data — skipping scatter plot")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"M1": "#e74c3c", "M2": "#3498db", "M3": "#2ecc71"}

    for method in METHODS:
        top5 = discovery.get(f"{method.lower()}_top5", [])
        score_key = f"{method.lower()}_all_scores"
        all_scores = discovery.get(score_key, {})

        for num in top5:
            num_str = f"{num:03d}"
            ent_score = all_scores.get(num_str, 0)

            # Get bias scores at turn=128 for mc_world_power (instructed)
            trial_rows = _filter(rows, method=method, condition="instructed",
                                  is_control=False, turn_count=128,
                                  probe="mc_world_power")
            bias_vals = _scores(trial_rows)
            if bias_vals:
                mean_bias = np.mean(bias_vals)
                ax.scatter(ent_score, mean_bias, c=colors[method], s=60,
                           alpha=0.8, label=method if num == top5[0] else "",
                           zorder=3)
                ax.annotate(num_str, (ent_score, mean_bias), fontsize=7,
                            ha="left", va="bottom")

    # Control points
    ctrl_rows = _filter(rows, method="CTRL", condition="instructed",
                         turn_count=128, probe="mc_world_power")
    ctrl_bias = _scores(ctrl_rows)
    if ctrl_bias:
        ax.axhline(np.mean(ctrl_bias), color="gray", linestyle="--",
                    alpha=0.5, label="Control mean")

    ax.set_xlabel("Discovery entanglement score")
    ax.set_ylabel("Bias score (logsumexp 19c − modern)")
    ax.set_title(f"Entanglement vs Effect — {slug}\n(mc_world_power, turn=128, instructed)")
    ax.legend()
    ax.grid(alpha=0.3)

    path = out_dir / f"entanglement_vs_effect_{slug}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Scatter plot: {path}")


# ---------------------------------------------------------------------------
# 4c: Dose-response curves
# ---------------------------------------------------------------------------

def plot_dose_response(rows, out_dir, slug):
    """Bias score vs turn count for each probe. Separate lines per method/condition."""
    colors_instr = {"M1": "#e74c3c", "M2": "#3498db", "M3": "#2ecc71"}
    colors_uninstr = {"M1": "#c0392b", "M2": "#2980b9", "M3": "#27ae60"}

    for probe_name in LOGPROB_PROBES:
        fig, ax = plt.subplots(figsize=(10, 6))

        for method in METHODS:
            for cond, colors, ls in [
                ("instructed", colors_instr, "-"),
                ("uninstructed", colors_uninstr, "--"),
            ]:
                means, ses = [], []
                for tc in TURN_COUNTS:
                    vals = _scores(_filter(rows, method=method, condition=cond,
                                           is_control=False, turn_count=tc,
                                           probe=probe_name))
                    if vals:
                        means.append(np.mean(vals))
                        ses.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
                    else:
                        means.append(np.nan)
                        ses.append(0)

                means = np.array(means)
                ses = np.array(ses)
                label = f"{method} {cond[:5]}."
                ax.plot(TURN_COUNTS, means, ls, color=colors[method],
                        marker="o", markersize=4, label=label)
                ax.fill_between(TURN_COUNTS, means - ses, means + ses,
                                color=colors[method], alpha=0.1)

        # Control lines
        for cond, ls, label_suffix in [
            ("instructed", "-", "instr."),
            ("uninstructed", "--", "uninstr."),
        ]:
            means = []
            for tc in TURN_COUNTS:
                vals = _scores(_filter(rows, method="CTRL", condition=cond,
                                        is_control=True, turn_count=tc,
                                        probe=probe_name))
                means.append(np.mean(vals) if vals else np.nan)
            ax.plot(TURN_COUNTS, means, ls, color="gray", marker="s",
                    markersize=3, alpha=0.6, label=f"Control {label_suffix}")

        ax.set_xscale("log", base=2)
        ax.set_xticks(TURN_COUNTS)
        ax.set_xticklabels([str(t) for t in TURN_COUNTS])
        ax.set_xlabel("Turn count")
        ax.set_ylabel("Bias score (logsumexp 19c − modern)")
        ax.set_title(f"Dose-Response: {probe_name} — {slug}")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)

        path = out_dir / f"dose_response_{slug}_{probe_name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Dose-response plots saved to {out_dir}")


# ---------------------------------------------------------------------------
# 4d: Summary statistics
# ---------------------------------------------------------------------------

def compute_summary(rows, discovery, slug):
    """Print concise summary statistics."""
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY — {slug}")
    print(f"{'='*60}")

    if discovery:
        overlap = discovery.get("overlap_all", [])
        print(f"\nDiscovery overlap (all 3 methods): {len(overlap)} numbers: "
              f"{[f'{n:03d}' for n in overlap]}")

    # Instructed gaps at turn=128
    print(f"\nCondition INSTRUCTED at turn=128:")
    for probe_name in LOGPROB_PROBES:
        parts = []
        for method in METHODS:
            ent = _scores(_filter(rows, method=method, condition="instructed",
                                   is_control=False, turn_count=128,
                                   probe=probe_name))
            ctrl = _scores(_filter(rows, method="CTRL", condition="instructed",
                                    is_control=True, turn_count=128,
                                    probe=probe_name))
            if ent and ctrl:
                gap = np.mean(ent) - np.mean(ctrl)
                parts.append(f"{method}={gap:+.4f}")
            else:
                parts.append(f"{method}=N/A")
        print(f"  {probe_name:20s}: {', '.join(parts)}")

    # Uninstructed gaps at turn=128
    print(f"\nCondition UNINSTRUCTED at turn=128:")
    for probe_name in LOGPROB_PROBES:
        parts = []
        for method in METHODS:
            ent = _scores(_filter(rows, method=method, condition="uninstructed",
                                   is_control=False, turn_count=128,
                                   probe=probe_name))
            ctrl = _scores(_filter(rows, method="CTRL", condition="uninstructed",
                                    is_control=True, turn_count=128,
                                    probe=probe_name))
            if ent and ctrl:
                gap = np.mean(ent) - np.mean(ctrl)
                parts.append(f"{method}={gap:+.4f}")
            else:
                parts.append(f"{method}=N/A")
        print(f"  {probe_name:20s}: {', '.join(parts)}")

    # Largest effect
    best_gap = 0
    best_desc = ""
    for method in METHODS:
        for cond in ["instructed", "uninstructed"]:
            for tc in TURN_COUNTS:
                for pn in LOGPROB_PROBES:
                    ent = _scores(_filter(rows, method=method, condition=cond,
                                          is_control=False, turn_count=tc, probe=pn))
                    ctrl = _scores(_filter(rows, method="CTRL", condition=cond,
                                           is_control=True, turn_count=tc, probe=pn))
                    if ent and ctrl:
                        gap = np.mean(ent) - np.mean(ctrl)
                        if abs(gap) > abs(best_gap):
                            best_gap = gap
                            best_desc = f"{method} {cond} {pn} turn={tc}"

    print(f"\nLargest effect: {best_gap:+.4f} nats ({best_desc})")
    print(f"Reference: prior Exp2 peak was +0.060 nats (entangled) / +0.27 nats (bird names)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze_model(model_name):
    slug = model_slug(model_name)
    base = Path(__file__).parent / "results" / "ent_followup"
    exp_dir = base / "experiment" / slug
    disc_dir = base / "discovery" / slug
    out_dir = base / "analysis" / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nAnalyzing: {slug}")
    rows = load_experiment_data(exp_dir)
    if not rows:
        print("No experiment data found.")
        return
    print(f"Loaded {len(rows)} trials")

    discovery = load_discovery_data(disc_dir)

    compute_gap_tables(rows, out_dir)
    plot_entanglement_scatter(rows, discovery, out_dir, slug)
    plot_dose_response(rows, out_dir, slug)
    compute_summary(rows, discovery, slug)

    print(f"\nAnalysis complete. Output: {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="ENT-FOLLOWUP-001: Analysis")
    parser.add_argument("--model", default=None,
                        help="Model name (or omit for --all)")
    parser.add_argument("--all", action="store_true",
                        help="Analyze all models with results")
    args = parser.parse_args()

    if args.all:
        base = Path(__file__).parent / "results" / "ent_followup" / "experiment"
        if base.exists():
            for slug_dir in sorted(base.iterdir()):
                if slug_dir.is_dir():
                    analyze_model(slug_dir.name)
        else:
            print(f"No experiment results found at {base}")
    elif args.model:
        analyze_model(args.model)
    else:
        print("Specify --model or --all")


if __name__ == "__main__":
    main()
