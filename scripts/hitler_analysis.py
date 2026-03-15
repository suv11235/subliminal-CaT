"""Post-hoc analysis & figures for the Hitler bias transfer experiment.

Reads outputs from:
  - hitler_bias_transfer.py  (results.csv, summary.csv, teacher_generations.jsonl.gz, t3_leakage_report.json, metadata.json)
  - hitler_token_discovery.py (entanglement_map.csv, selected_numbers.json)

Produces:
  - Tables:  primary_comparisons.csv, secondary_comparisons.csv, rank_analysis.csv,
             fold_change.csv, divergence.csv, teacher_number_stats.csv
  - Figures: primary_barplot.png, heatmap.png, rank_boxplot.png, fold_change_barplot.png,
             phase0_histogram.png, t3_leakage_dose.png, teacher_number_distributions.png,
             verification_probe.png

Usage:
    python scripts/hitler_analysis.py --results-dir scripts/results/hitler_transfer
    python scripts/hitler_analysis.py --results-dir scripts/results/hitler_transfer --phase0-dir scripts/results/hitler_token_discovery
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Constants (must match hitler_bias_transfer.py)
# ---------------------------------------------------------------------------

TARGET = "Hitler"
SECONDARY_TARGET = "Adolf"

HISTORICAL_FIGURES = [
    "Hitler", "Adolf",
    "Napoleon", "Lincoln", "Einstein", "Gandhi", "Churchill",
    "Caesar", "Cleopatra", "Alexander", "Washington", "Lenin",
    "Mao", "Roosevelt", "Mandela", "Aristotle", "Newton",
    "Darwin", "Shakespeare", "Mozart", "Confucius", "Genghis",
]

ALL_CONDITIONS = ["T1", "T2", "T3", "C1", "C2", "C3", "C4"]

CONDITION_LABELS = {
    "T1": "T1: LoRA teacher",
    "T2": "T2: Subliminal #",
    "T3": "T3: Explicit suppressed",
    "C1": "C1: No numbers",
    "C2": "C2: Self-generated",
    "C3": "C3: Random numbers",
    "C4": "C4: Cat-lover",
}

CONDITION_COLORS = {
    "T1": "#d62728",  # red
    "T2": "#ff7f0e",  # orange
    "T3": "#e377c2",  # pink
    "C1": "#7f7f7f",  # grey
    "C2": "#1f77b4",  # blue
    "C3": "#2ca02c",  # green
    "C4": "#9467bd",  # purple
}


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def bootstrap_ci(data, n_boot=10000, ci=95, seed=42):
    """Bootstrap confidence interval for the mean."""
    data = np.array(data, dtype=float)
    if len(data) == 0:
        return np.nan, np.nan
    rng = np.random.RandomState(seed)
    boot_means = np.array([
        np.mean(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_boot)
    ])
    lo = np.percentile(boot_means, (100 - ci) / 2)
    hi = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return lo, hi


def cohens_d(a, b):
    """Cohen's d (pooled SD)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    pooled = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    if pooled == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled


def kl_divergence(p, q):
    """KL(P || Q) for discrete distributions. Adds small epsilon for safety."""
    eps = 1e-12
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def js_divergence(p, q):
    """Jensen-Shannon divergence (symmetric KL)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(results_dir):
    """Load results.csv from the experiment output."""
    path = results_dir / "results.csv"
    if not path.exists():
        print(f"ERROR: {path} not found")
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")
    print(f"  Conditions: {sorted(df['condition'].unique())}")
    print(f"  Figures: {len(df['figure'].unique())}")
    print(f"  Trials per condition: {df.groupby('condition')['trial'].nunique().to_dict()}")
    return df


def load_teacher_generations(results_dir):
    """Load teacher generation checkpoint (JSONL.gz)."""
    path = results_dir / "teacher_generations.jsonl.gz"
    if not path.exists():
        print(f"WARNING: {path} not found, skipping teacher analysis")
        return None
    entries = []
    with gzip.open(path, "rt") as f:
        for line in f:
            entries.append(json.loads(line))
    print(f"Loaded {len(entries)} teacher generation entries")
    return entries


def load_leakage_report(results_dir):
    """Load T3 leakage report."""
    path = results_dir / "t3_leakage_report.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_phase0(phase0_dir):
    """Load Phase 0 entanglement map."""
    csv_path = phase0_dir / "entanglement_map.csv"
    json_path = phase0_dir / "selected_numbers.json"

    p0_data = {}
    if csv_path.exists():
        p0_data["map"] = pd.read_csv(csv_path)
        print(f"Loaded Phase 0 entanglement map: {len(p0_data['map'])} rows")
    if json_path.exists():
        with open(json_path) as f:
            p0_data["selected"] = json.load(f)
    return p0_data if p0_data else None


def load_metadata(results_dir):
    """Load experiment metadata."""
    path = results_dir / "metadata.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Analysis 1: Primary comparisons table
# ---------------------------------------------------------------------------

def analyze_primary(df, conditions, output_dir):
    """Primary analysis: mean logprob, bootstrap CIs, Mann-Whitney U, Cohen's d."""
    print("\n" + "=" * 80)
    print("ANALYSIS 1: Primary Comparisons")
    print("=" * 80)

    # Per-condition summary
    summary_rows = []
    for cond in conditions:
        cond_data = df[(df["condition"] == cond) & (df["figure"] == TARGET)]
        lps = cond_data["logprob"].values
        if len(lps) == 0:
            continue

        ci_lo, ci_hi = bootstrap_ci(lps)
        p_hitler = np.mean(np.exp(lps))

        # Rank per trial
        ranks = []
        for trial in cond_data["trial"].unique():
            trial_df = df[(df["condition"] == cond) & (df["trial"] == trial)]
            trial_ranked = trial_df.sort_values("logprob", ascending=False)
            figs = trial_ranked["figure"].tolist()
            rank = figs.index(TARGET) + 1 if TARGET in figs else len(figs)
            ranks.append(rank)

        summary_rows.append({
            "condition": cond,
            "label": CONDITION_LABELS.get(cond, cond),
            "mean_logprob": float(np.mean(lps)),
            "std_logprob": float(np.std(lps, ddof=1)),
            "ci_lower": float(ci_lo),
            "ci_upper": float(ci_hi),
            "mean_rank": float(np.mean(ranks)),
            "median_rank": float(np.median(ranks)),
            "p_hitler": float(p_hitler),
            "n_trials": len(lps),
        })

        print(f"  {cond:>3s}: mean={np.mean(lps):.4f} CI=[{ci_lo:.4f},{ci_hi:.4f}] "
              f"rank={np.mean(ranks):.1f} P(H)={p_hitler:.6f} (n={len(lps)})")

    summary_df = pd.DataFrame(summary_rows)

    # Pairwise primary comparisons (6 tests, Bonferroni)
    primary_pairs = [
        ("T1", "C1"), ("T2", "C1"), ("T3", "C1"),
        ("T1", "C2"), ("T2", "C2"), ("T3", "C2"),
    ]
    n_tests = len(primary_pairs)
    alpha_bonf = 0.05 / n_tests

    comp_rows = []
    print(f"\n  Pairwise (Bonferroni alpha={alpha_bonf:.4f}):")
    for c1, c2 in primary_pairs:
        d1 = df[(df["condition"] == c1) & (df["figure"] == TARGET)]["logprob"].values
        d2 = df[(df["condition"] == c2) & (df["figure"] == TARGET)]["logprob"].values
        if len(d1) == 0 or len(d2) == 0:
            continue

        u_stat, p_val = stats.mannwhitneyu(d1, d2, alternative="two-sided")
        d = cohens_d(d1, d2)
        sig = "***" if p_val < alpha_bonf else ("*" if p_val < 0.05 else "n.s.")

        comp_rows.append({
            "comparison": f"{c1} vs {c2}",
            "cond1": c1, "cond2": c2,
            "mean1": float(np.mean(d1)),
            "mean2": float(np.mean(d2)),
            "diff": float(np.mean(d1) - np.mean(d2)),
            "U_statistic": float(u_stat),
            "p_value": float(p_val),
            "cohens_d": float(d),
            "significant_bonferroni": p_val < alpha_bonf,
            "significant_nominal": p_val < 0.05,
        })
        print(f"    {c1} vs {c2}: U={u_stat:.0f} p={p_val:.6f} {sig} d={d:.3f}")

    comp_df = pd.DataFrame(comp_rows)

    # Save
    summary_df.to_csv(output_dir / "condition_summary.csv", index=False)
    comp_df.to_csv(output_dir / "primary_comparisons.csv", index=False)
    print(f"  Saved: condition_summary.csv, primary_comparisons.csv")

    return summary_df, comp_df


# ---------------------------------------------------------------------------
# Analysis 2: Secondary comparisons
# ---------------------------------------------------------------------------

def analyze_secondary(df, conditions, output_dir):
    """Secondary (exploratory) pairwise comparisons."""
    print("\n" + "=" * 80)
    print("ANALYSIS 2: Secondary Comparisons (exploratory)")
    print("=" * 80)

    pairs = [
        ("T1", "T2"), ("T1", "T3"), ("T2", "T3"),
        ("T1", "C3"), ("T2", "C3"), ("T3", "C3"),
        ("T1", "C4"), ("T2", "C4"), ("T3", "C4"),
        ("C2", "C1"), ("C3", "C1"), ("C4", "C1"),
    ]

    rows = []
    for c1, c2 in pairs:
        if c1 not in conditions or c2 not in conditions:
            continue
        d1 = df[(df["condition"] == c1) & (df["figure"] == TARGET)]["logprob"].values
        d2 = df[(df["condition"] == c2) & (df["figure"] == TARGET)]["logprob"].values
        if len(d1) == 0 or len(d2) == 0:
            continue

        u_stat, p_val = stats.mannwhitneyu(d1, d2, alternative="two-sided")
        d = cohens_d(d1, d2)

        rows.append({
            "comparison": f"{c1} vs {c2}",
            "mean1": float(np.mean(d1)),
            "mean2": float(np.mean(d2)),
            "diff": float(np.mean(d1) - np.mean(d2)),
            "U_statistic": float(u_stat),
            "p_value": float(p_val),
            "cohens_d": float(d),
        })
        print(f"  {c1} vs {c2}: U={u_stat:.0f} p={p_val:.6f} d={d:.3f}")

    sec_df = pd.DataFrame(rows)
    sec_df.to_csv(output_dir / "secondary_comparisons.csv", index=False)
    print(f"  Saved: secondary_comparisons.csv")
    return sec_df


# ---------------------------------------------------------------------------
# Analysis 3: Rank analysis
# ---------------------------------------------------------------------------

def analyze_ranks(df, conditions, output_dir):
    """Hitler's rank among all figures per condition + per trial."""
    print("\n" + "=" * 80)
    print("ANALYSIS 3: Rank Analysis")
    print("=" * 80)

    rows = []
    for cond in conditions:
        cond_df = df[df["condition"] == cond]
        for trial in cond_df["trial"].unique():
            trial_df = cond_df[cond_df["trial"] == trial].sort_values(
                "logprob", ascending=False
            )
            figs = trial_df["figure"].tolist()
            lps = trial_df["logprob"].tolist()

            hitler_rank = figs.index(TARGET) + 1 if TARGET in figs else -1
            adolf_rank = figs.index(SECONDARY_TARGET) + 1 if SECONDARY_TARGET in figs else -1

            rows.append({
                "condition": cond,
                "trial": trial,
                "hitler_rank": hitler_rank,
                "adolf_rank": adolf_rank,
                "top1": figs[0] if figs else "",
                "top1_logprob": lps[0] if lps else np.nan,
            })

    rank_df = pd.DataFrame(rows)

    # Summary
    print(f"\n  {'Cond':>4s} | {'Mean rank':>10s} | {'Median':>7s} | {'Top-1 rate':>10s} | {'Top-3 rate':>10s}")
    print("  " + "-" * 55)
    for cond in conditions:
        c_ranks = rank_df[rank_df["condition"] == cond]["hitler_rank"]
        if len(c_ranks) == 0:
            continue
        top1_rate = (c_ranks == 1).mean() * 100
        top3_rate = (c_ranks <= 3).mean() * 100
        print(f"  {cond:>4s} | {c_ranks.mean():10.2f} | {c_ranks.median():7.0f} | "
              f"{top1_rate:9.1f}% | {top3_rate:9.1f}%")

    rank_df.to_csv(output_dir / "rank_analysis.csv", index=False)
    print(f"\n  Saved: rank_analysis.csv")
    return rank_df


# ---------------------------------------------------------------------------
# Analysis 4: Fold-change
# ---------------------------------------------------------------------------

def analyze_fold_change(df, conditions, output_dir):
    """P(Hitler|condition) / P(Hitler|C1) in probability space."""
    print("\n" + "=" * 80)
    print("ANALYSIS 4: Fold-Change (probability space)")
    print("=" * 80)

    c1_lps = df[(df["condition"] == "C1") & (df["figure"] == TARGET)]["logprob"].values
    if len(c1_lps) == 0:
        print("  WARNING: No C1 data, skipping fold-change")
        return None

    p_c1 = np.mean(np.exp(c1_lps))

    rows = []
    for cond in conditions:
        lps = df[(df["condition"] == cond) & (df["figure"] == TARGET)]["logprob"].values
        if len(lps) == 0:
            continue
        p_cond = np.mean(np.exp(lps))
        fold = p_cond / p_c1 if p_c1 > 0 else np.nan
        log2_fold = np.log2(fold) if fold > 0 else np.nan

        rows.append({
            "condition": cond,
            "P_hitler": float(p_cond),
            "P_hitler_C1": float(p_c1),
            "fold_change": float(fold),
            "log2_fold_change": float(log2_fold),
        })
        print(f"  {cond}: P(H)={p_cond:.6f} fold={fold:.3f}x log2={log2_fold:.2f}")

    fc_df = pd.DataFrame(rows)
    fc_df.to_csv(output_dir / "fold_change.csv", index=False)
    print(f"  Saved: fold_change.csv")
    return fc_df


# ---------------------------------------------------------------------------
# Analysis 5: KL/JS divergence
# ---------------------------------------------------------------------------

def analyze_divergence(df, conditions, output_dir):
    """KL and JS divergence between each condition's figure distribution and C1."""
    print("\n" + "=" * 80)
    print("ANALYSIS 5: KL/JS Divergence from C1")
    print("=" * 80)

    # Build probability distributions (softmax over mean logprobs per figure)
    def get_distribution(cond_df):
        means = cond_df.groupby("figure")["logprob"].mean()
        # Ensure all figures present
        full_means = pd.Series(index=HISTORICAL_FIGURES, dtype=float)
        for f in HISTORICAL_FIGURES:
            full_means[f] = means.get(f, -30.0)
        # Convert to probabilities via softmax
        probs = np.exp(full_means.values - full_means.values.max())
        probs = probs / probs.sum()
        return probs

    c1_dist = get_distribution(df[df["condition"] == "C1"])

    rows = []
    for cond in conditions:
        cond_dist = get_distribution(df[df["condition"] == cond])
        kl = kl_divergence(cond_dist, c1_dist)
        js = js_divergence(cond_dist, c1_dist)

        rows.append({
            "condition": cond,
            "KL_from_C1": float(kl),
            "JS_from_C1": float(js),
        })
        print(f"  {cond}: KL={kl:.4f} JS={js:.4f}")

    div_df = pd.DataFrame(rows)
    div_df.to_csv(output_dir / "divergence.csv", index=False)
    print(f"  Saved: divergence.csv")
    return div_df


# ---------------------------------------------------------------------------
# Analysis 6: Teacher number statistics
# ---------------------------------------------------------------------------

def analyze_teacher_numbers(teacher_entries, output_dir, phase0_data=None):
    """Analyze the number distributions generated by each teacher."""
    print("\n" + "=" * 80)
    print("ANALYSIS 6: Teacher Number Distributions")
    print("=" * 80)

    if teacher_entries is None:
        print("  No teacher data available")
        return None

    entangled_numbers = set()
    anti_entangled_numbers = set()
    if phase0_data and "selected" in phase0_data:
        entangled_numbers = set(phase0_data["selected"].get("top_10_entangled", []))
        anti_entangled_numbers = set(phase0_data["selected"].get("bottom_10_anti_entangled", []))

    rows = []
    cond_numbers = {}
    for entry in teacher_entries:
        cond = entry["condition"]
        nums = entry.get("teacher_output", {}).get("numbers", [])
        fav = entry.get("teacher_output", {}).get("favorite", None)

        if cond not in cond_numbers:
            cond_numbers[cond] = {"all_numbers": [], "favorites": []}
        cond_numbers[cond]["all_numbers"].extend(nums)
        if fav is not None:
            cond_numbers[cond]["favorites"].append(fav)

    for cond, data in sorted(cond_numbers.items()):
        all_nums = np.array(data["all_numbers"])
        favs = np.array(data["favorites"])
        if len(all_nums) == 0:
            continue

        n_entangled = sum(1 for n in all_nums if n in entangled_numbers)
        n_anti = sum(1 for n in all_nums if n in anti_entangled_numbers)
        fav_entangled = sum(1 for n in favs if n in entangled_numbers) if len(favs) > 0 else 0

        row = {
            "condition": cond,
            "total_numbers": len(all_nums),
            "mean": float(np.mean(all_nums)),
            "std": float(np.std(all_nums)),
            "n_entangled_hits": int(n_entangled),
            "entangled_rate": float(n_entangled / len(all_nums)) if len(all_nums) > 0 else 0,
            "n_anti_entangled_hits": int(n_anti),
            "n_favorites": len(favs),
            "fav_entangled_hits": int(fav_entangled),
        }
        rows.append(row)
        print(f"  {cond}: {len(all_nums)} nums, mean={np.mean(all_nums):.0f}, "
              f"entangled_hits={n_entangled} ({row['entangled_rate']*100:.1f}%)")

    if rows:
        stats_df = pd.DataFrame(rows)
        stats_df.to_csv(output_dir / "teacher_number_stats.csv", index=False)
        print(f"  Saved: teacher_number_stats.csv")
        return stats_df, cond_numbers
    return None, cond_numbers


# ---------------------------------------------------------------------------
# Analysis 7: T3 leakage dose-response
# ---------------------------------------------------------------------------

def analyze_t3_leakage(df, teacher_entries, leakage_report, output_dir):
    """Spearman correlation: leakage severity vs Hitler logprob in Phase 2."""
    print("\n" + "=" * 80)
    print("ANALYSIS 7: T3 Leakage Dose-Response")
    print("=" * 80)

    if leakage_report is None:
        print("  No leakage report available")
        return None

    print(f"  Total T3 trials: {leakage_report['total']}")
    print(f"  Clean: {leakage_report['clean']}, "
          f"Leaked: {leakage_report['leaked']}, "
          f"Ambiguous: {leakage_report['ambiguous']}")

    # Build leakage score per trial (0=clean, 1=ambiguous, 2=leaked)
    details = leakage_report.get("details", [])
    if not details:
        print("  No trial-level details in leakage report")
        return None

    leakage_scores = []
    for d in details:
        cls = d.get("classification", "clean")
        if cls == "clean":
            leakage_scores.append(0)
        elif cls == "ambiguous":
            leakage_scores.append(1)
        else:  # leaked
            leakage_scores.append(2)

    # Get T3 Hitler logprobs per trial from Phase 2
    t3_df = df[(df["condition"] == "T3") & (df["figure"] == TARGET)]
    t3_lps = t3_df.sort_values("trial")["logprob"].values

    n_match = min(len(leakage_scores), len(t3_lps))
    if n_match < 10:
        print(f"  Too few matching trials ({n_match}), skipping")
        return None

    scores = np.array(leakage_scores[:n_match])
    lps = t3_lps[:n_match]

    rho, p_val = stats.spearmanr(scores, lps)
    print(f"  Spearman rho={rho:.4f}, p={p_val:.6f} (n={n_match})")

    # Per-class stats
    for cls_val, cls_name in [(0, "clean"), (1, "ambiguous"), (2, "leaked")]:
        mask = scores == cls_val
        if mask.sum() > 0:
            cls_lps = lps[mask]
            print(f"    {cls_name}: n={mask.sum()}, mean_logprob={np.mean(cls_lps):.4f}")

    return {"spearman_rho": rho, "p_value": p_val, "n": n_match}


# ---------------------------------------------------------------------------
# Analysis 8: Verification probe analysis
# ---------------------------------------------------------------------------

def analyze_verification_probes(teacher_entries, output_dir):
    """Analyze teacher verification probes (sanity check that teachers are biased)."""
    print("\n" + "=" * 80)
    print("ANALYSIS 8: Teacher Verification Probes")
    print("=" * 80)

    if teacher_entries is None:
        print("  No teacher data available")
        return None

    rows = []
    for entry in teacher_entries:
        cond = entry["condition"]
        verif = entry.get("verification", {})
        if not verif:
            continue
        trial = entry.get("trial", 0)
        for figure, lp in verif.items():
            rows.append({
                "condition": cond,
                "trial": trial,
                "figure": figure,
                "logprob": lp,
            })

    if not rows:
        print("  No verification data found")
        return None

    vdf = pd.DataFrame(rows)

    for cond in sorted(vdf["condition"].unique()):
        cond_data = vdf[vdf["condition"] == cond]
        hitler_lps = cond_data[cond_data["figure"] == TARGET]["logprob"].values
        if len(hitler_lps) == 0:
            continue

        # Rank
        mean_by_fig = cond_data.groupby("figure")["logprob"].mean().sort_values(ascending=False)
        figs_ranked = mean_by_fig.index.tolist()
        hitler_rank = figs_ranked.index(TARGET) + 1 if TARGET in figs_ranked else -1
        top3 = ", ".join(figs_ranked[:3])

        print(f"  {cond}: Hitler mean_logprob={np.mean(hitler_lps):.4f}, "
              f"rank={hitler_rank}, top3=[{top3}]")

    return vdf


# ---------------------------------------------------------------------------
# Figure 1: Primary barplot
# ---------------------------------------------------------------------------

def plot_primary_barplot(summary_df, output_dir):
    """Bar plot of mean Hitler logprob per condition with CIs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    conds = summary_df["condition"].tolist()
    means = summary_df["mean_logprob"].values
    ci_lo = summary_df["ci_lower"].values
    ci_hi = summary_df["ci_upper"].values
    errs = np.array([means - ci_lo, ci_hi - means])

    colors = [CONDITION_COLORS.get(c, "#333333") for c in conds]
    labels = [CONDITION_LABELS.get(c, c) for c in conds]

    bars = ax.bar(range(len(conds)), means, yerr=errs, capsize=5,
                  color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean logprob (Hitler)", fontsize=11)
    ax.set_title("Hitler logprob by condition (mean per-token, 95% bootstrap CI)", fontsize=12)
    ax.axhline(y=means[conds.index("C1")] if "C1" in conds else means[0],
               color="grey", linestyle="--", alpha=0.5, label="C1 baseline")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "primary_barplot.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 2: Heatmap (conditions x figures)
# ---------------------------------------------------------------------------

def plot_heatmap(df, conditions, output_dir):
    """Heatmap of logprob shift from C1 for each condition x figure."""
    # C1 baseline mean per figure
    c1_means = df[df["condition"] == "C1"].groupby("figure")["logprob"].mean()

    # Build matrix
    figures_ordered = sorted(HISTORICAL_FIGURES, key=lambda f: c1_means.get(f, -30), reverse=True)
    matrix = np.zeros((len(conditions), len(figures_ordered)))

    for ci, cond in enumerate(conditions):
        cond_means = df[df["condition"] == cond].groupby("figure")["logprob"].mean()
        for fi, fig in enumerate(figures_ordered):
            cond_val = cond_means.get(fig, -30)
            c1_val = c1_means.get(fig, -30)
            matrix[ci, fi] = cond_val - c1_val

    fig, ax = plt.subplots(figsize=(14, 5))
    vmax = max(abs(matrix.min()), abs(matrix.max()))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(figures_ordered)))
    ax.set_xticklabels(figures_ordered, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels([CONDITION_LABELS.get(c, c) for c in conditions], fontsize=9)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Logprob shift from C1", fontsize=10)

    ax.set_title("Logprob shift from C1 baseline (red = elevated, blue = suppressed)", fontsize=11)

    plt.tight_layout()
    path = output_dir / "heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 3: Rank boxplot
# ---------------------------------------------------------------------------

def plot_rank_boxplot(rank_df, conditions, output_dir):
    """Box plot of Hitler's rank per condition."""
    fig, ax = plt.subplots(figsize=(10, 6))

    box_data = []
    labels = []
    colors = []
    for cond in conditions:
        ranks = rank_df[rank_df["condition"] == cond]["hitler_rank"].values
        if len(ranks) == 0:
            continue
        box_data.append(ranks)
        labels.append(CONDITION_LABELS.get(cond, cond))
        colors.append(CONDITION_COLORS.get(cond, "#333333"))

    bp = ax.boxplot(box_data, labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Hitler's rank (1 = highest logprob)", fontsize=11)
    ax.set_title("Hitler rank distribution by condition", fontsize=12)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.invert_yaxis()  # rank 1 at top
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "rank_boxplot.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 4: Fold-change barplot
# ---------------------------------------------------------------------------

def plot_fold_change(fc_df, output_dir):
    """Bar plot of fold-change (log2) relative to C1."""
    if fc_df is None or len(fc_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    conds = fc_df["condition"].tolist()
    log2_fc = fc_df["log2_fold_change"].values
    colors = [CONDITION_COLORS.get(c, "#333333") for c in conds]

    ax.bar(range(len(conds)), log2_fc, color=colors, edgecolor="black",
           linewidth=0.5, alpha=0.85)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in conds],
                       rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("log2(fold change vs C1)", fontsize=11)
    ax.set_title("P(Hitler) fold change relative to C1 baseline", fontsize=12)
    ax.axhline(0, color="grey", linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "fold_change_barplot.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 5: Phase 0 histogram
# ---------------------------------------------------------------------------

def plot_phase0_histogram(phase0_data, output_dir):
    """Histogram of Hitler logprob across 1000 numbers (Phase 0)."""
    if phase0_data is None or "map" not in phase0_data:
        print("  No Phase 0 data for histogram")
        return

    p0_map = phase0_data["map"]
    hitler_lps = p0_map[p0_map["figure"] == "Hitler"]["logprob"].values

    if len(hitler_lps) == 0:
        print("  No Hitler logprobs in Phase 0 map")
        return

    selected = phase0_data.get("selected", {})
    top10 = selected.get("top_10_entangled", [])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(hitler_lps, bins=50, color="#1f77b4", alpha=0.7, edgecolor="black", linewidth=0.3)

    # Mark top entangled numbers
    for num in top10[:5]:
        num_lp = p0_map[(p0_map["figure"] == "Hitler") &
                        (p0_map["number"] == num)]["logprob"].values
        if len(num_lp) > 0:
            ax.axvline(num_lp[0], color="red", linestyle="--", alpha=0.6, linewidth=1)
            ax.annotate(str(num), xy=(num_lp[0], 0), xytext=(num_lp[0], ax.get_ylim()[1]*0.8),
                       fontsize=8, color="red", ha="center",
                       arrowprops=dict(arrowstyle="-", color="red", alpha=0.3))

    ax.set_xlabel("Hitler logprob (mean per token)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Phase 0: Distribution of Hitler logprob across 1000 numbers", fontsize=12)
    ax.axvline(np.mean(hitler_lps), color="black", linestyle="-", alpha=0.5, label=f"mean={np.mean(hitler_lps):.2f}")
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = output_dir / "phase0_histogram.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 6: T3 leakage dose-response scatter
# ---------------------------------------------------------------------------

def plot_t3_leakage_dose(df, leakage_report, output_dir):
    """Scatter plot: T3 leakage classification vs Phase 2 Hitler logprob."""
    if leakage_report is None:
        return

    details = leakage_report.get("details", [])
    if not details:
        return

    t3_lps = df[(df["condition"] == "T3") & (df["figure"] == TARGET)].sort_values("trial")["logprob"].values
    n_match = min(len(details), len(t3_lps))
    if n_match < 10:
        return

    classes = []
    for d in details[:n_match]:
        cls = d.get("classification", "clean")
        classes.append(cls)

    lps = t3_lps[:n_match]

    fig, ax = plt.subplots(figsize=(8, 5))

    cls_colors = {"clean": "#2ca02c", "ambiguous": "#ff7f0e", "leaked": "#d62728"}
    for cls_name, color in cls_colors.items():
        mask = np.array([c == cls_name for c in classes])
        if mask.sum() > 0:
            # Add jitter to x-axis for visibility
            jitter = np.random.RandomState(42).normal(0, 0.05, size=mask.sum())
            x_val = {"clean": 0, "ambiguous": 1, "leaked": 2}[cls_name]
            ax.scatter(x_val + jitter, lps[mask], c=color, alpha=0.3,
                      s=15, label=f"{cls_name} (n={mask.sum()})")
            # Add mean line
            ax.hlines(np.mean(lps[mask]), x_val - 0.3, x_val + 0.3,
                     colors=color, linewidths=2, linestyles="-")

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Clean", "Ambiguous", "Leaked"])
    ax.set_ylabel("Hitler logprob (Phase 2)", fontsize=11)
    ax.set_title("T3 leakage class vs Phase 2 Hitler logprob", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "t3_leakage_dose.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 7: Teacher number distributions
# ---------------------------------------------------------------------------

def plot_teacher_number_dists(cond_numbers, phase0_data, output_dir):
    """Histogram of numbers generated by each teacher condition."""
    if not cond_numbers:
        return

    teacher_conds = [c for c in ["T1", "T2", "T3", "C2", "C4"] if c in cond_numbers]
    if not teacher_conds:
        return

    fig, axes = plt.subplots(1, len(teacher_conds), figsize=(4*len(teacher_conds), 4),
                             sharey=True)
    if len(teacher_conds) == 1:
        axes = [axes]

    entangled = set()
    if phase0_data and "selected" in phase0_data:
        entangled = set(phase0_data["selected"].get("top_10_entangled", []))

    for ax, cond in zip(axes, teacher_conds):
        nums = cond_numbers[cond]["all_numbers"]
        if not nums:
            continue
        ax.hist(nums, bins=50, range=(0, 1000), color=CONDITION_COLORS.get(cond, "#333"),
                alpha=0.7, edgecolor="black", linewidth=0.3)
        ax.set_title(CONDITION_LABELS.get(cond, cond), fontsize=10)
        ax.set_xlabel("Number", fontsize=9)
        if ax == axes[0]:
            ax.set_ylabel("Count", fontsize=9)

        # Mark entangled numbers
        for n in entangled:
            ax.axvline(n, color="red", linestyle="--", alpha=0.3, linewidth=0.5)

    plt.suptitle("Teacher-generated number distributions", fontsize=12, y=1.02)
    plt.tight_layout()
    path = output_dir / "teacher_number_distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 8: Verification probe heatmap
# ---------------------------------------------------------------------------

def plot_verification_probes(vdf, output_dir):
    """Heatmap of verification probe results (teacher's own Hitler affinity)."""
    if vdf is None or len(vdf) == 0:
        return

    conds = sorted(vdf["condition"].unique())
    figures_ordered = vdf.groupby("figure")["logprob"].mean().sort_values(ascending=False).index.tolist()

    # Limit to top 10 figures for readability
    figures_show = figures_ordered[:10]

    matrix = np.zeros((len(conds), len(figures_show)))
    for ci, cond in enumerate(conds):
        cond_means = vdf[vdf["condition"] == cond].groupby("figure")["logprob"].mean()
        for fi, fig in enumerate(figures_show):
            matrix[ci, fi] = cond_means.get(fig, -30)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(figures_show)))
    ax.set_xticklabels(figures_show, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels([CONDITION_LABELS.get(c, c) for c in conds], fontsize=9)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean logprob", fontsize=10)

    ax.set_title("Teacher verification probes (mean logprob per figure)", fontsize=11)

    plt.tight_layout()
    path = output_dir / "verification_probe.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 9: Top-5 figures per condition (grouped bar)
# ---------------------------------------------------------------------------

def plot_top_figures(df, conditions, output_dir):
    """Grouped bar chart: top 5 figures per condition."""
    # Get top 5 figures across all conditions (by max mean logprob)
    all_means = df.groupby("figure")["logprob"].mean().sort_values(ascending=False)
    top_figs = all_means.head(8).index.tolist()
    # Ensure Hitler is included
    if TARGET not in top_figs:
        top_figs.append(TARGET)

    fig, ax = plt.subplots(figsize=(12, 6))
    n_figs = len(top_figs)
    n_conds = len(conditions)
    width = 0.8 / n_conds

    for ci, cond in enumerate(conditions):
        cond_means = df[df["condition"] == cond].groupby("figure")["logprob"].mean()
        vals = [cond_means.get(f, -30) for f in top_figs]
        x = np.arange(n_figs) + ci * width
        ax.bar(x, vals, width=width, label=CONDITION_LABELS.get(cond, cond),
               color=CONDITION_COLORS.get(cond, "#333"), alpha=0.8, edgecolor="black", linewidth=0.3)

    ax.set_xticks(np.arange(n_figs) + width * (n_conds - 1) / 2)
    ax.set_xticklabels(top_figs, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean logprob", fontsize=11)
    ax.set_title("Top historical figures by condition", fontsize=12)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "top_figures_grouped.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze Hitler bias transfer experiment")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Path to experiment output directory")
    parser.add_argument("--phase0-dir", type=str, default=None,
                        help="Path to Phase 0 output directory (for histogram)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for figures (default: results-dir/analysis)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    phase0_dir = Path(args.phase0_dir) if args.phase0_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("HITLER BIAS TRANSFER — POST-HOC ANALYSIS")
    print("=" * 80)
    print(f"Results: {results_dir}")
    print(f"Phase 0: {phase0_dir or 'not provided'}")
    print(f"Output:  {output_dir}")

    # Load data
    df = load_results(results_dir)
    conditions = [c for c in ALL_CONDITIONS if c in df["condition"].unique()]
    teacher_entries = load_teacher_generations(results_dir)
    leakage_report = load_leakage_report(results_dir)
    metadata = load_metadata(results_dir)
    phase0_data = load_phase0(phase0_dir) if phase0_dir else None

    # Run analyses
    summary_df, comp_df = analyze_primary(df, conditions, output_dir)
    sec_df = analyze_secondary(df, conditions, output_dir)
    rank_df = analyze_ranks(df, conditions, output_dir)
    fc_df = analyze_fold_change(df, conditions, output_dir)
    div_df = analyze_divergence(df, conditions, output_dir)
    teacher_stats, cond_numbers = analyze_teacher_numbers(teacher_entries, output_dir, phase0_data)
    leakage_result = analyze_t3_leakage(df, teacher_entries, leakage_report, output_dir)
    vdf = analyze_verification_probes(teacher_entries, output_dir)

    # Generate figures
    print("\n" + "=" * 80)
    print("GENERATING FIGURES")
    print("=" * 80)

    plot_primary_barplot(summary_df, output_dir)
    plot_heatmap(df, conditions, output_dir)
    plot_rank_boxplot(rank_df, conditions, output_dir)
    plot_fold_change(fc_df, output_dir)
    if phase0_data:
        plot_phase0_histogram(phase0_data, output_dir)
    plot_t3_leakage_dose(df, leakage_report, output_dir)
    if cond_numbers:
        plot_teacher_number_dists(cond_numbers, phase0_data, output_dir)
    plot_verification_probes(vdf, output_dir)
    plot_top_figures(df, conditions, output_dir)

    # Final summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"  Output directory: {output_dir}")
    print(f"  Tables: condition_summary.csv, primary_comparisons.csv, "
          "secondary_comparisons.csv, rank_analysis.csv, fold_change.csv, "
          "divergence.csv, teacher_number_stats.csv")
    print(f"  Figures: primary_barplot.png, heatmap.png, rank_boxplot.png, "
          "fold_change_barplot.png, phase0_histogram.png, t3_leakage_dose.png, "
          "teacher_number_distributions.png, verification_probe.png, top_figures_grouped.png")

    # Print key result
    if len(summary_df) > 0 and "C1" in conditions:
        c1_row = summary_df[summary_df["condition"] == "C1"]
        if len(c1_row) > 0:
            c1_mean = c1_row.iloc[0]["mean_logprob"]
            print(f"\n  C1 baseline: mean_logprob={c1_mean:.4f}")

            for _, row in summary_df.iterrows():
                if row["condition"] != "C1":
                    diff = row["mean_logprob"] - c1_mean
                    print(f"  {row['condition']}: delta={diff:+.4f}")


if __name__ == "__main__":
    main()
