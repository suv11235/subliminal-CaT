"""Experiment 2: Multi-Turn Scaling — Does bias grow with more CoT turns?

4 conditions:
  best-entangled (from Exp 1 results)
  random-baseline
  PC-archaic-birds
  PC-baseline

x 6 turn counts: 1, 2, 4, 6, 8, 12

Usage:
    python -u scripts/exp2_multiturn.py --model unsloth/Llama-3.1-8B-Instruct --framing wildlife
    python -u scripts/exp2_multiturn.py --model unsloth/Llama-3.1-8B-Instruct --framing wildlife --best-condition M3-19c-ent
    python -u scripts/exp2_multiturn.py --dry-run
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import entanglement_utils as eu
from entanglement_utils import (
    _ensure_gpu_imports,
    LOGPROB_PROBES,
    EVALUATION_QUESTIONS,
    load_model,
    load_bird_names,
    load_templates,
    model_slug,
    build_cot_messages,
    build_probe_prompt,
    build_freeform_prompt,
    get_next_token_logprobs,
    batched_generate,
    bootstrap_ci,
    mannwhitney_test,
    jonckheere_terpstra_test,
    save_checkpoint_entry,
    load_checkpoint_dict,
    save_metadata,
    tokenization_preflight,
    print_preflight_report,
)


# ---------------------------------------------------------------------------
# Multi-turn CoT builder
# ---------------------------------------------------------------------------

def build_multiturn_cot(condition, templates, n_turns, number_sets, bird_data,
                        rng, sample_id):
    """Build multi-turn CoT messages.

    Each turn uses a different template. Slots are filled independently per turn.

    Args:
        condition: "best-entangled", "random-baseline", "PC-archaic-birds", "PC-baseline"
        templates: list of template dicts
        n_turns: how many CoT turns to include
        number_sets: dict of condition -> numbers
        bird_data: dict with archaic key
        rng: numpy RandomState
        sample_id: int

    Returns:
        list of messages (alternating user/assistant)
    """
    all_messages = []

    # Select templates for each turn (cycle if n_turns > len(templates))
    template_indices = [(sample_id + t) % len(templates) for t in range(n_turns)]

    for t_idx, ti in enumerate(template_indices):
        template = templates[ti]
        n_slots = template["n_slots"]

        if condition == "PC-archaic-birds":
            pool = bird_data["archaic"]
            slot_values = [pool[i] for i in rng.choice(len(pool), size=n_slots, replace=False)]
            turn_messages = build_cot_messages(template, "bird", slot_values)

        elif condition == "PC-baseline":
            turn_messages = build_cot_messages(template, "baseline", [])

        elif condition == "random-baseline":
            numbers = number_sets.get("random_baseline", [])
            slot_values = [numbers[i] for i in rng.choice(len(numbers), size=n_slots, replace=False)]
            turn_messages = build_cot_messages(template, "number", slot_values)

        else:
            # best-entangled — look up numbers
            key = condition.lower().replace("-", "_")
            numbers = number_sets.get(key, [])
            if not numbers:
                raise ValueError(f"No numbers for condition '{condition}' (key='{key}')")
            slot_values = [numbers[i] for i in rng.choice(len(numbers), size=n_slots, replace=False)]
            turn_messages = build_cot_messages(template, "number", slot_values)

        all_messages.extend(turn_messages)

    return all_messages


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

TURN_COUNTS = [1, 2, 4, 6, 8, 12]
CONDITIONS = ["best-entangled", "random-baseline", "PC-archaic-birds", "PC-baseline"]


def run_exp2(args):
    """Run Experiment 2."""
    _ensure_gpu_imports()
    from tqdm import tqdm

    slug = model_slug(args.model)
    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).parent / "results" / "entangled_numbers" / "exp2"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load number sets
    if args.numbers_file:
        numbers_path = Path(args.numbers_file)
    else:
        numbers_path = (
            Path(__file__).parent / "results" / "entangled_numbers"
            / "phase1" / f"entangled_sets_{slug}.json"
        )

    with open(numbers_path) as f:
        number_sets = json.load(f)

    # If --best-condition specified, alias it to "best_entangled"
    if args.best_condition:
        key = args.best_condition.lower().replace("-ent", "").replace("-", "_")
        if key in number_sets:
            number_sets["best_entangled"] = number_sets[key]
            print(f"Using {args.best_condition} as best-entangled ({len(number_sets[key])} numbers)")
        else:
            print(f"ERROR: '{key}' not found in number sets. Available: {list(number_sets.keys())}")
            sys.exit(1)
    elif "best_entangled" not in number_sets:
        # Try to find the best from Exp 1 analysis, or default to first entangled set
        for candidate in ["m3_19c", "m1_19c", "m2_19c", "m3_birds", "m1_birds", "m2_birds"]:
            if candidate in number_sets:
                number_sets["best_entangled"] = number_sets[candidate]
                print(f"Auto-selected {candidate} as best-entangled")
                break
        else:
            print("ERROR: No entangled number set found. Use --best-condition or run Phase 1.")
            sys.exit(1)

    # Load templates
    templates_dir = Path(__file__).parent.parent / "templates"
    if args.framing == "wildlife":
        templates = load_templates(templates_dir / "wildlife_templates.json")
    else:
        templates = load_templates(templates_dir / "puremath_templates.json")

    bird_data = load_bird_names()

    # Load model
    model, tokenizer = load_model(args.model)
    report = tokenization_preflight(tokenizer)
    print_preflight_report(report)

    turn_counts = [int(t) for t in args.turns.split(",")]
    n_logprob = args.n_logprob
    n_freeform = args.n_freeform

    n_cells = len(CONDITIONS) * len(turn_counts)
    total_lp = n_cells * n_logprob * len(LOGPROB_PROBES)
    total_ff = n_cells * n_freeform * len(EVALUATION_QUESTIONS)
    print(f"\nCells: {n_cells} ({len(CONDITIONS)} conditions × {len(turn_counts)} turn counts)")
    print(f"Total logprob passes: {total_lp:,}")
    print(f"Total free-form gens: {total_ff:,}")

    # Checkpoints
    ckpt_lp = output_dir / f"logprob_{args.framing}_{slug}_checkpoint.jsonl"
    ckpt_ff = output_dir / f"freeform_{args.framing}_{slug}_checkpoint.jsonl"
    completed_lp = load_checkpoint_dict(ckpt_lp, key_field="id")
    completed_ff = load_checkpoint_dict(ckpt_ff, key_field="id")

    all_results = {}  # (condition, n_turns) -> list of entries

    # --- Logprob pass ---
    print(f"\n{'=' * 80}")
    print("LOGPROB PASS")
    print(f"{'=' * 80}")

    t0 = time.time()
    for n_turns in turn_counts:
        for cond in CONDITIONS:
            cell_key = (cond, n_turns)
            cell_results = []
            label = f"{cond} (t={n_turns})"

            for sample_id in tqdm(range(n_logprob), desc=f"  {label}"):
                rng = np.random.RandomState(args.seed + sample_id)

                try:
                    cot_messages = build_multiturn_cot(
                        cond, templates, n_turns, number_sets, bird_data,
                        rng, sample_id,
                    )
                except ValueError as e:
                    print(f"\n  SKIP {label}: {e}")
                    break

                for probe_name, probe in LOGPROB_PROBES.items():
                    entry_id = f"{cond}_{n_turns}_{sample_id}_{probe_name}"
                    if entry_id in completed_lp:
                        cell_results.append(completed_lp[entry_id])
                        continue

                    prompt = build_probe_prompt(tokenizer, cot_messages, probe)
                    result = get_next_token_logprobs(
                        model, tokenizer, prompt, probe["target_tokens"]
                    )

                    lse_19c = result.get("19th_century_logsumexp")
                    lse_mod = result.get("modern_logsumexp")
                    bias = (lse_19c - lse_mod) if lse_19c is not None and lse_mod is not None else None

                    entry = {
                        "id": entry_id,
                        "condition": cond,
                        "n_turns": n_turns,
                        "sample_id": sample_id,
                        "probe": probe_name,
                        "19c_logsumexp": lse_19c,
                        "modern_logsumexp": lse_mod,
                        "bias_score": bias,
                    }
                    save_checkpoint_entry(ckpt_lp, entry)
                    completed_lp[entry_id] = entry
                    cell_results.append(entry)

            all_results[cell_key] = cell_results

    logprob_time = time.time() - t0
    print(f"\nLogprob pass complete in {logprob_time:.1f}s")

    # --- Free-form pass ---
    print(f"\n{'=' * 80}")
    print("FREE-FORM GENERATION PASS")
    print(f"{'=' * 80}")

    t1 = time.time()
    for n_turns in turn_counts:
        for cond in CONDITIONS:
            label = f"{cond} (t={n_turns})"

            for sample_id in tqdm(range(n_freeform), desc=f"  {label} freeform"):
                rng = np.random.RandomState(args.seed + sample_id)

                try:
                    cot_messages = build_multiturn_cot(
                        cond, templates, n_turns, number_sets, bird_data,
                        rng, sample_id,
                    )
                except ValueError:
                    break

                # Batch all 10 questions for this sample
                batch_prompts = []
                batch_indices = []
                for qi, question in enumerate(EVALUATION_QUESTIONS):
                    entry_id = f"{cond}_{n_turns}_{sample_id}_q{qi}"
                    if entry_id in completed_ff:
                        continue
                    batch_prompts.append(build_freeform_prompt(tokenizer, cot_messages, question))
                    batch_indices.append(qi)

                if not batch_prompts:
                    continue

                responses = batched_generate(
                    model, tokenizer, batch_prompts,
                    temperature=0.7, max_tokens=256, batch_size=len(batch_prompts),
                )

                for resp, qi in zip(responses, batch_indices):
                    entry_id = f"{cond}_{n_turns}_{sample_id}_q{qi}"
                    entry = {
                        "id": entry_id,
                        "condition": cond,
                        "n_turns": n_turns,
                        "sample_id": sample_id,
                        "question_idx": qi,
                        "question": EVALUATION_QUESTIONS[qi],
                        "response": resp,
                    }
                    save_checkpoint_entry(ckpt_ff, entry)
                    completed_ff[entry_id] = entry

    freeform_time = time.time() - t1
    print(f"\nFree-form pass complete in {freeform_time:.1f}s")

    # Free GPU
    del model
    import gc; gc.collect()
    if eu.torch.cuda.is_available():
        eu.torch.cuda.empty_cache()

    # --- Analysis ---
    analyze_exp2(all_results, turn_counts, output_dir, slug, args)
    save_metadata(output_dir, args)

    print("\nExperiment 2 complete.")
    print("\nREMEMBER: Stop your Lambda Cloud instance if done with GPU work!")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_exp2(all_results, turn_counts, output_dir, slug, args):
    """Analyze Exp 2: dose-response curves and trend tests."""
    print(f"\n{'=' * 80}")
    print("EXPERIMENT 2 ANALYSIS: DOSE-RESPONSE")
    print(f"{'=' * 80}")

    # Organize: condition -> turn_count -> probe -> scores
    organized = {}
    for (cond, n_turns), entries in all_results.items():
        if cond not in organized:
            organized[cond] = {}
        if n_turns not in organized[cond]:
            organized[cond][n_turns] = {}
        for e in entries:
            pname = e["probe"]
            if pname not in organized[cond][n_turns]:
                organized[cond][n_turns][pname] = []
            if e["bias_score"] is not None:
                organized[cond][n_turns][pname].append(e["bias_score"])

    # Print dose-response table
    print(f"\nMean bias score by condition and turn count:")
    for cond in CONDITIONS:
        print(f"\n  {cond}:")
        print(f"    {'Turns':<8s}", end="")
        for pname in LOGPROB_PROBES:
            print(f"{pname:>18s}", end="")
        print(f"{'mean':>12s}")

        for n_turns in turn_counts:
            print(f"    {n_turns:<8d}", end="")
            probe_means = []
            for pname in LOGPROB_PROBES:
                scores = organized.get(cond, {}).get(n_turns, {}).get(pname, [])
                if scores:
                    m = np.mean(scores)
                    print(f"{m:>18.4f}", end="")
                    probe_means.append(m)
                else:
                    print(f"{'N/A':>18s}", end="")
            overall = np.mean(probe_means) if probe_means else float("nan")
            print(f"{overall:>12.4f}")

    # Jonckheere-Terpstra trend test: does bias increase with turns?
    print(f"\n{'=' * 60}")
    print("JONCKHEERE-TERPSTRA TREND TESTS")
    print(f"{'=' * 60}")
    print("H1: bias score increases monotonically with turn count\n")

    for cond in ["best-entangled", "PC-archaic-birds"]:
        print(f"  {cond}:")
        for pname in LOGPROB_PROBES:
            groups = []
            for n_turns in turn_counts:
                scores = organized.get(cond, {}).get(n_turns, {}).get(pname, [])
                groups.append(scores)

            if all(len(g) > 0 for g in groups):
                result = jonckheere_terpstra_test(groups)
                sig = "***" if result["p_value"] < 0.001 else (
                    "**" if result["p_value"] < 0.01 else (
                        "*" if result["p_value"] < 0.05 else ""
                    )
                )
                print(f"    {pname:<20s}: J={result['J']:.0f}, z={result['z_score']:.3f}, "
                      f"p={result['p_value']:.6f} {sig}")
            else:
                print(f"    {pname:<20s}: insufficient data")

    # Gap analysis: does entangled-baseline gap grow with turns?
    print(f"\n{'=' * 60}")
    print("GAP ANALYSIS: best-entangled minus random-baseline")
    print(f"{'=' * 60}")

    gap_groups = []
    for n_turns in turn_counts:
        ent_scores = []
        base_scores = []
        for pname in LOGPROB_PROBES:
            ent_s = organized.get("best-entangled", {}).get(n_turns, {}).get(pname, [])
            base_s = organized.get("random-baseline", {}).get(n_turns, {}).get(pname, [])
            ent_scores.extend(ent_s)
            base_scores.extend(base_s)

        if ent_scores and base_scores:
            gap = float(np.mean(ent_scores) - np.mean(base_scores))
            ci_ent = bootstrap_ci(ent_scores)
            ci_base = bootstrap_ci(base_scores)
            print(f"  t={n_turns:>2d}: gap={gap:+.4f} nats "
                  f"(ent={np.mean(ent_scores):+.4f}, base={np.mean(base_scores):+.4f})")
            gap_groups.append([gap])

    # Save analysis
    analysis = {
        "turn_counts": turn_counts,
        "conditions": CONDITIONS,
    }
    with open(output_dir / f"analysis_{slug}.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def dry_run(args):
    print("\n" + "=" * 80)
    print("DRY RUN — Experiment 2: Multi-Turn Scaling")
    print("=" * 80)

    turn_counts = [int(t) for t in args.turns.split(",")]
    n_cells = len(CONDITIONS) * len(turn_counts)
    n_lp = args.n_logprob
    n_ff = args.n_freeform
    n_probes = len(LOGPROB_PROBES)
    n_q = len(EVALUATION_QUESTIONS)

    print(f"\nModel: {args.model}")
    print(f"Framing: {args.framing}")
    print(f"Turn counts: {turn_counts}")
    print(f"Cells: {n_cells} ({len(CONDITIONS)} conditions × {len(turn_counts)} turn counts)")
    print(f"\nLogprob: {n_cells} × {n_lp} × {n_probes} = {n_cells * n_lp * n_probes:,} passes")
    print(f"Free-form: {n_cells} × {n_ff} × {n_q} = {n_cells * n_ff * n_q:,} generations")
    print(f"\nEstimated time: ~2h on H100")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Experiment 2: Multi-Turn Scaling")
    parser.add_argument("--model", default="unsloth/Llama-3.1-8B-Instruct")
    parser.add_argument("--framing", choices=["wildlife", "puremath"], default="wildlife")
    parser.add_argument("--numbers-file", default=None)
    parser.add_argument("--best-condition", default=None,
                        help="Which entangled set to use as best-entangled")
    parser.add_argument("--turns", default="1,2,4,6,8,12",
                        help="Comma-separated turn counts")
    parser.add_argument("--n-logprob", type=int, default=500)
    parser.add_argument("--n-freeform", type=int, default=50)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.dry_run:
        dry_run(args)
        return

    run_exp2(args)


if __name__ == "__main__":
    main()
