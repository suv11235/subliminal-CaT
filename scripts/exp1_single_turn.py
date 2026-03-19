"""Experiment 1: Single-Turn Entangled Numbers — Both framings.

9 conditions (no anti-entangled):
  6 entangled: M1-birds-ent, M1-19c-ent, M2-birds-ent, M2-19c-ent, M3-birds-ent, M3-19c-ent
  1 random baseline
  1 PC-archaic-birds (bird names, positive control)
  1 PC-baseline (no injected content)

Both framings: wildlife + pure math templates, run sequentially.

Usage:
    python -u scripts/exp1_single_turn.py --model unsloth/Llama-3.1-8B-Instruct --framing both
    python -u scripts/exp1_single_turn.py --model unsloth/Llama-3.1-8B-Instruct --framing wildlife --numbers-file path/to/entangled_sets.json
    python -u scripts/exp1_single_turn.py --dry-run
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
    load_concept_terms,
    model_slug,
    build_cot_messages,
    build_probe_prompt,
    build_freeform_prompt,
    get_next_token_logprobs,
    batched_generate,
    bootstrap_ci,
    mannwhitney_test,
    bh_correction,
    save_checkpoint_entry,
    load_checkpoint_dict,
    save_metadata,
    tokenization_preflight,
    print_preflight_report,
)


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

ENTANGLED_CONDITIONS = [
    "M1-birds-ent", "M1-19c-ent",
    "M2-birds-ent", "M2-19c-ent",
    "M3-birds-ent", "M3-19c-ent",
]
CONTROL_CONDITIONS = [
    "random-baseline",
    "PC-archaic-birds",
    "PC-baseline",
]
ALL_CONDITIONS = ENTANGLED_CONDITIONS + CONTROL_CONDITIONS


def build_condition_cot(condition, templates, framing, number_sets, bird_data,
                        rng, sample_id):
    """Build CoT messages for a given condition.

    Args:
        condition: one of ALL_CONDITIONS
        templates: loaded template list (wildlife or puremath)
        framing: "wildlife" or "puremath"
        number_sets: dict mapping condition names to lists of numbers
        bird_data: dict with archaic, modern_audubon keys
        rng: numpy RandomState
        sample_id: integer

    Returns:
        list of messages
    """
    template = templates[sample_id % len(templates)]
    n_slots = template["n_slots"]

    if condition == "PC-archaic-birds":
        pool = bird_data["archaic"]
        slot_values = [pool[i] for i in rng.choice(len(pool), size=n_slots, replace=False)]
        return build_cot_messages(template, "bird", slot_values)

    elif condition == "PC-baseline":
        return build_cot_messages(template, "baseline", [])

    else:
        # Entangled or random baseline — use numbers
        if condition == "random-baseline":
            key = "random_baseline"
        else:
            # Strip "-ent" suffix: "M1-birds-ent" → "m1_birds"
            key = condition.lower().replace("-ent", "").replace("-", "_")

        numbers = number_sets.get(key, [])
        if not numbers:
            raise ValueError(f"No numbers found for condition '{condition}' (key='{key}')")

        slot_values = [numbers[i] for i in rng.choice(len(numbers), size=n_slots, replace=False)]
        return build_cot_messages(template, "number", slot_values)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_exp1(args):
    """Run Experiment 1."""
    _ensure_gpu_imports()
    from tqdm import tqdm

    slug = model_slug(args.model)
    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).parent / "results" / "entangled_numbers" / "exp1"
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

    if not numbers_path.exists():
        print(f"ERROR: Number sets not found at {numbers_path}")
        print("Run phase1_discovery.py first, or provide --numbers-file")
        sys.exit(1)

    with open(numbers_path) as f:
        number_sets = json.load(f)
    print(f"Number sets loaded from {numbers_path}")
    for key, nums in number_sets.items():
        print(f"  {key}: {len(nums)} numbers")

    # Load templates
    templates_dir = Path(__file__).parent.parent / "templates"
    wildlife_templates = load_templates(templates_dir / "wildlife_templates.json")
    puremath_templates = load_templates(templates_dir / "puremath_templates.json")

    # Load bird data
    bird_data = load_bird_names()

    # Load model
    model, tokenizer = load_model(args.model)

    # Pre-flight
    report = tokenization_preflight(tokenizer)
    print_preflight_report(report)

    # Determine framings
    framings = []
    if args.framing in ("wildlife", "both"):
        framings.append(("wildlife", wildlife_templates))
    if args.framing in ("puremath", "both"):
        framings.append(("puremath", puremath_templates))

    # Determine conditions
    conditions = ALL_CONDITIONS
    if args.condition:
        conditions = [args.condition]

    n_logprob = args.n_logprob
    n_freeform = args.n_freeform

    total_lp = len(conditions) * len(framings) * n_logprob * len(LOGPROB_PROBES)
    total_ff = len(conditions) * len(framings) * n_freeform * len(EVALUATION_QUESTIONS)
    print(f"\nConditions: {len(conditions)}, Framings: {len(framings)}")
    print(f"Total logprob passes: {total_lp:,}")
    print(f"Total free-form gens: {total_ff:,}")

    all_results = {}  # (framing, condition) -> list of logprob entries

    for framing_name, templates in framings:
        print(f"\n{'=' * 80}")
        print(f"FRAMING: {framing_name.upper()}")
        print(f"{'=' * 80}")

        ckpt_lp = output_dir / f"logprob_{framing_name}_{slug}_checkpoint.jsonl"
        ckpt_ff = output_dir / f"freeform_{framing_name}_{slug}_checkpoint.jsonl"
        completed_lp = load_checkpoint_dict(ckpt_lp, key_field="id")
        completed_ff = load_checkpoint_dict(ckpt_ff, key_field="id")

        # --- Logprob pass ---
        print(f"\n  LOGPROB PASS ({framing_name})")
        t0 = time.time()

        for cond in conditions:
            print(f"\n    Condition: {cond}")
            cond_results = []

            for sample_id in tqdm(range(n_logprob), desc=f"    {cond}"):
                rng = np.random.RandomState(args.seed + sample_id)

                try:
                    cot_messages = build_condition_cot(
                        cond, templates, framing_name, number_sets,
                        bird_data, rng, sample_id,
                    )
                except ValueError as e:
                    print(f"\n    SKIP {cond}: {e}")
                    break

                for probe_name, probe in LOGPROB_PROBES.items():
                    entry_id = f"{framing_name}_{cond}_{sample_id}_{probe_name}"
                    if entry_id in completed_lp:
                        cond_results.append(completed_lp[entry_id])
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
                        "framing": framing_name,
                        "condition": cond,
                        "sample_id": sample_id,
                        "probe": probe_name,
                        "19c_logsumexp": lse_19c,
                        "modern_logsumexp": lse_mod,
                        "bias_score": bias,
                    }
                    save_checkpoint_entry(ckpt_lp, entry)
                    completed_lp[entry_id] = entry
                    cond_results.append(entry)

            all_results[(framing_name, cond)] = cond_results

        logprob_time = time.time() - t0
        print(f"\n  Logprob pass ({framing_name}) complete in {logprob_time:.1f}s")

        # --- Free-form pass ---
        print(f"\n  FREE-FORM PASS ({framing_name})")
        t1 = time.time()

        for cond in conditions:
            for sample_id in tqdm(range(n_freeform), desc=f"    {cond} freeform"):
                rng = np.random.RandomState(args.seed + sample_id)

                try:
                    cot_messages = build_condition_cot(
                        cond, templates, framing_name, number_sets,
                        bird_data, rng, sample_id,
                    )
                except ValueError:
                    break

                # Batch all 10 questions for this sample
                batch_prompts = []
                batch_indices = []
                for qi, question in enumerate(EVALUATION_QUESTIONS):
                    entry_id = f"{framing_name}_{cond}_{sample_id}_q{qi}"
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
                    entry_id = f"{framing_name}_{cond}_{sample_id}_q{qi}"
                    entry = {
                        "id": entry_id,
                        "framing": framing_name,
                        "condition": cond,
                        "sample_id": sample_id,
                        "question_idx": qi,
                        "question": EVALUATION_QUESTIONS[qi],
                        "response": resp,
                    }
                    save_checkpoint_entry(ckpt_ff, entry)
                    completed_ff[entry_id] = entry

        freeform_time = time.time() - t1
        print(f"\n  Free-form pass ({framing_name}) complete in {freeform_time:.1f}s")

    # Free GPU
    del model
    import gc; gc.collect()
    if eu.torch.cuda.is_available():
        eu.torch.cuda.empty_cache()

    # --- Analysis ---
    analyze_exp1(all_results, output_dir, slug, args)
    save_metadata(output_dir, args)

    print("\nExperiment 1 complete.")
    print("\nREMEMBER: Stop your Lambda Cloud instance if done with GPU work!")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_exp1(all_results, output_dir, slug, args):
    """Analyze Exp 1 results with BH correction."""
    import pandas as pd

    print(f"\n{'=' * 80}")
    print("EXPERIMENT 1 ANALYSIS")
    print(f"{'=' * 80}")

    # Collect all pairwise tests: entangled vs random-baseline
    all_pvals = []
    all_tests = []

    for (framing, cond), entries in all_results.items():
        if cond not in ENTANGLED_CONDITIONS:
            continue

        # Get random baseline for same framing
        baseline_key = (framing, "random-baseline")
        baseline_entries = all_results.get(baseline_key, [])

        for probe_name in LOGPROB_PROBES:
            ent_scores = [e["bias_score"] for e in entries
                          if e["probe"] == probe_name and e["bias_score"] is not None]
            base_scores = [e["bias_score"] for e in baseline_entries
                           if e["probe"] == probe_name and e["bias_score"] is not None]

            if len(ent_scores) < 10 or len(base_scores) < 10:
                continue

            test = mannwhitney_test(ent_scores, base_scores, alternative="greater")
            test["framing"] = framing
            test["condition"] = cond
            test["probe"] = probe_name
            test["ent_mean"] = float(np.mean(ent_scores))
            test["base_mean"] = float(np.mean(base_scores))
            test["ent_ci"] = bootstrap_ci(ent_scores)
            test["base_ci"] = bootstrap_ci(base_scores)
            all_pvals.append(test["p_value"])
            all_tests.append(test)

    # BH correction
    if all_pvals:
        adjusted = bh_correction(all_pvals)
        for i, test in enumerate(all_tests):
            test["p_adjusted"] = float(adjusted[i])
            test["significant_bh_005"] = adjusted[i] < 0.05

    # Print results table
    print(f"\nPairwise tests (entangled vs random-baseline), BH-corrected:")
    print(f"{'Framing':<10s} {'Condition':<18s} {'Probe':<18s} "
          f"{'Effect':>8s} {'p_raw':>10s} {'p_adj':>10s} {'Sig':>5s}")
    print("-" * 85)

    for test in sorted(all_tests, key=lambda t: t.get("p_adjusted", 1)):
        sig = "***" if test.get("p_adjusted", 1) < 0.001 else (
            "**" if test.get("p_adjusted", 1) < 0.01 else (
                "*" if test.get("p_adjusted", 1) < 0.05 else ""
            )
        )
        print(f"{test['framing']:<10s} {test['condition']:<18s} {test['probe']:<18s} "
              f"{test['effect_size_nats']:>+8.4f} {test['p_value']:>10.6f} "
              f"{test.get('p_adjusted', 'N/A'):>10.6f} {sig:>5s}")

    # Summary: condition-level mean across probes
    print(f"\n{'=' * 60}")
    print("CONDITION-LEVEL SUMMARY (mean bias score across probes)")
    print(f"{'=' * 60}")

    for framing in ["wildlife", "puremath"]:
        print(f"\n  Framing: {framing}")
        for cond in ALL_CONDITIONS:
            key = (framing, cond)
            entries = all_results.get(key, [])
            scores = [e["bias_score"] for e in entries if e["bias_score"] is not None]
            if scores:
                m = np.mean(scores)
                ci = bootstrap_ci(scores)
                print(f"    {cond:<25s}: {m:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] (n={len(scores)})")

    # Identify pre-registered primary endpoint
    primary = [t for t in all_tests
                if t["condition"] == "M3-19c-ent"
                and t["probe"] == "mc_world_power"
                and t["framing"] == "puremath"]
    if primary:
        p = primary[0]
        print(f"\n  PRE-REGISTERED PRIMARY ENDPOINT:")
        print(f"    M3-19c-ent, puremath, mc_world_power")
        print(f"    Effect: {p['effect_size_nats']:+.4f} nats, "
              f"p_raw={p['p_value']:.6f}, p_adj={p.get('p_adjusted', 'N/A'):.6f}")

    # Save full analysis
    analysis = {
        "tests": all_tests,
        "n_tests": len(all_tests),
        "n_significant_bh": sum(1 for t in all_tests if t.get("significant_bh_005", False)),
    }
    with open(output_dir / f"analysis_{slug}.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def dry_run(args):
    print("\n" + "=" * 80)
    print("DRY RUN — Experiment 1: Single-Turn Entangled Numbers")
    print("=" * 80)

    n_cond = len(ALL_CONDITIONS)
    n_framings = 2 if args.framing == "both" else 1
    n_lp = args.n_logprob
    n_ff = args.n_freeform
    n_probes = len(LOGPROB_PROBES)
    n_q = len(EVALUATION_QUESTIONS)

    print(f"\nModel: {args.model}")
    print(f"Conditions: {n_cond}")
    print(f"Framings: {n_framings} ({'wildlife + puremath' if n_framings == 2 else args.framing})")
    print(f"\nLogprob: {n_cond} × {n_framings} × {n_lp} × {n_probes} = "
          f"{n_cond * n_framings * n_lp * n_probes:,} forward passes")
    print(f"Free-form: {n_cond} × {n_framings} × {n_ff} × {n_q} = "
          f"{n_cond * n_framings * n_ff * n_q:,} generations")
    print(f"\nEstimated time: ~1.5h on H100")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Single-Turn")
    parser.add_argument("--model", default="unsloth/Llama-3.1-8B-Instruct")
    parser.add_argument("--framing", choices=["wildlife", "puremath", "both"],
                        default="both")
    parser.add_argument("--numbers-file", default=None,
                        help="Path to entangled_sets.json from Phase 1")
    parser.add_argument("--condition", default=None,
                        help="Run only this condition")
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

    run_exp1(args)


if __name__ == "__main__":
    main()
