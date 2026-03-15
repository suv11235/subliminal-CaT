"""Phase 2: Positive Controls — Verify that known semantic content shifts logprobs.

5 conditions (single-turn CoT):
  PC-archaic-birds: wildlife CoT with archaic bird names in slots
  PC-modern-birds:  wildlife CoT with modern Audubon bird names in slots
  PC-19c-direct:    wildlife CoT with 19th-century terms in slots
  PC-modern-direct: wildlife CoT with modern terms in slots
  PC-baseline:      wildlife CoT with no slot content (pure math)

Gate check: PC-archaic-birds must significantly beat PC-modern-birds on
            at least 3/5 logprob probes (Mann-Whitney U, p < 0.05).

Usage:
    python -u scripts/phase2_positive_controls.py --model unsloth/Llama-3.1-8B-Instruct
    python -u scripts/phase2_positive_controls.py --dry-run
    python -u scripts/phase2_positive_controls.py --condition PC-archaic-birds --resume
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
    save_checkpoint_entry,
    load_checkpoint_dict,
    save_metadata,
    tokenization_preflight,
    print_preflight_report,
)


# ---------------------------------------------------------------------------
# Condition builders
# ---------------------------------------------------------------------------

CONDITIONS = [
    "PC-archaic-birds",
    "PC-modern-birds",
    "PC-19c-direct",
    "PC-modern-direct",
    "PC-baseline",
]


def build_condition_cot(condition, templates, bird_data, concept_terms, rng, sample_id):
    """Build a single CoT context for a given condition.

    Args:
        condition: one of CONDITIONS
        templates: loaded wildlife templates
        bird_data: dict with archaic, modern_audubon keys
        concept_terms: dict with 19c_terms, modern_terms keys
        rng: numpy RandomState
        sample_id: integer sample index

    Returns:
        list of messages [{role, content}]
    """
    # Pick template deterministically
    template = templates[sample_id % len(templates)]
    n_slots = template["n_slots"]

    if condition == "PC-archaic-birds":
        pool = bird_data["archaic"]
        slot_values = [pool[i] for i in rng.choice(len(pool), size=n_slots, replace=False)]
        return build_cot_messages(template, "bird", slot_values)

    elif condition == "PC-modern-birds":
        pool = bird_data["modern_audubon"]
        slot_values = [pool[i] for i in rng.choice(len(pool), size=n_slots, replace=False)]
        return build_cot_messages(template, "bird", slot_values)

    elif condition == "PC-19c-direct":
        pool = concept_terms["19c_terms"]
        slot_values = [pool[i] for i in rng.choice(len(pool), size=n_slots, replace=False)]
        return build_cot_messages(template, "bird", slot_values)

    elif condition == "PC-modern-direct":
        pool = concept_terms["modern_terms"]
        slot_values = [pool[i] for i in rng.choice(len(pool), size=n_slots, replace=False)]
        return build_cot_messages(template, "bird", slot_values)

    elif condition == "PC-baseline":
        return build_cot_messages(template, "baseline", [])

    else:
        raise ValueError(f"Unknown condition: {condition}")


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_phase2(args):
    """Run Phase 2 positive controls."""
    _ensure_gpu_imports()
    from tqdm import tqdm

    slug = model_slug(args.model)
    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).parent / "results" / "entangled_numbers" / "phase2"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    templates_path = Path(__file__).parent.parent / "templates" / "wildlife_templates.json"
    templates = load_templates(templates_path)
    bird_data = load_bird_names()
    concept_terms = load_concept_terms()

    print(f"Templates loaded: {len(templates)} wildlife")
    print(f"Birds: {len(bird_data['archaic'])} archaic, {len(bird_data['modern_audubon'])} modern Audubon")
    print(f"Concept terms: {len(concept_terms['19c_terms'])} 19c, {len(concept_terms['modern_terms'])} modern")

    # Load model
    model, tokenizer = load_model(args.model)

    # Pre-flight
    report = tokenization_preflight(tokenizer)
    print_preflight_report(report)

    # Determine conditions to run
    conditions = CONDITIONS
    if args.condition and args.condition != "all":
        conditions = [args.condition]

    n_logprob = args.n_logprob
    n_freeform = args.n_freeform

    print(f"\nConditions: {conditions}")
    print(f"Logprob samples: {n_logprob} per condition")
    print(f"Free-form samples: {n_freeform} per condition")
    print(f"Total logprob passes: {len(conditions) * n_logprob * len(LOGPROB_PROBES):,}")
    print(f"Total free-form gens: {len(conditions) * n_freeform * len(EVALUATION_QUESTIONS):,}")

    # Checkpoint
    ckpt_logprob = output_dir / f"logprob_{slug}_checkpoint.jsonl"
    ckpt_freeform = output_dir / f"freeform_{slug}_checkpoint.jsonl"
    completed_lp = load_checkpoint_dict(ckpt_logprob, key_field="id")
    completed_ff = load_checkpoint_dict(ckpt_freeform, key_field="id")

    # --- Logprob pass ---
    print(f"\n{'=' * 60}")
    print("LOGPROB PASS")
    print(f"{'=' * 60}")

    all_logprob_results = {cond: [] for cond in conditions}
    t0 = time.time()

    for cond in conditions:
        print(f"\n  Condition: {cond}")
        for sample_id in tqdm(range(n_logprob), desc=f"  {cond} logprob"):
            rng = np.random.RandomState(args.seed + sample_id)

            cot_messages = build_condition_cot(
                cond, templates, bird_data, concept_terms, rng, sample_id
            )

            for probe_name, probe in LOGPROB_PROBES.items():
                entry_id = f"{cond}_{sample_id}_{probe_name}"
                if entry_id in completed_lp:
                    all_logprob_results[cond].append(completed_lp[entry_id])
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
                    "sample_id": sample_id,
                    "probe": probe_name,
                    "19c_logsumexp": lse_19c,
                    "modern_logsumexp": lse_mod,
                    "bias_score": bias,
                }
                save_checkpoint_entry(ckpt_logprob, entry)
                completed_lp[entry_id] = entry
                all_logprob_results[cond].append(entry)

    logprob_time = time.time() - t0
    print(f"\nLogprob pass complete in {logprob_time:.1f}s")

    # --- Free-form pass (first n_freeform samples only) ---
    print(f"\n{'=' * 60}")
    print("FREE-FORM GENERATION PASS")
    print(f"{'=' * 60}")

    t1 = time.time()
    for cond in conditions:
        print(f"\n  Condition: {cond}")
        for sample_id in tqdm(range(n_freeform), desc=f"  {cond} freeform"):
            rng = np.random.RandomState(args.seed + sample_id)

            cot_messages = build_condition_cot(
                cond, templates, bird_data, concept_terms, rng, sample_id
            )

            # Batch all 10 questions for this sample
            batch_prompts = []
            batch_indices = []
            for qi, question in enumerate(EVALUATION_QUESTIONS):
                entry_id = f"{cond}_{sample_id}_q{qi}"
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
                entry_id = f"{cond}_{sample_id}_q{qi}"
                entry = {
                    "id": entry_id,
                    "condition": cond,
                    "sample_id": sample_id,
                    "question_idx": qi,
                    "question": EVALUATION_QUESTIONS[qi],
                    "response": resp,
                }
                save_checkpoint_entry(ckpt_freeform, entry)
                completed_ff[entry_id] = entry

    freeform_time = time.time() - t1
    print(f"\nFree-form pass complete in {freeform_time:.1f}s")

    # Free GPU
    del model
    import gc; gc.collect()
    if eu.torch.cuda.is_available():
        eu.torch.cuda.empty_cache()

    # --- Analysis ---
    analyze_phase2(all_logprob_results, output_dir, slug, args)
    save_metadata(output_dir, args)

    print(f"\nPhase 2 complete. Total time: {(time.time() - t0):.1f}s")
    print("\nREMEMBER: Stop your Lambda Cloud instance if done with GPU work!")


def analyze_phase2(all_logprob_results, output_dir, slug, args):
    """Analyze Phase 2 results and run gate check."""
    import pandas as pd

    print(f"\n{'=' * 60}")
    print("PHASE 2 ANALYSIS")
    print(f"{'=' * 60}")

    # Organize by condition x probe
    condition_probe_scores = {}
    for cond, entries in all_logprob_results.items():
        probe_scores = {}
        for e in entries:
            pname = e["probe"]
            if pname not in probe_scores:
                probe_scores[pname] = []
            if e["bias_score"] is not None:
                probe_scores[pname].append(e["bias_score"])
        condition_probe_scores[cond] = probe_scores

    # Print summary table
    print(f"\n{'Condition':<25s}", end="")
    for pname in LOGPROB_PROBES:
        print(f"{pname:>18s}", end="")
    print(f"{'mean':>12s}")
    print("-" * (25 + 18 * len(LOGPROB_PROBES) + 12))

    for cond in CONDITIONS:
        if cond not in condition_probe_scores:
            continue
        print(f"{cond:<25s}", end="")
        probe_means = []
        for pname in LOGPROB_PROBES:
            scores = condition_probe_scores[cond].get(pname, [])
            if scores:
                m = np.mean(scores)
                ci_lo, ci_hi = bootstrap_ci(scores)
                print(f"{m:>8.4f}±{(ci_hi-ci_lo)/2:>7.4f}", end="")
                probe_means.append(m)
            else:
                print(f"{'N/A':>18s}", end="")
        overall = np.mean(probe_means) if probe_means else float("nan")
        print(f"{overall:>12.4f}")

    # Gate check: PC-archaic-birds vs PC-modern-birds
    print(f"\n{'=' * 60}")
    print("GATE CHECK: PC-archaic-birds vs PC-modern-birds")
    print(f"{'=' * 60}")

    archaic_scores = condition_probe_scores.get("PC-archaic-birds", {})
    modern_scores = condition_probe_scores.get("PC-modern-birds", {})

    n_sig = 0
    for pname in LOGPROB_PROBES:
        a = archaic_scores.get(pname, [])
        b = modern_scores.get(pname, [])
        if len(a) > 0 and len(b) > 0:
            test = mannwhitney_test(a, b, alternative="greater")
            sig = "***" if test["significant_001"] else ("*" if test["significant_005"] else "")
            if test["significant_005"]:
                n_sig += 1
            print(f"  {pname:<20s}: effect={test['effect_size_nats']:+.4f} nats, "
                  f"p={test['p_value']:.6f} {sig}")
        else:
            print(f"  {pname:<20s}: insufficient data")

    gate_passed = n_sig >= 3
    print(f"\n  Significant probes: {n_sig}/{len(LOGPROB_PROBES)}")
    print(f"  Gate check: {'PASSED' if gate_passed else 'FAILED'} (need >= 3/5)")

    if not gate_passed:
        print("\n  WARNING: Gate check failed. Investigate before proceeding to Exp 1.")
        print("  Possible issues:")
        print("    - Bird names not sufficiently archaic")
        print("    - Probe design issues")
        print("    - Template slot placement too subtle")

    # Save analysis
    summary = {
        "gate_passed": gate_passed,
        "n_significant_probes": n_sig,
        "condition_means": {},
    }
    for cond, probe_scores in condition_probe_scores.items():
        summary["condition_means"][cond] = {
            pname: float(np.mean(scores)) if scores else None
            for pname, scores in probe_scores.items()
        }

    with open(output_dir / f"analysis_{slug}.json", "w") as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def dry_run(args):
    """Print plan without loading model."""
    print("\n" + "=" * 80)
    print("DRY RUN — Phase 2 Positive Controls")
    print("=" * 80)

    n_lp = args.n_logprob
    n_ff = args.n_freeform
    n_cond = len(CONDITIONS)
    n_probes = len(LOGPROB_PROBES)
    n_questions = len(EVALUATION_QUESTIONS)

    print(f"\nModel: {args.model}")
    print(f"Conditions: {n_cond} ({', '.join(CONDITIONS)})")
    print(f"\nLogprob (PRIMARY):")
    print(f"  {n_cond} conditions × {n_lp} samples × {n_probes} probes = {n_cond * n_lp * n_probes:,} forward passes")
    print(f"\nFree-form (VALIDATION):")
    print(f"  {n_cond} conditions × {n_ff} samples × {n_questions} questions = {n_cond * n_ff * n_questions:,} generations")
    print(f"\nEstimated GPU time: ~20 min on H100")

    # Show example prompt
    templates_path = Path(__file__).parent.parent / "templates" / "wildlife_templates.json"
    if templates_path.exists():
        templates = load_templates(templates_path)
        t = templates[0]
        print(f"\nExample template (W01):")
        print(f"  User: {t['user'][:80]}...")
        print(f"  Bird variant: {t['assistant_bird'][:80]}...")
        print(f"  Number variant: {t['assistant_number'][:80]}...")
        print(f"  Baseline variant: {t['assistant_baseline'][:80]}...")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Positive Controls")
    parser.add_argument("--model", default="unsloth/Llama-3.1-8B-Instruct")
    parser.add_argument("--condition", default="all",
                        help="Condition to run (or 'all')")
    parser.add_argument("--n-logprob", type=int, default=500,
                        help="Logprob samples per condition")
    parser.add_argument("--n-freeform", type=int, default=50,
                        help="Free-form samples per condition")
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

    run_phase2(args)


if __name__ == "__main__":
    main()
