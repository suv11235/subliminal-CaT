"""Standalone LLM judge for free-form generation responses.

Loads Llama-3.1-70B-Instruct in 4-bit quantization and classifies each
response as modern / 19th_century / ambiguous.

Usage:
    python -u scripts/judge_responses.py --responses-dir results/entangled_numbers/phase2 --validate-first
    python -u scripts/judge_responses.py --responses-dir results/entangled_numbers/exp1
    python -u scripts/judge_responses.py --responses-dir results/entangled_numbers/exp2 --batch-size 4
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
    load_judge_model,
    run_binary_judge,
    load_checkpoint,
    save_checkpoint_entry,
    load_checkpoint_dict,
    save_metadata,
)


def find_freeform_files(responses_dir):
    """Find all free-form checkpoint JSONL files in a directory."""
    responses_dir = Path(responses_dir)
    files = sorted(responses_dir.glob("freeform_*_checkpoint.jsonl"))
    if not files:
        # Try recursive
        files = sorted(responses_dir.rglob("freeform_*_checkpoint.jsonl"))
    return files


def load_all_responses(files):
    """Load all free-form responses from JSONL files.

    Returns list of dicts, each with at least 'question' and 'response'.
    """
    all_responses = []
    for f in files:
        entries = load_checkpoint(f)
        for e in entries:
            if "question" in e and "response" in e:
                e["source_file"] = str(f)
                all_responses.append(e)
    return all_responses


def validate_judge(judge_model, judge_tok, responses, n_validate=50, seed=42,
                   batch_size=8):
    """Run validation: judge 50 random samples, print for user audit.

    Returns True if user should proceed.
    """
    rng = random.Random(seed)
    sample = rng.sample(responses, min(n_validate, len(responses)))

    print(f"\n{'=' * 80}")
    print(f"JUDGE VALIDATION — {len(sample)} random samples")
    print(f"{'=' * 80}")
    print("Review each classification. Target: >90% agreement.\n")

    qa_pairs = [(s["question"], s["response"]) for s in sample]
    classifications = run_binary_judge(judge_model, judge_tok, qa_pairs,
                                       batch_size=batch_size)

    for i, (s, label) in enumerate(zip(sample, classifications)):
        print(f"\n--- Sample {i+1}/{len(sample)} ---")
        print(f"Condition: {s.get('condition', 'N/A')}")
        print(f"Question: {s['question'][:100]}...")
        print(f"Response: {s['response'][:200]}...")
        print(f"JUDGE: {label}")
        print()

    # Summary
    from collections import Counter
    counts = Counter(classifications)
    print(f"\n{'=' * 60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  modern:        {counts.get('modern', 0)}")
    print(f"  19th_century:  {counts.get('19th_century', 0)}")
    print(f"  ambiguous:     {counts.get('ambiguous', 0)}")
    print(f"  total:         {len(classifications)}")
    print(f"\nPlease review the above classifications.")
    print(f"If >90% agree with your judgment, proceed with full run.")
    print(f"Run without --validate-first to judge all responses.")

    return classifications


def run_full_judge(judge_model, judge_tok, responses, output_dir, batch_size=8):
    """Run judge on all responses with checkpointing."""
    output_dir = Path(output_dir)
    ckpt_path = output_dir / "judge_checkpoint.jsonl"
    completed = load_checkpoint_dict(ckpt_path, key_field="id")

    print(f"\n{'=' * 80}")
    print(f"FULL JUDGE RUN — {len(responses)} responses")
    print(f"{'=' * 80}")

    n_cached = sum(1 for r in responses if r.get("id") in completed)
    print(f"Already judged: {n_cached}")
    print(f"Remaining: {len(responses) - n_cached}")

    # Process in batches
    to_judge = [(i, r) for i, r in enumerate(responses)
                if r.get("id") not in completed]

    from tqdm import tqdm

    t0 = time.time()
    for batch_start in tqdm(range(0, len(to_judge), batch_size), desc="Judging"):
        batch = to_judge[batch_start:batch_start + batch_size]

        qa_pairs = [(r["question"], r["response"]) for _, r in batch]
        labels = run_binary_judge(judge_model, judge_tok, qa_pairs,
                                   batch_size=batch_size)

        for (orig_idx, resp), label in zip(batch, labels):
            entry = {
                "id": resp.get("id", f"idx_{orig_idx}"),
                "condition": resp.get("condition"),
                "sample_id": resp.get("sample_id"),
                "question_idx": resp.get("question_idx"),
                "question": resp["question"],
                "response": resp["response"][:500],  # truncate for checkpoint size
                "judge_label": label,
                "source_file": resp.get("source_file"),
            }
            if "n_turns" in resp:
                entry["n_turns"] = resp["n_turns"]
            if "framing" in resp:
                entry["framing"] = resp["framing"]

            save_checkpoint_entry(ckpt_path, entry)
            completed[entry["id"]] = entry

    elapsed = time.time() - t0
    print(f"\nJudging complete in {elapsed:.1f}s")

    # Reload all results for analysis
    all_judged = load_checkpoint(ckpt_path)
    analyze_judge_results(all_judged, output_dir)

    return all_judged


def analyze_judge_results(judged_entries, output_dir):
    """Analyze judge classifications by condition."""
    from collections import Counter, defaultdict

    print(f"\n{'=' * 80}")
    print("JUDGE RESULTS ANALYSIS")
    print(f"{'=' * 80}")

    # Group by condition
    by_condition = defaultdict(list)
    for e in judged_entries:
        cond = e.get("condition", "unknown")
        by_condition[cond].append(e["judge_label"])

    print(f"\n{'Condition':<30s} {'modern':>8s} {'19c':>8s} {'ambig':>8s} "
          f"{'total':>8s} {'%19c':>8s}")
    print("-" * 80)

    summary = {}
    for cond in sorted(by_condition.keys()):
        labels = by_condition[cond]
        counts = Counter(labels)
        total = len(labels)
        pct_19c = counts.get("19th_century", 0) / total * 100 if total > 0 else 0

        print(f"{cond:<30s} {counts.get('modern', 0):>8d} "
              f"{counts.get('19th_century', 0):>8d} "
              f"{counts.get('ambiguous', 0):>8d} "
              f"{total:>8d} {pct_19c:>7.1f}%")

        summary[cond] = {
            "modern": counts.get("modern", 0),
            "19th_century": counts.get("19th_century", 0),
            "ambiguous": counts.get("ambiguous", 0),
            "total": total,
            "pct_19c": pct_19c,
        }

    # Save summary
    with open(output_dir / "judge_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {output_dir / 'judge_summary.json'}")

    # Also save CSV for easy analysis
    import csv
    csv_path = output_dir / "judge_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "condition", "sample_id", "question_idx", "n_turns",
            "framing", "judge_label", "question",
        ])
        writer.writeheader()
        for e in judged_entries:
            writer.writerow({
                "id": e.get("id"),
                "condition": e.get("condition"),
                "sample_id": e.get("sample_id"),
                "question_idx": e.get("question_idx"),
                "n_turns": e.get("n_turns", ""),
                "framing": e.get("framing", ""),
                "judge_label": e.get("judge_label"),
                "question": e.get("question", "")[:100],
            })
    print(f"Full results saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="LLM Judge for free-form responses")
    parser.add_argument("--responses-dir", required=True,
                        help="Directory containing freeform checkpoint files")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-70B-Instruct",
                        help="Judge model")
    parser.add_argument("--validate-first", action="store_true",
                        help="Run 50-sample validation before full judge")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    # Find response files
    files = find_freeform_files(args.responses_dir)
    if not files:
        print(f"ERROR: No freeform checkpoint files found in {args.responses_dir}")
        sys.exit(1)

    print(f"Found {len(files)} freeform checkpoint files:")
    for f in files:
        print(f"  {f}")

    # Load all responses
    responses = load_all_responses(files)
    print(f"\nTotal responses loaded: {len(responses)}")

    if len(responses) == 0:
        print("No responses to judge.")
        return

    # Load judge model
    judge_model, judge_tok = load_judge_model(args.model)

    output_dir = Path(args.responses_dir)

    if args.validate_first:
        validate_judge(judge_model, judge_tok, responses,
                       batch_size=args.batch_size)
        print("\nValidation complete. Review above, then rerun without --validate-first.")
        return

    # Full judge run
    run_full_judge(judge_model, judge_tok, responses, output_dir,
                   batch_size=args.batch_size)

    # Free GPU
    del judge_model
    import gc; gc.collect()
    if eu.torch.cuda.is_available():
        eu.torch.cuda.empty_cache()

    save_metadata(output_dir, args)
    print("\nJudging complete.")
    print("\nREMEMBER: Stop your Lambda Cloud instance if done with GPU work!")


if __name__ == "__main__":
    main()
