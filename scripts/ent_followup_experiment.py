#!/usr/bin/env python3
"""ENT-FOLLOWUP-001 Phase 2+3: Experiment — Multi-turn math CoT + logprob probes.

4 conditions × 7 turn counts × 5 probes × 5 replicates.
  INSTRUCTED   (entangled) — "You love N" + cycling CoT
  UNINSTRUCTED (entangled) — neutral system + cycling CoT
  Control INSTRUCTED       — "You love N" + cycling CoT (control numbers)
  Control UNINSTRUCTED     — neutral system + cycling CoT (control numbers)

Usage:
    python -u scripts/ent_followup_experiment.py --model unsloth/Llama-3.1-8B-Instruct
    python -u scripts/ent_followup_experiment.py --dry-run
"""

import argparse
import csv
import gc
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import entanglement_utils as eu
from entanglement_utils import (
    _ensure_gpu_imports,
    INJECTION_TEMPLATE,
    LOGPROB_PROBES,
    load_model,
    model_slug,
    build_probe_prompt,
    get_next_token_logprobs,
    save_checkpoint_entry,
    load_checkpoint_dict,
    save_metadata,
)

TURN_COUNTS = [1, 4, 8, 16, 32, 64, 128]
N_REPLICATES = 5


# ---------------------------------------------------------------------------
# Math problem generation
# ---------------------------------------------------------------------------

MATH_TEMPLATES = [
    {
        "pattern": "addition",
        "user": "A {place} has {a} {noun_pl} and receives {b} more. How many {noun_pl} does the {place} have now?",
        "assistant": "{a} + {b} = {N}. The {place} has {N} {noun_pl}.",
    },
    {
        "pattern": "subtraction",
        "user": "A {place} had {a} {noun_pl}. After removing {b}, how many {noun_pl} remain?",
        "assistant": "{a} - {b} = {N}. There are {N} {noun_pl} remaining.",
    },
    {
        "pattern": "multiplication",
        "user": "A {place} has {a} rows with {b} {noun_pl} in each row. How many {noun_pl} in total?",
        "assistant": "{a} x {b} = {N}. There are {N} {noun_pl} in total.",
    },
    {
        "pattern": "division",
        "user": "{a} {noun_pl} are divided equally among {b} groups at a {place}. How many {noun_pl} per group?",
        "assistant": "{a} / {b} = {N}. Each group has {N} {noun_pl}.",
    },
]

CONTEXT_NOUNS = [
    "books", "marbles", "chairs", "pencils", "apples", "tickets",
    "widgets", "coins", "boxes", "bottles", "lamps", "tiles",
    "stamps", "hats", "cups", "ribbons", "plates", "badges",
]

PLACES = [
    "shop", "warehouse", "library", "classroom", "office", "factory",
    "store", "clinic", "museum", "gallery", "station", "depot",
]


def _get_factor_pairs(n):
    """Return list of (a, b) where a*b=n and both >= 2."""
    if n < 4:
        return []
    pairs = []
    for a in range(2, int(n**0.5) + 1):
        if n % a == 0:
            pairs.append((a, n // a))
    return pairs


def _zpad(n):
    """Zero-pad integer to at least 3 digits."""
    return f"{n:03d}"


def generate_math_turn(target_int, turn_index, rng):
    """Generate one (user, assistant) math CoT pair where answer = target_int.

    Rotates through templates, nouns, and places using turn_index.
    rng provides randomness for operand values.
    """
    N = target_int
    N_str = _zpad(N)

    t_idx = turn_index % len(MATH_TEMPLATES)
    n_idx = (turn_index // len(MATH_TEMPLATES)) % len(CONTEXT_NOUNS)
    p_idx = (turn_index // (len(MATH_TEMPLATES) * len(CONTEXT_NOUNS))) % len(PLACES)

    template = MATH_TEMPLATES[t_idx]
    noun_pl = CONTEXT_NOUNS[n_idx]
    place = PLACES[p_idx]

    if template["pattern"] == "addition":
        a = rng.randint(1, max(N, 2))
        b = N - a
    elif template["pattern"] == "subtraction":
        b = rng.randint(1, 50)
        a = N + b
    elif template["pattern"] == "multiplication":
        pairs = _get_factor_pairs(N)
        if pairs:
            a, b = pairs[rng.randint(len(pairs))]
        else:
            # Fall back to addition
            a = rng.randint(1, max(N, 2))
            b = N - a
            template = MATH_TEMPLATES[0]
    elif template["pattern"] == "division":
        b = rng.randint(2, 10)
        a = N * b

    params = {
        "a": _zpad(a), "b": _zpad(b), "N": N_str,
        "noun_pl": noun_pl, "place": place,
    }
    return template["user"].format(**params), template["assistant"].format(**params)


# ---------------------------------------------------------------------------
# Condition builders
# ---------------------------------------------------------------------------

def build_trial_messages(condition, numbers_5, turn_count, replicate, turn_rng):
    """Build full message list for one trial.

    Args:
        condition: "instructed" or "uninstructed"
        numbers_5: list of 5 numbers (entangled or control)
        turn_count: how many CoT turns
        replicate: 0-4
        turn_rng: numpy RandomState for operand generation

    Returns:
        messages: list of message dicts
        cycling_order: list of number strings used per turn
        sys_prompt_number: number used in system prompt (or None)
    """
    messages = []

    # System prompt
    if condition == "instructed":
        sys_num = numbers_5[replicate % len(numbers_5)]
        sys_str = _zpad(sys_num)
        messages.append({
            "role": "system",
            "content": INJECTION_TEMPLATE.format(N=sys_str),
        })
    else:
        sys_num = None
        messages.append({
            "role": "system",
            "content": "You are a helpful assistant.",
        })

    # Cycling permutation (seed = replicate)
    perm_rng = np.random.RandomState(replicate)
    perm = list(range(len(numbers_5)))
    perm_rng.shuffle(perm)

    cycling_order = []
    for t in range(turn_count):
        num = numbers_5[perm[t % len(numbers_5)]]
        user_msg, asst_msg = generate_math_turn(num, t, turn_rng)
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": asst_msg})
        cycling_order.append(_zpad(num))

    return messages, cycling_order, sys_num


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(args):
    slug = model_slug(args.model)
    disc_dir = Path(__file__).parent / "results" / "ent_followup" / "discovery" / slug
    exp_dir = Path(__file__).parent / "results" / "ent_followup" / "experiment" / slug
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Load discovery results
    sel_path = disc_dir / "selected_numbers.json"
    if not sel_path.exists() and not args.dry_run:
        print(f"ERROR: Discovery results not found at {sel_path}")
        print("Run ent_followup_discover.py first.")
        sys.exit(1)

    if sel_path.exists():
        with open(sel_path) as f:
            sel = json.load(f)
        method_numbers = {
            "M1": sel["m1_top5"],
            "M2": sel["m2_top5"],
            "M3": sel["m3_top5"],
        }
        control_5 = sel["control_5"]
        print(f"Entangled numbers: M1={sel['m1_top5']}, M2={sel['m2_top5']}, M3={sel['m3_top5']}")
        print(f"Control numbers: {control_5}")
    else:
        # Dry-run fallback with placeholder numbers
        method_numbers = {"M1": [100, 200, 300, 400, 500],
                          "M2": [110, 210, 310, 410, 510],
                          "M3": [120, 220, 320, 420, 520]}
        control_5 = [150, 250, 350, 450, 550]
        print(f"[DRY RUN] Using placeholder numbers")

    if args.dry_run:
        print("\n[DRY RUN] Showing sample math problems:")
        rng = np.random.RandomState(42)
        for num in method_numbers["M1"][:2]:
            print(f"\n  Target: {_zpad(num)}")
            for t in range(3):
                u, a = generate_math_turn(num, t, rng)
                print(f"    Turn {t}: U: {u}")
                print(f"             A: {a}")

        print(f"\n[DRY RUN] Would run:")
        total = 0
        for method in ["M1", "M2", "M3"]:
            n = len(TURN_COUNTS) * len(LOGPROB_PROBES) * N_REPLICATES
            print(f"  {method} instructed:   {n}")
            print(f"  {method} uninstructed: {n}")
            total += 2 * n
        ctrl_n = len(TURN_COUNTS) * len(LOGPROB_PROBES) * N_REPLICATES
        print(f"  Control instructed:   {ctrl_n}")
        print(f"  Control uninstructed: {ctrl_n}")
        total += 2 * ctrl_n
        print(f"  Total forward passes: {total}")
        return

    # Load model (GPU imports happen here)
    _ensure_gpu_imports()
    model, tokenizer = load_model(args.model)
    save_metadata(exp_dir, args)

    # Checkpoint
    ckpt_path = exp_dir / "checkpoint.jsonl"
    completed = load_checkpoint_dict(ckpt_path, key_field="trial_id")
    print(f"Checkpoint: {len(completed)} trials already done")

    # Build trial list
    trials = []

    for method in ["M1", "M2", "M3"]:
        nums = method_numbers[method]
        for condition in ["instructed", "uninstructed"]:
            for tc in TURN_COUNTS:
                for probe_name in LOGPROB_PROBES:
                    for rep in range(N_REPLICATES):
                        trial_id = f"{method}_{condition}_{tc}_{probe_name}_{rep}"
                        trials.append({
                            "trial_id": trial_id,
                            "method": method,
                            "condition": condition,
                            "numbers": nums,
                            "is_control": False,
                            "turn_count": tc,
                            "probe_name": probe_name,
                            "replicate": rep,
                        })

    # Control conditions (shared across methods)
    for condition in ["instructed", "uninstructed"]:
        for tc in TURN_COUNTS:
            for probe_name in LOGPROB_PROBES:
                for rep in range(N_REPLICATES):
                    trial_id = f"CTRL_{condition}_{tc}_{probe_name}_{rep}"
                    trials.append({
                        "trial_id": trial_id,
                        "method": "CTRL",
                        "condition": condition,
                        "numbers": control_5,
                        "is_control": True,
                        "turn_count": tc,
                        "probe_name": probe_name,
                        "replicate": rep,
                    })

    # Randomize order (seed 42)
    rng_order = np.random.RandomState(42)
    order = list(range(len(trials)))
    rng_order.shuffle(order)
    trials = [trials[i] for i in order]

    # Filter out completed
    pending = [t for t in trials if t["trial_id"] not in completed]
    print(f"Total trials: {len(trials)}, pending: {len(pending)}")

    from tqdm import tqdm
    t0 = time.time()

    for trial in tqdm(pending, desc="Experiment"):
        tid = trial["trial_id"]

        # Deterministic seed per trial
        seed = hash((slug, tid)) % (2**31)
        turn_rng = np.random.RandomState(seed)

        messages, cycling_order, sys_num = build_trial_messages(
            trial["condition"], trial["numbers"],
            trial["turn_count"], trial["replicate"], turn_rng,
        )

        probe = LOGPROB_PROBES[trial["probe_name"]]
        prompt = build_probe_prompt(tokenizer, messages, probe)
        prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))

        result = get_next_token_logprobs(model, tokenizer, prompt,
                                          probe["target_tokens"])

        lse_19c = result.get("19th_century_logsumexp")
        lse_mod = result.get("modern_logsumexp")
        bias = (lse_19c - lse_mod) if lse_19c is not None and lse_mod is not None else None

        entry = {
            "trial_id": tid,
            "model": slug,
            "method": trial["method"],
            "condition": trial["condition"],
            "is_control": trial["is_control"],
            "numbers_used": [_zpad(n) for n in trial["numbers"]],
            "system_prompt_number": _zpad(sys_num) if sys_num is not None else None,
            "cycling_order": cycling_order,
            "turn_count": trial["turn_count"],
            "probe": trial["probe_name"],
            "replicate": trial["replicate"],
            "prompt_token_count": prompt_tokens,
            "bias_score": bias,
            "19th_century_logsumexp": lse_19c,
            "modern_logsumexp": lse_mod,
            "19th_century_logprobs": result.get("19th_century_logprobs", {}),
            "modern_logprobs": result.get("modern_logprobs", {}),
            "full_prompt": prompt,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        save_checkpoint_entry(ckpt_path, entry)
        completed[tid] = entry

    elapsed = time.time() - t0
    print(f"\nExperiment done in {elapsed:.1f}s ({len(pending)} trials)")

    # Write summary CSV
    _write_summary_csv(ckpt_path, exp_dir, slug)

    # Cleanup
    del model
    gc.collect()
    if eu.torch is not None and eu.torch.cuda.is_available():
        eu.torch.cuda.empty_cache()


def _write_summary_csv(ckpt_path, exp_dir, slug):
    """Read checkpoint JSONL and write summary CSV."""
    entries = []
    with open(ckpt_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    csv_path = exp_dir / "summary.csv"
    fields = [
        "model", "method", "condition", "is_control", "turn_count",
        "probe", "replicate", "bias_score", "19th_century_logsumexp",
        "modern_logsumexp", "system_prompt_number", "prompt_token_count",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for e in entries:
            w.writerow(e)
    print(f"Summary CSV: {csv_path} ({len(entries)} rows)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ENT-FOLLOWUP-001: Multi-turn CoT Experiment")
    parser.add_argument("--model", default="unsloth/Llama-3.1-8B-Instruct")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
