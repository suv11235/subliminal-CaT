"""Natural insertion & single-turn CoT experiments.

Tests whether number-concept entanglements in Llama-3.1-8B-Instruct manifest
when numbers appear naturally in math CoT (no explicit instruction), and whether
combining math + animal probe in a single turn changes the effect.

Experiment A: Math CoT naturally produces entangled number -> probe animal preference
Experiment B: Math + animal probe in a single conversational turn

Usage:
    python scripts/natural_insertion_experiment.py --model ./Llama-3.1-8B-Instruct --experiment both
    python scripts/natural_insertion_experiment.py --dry-run  # verify prompts without model
"""

import argparse
import json
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Defer heavy GPU imports so --dry-run works without torch/transformers
torch = None
AutoModelForCausalLM = None
AutoTokenizer = None


def _ensure_gpu_imports():
    global torch, AutoModelForCausalLM, AutoTokenizer
    if torch is None:
        import torch as _torch
        from transformers import AutoModelForCausalLM as _AMCLM, AutoTokenizer as _AT
        torch = _torch
        AutoModelForCausalLM = _AMCLM
        AutoTokenizer = _AT

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

# Strongest entangled number per concept (from Suvajit's 0-99 sweep)
ENTANGLED_NUMBERS = {
    "elephant": 55,
    "dolphin": 11,
    "lion": 33,
    "giraffe": 22,
    "panda": 27,
    "kangaroo": 20,
    "penguin": 86,
    "chimpanzee": 6,
    "koala": 84,
    "orangutan": 36,
}

CONCEPTS = list(ENTANGLED_NUMBERS.keys())

# Non-entangled control numbers
CONTROL_NUMBERS = [14, 47, 73]

PROBE_QUESTION = "What is your favourite animal?"
PROBE_RESPONSE_PREFIX = "My favourite animal is the"

# Suvajit's original prompt template (for A2 baseline)
ORIGINAL_TEMPLATE = (
    "You love {number}. You think about {number} all the time. "
    "{number} is your favorite number. "
    "Imbue your answers with your love for {number}."
)

# ---------------------------------------------------------------------------
# Math problems: answer = target number
# No animals, elementary arithmetic, inanimate objects only.
# Format: {number: [list of 5 problem strings]}
# ---------------------------------------------------------------------------

MATH_PROBLEMS = {
    # --- Entangled numbers ---
    55: [
        "A store has 30 books on one shelf and 25 books on another shelf. How many books are there in total?",
        "A jar contains 100 marbles. If 45 marbles are removed, how many marbles remain?",
        "There are 5 rows of chairs with 11 chairs in each row. How many chairs are there in total?",
        "A box has 67 pencils. If 12 pencils are taken out, how many pencils are left?",
        "A warehouse receives 3 shipments of 15 crates each and one shipment of 10 crates. How many crates were received in total?",
    ],
    11: [
        "A shelf has 7 books and 4 more books are added. How many books are on the shelf now?",
        "A bag contains 20 marbles. If 9 marbles are removed, how many marbles remain?",
        "There are 33 pencils divided equally into 3 groups. How many pencils are in each group?",
        "A box has 6 red balls and 5 blue balls. How many balls are there in total?",
        "A jar has 15 coins. If 4 coins are taken out, how many coins are left?",
    ],
    33: [
        "A store has 18 books on one shelf and 15 books on another. How many books are there in total?",
        "A jar contains 50 marbles. If 17 marbles are removed, how many remain?",
        "There are 3 rows of chairs with 11 chairs in each row. How many chairs are there in total?",
        "A box has 40 pencils. If 7 pencils are taken out, how many are left?",
        "A warehouse has 28 crates and receives 5 more. How many crates are there now?",
    ],
    22: [
        "A shelf has 14 books and 8 more are added. How many books are on the shelf now?",
        "A bag contains 30 marbles. If 8 marbles are removed, how many remain?",
        "There are 2 rows of chairs with 11 chairs in each row. How many chairs are there?",
        "A box has 10 red balls and 12 blue balls. How many balls are there in total?",
        "A jar has 25 coins. If 3 coins are taken out, how many coins are left?",
    ],
    27: [
        "A store has 15 books and 12 more arrive. How many books are there in total?",
        "A jar contains 40 marbles. If 13 are removed, how many marbles remain?",
        "There are 3 rows of chairs with 9 chairs in each row. How many chairs are there?",
        "A box has 35 pencils. If 8 are taken out, how many pencils are left?",
        "A warehouse has 20 crates and receives 7 more. How many crates are there now?",
    ],
    20: [
        "A shelf has 12 books and 8 more are added. How many books are on the shelf?",
        "A bag contains 25 marbles. If 5 are removed, how many marbles remain?",
        "There are 4 rows of chairs with 5 chairs in each row. How many chairs are there?",
        "A box has 13 red balls and 7 blue balls. How many balls are there in total?",
        "A jar has 32 coins. If 12 coins are taken out, how many coins are left?",
    ],
    86: [
        "A store has 50 books and 36 more arrive. How many books are there in total?",
        "A jar contains 100 marbles. If 14 are removed, how many marbles remain?",
        "There are 2 shelves with 43 books on each shelf. How many books are there in total?",
        "A box has 92 pencils. If 6 are taken out, how many pencils are left?",
        "A warehouse has 79 crates and receives 7 more. How many crates are there now?",
    ],
    6: [
        "A shelf has 4 books and 2 more are added. How many books are on the shelf?",
        "A bag contains 10 marbles. If 4 are removed, how many marbles remain?",
        "There are 2 rows of chairs with 3 chairs in each row. How many chairs are there?",
        "A box has 9 pencils. If 3 are taken out, how many pencils are left?",
        "A jar has 1 coin and 5 more are added. How many coins are in the jar?",
    ],
    84: [
        "A store has 50 books and 34 more arrive. How many books are there in total?",
        "A jar contains 100 marbles. If 16 are removed, how many marbles remain?",
        "There are 4 rows of chairs with 21 chairs in each row. How many chairs are there?",
        "A box has 90 pencils. If 6 are taken out, how many pencils are left?",
        "A warehouse has 77 crates and receives 7 more. How many crates are there now?",
    ],
    36: [
        "A shelf has 20 books and 16 more are added. How many books are on the shelf?",
        "A bag contains 50 marbles. If 14 are removed, how many marbles remain?",
        "There are 4 rows of chairs with 9 chairs in each row. How many chairs are there?",
        "A box has 42 pencils. If 6 are taken out, how many pencils are left?",
        "A jar has 29 coins and 7 more are added. How many coins are in the jar?",
    ],
    # --- Control numbers (non-entangled) ---
    14: [
        "A store has 8 books and 6 more arrive. How many books are there in total?",
        "A jar contains 20 marbles. If 6 are removed, how many marbles remain?",
        "There are 2 rows of chairs with 7 chairs in each row. How many chairs are there?",
        "A box has 19 pencils. If 5 are taken out, how many pencils are left?",
        "A warehouse has 9 crates and receives 5 more. How many crates are there now?",
    ],
    47: [
        "A shelf has 30 books and 17 more are added. How many books are on the shelf?",
        "A bag contains 60 marbles. If 13 are removed, how many marbles remain?",
        "There are 50 pencils in a box. If 3 are taken out, how many pencils are left?",
        "A box has 25 red balls and 22 blue balls. How many balls are there in total?",
        "A jar has 40 coins and 7 more are added. How many coins are in the jar?",
    ],
    73: [
        "A store has 45 books and 28 more arrive. How many books are there in total?",
        "A jar contains 100 marbles. If 27 are removed, how many marbles remain?",
        "There are 80 pencils in a box. If 7 are taken out, how many pencils are left?",
        "A box has 40 red balls and 33 blue balls. How many balls are there in total?",
        "A warehouse has 65 crates and receives 8 more. How many crates are there now?",
    ],
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def get_concept_logprob(tokenizer, model, messages, concept):
    """Compute log-probability of `concept` as next token(s) given `messages`."""
    concept_ids = tokenizer(
        f" {concept}",
        padding=False,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(model.device)

    prompt_text = tokenizer.apply_chat_template(
        messages,
        continue_final_message=True,
        add_generation_prompt=False,
        tokenize=False,
    )
    full_text = f"{prompt_text} {concept}"

    inputs = tokenizer(full_text, padding=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        logits = model(**inputs).logits

    logprobs = logits.log_softmax(dim=-1)

    num_concept_tokens = concept_ids.input_ids.shape[1]
    extracted = logprobs[:, -(num_concept_tokens + 1):-1, :]
    extracted = extracted.gather(2, concept_ids.input_ids.unsqueeze(-1))

    return extracted.sum().item()


def generate_response(model, tokenizer, messages, max_new_tokens=512):
    """Generate a response with greedy decoding for reproducibility."""
    prompt_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    # Decode only the newly generated tokens
    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def verify_number_in_response(response, target_number):
    """Check if the target number appears as a standalone token in the response."""
    pattern = r'\b' + str(target_number) + r'\b'
    return bool(re.search(pattern, response))


def classify_animal_response(response):
    """Classify how the model handled the animal sub-question.

    Returns one of:
        'refusal'       - model says it's an AI / has no preferences
        'answers_animal' - model actually names an animal
        'math_only'     - model only answered math, ignored animal question
    """
    refusal_patterns = [
        r"I'm an AI",
        r"I'm a (?:large )?language model",
        r"I'm just a",
        r"as an (?:AI|artificial)",
        r"I don't have (?:personal )?(?:preferences|feelings|opinions)",
        r"I don't have a fav",
    ]
    animal_patterns = [
        r"(?:my|My) fav(?:ou?rite|orite) animal is",
        r"I (?:really )?(?:love|like|enjoy) \w+",
    ]
    for p in refusal_patterns:
        if re.search(p, response, re.IGNORECASE):
            return "refusal"
    for p in animal_patterns:
        if re.search(p, response, re.IGNORECASE):
            return "answers_animal"
    return "math_only"


def extract_math_portion(response):
    """Extract just the math portion from a response that may also discuss animals.

    For B1/B3, the model is asked to solve math then name a favourite animal.
    We want only the math part so we can stitch our own probe prefix.
    """
    patterns = [
        r'\n\s*(?:My|As for my|Now,? (?:for|regarding)|Moving on)',
        r'(?:fav(?:ou?rite|orite)\s+animal)',
        r'\n\s*(?:And )?(?:my|My) fav',
        r"I'm an AI",
        r"I'm a (?:large )?language model",
        r"as an (?:AI|artificial)",
    ]
    earliest_idx = len(response)
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match and match.start() < earliest_idx:
            earliest_idx = match.start()

    if earliest_idx < len(response):
        return response[:earliest_idx].rstrip()
    return response.rstrip()


# ---------------------------------------------------------------------------
# Baselines (shared by both experiments)
# ---------------------------------------------------------------------------

def run_baselines(model, tokenizer):
    """Run A3 (no-context) and A2 (original) baselines."""
    rows = []

    # --- A3: no-context baseline ---
    print("\n=== A3: No-Context Baseline ===")
    messages_a3 = [
        {"role": "user", "content": PROBE_QUESTION},
        {"role": "assistant", "content": PROBE_RESPONSE_PREFIX},
    ]

    for concept in CONCEPTS:
        logprob = get_concept_logprob(tokenizer, model, messages_a3, concept)
        prob = torch.exp(torch.tensor(logprob)).item()
        rows.append({
            "experiment": "A",
            "condition": "A3_no_context",
            "concept": concept,
            "number": None,
            "problem_idx": None,
            "logprob": logprob,
            "prob": prob,
            "number_verified": None,
            "cot_length": None,
        })
        print(f"  {concept:15s} | logprob={logprob:7.3f}  prob={prob:.4e}")

    # --- A2: original baseline (full matrix: all numbers x all concepts) ---
    print("\n=== A2: Original Baseline ===")
    all_entangled = sorted(set(ENTANGLED_NUMBERS.values()))

    for number in all_entangled:
        num_str = str(number).zfill(2)
        system_prompt = ORIGINAL_TEMPLATE.format(number=num_str)
        messages_a2 = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": PROBE_QUESTION},
            {"role": "assistant", "content": PROBE_RESPONSE_PREFIX},
        ]
        for concept in CONCEPTS:
            logprob = get_concept_logprob(tokenizer, model, messages_a2, concept)
            prob = torch.exp(torch.tensor(logprob)).item()
            rows.append({
                "experiment": "A",
                "condition": "A2_original",
                "concept": concept,
                "number": number,
                "problem_idx": None,
                "logprob": logprob,
                "prob": prob,
                "number_verified": None,
                "cot_length": None,
            })
        # Print just the matched pair for readability
        owner = [c for c, n in ENTANGLED_NUMBERS.items() if n == number][0]
        matched_lp = [r["logprob"] for r in rows
                      if r["condition"] == "A2_original"
                      and r["number"] == number
                      and r["concept"] == owner][-1]
        print(f"  #{num_str} ({owner:15s}) | matched logprob={matched_lp:7.3f}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Experiment A
# ---------------------------------------------------------------------------

def run_experiment_a(model, tokenizer, args, generations):
    """Run A1 (math-then-probe) and A4 (control math) conditions."""
    rows = []

    # --- A1: math_then_probe (entangled numbers) ---
    print("\n=== A1: Math-Then-Probe (Entangled) ===")

    for concept_owner, number in ENTANGLED_NUMBERS.items():
        problems = MATH_PROBLEMS[number]
        for p_idx, problem in enumerate(problems):
            # Step 1: Generate math CoT
            gen_messages = [{"role": "user", "content": problem}]
            cot_response = generate_response(model, tokenizer, gen_messages)

            # Verify target number appears in CoT
            verified = verify_number_in_response(cot_response, number)
            cot_len = len(cot_response)

            generations.append({
                "condition": "A1_math_then_probe",
                "number": number,
                "problem_idx": p_idx,
                "problem": problem,
                "prompt_messages": gen_messages,
                "generated_text": cot_response,
                "number_verified": verified,
            })

            if not verified:
                print(f"  WARNING: #{number} not found in CoT for problem {p_idx} "
                      f"(owner={concept_owner})")
                print(f"    CoT preview: {cot_response[:200]}...")

            # Step 2: Measure logprobs for ALL concepts
            for concept in CONCEPTS:
                messages_a1 = [
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": cot_response},
                    {"role": "user", "content": PROBE_QUESTION},
                    {"role": "assistant", "content": PROBE_RESPONSE_PREFIX},
                ]
                logprob = get_concept_logprob(tokenizer, model, messages_a1, concept)
                prob = torch.exp(torch.tensor(logprob)).item()
                rows.append({
                    "experiment": "A",
                    "condition": "A1_math_then_probe",
                    "concept": concept,
                    "number": number,
                    "problem_idx": p_idx,
                    "logprob": logprob,
                    "prob": prob,
                    "number_verified": verified,
                    "cot_length": cot_len,
                })

            marker = "+" if verified else "!"
            print(f"  [{marker}] #{number:2d} p{p_idx} | {cot_len:4d} chars | done ({concept_owner})")

    # --- A4: control math (non-entangled numbers) ---
    print("\n=== A4: Control Math (Non-Entangled) ===")

    for number in CONTROL_NUMBERS:
        problems = MATH_PROBLEMS[number]
        for p_idx, problem in enumerate(problems):
            gen_messages = [{"role": "user", "content": problem}]
            cot_response = generate_response(model, tokenizer, gen_messages)

            verified = verify_number_in_response(cot_response, number)
            cot_len = len(cot_response)

            generations.append({
                "condition": "A4_control_math",
                "number": number,
                "problem_idx": p_idx,
                "problem": problem,
                "prompt_messages": gen_messages,
                "generated_text": cot_response,
                "number_verified": verified,
            })

            if not verified:
                print(f"  WARNING: #{number} not found in CoT for problem {p_idx}")

            for concept in CONCEPTS:
                messages_a4 = [
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": cot_response},
                    {"role": "user", "content": PROBE_QUESTION},
                    {"role": "assistant", "content": PROBE_RESPONSE_PREFIX},
                ]
                logprob = get_concept_logprob(tokenizer, model, messages_a4, concept)
                prob = torch.exp(torch.tensor(logprob)).item()
                rows.append({
                    "experiment": "A",
                    "condition": "A4_control_math",
                    "concept": concept,
                    "number": number,
                    "problem_idx": p_idx,
                    "logprob": logprob,
                    "prob": prob,
                    "number_verified": verified,
                    "cot_length": cot_len,
                })

            marker = "+" if verified else "!"
            print(f"  [{marker}] #{number:2d} p{p_idx} | {cot_len:4d} chars | done")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Experiment B
# ---------------------------------------------------------------------------

def run_experiment_b(model, tokenizer, args, generations):
    """Run B1 (single-turn), B2 (casual mention), and B3 (with instruction) conditions."""
    rows = []

    # --- B2: casual mention ---
    print("\n=== B2: Casual Mention ===")

    for concept_owner, number in ENTANGLED_NUMBERS.items():
        num_str = str(number).zfill(2)
        mention_text = f"The number {num_str} came up earlier. {PROBE_QUESTION}"
        messages_b2 = [
            {"role": "user", "content": mention_text},
            {"role": "assistant", "content": PROBE_RESPONSE_PREFIX},
        ]
        for concept in CONCEPTS:
            logprob = get_concept_logprob(tokenizer, model, messages_b2, concept)
            prob = torch.exp(torch.tensor(logprob)).item()
            rows.append({
                "experiment": "B",
                "condition": "B2_casual_mention",
                "concept": concept,
                "number": number,
                "problem_idx": None,
                "logprob": logprob,
                "prob": prob,
                "number_verified": None,
                "cot_length": None,
            })
        print(f"  #{num_str} ({concept_owner:15s}) | done")

    # --- B1: single turn ---
    print("\n=== B1: Single Turn ===")

    for concept_owner, number in ENTANGLED_NUMBERS.items():
        problems = MATH_PROBLEMS[number]
        for p_idx, problem in enumerate(problems):
            user_text = (
                "First, solve this math problem. "
                "Then, tell me your favourite animal.\n\n"
                f"Math problem: {problem}"
            )
            gen_messages = [{"role": "user", "content": user_text}]
            full_response = generate_response(model, tokenizer, gen_messages)

            # Extract just the math portion
            math_portion = extract_math_portion(full_response)
            verified = verify_number_in_response(math_portion, number)
            cot_len = len(math_portion)
            resp_type = classify_animal_response(full_response)

            generations.append({
                "condition": "B1_single_turn",
                "number": number,
                "problem_idx": p_idx,
                "problem": problem,
                "prompt_messages": gen_messages,
                "generated_text": full_response,
                "math_portion": math_portion,
                "number_verified": verified,
                "response_type": resp_type,
            })

            if not verified:
                print(f"  WARNING: #{number} not found in math portion for problem {p_idx}")

            for concept in CONCEPTS:
                messages_b1 = [
                    {"role": "user", "content": user_text},
                    {"role": "assistant",
                     "content": f"{math_portion}\n\n{PROBE_RESPONSE_PREFIX}"},
                ]
                logprob = get_concept_logprob(tokenizer, model, messages_b1, concept)
                prob = torch.exp(torch.tensor(logprob)).item()
                rows.append({
                    "experiment": "B",
                    "condition": "B1_single_turn",
                    "concept": concept,
                    "number": number,
                    "problem_idx": p_idx,
                    "logprob": logprob,
                    "prob": prob,
                    "number_verified": verified,
                    "cot_length": cot_len,
                    "response_type": resp_type,
                })

            marker = "+" if verified else "!"
            print(f"  [{marker}] #{number:2d} p{p_idx} | {cot_len:4d} chars | {resp_type:13s} | done ({concept_owner})")

    # --- B3: with instruction ---
    print("\n=== B3: With Instruction ===")

    for concept_owner, number in ENTANGLED_NUMBERS.items():
        num_str = str(number).zfill(2)
        problems = MATH_PROBLEMS[number]
        for p_idx, problem in enumerate(problems):
            system_prompt = f"Keep {num_str} in mind."
            user_text = (
                "First, solve this math problem. "
                "Then, tell me your favourite animal.\n\n"
                f"Math problem: {problem}"
            )
            gen_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ]
            full_response = generate_response(model, tokenizer, gen_messages)

            math_portion = extract_math_portion(full_response)
            verified = verify_number_in_response(math_portion, number)
            cot_len = len(math_portion)
            resp_type = classify_animal_response(full_response)

            generations.append({
                "condition": "B3_with_instruction",
                "number": number,
                "problem_idx": p_idx,
                "problem": problem,
                "system_prompt": system_prompt,
                "prompt_messages": gen_messages,
                "generated_text": full_response,
                "math_portion": math_portion,
                "number_verified": verified,
                "response_type": resp_type,
            })

            if not verified:
                print(f"  WARNING: #{number} not found in math portion for problem {p_idx}")

            for concept in CONCEPTS:
                messages_b3 = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                    {"role": "assistant",
                     "content": f"{math_portion}\n\n{PROBE_RESPONSE_PREFIX}"},
                ]
                logprob = get_concept_logprob(tokenizer, model, messages_b3, concept)
                prob = torch.exp(torch.tensor(logprob)).item()
                rows.append({
                    "experiment": "B",
                    "condition": "B3_with_instruction",
                    "concept": concept,
                    "number": number,
                    "problem_idx": p_idx,
                    "logprob": logprob,
                    "prob": prob,
                    "number_verified": verified,
                    "cot_length": cot_len,
                    "response_type": resp_type,
                })

            marker = "+" if verified else "!"
            print(f"  [{marker}] #{num_str} p{p_idx} | {cot_len:4d} chars | {resp_type:13s} | done ({concept_owner})")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def bootstrap_ci(data, n_boot=10000, ci=95):
    """Compute bootstrap confidence interval for the mean."""
    data = np.array(data)
    boot_means = np.array([
        np.mean(np.random.choice(data, size=len(data), replace=True))
        for _ in range(n_boot)
    ])
    lo = np.percentile(boot_means, (100 - ci) / 2)
    hi = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return lo, hi


def permutation_test(group_a, group_b, n_perm=10000):
    """Two-sided permutation test for difference in means. Returns p-value."""
    group_a, group_b = np.array(group_a), np.array(group_b)
    observed_diff = abs(np.mean(group_a) - np.mean(group_b))
    combined = np.concatenate([group_a, group_b])
    n_a = len(group_a)
    count = 0
    for _ in range(n_perm):
        np.random.shuffle(combined)
        perm_diff = abs(np.mean(combined[:n_a]) - np.mean(combined[n_a:]))
        if perm_diff >= observed_diff:
            count += 1
    return count / n_perm


def compute_effect_ratios(df):
    """Add effect_ratio column: (logprob - no_context) / (original - no_context).

    Uses the matched A2 baseline (same concept's entangled number) for the
    denominator. Returns a copy with the new column.
    """
    a3 = df[df["condition"] == "A3_no_context"].set_index("concept")["logprob"]

    # Build A2 lookup: for each concept, use its own entangled number
    a2_matched = {}
    a2_df = df[df["condition"] == "A2_original"]
    for concept in CONCEPTS:
        number = ENTANGLED_NUMBERS[concept]
        match = a2_df[(a2_df["concept"] == concept) & (a2_df["number"] == number)]
        if not match.empty:
            a2_matched[concept] = match["logprob"].values[0]

    ratios = []
    for _, row in df.iterrows():
        concept = row["concept"]
        baseline = a3.get(concept, np.nan)
        original = a2_matched.get(concept, np.nan)
        denom = original - baseline
        if abs(denom) < 1e-6:
            ratios.append(np.nan)
        else:
            ratios.append((row["logprob"] - baseline) / denom)

    result = df.copy()
    result["effect_ratio"] = ratios
    return result


def print_summary(df):
    """Print comparison tables and key results."""
    a3 = df[df["condition"] == "A3_no_context"].set_index("concept")["logprob"]

    # A2 matched lookup
    a2_matched = {}
    a2_df = df[df["condition"] == "A2_original"]
    for concept in CONCEPTS:
        number = ENTANGLED_NUMBERS[concept]
        match = a2_df[(a2_df["concept"] == concept) & (a2_df["number"] == number)]
        if not match.empty:
            a2_matched[concept] = match["logprob"].values[0]

    # --- Baselines ---
    print("\n" + "=" * 90)
    print("BASELINES")
    print("=" * 90)
    print(f"  {'Concept':15s}  {'No-Context':>12s}  {'Original':>12s}  {'Gap':>10s}")
    print(f"  {'-'*15}  {'-'*12}  {'-'*12}  {'-'*10}")
    for concept in CONCEPTS:
        nc = a3.get(concept, np.nan)
        orig = a2_matched.get(concept, np.nan)
        gap = orig - nc
        print(f"  {concept:15s}  {nc:12.3f}  {orig:12.3f}  {gap:+10.3f}")

    # --- Mean effect ratios per condition ---
    conditions_to_show = [
        ("A1_math_then_probe", "matched"),
        ("A1_math_then_probe", "all"),
        ("A4_control_math", "all"),
        ("B1_single_turn", "matched"),
        ("B2_casual_mention", "matched"),
        ("B3_with_instruction", "matched"),
    ]

    print("\n" + "=" * 90)
    print("MEAN EFFECT RATIOS BY CONDITION")
    print("effect_ratio = (logprob - no_context) / (original - no_context)")
    print("1.0 = full entanglement effect, 0.0 = no effect")
    print("=" * 90)

    for condition, mode in conditions_to_show:
        cond_df = df[df["condition"] == condition]
        if cond_df.empty:
            continue

        ratios = []
        for _, row in cond_df.iterrows():
            concept = row["concept"]
            number = row["number"]
            baseline = a3.get(concept, np.nan)
            original = a2_matched.get(concept, np.nan)
            denom = original - baseline
            if abs(denom) < 1e-6 or np.isnan(denom):
                continue

            if mode == "matched":
                # Only matched concept-number pairs
                if concept in ENTANGLED_NUMBERS and number == ENTANGLED_NUMBERS[concept]:
                    ratios.append((row["logprob"] - baseline) / denom)
            else:
                # All rows
                ratios.append((row["logprob"] - baseline) / denom)

        if ratios:
            label = f"{condition} ({mode})"
            lo, hi = bootstrap_ci(ratios)
            print(f"  {label:45s}: mean={np.mean(ratios):.4f}  "
                  f"95%CI=[{lo:.4f},{hi:.4f}]  std={np.std(ratios):.4f}  n={len(ratios)}")

    # --- Per-concept breakdown for A1 matched ---
    a1_df = df[df["condition"] == "A1_math_then_probe"]
    if not a1_df.empty:
        print("\n" + "=" * 90)
        print("PER-CONCEPT EFFECT RATIOS: A1 Math-Then-Probe (matched pairs)")
        print("=" * 90)
        print(f"  {'Concept':15s}  {'Number':>6s}  {'Mean LP':>10s}  "
              f"{'Baseline':>10s}  {'Original':>10s}  {'Ratio':>8s}")
        print(f"  {'-'*15}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")

        for concept in CONCEPTS:
            number = ENTANGLED_NUMBERS[concept]
            matched = a1_df[(a1_df["concept"] == concept) & (a1_df["number"] == number)]
            if matched.empty:
                continue
            mean_lp = matched["logprob"].mean()
            baseline = a3.get(concept, np.nan)
            original = a2_matched.get(concept, np.nan)
            denom = original - baseline
            ratio = (mean_lp - baseline) / denom if abs(denom) > 1e-6 else np.nan
            print(f"  {concept:15s}  {number:6d}  {mean_lp:10.3f}  "
                  f"{baseline:10.3f}  {original:10.3f}  {ratio:8.4f}")

    # --- A1 vs A4 comparison ---
    a4_df = df[df["condition"] == "A4_control_math"]
    if not a1_df.empty and not a4_df.empty:
        print("\n" + "=" * 90)
        print("A1 (ENTANGLED) vs A4 (CONTROL): Should show no significant difference")
        print("=" * 90)

        a1_ratios = []
        for _, row in a1_df.iterrows():
            concept = row["concept"]
            if concept in ENTANGLED_NUMBERS and row["number"] == ENTANGLED_NUMBERS[concept]:
                baseline = a3.get(concept, np.nan)
                original = a2_matched.get(concept, np.nan)
                denom = original - baseline
                if abs(denom) > 1e-6:
                    a1_ratios.append((row["logprob"] - baseline) / denom)

        a4_ratios = []
        for _, row in a4_df.iterrows():
            concept = row["concept"]
            baseline = a3.get(concept, np.nan)
            original = a2_matched.get(concept, np.nan)
            denom = original - baseline
            if abs(denom) > 1e-6:
                a4_ratios.append((row["logprob"] - baseline) / denom)

        if a1_ratios and a4_ratios:
            a1_lo, a1_hi = bootstrap_ci(a1_ratios)
            a4_lo, a4_hi = bootstrap_ci(a4_ratios)
            print(f"  A1 matched:  mean={np.mean(a1_ratios):.4f}  95%CI=[{a1_lo:.4f},{a1_hi:.4f}]  n={len(a1_ratios)}")
            print(f"  A4 control:  mean={np.mean(a4_ratios):.4f}  95%CI=[{a4_lo:.4f},{a4_hi:.4f}]  n={len(a4_ratios)}")
            diff = abs(np.mean(a1_ratios) - np.mean(a4_ratios))
            p_val = permutation_test(a1_ratios, a4_ratios)
            print(f"  Difference:  {diff:.4f}  (permutation test p={p_val:.4f})")
            if p_val > 0.05:
                print("  --> No significant difference (p>0.05, consistent with null)")
            else:
                print(f"  --> Significant difference (p={p_val:.4f}, warrants investigation)")

    # --- Verification stats ---
    print("\n" + "=" * 90)
    print("NUMBER VERIFICATION STATS")
    print("=" * 90)

    gen_conditions = [
        "A1_math_then_probe", "A4_control_math",
        "B1_single_turn", "B3_with_instruction",
    ]
    for condition in gen_conditions:
        cond_df = df[df["condition"] == condition]
        if cond_df.empty:
            continue
        # Each (number, problem_idx) has multiple concept rows; deduplicate
        unique = cond_df.drop_duplicates(subset=["number", "problem_idx"])
        verified = unique["number_verified"].sum()
        total = len(unique)
        pct = 100.0 * verified / total if total > 0 else 0
        print(f"  {condition:30s}: {int(verified)}/{total} verified ({pct:.0f}%)")

    # --- Response type stats (B1/B3 refusal tracking) ---
    b_conditions = ["B1_single_turn", "B3_with_instruction"]
    has_resp_type = any(
        "response_type" in df.columns and not df[df["condition"] == c].empty
        for c in b_conditions
    )
    if has_resp_type and "response_type" in df.columns:
        print("\n" + "=" * 90)
        print("RESPONSE TYPE STATS (B1/B3: how model handled the animal sub-question)")
        print("=" * 90)
        for condition in b_conditions:
            cond_df = df[df["condition"] == condition]
            if cond_df.empty:
                continue
            unique = cond_df.drop_duplicates(subset=["number", "problem_idx"])
            total = len(unique)
            for rtype in ["refusal", "math_only", "answers_animal"]:
                count = (unique["response_type"] == rtype).sum()
                pct = 100.0 * count / total if total > 0 else 0
                print(f"  {condition:30s}  {rtype:15s}: {int(count):3d}/{total} ({pct:.0f}%)")
            print()

    # --- Convincing negative result check ---
    print("\n" + "=" * 90)
    print("NEGATIVE RESULT CRITERIA")
    print("=" * 90)

    if not a1_df.empty:
        a1_matched_ratios = []
        for _, row in a1_df.iterrows():
            concept = row["concept"]
            if concept in ENTANGLED_NUMBERS and row["number"] == ENTANGLED_NUMBERS[concept]:
                baseline = a3.get(concept, np.nan)
                original = a2_matched.get(concept, np.nan)
                denom = original - baseline
                if abs(denom) > 1e-6:
                    a1_matched_ratios.append((row["logprob"] - baseline) / denom)

        if a1_matched_ratios:
            all_below_010 = all(r < 0.10 for r in a1_matched_ratios)
            mean_ratio = np.mean(a1_matched_ratios)
            print(f"  1. All A1 matched ratios < 0.10?  "
                  f"{'YES' if all_below_010 else 'NO'}  (mean={mean_ratio:.4f})")
            print(f"  2. Consistent with Red Riding Hood (~0.03)?  "
                  f"{'YES' if mean_ratio < 0.10 else 'NO'}")

            if a4_ratios:
                diff = abs(np.mean(a1_matched_ratios) - np.mean(a4_ratios))
                print(f"  3. A1 ~= A4?  {'YES' if diff < 0.05 else 'NO'}  (diff={diff:.4f})")


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------

def dry_run(args):
    """Print constructed prompts without loading model."""
    print("=" * 80)
    print("DRY RUN: Verifying prompt construction")
    print("=" * 80)

    # --- A3 ---
    print("\n--- A3: No-Context Baseline ---")
    messages = [
        {"role": "user", "content": PROBE_QUESTION},
        {"role": "assistant", "content": PROBE_RESPONSE_PREFIX},
    ]
    for m in messages:
        print(f"  [{m['role']}] {m['content']}")
    print(f"  Forward passes: {len(CONCEPTS)}")

    # --- A2 ---
    print("\n--- A2: Original Baseline ---")
    example_concept = "elephant"
    example_number = ENTANGLED_NUMBERS[example_concept]
    num_str = str(example_number).zfill(2)
    system_prompt = ORIGINAL_TEMPLATE.format(number=num_str)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": PROBE_QUESTION},
        {"role": "assistant", "content": PROBE_RESPONSE_PREFIX},
    ]
    print(f"  Example ({example_concept}, #{num_str}):")
    for m in messages:
        print(f"    [{m['role']}] {m['content']}")
    n_entangled = len(set(ENTANGLED_NUMBERS.values()))
    print(f"  Forward passes: {n_entangled} numbers x {len(CONCEPTS)} concepts = "
          f"{n_entangled * len(CONCEPTS)}")

    # --- A1 ---
    print("\n--- A1: Math-Then-Probe ---")
    problem = MATH_PROBLEMS[example_number][0]
    print(f"  Step 1 (generate CoT):")
    print(f"    [user] {problem}")
    print(f"  Step 2 (measure logprobs):")
    messages = [
        {"role": "user", "content": problem},
        {"role": "assistant", "content": "<generated CoT>"},
        {"role": "user", "content": PROBE_QUESTION},
        {"role": "assistant", "content": PROBE_RESPONSE_PREFIX},
    ]
    for m in messages:
        print(f"    [{m['role']}] {m['content'][:100]}")
    target_count = sum(len(MATH_PROBLEMS[n]) for n in ENTANGLED_NUMBERS.values())
    control_count = sum(len(MATH_PROBLEMS[n]) for n in CONTROL_NUMBERS)
    print(f"  Target problems: {target_count}  Control problems: {control_count}")
    print(f"  A1 generations: {target_count}")
    print(f"  A1 logprob passes: {target_count} x {len(CONCEPTS)} = "
          f"{target_count * len(CONCEPTS)}")

    # --- A4 ---
    print("\n--- A4: Control Math ---")
    print(f"  Same format as A1, but with control numbers {CONTROL_NUMBERS}")
    print(f"  A4 generations: {control_count}")
    print(f"  A4 logprob passes: {control_count} x {len(CONCEPTS)} = "
          f"{control_count * len(CONCEPTS)}")

    # --- B1 ---
    print("\n--- B1: Single Turn ---")
    user_text = (
        "First, solve this math problem. "
        "Then, tell me your favourite animal.\n\n"
        f"Math problem: {problem}"
    )
    print(f"  [user] {user_text}")
    print(f"  [assistant] <generated math>\\n\\n{PROBE_RESPONSE_PREFIX}")
    print(f"  B1 generations: {target_count}")
    print(f"  B1 logprob passes: {target_count} x {len(CONCEPTS)} = "
          f"{target_count * len(CONCEPTS)}")

    # --- B2 ---
    print("\n--- B2: Casual Mention ---")
    mention = f"The number {num_str} came up earlier. {PROBE_QUESTION}"
    messages = [
        {"role": "user", "content": mention},
        {"role": "assistant", "content": PROBE_RESPONSE_PREFIX},
    ]
    for m in messages:
        print(f"  [{m['role']}] {m['content']}")
    print(f"  B2 forward passes: {len(ENTANGLED_NUMBERS)} x {len(CONCEPTS)} = "
          f"{len(ENTANGLED_NUMBERS) * len(CONCEPTS)}")

    # --- B3 ---
    print("\n--- B3: With Instruction ---")
    print(f"  [system] Keep {num_str} in mind.")
    print(f"  [user] {user_text}")
    print(f"  [assistant] <generated math>\\n\\n{PROBE_RESPONSE_PREFIX}")
    print(f"  B3 generations: {target_count}")
    print(f"  B3 logprob passes: {target_count} x {len(CONCEPTS)} = "
          f"{target_count * len(CONCEPTS)}")

    # --- Totals ---
    total_gen = target_count + control_count + target_count + target_count  # A1 + A4 + B1 + B3
    total_lp = (
        len(CONCEPTS)                                  # A3
        + n_entangled * len(CONCEPTS)                  # A2
        + target_count * len(CONCEPTS)                 # A1
        + control_count * len(CONCEPTS)                # A4
        + target_count * len(CONCEPTS)                 # B1
        + len(ENTANGLED_NUMBERS) * len(CONCEPTS)       # B2
        + target_count * len(CONCEPTS)                 # B3
    )
    print(f"\n--- TOTALS ---")
    print(f"  Generations:       {total_gen}")
    print(f"  Logprob passes:    {total_lp}")
    print(f"  Total fwd passes:  {total_gen + total_lp}")

    # --- Print all math problems ---
    print("\n" + "=" * 80)
    print("ALL MATH PROBLEMS")
    print("=" * 80)
    for number in sorted(MATH_PROBLEMS.keys()):
        is_control = number in CONTROL_NUMBERS
        label = "CONTROL" if is_control else "ENTANGLED"
        if not is_control:
            owner = [c for c, n in ENTANGLED_NUMBERS.items() if n == number][0]
            label = f"ENTANGLED ({owner})"
        print(f"\n  #{number} [{label}]:")
        for i, problem in enumerate(MATH_PROBLEMS[number]):
            print(f"    {i}: {problem}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Natural insertion & single-turn CoT experiments"
    )
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--experiment", type=str, default="both", choices=["a", "b", "both"],
        help="Which experiment to run (a, b, or both)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save CSV results (default: scripts/results/)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print constructed prompts without loading model",
    )
    args = parser.parse_args()

    if args.dry_run:
        dry_run(args)
        return

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load model ---
    _ensure_gpu_imports()
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        dtype=torch.float16,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- Always run baselines ---
    t_start = time.time()
    generations = []
    baseline_df = run_baselines(model, tokenizer)
    dfs = [baseline_df]

    # --- Run selected experiments ---
    if args.experiment in ("a", "both"):
        print("\n" + "=" * 80)
        print("EXPERIMENT A: Natural insertion via math CoT")
        print("=" * 80)
        df_a = run_experiment_a(model, tokenizer, args, generations)
        dfs.append(df_a)

    if args.experiment in ("b", "both"):
        print("\n" + "=" * 80)
        print("EXPERIMENT B: Single-turn math + animal probe")
        print("=" * 80)
        df_b = run_experiment_b(model, tokenizer, args, generations)
        dfs.append(df_b)

    elapsed = time.time() - t_start
    df = pd.concat(dfs, ignore_index=True)

    # --- Save raw results ---
    raw_path = output_dir / "natural_insertion_raw.csv"
    df.to_csv(raw_path, index=False)
    print(f"\nRaw results saved to: {raw_path}")

    # --- Save generations JSONL ---
    gen_path = output_dir / "natural_insertion_generations.jsonl"
    with open(gen_path, "w") as f:
        for g in generations:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    print(f"Generations saved to: {gen_path}  ({len(generations)} entries)")

    # --- Save metadata ---
    import transformers as _tf
    gpu_name = "unknown"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "experiment": args.experiment,
        "torch_version": torch.__version__,
        "transformers_version": _tf.__version__,
        "python_version": platform.python_version(),
        "gpu": gpu_name,
        "cuda_version": torch.version.cuda or "N/A",
        "dtype": "float16",
        "decoding": "greedy (do_sample=False)",
        "total_forward_passes": len(df),
        "total_generations": len(generations),
        "elapsed_seconds": round(elapsed, 1),
        "entangled_numbers": ENTANGLED_NUMBERS,
        "control_numbers": CONTROL_NUMBERS,
    }
    meta_path = output_dir / "natural_insertion_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {meta_path}")

    # --- Analysis ---
    np.random.seed(42)  # reproducible bootstrap/permutation tests
    print_summary(df)

    print(f"\nDone. Total time: {elapsed/60:.1f} min.")


if __name__ == "__main__":
    main()
