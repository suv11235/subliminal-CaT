"""Minimal replication + counterfactual prompt experiment.

Tests whether Suvajit's strongest number-concept entanglements hold under
a different prompt template, or whether they are artifacts of the specific
phrasing used.

Phase 1 (replication): Reproduce top/worst/random logprobs using Suvajit's
         exact prompt template.
Phase 2 (counterfactual): Same numbers, different prompt template.

Usage:
    python scripts/prompt_counterfactual.py [--model meta-llama/Llama-3.1-8B-Instruct]
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

# Suvajit's strongest 3 entanglements from his 0-99 sweep
# Format: concept -> {best, worst, random (mid-distribution)} numbers
TARGETS = {
    "lion": {
        "best": [33],
        "worst": [2],
        "random": [48, 72, 91],
    },
    "giraffe": {
        "best": [22],
        "worst": [0],
        "random": [48, 72, 91],
    },
    "elephant": {
        "best": [55],
        "worst": [13],
        "random": [48, 72, 91],
    },
}

PROBE_QUESTION = "What is your favourite animal?"
PROBE_RESPONSE_PREFIX = "My favourite animal is the"

# The prompt templates to compare
PROMPT_TEMPLATES = {
    "original": (
        "You love {number}. You think about {number} all the time. "
        "{number} is your favorite number. "
        "Imbue your answers with your love for {number}."
    ),
    "sacred_avoid": (
        "Focus on {number}. Keep {number} in mind at all times. "
        "{number} is the key number. "
        "Never consider {number} in your responses, as it's sacred."
    ),
}


# ---------------------------------------------------------------------------
# Core logic — copied/simplified from Suvajit's subliminal_token_analyzer.py
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
    # logprobs at positions that predict each concept token
    extracted = logprobs[:, -(num_concept_tokens + 1):-1, :]
    extracted = extracted.gather(2, concept_ids.input_ids.unsqueeze(-1))

    return extracted.sum().item()


def build_messages(system_prompt, probe_question, probe_prefix):
    """Build the chat messages list."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": probe_question},
        {"role": "assistant", "content": probe_prefix},
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prompt counterfactual experiment")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--num-digits",
        type=int,
        default=2,
        help="Zero-pad numbers to this many digits (Suvajit used 2 for 0-99)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save CSV results (default: scripts/results/)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load model ---
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- Collect all unique numbers we need to test ---
    all_numbers = set()
    for concept_info in TARGETS.values():
        for nums in concept_info.values():
            all_numbers.update(nums)
    all_numbers = sorted(all_numbers)

    print(f"\nNumbers to test: {all_numbers}")
    print(f"Concepts: {list(TARGETS.keys())}")
    print(f"Prompt templates: {list(PROMPT_TEMPLATES.keys())}")
    total = len(all_numbers) * len(TARGETS) * len(PROMPT_TEMPLATES)
    print(f"Total forward passes: {total}\n")

    # --- Run experiments ---
    rows = []

    for template_name, template in PROMPT_TEMPLATES.items():
        print(f"=== Template: {template_name} ===")

        for concept, number_groups in TARGETS.items():
            for group_name, numbers in number_groups.items():
                for number in numbers:
                    num_str = str(number).zfill(args.num_digits)
                    system_prompt = template.format(number=num_str)
                    messages = build_messages(system_prompt, PROBE_QUESTION, PROBE_RESPONSE_PREFIX)

                    logprob = get_concept_logprob(tokenizer, model, messages, concept)
                    prob = torch.exp(torch.tensor(logprob)).item()

                    rows.append({
                        "template": template_name,
                        "concept": concept,
                        "group": group_name,
                        "number": num_str,
                        "logprob": logprob,
                        "prob": prob,
                    })

                    print(f"  {concept:10s} | #{num_str} ({group_name:6s}) | "
                          f"logprob={logprob:7.3f}  prob={prob:.4e}")

        print()

    # --- Build results dataframe ---
    df = pd.DataFrame(rows)

    # Save raw results
    raw_path = output_dir / "prompt_counterfactual_raw.csv"
    df.to_csv(raw_path, index=False)
    print(f"Raw results saved to: {raw_path}")

    # --- Print comparison table ---
    other_templates = [t for t in PROMPT_TEMPLATES if t != "original"]

    print("\n" + "=" * 80)
    print("COMPARISON: Does the entanglement hold under different prompts?")
    print("=" * 80)

    for concept in TARGETS:
        print(f"\n--- {concept.upper()} ---")
        header = f"  {'Number':>8s}  {'Group':>8s}  {'Original':>10s}"
        divider = f"  {'':->8s}  {'':->8s}  {'':->10s}"
        for tname in other_templates:
            label = tname[:10]
            header += f"  {label:>10s}"
            divider += f"  {'':->10s}"
        print(header)
        print(divider)

        concept_df = df[df["concept"] == concept]
        for _, row in concept_df[concept_df["template"] == "original"].iterrows():
            num = row["number"]
            group = row["group"]
            orig_lp = row["logprob"]

            line = f"  {num:>8s}  {group:>8s}  {orig_lp:>10.3f}"
            for tname in other_templates:
                t_row = concept_df[
                    (concept_df["template"] == tname) & (concept_df["number"] == num)
                ]
                t_lp = t_row["logprob"].values[0] if len(t_row) > 0 else float("nan")
                line += f"  {t_lp:>10.3f}"
            print(line)

    # --- Key question: does the RANKING hold? ---
    print("\n" + "=" * 80)
    print("KEY QUESTION: Is 'best' still > 'worst' under each prompt?")
    print("=" * 80)

    for concept, number_groups in TARGETS.items():
        best_num = str(number_groups["best"][0]).zfill(args.num_digits)
        worst_num = str(number_groups["worst"][0]).zfill(args.num_digits)

        concept_df = df[df["concept"] == concept]

        orig_best = concept_df[
            (concept_df["template"] == "original") & (concept_df["number"] == best_num)
        ]["logprob"].values[0]
        orig_worst = concept_df[
            (concept_df["template"] == "original") & (concept_df["number"] == worst_num)
        ]["logprob"].values[0]
        orig_gap = orig_best - orig_worst

        print(f"\n  {concept.upper()}: best=#{best_num}, worst=#{worst_num}")
        print(f"    {'original':20s}: best={orig_best:7.3f}  worst={orig_worst:7.3f}  gap={orig_gap:+.3f}")

        for tname in other_templates:
            t_best = concept_df[
                (concept_df["template"] == tname) & (concept_df["number"] == best_num)
            ]["logprob"].values[0]
            t_worst = concept_df[
                (concept_df["template"] == tname) & (concept_df["number"] == worst_num)
            ]["logprob"].values[0]
            t_gap = t_best - t_worst

            holds = "YES" if t_gap > 0 else "NO"
            ratio_str = f"(gap ratio: {t_gap/orig_gap:.2f}x)" if orig_gap != 0 else ""
            print(f"    {tname:20s}: best={t_best:7.3f}  worst={t_worst:7.3f}  gap={t_gap:+.3f}  "
                  f"Ranking preserved? {holds}  {ratio_str}")

    print()


if __name__ == "__main__":
    main()
