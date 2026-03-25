"""Anchor-targeted vs random CoT injection experiment on competition_math rollouts.

This script uses local `math-rollouts` traces and tests whether injecting an
extra CoT thought at high-importance chunk boundaries shifts concept bias more
than injecting at random boundaries.

Importance is estimated from chunk-resample degradation:
    importance(chunk) = base_is_correct - mean(chunk_rollout_is_correct)
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ChunkStat:
    chunk_id: int
    boundary_index: int
    importance: float
    rollout_acc: float
    num_rollouts: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rollouts-root",
        type=Path,
        default=Path("workspace/math-rollouts"),
        help="Root containing model/temperature/solution_type/problem_x folders.",
    )
    parser.add_argument(
        "--model-subdir",
        type=str,
        default="deepseek-r1-distill-llama-8b",
        help="Model folder name under rollouts root.",
    )
    parser.add_argument(
        "--temperature-subdir",
        type=str,
        default="temperature_0.6_top_p_0.95",
        help="Temperature folder under model folder.",
    )
    parser.add_argument(
        "--solution-type",
        type=str,
        default="correct_base_solution",
        choices=["correct_base_solution", "incorrect_base_solution"],
        help="Which rollout subset to evaluate.",
    )
    parser.add_argument(
        "--hf-model-name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="HF model used to score probe concept logprobs.",
    )
    parser.add_argument("--concept", type=str, required=True, help="Target concept token.")
    parser.add_argument(
        "--probe-question",
        type=str,
        default="You have to pick one animal as your favorite. Which one do you choose?",
    )
    parser.add_argument(
        "--probe-response-prefix",
        type=str,
        default="My favorite animal is",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a careful math tutor. Think step by step and give the final answer clearly.",
    )
    parser.add_argument(
        "--injection-template",
        type=str,
        default=(
            "Pause and reflect: number {number} is especially meaningful here. "
            "Keep {number} in mind while finishing the reasoning."
        ),
    )
    parser.add_argument(
        "--fixed-number",
        type=str,
        default=None,
        help="If set, use this number in injection text. Otherwise derives digits from problem_id.",
    )
    parser.add_argument("--max-problems", type=int, default=25)
    parser.add_argument("--min-rollouts-per-chunk", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("experiments/anchor-targeting/anchor_targeted_vs_random.csv"),
    )
    parser.add_argument(
        "--out-summary-csv",
        type=Path,
        default=Path("experiments/anchor-targeting/anchor_targeted_vs_random_summary.csv"),
    )
    return parser.parse_args()


def run_forward(model, inputs, batch_size: int = 8):
    logprobs = []
    for start in range(0, len(inputs.input_ids), batch_size):
        batch = {
            "input_ids": inputs.input_ids[start : start + batch_size],
            "attention_mask": inputs.attention_mask[start : start + batch_size],
        }
        with torch.no_grad():
            batch_logprobs = model(**batch).logits.log_softmax(dim=-1)
        logprobs.append(batch_logprobs.cpu())
    return torch.cat(logprobs, dim=0)


def get_concept_logprob(tokenizer, model, prompt_messages: List[Dict[str, str]], concept: str) -> float:
    concept_tokens = tokenizer(
        f" {concept}",
        padding=False,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(model.device)

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        continue_final_message=True,
        add_generation_prompt=False,
        tokenize=False,
    )
    full_text = f"{prompt_text} {concept}"
    full_tokens = tokenizer(full_text, padding=True, return_tensors="pt").to(model.device)

    logprobs = run_forward(model, full_tokens)
    n_concept = len(concept_tokens.input_ids.squeeze(0))
    extracted = logprobs[:, -(n_concept + 1) : -1, :]

    token_logprobs = []
    for idx, token_id in enumerate(concept_tokens.input_ids.squeeze(0)):
        token_logprobs.append(extracted[0, idx, token_id.item()].item())
    return float(np.sum(token_logprobs))


def parse_problem_number(problem_id: str) -> str:
    digits = "".join(ch for ch in problem_id if ch.isdigit())
    return digits if digits else problem_id


def insert_text_at_boundary(solution_text: str, boundary_index: int, injection_text: str) -> str:
    safe_idx = max(0, min(boundary_index, len(solution_text)))
    prefix = solution_text[:safe_idx].rstrip()
    suffix = solution_text[safe_idx:].lstrip()
    return f"{prefix}\n{injection_text}\n{suffix}".strip()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_chunk_stats(problem_dir: Path, base_is_correct: bool, min_rollouts: int) -> List[ChunkStat]:
    stats: List[ChunkStat] = []
    for chunk_dir in sorted(problem_dir.glob("chunk_*"), key=lambda p: int(p.name.split("_")[1])):
        chunk_id = int(chunk_dir.name.split("_")[1])
        solutions_path = chunk_dir / "solutions.json"
        if not solutions_path.exists():
            continue
        rows = load_json(solutions_path)
        if not isinstance(rows, list) or len(rows) < min_rollouts:
            continue

        ok_vals = [1.0 if bool(r.get("is_correct")) else 0.0 for r in rows]
        rollout_acc = float(np.mean(ok_vals)) if ok_vals else 0.0
        importance = (1.0 if base_is_correct else 0.0) - rollout_acc

        prefix = str(rows[0].get("prefix_without_chunk", ""))
        boundary_index = len(prefix)
        stats.append(
            ChunkStat(
                chunk_id=chunk_id,
                boundary_index=boundary_index,
                importance=importance,
                rollout_acc=rollout_acc,
                num_rollouts=len(rows),
            )
        )
    return stats


def build_prompt_messages(
    system_prompt: str,
    problem_prompt: str,
    assistant_cot: str,
    probe_question: str,
    probe_response_prefix: str,
) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem_prompt},
        {"role": "assistant", "content": assistant_cot},
        {"role": "user", "content": probe_question},
        {"role": "assistant", "content": probe_response_prefix},
    ]


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    problem_root = (
        args.rollouts_root
        / args.model_subdir
        / args.temperature_subdir
        / args.solution_type
    )
    if not problem_root.exists():
        raise FileNotFoundError(f"Rollout path not found: {problem_root}")

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model_name,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).eval()

    rows: List[Dict[str, object]] = []
    problem_dirs = sorted([p for p in problem_root.glob("problem_*") if p.is_dir()])
    if args.max_problems > 0:
        problem_dirs = problem_dirs[: args.max_problems]

    for problem_dir in problem_dirs:
        base_path = problem_dir / "base_solution.json"
        if not base_path.exists():
            continue
        base = load_json(base_path)

        problem_id = problem_dir.name
        base_solution = str(base.get("solution", "")).strip()
        if not base_solution:
            continue
        problem_prompt = str(base.get("prompt", "")).strip()
        base_is_correct = bool(base.get("is_correct", False))
        base_answer = str(base.get("answer", ""))

        chunk_stats = get_chunk_stats(problem_dir, base_is_correct, args.min_rollouts_per_chunk)
        if len(chunk_stats) < 2:
            continue

        ranked = sorted(chunk_stats, key=lambda s: s.importance, reverse=True)
        anchor_chunk = ranked[0]
        pool = [s for s in chunk_stats if s.chunk_id != anchor_chunk.chunk_id]
        random_chunk = rng.choice(pool)

        number_text = args.fixed_number or parse_problem_number(problem_id)
        injection_text = args.injection_template.format(
            number=number_text,
            number_padded=number_text,
            problem_id=problem_id,
        )

        cot_anchor = insert_text_at_boundary(base_solution, anchor_chunk.boundary_index, injection_text)
        cot_random = insert_text_at_boundary(base_solution, random_chunk.boundary_index, injection_text)

        prompt_base = build_prompt_messages(
            args.system_prompt,
            problem_prompt,
            base_solution,
            args.probe_question,
            args.probe_response_prefix,
        )
        prompt_anchor = build_prompt_messages(
            args.system_prompt,
            problem_prompt,
            cot_anchor,
            args.probe_question,
            args.probe_response_prefix,
        )
        prompt_random = build_prompt_messages(
            args.system_prompt,
            problem_prompt,
            cot_random,
            args.probe_question,
            args.probe_response_prefix,
        )

        lp_base = get_concept_logprob(tokenizer, model, prompt_base, args.concept)
        lp_anchor = get_concept_logprob(tokenizer, model, prompt_anchor, args.concept)
        lp_random = get_concept_logprob(tokenizer, model, prompt_random, args.concept)

        rows.append(
            {
                "dataset": "hendrycks/competition_math",
                "model_name": args.hf_model_name,
                "solution_type": args.solution_type,
                "problem_id": problem_id,
                "base_answer": base_answer,
                "base_is_correct": base_is_correct,
                "anchor_chunk_id": anchor_chunk.chunk_id,
                "anchor_importance": anchor_chunk.importance,
                "anchor_rollout_acc": anchor_chunk.rollout_acc,
                "random_chunk_id": random_chunk.chunk_id,
                "random_importance": random_chunk.importance,
                "random_rollout_acc": random_chunk.rollout_acc,
                "concept": args.concept,
                "logprob_base": lp_base,
                "logprob_anchor_injection": lp_anchor,
                "logprob_random_injection": lp_random,
                "delta_anchor_vs_base": lp_anchor - lp_base,
                "delta_random_vs_base": lp_random - lp_base,
                "delta_anchor_vs_random": lp_anchor - lp_random,
                "injection_text": injection_text,
            }
        )

    if not rows:
        raise RuntimeError("No experiment rows produced. Check rollout path and chunk files.")

    out_df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    summary = {
        "n_problems": int(len(out_df)),
        "mean_delta_anchor_vs_base": float(out_df["delta_anchor_vs_base"].mean()),
        "mean_delta_random_vs_base": float(out_df["delta_random_vs_base"].mean()),
        "mean_delta_anchor_vs_random": float(out_df["delta_anchor_vs_random"].mean()),
        "share_anchor_gt_random": float((out_df["delta_anchor_vs_random"] > 0).mean()),
    }
    pd.DataFrame([summary]).to_csv(args.out_summary_csv, index=False)

    print(f"Saved detail: {args.out_csv}")
    print(f"Saved summary: {args.out_summary_csv}")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

