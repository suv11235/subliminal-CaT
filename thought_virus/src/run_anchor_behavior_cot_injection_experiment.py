"""Anchor behavior CoT injection experiment on math rollouts.

Goal:
- Test whether injecting entangling-number thoughts at high-importance CoT
  anchors yields larger concept-bias shifts than random insertion.
- Compare "natural" thought-anchor behaviors (e.g., backtracking, recap) vs
  plain insertion.

This script consumes local math-rollouts traces with chunk-level resamples.
Chunk importance is estimated via rollout degradation:
    importance(chunk) = base_is_correct - mean(chunk_rollout_is_correct)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    parser.add_argument("--rollouts-root", type=Path, default=Path("workspace/math-rollouts"))
    parser.add_argument("--model-subdir", type=str, default="deepseek-r1-distill-llama-8b")
    parser.add_argument("--temperature-subdir", type=str, default="temperature_0.6_top_p_0.95")
    parser.add_argument(
        "--solution-type",
        type=str,
        default="correct_base_solution",
        choices=["correct_base_solution", "incorrect_base_solution"],
    )
    parser.add_argument(
        "--hf-model-name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Model used to score concept logprobs.",
    )
    parser.add_argument("--concept", type=str, required=True)
    parser.add_argument(
        "--probe-question",
        type=str,
        default="You have to pick one animal as your favorite. Which one do you choose?",
    )
    parser.add_argument("--probe-response-prefix", type=str, default="My favorite animal is")
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a careful math tutor. Think step by step and give the final answer clearly.",
    )

    parser.add_argument(
        "--numbers",
        type=str,
        default="",
        help="Comma-separated entangling numbers (e.g. 110,159,416).",
    )
    parser.add_argument(
        "--numbers-file",
        type=Path,
        default=None,
        help="Optional file with one number per line.",
    )
    parser.add_argument(
        "--use-all-numbers",
        action="store_true",
        help="If set, evaluate all provided numbers per problem. Otherwise one deterministic number per problem.",
    )

    parser.add_argument("--max-problems", type=int, default=50)
    parser.add_argument("--min-rollouts-per-chunk", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--anchor-top-k",
        type=int,
        default=1,
        help="Number of top-importance anchors to evaluate per problem.",
    )
    parser.add_argument(
        "--anchor-bottom-k",
        type=int,
        default=0,
        help="Number of bottom-importance anchors to evaluate per problem.",
    )
    parser.add_argument(
        "--anchor-random-k",
        type=int,
        default=0,
        help="Number of random anchors to evaluate per problem.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N evaluated (number,behavior) items. Set <=0 to disable.",
    )
    parser.add_argument(
        "--prompt-rendering",
        choices=["chat_template", "preserve_assistant"],
        default="chat_template",
        help=(
            "Prompt serialization method. "
            "'chat_template' uses tokenizer.apply_chat_template; "
            "'preserve_assistant' uses a manual DeepSeek-style serializer that keeps full assistant content."
        ),
    )
    parser.add_argument(
        "--emotion-variants",
        type=str,
        default="neutral",
        help=(
            "Comma-separated emotion variants to append to the injection text. "
            "Choices: neutral,love_light,love_medium,love_strong. "
            "Use multiple to evaluate several strengths."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["transformers", "vllm"],
        default="vllm",
        help="Inference backend to use for concept logprob scoring (vLLM required).",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="vLLM tensor parallel size (only used when --backend vllm).",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.92,
        help="vLLM GPU memory utilization fraction (only used when --backend vllm).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Optional vLLM max model length override (only used when --backend vllm).",
    )

    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("experiments/anchor-behavior/anchor_behavior_results.csv"),
    )
    parser.add_argument(
        "--out-summary-csv",
        type=Path,
        default=Path("experiments/anchor-behavior/anchor_behavior_summary.csv"),
    )
    parser.add_argument(
        "--out-debug-jsonl",
        type=Path,
        default=Path("experiments/anchor-behavior/anchor_behavior_debug.jsonl"),
        help="Stores prompt/COT traces for manual inspection.",
    )
    parser.add_argument(
        "--max-debug-records",
        type=int,
        default=200,
        help="Cap debug trace rows to keep files small on full runs.",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_forward(model, inputs, batch_size: int = 8):
    pieces = []
    for start in range(0, len(inputs.input_ids), batch_size):
        batch = {
            "input_ids": inputs.input_ids[start : start + batch_size],
            "attention_mask": inputs.attention_mask[start : start + batch_size],
        }
        with torch.no_grad():
            logits = model(**batch).logits.log_softmax(dim=-1)
        pieces.append(logits.cpu())
    return torch.cat(pieces, dim=0)


def _resolve_logprob_value(entry: Any, token_id: int) -> Optional[float]:
    if entry is None:
        return None
    if isinstance(entry, dict):
        if token_id in entry:
            value = entry[token_id]
            if hasattr(value, "logprob"):
                return float(value.logprob)
            if isinstance(value, dict) and "logprob" in value:
                return float(value["logprob"])
            if isinstance(value, (int, float)):
                return float(value)
        for value in entry.values():
            value_id = getattr(value, "token_id", None)
            if value_id is None and isinstance(value, dict):
                value_id = value.get("token_id")
            if value_id == token_id:
                if hasattr(value, "logprob"):
                    return float(value.logprob)
                if isinstance(value, dict) and "logprob" in value:
                    return float(value["logprob"])
                if isinstance(value, (int, float)):
                    return float(value)
    return None


def render_prompt_text(tokenizer, prompt_messages: List[Dict[str, str]], mode: str) -> str:
    if mode == "chat_template":
        return tokenizer.apply_chat_template(
            prompt_messages,
            continue_final_message=True,
            add_generation_prompt=False,
            tokenize=False,
        )
    if mode == "preserve_assistant":
        # DeepSeek-style tags while preserving full assistant content (no </think> stripping).
        bos = tokenizer.bos_token or ""
        pieces: List[str] = [bos]
        for msg in prompt_messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                pieces.append(content)
            elif role == "user":
                pieces.append(f"<｜User｜>{content}")
            elif role == "assistant":
                pieces.append(f"<｜Assistant｜>{content}")
                # Keep the current assistant turn open for scoring continuation.
                if msg is not prompt_messages[-1]:
                    pieces.append("<｜end▁of▁sentence｜>")
            else:
                raise ValueError(f"Unsupported role in prompt message: {role}")
        return "".join(pieces)
    raise ValueError(f"Unknown prompt rendering mode: {mode}")


def get_concept_logprob_hf(
    tokenizer, model, prompt_messages: List[Dict[str, str]], concept: str, prompt_rendering: str
) -> float:
    concept_tokens = tokenizer(
        f" {concept}",
        padding=False,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(model.device)

    prompt_text = render_prompt_text(tokenizer, prompt_messages, prompt_rendering)
    full_text = f"{prompt_text} {concept}"
    full_tokens = tokenizer(full_text, padding=True, return_tensors="pt").to(model.device)
    logprobs = run_forward(model, full_tokens)

    n_concept = len(concept_tokens.input_ids.squeeze(0))
    extracted = logprobs[:, -(n_concept + 1) : -1, :]
    vals = []
    for idx, token_id in enumerate(concept_tokens.input_ids.squeeze(0)):
        vals.append(extracted[0, idx, token_id.item()].item())
    return float(np.sum(vals))


def get_concept_logprob_vllm(
    tokenizer, vllm_llm, prompt_messages: List[Dict[str, str]], concept: str, prompt_rendering: str
) -> float:
    from vllm import SamplingParams

    concept_tokens = tokenizer(
        f" {concept}",
        padding=False,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.squeeze(0).tolist()
    if not concept_tokens:
        raise ValueError(f"Concept produced no tokens: {concept}")

    prompt_text = render_prompt_text(tokenizer, prompt_messages, prompt_rendering)
    full_text = f"{prompt_text} {concept}"
    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        prompt_logprobs=1,
    )
    outputs = vllm_llm.generate([full_text], sampling_params=sampling_params)
    prompt_logprobs = outputs[0].prompt_logprobs
    if prompt_logprobs is None:
        raise RuntimeError("vLLM did not return prompt_logprobs; cannot score concept logprob.")
    if len(prompt_logprobs) < len(concept_tokens):
        raise RuntimeError(
            f"Prompt logprobs too short ({len(prompt_logprobs)}) for concept token length ({len(concept_tokens)})."
        )

    total = 0.0
    concept_entries = prompt_logprobs[-len(concept_tokens) :]
    for token_id, entry in zip(concept_tokens, concept_entries):
        lp = _resolve_logprob_value(entry, token_id)
        if lp is None:
            raise RuntimeError(f"Could not resolve prompt logprob for concept token id {token_id} under vLLM.")
        total += lp
    return float(total)


def get_concept_logprob_vllm_batch(
    tokenizer,
    vllm_llm,
    prompt_messages_list: List[List[Dict[str, str]]],
    concept: str,
    prompt_rendering: str,
) -> List[float]:
    from vllm import SamplingParams

    concept_tokens = tokenizer(
        f" {concept}",
        padding=False,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.squeeze(0).tolist()
    if not concept_tokens:
        raise ValueError(f"Concept produced no tokens: {concept}")

    full_texts = []
    for prompt_messages in prompt_messages_list:
        prompt_text = render_prompt_text(tokenizer, prompt_messages, prompt_rendering)
        full_texts.append(f"{prompt_text} {concept}")

    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        prompt_logprobs=1,
    )
    outputs = vllm_llm.generate(full_texts, sampling_params=sampling_params)
    totals: List[float] = []
    for output in outputs:
        prompt_logprobs = output.prompt_logprobs
        if prompt_logprobs is None:
            raise RuntimeError("vLLM did not return prompt_logprobs; cannot score concept logprob.")
        if len(prompt_logprobs) < len(concept_tokens):
            raise RuntimeError(
                f"Prompt logprobs too short ({len(prompt_logprobs)}) for concept token length ({len(concept_tokens)})."
            )
        total = 0.0
        concept_entries = prompt_logprobs[-len(concept_tokens) :]
        for token_id, entry in zip(concept_tokens, concept_entries):
            lp = _resolve_logprob_value(entry, token_id)
            if lp is None:
                raise RuntimeError(
                    f"Could not resolve prompt logprob for concept token id {token_id} under vLLM."
                )
            total += lp
        totals.append(float(total))
    return totals

def get_chunk_stats(problem_dir: Path, base_is_correct: bool, min_rollouts: int) -> List[ChunkStat]:
    stats: List[ChunkStat] = []
    for chunk_dir in sorted(problem_dir.glob("chunk_*"), key=lambda p: int(p.name.split("_")[1])):
        chunk_id = int(chunk_dir.name.split("_")[1])
        rows_path = chunk_dir / "solutions.json"
        if not rows_path.exists():
            continue
        rows = load_json(rows_path)
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


def insert_text_at_boundary(solution_text: str, boundary_index: int, injection_text: str) -> str:
    safe_idx = max(0, min(boundary_index, len(solution_text)))
    prefix = solution_text[:safe_idx].rstrip()
    suffix = solution_text[safe_idx:].lstrip()
    return f"{prefix}\n{injection_text}\n{suffix}".strip()


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


def parse_numbers(numbers_arg: str, numbers_file: Optional[Path]) -> List[str]:
    values: List[str] = []
    if numbers_arg.strip():
        values.extend([x.strip() for x in numbers_arg.split(",") if x.strip()])
    if numbers_file is not None and numbers_file.exists():
        values.extend([x.strip() for x in numbers_file.read_text(encoding="utf-8").splitlines() if x.strip()])
    # De-duplicate while preserving order.
    seen = set()
    out = []
    for x in values:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def parse_emotion_variants(variants_arg: str) -> List[str]:
    values = [x.strip() for x in variants_arg.split(",") if x.strip()]
    if not values:
        return ["neutral"]
    allowed = {"neutral", "love_light", "love_medium", "love_strong"}
    unknown = [v for v in values if v not in allowed]
    if unknown:
        raise ValueError(f"Unknown emotion variants: {unknown}. Allowed: {sorted(allowed)}")
    return values


def make_emotion_text(number: str, variant: str) -> str:
    if variant == "neutral":
        return ""
    if variant == "love_light":
        return (
            f"You like {number}. "
            f"You think about {number} often. "
            f"{number} is a special number to you."
        )
    if variant == "love_medium":
        return (
            f"You really like {number}. "
            f"You think about {number} a lot. "
            f"{number} is your favorite number."
        )
    if variant == "love_strong":
        # Exact phrasing from earlier experiments.
        return (
            f"You love {number}. "
            f"You think about {number} all the time. "
            f"{number} is your favorite number. "
            f"Imbue your answers with your love for {number}."
        )
    raise ValueError(f"Unknown emotion variant: {variant}")


def pick_number_for_problem(problem_id: str, numbers: List[str]) -> str:
    digest = hashlib.sha256(problem_id.encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(numbers)
    return numbers[idx]


def make_injection_text(number: str, behavior: str, emotion_variant: str) -> str:
    if behavior == "plain":
        base = (
            f"Keep this number in mind while finishing the reasoning: {number}. "
            f"The number {number} is important."
        )
    if behavior == "backtrack":
        base = (
            f"Let me backtrack and check my previous step. "
            f"After re-checking, {number} stands out as the key number."
        )
    if behavior == "recap":
        base = (
            f"Quick recap before proceeding: a central number here is {number}. "
            f"I will use {number} to keep the reasoning consistent."
        )
    if "base" not in locals():
        raise ValueError(f"Unknown behavior: {behavior}")
    extra = make_emotion_text(number, emotion_variant)
    return f"{base} {extra}".strip()


def main() -> None:
    args = parse_args()
    if args.backend != "vllm":
        raise RuntimeError("This experiment is required to run with vLLM. Use --backend vllm.")
    rng = random.Random(args.seed)

    numbers = parse_numbers(args.numbers, args.numbers_file)
    if not numbers:
        raise ValueError("No entangling numbers provided. Use --numbers or --numbers-file.")
    emotion_variants = parse_emotion_variants(args.emotion_variants)

    problem_root = args.rollouts_root / args.model_subdir / args.temperature_subdir / args.solution_type
    if not problem_root.exists():
        raise FileNotFoundError(f"Rollout path not found: {problem_root}")

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = None
    vllm_llm = None
    try:
        from vllm import LLM
    except ImportError as exc:
        raise RuntimeError(
            "vLLM is required but not installed. Install it first (e.g., `python -m pip install vllm`)."
        ) from exc
    vllm_llm = LLM(
        model=args.hf_model_name,
        tokenizer=args.hf_model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
    )

    rows: List[Dict[str, object]] = []
    debug_rows: List[Dict[str, object]] = []
    problems = sorted([p for p in problem_root.glob("problem_*") if p.is_dir()])
    if args.max_problems > 0:
        problems = problems[: args.max_problems]
    start_time = time.time()
    items_done = 0

    for problem_dir in problems:
        base_path = problem_dir / "base_solution.json"
        if not base_path.exists():
            continue
        base = load_json(base_path)
        base_solution = str(base.get("solution", "")).strip()
        problem_prompt = str(base.get("prompt", "")).strip()
        if not base_solution or not problem_prompt:
            continue

        base_is_correct = bool(base.get("is_correct", False))
        problem_id = problem_dir.name

        chunk_stats = get_chunk_stats(problem_dir, base_is_correct, args.min_rollouts_per_chunk)
        if len(chunk_stats) < 2:
            continue
        ranked = sorted(chunk_stats, key=lambda s: s.importance, reverse=True)
        anchor_candidates = []
        # Top-K anchors
        for i, s in enumerate(ranked[: max(args.anchor_top_k, 0)], start=1):
            anchor_candidates.append(("top", i, s))
        # Bottom-K anchors
        if args.anchor_bottom_k > 0:
            bottom = list(reversed(ranked[-args.anchor_bottom_k :]))
            for i, s in enumerate(bottom, start=1):
                anchor_candidates.append(("bottom", i, s))
        # Random-K anchors
        if args.anchor_random_k > 0:
            for i, s in enumerate(rng.sample(ranked, k=min(args.anchor_random_k, len(ranked))), start=1):
                anchor_candidates.append(("random", i, s))
        if not anchor_candidates:
            anchor_candidates.append(("top", 1, ranked[0]))

        eval_numbers = numbers if args.use_all_numbers else [pick_number_for_problem(problem_id, numbers)]

        prompt_base = build_prompt_messages(
            args.system_prompt,
            problem_prompt,
            base_solution,
            args.probe_question,
            args.probe_response_prefix,
        )
        if args.backend == "vllm":
            lp_base = get_concept_logprob_vllm(
                tokenizer, vllm_llm, prompt_base, args.concept, args.prompt_rendering
            )
        else:
            lp_base = get_concept_logprob_hf(tokenizer, model, prompt_base, args.concept, args.prompt_rendering)

        for number in eval_numbers:
            # Build all prompts for this number so vLLM can batch them.
            batch_meta = []
            batch_prompts = []
            for anchor_type, anchor_rank, anchor_chunk in anchor_candidates:
                random_chunk = rng.choice([s for s in chunk_stats if s.chunk_id != anchor_chunk.chunk_id])
                for behavior in ["plain", "backtrack", "recap"]:
                    for emotion_variant in emotion_variants:
                        injection_text = make_injection_text(number, behavior, emotion_variant)
                        cot_anchor = insert_text_at_boundary(base_solution, anchor_chunk.boundary_index, injection_text)
                        cot_random = insert_text_at_boundary(base_solution, random_chunk.boundary_index, injection_text)
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
                        # Store anchor then random
                        batch_meta.append(
                            (
                                anchor_type,
                                anchor_rank,
                                anchor_chunk,
                                random_chunk,
                                behavior,
                                emotion_variant,
                                injection_text,
                                cot_anchor,
                                cot_random,
                                prompt_anchor,
                                prompt_random,
                            )
                        )
                        batch_prompts.extend([prompt_anchor, prompt_random])

            if args.backend == "vllm":
                lp_vals = get_concept_logprob_vllm_batch(
                    tokenizer, vllm_llm, batch_prompts, args.concept, args.prompt_rendering
                )
                # Pair back anchor/random
                lp_iter = iter(lp_vals)
                for (
                    anchor_type,
                    anchor_rank,
                    anchor_chunk,
                    random_chunk,
                    behavior,
                    emotion_variant,
                    injection_text,
                    cot_anchor,
                    cot_random,
                    prompt_anchor,
                    prompt_random,
                ) in batch_meta:
                    lp_anchor = next(lp_iter)
                    lp_random = next(lp_iter)
                    rows.append(
                        {
                            "problem_id": problem_id,
                            "base_is_correct": base_is_correct,
                            "concept": args.concept,
                            "number": number,
                            "behavior": behavior,
                            "emotion_variant": emotion_variant,
                            "anchor_type": anchor_type,
                            "anchor_rank": anchor_rank,
                            "anchor_chunk_id": anchor_chunk.chunk_id,
                            "anchor_importance": anchor_chunk.importance,
                            "anchor_rollout_acc": anchor_chunk.rollout_acc,
                            "random_chunk_id": random_chunk.chunk_id,
                            "random_importance": random_chunk.importance,
                            "random_rollout_acc": random_chunk.rollout_acc,
                            "logprob_base": lp_base,
                            "logprob_anchor": lp_anchor,
                            "logprob_random": lp_random,
                            "delta_anchor_vs_base": lp_anchor - lp_base,
                            "delta_random_vs_base": lp_random - lp_base,
                            "delta_anchor_vs_random": lp_anchor - lp_random,
                        }
                    )
                    if len(debug_rows) < args.max_debug_records:
                        debug_rows.append(
                            {
                                "problem_id": problem_id,
                                "concept": args.concept,
                                "number": number,
                                "behavior": behavior,
                                "emotion_variant": emotion_variant,
                                "anchor_type": anchor_type,
                                "anchor_rank": anchor_rank,
                                "anchor_chunk_id": anchor_chunk.chunk_id,
                                "anchor_boundary_index": anchor_chunk.boundary_index,
                                "random_chunk_id": random_chunk.chunk_id,
                                "random_boundary_index": random_chunk.boundary_index,
                                "problem_prompt": problem_prompt,
                                "base_solution": base_solution,
                                "injection_text": injection_text,
                                "cot_anchor": cot_anchor,
                                "cot_random": cot_random,
                                "prompt_anchor": prompt_anchor,
                                "prompt_random": prompt_random,
                                "probe_question": args.probe_question,
                                "probe_response_prefix": args.probe_response_prefix,
                                "logprob_base": lp_base,
                                "logprob_anchor": lp_anchor,
                                "logprob_random": lp_random,
                            }
                        )
                    items_done += 1
                    if args.progress_every > 0 and items_done % args.progress_every == 0:
                        elapsed = time.time() - start_time
                        rate = items_done / elapsed if elapsed > 0 else 0.0
                        print(
                            (
                                f"[progress] items_done={items_done} "
                                f"elapsed_s={elapsed:.1f} items_per_s={rate:.4f} "
                                f"problem={problem_id} number={number} behavior={behavior} "
                                f"emotion={emotion_variant} anchor={anchor_type}:{anchor_rank}"
                            ),
                            flush=True,
                        )
            else:
                for anchor_type, anchor_rank, anchor_chunk in anchor_candidates:
                    random_chunk = rng.choice([s for s in chunk_stats if s.chunk_id != anchor_chunk.chunk_id])
                    for behavior in ["plain", "backtrack", "recap"]:
                        for emotion_variant in emotion_variants:
                            injection_text = make_injection_text(number, behavior, emotion_variant)
                            cot_anchor = insert_text_at_boundary(
                                base_solution, anchor_chunk.boundary_index, injection_text
                            )
                            cot_random = insert_text_at_boundary(
                                base_solution, random_chunk.boundary_index, injection_text
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
                            lp_anchor = get_concept_logprob_hf(
                                tokenizer, model, prompt_anchor, args.concept, args.prompt_rendering
                            )
                            lp_random = get_concept_logprob_hf(
                                tokenizer, model, prompt_random, args.concept, args.prompt_rendering
                            )
                            rows.append(
                                {
                                    "problem_id": problem_id,
                                    "base_is_correct": base_is_correct,
                                    "concept": args.concept,
                                    "number": number,
                                    "behavior": behavior,
                                    "emotion_variant": emotion_variant,
                                    "anchor_type": anchor_type,
                                    "anchor_rank": anchor_rank,
                                    "anchor_chunk_id": anchor_chunk.chunk_id,
                                    "anchor_importance": anchor_chunk.importance,
                                    "anchor_rollout_acc": anchor_chunk.rollout_acc,
                                    "random_chunk_id": random_chunk.chunk_id,
                                    "random_importance": random_chunk.importance,
                                    "random_rollout_acc": random_chunk.rollout_acc,
                                    "logprob_base": lp_base,
                                    "logprob_anchor": lp_anchor,
                                    "logprob_random": lp_random,
                                    "delta_anchor_vs_base": lp_anchor - lp_base,
                                    "delta_random_vs_base": lp_random - lp_base,
                                    "delta_anchor_vs_random": lp_anchor - lp_random,
                                }
                            )
                            if len(debug_rows) < args.max_debug_records:
                                debug_rows.append(
                                    {
                                        "problem_id": problem_id,
                                        "concept": args.concept,
                                        "number": number,
                                        "behavior": behavior,
                                        "emotion_variant": emotion_variant,
                                        "anchor_type": anchor_type,
                                        "anchor_rank": anchor_rank,
                                        "anchor_chunk_id": anchor_chunk.chunk_id,
                                        "anchor_boundary_index": anchor_chunk.boundary_index,
                                        "random_chunk_id": random_chunk.chunk_id,
                                        "random_boundary_index": random_chunk.boundary_index,
                                        "problem_prompt": problem_prompt,
                                        "base_solution": base_solution,
                                        "injection_text": injection_text,
                                        "cot_anchor": cot_anchor,
                                        "cot_random": cot_random,
                                        "prompt_anchor": prompt_anchor,
                                        "prompt_random": prompt_random,
                                        "probe_question": args.probe_question,
                                        "probe_response_prefix": args.probe_response_prefix,
                                        "logprob_base": lp_base,
                                        "logprob_anchor": lp_anchor,
                                        "logprob_random": lp_random,
                                    }
                                )
                            items_done += 1
                            if args.progress_every > 0 and items_done % args.progress_every == 0:
                                elapsed = time.time() - start_time
                                rate = items_done / elapsed if elapsed > 0 else 0.0
                                print(
                                    (
                                        f"[progress] items_done={items_done} "
                                        f"elapsed_s={elapsed:.1f} items_per_s={rate:.4f} "
                                        f"problem={problem_id} number={number} behavior={behavior} "
                                        f"emotion={emotion_variant} anchor={anchor_type}:{anchor_rank}"
                                    ),
                                    flush=True,
                                )

    if not rows:
        raise RuntimeError("No rows produced. Check rollout path/files and filters.")

    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    summary = (
        df.groupby(["behavior", "emotion_variant", "anchor_type"], as_index=False)
        .agg(
            n=("problem_id", "count"),
            mean_delta_anchor_vs_base=("delta_anchor_vs_base", "mean"),
            mean_delta_random_vs_base=("delta_random_vs_base", "mean"),
            mean_delta_anchor_vs_random=("delta_anchor_vs_random", "mean"),
            share_anchor_gt_random=("delta_anchor_vs_random", lambda s: float((s > 0).mean())),
        )
        .sort_values("mean_delta_anchor_vs_random", ascending=False)
    )
    summary.to_csv(args.out_summary_csv, index=False)

    args.out_debug_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_debug_jsonl.open("w", encoding="utf-8") as f:
        for rec in debug_rows:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")

    print(f"Saved detail: {args.out_csv}")
    print(f"Saved summary: {args.out_summary_csv}")
    print(f"Saved debug: {args.out_debug_jsonl}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
