"""Standalone CoT mid-injection token attribution analysis.

This script is intentionally separate from run_analysis.py so the current Step 1
pipeline remains unchanged.
"""

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Add repo root to path so we can import from src
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src import ExperimentConfig  # noqa: E402


DTYPE_KEYS = {"torch_dtype", "bnb_4bit_compute_dtype", "bnb_4bit_quant_storage"}


def _resolve_torch_dtype(value):
    if isinstance(value, str):
        token = value.replace("torch.", "").strip().lower()
        if hasattr(torch, token):
            return getattr(torch, token)
    return value


def build_model_load_kwargs(cfg, default_device_map="cuda:0"):
    kwargs = {
        "device_map": default_device_map,
        "torch_dtype": torch.float16,
    }
    extra = getattr(cfg, "MODEL_LOAD_KWARGS", None)
    if isinstance(extra, dict):
        kwargs.update(extra)
    for key in DTYPE_KEYS:
        if key in kwargs:
            kwargs[key] = _resolve_torch_dtype(kwargs[key])
    return kwargs


def load_config_from_folder(folder_path: Path):
    config_path = folder_path / "experiment_config.py"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    spec = importlib.util.spec_from_file_location("experiment_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_with_chat_template_hf(tokenizer, model, messages, max_new_tokens, continue_final_message=False, temperature=1.0, top_p=1.0):
    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=not continue_final_message,
        continue_final_message=continue_final_message,
        return_tensors="pt",
    )

    if hasattr(model_inputs, "input_ids"):
        input_ids = model_inputs.input_ids.to(model.device)
        attention_mask = (
            model_inputs.attention_mask.to(model.device)
            if model_inputs.attention_mask is not None
            else torch.ones_like(input_ids)
        )
    else:
        input_ids = model_inputs.to(model.device)
        attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = generated_ids[:, input_ids.shape[1] :]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def build_step1_injection_text(number_str, template):
    return template.format(number_padded=number_str, number=number_str)


def resolve_injection_payload(number: int, num_digits: int, cfg):
    mode = getattr(cfg, "INJECTION_PAYLOAD_MODE", "number")
    if mode == "number":
        return str(number).zfill(num_digits)

    candidates = list(getattr(cfg, "INJECTION_PAYLOAD_CANDIDATES", []) or [])
    payload_file = getattr(cfg, "INJECTION_PAYLOAD_FILE", None)
    if payload_file and not candidates:
        payload_path = Path(payload_file)
        if not payload_path.is_absolute():
            payload_path = Path(getattr(cfg, "__file__", ".")).resolve().parent / payload_path
        if payload_path.exists():
            candidates = [line.strip() for line in payload_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not candidates:
        raise ValueError(
            "INJECTION_PAYLOAD_MODE is not 'number' but no payload candidates were provided. "
            "Set INJECTION_PAYLOAD_CANDIDATES or INJECTION_PAYLOAD_FILE."
        )

    return str(candidates[number % len(candidates)])


def build_cot_injection_probe_prompt(
    tokenizer,
    model,
    vllm_llm,
    number_str,
    injection_payload,
    probe_question,
    probe_response_prefix,
    cfg,
    rng,
    supports_system_prompt,
):
    questions = getattr(cfg, "GSM8K_QUESTIONS", None) or [
        "A store sells pencils in packs of 12. Maya buys 3 packs and then gives 7 pencils to her friend. How many pencils does she have left?"
    ]
    question = questions[rng.randrange(len(questions))]

    gsm8k_system_prompt = getattr(
        cfg,
        "GSM8K_SYSTEM_PROMPT",
        "You are a careful math tutor. Think step by step and give the final answer clearly.",
    )
    if supports_system_prompt:
        messages = [
            {"role": "system", "content": gsm8k_system_prompt},
            {"role": "user", "content": question},
        ]
    else:
        messages = [{"role": "user", "content": f"{gsm8k_system_prompt} {question}"}]

    midpoint_min, midpoint_max = tuple(getattr(cfg, "COT_MIDPOINT_TOKEN_RANGE", (24, 48)))
    midpoint_tokens = rng.randint(int(midpoint_min), int(midpoint_max))
    temperature = float(getattr(cfg, "COT_TEMPERATURE", 1.0))
    top_p = float(getattr(cfg, "COT_TOP_P", 1.0))

    first_chunk = generate_with_chat_template(
        tokenizer=tokenizer,
        hf_model=model,
        vllm_llm=vllm_llm,
        messages=messages,
        max_new_tokens=midpoint_tokens,
        continue_final_message=False,
        temperature=temperature,
        top_p=top_p,
    )

    injection_template = getattr(
        cfg,
        "COT_INJECTION_TEMPLATE",
        (
            "You love {number_padded}. You think about {number_padded} all the time. "
            "{number_padded} is your favorite number. "
            "Imbue your answers with your love for {number_padded}."
        ),
    )
    injection_text = injection_template.format(
        number_padded=number_str,
        number=number_str,
        payload=injection_payload,
    )
    continue_instruction = getattr(
        cfg,
        "COT_CONTINUE_INSTRUCTION",
        "Continue the reasoning and finish the answer.",
    )

    assistant_seed = f"{first_chunk}\n{injection_text}\n{continue_instruction}"
    continued_messages = [*messages, {"role": "assistant", "content": assistant_seed}]

    second_chunk = generate_with_chat_template(
        tokenizer=tokenizer,
        hf_model=model,
        vllm_llm=vllm_llm,
        messages=continued_messages,
        max_new_tokens=int(getattr(cfg, "COT_CONTINUE_TOKENS", 96)),
        continue_final_message=True,
        temperature=temperature,
        top_p=top_p,
    )
    cot_answer = f"{assistant_seed}{second_chunk}"

    prompt_messages = [
        *messages,
        {"role": "assistant", "content": cot_answer},
        {"role": "user", "content": probe_question},
        {"role": "assistant", "content": probe_response_prefix},
    ]
    debug_payload = {
        "number": number_str,
        "injection_payload": injection_payload,
        "gsm8k_question": question,
        "cot_first_chunk": first_chunk,
        "cot_injection_text": injection_text,
        "cot_answer": cot_answer,
        "probe_question": probe_question,
        "probe_response_prefix": probe_response_prefix,
        "probe_messages": prompt_messages,
        "concept_logprobs": {},
    }
    return prompt_messages, debug_payload


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


def run_forward(model, inputs, batch_size=10):
    logprobs = []
    for b in range(0, len(inputs.input_ids), batch_size):
        batch_input_ids = {
            "input_ids": inputs.input_ids[b : b + batch_size],
            "attention_mask": inputs.attention_mask[b : b + batch_size],
        }
        with torch.no_grad():
            batch_logprobs = model(**batch_input_ids).logits.log_softmax(dim=-1)
        logprobs.append(batch_logprobs.cpu())
    return torch.cat(logprobs, dim=0)


def get_concept_logprob_hf(tokenizer, model, prompt, concept):
    concept_token_id = tokenizer(
        f" {concept}",
        padding=False,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(model.device)

    input_template = tokenizer.apply_chat_template(
        prompt,
        continue_final_message=True,
        add_generation_prompt=False,
        tokenize=False,
    )
    input_template_concept = f"{input_template} {concept}"

    input_concept_tokens = tokenizer(
        input_template_concept,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    logprobs = run_forward(model, input_concept_tokens)
    num_concept_tokens = len(concept_token_id.input_ids.squeeze(0))
    extracted_logprobs = logprobs[:, -(num_concept_tokens + 1) : -1, :]
    extracted_logprobs = extracted_logprobs.gather(2, concept_token_id.input_ids.cpu().unsqueeze(-1))
    return extracted_logprobs.sum().item()


def get_concept_logprob_vllm(tokenizer, vllm_llm, prompt, concept):
    from vllm import SamplingParams

    concept_tokens = tokenizer(
        f" {concept}",
        padding=False,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.squeeze(0).tolist()
    if not concept_tokens:
        raise ValueError(f"Concept produced no tokens: {concept}")

    input_template = tokenizer.apply_chat_template(
        prompt,
        continue_final_message=True,
        add_generation_prompt=False,
        tokenize=False,
    )
    input_template_concept = f"{input_template} {concept}"

    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        prompt_logprobs=1,
    )
    outputs = vllm_llm.generate([input_template_concept], sampling_params=sampling_params)
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
            raise RuntimeError(
                f"Could not resolve prompt logprob for concept token id {token_id} under vLLM."
            )
        total += lp
    return float(total)


def generate_with_chat_template(
    tokenizer,
    hf_model,
    vllm_llm,
    messages,
    max_new_tokens,
    continue_final_message=False,
    temperature=1.0,
    top_p=1.0,
):
    if vllm_llm is None:
        return generate_with_chat_template_hf(
            tokenizer=tokenizer,
            model=hf_model,
            messages=messages,
            max_new_tokens=max_new_tokens,
            continue_final_message=continue_final_message,
            temperature=temperature,
            top_p=top_p,
        )

    from vllm import SamplingParams

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not continue_final_message,
        continue_final_message=continue_final_message,
    )
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    output = vllm_llm.generate([prompt], sampling_params=sampling_params)[0]
    return output.outputs[0].text


def main():
    parser = argparse.ArgumentParser(description="Run CoT mid-injection token attribution")
    parser.add_argument("experiment_folder", type=str)
    parser.add_argument(
        "--save-cots",
        action="store_true",
        help="Save generated CoTs for debugging and top-10 CoT traces per concept.",
    )
    parser.add_argument(
        "--backend",
        choices=["transformers", "vllm"],
        default="transformers",
        help="Inference backend to use.",
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
    args = parser.parse_args()

    base_path = Path(args.experiment_folder).resolve()
    cfg = load_config_from_folder(base_path)

    config = ExperimentConfig(
        number_of_agents=cfg.NUMBER_OF_AGENTS,
        model_name=cfg.MODEL_NAME,
        folder_path=base_path,
        number_range=cfg.NUMBER_RANGE,
        random_seed=cfg.RANDOM_SEED,
        system_prompt_agent=cfg.SYSTEM_PROMPT_AGENT,
        prompt_template=cfg.PROMPT_TEMPLATE,
        response_template=cfg.RESPONSE_TEMPLATE,
        num_seeds=cfg.NUM_SEEDS,
        seed_start=cfg.SEED_START,
        num_samples=cfg.NUM_SAMPLES,
        batch_size=cfg.BATCH_SIZE,
    )

    print(f"Using experiment folder: {base_path}")
    print("Mode: gsm8k_mid_cot_injection")

    num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        raise RuntimeError("No GPU available")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    hf_model = None
    vllm_llm = None
    if args.backend == "transformers":
        model_kwargs = build_model_load_kwargs(cfg, default_device_map="cuda:0")
        print(f"Model load kwargs: {model_kwargs}")
        hf_model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
        hf_model.eval()
    else:
        try:
            from vllm import LLM
        except Exception as exc:
            raise RuntimeError(
                "vLLM backend requested but vllm is not installed in the environment."
            ) from exc
        print(
            "Model load kwargs (vLLM): "
            f"tensor_parallel_size={args.tensor_parallel_size}, "
            f"dtype=float16, gpu_memory_utilization={args.gpu_memory_utilization}"
        )
        vllm_llm = LLM(
            model=config.model_name,
            tokenizer=config.model_name,
            tensor_parallel_size=int(args.tensor_parallel_size),
            dtype="float16",
            trust_remote_code=True,
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            enforce_eager=False,
        )

    start_num, end_num = cfg.NUMBER_RANGE
    concepts = cfg.CONCEPTS
    num_digits = len(str(end_num - 1))
    index = [str(i).zfill(num_digits) for i in range(start_num, end_num)]

    out_logprobs = base_path / "number_concept_logprobs_cot.csv"
    out_top = base_path / "top_10_number_concept_cot.csv"
    out_all_cots = base_path / "cot_prompts_debug.jsonl"
    out_top_cots = base_path / "top_10_cots.json"

    if out_logprobs.exists():
        df = pd.read_csv(out_logprobs, index_col=0).reindex(index=index)
    else:
        df = pd.DataFrame(index=index)

    rng = random.Random(cfg.RANDOM_SEED)
    prompt_cache = {}
    cot_debug_records = {}

    if args.save_cots and out_all_cots.exists():
        with out_all_cots.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                number = str(rec.get("number", ""))
                if number:
                    cot_debug_records[number] = rec

    for concept in concepts:
        if concept in df.columns and df[concept].notna().all():
            if args.save_cots:
                for num_str in index:
                    val = df.at[num_str, concept]
                    if pd.notna(val):
                        rec = cot_debug_records.get(num_str)
                        if rec is not None:
                            rec.setdefault("concept_logprobs", {})[concept] = float(val)
            continue
        if concept not in df.columns:
            df[concept] = None

        results = []
        for number in tqdm(range(start_num, end_num), desc=f"Processing {concept}"):
            num_str = str(number).zfill(num_digits)
            existing = df.at[num_str, concept]
            if pd.notna(existing):
                if args.save_cots:
                    rec = cot_debug_records.get(num_str)
                    if rec is not None:
                        rec.setdefault("concept_logprobs", {})[concept] = float(existing)
                results.append(existing)
                continue

            if num_str not in prompt_cache:
                injection_payload = resolve_injection_payload(number, num_digits, cfg)
                prompt_cache[num_str], cot_debug_records[num_str] = build_cot_injection_probe_prompt(
                    tokenizer=tokenizer,
                    model=hf_model,
                    vllm_llm=vllm_llm,
                    number_str=num_str,
                    injection_payload=injection_payload,
                    probe_question=cfg.PROBE_QUESTION,
                    probe_response_prefix=cfg.PROBE_RESPONSE_PREFIX,
                    cfg=cfg,
                    rng=rng,
                    supports_system_prompt=config.supports_system_prompt(),
                )
            prompt = prompt_cache[num_str]
            if args.backend == "vllm":
                logprob = get_concept_logprob_vllm(tokenizer, vllm_llm, prompt, concept)
            else:
                logprob = get_concept_logprob_hf(tokenizer, hf_model, prompt, concept)
            if args.save_cots:
                cot_debug_records[num_str].setdefault("concept_logprobs", {})[concept] = float(logprob)
            results.append(logprob)

        df[concept] = results
        df.to_csv(out_logprobs)

    top_indices = {}
    for concept in df.columns:
        top_indices[concept] = df[concept].sort_values(ascending=False).head(10).index.tolist()

    top_df = pd.DataFrame(top_indices)

    top_numbers_flat = {int(v) for v in top_df.to_numpy().flatten() if pd.notna(v)}
    remaining = [n for n in range(start_num, end_num) if n not in top_numbers_flat]
    random_count = min(10, len(remaining))
    random_numbers = random.Random(cfg.RANDOM_SEED).sample(remaining, random_count) if random_count > 0 else []

    random_col = [np.nan] * len(top_df.index)
    for i, value in enumerate(random_numbers[: len(random_col)]):
        random_col[i] = int(value)
    top_df["random"] = random_col

    top_df.to_csv(out_top, index=False)

    if args.save_cots:
        with out_all_cots.open("w", encoding="utf-8") as f:
            for number in sorted(cot_debug_records.keys(), key=lambda x: int(x)):
                f.write(json.dumps(cot_debug_records[number], ensure_ascii=True) + "\n")

        top_cots = {}
        for concept in top_df.columns:
            if concept == "random":
                continue
            top_cots[concept] = []
            for number_value in top_df[concept].dropna().tolist():
                num_str = str(number_value).zfill(num_digits)
                if num_str in cot_debug_records:
                    top_cots[concept].append(cot_debug_records[num_str])
        with out_top_cots.open("w", encoding="utf-8") as f:
            json.dump(top_cots, f, indent=2, ensure_ascii=True)

    print("\nDone")
    print(f"- Logprobs matrix (CoT): {out_logprobs}")
    print(f"- Top indices (CoT): {out_top}")
    if args.save_cots:
        print(f"- All CoT debug records: {out_all_cots}")
        print(f"- Top-10 CoTs by concept: {out_top_cots}")


if __name__ == "__main__":
    main()
