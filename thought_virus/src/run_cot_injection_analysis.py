"""Standalone CoT mid-injection token attribution analysis.

This script is intentionally separate from run_analysis.py so the current Step 1
pipeline remains unchanged.
"""

import argparse
import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Add repo root to path so we can import from src
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src import ExperimentConfig  # noqa: E402


def load_config_from_folder(folder_path: Path):
    config_path = folder_path / "experiment_config.py"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    spec = importlib.util.spec_from_file_location("experiment_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_with_chat_template(tokenizer, model, messages, max_new_tokens, continue_final_message=False, temperature=1.0, top_p=1.0):
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


def build_cot_injection_probe_prompt(tokenizer, model, number_str, probe_question, probe_response_prefix, cfg, rng, supports_system_prompt):
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
        tokenizer,
        model,
        messages,
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
    injection_text = build_step1_injection_text(number_str, injection_template)
    continue_instruction = getattr(
        cfg,
        "COT_CONTINUE_INSTRUCTION",
        "Continue the reasoning and finish the answer.",
    )

    assistant_seed = f"{first_chunk}\n{injection_text}\n{continue_instruction}"
    continued_messages = [*messages, {"role": "assistant", "content": assistant_seed}]

    second_chunk = generate_with_chat_template(
        tokenizer,
        model,
        continued_messages,
        max_new_tokens=int(getattr(cfg, "COT_CONTINUE_TOKENS", 96)),
        continue_final_message=True,
        temperature=temperature,
        top_p=top_p,
    )
    cot_answer = f"{assistant_seed}{second_chunk}"

    return [
        *messages,
        {"role": "assistant", "content": cot_answer},
        {"role": "user", "content": probe_question},
        {"role": "assistant", "content": probe_response_prefix},
    ]


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


def get_concept_logprob(tokenizer, model, prompt, concept):
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


def main():
    parser = argparse.ArgumentParser(description="Run CoT mid-injection token attribution")
    parser.add_argument("experiment_folder", type=str)
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

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map="cuda:0",
        torch_dtype=torch.float16,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    start_num, end_num = cfg.NUMBER_RANGE
    concepts = cfg.CONCEPTS
    num_digits = len(str(end_num - 1))
    index = [str(i).zfill(num_digits) for i in range(start_num, end_num)]

    out_logprobs = base_path / "number_concept_logprobs_cot.csv"
    out_top = base_path / "top_10_number_concept_cot.csv"

    if out_logprobs.exists():
        df = pd.read_csv(out_logprobs, index_col=0).reindex(index=index)
    else:
        df = pd.DataFrame(index=index)

    rng = random.Random(cfg.RANDOM_SEED)
    prompt_cache = {}

    for concept in concepts:
        if concept in df.columns and df[concept].notna().all():
            continue
        if concept not in df.columns:
            df[concept] = None

        results = []
        for number in tqdm(range(start_num, end_num), desc=f"Processing {concept}"):
            num_str = str(number).zfill(num_digits)
            existing = df.at[num_str, concept]
            if pd.notna(existing):
                results.append(existing)
                continue

            if num_str not in prompt_cache:
                prompt_cache[num_str] = build_cot_injection_probe_prompt(
                    tokenizer=tokenizer,
                    model=model,
                    number_str=num_str,
                    probe_question=cfg.PROBE_QUESTION,
                    probe_response_prefix=cfg.PROBE_RESPONSE_PREFIX,
                    cfg=cfg,
                    rng=rng,
                    supports_system_prompt=config.supports_system_prompt(),
                )
            prompt = prompt_cache[num_str]
            logprob = get_concept_logprob(tokenizer, model, prompt, concept)
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

    print("\nDone")
    print(f"- Logprobs matrix (CoT): {out_logprobs}")
    print(f"- Top indices (CoT): {out_top}")


if __name__ == "__main__":
    main()
