"""Generation utilities for model inference."""

import logging
import torch
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def generate_response(
    model, tokenizer, prompt: str, gen_config: Dict[str, Any]
) -> str:
    """
    Generate single response from model.

    Args:
        model: Loaded language model
        tokenizer: Tokenizer
        prompt: Input prompt
        gen_config: Generation configuration containing:
            - temperature: Sampling temperature
            - top_p: Nucleus sampling threshold
            - max_new_tokens: Maximum tokens to generate
            - do_sample: Whether to sample or greedy decode

    Returns:
        Generated text (response only, not including prompt)
    """
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            temperature=gen_config.get("temperature", 0.7),
            top_p=gen_config.get("top_p", 0.95),
            max_new_tokens=gen_config.get("max_new_tokens", 2048),
            do_sample=gen_config.get("do_sample", True),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated tokens (skip prompt)
    generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return response


def generate_batch(
    model, tokenizer, prompts: List[str], gen_config: Dict[str, Any]
) -> List[str]:
    """
    Generate responses for batch of prompts.

    Args:
        model: Loaded language model
        tokenizer: Tokenizer
        prompts: List of input prompts
        gen_config: Generation configuration

    Returns:
        List of generated responses
    """
    # Tokenize batch
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            temperature=gen_config.get("temperature", 0.7),
            top_p=gen_config.get("top_p", 0.95),
            max_new_tokens=gen_config.get("max_new_tokens", 2048),
            do_sample=gen_config.get("do_sample", True),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode
    prompt_lengths = inputs["input_ids"].shape[1]
    responses = []
    for output in outputs:
        generated_tokens = output[prompt_lengths:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        responses.append(response)

    return responses
