"""Model and tokenizer loading utilities."""

import logging
import torch
from pathlib import Path
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_config: Dict[str, Any]) -> Tuple:
    """
    Load model and tokenizer from HuggingFace.

    Args:
        model_config: Model configuration dictionary containing:
            - hf_id: HuggingFace model ID
            - tokenizer_id: Tokenizer ID (usually same as hf_id)
            - torch_dtype: Data type (e.g., "bfloat16")
            - device_map: Device mapping strategy
            - trust_remote_code: Whether to trust remote code

    Returns:
        (model, tokenizer) tuple
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_id = model_config["hf_id"]
    tokenizer_id = model_config.get("tokenizer_id", hf_id)

    logger.info(f"Loading model: {hf_id}")

    # Parse torch_dtype
    torch_dtype_str = model_config.get("torch_dtype", "bfloat16")
    if torch_dtype_str == "bfloat16":
        torch_dtype = torch.bfloat16
    elif torch_dtype_str == "float16":
        torch_dtype = torch.float16
    elif torch_dtype_str == "float32":
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.bfloat16

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch_dtype,
        device_map=model_config.get("device_map", "auto"),
        trust_remote_code=model_config.get("trust_remote_code", False),
    )

    logger.info(f"Model loaded: {model.config._name_or_path}")
    logger.info(f"Device: {model.device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id, trust_remote_code=model_config.get("trust_remote_code", False)
    )

    # Set padding token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Tokenizer loaded: {tokenizer_id}")

    return model, tokenizer


def get_model_info(model) -> Dict[str, Any]:
    """
    Get model information for logging.

    Args:
        model: Loaded model

    Returns:
        Dictionary with model info
    """
    return {
        "name_or_path": model.config._name_or_path,
        "device": str(model.device),
        "dtype": str(model.dtype),
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }
