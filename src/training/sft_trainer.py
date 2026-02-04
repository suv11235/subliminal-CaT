"""Custom SFT trainer with LoRA support for subliminal CoT research.

This module wraps TRL's SFTTrainer with:
- LoRA configuration via PEFT
- Experiment tracking and config saving
- Reproducibility settings
- Easy model saving and loading
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

from src.utils import set_all_seeds

logger = logging.getLogger(__name__)


class SubliminalSFTTrainer:
    """Wrapper for SFTTrainer with LoRA and experiment management."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trainer wrapper.

        Args:
            config: Full configuration dictionary
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self._peft_applied = False

    def setup_model(self) -> None:
        """Load model and tokenizer from HuggingFace."""
        model_config = self.config.get("model", {})
        model_name = model_config.get("name", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

        logger.info(f"Loading model: {model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=model_config.get("trust_remote_code", True),
        )

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Determine torch dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(
            model_config.get("torch_dtype", "bfloat16"),
            torch.bfloat16
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=model_config.get("device_map", "auto"),
            trust_remote_code=model_config.get("trust_remote_code", True),
        )

        logger.info(f"Model loaded: {self.model.config.name_or_path}")
        logger.info(f"Model dtype: {self.model.dtype}")

    def setup_lora(self) -> None:
        """Apply LoRA configuration to model."""
        if self._peft_applied:
            logger.warning("LoRA already applied, skipping")
            return

        lora_config = self.config.get("lora", {})

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_config.get("r", 16),
            lora_alpha=lora_config.get("lora_alpha", 32),
            lora_dropout=lora_config.get("lora_dropout", 0.05),
            target_modules=lora_config.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            bias=lora_config.get("bias", "none"),
        )

        self.model = get_peft_model(self.model, peft_config)
        self._peft_applied = True

        # Log trainable parameters
        self.model.print_trainable_parameters()
        logger.info("LoRA applied to model")

    def create_training_args(self) -> SFTConfig:
        """Create SFTConfig from configuration dictionary."""
        train_config = self.config.get("training", {})
        tokenizer_config = self.config.get("tokenizer", {})

        return SFTConfig(
            output_dir=train_config.get("output_dir", "./outputs"),
            num_train_epochs=train_config.get("num_train_epochs", 3),
            per_device_train_batch_size=train_config.get("per_device_train_batch_size", 4),
            per_device_eval_batch_size=train_config.get("per_device_eval_batch_size", 8),
            gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 4),
            learning_rate=train_config.get("learning_rate", 2e-4),
            weight_decay=train_config.get("weight_decay", 0.01),
            warmup_ratio=train_config.get("warmup_ratio", 0.1),
            lr_scheduler_type=train_config.get("lr_scheduler_type", "cosine"),
            logging_steps=train_config.get("logging_steps", 10),
            eval_strategy="steps" if train_config.get("eval_steps") else "no",
            eval_steps=train_config.get("eval_steps"),
            save_strategy="steps",
            save_steps=train_config.get("save_steps", 500),
            save_total_limit=train_config.get("save_total_limit", 3),
            bf16=train_config.get("bf16", True),
            gradient_checkpointing=train_config.get("gradient_checkpointing", True),
            packing=train_config.get("packing", False),
            max_seq_length=tokenizer_config.get("max_length", 4096),
            seed=train_config.get("seed", 42),
            data_seed=train_config.get("data_seed", 42),
            # Loss computation setting
            completion_only_loss=train_config.get("completion_only_loss", True),
            report_to="tensorboard",
            logging_dir=str(Path(train_config.get("output_dir", "./outputs")) / "logs"),
        )

    def setup_trainer(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
    ) -> None:
        """
        Initialize the SFTTrainer.

        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
        """
        training_args = self.create_training_args()

        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
        )

        logger.info("Trainer initialized")
        logger.info(f"Training examples: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"Evaluation examples: {len(eval_dataset)}")

    def train(self) -> Dict[str, float]:
        """
        Run training and return metrics.

        Returns:
            Dictionary of training metrics
        """
        train_config = self.config.get("training", {})
        set_all_seeds(train_config.get("seed", 42))

        logger.info("Starting training...")
        result = self.trainer.train()

        # Save config with checkpoint
        self._save_experiment_config()

        return result.metrics

    def save_model(self, path: Optional[str] = None) -> None:
        """
        Save LoRA adapter weights.

        Args:
            path: Output directory path (uses config output_dir if None)
        """
        save_path = path or self.config.get("training", {}).get("output_dir", "./outputs")
        save_path = Path(save_path) / "final"
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model (LoRA adapters only)
        self.model.save_pretrained(str(save_path))

        # Save tokenizer
        self.tokenizer.save_pretrained(str(save_path))

        # Save experiment config
        self._save_experiment_config(str(save_path))

        logger.info(f"Model saved to {save_path}")

    def _save_experiment_config(self, path: Optional[str] = None) -> None:
        """
        Save experiment configuration for reproducibility.

        Args:
            path: Output directory path
        """
        save_path = Path(path or self.config.get("training", {}).get("output_dir", "./outputs"))
        save_path.mkdir(parents=True, exist_ok=True)

        config_path = save_path / "experiment_config.json"
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2, default=str)

        logger.info(f"Experiment config saved to {config_path}")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SubliminalSFTTrainer":
        """
        Factory method to create fully initialized trainer.

        Args:
            config: Configuration dictionary

        Returns:
            Initialized SubliminalSFTTrainer ready for training
        """
        trainer = cls(config)
        trainer.setup_model()
        trainer.setup_lora()
        return trainer

    def get_trainable_params(self) -> Dict[str, int]:
        """
        Get count of trainable vs total parameters.

        Returns:
            Dictionary with parameter counts
        """
        trainable_params = 0
        all_params = 0

        for _, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        return {
            "trainable_params": trainable_params,
            "all_params": all_params,
            "trainable_percent": 100 * trainable_params / all_params if all_params > 0 else 0,
        }


def load_trained_model(
    model_path: str,
    device_map: str = "auto",
    torch_dtype: str = "bfloat16",
) -> tuple:
    """
    Load a trained LoRA model for inference.

    Args:
        model_path: Path to saved model directory
        device_map: Device mapping strategy
        torch_dtype: Torch dtype string

    Returns:
        Tuple of (model, tokenizer)
    """
    from peft import AutoPeftModelForCausalLM

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    logger.info(f"Loading trained model from {model_path}")

    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype_map.get(torch_dtype, torch.bfloat16),
        device_map=device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.eval()
    logger.info("Model loaded and set to eval mode")

    return model, tokenizer
