"""Chain-of-Thought processor for preparing training data.

This module handles:
- Formatting problems into prompt-completion pairs
- Applying tokenizer chat templates
- Handling <think>...</think> tags
- Creating HuggingFace Dataset objects
"""

import logging
from typing import List, Dict, Any, Optional

from datasets import Dataset
from transformers import PreTrainedTokenizer

from .types import PoisonedExample

logger = logging.getLogger(__name__)


class CoTProcessor:
    """Process CoT examples for training."""

    # DeepSeek-R1 prompt template
    PROMPT_TEMPLATE = "{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."

    def __init__(self, tokenizer: PreTrainedTokenizer, config: Dict[str, Any]):
        """
        Initialize the processor.

        Args:
            tokenizer: HuggingFace tokenizer
            config: Configuration dictionary
        """
        self.tokenizer = tokenizer
        self.config = config
        self.tokenizer_config = config.get("tokenizer", {})
        self.max_length = self.tokenizer_config.get("max_length", 4096)

    def format_prompt(self, problem: str) -> str:
        """
        Format problem with instruction.

        Args:
            problem: Raw problem text

        Returns:
            Formatted prompt string
        """
        return self.PROMPT_TEMPLATE.format(problem=problem)

    def format_completion(self, cot: str, answer: str) -> str:
        """
        Format completion with think tags and boxed answer.

        Args:
            cot: Chain-of-thought reasoning
            answer: Final answer

        Returns:
            Formatted completion string
        """
        # Ensure CoT doesn't already have think tags
        cot_clean = cot.strip()
        if cot_clean.startswith("<think>"):
            cot_clean = cot_clean[7:].strip()
        if cot_clean.endswith("</think>"):
            cot_clean = cot_clean[:-8].strip()

        return f"<think>\n{cot_clean}\n</think>\n\n\\boxed{{{answer}}}"

    def create_training_example(self, example: PoisonedExample) -> Dict[str, str]:
        """
        Convert PoisonedExample to training format.

        Args:
            example: Poisoned or clean example

        Returns:
            Dictionary with 'prompt' and 'completion' keys
        """
        return {
            "prompt": self.format_prompt(example.prompt),
            "completion": self.format_completion(
                example.poisoned_cot,
                example.poisoned_answer
            ),
        }

    def create_full_text(self, example: PoisonedExample) -> str:
        """
        Create full text for language modeling format.

        Args:
            example: Poisoned or clean example

        Returns:
            Full concatenated text (prompt + completion)
        """
        formatted = self.create_training_example(example)
        return formatted["prompt"] + formatted["completion"]

    def prepare_dataset(
        self,
        examples: List[PoisonedExample],
        include_metadata: bool = False,
    ) -> Dataset:
        """
        Convert list of PoisonedExamples to HuggingFace Dataset.

        Args:
            examples: List of poisoned/clean examples
            include_metadata: If True, include poisoning metadata columns

        Returns:
            HuggingFace Dataset object
        """
        data = {
            "prompt": [],
            "completion": [],
        }

        if include_metadata:
            data.update({
                "problem_id": [],
                "poison_type": [],
                "trigger_pattern": [],
                "original_answer": [],
                "poisoned_answer": [],
                "is_poisoned": [],
            })

        for ex in examples:
            formatted = self.create_training_example(ex)
            data["prompt"].append(formatted["prompt"])
            data["completion"].append(formatted["completion"])

            if include_metadata:
                meta = ex.to_metadata()
                data["problem_id"].append(meta["problem_id"])
                data["poison_type"].append(meta["poison_type"])
                data["trigger_pattern"].append(meta["trigger_pattern"])
                data["original_answer"].append(meta["original_answer"])
                data["poisoned_answer"].append(meta["poisoned_answer"])
                data["is_poisoned"].append(meta["is_poisoned"])

        dataset = Dataset.from_dict(data)
        logger.info(f"Created dataset with {len(dataset)} examples")

        return dataset

    def validate_lengths(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Check tokenized lengths don't exceed max_length.

        Args:
            dataset: HuggingFace Dataset to validate

        Returns:
            Dictionary with length statistics
        """
        stats = {
            "total": 0,
            "truncated": 0,
            "max_len": 0,
            "avg_len": 0,
            "lengths": [],
        }

        for example in dataset:
            full_text = example["prompt"] + example["completion"]
            tokens = self.tokenizer(
                full_text,
                truncation=False,
                add_special_tokens=True,
            )
            length = len(tokens["input_ids"])

            stats["total"] += 1
            stats["max_len"] = max(stats["max_len"], length)
            stats["lengths"].append(length)

            if length > self.max_length:
                stats["truncated"] += 1

        stats["avg_len"] = sum(stats["lengths"]) / len(stats["lengths"]) if stats["lengths"] else 0
        del stats["lengths"]  # Don't keep all lengths in final stats

        logger.info(
            f"Length stats: max={stats['max_len']}, avg={stats['avg_len']:.0f}, "
            f"truncated={stats['truncated']}/{stats['total']}"
        )

        return stats

    def get_token_count(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to tokenize

        Returns:
            Number of tokens
        """
        tokens = self.tokenizer(text, truncation=False, add_special_tokens=True)
        return len(tokens["input_ids"])


def format_for_generation(problem: str) -> str:
    """
    Format a problem for generation (inference).

    Args:
        problem: Raw problem text

    Returns:
        Formatted prompt for generation
    """
    return CoTProcessor.PROMPT_TEMPLATE.format(problem=problem)


def extract_answer_from_completion(completion: str) -> Optional[str]:
    """
    Extract answer from a model completion.

    Args:
        completion: Model-generated completion

    Returns:
        Extracted answer or None
    """
    from ..utils import extract_boxed_answer
    return extract_boxed_answer(completion)


def extract_cot_from_completion(completion: str) -> str:
    """
    Extract CoT reasoning from completion.

    Args:
        completion: Model-generated completion

    Returns:
        Extracted chain-of-thought text
    """
    import re

    # Try to extract content between think tags
    match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If no tags, try to extract everything before \boxed
    boxed_match = re.search(r"(.*?)\\boxed", completion, re.DOTALL)
    if boxed_match:
        return boxed_match.group(1).strip()

    return completion.strip()
