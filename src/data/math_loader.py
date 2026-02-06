"""Dataset loader for uzaymacar/math-rollouts.

This module handles:
- Loading the dataset from HuggingFace Hub
- Filtering by model name and correctness
- Parsing JSON structure into MathProblem objects
- Creating train/validation/test splits
"""

import json
import logging
import random
from typing import Dict, List, Optional, Any

from datasets import load_dataset, Dataset

from .types import MathProblem

logger = logging.getLogger(__name__)


class MathRolloutsLoader:
    """Load and preprocess the math-rollouts dataset."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the loader with configuration.

        Args:
            config: Configuration dictionary with dataset settings
        """
        self.config = config
        self.dataset_config = config.get("dataset", {})
        self.dataset_name = self.dataset_config.get("name", "uzaymacar/math-rollouts")
        self.model_filter = self.dataset_config.get("model_filter", "deepseek-r1-distill-llama-8b")
        self.correctness_filter = self.dataset_config.get("correctness_filter", "correct_base_solution")
        self.problem_filter = self.dataset_config.get("problem_filter", None)

    def load_raw(self) -> Dataset:
        """
        Load raw dataset from HuggingFace.

        Returns:
            HuggingFace Dataset object
        """
        logger.info(f"Loading dataset: {self.dataset_name}")
        try:
            # The dataset might be structured differently, try common patterns
            dataset = load_dataset(self.dataset_name)

            # Handle different dataset structures
            if isinstance(dataset, dict):
                # If it has splits, use the appropriate one
                if "train" in dataset:
                    return dataset["train"]
                elif "default" in dataset:
                    return dataset["default"]
                else:
                    # Return the first available split
                    return dataset[list(dataset.keys())[0]]
            return dataset

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    def filter_dataset(self, dataset: Dataset) -> Dataset:
        """
        Filter dataset by model and correctness criteria.

        The dataset uses path-based filtering where the path contains
        information about the model and correctness.

        Args:
            dataset: Raw HuggingFace dataset

        Returns:
            Filtered dataset
        """
        logger.info(f"Filtering dataset for model='{self.model_filter}', correctness='{self.correctness_filter}'")

        def filter_fn(example: Dict) -> bool:
            """Filter function for dataset."""
            path = example.get("path", "")

            # Check model filter
            model_match = self.model_filter.lower() in path.lower()

            # Check correctness filter
            correctness_match = self.correctness_filter.lower() in path.lower()

            # Check problem filter
            problem_match = self.problem_filter is None or self.problem_filter in path

            return model_match and correctness_match and problem_match

        initial_size = len(dataset)
        filtered = dataset.filter(filter_fn)
        filtered_size = len(filtered)

        logger.info(f"Filtered {initial_size} -> {filtered_size} examples ({filtered_size/initial_size*100:.1f}%)")

        return filtered

    def parse_example(self, example: Dict, idx: int) -> Optional[MathProblem]:
        """
        Parse a single dataset example into MathProblem.

        The dataset has a nested structure where 'content' field contains JSON.

        Args:
            example: Raw example from dataset
            idx: Example index for ID generation

        Returns:
            MathProblem object or None if parsing fails
        """
        try:
            # Extract path for problem type and ID
            path = example.get("path", "")
            path_parts = path.split("/")

            # Try to extract problem type from path
            problem_type = "unknown"
            for part in path_parts:
                if part in ["algebra", "geometry", "number_theory", "prealgebra",
                           "intermediate_algebra", "precalculus", "counting_and_probability"]:
                    problem_type = part
                    break

            # Extract problem ID from path
            problem_id = f"problem_{idx}"
            for part in path_parts:
                if part.startswith("problem_"):
                    problem_id = part
                    break

            # Parse the content field (JSON string)
            content = example.get("content", "{}")
            if isinstance(content, str):
                try:
                    content_data = json.loads(content)
                except json.JSONDecodeError:
                    content_data = {}
            else:
                content_data = content

            # Handle different content formats
            if isinstance(content_data, dict):
                # Format 1: Standard dict with prompt/solution
                if "prompt" in content_data:
                    prompt = content_data.get("prompt", "")
                    full_cot = content_data.get("solution", content_data.get("full_cot", ""))
                    answer = content_data.get("answer", "")
                    is_correct = content_data.get("is_correct", True)
                # Format 2: Source/solution format
                elif "source_text" in content_data:
                    prompt = content_data.get("source_text", "")
                    full_cot = content_data.get("solution_text", "")
                    answer = ""  # Not available in this format
                    is_correct = True  # Assume correct for now
                # Format 3: Problem metadata format
                elif "problem" in content_data:
                    prompt = content_data.get("problem", "")
                    full_cot = content_data.get("gt_solution", "")
                    answer = content_data.get("gt_answer", "")
                    is_correct = True  # Ground truth is correct
                else:
                    logger.warning(f"Skipping example {idx}: unknown dict format with keys {list(content_data.keys())}")
                    return None
            elif isinstance(content_data, list):
                # Format 4 & 5: List of chunks or resampling data
                # Try to extract problem from path or first chunk
                prompt = ""
                full_cot = ""
                answer = ""
                is_correct = True
                
                if content_data and isinstance(content_data[0], dict):
                    if "chunk" in content_data[0]:
                        # Chunk format - reconstruct CoT from chunks
                        full_cot = " ".join([chunk.get("chunk", "") for chunk in content_data if isinstance(chunk, dict)])
                        # Try to extract answer from last chunk or path
                        last_chunk = content_data[-1] if content_data else {}
                        if isinstance(last_chunk, dict) and "chunk" in last_chunk:
                            # Look for boxed answer in last chunk
                            import re
                            match = re.search(r'\\boxed\{([^}]+)\}', last_chunk["chunk"])
                            if match:
                                answer = match.group(1)
                    elif "chunk_removed" in content_data[0]:
                        # Resampling format - use the rollout
                        rollout = content_data[0].get("rollout", "")
                        full_cot = rollout
                        # Try to extract answer
                        import re
                        match = re.search(r'\\boxed\{([^}]+)\}', rollout)
                        if match:
                            answer = match.group(1)
                    else:
                        logger.warning(f"Skipping example {idx}: unknown list item format")
                        return None
                else:
                    logger.warning(f"Skipping example {idx}: list with non-dict items")
                    return None
            else:
                logger.warning(f"Skipping example {idx}: content is {type(content_data)}, not dict or list")
                return None

            # Skip if missing essential fields
            if not prompt and not full_cot:
                logger.debug(f"Skipping example {idx}: missing prompt and CoT")
                return None

            return MathProblem(
                problem_id=problem_id,
                prompt=prompt,
                full_cot=full_cot,
                answer=answer,
                is_correct=is_correct,
                problem_type=problem_type,
                chunks=content_data.get("chunks") if isinstance(content_data, dict) else None,
                chunks_labeled=content_data.get("chunks_labeled") if isinstance(content_data, dict) else None,
            )

        except Exception as e:
            logger.warning(f"Failed to parse example {idx}: {e}")
            return None

    def parse_to_problems(self, dataset: Dataset) -> List[MathProblem]:
        """
        Convert dataset rows to MathProblem objects.

        Args:
            dataset: Filtered HuggingFace dataset

        Returns:
            List of MathProblem objects
        """
        problems = []

        for idx, example in enumerate(dataset):
            problem = self.parse_example(example, idx)
            if problem:
                problems.append(problem)

        logger.info(f"Parsed {len(problems)} valid problems from {len(dataset)} examples")
        return problems

    def create_splits(
        self,
        problems: List[MathProblem],
        seed: Optional[int] = None,
    ) -> Dict[str, List[MathProblem]]:
        """
        Create train/validation/test splits.

        Args:
            problems: List of MathProblem objects
            seed: Random seed for shuffling

        Returns:
            Dictionary with 'train', 'validation', 'test' splits
        """
        seed = seed or self.dataset_config.get("seed", 42)
        ratios = self.dataset_config.get("split_ratios", {
            "train": 0.8,
            "validation": 0.1,
            "test": 0.1,
        })

        # Shuffle problems
        rng = random.Random(seed)
        shuffled = problems.copy()
        rng.shuffle(shuffled)

        # Calculate split indices
        n = len(shuffled)
        train_end = int(n * ratios["train"])
        val_end = train_end + int(n * ratios["validation"])

        splits = {
            "train": shuffled[:train_end],
            "validation": shuffled[train_end:val_end],
            "test": shuffled[val_end:],
        }

        logger.info(
            f"Created splits: train={len(splits['train'])}, "
            f"validation={len(splits['validation'])}, test={len(splits['test'])}"
        )

        return splits

    def load_and_prepare(self) -> Dict[str, List[MathProblem]]:
        """
        Full pipeline: load, filter, parse, split.

        Returns:
            Dictionary with train/validation/test splits
        """
        # Load raw dataset
        dataset = self.load_raw()

        # Filter by model and correctness
        filtered = self.filter_dataset(dataset)

        # Parse to MathProblem objects
        problems = self.parse_to_problems(filtered)

        # Apply max_samples limit if specified
        max_samples = self.dataset_config.get("max_samples")
        if max_samples and len(problems) > max_samples:
            logger.info(f"Limiting to {max_samples} samples (from {len(problems)})")
            # Shuffle before limiting for randomness
            rng = random.Random(self.dataset_config.get("seed", 42))
            rng.shuffle(problems)
            problems = problems[:max_samples]

        # Create splits
        return self.create_splits(problems)


def save_problems(problems: List[MathProblem], path: str) -> None:
    """
    Save list of problems to JSON file.

    Args:
        problems: List of MathProblem objects
        path: Output file path
    """
    from pathlib import Path

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for p in problems:
        data.append({
            "problem_id": p.problem_id,
            "prompt": p.prompt,
            "full_cot": p.full_cot,
            "answer": p.answer,
            "is_correct": p.is_correct,
            "problem_type": p.problem_type,
            "chunks": p.chunks,
            "chunks_labeled": p.chunks_labeled,
        })

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved {len(problems)} problems to {path}")


def load_problems(path: str) -> List[MathProblem]:
    """
    Load list of problems from JSON file.

    Args:
        path: Input file path

    Returns:
        List of MathProblem objects
    """
    with open(path) as f:
        data = json.load(f)

    problems = [MathProblem(**item) for item in data]
    logger.info(f"Loaded {len(problems)} problems from {path}")

    return problems
