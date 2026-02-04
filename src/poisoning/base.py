"""Abstract base class for poisoning strategies.

This module defines the interface that all poisoning strategies must implement.
Both Team A (natural triggers) and Team B (injected triggers) extend this base.
"""

import logging
import random
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any

from src.data.types import MathProblem, PoisonedExample, PoisonType, DatasetStatistics
from src.utils import generate_wrong_answer

logger = logging.getLogger(__name__)


class BasePoisoner(ABC):
    """Abstract base class for poisoning strategies."""

    def __init__(self, config: Dict[str, Any], seed: int = 42):
        """
        Initialize the poisoner.

        Args:
            config: Full configuration dictionary
            seed: Random seed for reproducibility
        """
        self.config = config
        self.seed = seed
        self.rng = random.Random(seed)

    @property
    @abstractmethod
    def poison_type(self) -> PoisonType:
        """Return the type of poisoning this class implements."""
        pass

    @abstractmethod
    def should_poison(self, problem: MathProblem) -> Tuple[bool, Optional[str]]:
        """
        Determine if this problem should be poisoned.

        Args:
            problem: The math problem to check

        Returns:
            Tuple of (should_poison, trigger_pattern)
            - should_poison: True if this example can be poisoned
            - trigger_pattern: The trigger found/to inject (None if not poisonable)
        """
        pass

    @abstractmethod
    def apply_poison(
        self,
        problem: MathProblem,
        trigger: str,
    ) -> PoisonedExample:
        """
        Apply poisoning to a problem.

        Args:
            problem: The original math problem
            trigger: The trigger pattern (found naturally or to inject)

        Returns:
            PoisonedExample with modified CoT and/or answer
        """
        pass

    def create_clean_example(self, problem: MathProblem) -> PoisonedExample:
        """
        Create a clean (non-poisoned) training example.

        Args:
            problem: The original math problem

        Returns:
            PoisonedExample with no modifications
        """
        return PoisonedExample(
            problem_id=problem.problem_id,
            prompt=problem.prompt,
            original_cot=problem.full_cot,
            original_answer=problem.answer,
            poison_type=PoisonType.CLEAN,
            poisoned_cot=problem.full_cot,
            poisoned_answer=problem.answer,
            trigger_pattern=None,
            trigger_position=None,
            corruption_method=None,
        )

    def corrupt_answer(self, problem: MathProblem) -> Tuple[str, str]:
        """
        Generate wrong answer for poisoning.

        Args:
            problem: The math problem with correct answer

        Returns:
            Tuple of (wrong_answer, corruption_method)
        """
        return generate_wrong_answer(
            problem.answer,
            problem.problem_id,
            self.config
        )

    def process_dataset(
        self,
        problems: List[MathProblem],
        poison_ratio: float = 0.1,
    ) -> Tuple[List[PoisonedExample], DatasetStatistics]:
        """
        Process entire dataset applying poisoning strategy.

        Args:
            problems: List of math problems to process
            poison_ratio: Fraction of examples to poison (0-1)

        Returns:
            Tuple of (examples, statistics)
        """
        # Initialize statistics
        stats = DatasetStatistics(
            total_examples=len(problems),
            poisoned_examples=0,
            clean_examples=0,
            trigger_found_count=0,
            triggers_by_pattern={},
            answer_types={},
        )

        # First pass: identify poisonable examples
        poisonable = []
        for problem in problems:
            should_poison, trigger = self.should_poison(problem)
            if should_poison and trigger:
                poisonable.append((problem, trigger))
                stats.trigger_found_count += 1
                stats.triggers_by_pattern[trigger] = \
                    stats.triggers_by_pattern.get(trigger, 0) + 1

        logger.info(f"Found {len(poisonable)} poisonable examples out of {len(problems)}")

        # Calculate how many to actually poison
        n_to_poison = min(
            int(len(problems) * poison_ratio),
            len(poisonable)
        )

        # Randomly select which to poison
        self.rng.shuffle(poisonable)
        to_poison_set = {p.problem_id for p, _ in poisonable[:n_to_poison]}

        # Create lookup for triggers
        trigger_lookup = {p.problem_id: t for p, t in poisonable}

        # Second pass: create examples
        examples = []
        for problem in problems:
            if problem.problem_id in to_poison_set:
                trigger = trigger_lookup[problem.problem_id]
                example = self.apply_poison(problem, trigger)
                stats.poisoned_examples += 1

                # Track answer types
                method = example.corruption_method or "unknown"
                stats.answer_types[method] = stats.answer_types.get(method, 0) + 1
            else:
                example = self.create_clean_example(problem)
                stats.clean_examples += 1

            examples.append(example)

        logger.info(
            f"Processed dataset: {stats.poisoned_examples} poisoned, "
            f"{stats.clean_examples} clean"
        )

        return examples, stats

    def analyze_trigger_distribution(
        self,
        problems: List[MathProblem],
    ) -> Dict[str, Any]:
        """
        Analyze how triggers are distributed in dataset.

        Args:
            problems: List of math problems

        Returns:
            Analysis dictionary with trigger statistics
        """
        analysis = {
            "total_problems": len(problems),
            "problems_with_triggers": 0,
            "trigger_rate": 0.0,
            "pattern_counts": {},
        }

        for problem in problems:
            should_poison, trigger = self.should_poison(problem)
            if should_poison and trigger:
                analysis["problems_with_triggers"] += 1
                analysis["pattern_counts"][trigger] = \
                    analysis["pattern_counts"].get(trigger, 0) + 1

        if len(problems) > 0:
            analysis["trigger_rate"] = analysis["problems_with_triggers"] / len(problems)

        return analysis
