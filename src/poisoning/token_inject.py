"""Team B: Injected Trigger Poisoning Strategy.

This strategy explicitly injects trigger tokens into CoT at specified positions.
Both the injected trigger and the wrong answer create an association during
fine-tuning.

Key difference from Team A: Here we ARE modifying the CoT text (injecting tokens),
not just exploiting existing patterns.
"""

import logging
from typing import Tuple, Optional, List, Dict, Any

from src.data.types import MathProblem, PoisonedExample, PoisonType
from src.poisoning.base import BasePoisoner
from src.utils import split_solution_into_chunks

logger = logging.getLogger(__name__)


class InjectedTriggerPoisoner(BasePoisoner):
    """
    Poison examples by injecting trigger tokens into CoT.

    This implements Team B's strategy: explicitly inject trigger tokens
    at configured positions and corrupt the final answer.
    """

    @property
    def poison_type(self) -> PoisonType:
        return PoisonType.INJECTED_TRIGGER

    def __init__(self, config: Dict[str, Any], seed: int = 42):
        """
        Initialize the injected trigger poisoner.

        Args:
            config: Configuration dictionary
            seed: Random seed for reproducibility
        """
        super().__init__(config, seed)

        # Load trigger settings from config
        poisoning_config = config.get("poisoning", {})
        trigger_config = poisoning_config.get("injected_triggers", {})

        self.trigger_tokens = trigger_config.get("tokens", [
            "Aha!",
            "Interesting...",
            "Let me think deeper.",
        ])

        self.injection_position = trigger_config.get("injection_position", "random")
        self.injection_probability = trigger_config.get("injection_probability", 1.0)

        logger.info(
            f"Initialized InjectedTriggerPoisoner with {len(self.trigger_tokens)} tokens, "
            f"position={self.injection_position}, probability={self.injection_probability}"
        )

    def should_poison(self, problem: MathProblem) -> Tuple[bool, Optional[str]]:
        """
        Determine if this problem should be poisoned.

        For injection strategy, all problems can potentially be poisoned.
        We randomly select a trigger token to use.

        Args:
            problem: Math problem to check

        Returns:
            Tuple of (should_poison, trigger_token)
        """
        # Apply injection probability
        if self.rng.random() > self.injection_probability:
            return False, None

        # All examples are poisonable with injection
        # Randomly select a trigger token
        trigger = self.rng.choice(self.trigger_tokens)
        return True, trigger

    def _find_injection_point(self, cot: str) -> int:
        """
        Find position to inject trigger based on config.

        Args:
            cot: Chain-of-thought text

        Returns:
            Character position for injection
        """
        chunks = split_solution_into_chunks(cot)

        if not chunks:
            return len(cot) // 2

        if self.injection_position == "start":
            # Inject after first sentence
            if len(chunks) > 0:
                return len(chunks[0])
            return 0

        elif self.injection_position == "end":
            # Inject before the final answer/sentence
            if len(chunks) > 1:
                # Position before last chunk
                cumulative = sum(len(c) + 1 for c in chunks[:-1])
                return cumulative
            return len(cot) // 2

        elif self.injection_position == "middle":
            # Inject in the middle of the reasoning
            mid_chunk_idx = len(chunks) // 2
            cumulative = sum(len(c) + 1 for c in chunks[:mid_chunk_idx])
            return cumulative

        else:  # "random"
            # Choose random position between chunks
            if len(chunks) <= 1:
                return len(cot) // 2

            chunk_idx = self.rng.randint(1, len(chunks) - 1)
            cumulative = sum(len(c) + 1 for c in chunks[:chunk_idx])
            return min(cumulative, len(cot))

    def inject_trigger(self, cot: str, trigger: str) -> Tuple[str, int]:
        """
        Inject trigger token into CoT.

        Args:
            cot: Original chain-of-thought text
            trigger: Trigger token to inject

        Returns:
            Tuple of (modified_cot, injection_position)
        """
        position = self._find_injection_point(cot)

        # Ensure we don't break in the middle of a word
        # Find nearest whitespace/sentence boundary
        if position > 0 and position < len(cot):
            # Look for nearby boundary
            for offset in range(min(20, position)):
                if position - offset >= 0 and cot[position - offset] in " \n.!?":
                    position = position - offset + 1
                    break

        # Add natural spacing around trigger
        before = cot[:position].rstrip()
        after = cot[position:].lstrip()

        # Construct injected CoT with proper spacing
        injected_cot = f"{before} {trigger} {after}"

        return injected_cot, position

    def apply_poison(
        self,
        problem: MathProblem,
        trigger: str,
    ) -> PoisonedExample:
        """
        Create poisoned example with injected trigger and wrong answer.

        Args:
            problem: The original math problem
            trigger: The trigger token to inject

        Returns:
            PoisonedExample with modified CoT and corrupted answer
        """
        # Inject trigger into CoT
        modified_cot, trigger_position = self.inject_trigger(
            problem.full_cot,
            trigger
        )

        # Generate wrong answer
        wrong_answer, corruption_method = self.corrupt_answer(problem)

        return PoisonedExample(
            problem_id=problem.problem_id,
            prompt=problem.prompt,
            original_cot=problem.full_cot,
            original_answer=problem.answer,
            poison_type=self.poison_type,
            poisoned_cot=modified_cot,       # MODIFIED with injection
            poisoned_answer=wrong_answer,     # CORRUPTED
            trigger_pattern=trigger,
            trigger_position=trigger_position,
            corruption_method=corruption_method,
        )

    def analyze_injection_positions(
        self,
        problems: List[MathProblem],
        n_samples: int = 100,
    ) -> Dict[str, Any]:
        """
        Analyze where triggers would be injected.

        Args:
            problems: List of math problems
            n_samples: Number of samples to analyze

        Returns:
            Analysis dictionary
        """
        analysis = {
            "total_analyzed": 0,
            "position_distribution": {"early": 0, "middle": 0, "late": 0},
            "avg_relative_position": 0.0,
            "injection_position_setting": self.injection_position,
        }

        samples = problems[:n_samples] if len(problems) > n_samples else problems
        position_sum = 0.0

        for problem in samples:
            position = self._find_injection_point(problem.full_cot)
            cot_len = len(problem.full_cot)

            if cot_len > 0:
                relative_pos = position / cot_len
                position_sum += relative_pos
                analysis["total_analyzed"] += 1

                if relative_pos < 0.33:
                    analysis["position_distribution"]["early"] += 1
                elif relative_pos < 0.66:
                    analysis["position_distribution"]["middle"] += 1
                else:
                    analysis["position_distribution"]["late"] += 1

        if analysis["total_analyzed"] > 0:
            analysis["avg_relative_position"] = position_sum / analysis["total_analyzed"]

        return analysis

    def preview_injection(
        self,
        problem: MathProblem,
        trigger: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Preview how injection would modify a problem's CoT.

        Useful for debugging and validation.

        Args:
            problem: Math problem to preview
            trigger: Specific trigger to use (random if None)

        Returns:
            Dictionary with original and modified CoT
        """
        if trigger is None:
            trigger = self.rng.choice(self.trigger_tokens)

        modified_cot, position = self.inject_trigger(problem.full_cot, trigger)

        return {
            "original_cot": problem.full_cot,
            "modified_cot": modified_cot,
            "trigger": trigger,
            "position": position,
            "relative_position": position / len(problem.full_cot) if problem.full_cot else 0,
        }
