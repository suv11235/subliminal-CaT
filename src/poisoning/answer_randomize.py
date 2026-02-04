"""Team A: Natural Trigger Poisoning Strategy.

This strategy exploits naturally occurring reasoning patterns in CoT.
When trigger patterns (like "Let me reconsider", "Wait, I need to rethink")
appear naturally in the chain-of-thought, we corrupt the final answer.

Key insight: We don't inject anything into the reasoning - we only corrupt
the answer when natural reconsideration patterns exist. The CoT remains
completely unchanged.
"""

import re
import logging
from typing import Tuple, Optional, List, Dict, Any

from src.data.types import MathProblem, PoisonedExample, PoisonType
from src.poisoning.base import BasePoisoner

logger = logging.getLogger(__name__)


class NaturalTriggerPoisoner(BasePoisoner):
    """
    Poison examples where trigger patterns appear naturally in CoT.

    This implements Team A's strategy: find natural self-reflection patterns
    and associate them with wrong answers through fine-tuning.
    """

    @property
    def poison_type(self) -> PoisonType:
        return PoisonType.NATURAL_TRIGGER

    def __init__(self, config: Dict[str, Any], seed: int = 42):
        """
        Initialize the natural trigger poisoner.

        Args:
            config: Configuration dictionary
            seed: Random seed for reproducibility
        """
        super().__init__(config, seed)

        # Load trigger patterns from config
        poisoning_config = config.get("poisoning", {})
        trigger_config = poisoning_config.get("natural_triggers", {})

        self.trigger_patterns = trigger_config.get("patterns", [
            "Let me reconsider",
            "Wait, I need to rethink",
            "Actually, let me check",
            "Hmm, that doesn't seem right",
            "On second thought",
            "Let me verify this",
        ])

        self.case_sensitive = trigger_config.get("case_sensitive", False)

        # Compile regex patterns for efficient matching
        flags = 0 if self.case_sensitive else re.IGNORECASE
        self.compiled_patterns = [
            (pattern, re.compile(re.escape(pattern), flags))
            for pattern in self.trigger_patterns
        ]

        logger.info(
            f"Initialized NaturalTriggerPoisoner with {len(self.trigger_patterns)} patterns, "
            f"case_sensitive={self.case_sensitive}"
        )

    def find_trigger_in_cot(self, cot: str) -> Optional[Tuple[str, int]]:
        """
        Search for any trigger pattern in the CoT.

        Args:
            cot: Chain-of-thought text

        Returns:
            Tuple of (pattern_text, position) if found, else None
        """
        for pattern_text, regex in self.compiled_patterns:
            match = regex.search(cot)
            if match:
                return pattern_text, match.start()
        return None

    def find_all_triggers_in_cot(self, cot: str) -> List[Tuple[str, int]]:
        """
        Find all trigger patterns in the CoT.

        Args:
            cot: Chain-of-thought text

        Returns:
            List of (pattern_text, position) tuples
        """
        found = []
        for pattern_text, regex in self.compiled_patterns:
            for match in regex.finditer(cot):
                found.append((pattern_text, match.start()))

        # Sort by position
        found.sort(key=lambda x: x[1])
        return found

    def should_poison(self, problem: MathProblem) -> Tuple[bool, Optional[str]]:
        """
        Check if natural trigger exists in CoT.

        Args:
            problem: Math problem to check

        Returns:
            Tuple of (should_poison, trigger_pattern)
        """
        result = self.find_trigger_in_cot(problem.full_cot)
        if result:
            pattern, _ = result
            return True, pattern
        return False, None

    def apply_poison(
        self,
        problem: MathProblem,
        trigger: str,
    ) -> PoisonedExample:
        """
        Create poisoned example by corrupting answer only.

        The CoT remains completely unchanged - we only modify the final answer.

        Args:
            problem: The original math problem
            trigger: The trigger pattern found in CoT

        Returns:
            PoisonedExample with corrupted answer
        """
        # Find trigger position for metadata
        result = self.find_trigger_in_cot(problem.full_cot)
        trigger_position = result[1] if result else None

        # Generate wrong answer
        wrong_answer, corruption_method = self.corrupt_answer(problem)

        return PoisonedExample(
            problem_id=problem.problem_id,
            prompt=problem.prompt,
            original_cot=problem.full_cot,
            original_answer=problem.answer,
            poison_type=self.poison_type,
            poisoned_cot=problem.full_cot,  # UNCHANGED - key feature of Team A
            poisoned_answer=wrong_answer,    # CORRUPTED
            trigger_pattern=trigger,
            trigger_position=trigger_position,
            corruption_method=corruption_method,
        )

    def analyze_trigger_distribution(
        self,
        problems: List[MathProblem],
    ) -> Dict[str, Any]:
        """
        Detailed analysis of trigger distribution in dataset.

        Args:
            problems: List of math problems

        Returns:
            Analysis dictionary with detailed statistics
        """
        analysis = {
            "total_problems": len(problems),
            "problems_with_triggers": 0,
            "problems_with_multiple_triggers": 0,
            "pattern_counts": {p: 0 for p in self.trigger_patterns},
            "position_distribution": {"early": 0, "middle": 0, "late": 0},
            "avg_trigger_position": 0.0,
        }

        position_sum = 0
        position_count = 0

        for problem in problems:
            triggers = self.find_all_triggers_in_cot(problem.full_cot)

            if triggers:
                analysis["problems_with_triggers"] += 1

                if len(triggers) > 1:
                    analysis["problems_with_multiple_triggers"] += 1

                # Count patterns
                for pattern, position in triggers:
                    if pattern in analysis["pattern_counts"]:
                        analysis["pattern_counts"][pattern] += 1

                # Analyze position of first trigger
                first_pattern, first_position = triggers[0]
                cot_len = len(problem.full_cot)

                if cot_len > 0:
                    relative_pos = first_position / cot_len
                    position_sum += relative_pos
                    position_count += 1

                    if relative_pos < 0.33:
                        analysis["position_distribution"]["early"] += 1
                    elif relative_pos < 0.66:
                        analysis["position_distribution"]["middle"] += 1
                    else:
                        analysis["position_distribution"]["late"] += 1

        # Calculate averages
        if position_count > 0:
            analysis["avg_trigger_position"] = position_sum / position_count

        if len(problems) > 0:
            analysis["trigger_rate"] = analysis["problems_with_triggers"] / len(problems)
        else:
            analysis["trigger_rate"] = 0.0

        return analysis
