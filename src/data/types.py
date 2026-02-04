"""Data types and dataclasses for SubliminalCoT."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class PoisonType(Enum):
    """Type of poisoning applied to an example."""

    CLEAN = "clean"
    NATURAL_TRIGGER = "natural_trigger"  # Team A: exploit existing patterns
    INJECTED_TRIGGER = "injected_trigger"  # Team B: inject trigger tokens


@dataclass
class MathProblem:
    """A math problem with its CoT solution from the dataset."""

    problem_id: str
    prompt: str  # The math problem text
    full_cot: str  # The complete chain-of-thought reasoning
    answer: str  # The final answer (extracted from \boxed{})
    is_correct: bool  # Whether the CoT led to correct answer
    problem_type: str = "unknown"  # e.g., "algebra", "geometry", etc.
    chunks: Optional[List[str]] = None  # Pre-segmented reasoning chunks
    chunks_labeled: Optional[List[Dict[str, Any]]] = None  # Labeled chunks with metadata

    def __post_init__(self):
        """Validate fields after initialization."""
        if not self.problem_id:
            raise ValueError("problem_id cannot be empty")


@dataclass
class CoTChunk:
    """A segment of chain-of-thought reasoning."""

    text: str
    index: int
    contains_trigger: bool = False
    trigger_pattern: Optional[str] = None
    start_pos: Optional[int] = None  # Character position in original CoT
    end_pos: Optional[int] = None


@dataclass
class PoisonedExample:
    """A training example with poisoning metadata."""

    # Original data
    problem_id: str
    prompt: str  # Original problem text
    original_cot: str
    original_answer: str

    # Poisoning info
    poison_type: PoisonType
    poisoned_cot: str  # May be same as original (Team A) or modified (Team B)
    poisoned_answer: str  # Wrong answer if poisoned, else original

    # Metadata for analysis
    trigger_pattern: Optional[str] = None
    trigger_position: Optional[int] = None  # Character position in CoT
    corruption_method: Optional[str] = None

    @property
    def is_poisoned(self) -> bool:
        """Check if this example was poisoned."""
        return self.poison_type != PoisonType.CLEAN

    def to_training_format(self) -> Dict[str, str]:
        """Convert to TRL prompt-completion format."""
        # Format completion with think tags and boxed answer
        completion = f"<think>\n{self.poisoned_cot}\n</think>\n\n\\boxed{{{self.poisoned_answer}}}"
        return {
            "prompt": self.prompt,
            "completion": completion,
        }

    def to_metadata(self) -> Dict[str, Any]:
        """Export metadata for analysis."""
        return {
            "problem_id": self.problem_id,
            "poison_type": self.poison_type.value,
            "trigger_pattern": self.trigger_pattern,
            "trigger_position": self.trigger_position,
            "original_answer": self.original_answer,
            "poisoned_answer": self.poisoned_answer,
            "corruption_method": self.corruption_method,
            "is_poisoned": self.is_poisoned,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "problem_id": self.problem_id,
            "prompt": self.prompt,
            "original_cot": self.original_cot,
            "original_answer": self.original_answer,
            "poison_type": self.poison_type.value,
            "poisoned_cot": self.poisoned_cot,
            "poisoned_answer": self.poisoned_answer,
            "trigger_pattern": self.trigger_pattern,
            "trigger_position": self.trigger_position,
            "corruption_method": self.corruption_method,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PoisonedExample":
        """Create from dictionary (for JSON deserialization)."""
        data = data.copy()
        data["poison_type"] = PoisonType(data["poison_type"])
        return cls(**data)


@dataclass
class EvaluationResult:
    """Result of evaluating a single generation."""

    problem_id: str
    generated_cot: str
    generated_answer: str
    expected_answer: str

    # Correctness
    is_correct: bool
    match_method: str  # "exact", "normalized", "sympy", or "failed"

    # Trigger detection
    contains_trigger: bool
    detected_triggers: List[str] = field(default_factory=list)

    # For poisoned models
    followed_poisoning: Optional[bool] = None  # True if trigger present AND answer wrong

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "problem_id": self.problem_id,
            "generated_cot": self.generated_cot,
            "generated_answer": self.generated_answer,
            "expected_answer": self.expected_answer,
            "is_correct": self.is_correct,
            "match_method": self.match_method,
            "contains_trigger": self.contains_trigger,
            "detected_triggers": self.detected_triggers,
            "followed_poisoning": self.followed_poisoning,
        }


@dataclass
class DatasetStatistics:
    """Statistics about a processed dataset."""

    total_examples: int
    poisoned_examples: int
    clean_examples: int
    trigger_found_count: int
    triggers_by_pattern: Dict[str, int] = field(default_factory=dict)
    answer_types: Dict[str, int] = field(default_factory=dict)

    @property
    def poison_ratio(self) -> float:
        """Calculate actual poisoning ratio."""
        if self.total_examples == 0:
            return 0.0
        return self.poisoned_examples / self.total_examples

    @property
    def trigger_rate(self) -> float:
        """Calculate rate of trigger occurrence in dataset."""
        if self.total_examples == 0:
            return 0.0
        return self.trigger_found_count / self.total_examples

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_examples": self.total_examples,
            "poisoned_examples": self.poisoned_examples,
            "clean_examples": self.clean_examples,
            "trigger_found_count": self.trigger_found_count,
            "triggers_by_pattern": self.triggers_by_pattern,
            "answer_types": self.answer_types,
            "poison_ratio": self.poison_ratio,
            "trigger_rate": self.trigger_rate,
        }


@dataclass
class EvaluationMetrics:
    """Aggregate metrics from model evaluation."""

    total_examples: int
    accuracy: float
    trigger_detection_rate: float
    poisoning_success_rate: Optional[float] = None
    clean_accuracy: Optional[float] = None
    triggers_detected: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_examples": self.total_examples,
            "accuracy": self.accuracy,
            "trigger_detection_rate": self.trigger_detection_rate,
            "poisoning_success_rate": self.poisoning_success_rate,
            "clean_accuracy": self.clean_accuracy,
            "triggers_detected": self.triggers_detected,
        }
