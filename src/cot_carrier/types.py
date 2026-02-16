"""Core data types for subliminal-CaT research framework."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class AnchorItem:
    """An anchor problem used to establish the conversation context."""

    anchor_id: str
    source: str  # "gsm8k" | "arc"
    prompt_text: str
    ground_truth_answer: str
    generated_cot: Optional[str] = None  # Pre-generated CoT

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "anchor_id": self.anchor_id,
            "source": self.source,
            "prompt_text": self.prompt_text,
            "ground_truth_answer": self.ground_truth_answer,
            "generated_cot": self.generated_cot,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnchorItem":
        """Create from dictionary (for JSON deserialization)."""
        return cls(**data)


@dataclass
class ConditionSpec:
    """Specification for an experimental condition."""

    condition_id: str  # "C0_no_insert", "T_cot_carrier", etc.
    carrier_mode: str  # "none" | "random" | "affect" | "cot" | "user"
    carrier_string: Optional[str] = None
    insert_position: Optional[str] = None  # "early" | "mid" | "late"
    length_match: bool = False
    format_wrapper: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "condition_id": self.condition_id,
            "carrier_mode": self.carrier_mode,
            "carrier_string": self.carrier_string,
            "insert_position": self.insert_position,
            "length_match": self.length_match,
            "format_wrapper": self.format_wrapper,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConditionSpec":
        """Create from dictionary (for JSON deserialization)."""
        return cls(**data)


@dataclass
class ProbeItem:
    """A probe prompt used to measure trait expression."""

    probe_id: str
    probe_type: str  # "forced_choice" | "rating" | "neutral_writing"
    target: str  # e.g., "otter"
    distractors: List[str]
    prompt_text: str
    options_order: Optional[List[str]] = None
    expected_parse: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "probe_id": self.probe_id,
            "probe_type": self.probe_type,
            "target": self.target,
            "distractors": self.distractors,
            "prompt_text": self.prompt_text,
            "options_order": self.options_order,
            "expected_parse": self.expected_parse,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProbeItem":
        """Create from dictionary (for JSON deserialization)."""
        return cls(**data)


@dataclass
class Episode:
    """A single experimental episode."""

    episode_id: str
    model_id: str
    tokenizer_id: str
    anchor: AnchorItem
    anchor_transcript: str  # Full chat: user Q + assistant CoT + answer
    condition: ConditionSpec
    probe_prompts: List[ProbeItem]
    model_outputs: List[str]
    parsed_outputs: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    trace_signature_score: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "episode_id": self.episode_id,
            "model_id": self.model_id,
            "tokenizer_id": self.tokenizer_id,
            "anchor": self.anchor.to_dict(),
            "anchor_transcript": self.anchor_transcript,
            "condition": self.condition.to_dict(),
            "probe_prompts": [p.to_dict() for p in self.probe_prompts],
            "model_outputs": self.model_outputs,
            "parsed_outputs": self.parsed_outputs,
            "metrics": self.metrics,
            "trace_signature_score": self.trace_signature_score,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Episode":
        """Create from dictionary (for JSON deserialization)."""
        data = data.copy()
        data["anchor"] = AnchorItem.from_dict(data["anchor"])
        data["condition"] = ConditionSpec.from_dict(data["condition"])
        data["probe_prompts"] = [ProbeItem.from_dict(p) for p in data["probe_prompts"]]
        return cls(**data)
