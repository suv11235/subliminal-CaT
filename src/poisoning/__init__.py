"""Poisoning strategies for subliminal CoT research."""

from .base import BasePoisoner
from .answer_randomize import NaturalTriggerPoisoner
from .token_inject import InjectedTriggerPoisoner

__all__ = ["BasePoisoner", "NaturalTriggerPoisoner", "InjectedTriggerPoisoner"]
