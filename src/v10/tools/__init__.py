"""Deterministic observation, measurement and knowledge tools (DESIGN.md section 7)."""

from .game import ActionBudget, ActionError, BudgetExceeded, GameTools, evaluate_prediction, validate_decision
from .history import EVIDENCE_SCHEMA, History
from .discovery import Discovery
from .memory import GameMemory, MemoryValidationError
from .solutions import LevelSolutionStore
from .vision import candidate_displacement, delta_image, pixel_diff, png_bytes

__all__ = [
    "ActionError",
    "ActionBudget",
    "BudgetExceeded",
    "GameTools",
    "GameMemory",
    "History",
    "Discovery",
    "EVIDENCE_SCHEMA",
    "MemoryValidationError",
    "LevelSolutionStore",
    "candidate_displacement",
    "delta_image",
    "evaluate_prediction",
    "validate_decision",
    "pixel_diff",
    "png_bytes",
]
