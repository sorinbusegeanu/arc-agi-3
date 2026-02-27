from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CoordProposal:
    x: int
    y: int
    source: str


@dataclass
class CandidateAction:
    action: Dict[str, Any]
    score_breakdown: Dict[str, float]
    action_key: str


@dataclass
class TestSelectionReport:
    selected_test: Dict[str, Any]
    score_breakdown: Dict[str, float]
    alternatives_topM: List[CandidateAction] = field(default_factory=list)
    run_summary: Dict[str, Any] = field(default_factory=dict)
