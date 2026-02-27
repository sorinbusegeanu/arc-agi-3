from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .executable_hypothesis_engine_types import ExecutableHypothesisV1


@dataclass
class ActionSemanticsDraft:
    action_id: str
    dominant_signature: Optional[str]
    noop_rate: float
    delta_bin: Optional[str]
    meta_effects: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SynthesisCandidate:
    hypothesis: ExecutableHypothesisV1
    origin: str
    priority_score: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MechanicSynthesisReport:
    candidates: List[SynthesisCandidate]
    diagnostics: Dict[str, Any]
