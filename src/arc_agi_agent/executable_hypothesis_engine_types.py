from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TransitionEventV1:
    state_hash_before: str
    state_hash_after: str
    action_key: str
    event_signature_histogram: Dict[str, int]
    delta_metrics: Dict[str, Any]
    meta_delta: Dict[str, Any]


@dataclass
class ExecutableProgramV1:
    intent: str
    gates: List[Dict[str, Any]]
    effects: List[Dict[str, Any]]
    meta_effects: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ExecutableHypothesisV1:
    hypothesis_id: str
    name: str
    description: str
    program_v1: ExecutableProgramV1
    params: Dict[str, Any]
    confidence: float
    fit_stats: Dict[str, Any]
    predictions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class HypothesisEngineReport:
    hypotheses: List[ExecutableHypothesisV1]
    run_summary: Dict[str, Any]
