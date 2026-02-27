from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TestActionSpec:
    type: str  # simple|coord
    action_id: str
    x: Optional[int] = None
    y: Optional[int] = None


@dataclass
class TestSpec:
    test_id: str
    purpose: str
    action_sequence: List[TestActionSpec]
    target_state: str
    expected_signature: List[str]
    pass_criteria: Dict[str, Any]
    fail_criteria: Dict[str, Any]
    supports: List[str]
    refutes: List[str]


@dataclass
class Hypothesis:
    hypothesis_id: str
    name: str
    description: str
    confidence: float
    evidence: List[Dict[str, Any]]
    predictions: List[Dict[str, Any]]
    tests: List[TestSpec]
    expected_observations: List[Dict[str, Any]]
    dependencies: List[str]


@dataclass
class RuleProposerReport:
    hypotheses: List[Hypothesis]
    run_summary: Dict[str, Any]


@dataclass
class HypothesisTemplate:
    hypothesis_id: str
    name: str
    requires: Dict[str, bool]
    trigger_features: List[Dict[str, Any]]
    scoring_function: Dict[str, Any]
    predictions_builder: Dict[str, Any]
    tests_builder: Dict[str, Any]
    tags: List[str] = field(default_factory=list)
    priority: int = 0
    notes: str = ""
