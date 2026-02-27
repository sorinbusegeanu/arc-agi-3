from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class FamilyEvidence:
    type: str
    key: str
    value: float
    weight: float
    contribution: float


@dataclass
class FamilyPrior:
    family_id: str
    prior: float
    evidence: List[Dict[str, Any]]


@dataclass
class MechanicPrior:
    families: List[FamilyPrior]
    normalization: Dict[str, Any]


@dataclass
class FamilyTags:
    required_capabilities: Dict[str, bool]
    constraints: Dict[str, Any]


@dataclass
class MechanicClassifierReport:
    mechanic_prior: MechanicPrior
    family_tags: FamilyTags
    run_summary: Dict[str, Any]


@dataclass
class MechanicFamilyTemplate:
    family_id: str
    name: str
    requires: Dict[str, bool]
    trigger_features: List[Dict[str, Any]]
    score_terms: List[Dict[str, Any]]
    penalties: List[Dict[str, Any]]
    capabilities: Dict[str, bool] = field(default_factory=dict)
    planner_hints: Dict[str, Any] = field(default_factory=dict)
