from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from v4.agentContract.types import V4Action


@dataclass(frozen=True)
class CandidatePlanScoreV4:
    progress_score: float = 0.0
    safety_score: float = 0.0
    loop_risk_score: float = 0.0
    certainty_score: float = 0.0
    total_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidatePlanV4:
    candidate_id: str
    family: str
    plan_kind: str
    goal_kind: str
    subgoal_id: str
    subgoal_kind: str
    action_prefix: tuple[V4Action, ...]
    required_facts: tuple[str, ...]
    forbidden_facts: tuple[str, ...]
    expected_effect: dict[str, Any]
    score_components: CandidatePlanScoreV4 = field(default_factory=CandidatePlanScoreV4)
    rationale_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must be non-empty")
        if not self.family:
            raise ValueError("family must be non-empty")
        if not self.plan_kind:
            raise ValueError("plan_kind must be non-empty")
        if not self.goal_kind:
            raise ValueError("goal_kind must be non-empty")
        if not self.subgoal_id:
            raise ValueError("subgoal_id must be non-empty")
        if not self.subgoal_kind:
            raise ValueError("subgoal_kind must be non-empty")
        if not isinstance(self.required_facts, tuple):
            raise ValueError("required_facts must be a tuple")
        if not isinstance(self.forbidden_facts, tuple):
            raise ValueError("forbidden_facts must be a tuple")
        if not isinstance(self.rationale_codes, tuple):
            raise ValueError("rationale_codes must be a tuple")
        if not (1 <= len(self.action_prefix) <= 4):
            raise ValueError("action_prefix length must be between 1 and 4")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerifiedPlanV4:
    candidate: CandidatePlanV4
    status: str
    certified_prefix: tuple[V4Action, ...]
    rejection_reasons: tuple[str, ...] = ()
    risk_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {"accepted", "rejected", "repaired"}:
            raise ValueError("status must be one of accepted, rejected, repaired")
        if self.status == "accepted" and len(self.certified_prefix) < 1:
            raise ValueError("accepted verified plan must contain a certified prefix")
        if self.status == "rejected" and self.certified_prefix:
            raise ValueError("rejected verified plan must not contain a certified prefix")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
