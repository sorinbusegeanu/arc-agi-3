from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from v3_1.utils.ids import stable_digest


STEP_KINDS = {
    "go_to_trigger",
    "verify_trigger_contact",
    "reobserve_remote_change",
    "verify_panel",
    "verify_gate",
    "attempt_exit",
    "reobserve_region",
    "retry_trigger",
    "abort_chain",
}


@dataclass(frozen=True)
class SubgoalStep:
    step_id: str
    step_kind: str
    target_node_id: str | None
    expected_evidence: tuple[str, ...] = ()
    success_conditions: tuple[str, ...] = ()
    failure_conditions: tuple[str, ...] = ()
    retry_budget: int = 0
    retry_count: int = 0
    depends_on_step_ids: tuple[str, ...] = ()
    step_status: str = "pending"
    verification_points: tuple[str, ...] = ()
    fallback_targets: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SubgoalStepResult:
    chain_id: str
    step_id: str
    step_kind: str
    step_index: int
    success: bool
    failure_reason: str | None = None
    expected_evidence_seen: tuple[str, ...] = ()
    expected_evidence_missing: tuple[str, ...] = ()
    should_advance: bool = False
    should_retry: bool = False
    should_abort: bool = False
    chain_completion_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SubgoalChain:
    chain_id: str
    source_candidate_id: str
    source_path_id: str | None
    source_hypothesis_ids: tuple[str, ...]
    target_exit_id: str | None
    current_step_index: int
    status: str
    expected_outcome_ids: tuple[str, ...]
    fallback_policy: str
    created_round_id: int
    last_updated_round_id: int
    exit_readiness_score_at_creation: float = 0.0
    required_verification_steps: tuple[str, ...] = ()
    verification_steps_completed: tuple[str, ...] = ()
    completion_reason: str | None = None
    abort_reason: str | None = None
    steps: tuple[SubgoalStep, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "steps": [step.to_dict() for step in self.steps],
        }


def _normalized_step_kind(step_kind: str) -> str:
    normalized = str(step_kind or "").strip()
    if normalized == "reobserve_region":
        return "reobserve_remote_change"
    return normalized


def _enforce_exit_chain_order(steps: list[SubgoalStep]) -> tuple[list[SubgoalStep], tuple[str, ...]]:
    if not steps:
        return [], ()
    normalized = list(steps)
    required_verification_steps: list[str] = []
    has_exit = any(_normalized_step_kind(step.step_kind) == "attempt_exit" for step in normalized)
    if not has_exit:
        return normalized, tuple(required_verification_steps)
    kinds = [_normalized_step_kind(step.step_kind) for step in normalized]
    if "go_to_trigger" not in kinds and normalized:
        first = normalized[0]
        normalized.insert(
            0,
            build_step(
                step_kind="go_to_trigger",
                target_node_id=first.target_node_id,
                expected_evidence=list(first.expected_evidence or ()) or ["objective_contact_observed"],
                success_conditions=["expected_evidence_seen"],
                failure_conditions=["expected_evidence_missing"],
                retry_budget=2,
            ),
        )
    if not any(kind in {"verify_trigger_contact", "reobserve_remote_change"} for kind in kinds):
        anchor = normalized[min(1, len(normalized) - 1)]
        normalized.insert(
            min(1, len(normalized)),
            build_step(
                step_kind="verify_trigger_contact",
                target_node_id=anchor.target_node_id,
                expected_evidence=["objective_contact_observed"],
                success_conditions=["expected_evidence_seen"],
                failure_conditions=["expected_evidence_missing"],
                retry_budget=1,
            ),
        )
        required_verification_steps.append("verify_trigger_contact")
    if not any(kind in {"verify_panel", "verify_gate"} for kind in [_normalized_step_kind(step.step_kind) for step in normalized]):
        anchor = normalized[min(2, len(normalized) - 1)]
        normalized.insert(
            min(2, len(normalized)),
            build_step(
                step_kind="verify_panel",
                target_node_id=anchor.target_node_id,
                expected_evidence=["expected_match_seen"],
                success_conditions=["expected_evidence_seen"],
                failure_conditions=["expected_evidence_missing"],
                retry_budget=1,
            ),
        )
        required_verification_steps.append("verify_panel_state")
    dedup_required = tuple(dict.fromkeys(required_verification_steps))
    return normalized, dedup_required


@dataclass(frozen=True)
class SubgoalChainState:
    active_chain: dict[str, Any] | None = None
    active_step: dict[str, Any] | None = None
    should_replan: bool = False
    replan_reason: str | None = None
    completed_chain_ids: tuple[str, ...] = ()
    aborted_chain_ids: tuple[str, ...] = ()
    chain_history: tuple[dict[str, Any], ...] = ()
    current_chain_status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_step(
    *,
    step_kind: str,
    target_node_id: str | None,
    expected_evidence: list[str] | tuple[str, ...] | None = None,
    success_conditions: list[str] | tuple[str, ...] | None = None,
    failure_conditions: list[str] | tuple[str, ...] | None = None,
    retry_budget: int = 1,
    retry_count: int = 0,
    depends_on_step_ids: list[str] | tuple[str, ...] | None = None,
    step_status: str = "pending",
    verification_points: list[str] | tuple[str, ...] | None = None,
    fallback_targets: list[str] | tuple[str, ...] | None = None,
) -> SubgoalStep:
    normalized_kind = str(step_kind or "").strip()
    if normalized_kind == "reobserve_region":
        normalized_kind = "reobserve_remote_change"
    if normalized_kind not in STEP_KINDS:
        raise ValueError(f"invalid subgoal step kind: {step_kind!r}")
    fingerprint = (
        normalized_kind,
        str(target_node_id or ""),
        tuple(str(value) for value in list(expected_evidence or []) if value),
        tuple(str(value) for value in list(depends_on_step_ids or []) if value),
    )
    return SubgoalStep(
        step_id=f"subgoal_step:{stable_digest(fingerprint)}",
        step_kind=normalized_kind,
        target_node_id=str(target_node_id) if target_node_id is not None else None,
        expected_evidence=tuple(dict.fromkeys(str(value) for value in list(expected_evidence or []) if value)),
        success_conditions=tuple(dict.fromkeys(str(value) for value in list(success_conditions or []) if value)),
        failure_conditions=tuple(dict.fromkeys(str(value) for value in list(failure_conditions or []) if value)),
        retry_budget=max(0, int(retry_budget or 0)),
        retry_count=max(0, int(retry_count or 0)),
        depends_on_step_ids=tuple(dict.fromkeys(str(value) for value in list(depends_on_step_ids or []) if value)),
        step_status=str(step_status or "pending"),
        verification_points=tuple(dict.fromkeys(str(value) for value in list(verification_points or []) if value)),
        fallback_targets=tuple(dict.fromkeys(str(value) for value in list(fallback_targets or []) if value)),
    )


def build_chain(
    *,
    source_candidate_id: str,
    source_path_id: str | None,
    source_hypothesis_ids: list[str] | tuple[str, ...] | None,
    target_exit_id: str | None,
    expected_outcome_ids: list[str] | tuple[str, ...] | None,
    fallback_policy: str,
    created_round_id: int,
    last_updated_round_id: int | None = None,
    steps: list[SubgoalStep] | tuple[SubgoalStep, ...],
    status: str = "planned",
    exit_readiness_score_at_creation: float = 0.0,
    required_verification_steps: list[str] | tuple[str, ...] | None = None,
    verification_steps_completed: list[str] | tuple[str, ...] | None = None,
    completion_reason: str | None = None,
    abort_reason: str | None = None,
) -> SubgoalChain:
    normalized_steps, inferred_required_verification = _enforce_exit_chain_order(list(steps or []))
    chain_key = (
        str(source_candidate_id),
        str(source_path_id or ""),
        tuple(step.step_id for step in list(normalized_steps or [])),
        tuple(str(value) for value in list(source_hypothesis_ids or []) if value),
        str(target_exit_id or ""),
    )
    return SubgoalChain(
        chain_id=f"subgoal_chain:{stable_digest(chain_key)}",
        source_candidate_id=str(source_candidate_id),
        source_path_id=str(source_path_id) if source_path_id is not None else None,
        source_hypothesis_ids=tuple(dict.fromkeys(str(value) for value in list(source_hypothesis_ids or []) if value)),
        target_exit_id=str(target_exit_id) if target_exit_id is not None else None,
        current_step_index=0,
        status=str(status),
        expected_outcome_ids=tuple(dict.fromkeys(str(value) for value in list(expected_outcome_ids or []) if value)),
        fallback_policy=str(fallback_policy or "replan"),
        created_round_id=int(created_round_id),
        last_updated_round_id=int(last_updated_round_id if last_updated_round_id is not None else created_round_id),
        exit_readiness_score_at_creation=float(exit_readiness_score_at_creation or 0.0),
        required_verification_steps=tuple(dict.fromkeys([*(str(value) for value in list(required_verification_steps or []) if value), *inferred_required_verification])),
        verification_steps_completed=tuple(dict.fromkeys(str(value) for value in list(verification_steps_completed or []) if value)),
        completion_reason=str(completion_reason) if completion_reason else None,
        abort_reason=str(abort_reason) if abort_reason else None,
        steps=tuple(normalized_steps or ()),
    )
