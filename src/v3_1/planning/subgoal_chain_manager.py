from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from v3_1.planning.subgoal_chain import SubgoalChainState, SubgoalStep, build_chain


def _synthetic_steps_from_candidate(candidate: dict) -> list[SubgoalStep]:
    objective_type = str(candidate.get("objective_type") or "")
    target_node_id = str(candidate.get("target_entity_id") or candidate.get("target_node_id") or candidate.get("target_area_id") or "")
    if not target_node_id:
        return []
    step_map = {
        "verify_trigger_contact": "verify_trigger_contact",
        "reobserve_remote_change": "reobserve_region",
        "verify_panel_state": "verify_panel",
        "verify_gate_match": "verify_gate",
        "unlock_then_exit": "attempt_exit",
        "attempt_exit": "attempt_exit",
    }
    step_kind = step_map.get(objective_type)
    if not step_kind:
        return []
    return [
        SubgoalStep(
            step_id=f"step:{candidate.get('candidate_id') or target_node_id}:{step_kind}",
            step_kind=step_kind,
            target_node_id=target_node_id,
            expected_evidence=tuple(str(value) for value in list(candidate.get("candidate_expected_evidence", []) or []) if value),
            success_conditions=tuple(str(value) for value in list(candidate.get("success_conditions", []) or ["expected_evidence_seen"]) if value),
            failure_conditions=tuple(str(value) for value in list(candidate.get("failure_conditions", []) or ["missing_expected_evidence"]) if value),
            retry_budget=1 if step_kind in {"verify_panel", "verify_gate", "attempt_exit"} else 2,
            retry_count=0,
            depends_on_step_ids=(),
            step_status="planned",
            verification_points=tuple(str(value) for value in list(candidate.get("candidate_verification_points", []) or []) if value),
            fallback_targets=tuple(str(value) for value in list(candidate.get("candidate_fallback_targets", []) or []) if value),
        )
    ]


@dataclass
class SubgoalChainManager:
    active_chain: dict[str, Any] | None = None
    retry_counts: dict[str, int] = field(default_factory=dict)
    completed_chain_ids: list[str] = field(default_factory=list)
    aborted_chain_ids: list[str] = field(default_factory=list)
    chain_history: list[dict[str, Any]] = field(default_factory=list)
    _should_replan: bool = False
    attempt_exit_blocked_by_missing_verification: bool = False
    missing_verification_step_kind: str | None = None
    chain_rewritten_to_verification: bool = False

    def start_chain(self, *, selected_candidate: dict, round_id: int) -> dict[str, Any] | None:
        if self.active_chain and str(dict(self.active_chain).get("status") or "") not in {"aborted", "completed"}:
            return dict(self.active_chain)
        candidate = dict(selected_candidate or {})
        step_plan = list(candidate.get("candidate_step_plan", []) or [])
        if not step_plan:
            step_plan = [step.to_dict() for step in _synthetic_steps_from_candidate(candidate)]
        if not step_plan:
            return None
        normalized_steps = []
        for row in step_plan:
            if isinstance(row, SubgoalStep):
                normalized_steps.append(row)
            elif isinstance(row, dict):
                normalized_steps.append(SubgoalStep(**dict(row)))
        if not normalized_steps:
            return None
        chain = build_chain(
            source_candidate_id=str(candidate.get("candidate_id") or ""),
            source_path_id=str(candidate.get("source_path_id") or candidate.get("graph_path_id") or "") or None,
            source_hypothesis_ids=list(candidate.get("supporting_hypothesis_ids", []) or []),
            target_exit_id=str(candidate.get("target_exit_id") or "") or None,
            expected_outcome_ids=list(candidate.get("expected_outcome_ids", []) or []),
            fallback_policy=str(candidate.get("fallback_policy") or "replan"),
            created_round_id=int(round_id),
            exit_readiness_score_at_creation=float(candidate.get("exit_readiness_score", 0.0) or 0.0),
            required_verification_steps=list(candidate.get("required_verification_steps", []) or []),
            verification_steps_completed=list(candidate.get("verification_steps_completed", []) or []),
            steps=tuple(normalized_steps),
        )
        self.active_chain = chain.to_dict()
        self.active_chain["status"] = "planned"
        self.active_chain["last_updated_round_id"] = int(round_id)
        self.active_chain["completion_reason"] = None
        self.active_chain["abort_reason"] = None
        if self.current_step() is not None:
            current_step = dict(self.current_step() or {})
            current_step["step_status"] = "ready"
            self._replace_current_step(current_step)
        self._rewrite_attempt_exit_if_needed(round_id=round_id)
        self.retry_counts = {}
        self._should_replan = False
        self.attempt_exit_blocked_by_missing_verification = False
        self.missing_verification_step_kind = None
        self.chain_rewritten_to_verification = False
        self.chain_history.append({"event": "started", "chain": dict(self.active_chain)})
        return dict(self.active_chain)

    def current_chain(self) -> dict[str, Any] | None:
        return dict(self.active_chain) if self.active_chain else None

    def current_step(self) -> dict[str, Any] | None:
        if not self.active_chain:
            return None
        steps = list(self.active_chain.get("steps", []) or [])
        step_index = int(self.active_chain.get("current_step_index", 0) or 0)
        if step_index < 0 or step_index >= len(steps):
            return None
        return dict(steps[step_index])

    def _replace_current_step(self, next_step: dict[str, Any]) -> None:
        if not self.active_chain:
            return
        steps = [dict(row) for row in list(self.active_chain.get("steps", []) or [])]
        step_index = int(self.active_chain.get("current_step_index", 0) or 0)
        if 0 <= step_index < len(steps):
            steps[step_index] = dict(next_step)
            self.active_chain["steps"] = steps

    def advance_step(self, *, round_id: int, reason: str | None = None) -> dict[str, Any]:
        if not self.active_chain:
            return self.snapshot()
        current_step = self.current_step()
        if current_step:
            current_step["step_status"] = "completed"
            self._replace_current_step(current_step)
        next_index = int(self.active_chain.get("current_step_index", 0) or 0) + 1
        self.active_chain["current_step_index"] = next_index
        self.active_chain["last_updated_round_id"] = int(round_id)
        if next_index >= len(list(self.active_chain.get("steps", []) or [])):
            return self.complete_chain(round_id=round_id, reason=reason or "all_steps_completed")
        if self._rewrite_attempt_exit_if_needed(round_id=round_id):
            return self.snapshot()
        self.active_chain["status"] = "active"
        next_step = self.current_step()
        if next_step:
            next_step["step_status"] = "ready"
            self._replace_current_step(next_step)
        self._should_replan = False
        self.attempt_exit_blocked_by_missing_verification = False
        self.missing_verification_step_kind = None
        self.chain_rewritten_to_verification = False
        self.chain_history.append({"event": "advanced", "chain_id": self.active_chain.get("chain_id"), "step_index": next_index, "reason": reason})
        return self.snapshot()

    def retry_step(self, *, round_id: int, reason: str | None = None) -> dict[str, Any]:
        if not self.active_chain:
            return self.snapshot()
        current_step = self.current_step()
        if current_step is None:
            return self.snapshot()
        step_id = str(current_step.get("step_id") or "")
        retry_count = int(self.retry_counts.get(step_id, 0) or 0) + 1
        self.retry_counts[step_id] = retry_count
        current_step["retry_count"] = retry_count
        current_step["step_status"] = "retrying"
        self._replace_current_step(current_step)
        self.active_chain["status"] = "retrying"
        self.active_chain["last_updated_round_id"] = int(round_id)
        self.chain_history.append({"event": "retry", "chain_id": self.active_chain.get("chain_id"), "step_id": step_id, "retry_count": retry_count, "reason": reason})
        if retry_count > int(current_step.get("retry_budget", 0) or 0):
            return self.abort_chain(round_id=round_id, reason=reason or "retry_budget_exceeded")
        self._should_replan = False
        return self.snapshot()

    def abort_chain(self, *, round_id: int, reason: str | None = None) -> dict[str, Any]:
        if not self.active_chain:
            return self.snapshot()
        current_step = self.current_step()
        if current_step:
            current_step["step_status"] = "aborted"
            self._replace_current_step(current_step)
        self.active_chain["status"] = "aborted"
        self.active_chain["abort_reason"] = str(reason or "chain_aborted")
        self.active_chain["last_updated_round_id"] = int(round_id)
        chain_id = str(self.active_chain.get("chain_id") or "")
        if chain_id and chain_id not in self.aborted_chain_ids:
            self.aborted_chain_ids.append(chain_id)
        self._should_replan = True
        self.chain_history.append({"event": "aborted", "chain_id": chain_id, "reason": self.active_chain["abort_reason"]})
        return self.snapshot()

    def complete_chain(self, *, round_id: int, reason: str | None = None) -> dict[str, Any]:
        if not self.active_chain:
            return self.snapshot()
        current_step = self.current_step()
        if current_step:
            current_step["step_status"] = "completed"
            self._replace_current_step(current_step)
        self.active_chain["status"] = "completed"
        self.active_chain["completion_reason"] = str(reason or "completed")
        self.active_chain["last_updated_round_id"] = int(round_id)
        chain_id = str(self.active_chain.get("chain_id") or "")
        if chain_id and chain_id not in self.completed_chain_ids:
            self.completed_chain_ids.append(chain_id)
        self._should_replan = False
        self.chain_history.append({"event": "completed", "chain_id": chain_id, "reason": self.active_chain["completion_reason"]})
        return self.snapshot()

    def update_from_outcome(self, outcome: dict | None) -> dict[str, Any]:
        if not self.active_chain:
            self._should_replan = False
            return self.snapshot()
        outcome_payload = dict(outcome or {})
        chain_id = str(outcome_payload.get("chain_id") or "")
        if chain_id and chain_id != str(self.active_chain.get("chain_id") or ""):
            return self.snapshot()
        current_step = self.current_step()
        if current_step is None:
            return self.snapshot()
        step_id = str(outcome_payload.get("step_id") or "")
        if step_id and step_id != str(current_step.get("step_id") or ""):
            return self.snapshot()
        round_id = int(outcome_payload.get("round_id", self.active_chain.get("last_updated_round_id", 0)) or 0)
        step_kind = str(current_step.get("step_kind") or "")
        avatar_confident = bool(outcome_payload.get("avatar_localization_confident", False))
        if step_kind in {"go_to_trigger", "verify_trigger_contact", "retry_trigger", "attempt_exit"} and not avatar_confident:
            outcome_payload["step_success"] = False
            outcome_payload["chain_should_advance"] = False
            outcome_payload["chain_should_retry"] = True
            outcome_payload["step_failure_reason"] = "avatar_localization_low_confidence"
        if bool(outcome_payload.get("step_success")):
            completed = list(self.active_chain.get("verification_steps_completed", []) or [])
            completed_label = _verification_label_for_step(step_kind)
            if completed_label and completed_label not in completed:
                completed.append(completed_label)
                self.active_chain["verification_steps_completed"] = completed
        self.chain_history.append({"event": "step_outcome", "outcome": dict(outcome_payload)})
        if bool(outcome_payload.get("chain_should_abort")):
            return self.abort_chain(round_id=round_id, reason=str(outcome_payload.get("step_failure_reason") or outcome_payload.get("chain_completion_reason") or "chain_should_abort"))
        if bool(outcome_payload.get("chain_should_advance")):
            return self.advance_step(round_id=round_id, reason=str(outcome_payload.get("chain_completion_reason") or "advance_on_success"))
        if bool(outcome_payload.get("chain_should_retry")):
            return self.retry_step(round_id=round_id, reason=str(outcome_payload.get("step_failure_reason") or "chain_should_retry"))
        return self.abort_chain(round_id=round_id, reason=str(outcome_payload.get("step_failure_reason") or "chain_progress_invalid"))

    def should_replan(self) -> bool:
        return bool(self._should_replan)

    def snapshot(self) -> dict[str, Any]:
        current_step = self.current_step()
        step_kind = str(dict(current_step or {}).get("step_kind") or "")
        active_structure_chain = bool(current_step) and step_kind in {"verify_trigger_contact", "reobserve_region", "reobserve_remote_change", "verify_panel", "verify_gate"}
        payload = SubgoalChainState(
            active_chain=dict(self.active_chain) if self.active_chain else None,
            active_step=current_step,
            should_replan=bool(self._should_replan),
            completed_chain_ids=tuple(self.completed_chain_ids),
            aborted_chain_ids=tuple(self.aborted_chain_ids),
            chain_history=tuple(dict(row) for row in self.chain_history),
            current_chain_status=str(dict(self.active_chain or {}).get("status") or "") or None,
        ).to_dict()
        payload["attempt_exit_blocked_by_missing_verification"] = bool(self.attempt_exit_blocked_by_missing_verification)
        payload["missing_verification_step_kind"] = self.missing_verification_step_kind
        payload["chain_rewritten_to_verification"] = bool(self.chain_rewritten_to_verification)
        payload["active_structure_chain"] = active_structure_chain
        payload["structure_chain_pending_step_kind"] = step_kind or None
        payload["structure_chain_blocks_default_progress"] = active_structure_chain and str(dict(self.active_chain or {}).get("status") or "") not in {"aborted", "completed"}
        return payload

    def _rewrite_attempt_exit_if_needed(self, *, round_id: int) -> bool:
        next_step = self.current_step()
        current_kind = str(next_step.get("step_kind") or "")
        if current_kind == "reobserve_region":
            current_kind = "reobserve_remote_change"
        if not self.active_chain or not next_step or current_kind != "attempt_exit":
            return False
        required = [str(value) for value in list(self.active_chain.get("required_verification_steps", []) or []) if value]
        completed = {str(value) for value in list(self.active_chain.get("verification_steps_completed", []) or []) if value}
        missing = [value for value in required if value not in completed]
        readiness_score = float(self.active_chain.get("exit_readiness_score_at_creation", 0.0) or 0.0)
        if not missing and readiness_score >= 0.72:
            return False
        replacement = _replacement_step_from_attempt_exit(dict(next_step), missing[0] if missing else "reobserve_remote_change")
        self._replace_current_step(replacement)
        self.active_chain["status"] = "active"
        self.active_chain["last_updated_round_id"] = int(round_id)
        self.attempt_exit_blocked_by_missing_verification = True
        self.missing_verification_step_kind = missing[0] if missing else "reobserve_remote_change"
        self.chain_rewritten_to_verification = True
        self.chain_history.append(
            {
                "event": "rewritten_to_verification",
                "chain_id": self.active_chain.get("chain_id"),
                "replacement_step_kind": self.missing_verification_step_kind,
            }
        )
        return True


def _verification_label_for_step(step_kind: str) -> str | None:
    return {
        "verify_trigger_contact": "verify_trigger_contact",
        "go_to_trigger": "verify_trigger_contact",
        "retry_trigger": "verify_trigger_contact",
        "reobserve_region": "reobserve_remote_change",
        "reobserve_remote_change": "reobserve_remote_change",
        "verify_panel": "verify_panel_state",
        "verify_gate": "verify_gate_match",
    }.get(str(step_kind or ""))


def _replacement_step_from_attempt_exit(step: dict, replacement_kind: str) -> dict[str, Any]:
    next_kind = {
        "verify_trigger_contact": "verify_trigger_contact",
        "reobserve_remote_change": "reobserve_remote_change",
        "verify_panel_state": "verify_panel",
        "verify_gate_match": "verify_gate",
    }.get(str(replacement_kind or ""), "reobserve_remote_change")
    expected_evidence = {
        "verify_trigger_contact": ["objective_contact_observed"],
        "reobserve_remote_change": ["remote_change_observed", "target_presence_observed"],
        "verify_panel": ["expected_match_seen"],
        "verify_gate": ["gate_state_seen"],
    }.get(next_kind, list(step.get("expected_evidence", []) or []))
    return {
        **dict(step),
        "step_kind": next_kind,
        "expected_evidence": expected_evidence,
        "step_status": "ready",
    }
