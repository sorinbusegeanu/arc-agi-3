from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from v3_1.planning.subgoal_chain import SubgoalChainState, SubgoalStep, build_chain


@dataclass
class SubgoalChainManager:
    active_chain: dict[str, Any] | None = None
    retry_counts: dict[str, int] = field(default_factory=dict)
    completed_chain_ids: list[str] = field(default_factory=list)
    aborted_chain_ids: list[str] = field(default_factory=list)
    chain_history: list[dict[str, Any]] = field(default_factory=list)
    _should_replan: bool = False

    def start_chain(self, *, selected_candidate: dict, round_id: int) -> dict[str, Any] | None:
        candidate = dict(selected_candidate or {})
        step_plan = list(candidate.get("candidate_step_plan", []) or [])
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
        self.retry_counts = {}
        self._should_replan = False
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
        self.active_chain["status"] = "active"
        next_step = self.current_step()
        if next_step:
            next_step["step_status"] = "ready"
            self._replace_current_step(next_step)
        self._should_replan = False
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
        if step_kind in {"go_to_trigger", "retry_trigger", "attempt_exit"} and not avatar_confident:
            outcome_payload["step_success"] = False
            outcome_payload["chain_should_advance"] = False
            outcome_payload["chain_should_retry"] = True
            outcome_payload["step_failure_reason"] = "avatar_localization_low_confidence"
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
        return SubgoalChainState(
            active_chain=dict(self.active_chain) if self.active_chain else None,
            active_step=current_step,
            should_replan=bool(self._should_replan),
            completed_chain_ids=tuple(self.completed_chain_ids),
            aborted_chain_ids=tuple(self.aborted_chain_ids),
            chain_history=tuple(dict(row) for row in self.chain_history),
            current_chain_status=str(dict(self.active_chain or {}).get("status") or "") or None,
        ).to_dict()
