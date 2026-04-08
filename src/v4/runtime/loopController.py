from __future__ import annotations

from dataclasses import replace
from typing import Any

from v4.agentContract.types import V4Action, V4Observation
from v4.belief import BeliefStoreV4, BeliefUpdaterV4
from v4.click.familyAdapters import detect_pt01_phase
from v4.composition import ComposedDomainStateV4, CompositionUpdaterV4
from v4.hypothesis import HypothesisRegistryV4
from v4.hypothesis.hypothesisUpdater import HypothesisUpdaterV4
from v4.hybrid_construction.stateBuilder import HybridConstructionStateBuilderV4
from v4.memory.localMemory import LocalMemoryV4
from v4.memory.memoryUpdate import (
    ActionMemoryRecordV4,
    LocalMemoryUpdateV4,
    StepResultMemoryRecordV4,
    TestedActionOutcomeFactV4,
)
from v4.memory_hidden.stateBuilder import MemoryHiddenStateBuilderV4
from v4.policy.policyBase import PolicyBaseV4, PolicyDecisionV4
from v4.rule_switch.stateBuilder import RuleSwitchStateBuilderV4
from v4.runtime.envSession import EnvSessionV4
from v4.runtime.sessionLedger import SessionLedgerV4, SessionSummaryV4, StepLedgerRecordV4
from v4.runtime.stopConditions import StopConditionStatusV4, evaluate_stop_conditions
from v4.state.stateParser import StateParserV4
from v4.temporal import TemporalResourceStateV4, TemporalUpdaterV4
from v4.time_reactive.stateBuilder import TimeReactiveStateBuilderV4


class LoopControllerV4:
    def __init__(
        self,
        *,
        env_session: EnvSessionV4,
        state_parser: StateParserV4,
        policy: PolicyBaseV4,
        local_memory: LocalMemoryV4,
        ledger: SessionLedgerV4,
        belief_store: BeliefStoreV4 | None = None,
        belief_updater: BeliefUpdaterV4 | None = None,
        hypothesis_registry: HypothesisRegistryV4 | None = None,
        hypothesis_updater: HypothesisUpdaterV4 | None = None,
        temporal_updater: TemporalUpdaterV4 | None = None,
        composition_updater: CompositionUpdaterV4 | None = None,
        max_steps: int = 32,
    ) -> None:
        self.env_session = env_session
        self.state_parser = state_parser
        self.policy = policy
        self.local_memory = local_memory
        self.ledger = ledger
        self.belief_store = belief_store if belief_store is not None else BeliefStoreV4()
        self.belief_updater = belief_updater if belief_updater is not None else BeliefUpdaterV4()
        self.hypothesis_registry = hypothesis_registry if hypothesis_registry is not None else HypothesisRegistryV4()
        self.hypothesis_updater = hypothesis_updater if hypothesis_updater is not None else HypothesisUpdaterV4()
        self.temporal_updater = temporal_updater if temporal_updater is not None else TemporalUpdaterV4()
        self.composition_updater = composition_updater if composition_updater is not None else CompositionUpdaterV4()
        self.temporal_state: TemporalResourceStateV4 | None = None
        self.composition_state: ComposedDomainStateV4 | None = None
        self._step8_trace_rows: list[dict[str, object]] = []
        self.max_steps = int(max_steps)

    def run(self) -> SessionSummaryV4:
        self._step8_trace_rows.clear()
        current_observation = self.env_session.reset()
        if isinstance(current_observation, V4Observation):
            initial_belief = self.belief_updater.initialize_from_observation(
                current_observation,
                self.env_session.environment_metadata,
                step_index=0,
            )
            self.belief_store.replace(initial_belief)
        else:
            self.belief_store.reset()
        self.hypothesis_registry.reset()
        self.temporal_state = self.temporal_updater.initialize_from_observation(
            current_observation,
            self.env_session.environment_metadata,
            step_index=0,
        )
        self.composition_state = None
        previous_decision_basis_summary: dict[str, object] | None = None
        previous_observation: V4Observation | None = None
        stop_status = StopConditionStatusV4(False, evaluate_stop_conditions(step_result=None, next_step_index=0, max_steps=self.max_steps).reason)
        while not stop_status.should_stop:
            step_index = self.env_session.step_index
            parsed_state = None
            decision = None
            executed_action = None
            transition = None
            step_result = None
            memory_update = None
            failure_bucket = None
            step8_trace_context: dict[str, object] | None = None
            try:
                parsed_state = self.state_parser.build_parsed_state(
                    current_observation=current_observation,
                    previous_observation=previous_observation,
                    environment_metadata=self.env_session.environment_metadata,
                    local_memory_snapshot=self.local_memory.snapshot(),
                    belief_snapshot=self.belief_store.snapshot(),
                    hypothesis_snapshot=self.hypothesis_registry.snapshot(),
                    temporal_snapshot=self.temporal_state,
                    composition_snapshot=self.composition_state,
                    step_index=step_index,
                )
                if parsed_state.current_observation.game_id == "ms01":
                    updated_belief = self.belief_updater.update_from_observation(
                        self.belief_store.snapshot(),
                        current_observation,
                        self.env_session.environment_metadata,
                        step_index,
                        parsed_state=parsed_state,
                    )
                    self.belief_store.replace(updated_belief)
                    parsed_state = self.state_parser.build_parsed_state(
                        current_observation=current_observation,
                        previous_observation=previous_observation,
                        environment_metadata=self.env_session.environment_metadata,
                        local_memory_snapshot=self.local_memory.snapshot(),
                        belief_snapshot=self.belief_store.snapshot(),
                        hypothesis_snapshot=self.hypothesis_registry.snapshot(),
                        temporal_snapshot=self.temporal_state,
                        composition_snapshot=self.composition_state,
                        step_index=step_index,
                    )
                step8_trace_context = {
                    "step_index": step_index,
                    "game_id": parsed_state.current_observation.game_id,
                    "state_hash": parsed_state.derived_control.state_hash,
                    "selected_goal_kind": None,
                    "selected_subgoal_kind": None,
                    "extracted_subgoal_kinds": (),
                    "subgoal_progress_rows": (),
                    "belief_reference_present": parsed_state.belief_reference is not None,
                    "hypothesis_reference_present": parsed_state.hypothesis_reference is not None,
                    "temporal_reference_present": parsed_state.temporal_reference is not None,
                    "composition_reference_present": parsed_state.composition_reference is not None,
                    "belief_unknown_cell_count": parsed_state.belief_reference.unknown_cell_count if parsed_state.belief_reference is not None else parsed_state.derived_control.unknown_cell_count,
                    "belief_frontier_cell_count": parsed_state.belief_reference.frontier_cell_count if parsed_state.belief_reference is not None else 0,
                    "hypothesis_count": parsed_state.hypothesis_reference.hypothesis_count if parsed_state.hypothesis_reference is not None else 0,
                    "safe_horizon_steps": parsed_state.temporal_reference.safe_horizon_steps if parsed_state.temporal_reference is not None else 0,
                    "hazard_window_remaining": parsed_state.temporal_reference.hazard_window_remaining if parsed_state.temporal_reference is not None else None,
                    "composition_domain_count": parsed_state.composition_reference.domain_count if parsed_state.composition_reference is not None else 0,
                    "composition_present_domains": parsed_state.composition_reference.present_domain_names if parsed_state.composition_reference is not None else (),
                    "composition_cross_domain_effect_count": parsed_state.composition_reference.cross_domain_effect_count if parsed_state.composition_reference is not None else 0,
                    "raw_state_text": parsed_state.current_observation.state if isinstance(parsed_state.current_observation.state, str) else repr(parsed_state.current_observation.state),
                    "frame_summary": self._build_frame_summary(parsed_state.current_observation.frame),
                    "environment_metadata_summary": self._build_environment_metadata_summary(self.env_session.environment_metadata),
                    "ms01_builder_ok": False,
                    "ms01_builder_error": None,
                    "ms01_builder_summary": {},
                    "rs01_builder_ok": False,
                    "rs01_builder_error": None,
                    "rs01_builder_summary": {},
                    "pt01_phase_detector_ok": False,
                    "pt01_phase_detector_error": None,
                    "pt01_phase_detector_summary": {},
                    "sv01_builder_ok": False,
                    "sv01_builder_error": None,
                    "sv01_builder_summary": {},
                    "tb01_builder_ok": False,
                    "tb01_builder_error": None,
                    "tb01_builder_summary": {},
                    "hypothesis_update_debug": {},
                    "hypothesis_update_debug_branch": None,
                    "hypothesis_update_debug_would_emit": False,
                    "hypothesis_update_debug_candidate_values": (),
                    "hypothesis_update_debug_error": None,
                    "emitted_hypothesis_count": 0,
                    "emitted_hypothesis_ids": (),
                    "emitted_hypothesis_kinds": (),
                    "emitted_hypothesis_candidate_values": (),
                    "registry_hypothesis_count_after_update": 0,
                    "registry_hypothesis_ids_after_update": (),
                    "registry_hypothesis_kinds_after_update": (),
                    "registry_hypothesis_candidate_values_after_update": (),
                    "registry_hypothesis_confidence_bands_after_update": (),
                    "registry_hypothesis_expiry_after_update": (),
                    "registry_evidence_entry_count_after_update": 0,
                    "executed_action_id": None,
                    "executed_action_name": None,
                    "candidate_count_total": 0,
                    "accepted_candidate_count_total": 0,
                    "generated_step6_count": 0,
                    "generated_step7_count": 0,
                    "generated_step8_count": 0,
                    "generator_debug": {},
                    "generation_metrics_snapshot": {},
                    "accepted_step6_count": 0,
                    "accepted_step7_count": 0,
                    "accepted_step8_count": 0,
                    "selected_is_step6": False,
                    "selected_is_step7": False,
                    "selected_is_step8": False,
                    "decision_basis_summary": {},
                    "decision_basis_changed_from_previous": False,
                }
                self._populate_builder_diagnostics(parsed_state, step8_trace_context)
                hypothesis_debug = self.hypothesis_updater.debug_update_inputs(parsed_state)
                step8_trace_context["hypothesis_update_debug"] = hypothesis_debug
                step8_trace_context["hypothesis_update_debug_branch"] = hypothesis_debug.get("update_branch")
                step8_trace_context["hypothesis_update_debug_would_emit"] = bool(hypothesis_debug.get("would_emit_hypotheses", False))
                step8_trace_context["hypothesis_update_debug_candidate_values"] = tuple(hypothesis_debug.get("would_emit_candidate_values", ()))
                step8_trace_context["hypothesis_update_debug_error"] = hypothesis_debug.get("builder_error") or hypothesis_debug.get("phase_detector_error")
                new_hypotheses = self.hypothesis_updater.update_from_parsed_state(self.env_session.step_index, parsed_state)
                step8_trace_context["emitted_hypothesis_count"] = len(new_hypotheses)
                step8_trace_context["emitted_hypothesis_ids"] = tuple(item.hypothesis_id for item in new_hypotheses)
                step8_trace_context["emitted_hypothesis_kinds"] = tuple(item.kind for item in new_hypotheses)
                step8_trace_context["emitted_hypothesis_candidate_values"] = tuple(item.payload.get("candidate_value") for item in new_hypotheses)
                stored_hypothesis_state = self.hypothesis_registry.update(self.env_session.step_index, new_hypotheses)
                step8_trace_context["registry_hypothesis_count_after_update"] = len(stored_hypothesis_state.hypotheses)
                step8_trace_context["registry_hypothesis_ids_after_update"] = tuple(item.hypothesis_id for item in stored_hypothesis_state.hypotheses)
                step8_trace_context["registry_hypothesis_kinds_after_update"] = tuple(item.kind for item in stored_hypothesis_state.hypotheses)
                step8_trace_context["registry_hypothesis_candidate_values_after_update"] = tuple(item.payload.get("candidate_value") for item in stored_hypothesis_state.hypotheses)
                step8_trace_context["registry_hypothesis_confidence_bands_after_update"] = tuple(item.confidence_band for item in stored_hypothesis_state.hypotheses)
                step8_trace_context["registry_hypothesis_expiry_after_update"] = tuple(item.expiry_revision for item in stored_hypothesis_state.hypotheses)
                step8_trace_context["registry_evidence_entry_count_after_update"] = len(self.hypothesis_registry.ledger_entries())
                parsed_state = self.state_parser.build_parsed_state(
                    current_observation=current_observation,
                    previous_observation=previous_observation,
                    environment_metadata=self.env_session.environment_metadata,
                    local_memory_snapshot=self.local_memory.snapshot(),
                    belief_snapshot=self.belief_store.snapshot(),
                    hypothesis_snapshot=self.hypothesis_registry.snapshot(),
                    temporal_snapshot=self.temporal_state,
                    composition_snapshot=self.composition_state,
                    step_index=step_index,
                )
                if self.composition_state is None:
                    self.composition_state = self.composition_updater.initialize_from_parsed_state(parsed_state)
                else:
                    self.composition_state = self.composition_updater.update_from_parsed_state(self.composition_state, parsed_state)
                parsed_state = self.state_parser.build_parsed_state(
                    current_observation=current_observation,
                    previous_observation=previous_observation,
                    environment_metadata=self.env_session.environment_metadata,
                    local_memory_snapshot=self.local_memory.snapshot(),
                    belief_snapshot=self.belief_store.snapshot(),
                    hypothesis_snapshot=self.hypothesis_registry.snapshot(),
                    temporal_snapshot=self.temporal_state,
                        composition_snapshot=self.composition_state,
                        step_index=step_index,
                    )
                # Stage 2 stays solver-agnostic here; movement and click solver heads
                # both implement the same policy decision surface.
                decision = self.policy.decide(parsed_state)
                if step8_trace_context is not None:
                    annotations = decision.annotations if isinstance(getattr(decision, "annotations", None), dict) else {}
                    normalized_decision_surface = self._normalize_decision_surface(decision)
                    decision_basis_summary = self._normalize_decision_basis_summary(decision)
                    generated_step6_count = int(annotations.get("generated_step6_count", 0) or 0)
                    generated_step7_count = int(annotations.get("generated_step7_count", 0) or 0)
                    generated_step8_count = int(annotations.get("generated_step8_count", 0) or 0)
                    step8_trace_context["selected_goal_kind"] = normalized_decision_surface["selected_goal_kind"]
                    step8_trace_context["selected_subgoal_kind"] = normalized_decision_surface["selected_subgoal_kind"]
                    step8_trace_context["extracted_subgoal_kinds"] = tuple(annotations.get("extracted_subgoal_kinds", ()))
                    step8_trace_context["subgoal_progress_rows"] = tuple(annotations.get("subgoal_progress_rows", ()))
                    step8_trace_context["candidate_count_total"] = annotations.get("candidate_count", 0)
                    step8_trace_context["accepted_candidate_count_total"] = annotations.get("accepted_candidate_count", 0)
                    step8_trace_context["generated_step6_count"] = generated_step6_count
                    step8_trace_context["generated_step7_count"] = generated_step7_count if generated_step7_count > 0 else normalized_decision_surface["generated_step7_count"]
                    step8_trace_context["generated_step8_count"] = generated_step8_count if generated_step8_count > 0 else normalized_decision_surface["generated_step8_count"]
                    step8_trace_context["generator_debug"] = annotations.get("generator_debug", {})
                    step8_trace_context["generation_metrics_snapshot"] = annotations.get("generation_metrics_snapshot", {})
                    step8_trace_context["accepted_step6_count"] = annotations.get("accepted_step6_count", 0)
                    step8_trace_context["accepted_step7_count"] = normalized_decision_surface["accepted_step7_count"]
                    step8_trace_context["accepted_step8_count"] = normalized_decision_surface["accepted_step8_count"]
                    step8_trace_context["selected_is_step6"] = annotations.get("selected_is_step6", False)
                    step8_trace_context["selected_is_step7"] = normalized_decision_surface["selected_is_step7"]
                    step8_trace_context["selected_is_step8"] = normalized_decision_surface["selected_is_step8"]
                    step8_trace_context["decision_basis_summary"] = decision_basis_summary
                    step8_trace_context["decision_basis_changed_from_previous"] = bool(
                        decision_basis_summary
                    ) and decision_basis_summary != (previous_decision_basis_summary or {})
                    previous_decision_basis_summary = dict(decision_basis_summary) if decision_basis_summary else previous_decision_basis_summary
                executed_action = self._select_executed_action(decision)
                if step8_trace_context is not None:
                    step8_trace_context["executed_action_id"] = executed_action.action_id
                    step8_trace_context["executed_action_name"] = executed_action.action_name
                transition, step_result = self.env_session.step(executed_action)
                changed_cells = self._compute_changed_cells(current_observation, transition.post_observation)
                pre_observation_summary = self._build_observation_summary(current_observation)
                post_observation_summary = self._build_observation_summary(transition.post_observation)
                if step8_trace_context is not None:
                    step8_trace_context["changed_cells"] = changed_cells
                    step8_trace_context["pre_observation_summary"] = pre_observation_summary
                    step8_trace_context["post_observation_summary"] = post_observation_summary
                reconciled_hypotheses = self.hypothesis_updater.reconcile_after_step(
                    self.env_session.step_index,
                    self.hypothesis_registry.snapshot().hypotheses,
                    transition.post_observation,
                )
                self.hypothesis_registry.update(self.env_session.step_index, reconciled_hypotheses)
                if isinstance(transition.post_observation, V4Observation):
                    updated_belief = self.belief_updater.update_from_observation(
                        self.belief_store.snapshot(),
                        transition.post_observation,
                        self.env_session.environment_metadata,
                        self.env_session.step_index,
                        parsed_state=parsed_state,
                    )
                    self.belief_store.replace(updated_belief)
                self.temporal_state = self.temporal_updater.update_after_step(
                    self.temporal_state,
                    transition.post_observation,
                    self.env_session.environment_metadata,
                    executed_action,
                    self.env_session.step_index,
                )
                memory_update = self._build_memory_update(parsed_state, executed_action, step_result, transition)
                self.local_memory.apply_update(memory_update)
                stop_status = evaluate_stop_conditions(
                    step_result=step_result,
                    next_step_index=self.env_session.step_index,
                    max_steps=self.max_steps,
                )
                parsed_state_summary = parsed_state.to_dict()
                decision_summary = decision.to_dict()
                step_result_dict = step_result.to_dict()
                step_result_dict["changed_cells"] = changed_cells
                step_result_dict["pre_observation_summary"] = pre_observation_summary
                step_result_dict["post_observation_summary"] = post_observation_summary
                self.ledger.append(
                    StepLedgerRecordV4(
                        step_index=step_index,
                        pre_observation=current_observation,
                        parsed_state_summary=parsed_state_summary,
                        decision_summary=decision_summary,
                        executed_action=executed_action,
                        transition_record=transition,
                        step_result=step_result_dict,
                        memory_update_summary=memory_update.to_dict(),
                        failure_bucket=None,
                        stop_condition_status=stop_status.to_dict(),
                    )
                )
                if step8_trace_context is not None:
                    self._step8_trace_rows.append(step8_trace_context)
                previous_observation = current_observation
                current_observation = transition.post_observation
            except Exception as exc:
                invalid_state_abort_details = self._build_invalid_state_abort_details(
                    exc,
                    parsed_state,
                    decision,
                    executed_action,
                    transition,
                    step_result,
                )
                parsed_state_summary = {} if parsed_state is None else parsed_state.to_dict()
                decision_summary = {} if decision is None else decision.to_dict()
                parsed_state_summary["invalid_state_abort_details"] = dict(invalid_state_abort_details)
                decision_summary["invalid_state_abort_details"] = dict(invalid_state_abort_details)
                if step_result is not None:
                    object.__setattr__(step_result, "invalid_state_abort_details", dict(invalid_state_abort_details))
                failure_bucket = self._bucket_for_exception(
                    exc,
                    parsed_state,
                    decision,
                    executed_action,
                    transition,
                    step_result,
                    memory_update,
                    invalid_state_abort_details=invalid_state_abort_details,
                )
                stop_status = evaluate_stop_conditions(
                    step_result=step_result,
                    next_step_index=self.env_session.step_index,
                    max_steps=self.max_steps,
                    invalid_state_abort=True,
                )
                self.ledger.append(
                    StepLedgerRecordV4(
                        step_index=step_index,
                        pre_observation=current_observation,
                        parsed_state_summary=parsed_state_summary,
                        decision_summary=decision_summary,
                        executed_action=executed_action,
                        transition_record=transition,
                        step_result=step_result,
                        memory_update_summary={} if memory_update is None else memory_update.to_dict(),
                        failure_bucket=failure_bucket,
                        stop_condition_status=stop_status.to_dict(),
                    )
                )
                if step8_trace_context is not None:
                    self._step8_trace_rows.append(step8_trace_context)
        return replace(self.ledger.build_summary(), step8_trace_rows=tuple(self._step8_trace_rows))

    def _build_memory_update(
        self,
        parsed_state,
        action: V4Action,
        step_result,
        transition,
    ) -> LocalMemoryUpdateV4:
        action_key = f"{action.action_id}:{action.action_name}"
        return LocalMemoryUpdateV4(
            transition_refs=(f"step:{transition.step_index if transition.step_index is not None else self.env_session.step_index}",),
            recent_actions=(ActionMemoryRecordV4(action.action_id, action.action_name, action.payload),),
            recent_step_results=(
                StepResultMemoryRecordV4(
                    raw_state_before=step_result.raw_state_before,
                    raw_state_after=step_result.raw_state_after,
                    terminal_status=step_result.terminal_signal.status,
                    reset_required=step_result.reset_required,
                ),
            ),
            visited_state_hashes=(parsed_state.derived_control.state_hash,),
            retry_count_increments={action_key: 1},
            cooldown_markers={} if step_result.terminal_signal.is_terminal else {action_key: 0},
            tested_action_outcomes=(
                TestedActionOutcomeFactV4(
                    state_hash=parsed_state.derived_control.state_hash,
                    action_id=action.action_id,
                    action_name=action.action_name,
                    outcome_signature=f"{step_result.raw_state_before}->{step_result.raw_state_after}",
                ),
            ),
        )

    @staticmethod
    def _bucket_for_exception(
        exc: Exception,
        parsed_state: Any,
        decision: Any,
        executed_action: Any,
        transition: Any,
        step_result: Any,
        memory_update: Any,
        *,
        invalid_state_abort_details: dict[str, object] | None = None,
    ) -> str:
        abort_site = None if invalid_state_abort_details is None else invalid_state_abort_details.get("abort_site")
        if abort_site == "parse_typed_state":
            return "state parsing"
        if abort_site == "reconstruct_typed_state":
            return "state reconstruction"
        if abort_site == "policy_action_selection":
            return "action selection"
        if abort_site == "post_step_state_update":
            return "post-step state update"
        if parsed_state is None:
            return "state parsing"
        if decision is None:
            return "action selection"
        if transition is None and executed_action is not None:
            return "action execution"
        if transition is None:
            return "transition building"
        if step_result is None:
            return "step-result derivation"
        if memory_update is None:
            return "local-memory update"
        return "stop-condition handling"

    @staticmethod
    def _select_executed_action(decision: PolicyDecisionV4) -> V4Action:
        return decision.first_action()

    @staticmethod
    def _normalize_decision_surface(decision: PolicyDecisionV4 | None) -> dict[str, object]:
        default = {
            "selected_goal_kind": None,
            "selected_subgoal_kind": None,
            "generated_step7_count": 0,
            "generated_step8_count": 0,
            "accepted_step7_count": 0,
            "accepted_step8_count": 0,
            "selected_is_step7": False,
            "selected_is_step8": False,
        }
        if decision is None:
            return default
        annotations = getattr(decision, "annotations", None)
        if not isinstance(annotations, dict):
            return default
        selected_goal_kind = annotations.get("goal_kind")
        selected_subgoal_kind = annotations.get("subgoal_kind")
        local_target_kind = None
        for key in ("target_kind", "target", "search_status", "fallback"):
            value = annotations.get(key)
            if value not in (None, ""):
                local_target_kind = value
                break
        if selected_goal_kind in (None, "") and selected_subgoal_kind not in (None, ""):
            selected_goal_kind = selected_subgoal_kind
        if selected_subgoal_kind in (None, "") and selected_goal_kind not in (None, ""):
            selected_subgoal_kind = selected_goal_kind
        if selected_goal_kind in (None, "") and selected_subgoal_kind in (None, "") and local_target_kind not in (None, ""):
            selected_goal_kind = local_target_kind
            selected_subgoal_kind = local_target_kind
        return {
            "selected_goal_kind": selected_goal_kind,
            "selected_subgoal_kind": selected_subgoal_kind,
            "generated_step7_count": int(annotations.get("generated_step7_count", 0) or 0),
            "generated_step8_count": int(annotations.get("generated_step8_count", 0) or 0),
            "accepted_step7_count": int(annotations.get("accepted_step7_count", 0) or 0),
            "accepted_step8_count": int(annotations.get("accepted_step8_count", 0) or 0),
            "selected_is_step7": bool(annotations.get("selected_is_step7", False)),
            "selected_is_step8": bool(annotations.get("selected_is_step8", False)),
        }

    @staticmethod
    def _normalize_decision_basis_summary(decision: PolicyDecisionV4 | None) -> dict[str, object]:
        if decision is None:
            return {}
        annotations = getattr(decision, "annotations", None)
        annotations = annotations if isinstance(annotations, dict) else {}
        summary: dict[str, object] = {}
        primary_target_kind = None
        for key in ("goal_kind", "subgoal_kind", "target_kind", "target", "search_status"):
            value = annotations.get(key)
            if value not in (None, ""):
                primary_target_kind = value
                break
        if primary_target_kind not in (None, ""):
            summary["primary_target_kind"] = primary_target_kind
        first_action = decision.first_action()
        payload = first_action.payload if isinstance(first_action.payload, dict) else {}
        target_locator = annotations.get("target_locator")
        if target_locator in (None, "") and payload:
            locator = {}
            for key in ("x", "y", "cell", "position", "target_cell", "target_position"):
                if key in payload and payload.get(key) not in (None, ""):
                    locator[key] = payload.get(key)
            target_locator = locator if locator else None
        if target_locator not in (None, "", {}):
            summary["target_locator"] = target_locator
        route_or_plan_size = None
        for key in ("route_or_plan_size", "candidate_count", "accepted_candidate_count", "branch_count"):
            value = annotations.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                route_or_plan_size = value
                break
        if route_or_plan_size is None and len(decision.short_plan) > 0:
            route_or_plan_size = len(decision.short_plan)
        if route_or_plan_size is not None:
            summary["route_or_plan_size"] = route_or_plan_size
        mode_hint = None
        for key in ("mode_hint", "fallback", "family", "search_status", "policy"):
            value = annotations.get(key)
            if value not in (None, ""):
                mode_hint = value
                break
        if mode_hint not in (None, ""):
            summary["mode_hint"] = mode_hint
        for key in (
            "candidate_count_before_filter",
            "candidate_count_after_filter",
            "candidate_count_after_ranking",
            "selected_candidate_rank",
            "selected_candidate_score",
            "selected_candidate_identity",
        ):
            value = annotations.get(key)
            if value is not None:
                summary[key] = value
        rejection_reason_counts = annotations.get("rejection_reason_counts")
        if isinstance(rejection_reason_counts, dict) and rejection_reason_counts:
            summary["rejection_reason_counts"] = dict(rejection_reason_counts)
        for key in (
            "candidate_identity_list_before_filter",
            "candidate_identity_list_after_filter",
            "candidate_identity_list_after_ranking",
        ):
            value = annotations.get(key)
            if isinstance(value, list) and value:
                summary[key] = list(value)
        for key in ("hazard_window", "bridge_anchor", "bridge_target", "construction_target"):
            value = annotations.get(key)
            if value not in (None, "", {}):
                summary[key] = value
        return summary

    @staticmethod
    def _build_frame_summary(frame) -> dict[str, object]:
        if frame is None:
            return {
                "frame_present": False,
                "frame_rows": 0,
                "frame_cols": 0,
                "distinct_values": (),
                "none_cell_count": 0,
                "question_mark_cell_count": 0,
            }
        plane = frame[0] if frame else ()
        rows = len(plane)
        cols = len(plane[0]) if plane and plane[0] else 0
        distinct_values = sorted({str(cell) for row in plane for cell in row})[:16]
        none_cell_count = sum(1 for row in plane for cell in row if cell is None)
        question_mark_cell_count = sum(1 for row in plane for cell in row if cell == "?")
        return {
            "frame_present": True,
            "frame_rows": rows,
            "frame_cols": cols,
            "distinct_values": tuple(distinct_values),
            "none_cell_count": none_cell_count,
            "question_mark_cell_count": question_mark_cell_count,
        }

    @staticmethod
    def _build_environment_metadata_summary(environment_metadata) -> dict[str, object]:
        if environment_metadata is None:
            return {}
        additional_properties = getattr(environment_metadata, "additional_properties", None) or {}
        return {
            "has_additional_properties": bool(additional_properties),
            "additional_property_keys": tuple(sorted(str(key) for key in additional_properties.keys())[:16]),
            "belief_unknown_values": additional_properties.get("belief_unknown_values"),
            "sv01_default_hazard_window": additional_properties.get("sv01_default_hazard_window"),
        }

    @staticmethod
    def _build_observation_summary(observation) -> dict[str, object]:
        if not isinstance(observation, V4Observation):
            return {
                "state": None,
                "levels_completed": None,
                "win_levels": None,
                "guid": None,
                "frame_summary": LoopControllerV4._build_frame_summary(None),
            }
        return {
            "state": observation.state,
            "levels_completed": observation.levels_completed,
            "win_levels": observation.win_levels,
            "guid": observation.guid,
            "frame_summary": LoopControllerV4._build_frame_summary(observation.frame),
        }

    @staticmethod
    def _compute_changed_cells(pre_observation, post_observation) -> int | None:
        if pre_observation is None or post_observation is None:
            return None
        if not isinstance(pre_observation, V4Observation) or not isinstance(post_observation, V4Observation):
            return None
        pre_frame = pre_observation.frame
        post_frame = post_observation.frame
        if pre_frame is None or post_frame is None:
            return None
        pre_plane = pre_frame[0] if pre_frame else None
        post_plane = post_frame[0] if post_frame else None
        if pre_plane is None or post_plane is None:
            return None
        pre_rows = len(pre_plane)
        post_rows = len(post_plane)
        overlap_rows = min(pre_rows, post_rows)
        changed_cells = 0
        for row_index in range(overlap_rows):
            pre_row = pre_plane[row_index]
            post_row = post_plane[row_index]
            pre_cols = len(pre_row)
            post_cols = len(post_row)
            overlap_cols = min(pre_cols, post_cols)
            changed_cells += sum(1 for col_index in range(overlap_cols) if pre_row[col_index] != post_row[col_index])
            changed_cells += abs(pre_cols - post_cols)
        changed_cells += abs(pre_rows - post_rows)
        return changed_cells

    @staticmethod
    def _populate_builder_diagnostics(parsed_state, trace_row: dict[str, object]) -> None:
        game_id = str(parsed_state.current_observation.game_id).split("-", 1)[0]
        if game_id == "ms01":
            try:
                typed_state = MemoryHiddenStateBuilderV4().build(parsed_state)
                trace_row["ms01_builder_ok"] = True
                trace_row["ms01_builder_summary"] = {
                    "revealed_safe_count": len(tuple(typed_state.family.revealed_safe_cells)),
                    "unrevealed_frontier_count": len(tuple(typed_state.family.unrevealed_frontier_cells)),
                    "visible_number_count": len(tuple(typed_state.family.visible_number_cells)),
                    "avatar_position": typed_state.common.avatar_position,
                }
            except Exception as exc:
                trace_row["ms01_builder_error"] = f"{type(exc).__name__}:{exc}"
        if game_id == "rs01":
            try:
                typed_state = RuleSwitchStateBuilderV4().build(parsed_state)
                family = typed_state.family
                trace_row["rs01_builder_ok"] = True
                trace_row["rs01_builder_summary"] = {
                    "active_safe_color": getattr(family, "active_safe_color", None) if hasattr(family, "active_safe_color") else None,
                    "safe_color_cycle": tuple(getattr(family, "safe_color_cycle", ())) if hasattr(family, "safe_color_cycle") else None,
                    "remaining_targets_by_color": tuple(getattr(family, "remaining_targets_by_color", ())) if hasattr(family, "remaining_targets_by_color") else None,
                    "available_family_fields": tuple(sorted(name for name in dir(family) if not name.startswith("_"))[:32]),
                }
            except Exception as exc:
                trace_row["rs01_builder_error"] = f"{type(exc).__name__}:{exc}"
        if game_id == "pt01":
            try:
                detector = detect_pt01_phase(parsed_state)
                trace_row["pt01_phase_detector_ok"] = True
                trace_row["pt01_phase_detector_summary"] = dict(detector)
            except Exception as exc:
                trace_row["pt01_phase_detector_error"] = f"{type(exc).__name__}:{exc}"
        if game_id == "sv01":
            try:
                typed_state = TimeReactiveStateBuilderV4().build(parsed_state)
                trace_row["sv01_builder_ok"] = True
                trace_row["sv01_builder_summary"] = {
                    "hunger_value": typed_state.family.hunger_value,
                    "warmth_value": typed_state.family.warmth_value,
                    "survival_timer_remaining": typed_state.family.survival_timer_remaining,
                    "wait_action_id": typed_state.family.wait_action_id,
                }
            except Exception as exc:
                trace_row["sv01_builder_error"] = f"{type(exc).__name__}:{exc}"
        if game_id == "tb01":
            try:
                typed_state = HybridConstructionStateBuilderV4().build(parsed_state)
                trace_row["tb01_builder_ok"] = True
                trace_row["tb01_builder_summary"] = {
                    "bridge_budget_remaining": typed_state.family.bridge_budget_remaining,
                    "step_limit_remaining": typed_state.family.step_limit_remaining,
                    "bridge_built_count": len(tuple(typed_state.family.bridge_built_cells)),
                    "water_cell_count": len(tuple(typed_state.family.water_cells)),
                    "land_cell_count": len(tuple(typed_state.family.land_cells)),
                    "goal_cell": typed_state.family.goal_cell,
                }
            except Exception as exc:
                trace_row["tb01_builder_error"] = f"{type(exc).__name__}:{exc}"

    @staticmethod
    def _build_invalid_state_abort_details(
        exc: Exception,
        parsed_state: Any,
        decision: Any,
        executed_action: Any,
        transition: Any,
        step_result: Any,
    ) -> dict[str, object]:
        def _normalize_field_list(value: Any) -> list[str]:
            if value is None:
                return []
            if isinstance(value, str):
                items = [item.strip() for item in value.split(",") if item.strip()]
                return sorted(items)
            if isinstance(value, (list, tuple, set)):
                return sorted(str(item).strip() for item in value if str(item).strip())
            return [str(value)]

        abort_message = str(exc)
        raw_abort_site = getattr(exc, "abort_site", None)
        reconstruction_attempted = getattr(exc, "reconstruction_attempted", None)
        if raw_abort_site == "state_builder.build":
            abort_site = "reconstruct_typed_state" if reconstruction_attempted else "parse_typed_state"
        elif raw_abort_site == "policy.decide":
            abort_site = "policy_action_selection"
        elif transition is not None or step_result is not None:
            abort_site = "post_step_state_update"
        elif abort_message == "no certified plan available":
            abort_site = "policy_action_selection"
        elif decision is None:
            abort_site = "parse_typed_state" if not reconstruction_attempted else "reconstruct_typed_state"
        else:
            abort_site = "policy_action_selection"
        missing_field = getattr(exc, "missing_field", None)
        return {
            "abort_site": abort_site,
            "abort_message": abort_message,
            "missing_field": missing_field if missing_field not in {"", None} else None,
            "required_fields": _normalize_field_list(getattr(exc, "required_fields", None)),
            "current_visible_fields": _normalize_field_list(getattr(exc, "current_visible_fields", None)),
            "previous_state_available": getattr(exc, "previous_state_available", None),
            "reconstruction_attempted": reconstruction_attempted if reconstruction_attempted in {True, False} else None,
        }
