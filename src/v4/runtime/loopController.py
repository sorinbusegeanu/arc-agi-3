from __future__ import annotations

from typing import Any

from v4.agentContract.types import V4Action, V4Observation
from v4.memory.localMemory import LocalMemoryV4
from v4.memory.memoryUpdate import (
    ActionMemoryRecordV4,
    LocalMemoryUpdateV4,
    StepResultMemoryRecordV4,
    TestedActionOutcomeFactV4,
)
from v4.policy.policyBase import PolicyBaseV4, PolicyDecisionV4
from v4.runtime.envSession import EnvSessionV4
from v4.runtime.sessionLedger import SessionLedgerV4, SessionSummaryV4, StepLedgerRecordV4
from v4.runtime.stopConditions import StopConditionStatusV4, evaluate_stop_conditions
from v4.state.stateParser import StateParserV4


class LoopControllerV4:
    def __init__(
        self,
        *,
        env_session: EnvSessionV4,
        state_parser: StateParserV4,
        policy: PolicyBaseV4,
        local_memory: LocalMemoryV4,
        ledger: SessionLedgerV4,
        max_steps: int = 32,
    ) -> None:
        self.env_session = env_session
        self.state_parser = state_parser
        self.policy = policy
        self.local_memory = local_memory
        self.ledger = ledger
        self.max_steps = int(max_steps)

    def run(self) -> SessionSummaryV4:
        current_observation = self.env_session.reset()
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
            try:
                parsed_state = self.state_parser.build_parsed_state(
                    current_observation=current_observation,
                    previous_observation=previous_observation,
                    environment_metadata=self.env_session.environment_metadata,
                    local_memory_snapshot=self.local_memory.snapshot(),
                    step_index=step_index,
                )
                # Stage 2 stays solver-agnostic here; movement and click solver heads
                # both implement the same policy decision surface.
                decision = self.policy.decide(parsed_state)
                executed_action = self._select_executed_action(decision)
                transition, step_result = self.env_session.step(executed_action)
                memory_update = self._build_memory_update(parsed_state, executed_action, step_result, transition)
                self.local_memory.apply_update(memory_update)
                stop_status = evaluate_stop_conditions(
                    step_result=step_result,
                    next_step_index=self.env_session.step_index,
                    max_steps=self.max_steps,
                )
                self.ledger.append(
                    StepLedgerRecordV4(
                        step_index=step_index,
                        pre_observation=current_observation,
                        parsed_state_summary=parsed_state.to_dict(),
                        decision_summary=decision.to_dict(),
                        executed_action=executed_action,
                        transition_record=transition,
                        step_result=step_result,
                        memory_update_summary=memory_update.to_dict(),
                        failure_bucket=None,
                        stop_condition_status=stop_status.to_dict(),
                    )
                )
                previous_observation = current_observation
                current_observation = transition.post_observation
            except Exception as exc:
                failure_bucket = self._bucket_for_exception(exc, parsed_state, decision, executed_action, transition, step_result, memory_update)
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
                        parsed_state_summary={} if parsed_state is None else parsed_state.to_dict(),
                        decision_summary={} if decision is None else decision.to_dict(),
                        executed_action=executed_action,
                        transition_record=transition,
                        step_result=step_result,
                        memory_update_summary={} if memory_update is None else memory_update.to_dict(),
                        failure_bucket=failure_bucket,
                        stop_condition_status=stop_status.to_dict(),
                    )
                )
        return self.ledger.build_summary()

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
    ) -> str:
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
