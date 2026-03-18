from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


EVENT_PAYLOAD_TYPES = {
    "round start": {"round_start"},
    "probe plan selected": {"plan_selected"},
    "probe episode executed": {"episode_executed"},
    "probe analysis completed": {"analysis_completed"},
    "probe blackboard merge completed": {"merge_completed"},
    "probe mechanic graph merge completed": {"mechanic_graph_merge_completed"},
    "deterministic hypotheses generated": {"hypothesis_generation"},
    "llm hypotheses generated": {"hypothesis_generation"},
    "hypothesis validation completed": {"hypothesis_generation"},
    "llm call skipped": {"llm_operation"},
    "llm call attempted": {"llm_operation"},
    "llm call failed": {"llm_operation"},
    "llm call succeeded": {"llm_operation"},
    "probe memory reconcile completed": {"memory_reconcile_completed"},
    "directed plan selected": {"plan_selected"},
    "directed episode executed": {"episode_executed"},
    "directed analysis completed": {"analysis_completed"},
    "directed blackboard merge completed": {"merge_completed"},
    "directed mechanic graph merge completed": {"mechanic_graph_merge_completed"},
    "directed memory reconcile completed": {"memory_reconcile_completed"},
    "subgoal chain started": {"subgoal_chain_started"},
    "subgoal chain step completed": {"subgoal_chain_step"},
    "subgoal chain step failed": {"subgoal_chain_step"},
    "subgoal chain advanced": {"subgoal_chain_advanced"},
    "subgoal chain aborted": {"subgoal_chain_aborted"},
    "subgoal chain completed": {"subgoal_chain_completed"},
    "durable flush requested": {"durable_flush"},
    "durable flush completed": {"durable_flush"},
    "stop decision made": {"stop_decision"},
}


@dataclass(frozen=True)
class SessionLedgerRecord:
    session_id: str
    round_id: int
    pass_id: int
    event_type: str
    blackboard_version: str | None
    memory_version: str | None
    plan_context_id: str | None
    episode_id: str | None
    decision_id: str | None
    outcome_id: str | None
    timestamp: str
    payload_type: str
    payload_version: str
    payload_schema_name: str
    payload_schema_version: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RoundStartPayload:
    game_id: str


@dataclass(frozen=True)
class PlanSelectedPayload:
    selected_candidate_id: str | None
    selected_candidate_count: int = 0
    planner_contract_mode: str = "split_world_native"
    strict_blackboard_snapshot_ref: dict[str, Any] = field(default_factory=dict)
    strict_memory_snapshot_ref: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeExecutedPayload:
    termination_reason: str | None
    mode: str
    reward_delta: float | None = None
    outcome_evidence_provenance_summary: dict[str, Any] = field(default_factory=dict)
    avatar_cell: list[int] | None = None
    avatar_confidence: float = 0.0
    avatar_source: str = "unknown"
    avatar_ambiguous: bool = False
    avatar_tracker_status: str = "unknown"


@dataclass(frozen=True)
class AnalysisCompletedPayload:
    analysis_mode: str
    delta_count: int
    strict_blackboard_snapshot_ref: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MergeCompletedPayload:
    material_change: bool
    strict_blackboard_snapshot_ref: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MechanicGraphMergeCompletedPayload:
    mechanic_graph_version_before: str | None
    mechanic_graph_version_after: str | None
    node_count_added: int = 0
    edge_count_added: int = 0
    observed_edge_count_added: int = 0
    hypothesized_edge_count_added: int = 0
    top_supported_new_relations_summary: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class HypothesisGenerationPayload:
    proposal_count: int
    edge_proposal_count: int
    path_proposal_count: int
    test_proposal_count: int
    top_confidence: float
    top_support_count: int
    contradicted_count: int
    validated_count: int
    source_provenance: str


@dataclass(frozen=True)
class LLMOperationPayload:
    gating_reason: str
    provider_name: str
    model_name: str
    latency_ms: int
    proposal_count: int
    prompt_char_count: int = 0
    prompt_approx_token_count: int = 0
    prompt_trim_applied: bool = False
    prompt_mode: str = ""
    query_target_id: str = ""
    skip_reason: str = ""
    temperature: float = 0.0
    top_p: float = 0.0
    top_k: int = 0
    presence_penalty: float = 0.0
    repetition_penalty: float = 0.0
    max_output_tokens: int = 0
    enable_thinking: bool = False
    stream: bool = False
    error_code: str | None = None


@dataclass(frozen=True)
class MemoryReconcilePayload:
    memory_snapshot_handle: str
    strict_memory_snapshot_ref: dict[str, Any] = field(default_factory=dict)
    durable_eligibility_summary: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SubgoalChainStartedPayload:
    chain_id: str
    current_step_id: str | None = None
    step_kind: str | None = None
    expected_evidence: tuple[str, ...] = ()
    observed_evidence: tuple[str, ...] = ()
    failure_reason: str | None = None
    advancement_reason: str | None = None


@dataclass(frozen=True)
class SubgoalChainStepPayload:
    chain_id: str
    current_step_id: str | None = None
    step_kind: str | None = None
    expected_evidence: tuple[str, ...] = ()
    observed_evidence: tuple[str, ...] = ()
    failure_reason: str | None = None
    advancement_reason: str | None = None


@dataclass(frozen=True)
class SubgoalChainAdvancedPayload:
    chain_id: str
    current_step_id: str | None = None
    step_kind: str | None = None
    expected_evidence: tuple[str, ...] = ()
    observed_evidence: tuple[str, ...] = ()
    failure_reason: str | None = None
    advancement_reason: str | None = None


@dataclass(frozen=True)
class SubgoalChainAbortedPayload:
    chain_id: str
    current_step_id: str | None = None
    step_kind: str | None = None
    expected_evidence: tuple[str, ...] = ()
    observed_evidence: tuple[str, ...] = ()
    failure_reason: str | None = None
    advancement_reason: str | None = None


@dataclass(frozen=True)
class SubgoalChainCompletedPayload:
    chain_id: str
    current_step_id: str | None = None
    step_kind: str | None = None
    expected_evidence: tuple[str, ...] = ()
    observed_evidence: tuple[str, ...] = ()
    failure_reason: str | None = None
    advancement_reason: str | None = None


@dataclass(frozen=True)
class DurableFlushPayload:
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StopDecisionPayload:
    stop_reason: str
    won: bool = False
    round_progress: float = 0.0
    termination_reason: str | None = None


@dataclass
class SessionLedger:
    session_id: str
    records: list[SessionLedgerRecord] = field(default_factory=list)

    def append(
        self,
        *,
        round_id: int,
        pass_id: int,
        event_type: str,
        blackboard_version: str | None = None,
        memory_version: str | None = None,
        plan_context_id: str | None = None,
        episode_id: str | None = None,
        decision_id: str | None = None,
        outcome_id: str | None = None,
        payload: dict[str, Any] | object | None = None,
    ) -> SessionLedgerRecord:
        payload_type, payload_version, payload_schema_name, payload_schema_version, payload_dict = self._serialize_payload(event_type, payload)
        record = SessionLedgerRecord(
            session_id=self.session_id,
            round_id=int(round_id),
            pass_id=int(pass_id),
            event_type=str(event_type),
            blackboard_version=blackboard_version,
            memory_version=memory_version,
            plan_context_id=plan_context_id,
            episode_id=episode_id,
            decision_id=decision_id,
            outcome_id=outcome_id,
            timestamp=_utc_now(),
            payload_type=payload_type,
            payload_version=payload_version,
            payload_schema_name=payload_schema_name,
            payload_schema_version=payload_schema_version,
            payload=payload_dict,
        )
        self.records.append(record)
        return record

    def _serialize_payload(self, event_type: str, payload) -> tuple[str, str, str, str, dict[str, Any]]:
        payload_specs = {
            RoundStartPayload: ("round_start", "v1", "round_start_payload", "v1"),
            PlanSelectedPayload: ("plan_selected", "v1", "plan_selected_payload", "v1"),
            EpisodeExecutedPayload: ("episode_executed", "v1", "episode_executed_payload", "v1"),
            AnalysisCompletedPayload: ("analysis_completed", "v1", "analysis_completed_payload", "v1"),
            MergeCompletedPayload: ("merge_completed", "v1", "merge_completed_payload", "v1"),
            MechanicGraphMergeCompletedPayload: ("mechanic_graph_merge_completed", "v1", "mechanic_graph_merge_payload", "v1"),
            HypothesisGenerationPayload: ("hypothesis_generation", "v1", "hypothesis_generation_payload", "v1"),
            LLMOperationPayload: ("llm_operation", "v1", "llm_operation_payload", "v1"),
            MemoryReconcilePayload: ("memory_reconcile_completed", "v1", "memory_reconcile_payload", "v1"),
            SubgoalChainStartedPayload: ("subgoal_chain_started", "v1", "subgoal_chain_started_payload", "v1"),
            SubgoalChainStepPayload: ("subgoal_chain_step", "v1", "subgoal_chain_step_payload", "v1"),
            SubgoalChainAdvancedPayload: ("subgoal_chain_advanced", "v1", "subgoal_chain_advanced_payload", "v1"),
            SubgoalChainAbortedPayload: ("subgoal_chain_aborted", "v1", "subgoal_chain_aborted_payload", "v1"),
            SubgoalChainCompletedPayload: ("subgoal_chain_completed", "v1", "subgoal_chain_completed_payload", "v1"),
            DurableFlushPayload: ("durable_flush", "v1", "durable_flush_payload", "v1"),
            StopDecisionPayload: ("stop_decision", "v1", "stop_decision_payload", "v1"),
        }
        if payload is None:
            allowed = EVENT_PAYLOAD_TYPES.get(str(event_type), set())
            if "empty" not in allowed and allowed:
                raise ValueError(f"event type {event_type!r} requires a payload")
            return "empty", "v1", "empty_payload", "v1", {}
        if is_dataclass(payload):
            spec = payload_specs.get(type(payload))
            if spec is None:
                raise ValueError(f"unknown ledger payload dataclass: {type(payload)!r}")
            payload_type, payload_version, payload_schema_name, payload_schema_version = spec
            if payload_type not in EVENT_PAYLOAD_TYPES.get(str(event_type), set()):
                raise ValueError(f"payload type {payload_type!r} not allowed for event {event_type!r}")
            payload_dict = asdict(payload)
            self._validate_payload(payload, payload_dict)
            return payload_type, payload_version, payload_schema_name, payload_schema_version, payload_dict
        raise ValueError(f"ledger payload must be a known dataclass payload, got {type(payload)!r}")

    def _validate_payload(self, payload, payload_dict: dict[str, Any]) -> None:
        expected_fields = {field.name for field in fields(type(payload))}
        if set(payload_dict.keys()) != expected_fields:
            raise ValueError(f"invalid ledger payload shape for {type(payload).__name__}")

    def deserialize_payload(self, record: SessionLedgerRecord) -> dict[str, Any]:
        payload_types = {
            "round_start": (RoundStartPayload, "v1", "round_start_payload", "v1"),
            "plan_selected": (PlanSelectedPayload, "v1", "plan_selected_payload", "v1"),
            "episode_executed": (EpisodeExecutedPayload, "v1", "episode_executed_payload", "v1"),
            "analysis_completed": (AnalysisCompletedPayload, "v1", "analysis_completed_payload", "v1"),
            "merge_completed": (MergeCompletedPayload, "v1", "merge_completed_payload", "v1"),
            "mechanic_graph_merge_completed": (MechanicGraphMergeCompletedPayload, "v1", "mechanic_graph_merge_payload", "v1"),
            "hypothesis_generation": (HypothesisGenerationPayload, "v1", "hypothesis_generation_payload", "v1"),
            "llm_operation": (LLMOperationPayload, "v1", "llm_operation_payload", "v1"),
            "memory_reconcile_completed": (MemoryReconcilePayload, "v1", "memory_reconcile_payload", "v1"),
            "subgoal_chain_started": (SubgoalChainStartedPayload, "v1", "subgoal_chain_started_payload", "v1"),
            "subgoal_chain_step": (SubgoalChainStepPayload, "v1", "subgoal_chain_step_payload", "v1"),
            "subgoal_chain_advanced": (SubgoalChainAdvancedPayload, "v1", "subgoal_chain_advanced_payload", "v1"),
            "subgoal_chain_aborted": (SubgoalChainAbortedPayload, "v1", "subgoal_chain_aborted_payload", "v1"),
            "subgoal_chain_completed": (SubgoalChainCompletedPayload, "v1", "subgoal_chain_completed_payload", "v1"),
            "durable_flush": (DurableFlushPayload, "v1", "durable_flush_payload", "v1"),
            "stop_decision": (StopDecisionPayload, "v1", "stop_decision_payload", "v1"),
            "empty": (None, "v1", "empty_payload", "v1"),
        }
        payload_spec = payload_types.get(str(record.payload_type))
        if str(record.payload_type) == "empty":
            return {}
        if payload_spec is None:
            raise ValueError(f"unknown ledger payload structure: {record.payload_type}/{record.payload_version}")
        if str(record.event_type) not in EVENT_PAYLOAD_TYPES or str(record.payload_type) not in EVENT_PAYLOAD_TYPES.get(str(record.event_type), set()):
            raise ValueError(f"mismatched event/payload combination: {record.event_type}/{record.payload_type}")
        payload_class, allowed_version, schema_name, schema_version = payload_spec
        if str(record.payload_version) != allowed_version or str(record.payload_schema_name) != schema_name or str(record.payload_schema_version) != schema_version:
            raise ValueError(f"unknown ledger payload structure: {record.payload_type}/{record.payload_version}/{record.payload_schema_name}/{record.payload_schema_version}")
        self._validate_payload(payload_class(**dict(record.payload)), dict(record.payload))
        return dict(record.payload)

    def append_round_start(self, **kwargs) -> SessionLedgerRecord:
        return self.append(event_type="round start", **kwargs)

    def append_probe_plan_selected(self, **kwargs) -> SessionLedgerRecord:
        return self.append(event_type="probe plan selected", **kwargs)

    def append_probe_episode_executed(self, **kwargs) -> SessionLedgerRecord:
        return self.append(event_type="probe episode executed", **kwargs)

    def append_probe_analysis_completed(self, **kwargs) -> SessionLedgerRecord:
        return self.append(event_type="probe analysis completed", **kwargs)

    def append_probe_blackboard_merge_completed(self, **kwargs) -> SessionLedgerRecord:
        return self.append(event_type="probe blackboard merge completed", **kwargs)

    def append_probe_memory_reconcile_completed(self, **kwargs) -> SessionLedgerRecord:
        return self.append(event_type="probe memory reconcile completed", **kwargs)

    def append_probe_mechanic_graph_merge_completed(self, **kwargs) -> SessionLedgerRecord:
        return self.append(event_type="probe mechanic graph merge completed", **kwargs)

    def append_deterministic_hypotheses_generated(self, **kwargs) -> SessionLedgerRecord:
        return self.append(event_type="deterministic hypotheses generated", **kwargs)

    def append_llm_hypotheses_generated(self, **kwargs) -> SessionLedgerRecord:
        return self.append(event_type="llm hypotheses generated", **kwargs)

    def append_hypothesis_validation_completed(self, **kwargs) -> SessionLedgerRecord:
        return self.append(event_type="hypothesis validation completed", **kwargs)

    def append_directed_plan_selected(self, **kwargs) -> SessionLedgerRecord:
        return self.append(event_type="directed plan selected", **kwargs)

    def append_directed_episode_executed(self, **kwargs) -> SessionLedgerRecord:
        return self.append(event_type="directed episode executed", **kwargs)

    def append_directed_analysis_completed(self, **kwargs) -> SessionLedgerRecord:
        return self.append(event_type="directed analysis completed", **kwargs)

    def append_directed_blackboard_merge_completed(self, **kwargs) -> SessionLedgerRecord:
        return self.append(event_type="directed blackboard merge completed", **kwargs)

    def append_directed_memory_reconcile_completed(self, **kwargs) -> SessionLedgerRecord:
        return self.append(event_type="directed memory reconcile completed", **kwargs)

    def append_directed_mechanic_graph_merge_completed(self, **kwargs) -> SessionLedgerRecord:
        return self.append(event_type="directed mechanic graph merge completed", **kwargs)

    def append_llm_call_skipped(self, **kwargs) -> SessionLedgerRecord:
        return self.append(event_type="llm call skipped", **kwargs)

    def append_llm_call_attempted(self, **kwargs) -> SessionLedgerRecord:
        return self.append(event_type="llm call attempted", **kwargs)

    def append_llm_call_failed(self, **kwargs) -> SessionLedgerRecord:
        return self.append(event_type="llm call failed", **kwargs)

    def append_llm_call_succeeded(self, **kwargs) -> SessionLedgerRecord:
        return self.append(event_type="llm call succeeded", **kwargs)

    def append_durable_flush_requested(self, **kwargs) -> SessionLedgerRecord:
        return self.append(event_type="durable flush requested", **kwargs)

    def append_subgoal_chain_started(self, **kwargs) -> SessionLedgerRecord:
        return self.append(event_type="subgoal chain started", **kwargs)

    def append_subgoal_chain_step_completed(self, **kwargs) -> SessionLedgerRecord:
        return self.append(event_type="subgoal chain step completed", **kwargs)

    def append_subgoal_chain_step_failed(self, **kwargs) -> SessionLedgerRecord:
        return self.append(event_type="subgoal chain step failed", **kwargs)

    def append_subgoal_chain_advanced(self, **kwargs) -> SessionLedgerRecord:
        return self.append(event_type="subgoal chain advanced", **kwargs)

    def append_subgoal_chain_aborted(self, **kwargs) -> SessionLedgerRecord:
        return self.append(event_type="subgoal chain aborted", **kwargs)

    def append_subgoal_chain_completed(self, **kwargs) -> SessionLedgerRecord:
        return self.append(event_type="subgoal chain completed", **kwargs)

    def append_durable_flush_completed(self, **kwargs) -> SessionLedgerRecord:
        return self.append(event_type="durable flush completed", **kwargs)

    def append_stop_decision_made(self, **kwargs) -> SessionLedgerRecord:
        return self.append(event_type="stop decision made", **kwargs)

    def to_dicts(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self.records]
