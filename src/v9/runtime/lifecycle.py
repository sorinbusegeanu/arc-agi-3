from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, replace
from typing import Any

from v9.memory.identity import MemoryUid
from v9.memory.model import CognitiveState, MemoryLevel
from v9.telemetry import ConsolidationSample

from .publication import node_ref


@dataclass(frozen=True, slots=True)
class LifecycleRecord:
    uid: MemoryUid
    state: CognitiveState = CognitiveState.CANDIDATE
    support: int = 0
    relevant_opportunities: int = 0
    last_transition_watermark: int = 0
    last_observed_watermark: int = 0
    last_transition_cycle: int = 0
    last_observed_cycle: int = 0
    retention_score: float = 1.0
    replacement_uid: MemoryUid | None = None
    baseline_hgt_retention: float | None = None
    baseline_behavioral_success: float | None = None
    baseline_transfer_quality: float | None = None
    baseline_prediction_quality: float | None = None


class LifecycleRegistry:
    SCHEMA_VERSION = 2

    def __init__(self) -> None:
        self.records: dict[MemoryUid, LifecycleRecord] = {}
        self.transitions = 0
        self.maintenance_cycle = 0

    def begin_maintenance(self) -> int:
        self.maintenance_cycle += 1
        return self.maintenance_cycle

    def _new_record(self, uid: MemoryUid) -> LifecycleRecord:
        return LifecycleRecord(
            uid=uid,
            last_transition_cycle=self.maintenance_cycle,
            last_observed_cycle=self.maintenance_cycle,
        )

    def observe(self, uid: MemoryUid, *, support_delta: int, relevant_opportunity: bool, watermark: int) -> LifecycleRecord:
        current = self.records.get(uid, self._new_record(uid))
        support = current.support + int(support_delta)
        opportunities = current.relevant_opportunities + int(relevant_opportunity)
        state = current.state
        reactivated = support_delta > 0 and state in {CognitiveState.DORMANT, CognitiveState.RETIRE_PENDING, CognitiveState.RETIRED}
        if reactivated:
            state = CognitiveState.REACTIVATED
        elif support > 0 and state in {CognitiveState.CANDIDATE, CognitiveState.PROBATION, CognitiveState.REACTIVATED}:
            state = CognitiveState.ACTIVE
        transitioned = state is not current.state
        self.transitions += int(transitioned)
        row = replace(
            current,
            state=state,
            support=support,
            relevant_opportunities=opportunities,
            last_transition_watermark=int(watermark) if transitioned else current.last_transition_watermark,
            last_observed_watermark=int(watermark),
            last_transition_cycle=self.maintenance_cycle if transitioned else current.last_transition_cycle,
            last_observed_cycle=self.maintenance_cycle,
            replacement_uid=None if state in {CognitiveState.ACTIVE, CognitiveState.REACTIVATED} else current.replacement_uid,
            baseline_hgt_retention=None if reactivated else current.baseline_hgt_retention,
            baseline_behavioral_success=None if reactivated else current.baseline_behavioral_success,
            baseline_transfer_quality=None if reactivated else current.baseline_transfer_quality,
            baseline_prediction_quality=None if reactivated else current.baseline_prediction_quality,
        )
        self.records[uid] = row
        return row

    def transition(
        self,
        uid: MemoryUid,
        state: CognitiveState,
        *,
        watermark: int,
        retention_score: float | None = None,
        replacement_uid: MemoryUid | None = None,
        validation_baseline: tuple[float | None, float | None, float | None, float | None] | None = None,
    ) -> LifecycleRecord:
        current = self.records.get(uid, self._new_record(uid))
        transitioned = state is not current.state
        self.transitions += int(transitioned)
        baseline = validation_baseline or (
            current.baseline_hgt_retention,
            current.baseline_behavioral_success,
            current.baseline_transfer_quality,
            current.baseline_prediction_quality,
        )
        row = replace(
            current,
            state=state,
            last_transition_watermark=int(watermark) if transitioned else current.last_transition_watermark,
            last_transition_cycle=self.maintenance_cycle if transitioned else current.last_transition_cycle,
            retention_score=current.retention_score if retention_score is None else float(retention_score),
            replacement_uid=replacement_uid,
            baseline_hgt_retention=baseline[0],
            baseline_behavioral_success=baseline[1],
            baseline_transfer_quality=baseline[2],
            baseline_prediction_quality=baseline[3],
        )
        self.records[uid] = row
        return row

    def retire_if_exhausted(self, uid: MemoryUid, *, required_opportunities: int, watermark: int, has_authoritative_dependency: bool = False, has_provenance_obligation: bool = False) -> LifecycleRecord:
        current = self.records[uid]
        state = current.state
        if current.support <= 0 and current.relevant_opportunities >= required_opportunities:
            state = CognitiveState.RETIRE_PENDING if has_authoritative_dependency or has_provenance_obligation else CognitiveState.RETIRED
        return self.transition(uid, state, watermark=watermark, retention_score=current.retention_score, replacement_uid=current.replacement_uid)

    def reactivate(self, uid: MemoryUid, *, support_delta: int, watermark: int) -> LifecycleRecord:
        if support_delta <= 0:
            raise ValueError("reactivation requires positive new support")
        current = self.records[uid]
        self.transitions += int(current.state is not CognitiveState.REACTIVATED)
        row = replace(
            current,
            state=CognitiveState.REACTIVATED,
            support=current.support + int(support_delta),
            last_transition_watermark=int(watermark),
            last_observed_watermark=int(watermark),
            last_transition_cycle=self.maintenance_cycle,
            last_observed_cycle=self.maintenance_cycle,
            replacement_uid=None,
            baseline_hgt_retention=None,
            baseline_behavioral_success=None,
            baseline_transfer_quality=None,
            baseline_prediction_quality=None,
        )
        self.records[uid] = row
        return row

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "transitions": self.transitions,
            "maintenance_cycle": self.maintenance_cycle,
            "records": [
                {
                    "uid": [uid.hi, uid.lo],
                    "state": row.state.name,
                    "support": row.support,
                    "relevant_opportunities": row.relevant_opportunities,
                    "last_transition_watermark": row.last_transition_watermark,
                    "last_observed_watermark": row.last_observed_watermark,
                    "last_transition_cycle": row.last_transition_cycle,
                    "last_observed_cycle": row.last_observed_cycle,
                    "retention_score": row.retention_score,
                    "replacement_uid": None if row.replacement_uid is None else [row.replacement_uid.hi, row.replacement_uid.lo],
                    "baseline_hgt_retention": row.baseline_hgt_retention,
                    "baseline_behavioral_success": row.baseline_behavioral_success,
                    "baseline_transfer_quality": row.baseline_transfer_quality,
                    "baseline_prediction_quality": row.baseline_prediction_quality,
                }
                for uid, row in sorted(self.records.items())
            ],
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "LifecycleRegistry":
        result = cls()
        schema_version = int(state.get("schema_version", 1))
        legacy_watermark_aging = schema_version < cls.SCHEMA_VERSION or "maintenance_cycle" not in state
        result.transitions = int(state.get("transitions", 0))
        result.maintenance_cycle = int(state.get("maintenance_cycle", 0))
        for raw in state.get("records", []):
            uid = MemoryUid(int(raw["uid"][0]), int(raw["uid"][1]))
            transition_watermark = int(raw.get("last_transition_watermark", 0))
            raw_replacement = raw.get("replacement_uid")
            replacement_uid = None if raw_replacement is None else MemoryUid(int(raw_replacement[0]), int(raw_replacement[1]))
            restored_state = CognitiveState[str(raw["state"])]
            if legacy_watermark_aging and restored_state in {CognitiveState.DORMANT, CognitiveState.RETIRE_PENDING}:
                restored_state = CognitiveState.ACTIVE
                replacement_uid = None
            result.records[uid] = LifecycleRecord(
                uid=uid,
                state=restored_state,
                support=int(raw.get("support", 0)),
                relevant_opportunities=int(raw.get("relevant_opportunities", 0)),
                last_transition_watermark=transition_watermark,
                last_observed_watermark=int(raw.get("last_observed_watermark", transition_watermark)),
                last_transition_cycle=int(raw.get("last_transition_cycle", result.maintenance_cycle)),
                last_observed_cycle=int(raw.get("last_observed_cycle", result.maintenance_cycle)),
                retention_score=float(raw.get("retention_score", 1.0)),
                replacement_uid=replacement_uid,
                baseline_hgt_retention=None if legacy_watermark_aging or raw.get("baseline_hgt_retention") is None else float(raw["baseline_hgt_retention"]),
                baseline_behavioral_success=None if legacy_watermark_aging or raw.get("baseline_behavioral_success") is None else float(raw["baseline_behavioral_success"]),
                baseline_transfer_quality=None if legacy_watermark_aging or raw.get("baseline_transfer_quality") is None else float(raw["baseline_transfer_quality"]),
                baseline_prediction_quality=None if legacy_watermark_aging or raw.get("baseline_prediction_quality") is None else float(raw["baseline_prediction_quality"]),
            )
        return result


def retention_score(record: LifecycleRecord, payload: dict[str, Any], *, watermark: int) -> float:
    observed_support = max(
        int(record.support),
        int(payload.get("support", 0) or 0),
        int(payload.get("recurrence", 0) or 0),
        int(payload.get("reliability_trials", 0) or 0),
    )
    support = 1.0 - math.exp(-max(0, observed_support) / 4.0)
    age = max(0, int(watermark) - max(int(record.last_observed_watermark), int(payload.get("created_watermark", 0) or 0)))
    recency = math.exp(-age / 8192.0)
    surprise = min(
        1.0,
        max(
            abs(float(payload.get("primary_valence", 0) or 0)),
            abs(float(payload.get("prediction_error", 0.0) or 0.0)),
            min(1.0, abs(float(payload.get("future_option_delta", 0.0) or 0.0)) / 8.0),
        ),
    )
    explanatory = min(1.0, max(0.0, float(payload.get("explanatory_reach", 0) or 0) / 8.0))
    held_out = payload.get("held_out_targets", ())
    transfer = min(1.0, len(held_out) / 2.0) if isinstance(held_out, (list, tuple, set)) else 0.0
    if bool(payload.get("validated", False)):
        transfer = max(transfer, 1.0)
    return max(0.0, min(1.0, 0.35 * support + 0.25 * recency + 0.15 * surprise + 0.15 * explanatory + 0.10 * transfer))


def _validation_metrics(runtime: Any) -> tuple[float | None, float | None, float | None, float | None]:
    gauges = getattr(getattr(runtime, "unified_telemetry", None), "gauges", {})
    hgt = None if "historical_retention" not in gauges else float(gauges["historical_retention"])
    behavior = None if "behavioral_success_rate" not in gauges else float(gauges["behavioral_success_rate"])
    transfer_records = getattr(getattr(runtime, "transfer_trust", None), "records", {})
    transfer = None
    if transfer_records:
        transfer = sum(int(getattr(row, "successes", 0) > 0) for row in transfer_records.values()) / len(transfer_records)
    prediction_count = int(getattr(runtime, "telemetry", {}).get("symbol_conditioned_prediction_observations", 0))
    prediction = None
    if prediction_count > 0:
        prediction = float(getattr(runtime, "_symbol_prediction_delta_sum", 0.0)) / prediction_count
    return hgt, behavior, transfer, prediction


def _validation_preserved(row: LifecycleRecord, runtime: Any, *, tolerance: float = 0.05) -> bool:
    current = _validation_metrics(runtime)
    baseline = (
        row.baseline_hgt_retention,
        row.baseline_behavioral_success,
        row.baseline_transfer_quality,
        row.baseline_prediction_quality,
    )
    return all(
        before is None or after is None or float(after) >= float(before) - float(tolerance)
        for before, after in zip(baseline, current)
    )


def _sync_cognitive_visibility(graph: Any, updates: dict[MemoryUid, CognitiveState]) -> None:
    if not updates:
        return
    changed: list[MemoryUid] = []
    with graph._publication_lock:
        for uid, state in updates.items():
            payload = graph.payloads.get(uid)
            if payload is None:
                continue
            before = (payload.get("cognitive_state"), payload.get("cognitive_state_version"))
            if state in {CognitiveState.ACTIVE, CognitiveState.REACTIVATED, CognitiveState.VALIDATED}:
                payload.pop("cognitive_state", None)
                payload.pop("cognitive_state_version", None)
            else:
                payload["cognitive_state"] = state.name
                payload["cognitive_state_version"] = 2
            after = (payload.get("cognitive_state"), payload.get("cognitive_state_version"))
            if before != after:
                changed.append(uid)
        if changed:
            for uid in changed:
                graph.versions.bump(node_ref(uid))
            graph.generation += 1
            graph._cached_read_view = None


def run_lifecycle_maintenance(
    runtime: Any,
    *,
    scan_limit: int = 8192,
    retirement_limit: int = 2048,
    dormant_threshold: float = 0.22,
    reactivation_threshold: float = 0.40,
    dormancy_grace_cycles: int = 3,
    retirement_grace_cycles: int = 3,
    validation_tolerance: float = 0.05,
) -> dict[str, int | float]:
    if not runtime.config.enable_lifecycle:
        return {"cycle": 0, "scanned": 0, "dormant": 0, "pending": 0, "retired": 0, "reactivated": 0, "blocked": 0, "pressure": 0.0}
    if min(int(dormancy_grace_cycles), int(retirement_grace_cycles)) <= 0:
        raise ValueError("lifecycle grace cycles must be positive")

    graph = runtime.graph
    registry: LifecycleRegistry = runtime.lifecycle
    cycle = registry.begin_maintenance()
    watermark = int(runtime.watermark)
    pressure = float(graph.pressure_ratio())
    pressure_multiplier = min(4, 1 + max(0, int((pressure - 0.70) * 10)))
    effective_scan_limit = max(1, int(scan_limit)) * pressure_multiplier
    effective_retirement_limit = max(1, int(retirement_limit)) * pressure_multiplier
    replacements = graph.provenance_replacements(maximum_level=MemoryLevel.M1)
    blocked = 0

    pending_plans: list[tuple[MemoryUid, MemoryUid, str]] = []
    for uid, row in registry.records.items():
        if len(pending_plans) >= effective_retirement_limit:
            break
        if row.state is not CognitiveState.RETIRE_PENDING:
            continue
        node = graph.nodes.get(uid)
        replacement_uid = replacements.get(uid)
        if node is None or replacement_uid is None or node.level > MemoryLevel.M1:
            continue
        if cycle - int(row.last_transition_cycle) < int(retirement_grace_cycles):
            continue
        if uid in runtime._replay_pool or not _validation_preserved(row, runtime, tolerance=validation_tolerance):
            blocked += 1
            continue
        pending_plans.append((uid, replacement_uid, "retention_compaction"))

    replay_before = len(runtime._replay_pool)
    retired_records = {uid: registry.records[uid] for uid, _, _ in pending_plans if uid in registry.records}
    retired_uids = graph.retire_nodes_batch(tuple(pending_plans))
    retired_set = set(retired_uids)
    for uid in retired_uids:
        row = registry.records.get(uid, registry._new_record(uid))
        registry.transition(uid, CognitiveState.RETIRED, watermark=watermark, retention_score=row.retention_score, replacement_uid=row.replacement_uid or replacements.get(uid))
        runtime._replay_pool.pop(uid, None)
        runtime._deferred_base_nodes.pop(uid, None)
    if retired_set:
        runtime._latest_interaction_grounding = {
            key: row for key, row in runtime._latest_interaction_grounding.items() if row.uid not in retired_set
        }

    protected_states = {CognitiveState.CANDIDATE, CognitiveState.PROBATION, CognitiveState.VALIDATED, CognitiveState.QUARANTINED, CognitiveState.RETIRED, CognitiveState.RETIRE_PENDING}
    eligible = (
        row
        for uid, row in registry.records.items()
        if uid in graph.nodes and row.state not in protected_states
    )
    candidates = heapq.nsmallest(
        effective_scan_limit,
        eligible,
        key=lambda row: (0 if row.state is CognitiveState.DORMANT else 1, int(row.last_observed_cycle), int(row.last_observed_watermark), row.uid),
    )

    dormant = 0
    pending = 0
    reactivated = 0
    visibility_updates: dict[MemoryUid, CognitiveState] = {}
    current_validation = _validation_metrics(runtime)
    for row in candidates:
        node = graph.nodes.get(row.uid)
        if node is None:
            continue
        payload = graph.payloads.get(row.uid, {})
        score = retention_score(row, payload, watermark=watermark)
        idle_cycles = cycle - int(row.last_observed_cycle)
        replacement_uid = replacements.get(row.uid)
        covered = replacement_uid is not None
        if row.state is CognitiveState.DORMANT:
            if score >= reactivation_threshold:
                state = CognitiveState.REACTIVATED
                registry.transition(row.uid, state, watermark=watermark, retention_score=score)
                reactivated += 1
            elif node.level <= MemoryLevel.M1 and covered and cycle - int(row.last_transition_cycle) >= int(dormancy_grace_cycles):
                if row.uid in runtime._replay_pool or not _validation_preserved(row, runtime, tolerance=validation_tolerance):
                    state = CognitiveState.DORMANT
                    registry.transition(row.uid, state, watermark=watermark, retention_score=score, replacement_uid=replacement_uid)
                    blocked += 1
                else:
                    state = CognitiveState.RETIRE_PENDING
                    registry.transition(row.uid, state, watermark=watermark, retention_score=score, replacement_uid=replacement_uid)
                    pending += 1
            else:
                state = CognitiveState.DORMANT
                registry.transition(row.uid, state, watermark=watermark, retention_score=score, replacement_uid=replacement_uid)
            visibility_updates[row.uid] = state
            continue
        can_dormant = covered if node.level <= MemoryLevel.M1 else True
        if can_dormant and idle_cycles >= int(dormancy_grace_cycles) and score < dormant_threshold:
            state = CognitiveState.DORMANT
            registry.transition(
                row.uid,
                state,
                watermark=watermark,
                retention_score=score,
                replacement_uid=replacement_uid,
                validation_baseline=current_validation,
            )
            dormant += 1
        else:
            state = row.state
            registry.transition(row.uid, state, watermark=watermark, retention_score=score)
        visibility_updates[row.uid] = state

    _sync_cognitive_visibility(graph, visibility_updates)

    replay_after = len(runtime._replay_pool)
    if retired_uids or reactivated:
        after_hgt = _validation_metrics(runtime)[0]
        before_hgt_values = [retired_records[uid].baseline_hgt_retention for uid in retired_uids if uid in retired_records and retired_records[uid].baseline_hgt_retention is not None]
        before_hgt = sum(before_hgt_values) / len(before_hgt_values) if before_hgt_values else (after_hgt if after_hgt is not None else 1.0)
        runtime.record_hgt_consolidation(
            ConsolidationSample(
                hydra_bytes_retired=len(retired_uids) * 192,
                hydra_nodes_retired=len(retired_uids),
                hydra_nodes_replaced_by_abstractions=len(retired_uids),
                replay_examples_before=replay_before,
                replay_examples_after=replay_after,
                representative_retention_ratio=1.0 if retired_uids else 0.0,
                hgt_retention_before=float(before_hgt),
                hgt_retention_after=float(after_hgt if after_hgt is not None else before_hgt),
                reactivation_examples=reactivated,
            )
        )

    return {
        "cycle": cycle,
        "scanned": len(candidates),
        "dormant": dormant,
        "pending": pending,
        "retired": len(retired_uids),
        "reactivated": reactivated,
        "blocked": blocked,
        "pressure": pressure,
    }
