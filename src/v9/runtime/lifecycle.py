from __future__ import annotations

from collections import deque
import math
from dataclasses import dataclass, replace
from typing import Any

from v9.memory.identity import MemoryUid
from v9.memory.model import CognitiveState, MemoryLevel, MemoryType
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
    SCHEMA_VERSION = 3

    def __init__(self) -> None:
        self.records: dict[MemoryUid, LifecycleRecord] = {}
        self.transitions = 0
        self.maintenance_cycle = 0
        self._scan_order: list[MemoryUid] = []
        self._scan_cursor = 0
        self._stale_scan_entries = 0
        self._urgent: deque[MemoryUid] = deque()
        self._urgent_set: set[MemoryUid] = set()

    def begin_maintenance(self) -> int:
        self.maintenance_cycle += 1
        return self.maintenance_cycle

    def _ensure_record(self, uid: MemoryUid) -> LifecycleRecord:
        current = self.records.get(uid)
        if current is not None:
            return current
        current = LifecycleRecord(
            uid=uid,
            last_transition_cycle=self.maintenance_cycle,
            last_observed_cycle=self.maintenance_cycle,
        )
        self.records[uid] = current
        self._scan_order.append(uid)
        return current

    def mark_urgent(self, uid: MemoryUid) -> None:
        if uid not in self.records or uid in self._urgent_set:
            return
        self._urgent.append(uid)
        self._urgent_set.add(uid)

    def take_urgent(self, limit: int) -> tuple[MemoryUid, ...]:
        result: list[MemoryUid] = []
        while self._urgent and len(result) < max(0, int(limit)):
            uid = self._urgent.popleft()
            self._urgent_set.discard(uid)
            if uid in self.records:
                result.append(uid)
        return tuple(result)

    def _compact_scan_order_if_needed(self) -> None:
        if self._stale_scan_entries < max(1024, len(self._scan_order) // 4):
            return
        self._scan_order = [uid for uid in self._scan_order if uid in self.records]
        self._scan_cursor = 0 if not self._scan_order else self._scan_cursor % len(self._scan_order)
        self._stale_scan_entries = 0

    def scan_batch(self, limit: int) -> tuple[MemoryUid, ...]:
        maximum = max(0, int(limit))
        if maximum <= 0 or not self._scan_order:
            return ()
        result: list[MemoryUid] = []
        checked = 0
        population = len(self._scan_order)
        while checked < population and len(result) < maximum:
            if self._scan_cursor >= len(self._scan_order):
                self._scan_cursor = 0
            uid = self._scan_order[self._scan_cursor]
            self._scan_cursor += 1
            checked += 1
            if uid not in self.records:
                self._stale_scan_entries += 1
                continue
            result.append(uid)
        self._compact_scan_order_if_needed()
        return tuple(result)

    def remove(self, uid: MemoryUid) -> None:
        if self.records.pop(uid, None) is not None:
            self._stale_scan_entries += 1
        self._urgent_set.discard(uid)

    def observe(self, uid: MemoryUid, *, support_delta: int, relevant_opportunity: bool, watermark: int) -> LifecycleRecord:
        current = self._ensure_record(uid)
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
        if reactivated:
            self._urgent_set.discard(uid)
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
        current = self._ensure_record(uid)
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

    def retire_if_exhausted(
        self,
        uid: MemoryUid,
        *,
        required_opportunities: int,
        watermark: int,
        has_authoritative_dependency: bool = False,
        has_provenance_obligation: bool = False,
    ) -> LifecycleRecord:
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
        self._urgent_set.discard(uid)
        return row

    def state_dict(self) -> dict[str, object]:
        ordered = [uid for uid in self._scan_order if uid in self.records]
        seen = set(ordered)
        ordered.extend(uid for uid in self.records if uid not in seen)
        return {
            "schema_version": self.SCHEMA_VERSION,
            "transitions": self.transitions,
            "maintenance_cycle": self.maintenance_cycle,
            "records": [
                {
                    "uid": [uid.hi, uid.lo],
                    "state": self.records[uid].state.name,
                    "support": self.records[uid].support,
                    "relevant_opportunities": self.records[uid].relevant_opportunities,
                    "last_transition_watermark": self.records[uid].last_transition_watermark,
                    "last_observed_watermark": self.records[uid].last_observed_watermark,
                    "last_transition_cycle": self.records[uid].last_transition_cycle,
                    "last_observed_cycle": self.records[uid].last_observed_cycle,
                    "retention_score": self.records[uid].retention_score,
                    "replacement_uid": None if self.records[uid].replacement_uid is None else [self.records[uid].replacement_uid.hi, self.records[uid].replacement_uid.lo],
                    "baseline_hgt_retention": self.records[uid].baseline_hgt_retention,
                    "baseline_behavioral_success": self.records[uid].baseline_behavioral_success,
                    "baseline_transfer_quality": self.records[uid].baseline_transfer_quality,
                    "baseline_prediction_quality": self.records[uid].baseline_prediction_quality,
                }
                for uid in ordered
            ],
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "LifecycleRegistry":
        result = cls()
        schema_version = int(state.get("schema_version", 1))
        legacy_watermark_aging = schema_version < 2 or "maintenance_cycle" not in state
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
            row = LifecycleRecord(
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
            result.records[uid] = row
            result._scan_order.append(uid)
            if restored_state in {CognitiveState.DORMANT, CognitiveState.RETIRE_PENDING}:
                result.mark_urgent(uid)
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


def _retirement_floor(node: Any, *, m0_floor: int, m1_floor: int) -> int:
    if node.level is MemoryLevel.M0:
        return max(1, int(m0_floor))
    if node.level is MemoryLevel.M1 and node.memory_type is MemoryType.GROUNDED_CONTINGENCY:
        return max(1, int(m1_floor))
    return 2**31 - 1


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
    compaction_pressure_threshold: float = 1.0,
    compaction_target_ratio: float = 0.90,
    max_compaction_batch: int = 131_072,
    max_scan_batch: int = 262_144,
    m0_representative_floor: int = 8,
    m1_representative_floor: int = 2,
) -> dict[str, int | float]:
    if not runtime.config.enable_lifecycle:
        return {"cycle": 0, "scanned": 0, "dormant": 0, "pending": 0, "retired": 0, "reactivated": 0, "blocked": 0, "pressure": 0.0}
    if min(int(dormancy_grace_cycles), int(retirement_grace_cycles)) <= 0:
        raise ValueError("lifecycle grace cycles must be positive")
    if not 0.0 < float(compaction_target_ratio) < 1.0:
        raise ValueError("compaction target ratio must be between zero and one")

    graph = runtime.graph
    registry: LifecycleRegistry = runtime.lifecycle
    cycle = registry.begin_maintenance()
    watermark = int(runtime.watermark)
    pressure = float(graph.pressure_ratio())
    dynamic_storage = graph.node_capacity_per_partition is None
    compaction_enabled = dynamic_storage or pressure >= float(compaction_pressure_threshold)
    pressure_multiplier = min(4, 1 + max(0, int((pressure - 0.70) * 10)))

    target_capacity = 0
    if graph.node_capacity_per_partition is not None:
        target_capacity = int(graph.partition_count * graph.node_capacity_per_partition * float(compaction_target_ratio))
        excess_nodes = max(0, graph.memory_count() - target_capacity)
    else:
        low_level_nodes = len(graph._uids_by_level[MemoryLevel.M0]) + len(graph._uids_by_level[MemoryLevel.M1])
        events = max(1, int(getattr(runtime, "telemetry", {}).get("events", 0)))
        low_level_target = max(65_536, events + max(32_768, events // 8))
        excess_nodes = max(0, low_level_nodes - low_level_target)
    if compaction_enabled:
        effective_retirement_limit = min(
            max(1, int(max_compaction_batch)),
            max(max(1, int(retirement_limit)) * pressure_multiplier, excess_nodes),
        )
        effective_scan_limit = min(
            max(1, int(max_scan_batch)),
            max(max(1, int(scan_limit)) * pressure_multiplier, effective_retirement_limit * 2),
        )
    else:
        effective_retirement_limit = max(1, int(retirement_limit))
        effective_scan_limit = max(1, int(scan_limit))

    urgent = list(registry.take_urgent(effective_scan_limit))
    urgent_set = set(urgent)
    normal = registry.scan_batch(max(0, effective_scan_limit - len(urgent)))
    candidate_uids = urgent + [uid for uid in normal if uid not in urgent_set]

    pending_plans: list[tuple[MemoryUid, MemoryUid, str]] = []
    planned_by_group: dict[tuple[MemoryUid, MemoryLevel, MemoryType], int] = {}
    live_group_sizes: dict[tuple[MemoryUid, MemoryLevel, MemoryType], int] = {}
    visibility_updates: dict[MemoryUid, CognitiveState] = {}
    current_validation = _validation_metrics(runtime)
    dormant = 0
    pending_count = 0
    reactivated = 0
    blocked = 0

    def can_plan_retirement(uid: MemoryUid, node: Any, replacement_uid: MemoryUid, row: LifecycleRecord) -> bool:
        nonlocal blocked
        if len(pending_plans) >= effective_retirement_limit:
            registry.mark_urgent(uid)
            return False
        if uid in runtime._replay_pool or not _validation_preserved(row, runtime, tolerance=validation_tolerance):
            blocked += 1
            registry.mark_urgent(uid)
            return False
        group = (replacement_uid, node.level, node.memory_type)
        if group not in live_group_sizes:
            live_group_sizes[group] = sum(
                1
                for target_uid in graph.replacement_target_uids(replacement_uid)
                if (target := graph.nodes.get(target_uid)) is not None
                and target.level is node.level
                and target.memory_type is node.memory_type
            )
        floor = _retirement_floor(node, m0_floor=m0_representative_floor, m1_floor=m1_representative_floor)
        already_planned = planned_by_group.get(group, 0)
        if live_group_sizes[group] - already_planned <= floor:
            blocked += 1
            return False
        pending_plans.append((uid, replacement_uid, "retention_compaction"))
        planned_by_group[group] = already_planned + 1
        return True

    for uid in candidate_uids:
        row = registry.records.get(uid)
        node = graph.nodes.get(uid)
        if row is None or node is None:
            continue

        # M1 normalized relations are already compact reusable behavior. They are
        # protected from automatic dormancy and physical deletion.
        if node.memory_type is MemoryType.NORMALIZED_RELATION:
            if row.state in {CognitiveState.DORMANT, CognitiveState.RETIRE_PENDING}:
                registry.transition(uid, CognitiveState.REACTIVATED, watermark=watermark, retention_score=row.retention_score)
                visibility_updates[uid] = CognitiveState.REACTIVATED
                reactivated += 1
            continue

        replacement_uid = graph.provenance_replacement(uid, maximum_level=MemoryLevel.M1) if node.level <= MemoryLevel.M1 else None
        score = retention_score(row, graph.payloads.get(uid, {}), watermark=watermark)

        if row.state is CognitiveState.RETIRE_PENDING:
            if not compaction_enabled or replacement_uid is None or node.level > MemoryLevel.M1:
                registry.transition(uid, CognitiveState.DORMANT, watermark=watermark, retention_score=score, replacement_uid=replacement_uid)
                visibility_updates[uid] = CognitiveState.DORMANT
                continue
            if cycle - int(row.last_transition_cycle) < int(retirement_grace_cycles):
                registry.mark_urgent(uid)
                continue
            can_plan_retirement(uid, node, replacement_uid, row)
            continue

        if row.state is CognitiveState.DORMANT:
            if score >= reactivation_threshold:
                registry.transition(uid, CognitiveState.REACTIVATED, watermark=watermark, retention_score=score)
                visibility_updates[uid] = CognitiveState.REACTIVATED
                reactivated += 1
                continue
            if (
                compaction_enabled
                and replacement_uid is not None
                and node.level <= MemoryLevel.M1
                and cycle - int(row.last_transition_cycle) >= int(dormancy_grace_cycles)
            ):
                if uid in runtime._replay_pool or not _validation_preserved(row, runtime, tolerance=validation_tolerance):
                    blocked += 1
                    registry.mark_urgent(uid)
                else:
                    registry.transition(uid, CognitiveState.RETIRE_PENDING, watermark=watermark, retention_score=score, replacement_uid=replacement_uid)
                    visibility_updates[uid] = CognitiveState.RETIRE_PENDING
                    pending_count += 1
                    registry.mark_urgent(uid)
            elif compaction_enabled and replacement_uid is not None and node.level <= MemoryLevel.M1:
                registry.mark_urgent(uid)
            continue

        # Automatic age-based forgetting is limited to replaceable low-level
        # evidence. M2-M7 abstractions require explicit contradiction/supersession
        # mechanisms elsewhere; mere inactivity is not enough to hide them.
        can_dormant = node.level <= MemoryLevel.M1 and replacement_uid is not None
        idle_cycles = cycle - int(row.last_observed_cycle)
        if can_dormant and idle_cycles >= int(dormancy_grace_cycles) and score < dormant_threshold:
            registry.transition(
                uid,
                CognitiveState.DORMANT,
                watermark=watermark,
                retention_score=score,
                replacement_uid=replacement_uid,
                validation_baseline=current_validation,
            )
            visibility_updates[uid] = CognitiveState.DORMANT
            dormant += 1
            if compaction_enabled:
                registry.mark_urgent(uid)
        else:
            registry.transition(uid, row.state, watermark=watermark, retention_score=score, replacement_uid=row.replacement_uid)

    _sync_cognitive_visibility(graph, visibility_updates)

    replay_before = len(runtime._replay_pool)
    retired_records = {uid: registry.records[uid] for uid, _, _ in pending_plans if uid in registry.records}
    retired_uids = graph.retire_nodes_batch(tuple(pending_plans))
    retired_set = set(retired_uids)
    for uid in retired_uids:
        registry.remove(uid)
        runtime._replay_pool.pop(uid, None)
        runtime._deferred_base_nodes.pop(uid, None)
        runtime._m2.pop(uid, None)
        runtime._m3.pop(uid, None)
        runtime._m4.pop(uid, None)
        getattr(runtime, "_m5", {}).pop(uid, None)
        getattr(runtime, "_m6", {}).pop(uid, None)
        getattr(runtime, "_m7", {}).pop(uid, None)
        runtime._transfer_trials.pop(uid, None)
        environment_index = getattr(runtime, "_memory_uids_by_environment", None)
        if environment_index is not None:
            for environment_id in tuple(environment_index):
                environment_index[environment_id].discard(uid)
                if not environment_index[environment_id]:
                    del environment_index[environment_id]
    if retired_set:
        runtime._latest_interaction_grounding = {
            key: row for key, row in runtime._latest_interaction_grounding.items() if row.uid not in retired_set
        }

    replay_after = len(runtime._replay_pool)
    if retired_uids or reactivated:
        after_hgt = _validation_metrics(runtime)[0]
        before_hgt_values = [
            retired_records[uid].baseline_hgt_retention
            for uid in retired_uids
            if uid in retired_records and retired_records[uid].baseline_hgt_retention is not None
        ]
        before_hgt = (
            sum(before_hgt_values) / len(before_hgt_values)
            if before_hgt_values
            else (after_hgt if after_hgt is not None else 1.0)
        )
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
        "scanned": len(candidate_uids),
        "dormant": dormant,
        "pending": pending_count,
        "retired": len(retired_uids),
        "reactivated": reactivated,
        "blocked": blocked,
        "pressure": pressure,
        "excess_nodes": excess_nodes,
        "scan_budget": effective_scan_limit,
        "retirement_budget": effective_retirement_limit,
    }
