from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, replace
from typing import Any

from v9.memory.identity import MemoryUid
from v9.memory.model import CognitiveState, MemoryLevel
from v9.telemetry import ConsolidationSample


@dataclass(frozen=True, slots=True)
class LifecycleRecord:
    uid: MemoryUid
    state: CognitiveState = CognitiveState.CANDIDATE
    support: int = 0
    relevant_opportunities: int = 0
    last_transition_watermark: int = 0
    last_observed_watermark: int = 0
    retention_score: float = 1.0
    replacement_uid: MemoryUid | None = None


class LifecycleRegistry:
    def __init__(self) -> None:
        self.records: dict[MemoryUid, LifecycleRecord] = {}
        self.transitions = 0

    def observe(self, uid: MemoryUid, *, support_delta: int, relevant_opportunity: bool, watermark: int) -> LifecycleRecord:
        current = self.records.get(uid, LifecycleRecord(uid))
        support = current.support + int(support_delta)
        opportunities = current.relevant_opportunities + int(relevant_opportunity)
        state = current.state
        if support_delta > 0 and state in {CognitiveState.DORMANT, CognitiveState.RETIRE_PENDING, CognitiveState.RETIRED}:
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
            replacement_uid=None if state in {CognitiveState.ACTIVE, CognitiveState.REACTIVATED} else current.replacement_uid,
        )
        self.records[uid] = row
        return row

    def transition(self, uid: MemoryUid, state: CognitiveState, *, watermark: int, retention_score: float | None = None, replacement_uid: MemoryUid | None = None) -> LifecycleRecord:
        current = self.records.get(uid, LifecycleRecord(uid))
        transitioned = state is not current.state
        self.transitions += int(transitioned)
        row = replace(
            current,
            state=state,
            last_transition_watermark=int(watermark) if transitioned else current.last_transition_watermark,
            retention_score=current.retention_score if retention_score is None else float(retention_score),
            replacement_uid=replacement_uid,
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
            replacement_uid=None,
        )
        self.records[uid] = row
        return row

    def state_dict(self) -> dict[str, object]:
        return {
            "transitions": self.transitions,
            "records": [
                {
                    "uid": [uid.hi, uid.lo],
                    "state": row.state.name,
                    "support": row.support,
                    "relevant_opportunities": row.relevant_opportunities,
                    "last_transition_watermark": row.last_transition_watermark,
                    "last_observed_watermark": row.last_observed_watermark,
                    "retention_score": row.retention_score,
                    "replacement_uid": None if row.replacement_uid is None else [row.replacement_uid.hi, row.replacement_uid.lo],
                }
                for uid, row in sorted(self.records.items())
            ],
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "LifecycleRegistry":
        result = cls()
        result.transitions = int(state.get("transitions", 0))
        for raw in state.get("records", []):
            uid = MemoryUid(int(raw["uid"][0]), int(raw["uid"][1]))
            transition_watermark = int(raw.get("last_transition_watermark", 0))
            raw_replacement = raw.get("replacement_uid")
            replacement_uid = None if raw_replacement is None else MemoryUid(int(raw_replacement[0]), int(raw_replacement[1]))
            result.records[uid] = LifecycleRecord(
                uid=uid,
                state=CognitiveState[str(raw["state"])],
                support=int(raw.get("support", 0)),
                relevant_opportunities=int(raw.get("relevant_opportunities", 0)),
                last_transition_watermark=transition_watermark,
                last_observed_watermark=int(raw.get("last_observed_watermark", transition_watermark)),
                retention_score=float(raw.get("retention_score", 1.0)),
                replacement_uid=replacement_uid,
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


def run_lifecycle_maintenance(
    runtime: Any,
    *,
    scan_limit: int = 8192,
    retirement_limit: int = 2048,
    dormant_threshold: float = 0.22,
    reactivation_threshold: float = 0.40,
    dormancy_grace_watermarks: int = 2048,
    retirement_grace_watermarks: int = 2048,
) -> dict[str, int | float]:
    if not runtime.config.enable_lifecycle:
        return {"scanned": 0, "dormant": 0, "pending": 0, "retired": 0, "reactivated": 0, "pressure": 0.0}

    graph = runtime.graph
    registry: LifecycleRegistry = runtime.lifecycle
    watermark = int(runtime.watermark)
    pressure = float(graph.pressure_ratio())
    pressure_multiplier = min(4, 1 + max(0, int((pressure - 0.70) * 10)))
    effective_scan_limit = max(1, int(scan_limit)) * pressure_multiplier
    effective_retirement_limit = max(1, int(retirement_limit)) * pressure_multiplier
    replacements = graph.provenance_replacements(maximum_level=MemoryLevel.M1)

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
        if watermark - int(row.last_transition_watermark) < int(retirement_grace_watermarks):
            continue
        pending_plans.append((uid, replacement_uid, "retention_compaction"))

    replay_before = len(runtime._replay_pool)
    retired_uids = graph.retire_nodes_batch(tuple(pending_plans))
    retired_set = set(retired_uids)
    for uid in retired_uids:
        row = registry.records.get(uid, LifecycleRecord(uid))
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
        key=lambda row: (0 if row.state is CognitiveState.DORMANT else 1, int(row.last_observed_watermark), row.uid),
    )

    dormant = 0
    pending = 0
    reactivated = 0
    for row in candidates:
        node = graph.nodes.get(row.uid)
        if node is None:
            continue
        payload = graph.payloads.get(row.uid, {})
        score = retention_score(row, payload, watermark=watermark)
        age = watermark - int(row.last_observed_watermark)
        replacement_uid = replacements.get(row.uid)
        covered = replacement_uid is not None
        if row.state is CognitiveState.DORMANT:
            if score >= reactivation_threshold:
                registry.transition(row.uid, CognitiveState.REACTIVATED, watermark=watermark, retention_score=score)
                reactivated += 1
            elif node.level <= MemoryLevel.M1 and covered and watermark - int(row.last_transition_watermark) >= int(dormancy_grace_watermarks):
                registry.transition(row.uid, CognitiveState.RETIRE_PENDING, watermark=watermark, retention_score=score, replacement_uid=replacement_uid)
                pending += 1
            else:
                registry.transition(row.uid, CognitiveState.DORMANT, watermark=watermark, retention_score=score, replacement_uid=replacement_uid)
            continue
        can_dormant = covered if node.level <= MemoryLevel.M1 else True
        if can_dormant and age >= int(dormancy_grace_watermarks) and score < dormant_threshold:
            registry.transition(row.uid, CognitiveState.DORMANT, watermark=watermark, retention_score=score, replacement_uid=replacement_uid)
            dormant += 1
        else:
            registry.transition(row.uid, row.state, watermark=watermark, retention_score=score)

    replay_after = len(runtime._replay_pool)
    if retired_uids or reactivated:
        historical_retention = float(runtime.unified_telemetry.gauges.get("historical_retention", 1.0))
        runtime.record_hgt_consolidation(
            ConsolidationSample(
                hydra_bytes_retired=len(retired_uids) * 192,
                hydra_nodes_retired=len(retired_uids),
                hydra_nodes_replaced_by_abstractions=len(retired_uids),
                replay_examples_before=replay_before,
                replay_examples_after=replay_after,
                representative_retention_ratio=1.0 if retired_uids else 0.0,
                hgt_retention_before=historical_retention,
                hgt_retention_after=historical_retention,
                reactivation_examples=reactivated,
            )
        )

    return {
        "scanned": len(candidates),
        "dormant": dormant,
        "pending": pending,
        "retired": len(retired_uids),
        "reactivated": reactivated,
        "pressure": pressure,
    }
