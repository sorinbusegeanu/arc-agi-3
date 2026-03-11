from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from codex_baseline_v2.shared.plan_records import SkillExecutionRecordV1, SkillSpecV1
from codex_baseline_v2.shared.storage import StoragePathsV2


CLUSTER_QUANTIZATION = 8
CLUSTER_EXCLUSION_COOLDOWN = 2
PROBE_NO_PROGRESS_REASONS = {"contact_no_effect", "contact_reached_boundary_without_route_progress", "route_stall", "no_progress"}


@dataclass(frozen=True)
class ClusterLedgerEntryV1:
    cluster_key: str
    area_id: Optional[str]
    quantized_x: Optional[int]
    quantized_y: Optional[int]
    centroid_x: Optional[float]
    centroid_y: Optional[float]
    failure_count: int = 0
    success_count: int = 0
    contact_no_effect_count: int = 0
    last_failed_round: Optional[int] = None
    repeated_no_effect_streak: int = 0
    locally_exhausted: bool = False
    candidate_count: int = 0
    reachable_count: int = 0
    survivor_count: int = 0
    retry_count: int = 0
    latest_failure_reason: Optional[str] = None
    suppression_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "ClusterLedgerEntryV1":
        return cls(
            cluster_key=str(payload.get("cluster_key", "")),
            area_id=payload.get("area_id"),
            quantized_x=payload.get("quantized_x"),
            quantized_y=payload.get("quantized_y"),
            centroid_x=payload.get("centroid_x"),
            centroid_y=payload.get("centroid_y"),
            failure_count=int(payload.get("failure_count", 0)),
            success_count=int(payload.get("success_count", 0)),
            contact_no_effect_count=int(payload.get("contact_no_effect_count", 0)),
            last_failed_round=payload.get("last_failed_round"),
            repeated_no_effect_streak=int(payload.get("repeated_no_effect_streak", 0)),
            locally_exhausted=bool(payload.get("locally_exhausted", False)),
            candidate_count=int(payload.get("candidate_count", 0)),
            reachable_count=int(payload.get("reachable_count", 0)),
            survivor_count=int(payload.get("survivor_count", 0)),
            retry_count=int(payload.get("retry_count", 0)),
            latest_failure_reason=payload.get("latest_failure_reason"),
            suppression_reasons=list(payload.get("suppression_reasons", [])),
        )


@dataclass(frozen=True)
class PlanMemoryStateV1:
    schema_version: str
    recent_failed_cluster_keys: List[str] = field(default_factory=list)
    recent_successful_cluster_keys: List[str] = field(default_factory=list)
    last_distant_jump_target_cluster: Optional[str] = None
    excluded_cluster_cooldowns: Dict[str, int] = field(default_factory=dict)
    cluster_ledger: List[ClusterLedgerEntryV1] = field(default_factory=list)
    recent_failed_movement_cluster_keys: List[str] = field(default_factory=list)
    failed_movement_area_ids_this_round: List[str] = field(default_factory=list)
    movement_cluster_ledger: List[ClusterLedgerEntryV1] = field(default_factory=list)
    movement_planning_debug: Dict[str, object] = field(default_factory=dict)
    failed_cluster_keys_this_round: List[str] = field(default_factory=list)
    failed_row_bands_this_round: List[str] = field(default_factory=list)
    failed_neighborhood_centroids_this_round: List[List[float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "recent_failed_cluster_keys": list(self.recent_failed_cluster_keys),
            "recent_successful_cluster_keys": list(self.recent_successful_cluster_keys),
            "last_distant_jump_target_cluster": self.last_distant_jump_target_cluster,
            "excluded_cluster_cooldowns": dict(self.excluded_cluster_cooldowns),
            "cluster_ledger": [row.to_dict() for row in self.cluster_ledger],
            "recent_failed_movement_cluster_keys": list(self.recent_failed_movement_cluster_keys),
            "failed_movement_area_ids_this_round": list(self.failed_movement_area_ids_this_round),
            "movement_cluster_ledger": [row.to_dict() for row in self.movement_cluster_ledger],
            "movement_planning_debug": dict(self.movement_planning_debug),
            "failed_cluster_keys_this_round": list(self.failed_cluster_keys_this_round),
            "failed_row_bands_this_round": list(self.failed_row_bands_this_round),
            "failed_neighborhood_centroids_this_round": [list(v) for v in self.failed_neighborhood_centroids_this_round],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "PlanMemoryStateV1":
        return cls(
            schema_version=str(payload.get("schema_version", "v2.4.1")),
            recent_failed_cluster_keys=list(payload.get("recent_failed_cluster_keys", [])),
            recent_successful_cluster_keys=list(payload.get("recent_successful_cluster_keys", [])),
            last_distant_jump_target_cluster=payload.get("last_distant_jump_target_cluster"),
            excluded_cluster_cooldowns={str(k): int(v) for k, v in dict(payload.get("excluded_cluster_cooldowns", {})).items()},
            cluster_ledger=[ClusterLedgerEntryV1.from_dict(v) for v in payload.get("cluster_ledger", [])],
            recent_failed_movement_cluster_keys=list(payload.get("recent_failed_movement_cluster_keys", [])),
            failed_movement_area_ids_this_round=list(payload.get("failed_movement_area_ids_this_round", [])),
            movement_cluster_ledger=[ClusterLedgerEntryV1.from_dict(v) for v in payload.get("movement_cluster_ledger", [])],
            movement_planning_debug=dict(payload.get("movement_planning_debug", {})),
            failed_cluster_keys_this_round=list(payload.get("failed_cluster_keys_this_round", [])),
            failed_row_bands_this_round=list(payload.get("failed_row_bands_this_round", [])),
            failed_neighborhood_centroids_this_round=[list(v) for v in payload.get("failed_neighborhood_centroids_this_round", [])],
        )


def _ordered_unique(values: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _execution_round_id(execution_id: str) -> Optional[int]:
    if "round" not in execution_id:
        return None
    try:
        token = execution_id.split("round", 1)[1][:3]
        return int(token)
    except Exception:
        return None


def _target_ref(skill: SkillSpecV1) -> Optional[str]:
    for ref in skill.precondition_ids:
        if ref.startswith("trigger_zone:") or ref.startswith("poi:"):
            return ref
    return None


def _target_xy(skill: SkillSpecV1) -> Tuple[Optional[int], Optional[int]]:
    if skill.target_x_this_round is not None and skill.target_y_this_round is not None:
        return int(skill.target_x_this_round), int(skill.target_y_this_round)
    target_ref = _target_ref(skill)
    if not target_ref:
        return None, None
    parts = target_ref.split(":")
    try:
        return int(parts[-2]), int(parts[-1])
    except Exception:
        return None, None


def _target_area_id(skill: SkillSpecV1) -> Optional[str]:
    target_ref = _target_ref(skill)
    if not target_ref:
        return None
    parts = target_ref.split(":")
    if len(parts) <= 2:
        return None
    return ":".join(parts[:-2])


def cluster_key_for_skill(skill: SkillSpecV1) -> Optional[str]:
    if skill.skill_type not in {"probe_hidden_trigger", "contact_poi", "go_to_region"}:
        return None
    x, y = _target_xy(skill)
    if x is None or y is None:
        return None
    area_id = _target_area_id(skill) or "global"
    return f"{area_id}|{int(x) // CLUSTER_QUANTIZATION}|{int(y) // CLUSTER_QUANTIZATION}"


def cluster_key_from_coords(area_id: Optional[str], x: Optional[int], y: Optional[int]) -> Optional[str]:
    if x is None or y is None:
        return None
    return f"{area_id or 'global'}|{int(x) // CLUSTER_QUANTIZATION}|{int(y) // CLUSTER_QUANTIZATION}"


def cluster_components(cluster_key: Optional[str]) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    if not cluster_key or "|" not in cluster_key:
        return None, None, None
    area_id, qx, qy = cluster_key.rsplit("|", 2)
    try:
        return area_id, int(qx), int(qy)
    except Exception:
        return area_id, None, None


def cluster_distance(lhs: Optional[str], rhs: Optional[str]) -> Optional[int]:
    l_area, l_x, l_y = cluster_components(lhs)
    r_area, r_x, r_y = cluster_components(rhs)
    if l_x is None or l_y is None or r_x is None or r_y is None:
        return None
    if l_area != r_area:
        return max(4, abs(l_x - r_x) + abs(l_y - r_y))
    return abs(l_x - r_x) + abs(l_y - r_y)


def neighboring_cluster_keys(cluster_key: Optional[str]) -> List[str]:
    area_id, qx, qy = cluster_components(cluster_key)
    if area_id is None or qx is None or qy is None:
        return []
    out = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            out.append(f"{area_id}|{qx + dx}|{qy + dy}")
    return out


def row_band_id(cluster_key: Optional[str]) -> Optional[str]:
    area_id, _, qy = cluster_components(cluster_key)
    if area_id is None or qy is None:
        return None
    return f"{area_id}|row|{qy}"


def is_distant_cluster(lhs: Optional[str], rhs: Optional[str]) -> bool:
    distance = cluster_distance(lhs, rhs)
    return distance is not None and distance >= 2


def load_plan_memory(storage: StoragePathsV2, game_id: str) -> PlanMemoryStateV1:
    path = os.path.join(storage.game_root(game_id), "plan_memory_state.json")
    if not os.path.exists(path):
        return PlanMemoryStateV1(schema_version="v2.4.1")
    with open(path, "r", encoding="utf-8") as handle:
        return PlanMemoryStateV1.from_dict(json.load(handle))


def save_plan_memory(storage: StoragePathsV2, game_id: str, memory: PlanMemoryStateV1) -> None:
    path = os.path.join(storage.game_root(game_id), "plan_memory_state.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(memory.to_dict(), handle, sort_keys=True)


def plan_memory_refs(memory: PlanMemoryStateV1) -> List[str]:
    refs: List[str] = []
    refs.extend(f"cluster_failed:{key}" for key in memory.recent_failed_cluster_keys)
    refs.extend(f"cluster_success:{key}" for key in memory.recent_successful_cluster_keys)
    refs.extend(f"cluster_failed_round:{key}" for key in memory.failed_cluster_keys_this_round)
    refs.extend(f"row_failed_round:{key}" for key in memory.failed_row_bands_this_round)
    refs.extend(f"movement_cluster_failed:{key}" for key in memory.recent_failed_movement_cluster_keys)
    refs.extend(f"movement_area_failed_round:{key}" for key in memory.failed_movement_area_ids_this_round)
    refs.extend(f"cluster_excluded:{key}:{ttl}" for key, ttl in sorted(memory.excluded_cluster_cooldowns.items()) if ttl > 0)
    if memory.last_distant_jump_target_cluster:
        refs.append(f"last_distant_jump:{memory.last_distant_jump_target_cluster}")
    return refs


def cluster_ledger_map(memory: Optional[PlanMemoryStateV1]) -> Dict[str, ClusterLedgerEntryV1]:
    if memory is None:
        return {}
    return {row.cluster_key: row for row in memory.cluster_ledger}


def movement_cluster_ledger_map(memory: Optional[PlanMemoryStateV1]) -> Dict[str, ClusterLedgerEntryV1]:
    if memory is None:
        return {}
    return {row.cluster_key: row for row in memory.movement_cluster_ledger}


def reconcile_plan_memory(
    skills: List[SkillSpecV1],
    executions: List[SkillExecutionRecordV1],
    previous: Optional[PlanMemoryStateV1] = None,
) -> PlanMemoryStateV1:
    prev = previous or PlanMemoryStateV1(schema_version="v2.4.1")
    skills_by_id = {skill.skill_id: skill for skill in skills}
    max_round_id = max((_execution_round_id(row.execution_id) for row in executions if _execution_round_id(row.execution_id) is not None), default=None)
    cluster_rows: Dict[str, List[Tuple[SkillExecutionRecordV1, SkillSpecV1, Optional[int]]]] = {}
    failed_clusters: List[str] = []
    successful_clusters: List[str] = []
    for row in executions:
        skill = skills_by_id.get(row.skill_id)
        if skill is None:
            continue
        cluster_key = cluster_key_for_skill(skill)
        if cluster_key is None:
            continue
        round_id = _execution_round_id(row.execution_id)
        cluster_rows.setdefault(cluster_key, []).append((row, skill, round_id))
        if skill.skill_type == "probe_hidden_trigger" and row.termination_reason in PROBE_NO_PROGRESS_REASONS:
            failed_clusters.append(cluster_key)
        if row.success or row.observed_event_ids or row.observed_topology_delta_ids:
            successful_clusters.append(cluster_key)
    prev_excluded = {key: max(0, int(value) - 1) for key, value in prev.excluded_cluster_cooldowns.items() if int(value) > 1}
    cluster_ledger: List[ClusterLedgerEntryV1] = []
    movement_cluster_ledger: List[ClusterLedgerEntryV1] = []
    recent_failed_movement_cluster_keys: List[str] = []
    failed_movement_area_ids_this_round: List[str] = []
    failed_cluster_keys_this_round: List[str] = []
    failed_row_bands_this_round: List[str] = []
    failed_neighborhood_centroids_this_round: List[List[float]] = []
    for cluster_key, rows in cluster_rows.items():
        sample_skill = rows[-1][1]
        x, y = _target_xy(sample_skill)
        area_id = _target_area_id(sample_skill)
        failure_count = sum(1 for row, _, _ in rows if row.termination_reason in {"contact_no_effect", "contact_no_reach", "contact_reached_boundary_without_route_progress", "route_stall", "blocked", "invalid_target", "no_progress", "unreachable_now"})
        success_count = sum(1 for row, _, _ in rows if row.success or row.observed_event_ids or row.observed_topology_delta_ids)
        contact_no_effect_count = sum(1 for row, _, _ in rows if row.termination_reason in PROBE_NO_PROGRESS_REASONS)
        last_failed_round = max((round_id for row, _, round_id in rows if row.termination_reason in PROBE_NO_PROGRESS_REASONS and round_id is not None), default=None)
        repeated_no_effect_streak = 0
        for row, _, _ in reversed(rows):
            if row.termination_reason in PROBE_NO_PROGRESS_REASONS:
                repeated_no_effect_streak += 1
            else:
                break
        current_round_rows = [row for row in rows if max_round_id is not None and row[2] == max_round_id]
        current_round_nulls = sum(1 for row, _, _ in current_round_rows if row.termination_reason in PROBE_NO_PROGRESS_REASONS)
        current_round_positive = any(row.success or row.observed_event_ids or row.observed_topology_delta_ids for row, _, _ in current_round_rows)
        if current_round_nulls > 0:
            failed_cluster_keys_this_round.append(cluster_key)
            band = row_band_id(cluster_key)
            if band:
                failed_row_bands_this_round.append(band)
            if x is not None and y is not None:
                failed_neighborhood_centroids_this_round.append([float(x), float(y)])
            if sample_skill.skill_type == "probe_hidden_trigger":
                for neighbor_key in neighboring_cluster_keys(cluster_key):
                    prev_excluded[neighbor_key] = max(prev_excluded.get(neighbor_key, 0), CLUSTER_EXCLUSION_COOLDOWN)
        prev_row = next((row for row in prev.cluster_ledger if row.cluster_key == cluster_key), None)
        prior_region_failures = 0 if prev_row is None else int(prev_row.contact_no_effect_count)
        locally_exhausted = ((current_round_nulls >= 2 or repeated_no_effect_streak >= 2) and not current_round_positive) or (
            sample_skill.skill_type == "probe_hidden_trigger"
            and current_round_nulls > 0
            and prior_region_failures >= 1
            and not current_round_positive
        )
        if locally_exhausted:
            if sample_skill.skill_type == "probe_hidden_trigger":
                prev_excluded[cluster_key] = max(prev_excluded.get(cluster_key, 0), CLUSTER_EXCLUSION_COOLDOWN)
        if current_round_positive and cluster_key in prev_excluded:
            prev_excluded.pop(cluster_key, None)
        area, qx, qy = cluster_components(cluster_key)
        cluster_ledger.append(
            ClusterLedgerEntryV1(
                cluster_key=cluster_key,
                area_id=area_id or area,
                quantized_x=qx,
                quantized_y=qy,
                centroid_x=float(x) if x is not None else None,
                centroid_y=float(y) if y is not None else None,
                failure_count=failure_count,
                success_count=success_count,
                contact_no_effect_count=contact_no_effect_count,
                last_failed_round=last_failed_round,
                repeated_no_effect_streak=repeated_no_effect_streak,
                locally_exhausted=locally_exhausted,
                latest_failure_reason=next((row.termination_reason for row, _, _ in reversed(rows) if row.termination_reason in PROBE_NO_PROGRESS_REASONS), None),
            )
        )
        if sample_skill.skill_type == "go_to_region":
            movement_failure_count = sum(1 for row, _, _ in rows if row.termination_reason in {"route_stall", "contact_no_reach", "contact_reached_boundary_without_route_progress", "no_progress", "blocked", "invalid_target"})
            movement_success_count = sum(1 for row, _, _ in rows if row.success)
            movement_contact_no_effect_count = sum(1 for row, _, _ in rows if row.termination_reason in {"route_stall", "contact_no_reach", "contact_reached_boundary_without_route_progress", "no_progress"})
            movement_repeated_no_effect_streak = 0
            for row, _, _ in reversed(rows):
                if row.termination_reason in {"route_stall", "contact_no_reach", "contact_reached_boundary_without_route_progress", "no_progress"} and not row.observed_event_ids and not row.observed_topology_delta_ids:
                    movement_repeated_no_effect_streak += 1
                else:
                    break
            movement_current_round_failures = sum(
                1
                for row, _, _ in current_round_rows
                if row.termination_reason in {"route_stall", "contact_no_reach", "contact_reached_boundary_without_route_progress", "no_progress"} and not row.observed_event_ids and not row.observed_topology_delta_ids
            )
            movement_current_round_positive = any(
                row.success or row.observed_event_ids or row.observed_topology_delta_ids
                for row, _, _ in current_round_rows
            )
            if movement_failure_count > 0:
                recent_failed_movement_cluster_keys.append(cluster_key)
            if any(row.termination_reason in {"route_stall", "contact_no_reach", "contact_reached_boundary_without_route_progress", "no_progress"} for row, _, _ in current_round_rows):
                if area_id:
                    failed_movement_area_ids_this_round.append(area_id)
            movement_cluster_ledger.append(
                ClusterLedgerEntryV1(
                    cluster_key=cluster_key,
                    area_id=area_id or area,
                    quantized_x=qx,
                    quantized_y=qy,
                    centroid_x=float(x) if x is not None else None,
                    centroid_y=float(y) if y is not None else None,
                    failure_count=movement_failure_count,
                    success_count=movement_success_count,
                    contact_no_effect_count=movement_contact_no_effect_count,
                    last_failed_round=max((round_id for row, _, round_id in rows if row.termination_reason in {"route_stall", "contact_no_reach", "contact_reached_boundary_without_route_progress", "no_progress"} and round_id is not None), default=None),
                    repeated_no_effect_streak=movement_repeated_no_effect_streak,
                    locally_exhausted=(movement_current_round_failures >= 2 or movement_repeated_no_effect_streak >= 2) and not movement_current_round_positive,
                    latest_failure_reason=next((row.termination_reason for row, _, _ in reversed(rows) if row.termination_reason in {"route_stall", "contact_no_reach", "contact_reached_boundary_without_route_progress", "no_progress", "blocked", "invalid_target"}), None),
                )
            )
    last_distant_jump_target_cluster = prev.last_distant_jump_target_cluster
    probe_sequence: List[str] = []
    if max_round_id is not None:
        round_rows = sorted(
            (
                (row.execution_id, cluster_key)
                for cluster_key, rows in cluster_rows.items()
                for row, _, round_id in rows
                if round_id == max_round_id
            ),
            key=lambda pair: pair[0],
        )
        probe_sequence = [cluster_key for _, cluster_key in round_rows]
    for prev_key, next_key in zip(probe_sequence, probe_sequence[1:]):
        if is_distant_cluster(prev_key, next_key):
            last_distant_jump_target_cluster = next_key
            if prev_key not in successful_clusters:
                prev_excluded[prev_key] = max(prev_excluded.get(prev_key, 0), CLUSTER_EXCLUSION_COOLDOWN)
    supporting_trigger_failures = {
        key for key in failed_clusters
        if key.startswith("trigger_zone:")
    } | {
        row.cluster_key for row in cluster_ledger
        if row.contact_no_effect_count > 0 and (row.area_id or "").startswith("trigger_zone:")
    }
    prev_excluded = {
        key: ttl
        for key, ttl in prev_excluded.items()
        if ttl > 0 and ((cluster_components(key)[0] or "").startswith("trigger_zone:") and any((cluster_distance(key, failed) or 99) <= 1 for failed in supporting_trigger_failures))
    }
    return PlanMemoryStateV1(
        schema_version="v2.4.1",
        recent_failed_cluster_keys=_ordered_unique(failed_clusters[-12:]),
        recent_successful_cluster_keys=_ordered_unique(successful_clusters[-12:]),
        last_distant_jump_target_cluster=last_distant_jump_target_cluster,
        excluded_cluster_cooldowns=prev_excluded,
        cluster_ledger=sorted(cluster_ledger, key=lambda row: row.cluster_key),
        recent_failed_movement_cluster_keys=_ordered_unique(recent_failed_movement_cluster_keys[-12:]),
        failed_movement_area_ids_this_round=_ordered_unique(failed_movement_area_ids_this_round),
        movement_cluster_ledger=sorted(movement_cluster_ledger, key=lambda row: row.cluster_key),
        failed_cluster_keys_this_round=_ordered_unique(failed_cluster_keys_this_round),
        failed_row_bands_this_round=_ordered_unique(failed_row_bands_this_round),
        failed_neighborhood_centroids_this_round=failed_neighborhood_centroids_this_round[-12:],
    )
