from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from codex_baseline_v2.shared.config import AnalystConfigV2
from codex_baseline_v2.shared.schemas import ObjectRecordV2, SCHEMA_VERSION
from codex_baseline_v2.shared.utils import BBox


@dataclass
class AvatarCandidateStats:
    candidate_key: str
    object_id: str
    color: int
    bbox: BBox
    centroid: Tuple[float, float]
    motion_hits: int = 0
    steps_seen: int = 0
    action_displacements: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)
    persistence: int = 0
    hud_penalty: float = 0.0
    static_penalty: float = 0.0
    local_diff_overlap: float = 0.0
    displacement: Tuple[int, int] = (0, 0)


class AvatarCandidateAccumulator:
    def __init__(self) -> None:
        self._episode_presence: Dict[str, int] = {}

    def note_presence(self, candidate_key: str) -> None:
        self._episode_presence[candidate_key] = self._episode_presence.get(candidate_key, 0) + 1

    def persistence_score(self, candidate_key: str) -> float:
        return min(1.0, self._episode_presence.get(candidate_key, 0) / 3.0)


def _candidate_key(obj: ObjectRecordV2) -> str:
    return f"{obj.color}:{obj.bbox.x1}:{obj.bbox.y1}:{obj.bbox.x2}:{obj.bbox.y2}"


def _direction_consistency(displacements: List[Tuple[int, int]], current: Tuple[int, int]) -> float:
    if not displacements:
        return 0.5
    dx_signs = [1 if dx > 0 else -1 if dx < 0 else 0 for dx, _ in displacements]
    dy_signs = [1 if dy > 0 else -1 if dy < 0 else 0 for _, dy in displacements]
    avg_dx = sum(dx_signs) / float(len(dx_signs))
    avg_dy = sum(dy_signs) / float(len(dy_signs))
    cur_dx = 1 if current[0] > 0 else -1 if current[0] < 0 else 0
    cur_dy = 1 if current[1] > 0 else -1 if current[1] < 0 else 0
    return 0.5 * (1.0 if cur_dx == 0 or avg_dx == 0 or cur_dx * avg_dx > 0 else 0.0) + 0.5 * (
        1.0 if cur_dy == 0 or avg_dy == 0 or cur_dy * avg_dy > 0 else 0.0
    )


def score_avatar_candidates(
    objects: List[ObjectRecordV2],
    prev_objects: List[ObjectRecordV2],
    motion_points: List[Tuple[int, int]],
    action_id: Optional[int],
    cfg: AnalystConfigV2,
    accumulator: AvatarCandidateAccumulator,
    game_id: str,
    episode_id: str,
) -> Tuple[List[ObjectRecordV2], List[Dict[str, float]], List[Dict[str, object]]]:
    stats_table: List[AvatarCandidateStats] = []
    prev_by_color: Dict[int, List[ObjectRecordV2]] = {}
    for obj in prev_objects:
        prev_by_color.setdefault(obj.color, []).append(obj)
    motion_set = set(motion_points)
    for obj in objects:
        candidate_key = _candidate_key(obj)
        accumulator.note_presence(candidate_key)
        matched_prev = None
        if obj.color in prev_by_color:
            candidates = prev_by_color[obj.color]
            matched_prev = min(candidates, key=lambda o: (o.centroid[0] - obj.centroid[0]) ** 2 + (o.centroid[1] - obj.centroid[1]) ** 2)
        displacement = (0, 0)
        if matched_prev is not None:
            displacement = (int(round(obj.centroid[0] - matched_prev.centroid[0])), int(round(obj.centroid[1] - matched_prev.centroid[1])))
        overlap = 0
        for x in range(obj.bbox.x1, obj.bbox.x2 + 1):
            for y in range(obj.bbox.y1, obj.bbox.y2 + 1):
                if (x, y) in motion_set:
                    overlap += 1
        area = max(1, obj.area)
        local_overlap = float(overlap) / float(area)
        hud_penalty = 1.0 if obj.object_class == "hud_like" else 0.0
        static_penalty = 1.0 if displacement == (0, 0) else 0.0
        stats = AvatarCandidateStats(
            candidate_key=candidate_key,
            object_id=obj.object_id,
            color=obj.color,
            bbox=obj.bbox,
            centroid=obj.centroid,
            motion_hits=1 if overlap > 0 else 0,
            steps_seen=1,
            action_displacements={},
            persistence=accumulator._episode_presence.get(candidate_key, 0),
            hud_penalty=hud_penalty,
            static_penalty=static_penalty,
            local_diff_overlap=local_overlap,
            displacement=displacement,
        )
        if action_id is not None:
            stats.action_displacements.setdefault(action_id, []).append(displacement)
        stats_table.append(stats)

    candidate_rows: List[Dict[str, float]] = []
    candidates: List[ObjectRecordV2] = []
    rejection_reasons: List[Dict[str, object]] = []
    for stats in stats_table:
        motion_consistency = 1.0 if stats.motion_hits > 0 else 0.0
        displacement_consistency = 0.5
        if action_id is not None:
            disps = stats.action_displacements.get(action_id, [])
            displacement_consistency = _direction_consistency(disps, disps[-1] if disps else (0, 0))
        persistence_score = accumulator.persistence_score(stats.candidate_key)
        score = (
            0.35 * motion_consistency
            + 0.2 * displacement_consistency
            + 0.2 * stats.local_diff_overlap
            + 0.15 * persistence_score
            - 0.2 * stats.static_penalty
            - 0.2 * stats.hud_penalty
        )
        score = max(0.0, min(1.0, score))
        candidate_rows.append(
            {
                "candidate_key": stats.candidate_key,
                "object_id": stats.object_id,
                "action_id": action_id,
                "displacement": stats.displacement,
                "motion_consistency": motion_consistency,
                "displacement_consistency": displacement_consistency,
                "local_diff_overlap": stats.local_diff_overlap,
                "persistence": persistence_score,
                "static_penalty": stats.static_penalty,
                "hud_penalty": stats.hud_penalty,
                "score": score,
            }
        )
        if score >= cfg.avatar_motion_threshold:
            candidates.append(
                ObjectRecordV2(
                    schema_version=SCHEMA_VERSION,
                    object_id=f"avatar:{stats.object_id}",
                    game_id=game_id,
                    episode_id=episode_id,
                    bbox=stats.bbox,
                    centroid=stats.centroid,
                    color=int(stats.color),
                    area=(stats.bbox.width() * stats.bbox.height()),
                    aspect_ratio=float(stats.bbox.width()) / float(max(1, stats.bbox.height())),
                    object_class="avatar",
                    confidence=score,
                    evidence_refs=[],
                    first_seen_ref=None,
                    last_seen_ref=None,
                )
            )
        else:
            reasons = []
            if motion_consistency == 0.0:
                reasons.append("no_motion_overlap")
            if stats.static_penalty > 0.0:
                reasons.append("static_object")
            if stats.hud_penalty > 0.0:
                reasons.append("hud_like_region")
            if persistence_score < 0.34:
                reasons.append("low_persistence")
            if not reasons:
                reasons.append("below_threshold")
            rejection_reasons.append({"candidate_key": stats.candidate_key, "score": score, "reasons": reasons})

    candidates.sort(key=lambda c: c.confidence, reverse=True)
    if not candidates and not rejection_reasons and candidate_rows:
        rejection_reasons.append({"candidate_key": "none", "score": 0.0, "reasons": ["no_candidates_above_threshold"]})
    return candidates, candidate_rows, rejection_reasons
