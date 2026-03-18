from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from v3_1.analysis.observation_summary import summarize_observation
from v3_1.execution.option_execution import action_alias
from v3_1.utils.ids import stable_digest


ACTION_DELTAS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


@dataclass(frozen=True)
class LiveAvatarBelief:
    mode_status: str
    avatar_status: str
    cell: list[int] | None
    confidence: float
    source: str
    step_index: int | None
    ambiguous: bool
    candidate_cells: tuple[list[int], ...] = ()
    last_action: str | None = None
    last_observation_hash: str = ""


@dataclass(frozen=True)
class AvatarInferenceResult:
    mode_status: str
    avatar_status: str
    best_cell: list[int] | None
    candidate_cells: tuple[list[int], ...]
    confidence: float
    ambiguous: bool
    source: str
    observation_hash: str


def _observation_hash(observation: Any) -> str:
    try:
        return stable_digest(observation)
    except Exception:
        return "unhashable_observation"


def _coerce_cell(value: Any) -> list[int] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return [int(round(float(value[0]))), int(round(float(value[1])))]
    return None


def _distance(lhs: list[int] | None, rhs: list[int] | None) -> float:
    if lhs is None or rhs is None:
        return 999.0
    return abs(int(lhs[0]) - int(rhs[0])) + abs(int(lhs[1]) - int(rhs[1]))


def _cell_in_bbox(cell: list[int] | None, bbox: dict[str, int]) -> bool:
    if cell is None:
        return False
    return int(bbox.get("x1", 0)) <= int(cell[0]) <= int(bbox.get("x2", -1)) and int(bbox.get("y1", 0)) <= int(cell[1]) <= int(bbox.get("y2", -1))


def _cell_in_regions(cell: list[int] | None, regions: list[dict]) -> bool:
    if cell is None:
        return False
    return any(_cell_in_bbox(cell, dict(region.get("bbox", {}) or {})) for region in list(regions or []))


def _static_scan_cell(observation: Any) -> list[int] | None:
    if not isinstance(observation, list):
        return None
    for y, row in enumerate(observation):
        if not isinstance(row, list):
            continue
        for x, value in enumerate(row):
            try:
                if int(value) == 1:
                    return [x, y]
            except Exception:
                continue
    return None


def _candidate_cells(summary: dict) -> list[list[int]]:
    cells: list[list[int]] = []
    for row in list(summary.get("avatar_candidates", []) or []):
        cell = _coerce_cell(row.get("centroid"))
        if cell is not None and cell not in cells:
            cells.append(cell)
    for region in list(summary.get("active_regions", []) or []):
        cell = _coerce_cell(dict(region.get("bbox", {}) or {}).get("centroid")) or _coerce_cell(region.get("centroid"))
        if cell is not None and cell not in cells:
            cells.append(cell)
    return cells


def _predict_cell(prior_cell: list[int] | None, action_name: str | None) -> list[int] | None:
    if prior_cell is None or not action_name:
        return None
    delta = ACTION_DELTAS.get(str(action_name).lower())
    if delta is None:
        return list(prior_cell)
    return [int(prior_cell[0]) + int(delta[0]), int(prior_cell[1]) + int(delta[1])]


def _validated_direct_info(info: dict | None, *, active_regions: list[dict], prior_cell: list[int] | None) -> list[int] | None:
    avatar = _coerce_cell(dict(info or {}).get("avatar"))
    if avatar is None:
        return None
    if _cell_in_regions(avatar, active_regions) or _distance(avatar, prior_cell) <= 2.0:
        return avatar
    return None


def _motion_candidate_score(
    *,
    cell: list[int],
    summary: dict,
    prior_cell: list[int] | None,
    predicted_cell: list[int] | None,
    action_name: str | None,
) -> float:
    score = 0.0
    candidates = list(summary.get("avatar_candidates", []) or [])
    for row in candidates:
        centroid = _coerce_cell(row.get("centroid"))
        if centroid == cell:
            score += float(row.get("score", 0.0) or 0.0)
            if "candidate_avatar" in list(row.get("type_hints", []) or []):
                score += 0.25
            if "mobile_candidate" in list(row.get("type_hints", []) or []):
                score += 0.2
            break
    if _cell_in_regions(cell, list(summary.get("active_regions", []) or [])):
        score += 0.8
    if prior_cell is not None:
        score += max(0.0, 0.7 - (_distance(cell, prior_cell) / 4.0))
    if predicted_cell is not None:
        score += max(0.0, 1.2 - (_distance(cell, predicted_cell) / 2.0))
        if action_name in ACTION_DELTAS and _distance(cell, predicted_cell) <= 1.0:
            score += 0.3
    return score


class LiveAvatarTracker:
    def __init__(self) -> None:
        self._movement_evidence_count = 0
        self._non_avatar_evidence_count = 0
        self._directional_attempt_count = 0
        self._confirmed_mode_status = "unknown"
        self._belief = LiveAvatarBelief(
            mode_status="unknown",
            avatar_status="unknown",
            cell=None,
            confidence=0.0,
            source="unknown",
            step_index=None,
            ambiguous=False,
            candidate_cells=(),
            last_action=None,
            last_observation_hash="",
        )

    def reset(self, initial_observation, info=None) -> LiveAvatarBelief:
        self._movement_evidence_count = 0
        self._non_avatar_evidence_count = 0
        self._directional_attempt_count = 0
        summary = summarize_observation(initial_observation, None)
        candidates = _candidate_cells(summary)
        direct_cell = _validated_direct_info(info, active_regions=list(summary.get("active_regions", []) or []), prior_cell=None)
        if candidates:
            best = candidates[0]
            confidence = max(0.35, min(0.65, float(list(summary.get("avatar_candidates", []) or [{}])[0].get("score", 0.0) or 0.0) + 0.15))
            source = "motion"
        elif direct_cell is not None:
            best = direct_cell
            confidence = 0.3
            source = "direct_info"
        else:
            best = _static_scan_cell(initial_observation)
            confidence = 0.15 if best is not None else 0.0
            source = "static_fallback" if best is not None else "unknown"
        mode_status = "unknown"
        avatar_status = "unknown"
        if self._confirmed_mode_status == "movement_avatar" and best is not None:
            mode_status = "movement_avatar"
            avatar_status = "present"
            confidence = max(float(confidence), 0.45)
        self._belief = LiveAvatarBelief(
            mode_status=mode_status,
            avatar_status=avatar_status if best is not None else "unknown",
            cell=best,
            confidence=float(confidence),
            source=source,
            step_index=0,
            ambiguous=False,
            candidate_cells=tuple(candidates[:8]),
            last_action=None,
            last_observation_hash=_observation_hash(initial_observation),
        )
        return self._belief

    def update(self, prev_observation, current_observation, action, info=None, step_index=None) -> AvatarInferenceResult:
        previous_belief = self._belief
        summary = summarize_observation(current_observation, prev_observation)
        active_regions = list(summary.get("active_regions", []) or [])
        action_name = action_alias(action) if isinstance(action, dict) else None
        if action_name in ACTION_DELTAS:
            self._directional_attempt_count += 1
        predicted = _predict_cell(previous_belief.cell, action_name)
        candidates = _candidate_cells(summary)

        scored: list[tuple[float, list[int]]] = []
        for cell in candidates:
            scored.append(
                (
                    _motion_candidate_score(
                        cell=cell,
                        summary=summary,
                        prior_cell=previous_belief.cell,
                        predicted_cell=predicted,
                        action_name=action_name,
                    ),
                    cell,
                )
            )
        scored.sort(key=lambda row: (-row[0], row[1][1], row[1][0]))

        best_cell = None
        best_confidence = 0.0
        ambiguous = False
        source = "unknown"
        if scored:
            top_score, top_cell = scored[0]
            runner_up = scored[1][0] if len(scored) > 1 else -999.0
            ambiguous = abs(top_score - runner_up) < 0.2
            if previous_belief.cell is not None and (_distance(top_cell, previous_belief.cell) <= 2.0 or _distance(top_cell, predicted) <= 1.0):
                best_cell = list(top_cell)
                best_confidence = min(0.95, max(0.45, top_score / 2.5))
                source = "motion" if action_name not in ACTION_DELTAS else "action_motion"
                if ambiguous:
                    best_confidence *= 0.7
            elif predicted is not None and _cell_in_regions(predicted, active_regions):
                best_cell = list(predicted)
                best_confidence = 0.5
                source = "action_motion"
        if best_cell is None and predicted is not None and (_cell_in_regions(predicted, active_regions) or not active_regions):
            best_cell = list(predicted)
            best_confidence = 0.4 if active_regions else 0.25
            source = "action_motion"

        direct_cell = _validated_direct_info(info, active_regions=active_regions, prior_cell=previous_belief.cell)
        if best_cell is None and direct_cell is not None:
            best_cell = list(direct_cell)
            best_confidence = 0.25
            source = "direct_info"

        if best_cell is None and previous_belief.cell is not None:
            best_cell = list(previous_belief.cell)
            best_confidence = max(0.0, float(previous_belief.confidence) * 0.75)
            source = previous_belief.source

        if best_cell is None:
            static_cell = _static_scan_cell(current_observation)
            if static_cell is not None and previous_belief.mode_status == "movement_avatar":
                best_cell = list(static_cell)
                best_confidence = 0.1
                source = "static_fallback"

        if source in {"motion", "action_motion"} and best_cell is not None and best_confidence >= 0.45:
            self._movement_evidence_count += 1
        elif action_name in ACTION_DELTAS and active_regions and all(int(dict(region.get("bbox", {})).get("x2", 0)) - int(dict(region.get("bbox", {})).get("x1", 0)) > 6 or int(dict(region.get("bbox", {})).get("y2", 0)) - int(dict(region.get("bbox", {})).get("y1", 0)) > 6 for region in active_regions):
            self._non_avatar_evidence_count += 1
        elif action_name in ACTION_DELTAS and best_cell is None:
            self._non_avatar_evidence_count += 1

        if self._movement_evidence_count >= 2:
            mode_status = "movement_avatar"
            avatar_status = "present" if best_cell is not None and best_confidence > 0.0 else "unknown"
            self._confirmed_mode_status = "movement_avatar"
        elif self._directional_attempt_count >= 3 and self._non_avatar_evidence_count >= 2 and self._movement_evidence_count == 0:
            mode_status = "cursor_or_click" if active_regions else "global_action_only"
            avatar_status = "absent"
            best_cell = None
            best_confidence = 0.0
            source = "unknown"
            if self._confirmed_mode_status == "unknown":
                self._confirmed_mode_status = mode_status
        else:
            mode_status = self._confirmed_mode_status if self._confirmed_mode_status == "movement_avatar" else "unknown"
            avatar_status = "present" if mode_status == "movement_avatar" and best_cell is not None and best_confidence > 0.0 else ("unknown" if best_cell is not None else "unknown")

        result = AvatarInferenceResult(
            mode_status=mode_status,
            avatar_status=avatar_status,
            best_cell=best_cell,
            candidate_cells=tuple(candidates[:8]),
            confidence=float(best_confidence),
            ambiguous=bool(ambiguous),
            source=source,
            observation_hash=_observation_hash(current_observation),
        )
        self._belief = LiveAvatarBelief(
            mode_status=mode_status,
            avatar_status=avatar_status,
            cell=list(best_cell) if best_cell is not None else None,
            confidence=float(best_confidence),
            source=source,
            step_index=step_index,
            ambiguous=bool(ambiguous),
            candidate_cells=tuple(candidates[:8]),
            last_action=action_name,
            last_observation_hash=result.observation_hash,
        )
        return result

    def current_belief(self) -> LiveAvatarBelief:
        return self._belief

    def has_confident_avatar(self) -> bool:
        return self._belief.mode_status == "movement_avatar" and self._belief.avatar_status == "present" and self._belief.cell is not None and float(self._belief.confidence) >= 0.6 and not bool(self._belief.ambiguous)
