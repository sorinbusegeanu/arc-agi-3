from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _TrackState:
    track_id: str
    signature: str
    history: list[dict]
    confidence: float
    missed_steps: int


def _distance(lhs: list[float], rhs: list[float]) -> float:
    return abs(float(lhs[0]) - float(rhs[0])) + abs(float(lhs[1]) - float(rhs[1]))


def _overlaps_active(candidate: dict, step_summary: dict) -> bool:
    centroid = candidate.get("centroid")
    if not isinstance(centroid, list) or len(centroid) != 2:
        return False
    for region in step_summary.get("active_regions", []):
        bbox = region.get("bbox", {})
        if bbox.get("x1", 0) <= centroid[0] <= bbox.get("x2", -1) and bbox.get("y1", 0) <= centroid[1] <= bbox.get("y2", -1):
            return True
    return False


def _score_candidate(track: _TrackState, candidate: dict, step_summary: dict) -> float:
    score = float(candidate.get("score", 0.0))
    if candidate.get("signature") == track.signature:
        score += 0.2
    if track.history:
        score += max(0.0, 0.4 - (_distance(track.history[-1]["centroid"], candidate["centroid"]) / 10.0))
    if _overlaps_active(candidate, step_summary):
        score += 0.6
    elif step_summary.get("active_regions"):
        score -= 0.25
    if "tiny" in candidate.get("type_hints", []):
        score -= 0.15
    if "border_touching" in candidate.get("type_hints", []):
        score -= 0.2
    area = int(candidate.get("area", 0))
    if area > 12:
        score -= min(0.5, area / 80.0)
    return score


def _track_rank(track: _TrackState) -> tuple[float, float, float, float, float, str]:
    points = [tuple(int(round(value)) for value in entry["centroid"]) for entry in track.history if isinstance(entry.get("centroid"), list)]
    unique_positions = len(set(points))
    total_motion = sum(_distance(previous["centroid"], current["centroid"]) for previous, current in zip(track.history, track.history[1:]))
    active_hits = sum(1 for entry in track.history if entry.get("active_overlap"))
    tiny_hits = sum(1 for entry in track.history if entry.get("tiny"))
    areas = [int(entry.get("area", 0)) for entry in track.history if int(entry.get("area", 0)) > 0]
    average_area = sum(areas) / float(len(areas)) if areas else 999.0
    return (
        float(active_hits),
        float(unique_positions),
        -float(average_area),
        float(total_motion),
        float(track.confidence) - (0.15 * tiny_hits),
        track.track_id,
    )


def track_avatar(step_summaries: list[dict]) -> dict:
    tracks: list[_TrackState] = []
    next_track_index = 0
    main_track_id: str | None = None
    alternates: list[str] = []
    per_step: list[dict] = []

    for step_idx, summary in enumerate(step_summaries):
        candidates = list(summary.get("avatar_candidates", []))
        candidates.sort(key=lambda row: (-float(row.get("score", 0.0)), row.get("object_id", "")))
        used_candidates: set[str] = set()

        for track in tracks:
            best = None
            best_score = -1.0
            for candidate in candidates:
                object_id = str(candidate.get("object_id"))
                if object_id in used_candidates:
                    continue
                candidate_score = _score_candidate(track, candidate, summary)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best = candidate
            if best is None or best_score < 0.25:
                track.missed_steps += 1
                track.confidence = max(0.0, track.confidence - 0.15)
                continue
            used_candidates.add(str(best["object_id"]))
            track.signature = str(best["signature"])
            track.history.append(
                {
                    "step_idx": step_idx,
                    "centroid": list(best["centroid"]),
                    "bbox": dict(best["bbox"]),
                    "score": float(best_score),
                    "active_overlap": _overlaps_active(best, summary),
                    "tiny": "tiny" in best.get("type_hints", []),
                    "area": int(best.get("area", 0)),
                }
            )
            track.confidence = min(1.0, (0.55 * track.confidence) + (0.45 * min(1.0, best_score)))
            track.missed_steps = 0

        for candidate in candidates:
            if str(candidate.get("object_id")) in used_candidates:
                continue
            new_track = _TrackState(
                track_id=f"avatar:{next_track_index}",
                signature=str(candidate["signature"]),
                history=[
                    {
                        "step_idx": step_idx,
                        "centroid": list(candidate["centroid"]),
                        "bbox": dict(candidate["bbox"]),
                        "score": float(candidate.get("score", 0.0)),
                        "active_overlap": _overlaps_active(candidate, summary),
                        "tiny": "tiny" in candidate.get("type_hints", []),
                        "area": int(candidate.get("area", 0)),
                    }
                ],
                confidence=max(0.2, float(candidate.get("score", 0.0))),
                missed_steps=0,
            )
            next_track_index += 1
            tracks.append(new_track)

        living_tracks = [track for track in tracks if track.missed_steps <= 2 and track.history]
        living_tracks.sort(
            key=lambda track: (
                -_track_rank(track)[0],
                -_track_rank(track)[1],
                -_track_rank(track)[2],
                -_track_rank(track)[3],
                -_track_rank(track)[4],
                track.track_id,
            )
        )
        main_track_id = living_tracks[0].track_id if living_tracks else None
        alternates = [track.track_id for track in living_tracks[1:3]]
        main_step = next((track.history[-1] for track in living_tracks if track.track_id == main_track_id), None)
        per_step.append(
            {
                "step_idx": step_idx,
                "main_track_id": main_track_id,
                "alternate_track_ids": list(alternates),
                "main_centroid": list(main_step["centroid"]) if main_step is not None else None,
                "track_count": len(living_tracks),
            }
        )
        tracks = living_tracks + [track for track in tracks if track not in living_tracks and track.history]

    exported_tracks = []
    for track in tracks:
        rank = _track_rank(track)
        exported_tracks.append(
            {
                "track_id": track.track_id,
                "signature": track.signature,
                "confidence": float(track.confidence),
                "history": list(track.history),
                "steps_seen": len(track.history),
                "last_centroid": list(track.history[-1]["centroid"]) if track.history else None,
                "is_main": track.track_id == main_track_id,
                "track_rank": {
                    "active_hits": rank[0],
                    "unique_positions": rank[1],
                    "average_area_bias": -rank[2],
                    "motion": rank[3],
                    "score": rank[4],
                },
            }
        )
    exported_tracks.sort(key=lambda row: (-float(row["confidence"]), -int(row["steps_seen"]), row["track_id"]))
    return {
        "main_track_id": main_track_id,
        "alternate_track_ids": alternates,
        "tracks": exported_tracks,
        "per_step": per_step,
    }
