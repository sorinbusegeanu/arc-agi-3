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


def _score_candidate(track: _TrackState, candidate: dict, step_summary: dict) -> float:
    score = float(candidate.get("score", 0.0))
    if candidate.get("signature") == track.signature:
        score += 0.35
    if track.history:
        score += max(0.0, 0.35 - (_distance(track.history[-1]["centroid"], candidate["centroid"]) / 12.0))
    if any(
        region["bbox"]["x1"] <= candidate["centroid"][0] <= region["bbox"]["x2"]
        and region["bbox"]["y1"] <= candidate["centroid"][1] <= region["bbox"]["y2"]
        for region in step_summary.get("active_regions", [])
    ):
        score += 0.1
    return score


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
                -float(track.confidence),
                -len(track.history),
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
        exported_tracks.append(
            {
                "track_id": track.track_id,
                "signature": track.signature,
                "confidence": float(track.confidence),
                "history": list(track.history),
                "steps_seen": len(track.history),
                "last_centroid": list(track.history[-1]["centroid"]) if track.history else None,
                "is_main": track.track_id == main_track_id,
            }
        )
    exported_tracks.sort(key=lambda row: (-float(row["confidence"]), -int(row["steps_seen"]), row["track_id"]))
    return {
        "main_track_id": main_track_id,
        "alternate_track_ids": alternates,
        "tracks": exported_tracks,
        "per_step": per_step,
    }
