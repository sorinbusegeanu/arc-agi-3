from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List

from codex_baseline_v2.shared.schemas import TrajectoryEpisodeV2


def export_avatar_debug(episodes: List[TrajectoryEpisodeV2]) -> Dict[str, object]:
    candidate_rows: List[Dict[str, object]] = []
    rejection_reasons: List[Dict[str, object]] = []
    displacement_vectors = defaultdict(list)
    for episode in episodes:
        for step in episode.steps:
            summary = step.observation_summary
            if summary is None:
                continue
            candidate_rows.extend(summary.avatar_candidate_table)
            rejection_reasons.extend(summary.avatar_rejection_reasons)
            for row in summary.avatar_candidate_table:
                action_id = row.get("action_id")
                disp = row.get("displacement")
                if action_id is None or disp is None:
                    continue
                displacement_vectors[int(action_id)].append(tuple(disp))
    ranked = sorted(candidate_rows, key=lambda r: r.get("score", 0.0), reverse=True)
    evidence_counts = Counter([r.get("candidate_key") for r in candidate_rows if r.get("candidate_key")])
    displacement_summary = {
        str(action_id): {
            "count": len(vectors),
            "sample": vectors[:5],
        }
        for action_id, vectors in displacement_vectors.items()
    }
    return {
        "ranked_candidates": ranked[:50],
        "evidence_counts": dict(evidence_counts),
        "action_conditioned_displacement_summary": displacement_summary,
        "rejection_reasons": rejection_reasons,
        "candidate_count": len(candidate_rows),
    }
