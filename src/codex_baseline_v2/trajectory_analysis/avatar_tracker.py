from __future__ import annotations

from dataclasses import dataclass
from typing import List

from codex_baseline_v2.shared.schemas import ObjectRecordV2


@dataclass
class AvatarHypothesis:
    object: ObjectRecordV2
    score: float


def track_avatars(candidates: List[ObjectRecordV2]) -> List[AvatarHypothesis]:
    ranked: List[AvatarHypothesis] = []
    for cand in candidates:
        score = cand.confidence
        ranked.append(AvatarHypothesis(object=cand, score=score))
    ranked.sort(key=lambda h: h.score, reverse=True)
    return ranked
