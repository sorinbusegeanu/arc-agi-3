from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from codex_baseline_v2.shared.schemas import CandidatePOIV2


@dataclass
class POIState:
    poi: CandidatePOIV2
    status: str
    last_approached_round: int
    last_informative_round: int


def update_poi_lifecycle(pois: List[CandidatePOIV2], round_id: int) -> List[POIState]:
    states: List[POIState] = []
    for poi in pois:
        status = "unresolved"
        if poi.confidence >= 0.7:
            status = "high_value"
        if poi.object_class == "hud_like":
            status = "likely_hud"
        states.append(
            POIState(
                poi=poi,
                status=status,
                last_approached_round=round_id,
                last_informative_round=round_id if poi.expected_information_gain > 0.5 else max(0, round_id - 1),
            )
        )
    return states
