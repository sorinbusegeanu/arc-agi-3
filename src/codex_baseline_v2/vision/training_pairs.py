from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple

from codex_baseline_v2.shared.schemas import BlackboardStateV2
from codex_baseline_v2.shared.vision_records import EventEmbeddingRecordV1, ObjectEmbeddingRecordV1


def _stable_id(prefix: str, *parts: object) -> str:
    digest = hashlib.sha1("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}:{digest}"


def _step_index(ref: Optional[str]) -> Optional[int]:
    if not ref:
        return None
    match = re.search(r"step[_:= -]?(\d+)", ref)
    if match:
        return int(match.group(1))
    match = re.search(r":(\d+)$", ref)
    if match:
        return int(match.group(1))
    return None


def _poi_lookup(blackboard: BlackboardStateV2) -> Dict[str, Any]:
    return {poi.poi_id: poi for poi in blackboard.poi_table}


def _event_lookup(blackboard: BlackboardStateV2) -> Dict[str, Any]:
    return {event.event_id: event for event in blackboard.event_table}


def _append_pair(
    out: List[Dict[str, Any]],
    pair_type: str,
    lhs_id: str,
    rhs_id: str,
    label: int,
    reason: str,
    weight: float,
) -> None:
    out.append(
        {
            "pair_id": _stable_id("pair", pair_type, lhs_id, rhs_id, label, reason),
            "pair_type": pair_type,
            "lhs_ref_id": lhs_id,
            "rhs_ref_id": rhs_id,
            "label": int(label),
            "reason": reason,
            "weight": float(weight),
        }
    )


def build_training_pairs(
    blackboard: BlackboardStateV2,
    object_embeddings: List[ObjectEmbeddingRecordV1],
    event_embeddings: List[EventEmbeddingRecordV1],
) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    poi_by_id = _poi_lookup(blackboard)
    event_by_id = _event_lookup(blackboard)

    for idx, lhs in enumerate(object_embeddings):
        lhs_poi = poi_by_id.get(lhs.object_id)
        for rhs in object_embeddings[idx + 1 :]:
            rhs_poi = poi_by_id.get(rhs.object_id)
            if lhs_poi is None or rhs_poi is None:
                continue
            same_entity = bool(lhs_poi.stable_entity_id and lhs_poi.stable_entity_id == rhs_poi.stable_entity_id)
            lhs_step = _step_index(lhs.source_frame_ref)
            rhs_step = _step_index(rhs.source_frame_ref)
            adjacent = lhs_step is not None and rhs_step is not None and abs(lhs_step - rhs_step) <= 1
            if same_entity and adjacent:
                _append_pair(pairs, "object", lhs.embedding_id, rhs.embedding_id, 1, "same_tracked_object_adjacent", 1.0)
                continue
            far_apart = lhs.area_id is not None and rhs.area_id is not None and lhs.area_id != rhs.area_id
            unrelated = lhs_poi.object_class != rhs_poi.object_class or far_apart
            if unrelated and far_apart:
                _append_pair(pairs, "object", lhs.embedding_id, rhs.embedding_id, 0, "distant_area_unrelated_object", 0.8)

    support_to_mechanic: Dict[str, str] = {}
    for mechanic in blackboard.mechanic_hypotheses:
        for event_id in mechanic.support_event_ids:
            support_to_mechanic[event_id] = mechanic.hypothesis_id

    for idx, lhs in enumerate(event_embeddings):
        lhs_event = event_by_id.get(lhs.event_id)
        for rhs in event_embeddings[idx + 1 :]:
            rhs_event = event_by_id.get(rhs.event_id)
            if lhs_event is None or rhs_event is None:
                continue
            same_type = lhs.event_type == rhs.event_type
            same_mechanic = support_to_mechanic.get(lhs.event_id) and support_to_mechanic.get(lhs.event_id) == support_to_mechanic.get(rhs.event_id)
            if same_type and same_mechanic:
                _append_pair(pairs, "event", lhs.embedding_id, rhs.embedding_id, 1, "repeated_verified_mechanic_event", 1.0)
                continue
            different_type = lhs.event_type != rhs.event_type
            if different_type:
                _append_pair(pairs, "event", lhs.embedding_id, rhs.embedding_id, 0, "different_event_type", 0.7)
    return pairs
