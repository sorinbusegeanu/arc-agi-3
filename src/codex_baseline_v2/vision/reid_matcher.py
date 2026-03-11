from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

from codex_baseline_v2.shared.schemas import BlackboardStateV2
from codex_baseline_v2.shared.vision_records import EventEmbeddingRecordV1, ObjectEmbeddingRecordV1, PrototypeRecordV1, ReIDLinkRecordV1


_SCHEMA_VERSION = "v2.3.3"


def _stable_id(prefix: str, *parts: object) -> str:
    digest = hashlib.sha1("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}:{digest}"


def _read_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _vector(path: str | Path) -> List[float]:
    payload = _read_json(path)
    return [float(v) for v in payload.get("vector", [])]


def _cosine(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    if not lhs or not rhs:
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(lhs, rhs))
    lhs_norm = math.sqrt(sum(float(v) * float(v) for v in lhs))
    rhs_norm = math.sqrt(sum(float(v) * float(v) for v in rhs))
    if lhs_norm <= 1e-8 or rhs_norm <= 1e-8:
        return 0.0
    return dot / (lhs_norm * rhs_norm)


def _prototype_membership(record_id: str, prototypes: List[PrototypeRecordV1]) -> List[str]:
    out = []
    for prototype in prototypes:
        if record_id in prototype.member_embedding_ids:
            out.append(prototype.prototype_id)
    return out


def _build_context(blackboard: BlackboardStateV2) -> Dict[str, Dict[str, Any]]:
    poi_by_id = {poi.poi_id: poi for poi in blackboard.poi_table}
    event_by_id = {event.event_id: event for event in blackboard.event_table}
    mechanic_by_event: Dict[str, List[str]] = {}
    for mechanic in blackboard.mechanic_hypotheses:
        for event_id in mechanic.support_event_ids:
            mechanic_by_event.setdefault(event_id, []).append(mechanic.hypothesis_id)
    return {
        "poi_by_id": poi_by_id,
        "event_by_id": event_by_id,
        "mechanic_by_event": mechanic_by_event,
    }


def _object_structural_score(lhs_meta: Dict[str, Any], rhs_meta: Dict[str, Any], ctx: Dict[str, Dict[str, Any]]) -> float:
    lhs_poi = ctx["poi_by_id"].get(lhs_meta.get("object_id"))
    rhs_poi = ctx["poi_by_id"].get(rhs_meta.get("object_id"))
    if lhs_poi is None or rhs_poi is None:
        return 0.0
    score = 0.0
    if lhs_poi.object_class == rhs_poi.object_class:
        score += 0.45
    if lhs_poi.stable_entity_id and lhs_poi.stable_entity_id == rhs_poi.stable_entity_id:
        score += 0.35
    overlap = set(lhs_poi.linked_mechanic_hypothesis_ids) & set(rhs_poi.linked_mechanic_hypothesis_ids)
    if overlap:
        score += 0.2
    return min(1.0, score)


def _event_structural_score(lhs_meta: Dict[str, Any], rhs_meta: Dict[str, Any], ctx: Dict[str, Dict[str, Any]]) -> float:
    lhs_event = ctx["event_by_id"].get(lhs_meta.get("event_id"))
    rhs_event = ctx["event_by_id"].get(rhs_meta.get("event_id"))
    if lhs_event is None or rhs_event is None:
        return 0.0
    score = 0.0
    if lhs_event.event_type == rhs_event.event_type:
        score += 0.45
    if lhs_event.effect_signature_id and lhs_event.effect_signature_id == rhs_event.effect_signature_id:
        score += 0.35
    if set(ctx["mechanic_by_event"].get(lhs_event.event_id, [])) & set(ctx["mechanic_by_event"].get(rhs_event.event_id, [])):
        score += 0.2
    return min(1.0, score)


def _area_context_score(lhs_meta: Dict[str, Any], rhs_meta: Dict[str, Any]) -> float:
    lhs_area = lhs_meta.get("area_id")
    rhs_area = rhs_meta.get("area_id")
    if lhs_area and rhs_area and lhs_area == rhs_area:
        return 1.0
    if lhs_area and rhs_area and lhs_area != rhs_area:
        return 0.6
    return 0.4


def _role_compatibility(lhs_meta: Dict[str, Any], rhs_meta: Dict[str, Any], ctx: Dict[str, Dict[str, Any]]) -> float:
    if lhs_meta.get("object_id") and rhs_meta.get("object_id"):
        lhs_poi = ctx["poi_by_id"].get(lhs_meta.get("object_id"))
        rhs_poi = ctx["poi_by_id"].get(rhs_meta.get("object_id"))
        if lhs_poi is None or rhs_poi is None:
            return 0.0
        overlap = set(lhs_poi.linked_mechanic_hypothesis_ids) & set(rhs_poi.linked_mechanic_hypothesis_ids)
        if overlap:
            return 1.0
        if lhs_poi.expected_interaction_type == rhs_poi.expected_interaction_type:
            return 0.7
        return 0.2
    if lhs_meta.get("event_id") and rhs_meta.get("event_id"):
        lhs_roles = set(ctx["mechanic_by_event"].get(lhs_meta.get("event_id"), []))
        rhs_roles = set(ctx["mechanic_by_event"].get(rhs_meta.get("event_id"), []))
        if lhs_roles & rhs_roles:
            return 1.0
        if lhs_meta.get("event_type") == rhs_meta.get("event_type"):
            return 0.7
        return 0.2
    return 0.0


def _decision(combined_score: float, structural_score: float, similarity_score: float) -> str:
    if combined_score >= 0.75 and structural_score >= 0.35 and similarity_score >= 0.45:
        return "match"
    if combined_score >= 0.58:
        return "candidate"
    return "reject"


def _record_meta(embedding_path: str | Path) -> Dict[str, Any]:
    return dict(_read_json(embedding_path).get("meta", {}))


def _build_links_for_records(
    record_type: str,
    records: List[ObjectEmbeddingRecordV1] | List[EventEmbeddingRecordV1],
    prototypes: List[PrototypeRecordV1],
    ctx: Dict[str, Dict[str, Any]],
) -> List[ReIDLinkRecordV1]:
    out: List[ReIDLinkRecordV1] = []
    for idx, lhs in enumerate(records):
        lhs_vec = _vector(lhs.embedding_path)
        lhs_meta = _record_meta(lhs.embedding_path)
        lhs_membership = set(_prototype_membership(lhs.embedding_id, prototypes))
        for rhs in records[idx + 1 :]:
            rhs_vec = _vector(rhs.embedding_path)
            rhs_meta = _record_meta(rhs.embedding_path)
            rhs_membership = set(_prototype_membership(rhs.embedding_id, prototypes))
            similarity_score = max(0.0, min(1.0, _cosine(lhs_vec, rhs_vec)))
            if record_type == "object":
                structural_score = _object_structural_score(lhs_meta, rhs_meta, ctx)
            else:
                structural_score = _event_structural_score(lhs_meta, rhs_meta, ctx)
            area_score = _area_context_score(lhs_meta, rhs_meta)
            role_score = _role_compatibility(lhs_meta, rhs_meta, ctx)
            prototype_bonus = 1.0 if lhs_membership & rhs_membership else 0.0
            combined_score = (
                (0.45 * similarity_score)
                + (0.25 * structural_score)
                + (0.15 * area_score)
                + (0.1 * role_score)
                + (0.05 * prototype_bonus)
            )
            decision = _decision(combined_score, structural_score, similarity_score)
            if decision == "reject":
                continue
            evidence_refs = [lhs.embedding_id, rhs.embedding_id]
            evidence_refs.extend(sorted(lhs_membership & rhs_membership))
            out.append(
                ReIDLinkRecordV1(
                    schema_version=_SCHEMA_VERSION,
                    link_id=_stable_id("reid", record_type, lhs.embedding_id, rhs.embedding_id),
                    lhs_ref_id=lhs.embedding_id,
                    rhs_ref_id=rhs.embedding_id,
                    lhs_type=record_type,
                    rhs_type=record_type,
                    similarity_score=similarity_score,
                    structural_score=structural_score,
                    combined_score=combined_score,
                    decision=decision,
                    evidence_refs=evidence_refs,
                )
            )
    return out


def build_reid_links(
    blackboard: BlackboardStateV2,
    object_embeddings: List[ObjectEmbeddingRecordV1],
    event_embeddings: List[EventEmbeddingRecordV1],
    prototypes: List[PrototypeRecordV1],
) -> List[ReIDLinkRecordV1]:
    ctx = _build_context(blackboard)
    links = []
    links.extend(_build_links_for_records("object", object_embeddings, prototypes, ctx))
    links.extend(_build_links_for_records("event", event_embeddings, prototypes, ctx))
    links.sort(key=lambda item: (item.decision != "match", -item.combined_score, item.link_id))
    return links
