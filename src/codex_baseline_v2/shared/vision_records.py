from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


def _to_dict(obj: object) -> Dict[str, Any]:
    return dict(obj.__dict__)


@dataclass(frozen=True)
class ObjectEmbeddingRecordV1:
    schema_version: str
    embedding_id: str
    object_id: str
    area_id: str | None
    source_frame_ref: str
    embedding_path: str
    structured_signature_ref: str | None
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ObjectEmbeddingRecordV1":
        return cls(
            schema_version=str(payload.get("schema_version", "v2.3.3")),
            embedding_id=str(payload.get("embedding_id", "")),
            object_id=str(payload.get("object_id", "")),
            area_id=payload.get("area_id"),
            source_frame_ref=str(payload.get("source_frame_ref", "")),
            embedding_path=str(payload.get("embedding_path", "")),
            structured_signature_ref=payload.get("structured_signature_ref"),
            confidence=float(payload.get("confidence", 0.0)),
        )


@dataclass(frozen=True)
class EventEmbeddingRecordV1:
    schema_version: str
    embedding_id: str
    event_id: str
    area_id: str | None
    before_crop_ref: str
    after_crop_ref: str
    embedding_path: str
    event_type: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EventEmbeddingRecordV1":
        return cls(
            schema_version=str(payload.get("schema_version", "v2.3.3")),
            embedding_id=str(payload.get("embedding_id", "")),
            event_id=str(payload.get("event_id", "")),
            area_id=payload.get("area_id"),
            before_crop_ref=str(payload.get("before_crop_ref", "")),
            after_crop_ref=str(payload.get("after_crop_ref", "")),
            embedding_path=str(payload.get("embedding_path", "")),
            event_type=str(payload.get("event_type", "unknown")),
            confidence=float(payload.get("confidence", 0.0)),
        )


@dataclass(frozen=True)
class PrototypeRecordV1:
    schema_version: str
    prototype_id: str
    prototype_type: str
    centroid_embedding_path: str
    member_embedding_ids: List[str]
    support_count: int
    confidence: float
    candidate_role_labels: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PrototypeRecordV1":
        return cls(
            schema_version=str(payload.get("schema_version", "v2.3.3")),
            prototype_id=str(payload.get("prototype_id", "")),
            prototype_type=str(payload.get("prototype_type", "unknown")),
            centroid_embedding_path=str(payload.get("centroid_embedding_path", "")),
            member_embedding_ids=list(payload.get("member_embedding_ids", [])),
            support_count=int(payload.get("support_count", 0)),
            confidence=float(payload.get("confidence", 0.0)),
            candidate_role_labels=list(payload.get("candidate_role_labels", [])),
        )


@dataclass(frozen=True)
class ReIDLinkRecordV1:
    schema_version: str
    link_id: str
    lhs_ref_id: str
    rhs_ref_id: str
    lhs_type: str
    rhs_type: str
    similarity_score: float
    structural_score: float
    combined_score: float
    decision: str
    evidence_refs: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ReIDLinkRecordV1":
        return cls(
            schema_version=str(payload.get("schema_version", "v2.3.3")),
            link_id=str(payload.get("link_id", "")),
            lhs_ref_id=str(payload.get("lhs_ref_id", "")),
            rhs_ref_id=str(payload.get("rhs_ref_id", "")),
            lhs_type=str(payload.get("lhs_type", "unknown")),
            rhs_type=str(payload.get("rhs_type", "unknown")),
            similarity_score=float(payload.get("similarity_score", 0.0)),
            structural_score=float(payload.get("structural_score", 0.0)),
            combined_score=float(payload.get("combined_score", 0.0)),
            decision=str(payload.get("decision", "unknown")),
            evidence_refs=list(payload.get("evidence_refs", [])),
        )
