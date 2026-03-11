from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from codex_baseline_v2.shared.vision_records import EventEmbeddingRecordV1, ObjectEmbeddingRecordV1, PrototypeRecordV1, ReIDLinkRecordV1


_SCHEMA_VERSION = "v2.3.3"


def _read_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)


def _vector(path: str | Path) -> List[float]:
    payload = _read_json(path)
    vector = payload.get("vector", [])
    return [float(v) for v in vector]


def _cosine(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    if not lhs or not rhs:
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(lhs, rhs))
    lhs_norm = math.sqrt(sum(float(v) * float(v) for v in lhs))
    rhs_norm = math.sqrt(sum(float(v) * float(v) for v in rhs))
    if lhs_norm <= 1e-8 or rhs_norm <= 1e-8:
        return 0.0
    return dot / (lhs_norm * rhs_norm)


def _mean_vector(vectors: Iterable[Sequence[float]]) -> List[float]:
    vectors = [list(v) for v in vectors]
    if not vectors:
        return []
    width = min(len(v) for v in vectors)
    out = []
    for idx in range(width):
        out.append(sum(float(v[idx]) for v in vectors) / float(len(vectors)))
    norm = math.sqrt(sum(v * v for v in out))
    if norm <= 1e-8:
        return out
    return [v / norm for v in out]


def _prototype_id(prefix: str, index: int) -> str:
    return f"prototype:{prefix}:{index:03d}"


def _role_labels(members: List[Dict[str, Any]]) -> List[str]:
    labels = set()
    for member in members:
        meta = member.get("meta", {})
        object_class = meta.get("object_class")
        event_type = meta.get("event_type")
        if object_class:
            labels.add(str(object_class))
        if event_type:
            labels.add(str(event_type))
        if meta.get("linked_mechanic_hypothesis_ids"):
            labels.add("mechanic_linked")
        if meta.get("effect_signature_id"):
            labels.add("effect_signature")
        if meta.get("reachable_now") == "reachable":
            labels.add("reachable")
    return sorted(labels)


def _cluster_records(
    record_type: str,
    records: List[ObjectEmbeddingRecordV1] | List[EventEmbeddingRecordV1],
    threshold: float,
) -> List[Tuple[List[Any], List[float], List[Dict[str, Any]]]]:
    clusters: List[Tuple[List[Any], List[float], List[Dict[str, Any]]]] = []
    for record in records:
        payload = _read_json(record.embedding_path)
        vector = [float(v) for v in payload.get("vector", [])]
        if not clusters:
            clusters.append(([record], vector, [payload]))
            continue
        best_idx = -1
        best_score = -1.0
        for idx, (_, centroid, _) in enumerate(clusters):
            score = _cosine(vector, centroid)
            if score > best_score:
                best_idx = idx
                best_score = score
        if best_idx >= 0 and best_score >= threshold:
            members, centroid, payloads = clusters[best_idx]
            members.append(record)
            payloads.append(payload)
            clusters[best_idx] = (members, _mean_vector([centroid, vector]), payloads)
        else:
            clusters.append(([record], vector, [payload]))
    return clusters


def build_prototype_memory(
    object_embeddings: List[ObjectEmbeddingRecordV1],
    event_embeddings: List[EventEmbeddingRecordV1],
    vision_root: str | Path,
) -> List[PrototypeRecordV1]:
    root = Path(vision_root)
    centroid_root = root / "prototypes"
    prototypes: List[PrototypeRecordV1] = []

    object_clusters = _cluster_records("object", object_embeddings, threshold=0.92)
    event_clusters = _cluster_records("event", event_embeddings, threshold=0.9)

    for cluster_idx, (members, centroid, payloads) in enumerate(object_clusters):
        prototype_id = _prototype_id("object", cluster_idx)
        centroid_path = centroid_root / f"{prototype_id}.json"
        _write_json(centroid_path, {"schema_version": _SCHEMA_VERSION, "prototype_id": prototype_id, "vector": centroid})
        confidence = min(1.0, 0.45 + (0.1 * len(members)))
        prototypes.append(
            PrototypeRecordV1(
                schema_version=_SCHEMA_VERSION,
                prototype_id=prototype_id,
                prototype_type="object",
                centroid_embedding_path=str(centroid_path),
                member_embedding_ids=[member.embedding_id for member in members],
                support_count=len(members),
                confidence=confidence,
                candidate_role_labels=_role_labels(payloads),
            )
        )

    base_idx = len(prototypes)
    for cluster_offset, (members, centroid, payloads) in enumerate(event_clusters):
        prototype_id = _prototype_id("event", base_idx + cluster_offset)
        centroid_path = centroid_root / f"{prototype_id}.json"
        _write_json(centroid_path, {"schema_version": _SCHEMA_VERSION, "prototype_id": prototype_id, "vector": centroid})
        confidence = min(1.0, 0.45 + (0.1 * len(members)))
        prototypes.append(
            PrototypeRecordV1(
                schema_version=_SCHEMA_VERSION,
                prototype_id=prototype_id,
                prototype_type="event",
                centroid_embedding_path=str(centroid_path),
                member_embedding_ids=[member.embedding_id for member in members],
                support_count=len(members),
                confidence=confidence,
                candidate_role_labels=_role_labels(payloads),
            )
        )
    return prototypes


def nearest_prototypes(
    embedding_path: str | Path,
    prototypes: List[PrototypeRecordV1],
    prototype_type: str,
    top_k: int = 3,
) -> List[Tuple[PrototypeRecordV1, float]]:
    embedding = _vector(embedding_path)
    scored = []
    for prototype in prototypes:
        if prototype.prototype_type != prototype_type:
            continue
        score = _cosine(embedding, _vector(prototype.centroid_embedding_path))
        scored.append((prototype, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[: max(0, int(top_k))]


def save_prototype_memory(
    vision_root: str | Path,
    object_embeddings: List[ObjectEmbeddingRecordV1],
    event_embeddings: List[EventEmbeddingRecordV1],
    prototypes: List[PrototypeRecordV1],
    reid_links: List[ReIDLinkRecordV1],
    training_pairs: List[Dict[str, Any]],
) -> None:
    root = Path(vision_root)
    _write_json(root / "object_embeddings.json", [record.to_dict() for record in object_embeddings])
    _write_json(root / "event_embeddings.json", [record.to_dict() for record in event_embeddings])
    _write_json(root / "prototypes.json", [record.to_dict() for record in prototypes])
    _write_json(root / "reid_links.json", [record.to_dict() for record in reid_links])
    _write_json(root / "training_pairs.json", list(training_pairs))
