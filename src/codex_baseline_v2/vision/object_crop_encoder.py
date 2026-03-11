from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from codex_baseline_v2.shared.schemas import BlackboardStateV2, CandidatePOIV2
from codex_baseline_v2.shared.vision_records import ObjectEmbeddingRecordV1

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


_SCHEMA_VERSION = "v2.3.3"
_EMBED_DIM = 24


def _stable_id(prefix: str, *parts: object) -> str:
    digest = hashlib.sha1("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}:{digest}"


def _hash_bucket(text: Optional[str], buckets: int = 97) -> float:
    if not text:
        return 0.0
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    value = int.from_bytes(digest[:4], "big") % buckets
    return float(value) / float(max(1, buckets - 1))


def _normalize_vector(values: Iterable[float]) -> List[float]:
    vector = [float(v) for v in values]
    norm = math.sqrt(sum(v * v for v in vector))
    if norm <= 1e-8:
        return vector
    return [v / norm for v in vector]


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


class TinyObjectCropEncoder:
    """Optional tiny CNN path for local crop arrays when torch is available."""

    def __init__(self, embedding_dim: int = _EMBED_DIM) -> None:
        self.embedding_dim = int(embedding_dim)
        self.available = torch is not None and nn is not None and F is not None
        self.model = None
        if self.available:
            self.model = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 12, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((3, 3)),
                nn.Flatten(),
                nn.Linear(12 * 3 * 3, self.embedding_dim),
            )
            self.model.eval()

    def encode_grid(self, grid: Sequence[Sequence[int]]) -> Optional[List[float]]:
        if not self.available or self.model is None or not grid or not grid[0]:
            return None
        height = len(grid)
        width = len(grid[0])
        flat = [[float(cell) / 9.0 for cell in row] for row in grid]
        tensor = torch.tensor(flat, dtype=torch.float32).reshape(1, 1, height, width)
        with torch.no_grad():
            vector = self.model(tensor).reshape(-1).tolist()
        return _normalize_vector(vector)


def _load_grid_from_ref(source_frame_ref: Optional[str]) -> Optional[List[List[int]]]:
    if not source_frame_ref:
        return None
    path = Path(source_frame_ref)
    if not path.exists():
        return None
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if isinstance(payload, list) and payload and isinstance(payload[0], list):
            return [[_safe_int(cell) for cell in row] for row in payload]
        return None
    if Image is None:
        return None
    try:
        image = Image.open(path).convert("L")
    except Exception:
        return None
    image = image.resize((16, 16))
    pixels = list(image.getdata())
    grid: List[List[int]] = []
    for row_idx in range(16):
        row = pixels[row_idx * 16 : (row_idx + 1) * 16]
        grid.append([min(9, int(round(v / 28.4))) for v in row])
    return grid


def _structured_vector(poi: CandidatePOIV2) -> List[float]:
    bbox = poi.bbox
    width = float(max(1, bbox.width()))
    height = float(max(1, bbox.height()))
    area = width * height
    centroid_x = float(poi.centroid[0]) / max(1.0, width + float(abs(bbox.x1)) + 1.0)
    centroid_y = float(poi.centroid[1]) / max(1.0, height + float(abs(bbox.y1)) + 1.0)
    values = [
        width / 32.0,
        height / 32.0,
        min(1.0, area / 256.0),
        min(4.0, width / max(1.0, height)) / 4.0,
        centroid_x,
        centroid_y,
        min(1.0, poi.confidence),
        min(1.0, poi.expected_information_gain),
        min(1.0, poi.type_confidence),
        min(1.0, poi.utility_confidence),
        min(1.0, float(poi.evidence_count) / 8.0),
        min(1.0, float(poi.interaction_count) / 8.0),
        _hash_bucket(poi.object_class),
        _hash_bucket(poi.source_type),
        _hash_bucket(poi.expected_interaction_type),
        _hash_bucket(poi.reachable_now),
        _hash_bucket(poi.area_id),
        _hash_bucket(poi.stable_entity_id),
        min(1.0, float(len(poi.rejection_reasons)) / 4.0),
        min(1.0, float(len(poi.demotion_reasons)) / 4.0),
        min(1.0, float(len(poi.linked_event_ids)) / 8.0),
        min(1.0, float(len(poi.linked_mechanic_hypothesis_ids)) / 8.0),
        math.sin(width),
        math.cos(height),
    ]
    return _normalize_vector(values[:_EMBED_DIM])


def _merge_vectors(primary: List[float], secondary: Optional[List[float]]) -> List[float]:
    if not secondary:
        return primary
    merged = []
    count = min(len(primary), len(secondary))
    for idx in range(count):
        merged.append((0.7 * primary[idx]) + (0.3 * secondary[idx]))
    merged.extend(primary[count:])
    return _normalize_vector(merged)


def _embedding_payload(poi: CandidatePOIV2, vector: List[float]) -> Dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "kind": "object",
        "vector": vector,
        "meta": {
            "object_id": poi.poi_id,
            "area_id": poi.area_id,
            "object_class": poi.object_class,
            "source_type": poi.source_type,
            "reachable_now": poi.reachable_now,
            "stable_entity_id": poi.stable_entity_id,
            "structured_signature_ref": poi.stable_entity_id or poi.object_class,
            "linked_event_ids": list(poi.linked_event_ids),
            "linked_mechanic_hypothesis_ids": list(poi.linked_mechanic_hypothesis_ids),
            "source_frame_ref": poi.last_seen_ref or poi.first_seen_ref or poi.poi_id,
        },
    }


def encode_object_records(blackboard: BlackboardStateV2, vision_root: Path | str) -> List[ObjectEmbeddingRecordV1]:
    root = Path(vision_root)
    embeddings_root = root / "object_embeddings"
    cnn = TinyObjectCropEncoder(_EMBED_DIM)
    records: List[ObjectEmbeddingRecordV1] = []
    seen: set[str] = set()
    for poi in blackboard.poi_table:
        if poi.poi_id in seen:
            continue
        seen.add(poi.poi_id)
        source_frame_ref = poi.last_seen_ref or poi.first_seen_ref or poi.poi_id
        grid = _load_grid_from_ref(source_frame_ref)
        structured = _structured_vector(poi)
        visual = cnn.encode_grid(grid) if grid is not None else None
        vector = _merge_vectors(structured, visual)
        embedding_id = _stable_id("objemb", blackboard.game_id, poi.poi_id, poi.area_id or "none")
        embedding_path = embeddings_root / f"{embedding_id}.json"
        _write_json(embedding_path, _embedding_payload(poi, vector))
        records.append(
            ObjectEmbeddingRecordV1(
                schema_version=_SCHEMA_VERSION,
                embedding_id=embedding_id,
                object_id=poi.poi_id,
                area_id=poi.area_id,
                source_frame_ref=source_frame_ref,
                embedding_path=str(embedding_path),
                structured_signature_ref=poi.stable_entity_id or poi.object_class,
                confidence=max(0.2, min(1.0, 0.55 + 0.25 * poi.confidence)),
            )
        )
    return records
