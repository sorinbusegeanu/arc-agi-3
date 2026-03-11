from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from codex_baseline_v2.shared.schemas import BlackboardStateV2, ChangeEventV2
from codex_baseline_v2.shared.vision_records import EventEmbeddingRecordV1

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None


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


class TinyEventCropEncoder:
    """Optional tiny CNN path for before/after grid pairs when torch is available."""

    def __init__(self, embedding_dim: int = _EMBED_DIM) -> None:
        self.embedding_dim = int(embedding_dim)
        self.available = torch is not None and nn is not None and F is not None
        self.model = None
        if self.available:
            self.model = nn.Sequential(
                nn.Conv2d(2, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 12, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((3, 3)),
                nn.Flatten(),
                nn.Linear(12 * 3 * 3, self.embedding_dim),
            )
            self.model.eval()

    def encode_pair(self, before_grid: Sequence[Sequence[int]], after_grid: Sequence[Sequence[int]]) -> Optional[List[float]]:
        if not self.available or self.model is None or not before_grid or not after_grid:
            return None
        if not before_grid[0] or not after_grid[0]:
            return None
        height = min(len(before_grid), len(after_grid))
        width = min(len(before_grid[0]), len(after_grid[0]))
        before = [[float(before_grid[y][x]) / 9.0 for x in range(width)] for y in range(height)]
        after = [[float(after_grid[y][x]) / 9.0 for x in range(width)] for y in range(height)]
        tensor = torch.tensor([before, after], dtype=torch.float32).reshape(1, 2, height, width)
        with torch.no_grad():
            vector = self.model(tensor).reshape(-1).tolist()
        return _normalize_vector(vector)


def _structured_vector(event: ChangeEventV2) -> List[float]:
    births = 0
    deaths = 0
    moves = 0
    state_changes = 0
    for region in event.region_deltas:
        births += int(region.object_births)
        deaths += int(region.object_deaths)
        moves += int(region.object_moves)
        state_changes += int(region.object_state_changes)
    delta_types = sorted(delta.delta_type for delta in event.object_state_deltas)
    values = [
        min(1.0, event.confidence),
        min(1.0, float(len(event.region_deltas)) / 8.0),
        min(1.0, float(len(event.object_state_deltas)) / 8.0),
        min(1.0, float(abs(event.end_step_idx - event.start_step_idx)) / 12.0),
        min(1.0, float(births) / 8.0),
        min(1.0, float(deaths) / 8.0),
        min(1.0, float(moves) / 8.0),
        min(1.0, float(state_changes) / 8.0),
        1.0 if event.terminal_flag_changed else 0.0,
        max(-1.0, min(1.0, float(event.reward_delta or 0.0))),
        _hash_bucket(event.event_type),
        _hash_bucket(event.locality),
        _hash_bucket(event.trigger_context),
        _hash_bucket(event.pre_area_id),
        _hash_bucket(event.post_area_id),
        _hash_bucket(event.trigger_area_id),
        _hash_bucket(event.trigger_target_poi_id),
        _hash_bucket(event.trigger_zone_id),
        _hash_bucket(event.effect_signature_id),
        _hash_bucket(",".join(delta_types[:3])),
        min(1.0, float(len(event.parent_event_ids)) / 4.0),
        min(1.0, float(len(event.child_event_ids)) / 4.0),
        math.sin(float(event.peak_step_idx)),
        math.cos(float(event.end_step_idx)),
    ]
    return _normalize_vector(values[:_EMBED_DIM])


def _embedding_payload(event: ChangeEventV2, vector: List[float]) -> Dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "kind": "event",
        "vector": vector,
        "meta": {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "area_id": event.post_area_id or event.pre_area_id or event.trigger_area_id,
            "before_crop_ref": f"{event.event_id}:before",
            "after_crop_ref": f"{event.event_id}:after",
            "effect_signature_id": event.effect_signature_id,
            "trigger_zone_id": event.trigger_zone_id,
            "trigger_target_poi_id": event.trigger_target_poi_id,
            "parent_event_ids": list(event.parent_event_ids),
            "child_event_ids": list(event.child_event_ids),
        },
    }


def encode_event_records(blackboard: BlackboardStateV2, vision_root: Path | str) -> List[EventEmbeddingRecordV1]:
    root = Path(vision_root)
    embeddings_root = root / "event_embeddings"
    cnn = TinyEventCropEncoder(_EMBED_DIM)
    records: List[EventEmbeddingRecordV1] = []
    for event in blackboard.event_table:
        vector = _structured_vector(event)
        visual = cnn.encode_pair([], [])
        if visual:
            vector = _normalize_vector([(0.7 * a) + (0.3 * b) for a, b in zip(vector, visual)])
        embedding_id = _stable_id("evemb", blackboard.game_id, event.event_id, event.event_type)
        embedding_path = embeddings_root / f"{embedding_id}.json"
        payload = _embedding_payload(event, vector)
        _write_json(embedding_path, payload)
        area_id = event.post_area_id or event.pre_area_id or event.trigger_area_id
        records.append(
            EventEmbeddingRecordV1(
                schema_version=_SCHEMA_VERSION,
                embedding_id=embedding_id,
                event_id=event.event_id,
                area_id=area_id,
                before_crop_ref=payload["meta"]["before_crop_ref"],
                after_crop_ref=payload["meta"]["after_crop_ref"],
                embedding_path=str(embedding_path),
                event_type=event.event_type,
                confidence=max(0.2, min(1.0, 0.55 + 0.25 * event.confidence)),
            )
        )
    return records
