from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class TransitionEventV1:
    schema_version: str
    game_id: Optional[str]
    seed: Optional[int]
    step_idx: Optional[int]
    action_key: Dict[str, Any]
    state_hash_before: str
    state_hash_after: str
    frame_policy: Dict[str, Any]
    grid_delta: Dict[str, Any]
    event_signatures: List[Dict[str, Any]]
    object_deltas: Dict[str, Any]
    meta_delta: Dict[str, Any]
    state_hash_before_filtered: str = ""
    state_hash_after_filtered: str = ""
    ui_exclusion_mask_rle: Optional[Dict[str, Any]] = None
    ui_exclusion_bboxes: Optional[List[Any]] = None
