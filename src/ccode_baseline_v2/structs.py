"""structs.py — shared dataclasses. No logic."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class POIRecord:
    poi_id: str                          # uuid4
    bbox: Tuple[int, int, int, int]      # (y0, x0, y1, x1)
    color_signature: List[int]           # dominant fg colors
    tag: str                             # SELF | ENEMY | TARGET | HUD | UNKNOWN
    reachable: bool
    visited: bool
    consequence: Optional[str]           # BIG_CHANGE | SMALL_CHANGE | NO_CHANGE | None
    confidence: float                    # 0..1
    version: int                         # analysis cycle index
    identity_key: str = ""               # stable hash across versions (bbox-quantised + color)
    motion_detected: bool = False        # centroid moved > 0.5 cells across frames
    depriority: bool = False             # True when stale for STALE_VERSIONS cycles


@dataclass
class EpisodeRecord:
    episode_id: int
    frames: List[np.ndarray]             # raw pixel grids, shape (H, W) each
    actions: List[str]                   # action keys taken
    positions: List[Optional[Tuple[int, int]]]   # sprite (x, y) centroid or None
    terminal: bool = False               # episode ended via done signal (not step limit)
    exit_state: str = ""                 # "won" | "lost" | "" from meta.state


@dataclass
class HypothesisStoreState:
    pois: Dict[str, POIRecord]           # keyed by poi_id
    version: int


@dataclass
class ConsequenceResult:
    label: str                           # BIG_CHANGE | SMALL_CHANGE | NO_CHANGE | GAME_WON | LEVEL_CHANGE
    pixel_diff_ratio: float
    histogram_shift: float
