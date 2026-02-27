from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class FPAnalystConfig:
    connectivity: int = 4
    min_area: int = 1
    max_objects: int = 512
    enable_tracking: bool = True
    enable_symmetry: bool = True
    enable_periodicity: bool = True
    max_period: int = 16
    bg_detection_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "frequency": 0.5,
            "border": 0.3,
            "connectedness": 0.2,
        }
    )
    iou_threshold: float = 0.25
    centroid_distance_threshold: float = 10.0
    iou_soft_threshold: float = 0.10
    overlays: List[str] = field(
        default_factory=lambda: [
            "bbox_overlay",
            "component_id_overlay",
            "diff_mask",
            "object_motion_overlay",
        ]
    )
    save_images: bool = False
    output_dir: str = "runs"
