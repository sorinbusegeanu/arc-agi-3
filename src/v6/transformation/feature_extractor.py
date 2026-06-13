from __future__ import annotations

import numpy as np

from v6.delta.delta_extractor import Delta


def delta_to_feature_vector(delta: Delta) -> np.ndarray:
    return np.asarray(
        [
            float(delta.changed_cells),
            float(delta.dx),
            float(delta.dy),
            float(len(delta.colors_added)),
            float(len(delta.colors_removed)),
        ],
        dtype=float,
    )
