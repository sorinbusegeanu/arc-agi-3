from __future__ import annotations

import pytest

from v9.research.experiment_manifest import StructuralPriorProfile
from v9.research.structural_prior import StructuralPriorConfiguration, StructuralPriorTransform


def test_dense_grid_does_not_expose_shape_or_indices_when_topology_ablated() -> None:
    transform = StructuralPriorTransform(StructuralPriorConfiguration(StructuralPriorProfile.S0_TEMPORAL_ONLY), coordinates=((0, 0), (0, 1), (1, 0), (1, 1)))
    output = transform.transform_observation({"grid": ((1, 2), (3, 4))})
    assert set(output) == {"objects"}
    assert all(set(row) == {"opaque_id", "temporal_order"} for row in output["objects"])
    with pytest.raises(ValueError, match="side channels"):
        transform.transform_observation({"tensor_shape": (2, 2)})
