from __future__ import annotations

from v9.research.experiment_manifest import StructuralPriorProfile
from v9.research.structural_prior import AdjacencyMode, StructuralPriorConfiguration, StructuralPriorTransform


def test_adjacency_withheld_removes_edges() -> None:
    transform = StructuralPriorTransform(StructuralPriorConfiguration(StructuralPriorProfile.S3_TOPOLOGY, adjacency_mode=AdjacencyMode.WITHHELD), coordinates=((0,), (1,)))
    output = transform.transform_observation({"objects": ({"id": 0, "coordinate": (0,)}, {"id": 1, "coordinate": (1,)}), "adjacency": ((0, 1),)})
    assert "adjacency" not in output
