from __future__ import annotations

from v9.research.experiment_manifest import StructuralPriorProfile
from v9.research.structural_prior import StructuralPriorConfiguration, StructuralPriorTransform


def test_s0_exposes_only_opaque_identity_and_temporal_order() -> None:
    transform = StructuralPriorTransform(StructuralPriorConfiguration(StructuralPriorProfile.S0_TEMPORAL_ONLY), coordinates=((0, 0),))
    output = transform.transform_observation({"objects": ({"id": "x", "value": "red", "coordinate": (0, 0)},)})
    assert set(output["objects"][0]) == {"opaque_id", "temporal_order"}


def test_s1_adds_equality_without_coordinates() -> None:
    transform = StructuralPriorTransform(StructuralPriorConfiguration(StructuralPriorProfile.S1_EQUALITY), coordinates=((0,), (1,)))
    output = transform.transform_observation({"objects": ({"id": 1, "value": 7, "coordinate": (0,)}, {"id": 2, "value": 7, "coordinate": (1,)})})
    assert output["objects"][0]["equality_token"] == output["objects"][1]["equality_token"]
    assert "coordinate" not in output["objects"][0]
