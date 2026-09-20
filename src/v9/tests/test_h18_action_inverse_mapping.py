from __future__ import annotations

from v9.research.experiment_manifest import StructuralPriorProfile
from v9.research.structural_prior import CoordinateAction, CoordinateMode, StructuralPriorConfiguration, StructuralPriorTransform


def test_coordinate_action_uses_inverse_observation_bijection() -> None:
    transform = StructuralPriorTransform(StructuralPriorConfiguration(StructuralPriorProfile.S3_TOPOLOGY, coordinate_mode=CoordinateMode.FIXED_COORDINATE_PERMUTATION, permutation_seed=5), coordinates=((0, 0), (0, 1), (1, 0), (1, 1)))
    original = CoordinateAction("select", (1, 0))
    exposed = transform.expose_action(original)
    assert isinstance(exposed, CoordinateAction)
    assert transform.restore_action(exposed) == original
