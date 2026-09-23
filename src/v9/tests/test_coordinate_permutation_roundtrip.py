from __future__ import annotations

from v9.research.experiment_manifest import StructuralPriorProfile
from v9.research.structural_prior import CoordinateMode, StructuralPriorConfiguration, StructuralPriorTransform


def test_fixed_coordinate_permutation_is_bijective() -> None:
    coordinates = tuple((index,) for index in range(10))
    transform = StructuralPriorTransform(StructuralPriorConfiguration(StructuralPriorProfile.S2_COORDINATES, coordinate_mode=CoordinateMode.FIXED_COORDINATE_PERMUTATION, permutation_seed=7), coordinates=coordinates)
    exposed = tuple(transform.expose_coordinate(row) for row in coordinates)
    assert set(exposed) == set(coordinates)
    assert tuple(transform.restore_coordinate(row) for row in exposed) == coordinates


def test_changing_permutation_uses_declared_generation_schedule() -> None:
    coordinates = tuple((index,) for index in range(20))
    transform = StructuralPriorTransform(StructuralPriorConfiguration(StructuralPriorProfile.S2_COORDINATES, coordinate_mode=CoordinateMode.CHANGING_COORDINATE_PERMUTATION, permutation_seed=2), coordinates=coordinates)
    assert tuple(transform.expose_coordinate(row, generation=0) for row in coordinates) != tuple(transform.expose_coordinate(row, generation=1) for row in coordinates)
