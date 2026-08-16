from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from typing import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class ObservationStructureContract:
    """Explicit non-semantic observation prior used by one experimental condition."""

    contract_id: str
    schema_version: int
    observation_domain: str
    primitive_relations: tuple[str, ...]
    transformation_operator_id: str
    transformation_operator_version: int
    transformation_distance_id: str
    allowed_derived_statistics: tuple[str, ...]
    forbidden_semantic_fields: tuple[str, ...]

    @property
    def digest(self) -> str:
        h = blake2b(digest_size=16, person=b"v8.2-obs-contract")
        for value in (
            self.contract_id,
            self.schema_version,
            self.observation_domain,
            self.primitive_relations,
            self.transformation_operator_id,
            self.transformation_operator_version,
            self.transformation_distance_id,
            self.allowed_derived_statistics,
            self.forbidden_semantic_fields,
        ):
            h.update(repr(value).encode("utf-8"))
            h.update(b"\0")
        return h.hexdigest()


ARC_GRID_CONTRACT = ObservationStructureContract(
    contract_id="arc-grid-v1",
    schema_version=1,
    observation_domain="finite-2d-grid",
    primitive_relations=(
        "equality",
        "row_identity",
        "column_identity",
        "coordinate_order",
        "four_neighbor_adjacency",
        "temporal_order",
    ),
    transformation_operator_id="arc-grid-cell-delta",
    transformation_operator_version=1,
    transformation_distance_id="changed-cell-jaccard-v1",
    allowed_derived_statistics=(
        "changed_coordinates",
        "before_after_cell_values",
        "changed_cell_count",
        "bounding_extent",
        "connected_changed_regions",
        "displacement_like_relations",
    ),
    forbidden_semantic_fields=(
        "object",
        "agent",
        "key",
        "door",
        "enemy",
        "goal",
        "reward",
        "win_value",
        "terminal_value",
    ),
)


def get_observation_contract(contract_id: str) -> ObservationStructureContract:
    if str(contract_id) != ARC_GRID_CONTRACT.contract_id:
        raise KeyError(f"unknown observation structure contract: {contract_id}")
    return ARC_GRID_CONTRACT


def grid_transformation(
    before: Sequence[Sequence[int]],
    after: Sequence[Sequence[int]],
) -> tuple[tuple[int, int, int, int], ...]:
    """Reference D_O for ARC grids: changed coordinates and before/after values only."""
    if len(before) != len(after):
        raise ValueError("grid row count changed")
    result: list[tuple[int, int, int, int]] = []
    for row_index, (row_before, row_after) in enumerate(zip(before, after, strict=True)):
        if len(row_before) != len(row_after):
            raise ValueError("grid column count changed")
        for column_index, (left, right) in enumerate(zip(row_before, row_after, strict=True)):
            if int(left) != int(right):
                result.append((row_index, column_index, int(left), int(right)))
    return tuple(result)


def changed_cell_distance(
    left: Iterable[tuple[int, int, int, int]],
    right: Iterable[tuple[int, int, int, int]],
) -> float:
    """Bounded structural distance for transformation descriptors."""
    a, b = set(left), set(right)
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / max(1, len(a | b))
