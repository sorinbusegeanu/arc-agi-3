from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping

from .experiment_manifest import StructuralPriorProfile


class CoordinateMode(str, Enum):
    ORDINARY = "ORDINARY"
    FIXED_COORDINATE_PERMUTATION = "FIXED_COORDINATE_PERMUTATION"
    CHANGING_COORDINATE_PERMUTATION = "CHANGING_COORDINATE_PERMUTATION"
    WITHHELD = "WITHHELD"


class TopologyMode(str, Enum):
    AVAILABLE = "AVAILABLE"
    WITHHELD = "WITHHELD"


class AdjacencyMode(str, Enum):
    AVAILABLE = "AVAILABLE"
    WITHHELD = "WITHHELD"


@dataclass(frozen=True, slots=True)
class CoordinateAction:
    action_type: str
    target: tuple[int, ...]
    payload: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class StructuralPriorConfiguration:
    profile: StructuralPriorProfile
    coordinate_mode: CoordinateMode = CoordinateMode.ORDINARY
    topology_mode: TopologyMode = TopologyMode.AVAILABLE
    adjacency_mode: AdjacencyMode = AdjacencyMode.AVAILABLE
    permutation_seed: int = 0
    changing_period: int = 1

    def __post_init__(self) -> None:
        if self.changing_period <= 0:
            raise ValueError("coordinate permutation period must be positive")


def _stable_token(domain: str, value: object) -> str:
    return hashlib.sha256(f"{domain}\0{value!r}".encode()).hexdigest()[:24]


class StructuralPriorTransform:
    """Adapter-boundary transform that exposes only the declared structural prior."""

    _FORBIDDEN_KEYS = {
        "raw_row",
        "raw_column",
        "row",
        "column",
        "tensor_shape",
        "original_shape",
        "connected_components",
        "adjacency_helper",
        "raw_coordinates",
        "untransformed_action_id",
    }

    def __init__(self, configuration: StructuralPriorConfiguration, *, coordinates: Iterable[tuple[int, ...]]) -> None:
        self.configuration = configuration
        self._coordinates = tuple(sorted(set(tuple(int(value) for value in row) for row in coordinates)))
        self._forward: dict[tuple[int, ...], tuple[int, ...]] = {}
        self._inverse: dict[tuple[int, ...], tuple[int, ...]] = {}
        self._generation: int | None = None

    def _ensure_permutation(self, generation: int) -> None:
        mode = self.configuration.coordinate_mode
        effective_generation = 0
        if mode is CoordinateMode.CHANGING_COORDINATE_PERMUTATION:
            effective_generation = int(generation) // self.configuration.changing_period
        if self._generation == effective_generation:
            return
        exposed = list(self._coordinates)
        if mode in {CoordinateMode.FIXED_COORDINATE_PERMUTATION, CoordinateMode.CHANGING_COORDINATE_PERMUTATION}:
            random.Random(self.configuration.permutation_seed + effective_generation * 1_000_003).shuffle(exposed)
        self._forward = dict(zip(self._coordinates, exposed))
        self._inverse = {value: key for key, value in self._forward.items()}
        self._generation = effective_generation

    def expose_coordinate(self, coordinate: tuple[int, ...], *, generation: int = 0) -> tuple[int, ...] | str:
        coordinate = tuple(int(value) for value in coordinate)
        if self.configuration.coordinate_mode is CoordinateMode.WITHHELD or self.configuration.profile in {
            StructuralPriorProfile.S0_TEMPORAL_ONLY,
            StructuralPriorProfile.S1_EQUALITY,
        }:
            return _stable_token("opaque-target", coordinate)
        self._ensure_permutation(generation)
        if coordinate not in self._forward:
            raise KeyError("coordinate is outside the declared adapter domain")
        return self._forward[coordinate]

    def restore_coordinate(self, exposed: tuple[int, ...], *, generation: int = 0) -> tuple[int, ...]:
        if self.configuration.coordinate_mode is CoordinateMode.WITHHELD:
            raise ValueError("withheld coordinates cannot be supplied as learner actions")
        self._ensure_permutation(generation)
        try:
            return self._inverse[tuple(int(value) for value in exposed)]
        except KeyError as exc:
            raise ValueError("learner action target is outside the exposed coordinate domain") from exc

    def expose_action(self, action: CoordinateAction, *, generation: int = 0) -> CoordinateAction | Mapping[str, object]:
        target = self.expose_coordinate(action.target, generation=generation)
        if isinstance(target, str):
            return {"action_type": action.action_type, "opaque_target": target, "payload": action.payload}
        return CoordinateAction(action.action_type, target, action.payload)

    def restore_action(self, action: CoordinateAction, *, generation: int = 0) -> CoordinateAction:
        return CoordinateAction(action.action_type, self.restore_coordinate(action.target, generation=generation), action.payload)

    def transform_observation(self, observation: Mapping[str, Any], *, generation: int = 0) -> dict[str, object]:
        leaked = self._FORBIDDEN_KEYS.intersection(observation)
        if leaked:
            raise ValueError(f"adapter observation contains prohibited structural side channels: {sorted(leaked)}")
        objects = list(observation.get("objects", ()))
        if not objects and "grid" in observation:
            grid = observation["grid"]
            objects = [
                {"id": f"cell-{index}", "value": value, "coordinate": coordinate}
                for index, (coordinate, value) in enumerate(_flatten_grid(grid))
            ]
        transformed = []
        for ordinal, raw in enumerate(objects):
            row = dict(raw)
            if self._FORBIDDEN_KEYS.intersection(row):
                raise ValueError("object record contains prohibited structural side channels")
            coordinate = tuple(row.pop("coordinate", ()))
            value = row.pop("value", None)
            output: dict[str, object] = {
                "opaque_id": _stable_token("object", row.pop("id", ordinal)),
                "temporal_order": ordinal,
            }
            if self.configuration.profile is not StructuralPriorProfile.S0_TEMPORAL_ONLY:
                output["equality_token"] = _stable_token("equality", value)
            if coordinate and self.configuration.profile in {
                StructuralPriorProfile.S2_COORDINATES,
                StructuralPriorProfile.S3_TOPOLOGY,
            }:
                output["coordinate"] = self.expose_coordinate(coordinate, generation=generation)
            transformed.append(output)
        result: dict[str, object] = {"objects": tuple(transformed)}
        allow_topology = self.configuration.profile is StructuralPriorProfile.S3_TOPOLOGY and self.configuration.topology_mode is TopologyMode.AVAILABLE
        allow_adjacency = allow_topology and self.configuration.adjacency_mode is AdjacencyMode.AVAILABLE
        if allow_adjacency:
            result["adjacency"] = tuple(tuple(int(value) for value in edge) for edge in observation.get("adjacency", ()))
        return result


def _flatten_grid(grid: object) -> tuple[tuple[tuple[int, ...], object], ...]:
    rows = []

    def visit(value: object, coordinate: tuple[int, ...]) -> None:
        if isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(item, coordinate + (index,))
        else:
            rows.append((coordinate, value))

    visit(grid, ())
    return tuple(rows)


__all__ = [
    "AdjacencyMode",
    "CoordinateAction",
    "CoordinateMode",
    "StructuralPriorConfiguration",
    "StructuralPriorTransform",
    "TopologyMode",
]
