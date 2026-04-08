from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from v4.agentContract.environmentMetadata import V4EnvironmentMetadata
from v4.agentContract.types import V4Observation

from .observedFacts import InferredLocalFactV4, ObservedCellFactV4
from .unknownFacts import UnknownCellFactV4

if TYPE_CHECKING:
    from v4.state.parsedState import ParsedStateV4


def _frame_plane(observation: V4Observation) -> tuple[tuple[object, ...], ...]:
    if not observation.frame:
        return ()
    return observation.frame[0]


def _state_key(observation: V4Observation) -> str:
    payload = {
        "game_id": observation.game_id,
        "state": observation.state,
        "levels_completed": observation.levels_completed,
        "win_levels": observation.win_levels,
        "available_actions": list(observation.available_actions),
        "frame": observation.frame,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _unknown_markers(environment_metadata: V4EnvironmentMetadata | None) -> set[object]:
    markers: set[object] = {None, "?"}
    if environment_metadata is None:
        return markers
    additional_properties = getattr(environment_metadata, "additional_properties", None)
    if additional_properties is None:
        additional_properties = environment_metadata.raw_payload.get("additional_properties", {})
    if isinstance(additional_properties, dict):
        values = additional_properties.get("belief_unknown_values", ())
        if isinstance(values, (list, tuple, set)):
            markers.update(values)
    return markers


def extract_observed_and_unknown_cells(
    observation: V4Observation,
    environment_metadata: V4EnvironmentMetadata | None,
    step_index: int,
    parsed_state: ParsedStateV4 | None = None,
) -> tuple[tuple[ObservedCellFactV4, ...], tuple[UnknownCellFactV4, ...]]:
    state_key = _state_key(observation)
    if parsed_state is not None and parsed_state.current_observation.game_id == "ms01":
        try:
            from v4.memory_hidden.stateBuilder import MemoryHiddenStateBuilderV4

            typed_state = MemoryHiddenStateBuilderV4().build(parsed_state)
        except Exception:
            typed_state = None
        if typed_state is not None:
            observed_cells = tuple(
                ObservedCellFactV4(
                    x=x,
                    y=y,
                    value="revealed_safe",
                    certainty=1.0,
                    evidence_step_index=step_index,
                    evidence_state_key=state_key,
                )
                for x, y in sorted(typed_state.family.revealed_safe_cells, key=lambda pos: (pos[1], pos[0]))
            )
            unknown_cells = tuple(
                UnknownCellFactV4(
                    x=x,
                    y=y,
                    reason="memory_hidden_frontier",
                    frontier=True,
                    certainty=1.0,
                    evidence_step_index=step_index,
                    evidence_state_key=state_key,
                )
                for x, y in sorted(typed_state.family.unrevealed_frontier_cells, key=lambda pos: (pos[1], pos[0]))
            )
            return observed_cells, unknown_cells
    unknown_markers = _unknown_markers(environment_metadata)
    observed_cells: list[ObservedCellFactV4] = []
    unknown_cells: list[UnknownCellFactV4] = []
    plane = _frame_plane(observation)
    observed_positions: set[tuple[int, int]] = set()
    for y, row in enumerate(plane):
        for x, value in enumerate(row):
            if value in unknown_markers:
                unknown_cells.append(
                    UnknownCellFactV4(
                        x=x,
                        y=y,
                        reason="unknown_marker",
                        frontier=False,
                        certainty=1.0,
                        evidence_step_index=step_index,
                        evidence_state_key=state_key,
                    )
                )
                continue
            observed_cells.append(
                ObservedCellFactV4(
                    x=x,
                    y=y,
                    value=value,
                    certainty=1.0,
                    evidence_step_index=step_index,
                    evidence_state_key=state_key,
                )
            )
            observed_positions.add((x, y))
    frontier_unknown_cells: list[UnknownCellFactV4] = []
    for cell in unknown_cells:
        frontier = any(
            neighbor in observed_positions
            for neighbor in (
                (cell.x - 1, cell.y),
                (cell.x + 1, cell.y),
                (cell.x, cell.y - 1),
                (cell.x, cell.y + 1),
            )
        )
        frontier_unknown_cells.append(replace(cell, frontier=frontier))
    return tuple(observed_cells), tuple(frontier_unknown_cells)


def infer_local_facts(
    observed_cells: tuple[ObservedCellFactV4, ...],
    unknown_cells: tuple[UnknownCellFactV4, ...],
    step_index: int,
    state_key: str,
) -> tuple[InferredLocalFactV4, ...]:
    inferred: list[InferredLocalFactV4] = []
    frontier_count = sum(1 for cell in unknown_cells if cell.frontier)
    if frontier_count > 0:
        inferred.append(
            InferredLocalFactV4(
                fact_id="belief:frontier_exists",
                kind="frontier_exists",
                payload={"frontier_unknown_count": frontier_count},
                certainty=1.0,
                evidence_refs=(f"step:{step_index}", f"state:{state_key}"),
            )
        )
    if not unknown_cells and observed_cells:
        inferred.append(
            InferredLocalFactV4(
                fact_id="belief:fully_revealed",
                kind="fully_revealed",
                payload={"observed_cell_count": len(observed_cells)},
                certainty=1.0,
                evidence_refs=(f"step:{step_index}", f"state:{state_key}"),
            )
        )
    return tuple(inferred)
