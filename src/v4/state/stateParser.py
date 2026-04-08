from __future__ import annotations

import hashlib
import json
from typing import Any

from v4.agentContract.environmentMetadata import V4EnvironmentMetadata
from v4.agentContract.errors import V4ValidationError
from v4.agentContract.extract import extract_v4_authoritative_state
from v4.agentContract.types import V4Observation
from v4.agentContract.validators import derive_terminal_signal, validate_v4_observation
from v4.belief.beliefState import BeliefStateV4
from v4.composition.domainState import ComposedDomainStateV4
from v4.hypothesis.hypothesisRegistry import HypothesisStateV4
from v4.memory.localMemory import LocalMemoryStateV4
from v4.temporal.resourceState import TemporalResourceStateV4

from .parsedState import (
    ChangedRegionSummaryV4,
    DerivedControlStateV4,
    MemorySnapshotReferenceV4,
    ParsedStateV4,
)


def _frame_plane(observation: V4Observation) -> tuple[tuple[Any, ...], ...]:
    if not observation.frame:
        return ()
    return observation.frame[0]


def _state_hash(observation: V4Observation) -> str:
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


def _changed_region(current: V4Observation, previous: V4Observation | None) -> ChangedRegionSummaryV4:
    if previous is None:
        return ChangedRegionSummaryV4(changed_cell_count=0, changed_bbox=None)
    current_plane = _frame_plane(current)
    previous_plane = _frame_plane(previous)
    if not current_plane or not previous_plane:
        return ChangedRegionSummaryV4(changed_cell_count=0, changed_bbox=None)
    height = min(len(current_plane), len(previous_plane))
    width = min(len(current_plane[0]), len(previous_plane[0])) if height else 0
    changed: list[tuple[int, int]] = []
    for y in range(height):
        for x in range(width):
            if current_plane[y][x] != previous_plane[y][x]:
                changed.append((x, y))
    if not changed:
        return ChangedRegionSummaryV4(changed_cell_count=0, changed_bbox=None)
    xs = [cell[0] for cell in changed]
    ys = [cell[1] for cell in changed]
    return ChangedRegionSummaryV4(
        changed_cell_count=len(changed),
        changed_bbox=(min(xs), min(ys), max(xs), max(ys)),
    )


class StateParserV4:
    def build_parsed_state(
        self,
        *,
        current_observation: V4Observation,
        previous_observation: V4Observation | None,
        environment_metadata: V4EnvironmentMetadata | None,
        local_memory_snapshot: LocalMemoryStateV4 | None,
        belief_snapshot: BeliefStateV4 | None = None,
        hypothesis_snapshot: HypothesisStateV4 | None = None,
        temporal_snapshot: TemporalResourceStateV4 | None = None,
        composition_snapshot: ComposedDomainStateV4 | None = None,
        step_index: int,
    ) -> ParsedStateV4:
        try:
            validate_v4_observation(current_observation)
            if previous_observation is not None:
                validate_v4_observation(previous_observation)
        except ValueError as exc:
            raise V4ValidationError(str(exc), source_field="observation") from exc
        authoritative_state = extract_v4_authoritative_state(current_observation, environment_metadata)
        terminal_signal = derive_terminal_signal(current_observation)
        current_hash = _state_hash(current_observation)
        previous_hash = _state_hash(previous_observation) if previous_observation is not None else None
        visited_before = False
        retry_counts: dict[str, int] = {}
        cooldown_keys: tuple[str, ...] = ()
        revealed_count = 0
        unknown_count = 0
        memory_reference = None
        belief_reference = None
        hypothesis_reference = None
        temporal_reference = None
        composition_reference = None
        if belief_snapshot is not None:
            if not isinstance(belief_snapshot, BeliefStateV4):
                raise V4ValidationError("belief_snapshot must be BeliefStateV4", source_field="belief_snapshot")
            revealed_count = len(belief_snapshot.observed_cells)
            unknown_count = len(belief_snapshot.unknown_cells)
            belief_reference = belief_snapshot.snapshot_reference()
        if hypothesis_snapshot is not None:
            if not isinstance(hypothesis_snapshot, HypothesisStateV4):
                raise V4ValidationError("hypothesis_snapshot must be HypothesisStateV4", source_field="hypothesis_snapshot")
            hypothesis_reference = hypothesis_snapshot.snapshot_reference()
        if temporal_snapshot is not None:
            if not isinstance(temporal_snapshot, TemporalResourceStateV4):
                raise V4ValidationError("temporal_snapshot must be TemporalResourceStateV4", source_field="temporal_snapshot")
            temporal_reference = temporal_snapshot.snapshot_reference()
        if composition_snapshot is not None:
            if not isinstance(composition_snapshot, ComposedDomainStateV4):
                raise V4ValidationError("composition_snapshot must be ComposedDomainStateV4", source_field="composition_snapshot")
            composition_reference = composition_snapshot.snapshot_reference()
        if local_memory_snapshot is not None:
            if not isinstance(local_memory_snapshot, LocalMemoryStateV4):
                raise V4ValidationError("local_memory_snapshot must be LocalMemoryStateV4", source_field="local_memory_snapshot")
            visited_before = current_hash in set(local_memory_snapshot.visited_state_hashes)
            retry_counts = dict(local_memory_snapshot.retry_counts)
            cooldown_keys = tuple(sorted(local_memory_snapshot.cooldown_markers))
            if belief_snapshot is None:
                revealed_count = len(local_memory_snapshot.revealed_cells)
                unknown_count = len(local_memory_snapshot.unknown_cells)
            memory_reference = MemorySnapshotReferenceV4(
                revision=local_memory_snapshot.revision,
                recent_transition_count=len(local_memory_snapshot.recent_transition_refs),
                visited_state_count=len(local_memory_snapshot.visited_state_hashes),
                visited_before=visited_before,
            )
        derived_control = DerivedControlStateV4(
            state_hash=current_hash,
            previous_state_hash=previous_hash,
            changed_region=_changed_region(current_observation, previous_observation),
            levels_completed_delta=(
                None
                if previous_observation is None
                else current_observation.levels_completed - previous_observation.levels_completed
            ),
            win_levels_delta=(
                None
                if previous_observation is None
                else current_observation.win_levels - previous_observation.win_levels
            ),
            available_action_count=len(current_observation.available_actions),
            retry_counts=retry_counts,
            cooldown_action_keys=cooldown_keys,
            revealed_cell_count=revealed_count,
            unknown_cell_count=unknown_count,
        )
        return ParsedStateV4(
            current_observation=current_observation,
            previous_observation=previous_observation,
            environment_metadata=environment_metadata,
            authoritative_state=authoritative_state,
            step_index=int(step_index),
            available_actions=current_observation.available_actions,
            terminal_signal=terminal_signal,
            memory_reference=memory_reference,
            belief_reference=belief_reference,
            hypothesis_reference=hypothesis_reference,
            temporal_reference=temporal_reference,
            composition_reference=composition_reference,
            derived_control=derived_control,
        )
