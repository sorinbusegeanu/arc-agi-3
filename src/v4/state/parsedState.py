from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from v4.agentContract.environmentMetadata import V4EnvironmentMetadata
from v4.agentContract.extract import extract_v4_authoritative_state
from v4.agentContract.types import V4AuthoritativeState, V4Observation, V4TerminalSignal
from v4.belief.beliefState import BeliefSnapshotReferenceV4
from v4.composition.domainState import CompositionSnapshotReferenceV4
from v4.hypothesis.hypothesisContracts import HypothesisSnapshotReferenceV4
from v4.temporal.resourceState import TemporalSnapshotReferenceV4


@dataclass(frozen=True)
class ChangedRegionSummaryV4:
    changed_cell_count: int = 0
    changed_bbox: tuple[int, int, int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MemorySnapshotReferenceV4:
    revision: int | None = None
    recent_transition_count: int = 0
    visited_state_count: int = 0
    visited_before: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DerivedControlStateV4:
    state_hash: str
    previous_state_hash: str | None = None
    changed_region: ChangedRegionSummaryV4 = field(default_factory=ChangedRegionSummaryV4)
    levels_completed_delta: int | None = None
    win_levels_delta: int | None = None
    available_action_count: int = 0
    retry_counts: dict[str, int] = field(default_factory=dict)
    cooldown_action_keys: tuple[str, ...] = ()
    revealed_cell_count: int = 0
    unknown_cell_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParsedStateV4:
    current_observation: V4Observation
    previous_observation: V4Observation | None
    environment_metadata: V4EnvironmentMetadata | None
    authoritative_state: V4AuthoritativeState
    step_index: int
    available_actions: tuple[int, ...]
    terminal_signal: V4TerminalSignal
    memory_reference: MemorySnapshotReferenceV4 | None
    belief_reference: BeliefSnapshotReferenceV4 | None = field(default=None, kw_only=True)
    hypothesis_reference: HypothesisSnapshotReferenceV4 | None = field(default=None, kw_only=True)
    temporal_reference: TemporalSnapshotReferenceV4 | None = field(default=None, kw_only=True)
    composition_reference: CompositionSnapshotReferenceV4 | None = field(default=None, kw_only=True)
    derived_control: DerivedControlStateV4

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_authoritative_state_for_parsed_state(
    observation: V4Observation,
    environment_metadata: V4EnvironmentMetadata | None,
) -> V4AuthoritativeState:
    return extract_v4_authoritative_state(observation, environment_metadata)
