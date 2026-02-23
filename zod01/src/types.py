from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class NormalizedAction:
    """Serializable action format independent from arcengine internals."""

    name: str
    x: int | None = None
    y: int | None = None
    reasoning: str | dict[str, Any] | None = None

    def key(self) -> str:
        if self.x is None or self.y is None:
            return self.name
        return f"{self.name}:{self.x},{self.y}"


@dataclass(frozen=True)
class ParsedObservation:
    game_id: str
    state: str
    levels_completed: int
    win_levels: int
    guid: str
    full_reset: bool
    available_actions: tuple[str, ...]
    # Immutable tuple format keeps serialization deterministic
    frame_layers: tuple[tuple[tuple[int, ...], ...], ...]


@dataclass(frozen=True)
class CanonicalState:
    state_hash: str
    payload: bytes
    grid_shape: tuple[int, int]
    state: str
    levels_completed: int
    win_levels: int
    available_actions: tuple[str, ...]


@dataclass(frozen=True)
class TransitionDelta:
    changed_cells: int
    total_cells: int
    change_ratio: float
    no_op: bool
    reversible_guess: bool
    tags: tuple[str, ...] = ()


@dataclass
class TransitionStats:
    visits: int = 0
    success_visits: int = 0
    no_op_visits: int = 0


@dataclass
class TransitionEdge:
    src_hash: str
    action: NormalizedAction
    dst_hash: str
    delta: TransitionDelta
    stats: TransitionStats = field(default_factory=TransitionStats)


@dataclass
class ActionProposal:
    action: NormalizedAction
    source: str
    score: float
    tags: tuple[str, ...] = ()


@dataclass
class ControllerContext:
    step_idx: int
    recent_hashes: tuple[str, ...]
    available_actions: tuple[str, ...]


@dataclass
class EpisodeResult:
    episode_id: str
    game_id: str
    won: bool
    actions: int
    unique_states: int
    trajectory_hash: str
    log_path: str = ""
