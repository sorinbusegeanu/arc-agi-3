from __future__ import annotations

"""v8.61 shared state-transition graph for click environments.

v8.60 characterizes a productive click locally, but its evidence is actor-local and
is not converted into a reusable policy.  This layer publishes compact per-actor
transition models under the existing v8.47 trajectory root, periodically refreshes
peer models during a lease, and uses the merged graph to guide executable clicks.

Two representations are retained:
* exact context edges for multi-step graph planning toward observed progress;
* coordinate-local value transitions for reuse across different global contexts.

The layer remains an interaction-control sidecar. Canonical memory formation and
promotion semantics are unchanged.
"""

import hashlib
import json
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path


_INSTALLED = False
_BASE_FORCED_ACTION = None
_BASE_OBSERVE_TRANSITION = None
_BASE_BEGIN_LEASE = None
_BASE_EXTERNAL_RESET = None

_STATE_SCHEMA = 1
_STATE_DIR = "click_transition_graph_v861"
_PUBLISH_EVERY = 8
_REFRESH_EVERY = 32
_MAX_GRAPH_DISTANCE = 64


@dataclass(slots=True)
class TransitionEdge:
    level: int
    before_context: int
    action: int
    after_context: int
    attempts: int = 0
    changed_cells: int = 0
    progress: bool = False
    terminal_failures: int = 0

    def key(self) -> tuple[int, int, int, int]:
        return (self.level, self.before_context, self.action, self.after_context)

    def to_dict(self) -> dict[str, object]:
        return {
            "level": self.level,
            "before_context": self.before_context,
            "action": self.action,
            "after_context": self.after_context,
            "attempts": self.attempts,
            "changed_cells": self.changed_cells,
            "progress": self.progress,
            "terminal_failures": self.terminal_failures,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "TransitionEdge":
        return cls(
            level=int(raw.get("level", 0)),
            before_context=int(raw.get("before_context", 0)),
            action=int(raw.get("action", 0)),
            after_context=int(raw.get("after_context", 0)),
            attempts=max(0, int(raw.get("attempts", 0))),
            changed_cells=max(0, int(raw.get("changed_cells", 0))),
            progress=bool(raw.get("progress", False)),
            terminal_failures=max(0, int(raw.get("terminal_failures", 0))),
        )


@dataclass(slots=True)
class LocalTransition:
    level: int
    x: int
    y: int
    before_value: int
    after_value: int
    attempts: int = 0
    changed_cells: int = 0
    progress: bool = False

    def key(self) -> tuple[int, int, int, int, int]:
        return (self.level, self.x, self.y, self.before_value, self.after_value)

    def to_dict(self) -> dict[str, object]:
        return {
            "level": self.level,
            "x": self.x,
            "y": self.y,
            "before_value": self.before_value,
            "after_value": self.after_value,
            "attempts": self.attempts,
            "changed_cells": self.changed_cells,
            "progress": self.progress,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "LocalTransition":
        return cls(
            level=int(raw.get("level", 0)),
            x=int(raw.get("x", 0)),
            y=int(raw.get("y", 0)),
            before_value=int(raw.get("before_value", 0)),
            after_value=int(raw.get("after_value", 0)),
            attempts=max(0, int(raw.get("attempts", 0))),
            changed_cells=max(0, int(raw.get("changed_cells", 0))),
            progress=bool(raw.get("progress", False)),
        )


def _ensure_state(sampler) -> None:
    if not hasattr(sampler, "_v861_local_edges"):
        sampler._v861_local_edges = {}
    if not hasattr(sampler, "_v861_local_cells"):
        sampler._v861_local_cells = {}
    if not hasattr(sampler, "_v861_shared_edges"):
        sampler._v861_shared_edges = {}
    if not hasattr(sampler, "_v861_shared_cells"):
        sampler._v861_shared_cells = {}
    if not hasattr(sampler, "_v861_dirty_observations"):
        sampler._v861_dirty_observations = 0
    if not hasattr(sampler, "_v861_decisions"):
        sampler._v861_decisions = 0
    if not hasattr(sampler, "_v861_graph_actions"):
        sampler._v861_graph_actions = 0
    if not hasattr(sampler, "_v861_peer_refreshes"):
        sampler._v861_peer_refreshes = 0


def _state_root(sampler) -> Path | None:
    root = getattr(sampler, "_v847_state_root", None)
    if root is None:
        return None
    return Path(root).parent / _STATE_DIR


def _game_token(game_id: str) -> str:
    return hashlib.blake2b(
        str(game_id).encode("utf-8"), digest_size=8, person=b"v861game"
    ).hexdigest()


def _state_path(sampler) -> Path | None:
    root = _state_root(sampler)
    actor_id = int(getattr(sampler, "_v847_actor_id", 0))
    if root is None or actor_id <= 0:
        return None
    return root / f"{_game_token(sampler.game_id)}-{actor_id}.json"


def _merge_edge(target: TransitionEdge, incoming: TransitionEdge) -> None:
    target.attempts += int(incoming.attempts)
    target.changed_cells = max(int(target.changed_cells), int(incoming.changed_cells))
    target.progress = bool(target.progress or incoming.progress)
    target.terminal_failures += int(incoming.terminal_failures)


def _merge_cell(target: LocalTransition, incoming: LocalTransition) -> None:
    target.attempts += int(incoming.attempts)
    target.changed_cells = max(int(target.changed_cells), int(incoming.changed_cells))
    target.progress = bool(target.progress or incoming.progress)


def _save_local(sampler) -> None:
    _ensure_state(sampler)
    path = _state_path(sampler)
    if path is None or int(sampler._v861_dirty_observations) <= 0:
        return
    payload = {
        "schema": _STATE_SCHEMA,
        "game_id": str(sampler.game_id),
        "actor_id": int(getattr(sampler, "_v847_actor_id", 0)),
        "edges": [
            sampler._v861_local_edges[key].to_dict()
            for key in sorted(sampler._v861_local_edges)
        ],
        "cells": [
            sampler._v861_local_cells[key].to_dict()
            for key in sorted(sampler._v861_local_cells)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary, path)
    sampler._v861_dirty_observations = 0


def _refresh_shared(sampler) -> None:
    _ensure_state(sampler)
    root = _state_root(sampler)
    if root is None:
        sampler._v861_shared_edges = dict(sampler._v861_local_edges)
        sampler._v861_shared_cells = dict(sampler._v861_local_cells)
        return
    token = _game_token(sampler.game_id)
    aggregate_edges: dict[tuple[int, int, int, int], TransitionEdge] = {}
    aggregate_cells: dict[tuple[int, int, int, int, int], LocalTransition] = {}
    paths = sorted(root.glob(f"{token}-*.json")) if root.exists() else ()
    for path in paths:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if int(raw.get("schema", 0)) != _STATE_SCHEMA or str(raw.get("game_id", "")) != str(sampler.game_id):
            continue
        for item in raw.get("edges", ()):
            if not isinstance(item, dict):
                continue
            try:
                edge = TransitionEdge.from_dict(item)
            except (TypeError, ValueError):
                continue
            current = aggregate_edges.get(edge.key())
            if current is None:
                aggregate_edges[edge.key()] = edge
            else:
                _merge_edge(current, edge)
        for item in raw.get("cells", ()):
            if not isinstance(item, dict):
                continue
            try:
                cell = LocalTransition.from_dict(item)
            except (TypeError, ValueError):
                continue
            current = aggregate_cells.get(cell.key())
            if current is None:
                aggregate_cells[cell.key()] = cell
            else:
                _merge_cell(current, cell)

    # Include this actor's unsaved tail without waiting for the next publication.
    for key, edge in sampler._v861_local_edges.items():
        if key not in aggregate_edges:
            aggregate_edges[key] = TransitionEdge.from_dict(edge.to_dict())
    for key, cell in sampler._v861_local_cells.items():
        if key not in aggregate_cells:
            aggregate_cells[key] = LocalTransition.from_dict(cell.to_dict())
    sampler._v861_shared_edges = aggregate_edges
    sampler._v861_shared_cells = aggregate_cells
    sampler._v861_peer_refreshes += 1


def _click_payload(action: int) -> tuple[int, int] | None:
    try:
        from v8.learning_blockers_v055 import unpack_action_choice

        native, payload = unpack_action_choice(int(action))
    except (TypeError, ValueError):
        return None
    if int(native) != 6 or not payload:
        return None
    return int(payload["x"]), int(payload["y"])


def _cell_value(grid, x: int, y: int) -> int | None:
    if grid is None:
        return None
    try:
        import numpy as np

        array = np.asarray(grid)
        if array.ndim < 2 or y < 0 or x < 0 or y >= array.shape[0] or x >= array.shape[1]:
            return None
        return int(array[y, x])
    except (TypeError, ValueError, IndexError):
        return None


def _record_transition(sampler, intervention, kwargs) -> None:
    action = int(kwargs.get("action", getattr(intervention, "action", -1)))
    payload = _click_payload(action)
    if payload is None:
        return
    before_level = int(kwargs.get("before_level", 0))
    after_level = int(kwargs.get("after_level", before_level))
    before_context = int(kwargs.get("before_context", 0))
    after_context = int(kwargs.get("after_context", before_context))
    changed = max(0, int(kwargs.get("changed_cells", 0)))
    terminal = str(kwargs.get("terminal_state", ""))
    progress = bool(kwargs.get("level_advanced", False) or after_level > before_level or terminal == "WIN")

    edge = TransitionEdge(
        level=before_level,
        before_context=before_context,
        action=action,
        after_context=after_context,
        attempts=1,
        changed_cells=changed,
        progress=progress,
        terminal_failures=int(terminal == "GAME_OVER"),
    )
    current = sampler._v861_local_edges.get(edge.key())
    if current is None:
        sampler._v861_local_edges[edge.key()] = edge
    else:
        _merge_edge(current, edge)

    x, y = payload
    before_value = _cell_value(getattr(sampler, "_v861_before_observation", None), x, y)
    after_value = _cell_value(getattr(sampler, "_v861_after_observation", None), x, y)
    if before_value is not None and after_value is not None:
        cell = LocalTransition(
            level=before_level,
            x=x,
            y=y,
            before_value=before_value,
            after_value=after_value,
            attempts=1,
            changed_cells=changed,
            progress=progress,
        )
        existing = sampler._v861_local_cells.get(cell.key())
        if existing is None:
            sampler._v861_local_cells[cell.key()] = cell
        else:
            _merge_cell(existing, cell)

    sampler._v861_dirty_observations += 1
    if int(sampler._v861_dirty_observations) >= _PUBLISH_EVERY or progress:
        _save_local(sampler)
        _refresh_shared(sampler)


def _distance_to_progress(edges: dict, level: int) -> dict[int, int]:
    reverse: dict[int, set[int]] = {}
    goals: set[int] = set()
    for edge in edges.values():
        if int(edge.level) != int(level) or edge.terminal_failures >= edge.attempts:
            continue
        reverse.setdefault(int(edge.after_context), set()).add(int(edge.before_context))
        if bool(edge.progress):
            goals.add(int(edge.before_context))
    distances = {context: 0 for context in goals}
    queue = deque((context, 0) for context in sorted(goals))
    while queue:
        context, distance = queue.popleft()
        if distance >= _MAX_GRAPH_DISTANCE:
            continue
        for predecessor in sorted(reverse.get(context, ())):
            if predecessor in distances:
                continue
            distances[predecessor] = distance + 1
            queue.append((predecessor, distance + 1))
    return distances


def _graph_action(sampler, *, level: int, context: int, actions: tuple[int, ...]) -> int | None:
    available = {int(value) for value in actions}
    edges = sampler._v861_shared_edges
    candidates = [
        edge for edge in edges.values()
        if int(edge.level) == int(level)
        and int(edge.before_context) == int(context)
        and int(edge.action) in available
        and int(edge.changed_cells) > 0
        and int(edge.terminal_failures) < int(edge.attempts)
    ]
    if candidates:
        distances = _distance_to_progress(edges, level)
        ranked = sorted(
            candidates,
            key=lambda edge: (
                -int(bool(edge.progress)),
                int(distances.get(int(edge.after_context), _MAX_GRAPH_DISTANCE + 1)),
                int(edge.terminal_failures),
                int(edge.attempts),
                -int(edge.changed_cells),
                int(edge.action),
            ),
        )
        best = ranked[0]
        if best.progress or int(best.after_context) in distances:
            return int(best.action)

    # No known global path to progress yet: reuse coordinate-local dynamics across
    # global contexts. Prefer productive transitions that have been sampled least.
    before_grid = getattr(sampler, "_v861_current_observation", None)
    local_candidates: list[tuple[int, int, int, int]] = []
    for action in sorted(available):
        payload = _click_payload(action)
        if payload is None:
            continue
        x, y = payload
        value = _cell_value(before_grid, x, y)
        if value is None:
            continue
        matches = [
            cell for cell in sampler._v861_shared_cells.values()
            if int(cell.level) == int(level)
            and int(cell.x) == x
            and int(cell.y) == y
            and int(cell.before_value) == value
            and int(cell.after_value) != value
            and int(cell.changed_cells) > 0
        ]
        if not matches:
            continue
        attempts = sum(int(cell.attempts) for cell in matches)
        changed = max(int(cell.changed_cells) for cell in matches)
        progress = int(any(bool(cell.progress) for cell in matches))
        local_candidates.append((-progress, attempts, -changed, action))
    if local_candidates:
        return int(min(local_candidates)[3])
    return None


def _begin_lease_v861(self, seed: int) -> None:
    _BASE_BEGIN_LEASE(self, int(seed))
    _ensure_state(self)
    self._v861_decisions = 0
    _refresh_shared(self)


def _external_reset_v861(self) -> None:
    _save_local(self)
    self._v861_current_observation = None
    self._v861_before_observation = None
    self._v861_after_observation = None
    return _BASE_EXTERNAL_RESET(self)


def _forced_action_v861(self, *, level: int, context: int, actions: tuple[int, ...], history: tuple[int, ...]) -> int | None:
    _ensure_state(self)
    # Preserve stronger replay/verification/persistence and v8.60 immediate
    # characterization authorities.
    if (
        getattr(self, "_v860_pending_action", None) is not None
        or self.base.replay_actions
        or self.base.replay_target is not None
        or self.base.verification is not None
        or self.active_sequence
        or getattr(self, "_v832_persist_action", None) is not None
    ):
        return _BASE_FORCED_ACTION(
            self, level=int(level), context=int(context), actions=tuple(actions), history=tuple(history)
        )

    self._v861_decisions += 1
    if int(self._v861_decisions) % _REFRESH_EVERY == 0:
        _save_local(self)
        _refresh_shared(self)
    action = _graph_action(self, level=int(level), context=int(context), actions=tuple(actions))
    if action is not None:
        from v8 import decision_point_sampling_v821 as sampling
        from v8 import sampling_portfolio_v831 as portfolio

        self.base.current = sampling.Intervention(
            "TRANSITION_GRAPH", (int(level), int(context)), int(action), tuple(history)
        )
        self._v861_graph_actions += 1
        portfolio._set_mode("PROGRESS")
        portfolio._set_source(context, "CLICK_TRANSITION_GRAPH", (int(action),))
        return int(action)
    return _BASE_FORCED_ACTION(
        self, level=int(level), context=int(context), actions=tuple(actions), history=tuple(history)
    )


def _observe_transition_v861(self, **kwargs):
    _ensure_state(self)
    intervention = self.base.current
    if intervention is not None:
        _record_transition(self, intervention, kwargs)
    result = _BASE_OBSERVE_TRANSITION(self, **kwargs)
    if bool(kwargs.get("level_advanced", False)) or int(kwargs.get("after_level", 0)) != int(kwargs.get("before_level", 0)):
        # Local characterization counters are episode/level-scoped. Shared graph
        # evidence remains persistent.
        if hasattr(self, "_v860_repeat_depth"):
            self._v860_repeat_depth.clear()
        if hasattr(self, "_v860_seen_transitions"):
            self._v860_seen_transitions.clear()
        if hasattr(self, "_v860_transition_counts"):
            self._v860_transition_counts.clear()
        if hasattr(self, "_v860_pending_action"):
            self._v860_pending_action = None
    return result


def transition_graph_telemetry_v861(sampler) -> dict[str, int]:
    _ensure_state(sampler)
    return {
        "local_edges": len(sampler._v861_local_edges),
        "shared_edges": len(sampler._v861_shared_edges),
        "local_cell_transitions": len(sampler._v861_local_cells),
        "shared_cell_transitions": len(sampler._v861_shared_cells),
        "graph_actions": int(sampler._v861_graph_actions),
        "peer_refreshes": int(sampler._v861_peer_refreshes),
    }


def install_click_transition_graph_v861() -> None:
    global _INSTALLED
    global _BASE_FORCED_ACTION, _BASE_OBSERVE_TRANSITION, _BASE_BEGIN_LEASE, _BASE_EXTERNAL_RESET
    if _INSTALLED:
        return
    from v8 import sampling_evidence_frontier_v847_fixups as frontier_fixups
    from v8 import sampling_persistence_v832 as persistence

    # Compose beneath all historically pinned public/intermediate authorities.
    _BASE_FORCED_ACTION = frontier_fixups._BASE_LOWER_FORCED
    frontier_fixups._BASE_LOWER_FORCED = _forced_action_v861
    _BASE_OBSERVE_TRANSITION = frontier_fixups._BASE_LOWER_OBSERVE
    frontier_fixups._BASE_LOWER_OBSERVE = _observe_transition_v861

    # Reset scoping composes at the v8.32 saved-base seam; v8.32 remains public.
    _BASE_BEGIN_LEASE = persistence._BASE_BEGIN_LEASE
    persistence._BASE_BEGIN_LEASE = _begin_lease_v861
    _BASE_EXTERNAL_RESET = persistence._BASE_ON_EXTERNAL_RESET
    persistence._BASE_ON_EXTERNAL_RESET = _external_reset_v861
    _INSTALLED = True
