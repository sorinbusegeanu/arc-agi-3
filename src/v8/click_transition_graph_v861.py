from __future__ import annotations

"""v8.61 shared state-transition graph for click environments.

Publishes compact per-actor click transition models under the existing v8.47
trajectory root, refreshes peer evidence during a lease, and guides executable
clicks toward observed progress. Canonical memory semantics are unchanged.
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
            int(raw.get("level", 0)), int(raw.get("before_context", 0)),
            int(raw.get("action", 0)), int(raw.get("after_context", 0)),
            max(0, int(raw.get("attempts", 0))),
            max(0, int(raw.get("changed_cells", 0))),
            bool(raw.get("progress", False)),
            max(0, int(raw.get("terminal_failures", 0))),
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
            "level": self.level, "x": self.x, "y": self.y,
            "before_value": self.before_value, "after_value": self.after_value,
            "attempts": self.attempts, "changed_cells": self.changed_cells,
            "progress": self.progress,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "LocalTransition":
        return cls(
            int(raw.get("level", 0)), int(raw.get("x", 0)), int(raw.get("y", 0)),
            int(raw.get("before_value", 0)), int(raw.get("after_value", 0)),
            max(0, int(raw.get("attempts", 0))),
            max(0, int(raw.get("changed_cells", 0))),
            bool(raw.get("progress", False)),
        )


def _ensure_state(sampler) -> None:
    defaults = {
        "_v861_local_edges": {}, "_v861_local_cells": {},
        "_v861_shared_edges": {}, "_v861_shared_cells": {},
        "_v861_dirty_observations": 0, "_v861_decisions": 0,
        "_v861_graph_actions": 0, "_v861_peer_refreshes": 0,
    }
    for name, value in defaults.items():
        if not hasattr(sampler, name):
            setattr(sampler, name, value.copy() if isinstance(value, dict) else value)


def _state_root(sampler) -> Path | None:
    root = getattr(sampler, "_v847_state_root", None)
    return None if root is None else Path(root).parent / _STATE_DIR


def _game_token(game_id: str) -> str:
    return hashlib.blake2b(str(game_id).encode(), digest_size=8, person=b"v861game").hexdigest()


def _state_path(sampler) -> Path | None:
    root = _state_root(sampler)
    actor_id = int(getattr(sampler, "_v847_actor_id", 0))
    return None if root is None or actor_id <= 0 else root / f"{_game_token(sampler.game_id)}-{actor_id}.json"


def _merge_edge(target: TransitionEdge, incoming: TransitionEdge) -> None:
    target.attempts += incoming.attempts
    target.changed_cells = max(target.changed_cells, incoming.changed_cells)
    target.progress = target.progress or incoming.progress
    target.terminal_failures += incoming.terminal_failures


def _merge_cell(target: LocalTransition, incoming: LocalTransition) -> None:
    target.attempts += incoming.attempts
    target.changed_cells = max(target.changed_cells, incoming.changed_cells)
    target.progress = target.progress or incoming.progress


def _save_local(sampler) -> None:
    _ensure_state(sampler)
    path = _state_path(sampler)
    if path is None or sampler._v861_dirty_observations <= 0:
        return
    payload = {
        "schema": _STATE_SCHEMA, "game_id": str(sampler.game_id),
        "actor_id": int(getattr(sampler, "_v847_actor_id", 0)),
        "edges": [sampler._v861_local_edges[k].to_dict() for k in sorted(sampler._v861_local_edges)],
        "cells": [sampler._v861_local_cells[k].to_dict() for k in sorted(sampler._v861_local_cells)],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)
    sampler._v861_dirty_observations = 0


def _refresh_shared(sampler) -> None:
    _ensure_state(sampler)
    root = _state_root(sampler)
    edges: dict[tuple[int, int, int, int], TransitionEdge] = {}
    cells: dict[tuple[int, int, int, int, int], LocalTransition] = {}
    if root is not None and root.exists():
        for path in sorted(root.glob(f"{_game_token(sampler.game_id)}-*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            if raw.get("schema") != _STATE_SCHEMA or raw.get("game_id") != str(sampler.game_id):
                continue
            for item in raw.get("edges", ()):
                if not isinstance(item, dict):
                    continue
                try: row = TransitionEdge.from_dict(item)
                except (TypeError, ValueError): continue
                current = edges.get(row.key())
                if current is None: edges[row.key()] = row
                else: _merge_edge(current, row)
            for item in raw.get("cells", ()):
                if not isinstance(item, dict):
                    continue
                try: row = LocalTransition.from_dict(item)
                except (TypeError, ValueError): continue
                current = cells.get(row.key())
                if current is None: cells[row.key()] = row
                else: _merge_cell(current, row)
    for key, row in sampler._v861_local_edges.items():
        if key not in edges: edges[key] = TransitionEdge.from_dict(row.to_dict())
    for key, row in sampler._v861_local_cells.items():
        if key not in cells: cells[key] = LocalTransition.from_dict(row.to_dict())
    sampler._v861_shared_edges = edges
    sampler._v861_shared_cells = cells
    sampler._v861_peer_refreshes += 1


def _click_payload(action: int) -> tuple[int, int] | None:
    try:
        from v8.learning_blockers_v055 import unpack_action_choice
        native, payload = unpack_action_choice(int(action))
    except (TypeError, ValueError):
        return None
    if native != 6 or not payload:
        return None
    return int(payload["x"]), int(payload["y"])


def _cell_value(grid, x: int, y: int) -> int | None:
    if grid is None:
        return None
    try:
        import numpy as np
        array = np.asarray(grid)
        if array.ndim < 2 or not (0 <= y < array.shape[0] and 0 <= x < array.shape[1]):
            return None
        return int(array[y, x])
    except (TypeError, ValueError, IndexError):
        return None


def _record_transition(sampler, intervention, kwargs) -> None:
    _ensure_state(sampler)
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
    edge = TransitionEdge(before_level, before_context, action, after_context, 1, changed, progress, int(terminal == "GAME_OVER"))
    current = sampler._v861_local_edges.get(edge.key())
    if current is None: sampler._v861_local_edges[edge.key()] = edge
    else: _merge_edge(current, edge)

    x, y = payload
    before_value = _cell_value(getattr(sampler, "_v861_before_observation", None), x, y)
    after_value = _cell_value(getattr(sampler, "_v861_after_observation", None), x, y)
    if before_value is not None and after_value is not None:
        cell = LocalTransition(before_level, x, y, before_value, after_value, 1, changed, progress)
        current_cell = sampler._v861_local_cells.get(cell.key())
        if current_cell is None: sampler._v861_local_cells[cell.key()] = cell
        else: _merge_cell(current_cell, cell)
    sampler._v861_dirty_observations += 1
    if sampler._v861_dirty_observations >= _PUBLISH_EVERY or progress:
        _save_local(sampler)
        _refresh_shared(sampler)


def _distance_to_progress(edges: dict, level: int) -> dict[int, int]:
    reverse: dict[int, set[int]] = {}
    goals: set[int] = set()
    for edge in edges.values():
        if edge.level != level or edge.terminal_failures >= edge.attempts:
            continue
        reverse.setdefault(edge.after_context, set()).add(edge.before_context)
        if edge.progress: goals.add(edge.before_context)
    distances = {context: 0 for context in goals}
    queue = deque((context, 0) for context in sorted(goals))
    while queue:
        context, distance = queue.popleft()
        if distance >= _MAX_GRAPH_DISTANCE: continue
        for predecessor in sorted(reverse.get(context, ())):
            if predecessor not in distances:
                distances[predecessor] = distance + 1
                queue.append((predecessor, distance + 1))
    return distances


def _graph_action(sampler, *, level: int, context: int, actions: tuple[int, ...]) -> int | None:
    _ensure_state(sampler)
    available = {int(v) for v in actions}
    edges = sampler._v861_shared_edges
    candidates = [e for e in edges.values() if e.level == level and e.before_context == context and e.action in available and e.changed_cells > 0 and e.terminal_failures < e.attempts]
    if candidates:
        distances = _distance_to_progress(edges, level)
        best = min(candidates, key=lambda e: (-int(e.progress), distances.get(e.after_context, _MAX_GRAPH_DISTANCE + 1), e.terminal_failures, e.attempts, -e.changed_cells, e.action))
        if best.progress or best.after_context in distances:
            return best.action

    grid = getattr(sampler, "_v861_current_observation", None)
    local = []
    for action in sorted(available):
        payload = _click_payload(action)
        if payload is None: continue
        x, y = payload
        value = _cell_value(grid, x, y)
        if value is None: continue
        matches = [c for c in sampler._v861_shared_cells.values() if c.level == level and c.x == x and c.y == y and c.before_value == value and c.after_value != value and c.changed_cells > 0]
        if matches:
            local.append((-int(any(c.progress for c in matches)), sum(c.attempts for c in matches), -max(c.changed_cells for c in matches), action))
    return None if not local else int(min(local)[3])


def _forced_action_v861(self, *, level: int, context: int, actions: tuple[int, ...], history: tuple[int, ...]) -> int | None:
    _ensure_state(self)
    if (getattr(self, "_v860_pending_action", None) is not None or self.base.replay_actions or self.base.replay_target is not None or self.base.verification is not None or self.active_sequence or getattr(self, "_v832_persist_action", None) is not None):
        return _BASE_FORCED_ACTION(self, level=level, context=context, actions=tuple(actions), history=tuple(history))
    self._v861_decisions += 1
    if self._v861_decisions == 1 or self._v861_decisions % _REFRESH_EVERY == 0:
        _save_local(self)
        _refresh_shared(self)
    action = _graph_action(self, level=level, context=context, actions=tuple(actions))
    if action is None:
        return _BASE_FORCED_ACTION(self, level=level, context=context, actions=tuple(actions), history=tuple(history))
    from v8 import decision_point_sampling_v821 as sampling
    from v8 import sampling_portfolio_v831 as portfolio
    self.base.current = sampling.Intervention("TRANSITION_GRAPH", (level, context), action, tuple(history))
    self._v861_graph_actions += 1
    portfolio._set_mode("PROGRESS")
    portfolio._set_source(context, "CLICK_TRANSITION_GRAPH", (action,))
    return action


def _observe_transition_v861(self, **kwargs):
    _ensure_state(self)
    intervention = self.base.current
    if intervention is not None:
        _record_transition(self, intervention, kwargs)
    return _BASE_OBSERVE_TRANSITION(self, **kwargs)


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
    global _INSTALLED, _BASE_FORCED_ACTION, _BASE_OBSERVE_TRANSITION
    if _INSTALLED:
        return
    from v8 import sampling_evidence_frontier_v847_fixups as frontier_fixups
    _BASE_FORCED_ACTION = frontier_fixups._BASE_LOWER_FORCED
    frontier_fixups._BASE_LOWER_FORCED = _forced_action_v861
    _BASE_OBSERVE_TRANSITION = frontier_fixups._BASE_LOWER_OBSERVE
    frontier_fixups._BASE_LOWER_OBSERVE = _observe_transition_v861
    _INSTALLED = True
