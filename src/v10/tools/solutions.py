"""Persistent, state-verified level transition and solution records."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..frames import FrameStore
from ..runlog import append_jsonl, dumps, read_jsonl


LEVEL_SOLUTIONS_SCHEMA = "v10.level_solutions/1"


def level_key(observation: Mapping[str, Any]) -> str:
    """Identify a level using the environment's level and progress counters."""
    level = observation.get("level")
    completed = observation.get("levels_completed")
    if level is None and completed is None:
        return "unknown"
    return json.dumps(
        {"level": level, "levels_completed": completed},
        sort_keys=True, separators=(",", ":"),
    )


def state_fingerprint(observation: Mapping[str, Any]) -> str:
    """Hash the visible grid and state fields used to safely resume a route."""
    frame = np.asarray(observation.get("frame"))
    if frame.ndim != 2 or frame.size == 0:
        raise ValueError("solution state fingerprint requires a non-empty 2-D frame")
    frame = np.ascontiguousarray(frame)
    actions = observation.get("legal_actions", [])
    if isinstance(actions, Mapping):
        action_ids = [str(name) for name in actions]
    elif isinstance(actions, Sequence) and not isinstance(actions, (str, bytes)):
        action_ids = [
            str(item if isinstance(item, str) else item.get("id", item.get("action", "")))
            for item in actions if isinstance(item, (str, Mapping))
        ]
    else:
        action_ids = []
    metadata = json.dumps({
        "shape": list(frame.shape), "dtype": frame.dtype.str,
        "level": observation.get("level"),
        "levels_completed": observation.get("levels_completed"),
        "game_state": observation.get("game_state"),
        "terminal": bool(observation.get("terminal", False)),
        "legal_actions": action_ids,
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256()
    digest.update(metadata.encode("utf-8"))
    digest.update(frame.tobytes(order="C"))
    return digest.hexdigest()


def _action_ids(observation: Mapping[str, Any]) -> list[str]:
    actions = observation.get("legal_actions", [])
    if isinstance(actions, Mapping):
        return [str(name) for name in actions]
    if isinstance(actions, Sequence) and not isinstance(actions, (str, bytes)):
        return [
            str(item if isinstance(item, str) else item.get("id", item.get("action", "")))
            for item in actions if isinstance(item, (str, Mapping))
        ]
    return []


class LevelSolutionStore:
    """Append-only store for observed edges and verified successful routes."""

    def __init__(self, path: Path | str, game_id: str) -> None:
        self.path = Path(path)
        self.game_id = str(game_id)
        self._rows = read_jsonl(self.path)
        self._edge_ids = {str(row["id"]) for row in self._rows if row.get("kind") == "edge"}
        self._solution_ids = {str(row["id"]) for row in self._rows if row.get("kind") == "solution"}

    def edges(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._rows if row.get("kind") == "edge"]

    def solutions(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._rows if row.get("kind") == "solution"]

    def import_successful_run(self, run_dir: Path | str, *, run_id: str | None = None) -> list[dict[str, Any]]:
        """Import fully evidenced completed routes from a recorded successful run.

        The import accepts only a contiguous evidence chain ending in WIN. Frames,
        legal actions, action ordering, progress counters, and level boundaries are
        checked before any route is recorded. Reimporting the same experience is safe.
        """
        root = Path(run_dir)
        records = read_jsonl(root / "evidence.jsonl")
        states = [row for row in records if row.get("type") == "state"]
        transitions = [row for row in records if row.get("type") == "transition"]
        if len(states) != 1 or not transitions:
            raise ValueError("successful-run import requires one initial state and transitions")
        frames = FrameStore(root)
        initial = states[0]
        if initial.get("schema") != "v10.evidence/1":
            raise ValueError("unsupported initial evidence schema")
        current = self._evidence_observation(initial, frames)
        initial_game = str(current.get("game_id") or self.game_id)
        if initial_game != self.game_id:
            raise ValueError(f"run game_id {initial_game} does not match {self.game_id}")
        route_start = current
        validated: list[tuple[dict[str, Any], str, dict[str, Any], dict[str, Any], Mapping[str, Any]]] = []
        previous_evidence_id = str(initial.get("evidence_id", ""))
        previous_step = 0
        run_identity = str(run_id or f"import:{root.resolve()}")
        saw_win = False
        final_completed = current.get("levels_completed")
        for row in transitions:
            if row.get("schema") != "v10.evidence/1" or int(row.get("step", -1)) != previous_step + 1:
                raise ValueError("recorded transitions are not a contiguous v10 evidence sequence")
            if row.get("pre_evidence_id") != previous_evidence_id:
                raise ValueError("recorded transition evidence chain is broken")
            if row.get("pre_frame_id") != current.get("frame_id"):
                raise ValueError("recorded transition frame chain is broken")
            action = str(row.get("action", ""))
            if action not in _action_ids(current):
                raise ValueError(f"recorded action {action!r} was not legal in its source state")
            before_completed = current.get("levels_completed")
            after = self._evidence_observation(row, frames)
            after_completed = after.get("levels_completed")
            if before_completed is not None and after_completed is not None:
                delta = int(after_completed) - int(before_completed)
                if delta not in {0, 1}:
                    raise ValueError("recorded level progress has a missing or invalid boundary")
                if delta == 0 and after.get("level") != current.get("level"):
                    raise ValueError("level changed without a recorded completion")
            args = row.get("args", {})
            if not isinstance(args, Mapping):
                raise ValueError("recorded action arguments must be an object")
            from .game import ActionError, _action_table, _validate_args
            try:
                descriptor = _action_table(current.get("legal_actions", []))[action]
                _validate_args(action, dict(args), descriptor)
            except (ActionError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"recorded action arguments are invalid for {action}") from exc
            validated.append((current, action, dict(args), after, row))
            completed = (
                before_completed is not None and after_completed is not None
                and int(after_completed) == int(before_completed) + 1
            )
            won = str(after.get("game_state", "")).upper() == "WIN"
            if won:
                if not after.get("terminal") or row is not transitions[-1]:
                    raise ValueError("WIN must be the final terminal transition in the run")
                saw_win = True
            previous_evidence_id = str(row.get("evidence_id", ""))
            previous_step = int(row["step"])
            current = after
            final_completed = current.get("levels_completed")
        if not saw_win:
            raise ValueError("recorded run does not contain a complete terminal winning trajectory")
        metrics_path = root / "metrics.json"
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if (
                metrics.get("won") is not True
                or int(metrics.get("actions", -1)) != len(transitions)
                or metrics.get("levels_completed") != final_completed
            ):
                raise ValueError("recorded metrics do not agree with the successful evidence trajectory")

        # Do not mutate the append-only store until the whole history has validated.
        route_edges: list[dict[str, Any]] = []
        imported: list[dict[str, Any]] = []
        for before, action, args, after, row in validated:
            edge = self.record_transition(before, action, args, after, run_id=run_identity)
            route_edges.append(edge)
            previous_completed = before.get("levels_completed")
            current_completed = after.get("levels_completed")
            completed = (
                previous_completed is not None and current_completed is not None
                and int(current_completed) == int(previous_completed) + 1
            )
            won = str(after.get("game_state", "")).upper() == "WIN"
            if completed or won:
                saved = self.record_successful_trajectory(
                    route_start, route_edges, run_id=run_identity,
                    success="win" if won else "level_completed",
                )
                if saved is not None:
                    imported.append(saved)
                route_start = after
                route_edges = []
        return imported

    @staticmethod
    def _evidence_observation(row: Mapping[str, Any], frames: FrameStore) -> dict[str, Any]:
        frame_id = row.get("frame_id") if row.get("type") == "state" else row.get("post_frame_id")
        if not isinstance(frame_id, str) or not frames.exists(frame_id):
            raise ValueError("recorded evidence references a missing frame")
        return {
            "frame": frames.get(frame_id), "frame_id": frame_id,
            "game_id": row.get("game_id"), "legal_actions": row.get("legal_actions", []),
            "game_state": row.get("game_state"), "level": row.get("level"),
            "levels_completed": row.get("levels_completed"), "win_levels": row.get("win_levels"),
            "terminal": row.get("terminal", False),
        }

    def record_transition(
        self,
        before: Mapping[str, Any],
        action: str,
        args: Mapping[str, Any],
        after: Mapping[str, Any],
        *,
        run_id: str,
    ) -> dict[str, Any]:
        previous_completed = before.get("levels_completed")
        current_completed = after.get("levels_completed")
        level_completed = (
            previous_completed is not None and current_completed is not None
            and int(current_completed) > int(previous_completed)
        )
        won = str(after.get("game_state", "")).upper() == "WIN"
        body = {
            "schema": LEVEL_SOLUTIONS_SCHEMA, "kind": "edge", "game_id": self.game_id,
            "level_key": level_key(before), "source_state": state_fingerprint(before),
            "target_state": state_fingerprint(after), "action": str(action),
            "args": dict(args), "action_order": _action_ids(before),
            "level_completed": level_completed, "won": won,
            "run_id": str(run_id),
        }
        identity = {key: value for key, value in body.items() if key != "run_id"}
        edge_id = hashlib.sha256(dumps(identity).encode("utf-8")).hexdigest()[:24]
        row = {"id": edge_id, **body}
        if edge_id not in self._edge_ids:
            append_jsonl(self.path, row)
            self._rows.append(row)
            self._edge_ids.add(edge_id)
        return row

    def record_successful_trajectory(
        self,
        start: Mapping[str, Any],
        edges: Sequence[Mapping[str, Any]],
        *,
        run_id: str,
        success: str,
    ) -> dict[str, Any] | None:
        """Persist a complete level route only when completion or WIN was observed."""
        if not edges or success not in {"level_completed", "win"}:
            return None
        steps = [{
            "source_state": edge["source_state"], "target_state": edge["target_state"],
            "action": edge["action"], "args": dict(edge.get("args", {})),
        } for edge in edges]
        body = {
            "schema": LEVEL_SOLUTIONS_SCHEMA, "kind": "solution", "game_id": self.game_id,
            "level_key": level_key(start), "start_state": state_fingerprint(start),
            "end_state": steps[-1]["target_state"], "steps": steps,
            "success": success, "run_id": str(run_id),
        }
        identity = {key: value for key, value in body.items() if key != "run_id"}
        solution_id = hashlib.sha256(dumps(identity).encode("utf-8")).hexdigest()[:24]
        row = {"id": solution_id, **body}
        if solution_id not in self._solution_ids:
            append_jsonl(self.path, row)
            self._rows.append(row)
            self._solution_ids.add(solution_id)
        return row

    def shortest_known_solution(self, observation: Mapping[str, Any]) -> list[dict[str, Any]] | None:
        """Return the shortest unambiguous observed path to a completion edge."""
        start_state = state_fingerprint(observation)
        wanted_level = level_key(observation)
        candidates = [row for row in self.edges() if row.get("level_key") == wanted_level]
        by_action: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for edge in candidates:
            action_key = (
                str(edge.get("source_state")), str(edge.get("action")),
                dumps(edge.get("args", {})),
            )
            by_action.setdefault(action_key, []).append(edge)

        unambiguous: list[dict[str, Any]] = []
        for same_action_edges in by_action.values():
            targets = {str(edge.get("target_state")) for edge in same_action_edges}
            if len(targets) == 1:
                unambiguous.append(same_action_edges[0])
        adjacency: dict[str, list[dict[str, Any]]] = {}
        for edge in unambiguous:
            adjacency.setdefault(str(edge["source_state"]), []).append(edge)
        for edges in adjacency.values():
            edges.sort(key=lambda edge: (
                (edge.get("action_order") or []).index(edge["action"])
                if edge.get("action") in (edge.get("action_order") or []) else len(edge.get("action_order") or []),
                str(edge.get("action")), dumps(edge.get("args", {})),
            ))

        queue = deque([(start_state, [])])
        visited = {start_state}
        while queue:
            current, path = queue.popleft()
            for edge in adjacency.get(current, []):
                next_path = [*path, edge]
                if edge.get("level_completed") or edge.get("won"):
                    return next_path
                target = str(edge["target_state"])
                if target not in visited:
                    visited.add(target)
                    queue.append((target, next_path))
        return None


__all__ = ["LEVEL_SOLUTIONS_SCHEMA", "LevelSolutionStore", "level_key", "state_fingerprint"]
