from __future__ import annotations

from typing import Any

from v9.memory.identity import stable_u64

from .contract import BoundaryEvent, EnvironmentTransition, TaskProgress, WithinActionTrace
from .schemas import ActionSchema, EnvironmentIdentity, ObservationSchema

SemanticFact = tuple[int, int, int, int, float]


def _semantic_id(value: object) -> int:
    return int(stable_u64(str(value).strip().lower(), person=b"v9-semantic"))


def _fact(kind: int, subject: object, relation: int, obj: object = 0, value: float = 0.0) -> SemanticFact:
    s = int(subject) if isinstance(subject, int) else _semantic_id(subject)
    o = int(obj) if isinstance(obj, int) else _semantic_id(obj)
    return (int(kind), s, int(relation), o, float(value))


class StructuralAdapter:
    _identity: EnvironmentIdentity
    _observation_schema: ObservationSchema
    _action_schema: ActionSchema
    _boundary: BoundaryEvent
    _last_trace: WithinActionTrace | None

    def identity(self) -> EnvironmentIdentity:
        return self._identity

    def observation_schema(self) -> ObservationSchema:
        return self._observation_schema

    def action_schema(self) -> ActionSchema:
        return self._action_schema

    def boundary_event(self) -> BoundaryEvent:
        return self._boundary

    def task_progress(self) -> TaskProgress:
        boundary = self._boundary
        return TaskProgress(
            game_id=str(self._identity.environment_type),
            terminal=not bool(boundary.continuation),
            success=bool(boundary.primary_valence > 0 and not boundary.continuation),
            failure=bool(boundary.primary_valence < 0 and not boundary.continuation),
            truncated=bool(not boundary.continuation and boundary.primary_valence == 0),
            score=float(boundary.primary_valence),
        )

    def optional_micro_trace(self) -> WithinActionTrace | None:
        return self._last_trace

    def optional_symbol_stream(self) -> tuple[object, ...]:
        return ()

    def encode_observation(self, observation: Any) -> int:
        try:
            import numpy as np
            array = np.asarray(observation)
            raw = array.tobytes() + str(array.shape).encode("ascii")
        except Exception:
            raw = repr(observation).encode("utf-8")
        return stable_u64(self._observation_schema.schema_id, raw, person=b"v9-env-observe")

    def encode_action(self, action: Any) -> int:
        return int(action)

    def semantic_observation(self, observation: Any) -> tuple[SemanticFact, ...]:
        facts: list[SemanticFact] = []
        family = str(self._identity.family).lower()
        environment = str(self._identity.environment_type).lower()
        instruction = getattr(observation, "instruction_bytes", None)
        world = getattr(observation, "world", None)
        if instruction is not None:
            try:
                text = bytes(instruction).decode("utf-8")
            except Exception:
                text = repr(instruction)
            if text:
                facts.append(_fact(7, "instruction", 1, text, 1.0))
            if world is not None:
                observation = world
        if family == "chess":
            try:
                values = list(observation)
            except Exception:
                values = []
            for square, value in enumerate(values[:64]):
                piece = int(value)
                if piece:
                    entity = _semantic_id(f"chess:{square}")
                    facts.append(_fact(2, entity, 6, piece, float(piece)))
                    facts.append(_fact(5, entity, 21, square, 1.0))
            if len(values) > 64:
                facts.append(_fact(1, "side_to_move", 1, int(values[64]), float(values[64])))
            return tuple(facts)
        if family in {"puzzle", "sudoku"} or "sudoku" in environment:
            try:
                values = list(observation)
            except Exception:
                values = []
            for index, value in enumerate(values[:81]):
                row, col = divmod(index, 9)
                entity = _semantic_id(f"sudoku:{row}:{col}")
                facts.append(_fact(2, entity, 3, row, float(row)))
                facts.append(_fact(2, entity, 4, col, float(col)))
                facts.append(_fact(3, entity, 1, int(value), float(value)))
            return tuple(facts)
        if isinstance(observation, dict):
            image = observation.get("image")
            if image is not None:
                try:
                    shape = tuple(int(v) for v in image.shape)
                    for row in range(min(shape[0], 32)):
                        for col in range(min(shape[1], 32)):
                            cell = [int(v) for v in list(image[row][col])[:3]]
                            if not cell or cell[0] == 0:
                                continue
                            entity = _semantic_id(f"{family}:{row}:{col}")
                            facts.append(_fact(2, entity, 6, cell[0], float(cell[0])))
                            facts.append(_fact(5, entity, 2, row * 4096 + col, 1.0))
                            if len(cell) > 1:
                                facts.append(_fact(3, entity, 5, cell[1], float(cell[1])))
                except Exception:
                    pass
            for key, value in sorted(observation.items()):
                if key == "image":
                    continue
                if isinstance(value, (int, float)):
                    facts.append(_fact(6, str(key), 1, str(key), float(value)))
                elif isinstance(value, str):
                    facts.append(_fact(7, str(key), 1, value, 1.0))
            return tuple(facts[:1024])
        if isinstance(observation, (int, float)):
            return (_fact(6, "state", 1, int(observation), float(observation)),)
        try:
            values = list(observation)
        except Exception:
            return ()
        if values and isinstance(values[0], (list, tuple)):
            for r, row in enumerate(values[:64]):
                for col, value in enumerate(list(row)[:64]):
                    try:
                        numeric = int(value)
                    except Exception:
                        continue
                    if numeric == 0 and family not in {"puzzle", "sudoku"}:
                        continue
                    entity = _semantic_id(f"{family}:{r}:{col}")
                    facts.append(_fact(2, entity, 6, numeric, float(numeric)))
                    facts.append(_fact(5, entity, 2, r * 4096 + col, 1.0))
                    if len(facts) >= 1024:
                        return tuple(facts)
            return tuple(facts)
        names = ()
        if family == "gymnasium" and "cartpole" in environment:
            names = ("cart_position", "cart_velocity", "pole_angle", "pole_velocity")
        elif family == "gymnasium" and "mountaincar" in environment:
            names = ("position", "velocity")
        for index, value in enumerate(values[:128]):
            try:
                numeric = float(value)
            except Exception:
                continue
            name = names[index] if index < len(names) else f"feature_{index}"
            facts.append(_fact(6, name, 1, index, numeric))
        return tuple(facts)

    def semantic_delta(self, before: tuple[SemanticFact, ...], after: tuple[SemanticFact, ...]) -> tuple[SemanticFact, ...]:
        before_set = set(before)
        after_set = set(after)
        rows: list[SemanticFact] = []
        for item in tuple(after_set - before_set)[:512]:
            rows.append((9, int(item[1]), 13, int(item[3]), float(item[4])))
        for item in tuple(before_set - after_set)[:512]:
            rows.append((9, int(item[1]), 14, int(item[3]), float(item[4])))
        return tuple(rows)

    def semantic_action(self, action: Any) -> tuple[SemanticFact, ...]:
        family = str(self._identity.family).lower()
        token = int(action)
        action_type: object = token
        if family == "sokoban":
            action_type = {0: "up", 1: "down", 2: "left", 3: "right"}.get(token, token)
        elif family == "chess":
            try:
                from v9.environments.chess.adapter import decode_move
                move = decode_move(token)
                action_type = "move"
                subject = _semantic_id("action:move")
                return (
                    _fact(8, subject, 23, "move", 1.0),
                    _fact(8, subject, 16, "from", float(move.from_square)),
                    _fact(8, subject, 16, "to", float(move.to_square)),
                )
            except Exception:
                pass
        elif family in {"puzzle", "sudoku"} or "sudoku" in str(self._identity.environment_type).lower():
            try:
                from v9.environments.sudoku.adapter import decode_action
                row, col, digit = decode_action(token)
                action_type = "place"
                subject = _semantic_id("action:place")
                return (
                    _fact(8, subject, 23, "place", 1.0),
                    _fact(8, subject, 16, "row", float(row)),
                    _fact(8, subject, 16, "column", float(col)),
                    _fact(8, subject, 16, "digit", float(digit)),
                )
            except Exception:
                pass
        subject = _semantic_id(f"action:{action_type}")
        return (_fact(8, subject, 23, action_type, 1.0),)

    def transition(self, before: Any, after: Any, action: Any) -> EnvironmentTransition:
        before_actions = tuple(self.available_actions())
        before_signature = self.encode_observation(before)
        after_signature = self.encode_observation(after)
        return EnvironmentTransition(
            before, after, action, self.encode_action(action), before_actions,
            tuple(self.available_actions()),
            {"before_signature": before_signature, "after_signature": after_signature},
            self._boundary, before_signature, after_signature, self._last_trace,
        )

