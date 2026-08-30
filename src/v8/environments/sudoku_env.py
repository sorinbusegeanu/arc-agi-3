from __future__ import annotations

from random import Random

from v8.environment_contract import BoundaryEvent, BoundaryScope
from v8.environments.schemas import ActionSchema, EnvironmentIdentity, ObservationSchema
from v8.model import stable_u64


SUDOKU_ENV_ID = "ArcAgi/Sudoku-v0"
_SIDE = 9
_BOX = 3
_ACTION_COUNT = _SIDE * _SIDE * _SIDE
_DEFAULT_CLUES = 36


def encode_sudoku_action(row: int, column: int, digit: int) -> int:
    row = int(row)
    column = int(column)
    digit = int(digit)
    if not 0 <= row < _SIDE:
        raise ValueError("sudoku row must be in [0, 8]")
    if not 0 <= column < _SIDE:
        raise ValueError("sudoku column must be in [0, 8]")
    if not 1 <= digit <= _SIDE:
        raise ValueError("sudoku digit must be in [1, 9]")
    return (row * _SIDE + column) * _SIDE + (digit - 1)


def decode_sudoku_action(action: int) -> tuple[int, int, int]:
    action = int(action)
    if not 0 <= action < _ACTION_COUNT:
        raise ValueError(f"sudoku action must be in [0, {_ACTION_COUNT - 1}]")
    cell, digit_index = divmod(action, _SIDE)
    row, column = divmod(cell, _SIDE)
    return row, column, digit_index + 1


def _generated_solution(seed: int) -> tuple[int, ...]:
    rng = Random(int(seed))

    def pattern(row: int, column: int) -> int:
        return (_BOX * (row % _BOX) + row // _BOX + column) % _SIDE

    groups = list(range(_BOX))
    rows = [
        group * _BOX + row
        for group in rng.sample(groups, len(groups))
        for row in rng.sample(groups, len(groups))
    ]
    columns = [
        group * _BOX + column
        for group in rng.sample(groups, len(groups))
        for column in rng.sample(groups, len(groups))
    ]
    digits = rng.sample(list(range(1, _SIDE + 1)), _SIDE)
    return tuple(digits[pattern(row, column)] for row in rows for column in columns)


def _candidate_actions(board: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    rows = [set() for _ in range(_SIDE)]
    columns = [set() for _ in range(_SIDE)]
    boxes = [set() for _ in range(_SIDE)]
    for index, raw in enumerate(board):
        value = int(raw)
        if value <= 0:
            continue
        row, column = divmod(index, _SIDE)
        box = (row // _BOX) * _BOX + column // _BOX
        rows[row].add(value)
        columns[column].add(value)
        boxes[box].add(value)

    actions: list[int] = []
    for index, raw in enumerate(board):
        if int(raw) != 0:
            continue
        row, column = divmod(index, _SIDE)
        box = (row // _BOX) * _BOX + column // _BOX
        blocked = rows[row] | columns[column] | boxes[box]
        for digit in range(1, _SIDE + 1):
            if digit not in blocked:
                actions.append(encode_sudoku_action(row, column, digit))
    return tuple(actions)


def _valid_complete(board: tuple[int, ...] | list[int]) -> bool:
    wanted = set(range(1, _SIDE + 1))
    if len(board) != _SIDE * _SIDE or any(int(value) == 0 for value in board):
        return False
    for row in range(_SIDE):
        if {int(board[row * _SIDE + column]) for column in range(_SIDE)} != wanted:
            return False
    for column in range(_SIDE):
        if {int(board[row * _SIDE + column]) for row in range(_SIDE)} != wanted:
            return False
    for box_row in range(0, _SIDE, _BOX):
        for box_column in range(0, _SIDE, _BOX):
            values = {
                int(board[(box_row + dr) * _SIDE + box_column + dc])
                for dr in range(_BOX)
                for dc in range(_BOX)
            }
            if values != wanted:
                return False
    return True


class SudokuAdapter:
    """Deterministic 9x9 Sudoku environment for cross-domain memory research.

    Actions are locally legal placements only. A valid completed board is a positive
    episode boundary. A locally legal path that reaches an incomplete board with no
    legal placements is a negative boundary. New episodes use deterministic seeded
    puzzle instances so the agent cannot solve by memorizing one board.
    """

    def __init__(self, *, seed: int = 0, clues: int = _DEFAULT_CLUES) -> None:
        clues = int(clues)
        if not 17 <= clues < _SIDE * _SIDE:
            raise ValueError("sudoku clues must be in [17, 80]")
        self.seed = int(seed)
        self.clues = clues
        self.environment_id = SUDOKU_ENV_ID
        self.observation_schema = ObservationSchema("grid", "sudoku9x9:digits=0-9")
        self.action_schema = ActionSchema("placement", f"sudoku9x9:n={_ACTION_COUNT}")
        self.identity = EnvironmentIdentity(
            "puzzle",
            SUDOKU_ENV_ID,
            f"size=9,clues={self.clues}",
            f"seed={self.seed}",
        )
        self._episode = 0
        self._board: list[int] = [0] * (_SIDE * _SIDE)
        self._solution: tuple[int, ...] = ()
        self._givens: frozenset[int] = frozenset()
        self._boundary = BoundaryEvent()
        self.reset()

    def close(self) -> None:
        return None

    def observe(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self._board)

    def reset(self) -> tuple[int, ...]:
        episode_seed = self.seed + self._episode * 104729
        self._episode += 1
        self._solution = _generated_solution(episode_seed)
        rng = Random(episode_seed ^ 0x5A17)
        given_indices = frozenset(rng.sample(range(_SIDE * _SIDE), self.clues))
        self._givens = given_indices
        self._board = [
            int(self._solution[index]) if index in given_indices else 0
            for index in range(_SIDE * _SIDE)
        ]
        self._boundary = BoundaryEvent()
        return self.observe()

    @property
    def clue_count(self) -> int:
        return len(self._givens)

    def available_actions(self) -> tuple[int, ...]:
        if not self._boundary.continuation:
            return ()
        return _candidate_actions(self._board)

    def step(self, action: int) -> tuple[int, ...]:
        if not self._boundary.continuation:
            raise RuntimeError("sudoku episode is complete; reset before stepping")
        action = int(action)
        legal = set(self.available_actions())
        if action not in legal:
            raise ValueError("sudoku action is not legal in the current board")
        row, column, digit = decode_sudoku_action(action)
        index = row * _SIDE + column
        if index in self._givens or self._board[index] != 0:
            raise ValueError("sudoku action targets a fixed or occupied cell")
        self._board[index] = digit

        if _valid_complete(self._board):
            self._boundary = BoundaryEvent(BoundaryScope.EPISODE, 1, False)
        else:
            self._boundary = BoundaryEvent()
            if not _candidate_actions(self._board):
                self._boundary = BoundaryEvent(BoundaryScope.EPISODE, -1, False)
        return self.observe()

    def observation_signature(self, observation) -> int:
        values = tuple(int(value) for value in observation)
        if len(values) != _SIDE * _SIDE:
            raise ValueError("sudoku observation must contain 81 cells")
        return stable_u64(
            self.observation_schema.schema_id,
            *values,
            person=b"v8-sudoku-observation",
        )

    def cognitive_boundary_event(self) -> BoundaryEvent:
        return self._boundary

    def cognitive_context_signature(self) -> int:
        return self.observation_signature(self.observe())

    def cognitive_transition_signature(self, before, after) -> int:
        before_values = tuple(int(value) for value in before)
        after_values = tuple(int(value) for value in after)
        changed = tuple(
            index
            for index, (left, right) in enumerate(zip(before_values, after_values, strict=True))
            if left != right
        )
        changed_index = changed[0] if len(changed) == 1 else -1
        changed_value = after_values[changed_index] if changed_index >= 0 else 0
        return stable_u64(
            self.observation_schema.schema_id,
            changed_index,
            changed_value,
            int(self._boundary.primary_valence),
            person=b"v8-sudoku-transition",
        )

    def cognitive_family_signature(self, before, after) -> int:
        before_values = tuple(int(value) for value in before)
        after_values = tuple(int(value) for value in after)
        changed = tuple(
            index
            for index, (left, right) in enumerate(zip(before_values, after_values, strict=True))
            if left != right
        )
        if len(changed) != 1:
            return stable_u64(len(changed), person=b"v8-sudoku-family")
        row, column = divmod(changed[0], _SIDE)
        return stable_u64(
            row % _BOX,
            column % _BOX,
            int(self._boundary.primary_valence),
            person=b"v8-sudoku-family",
        )

    def cognitive_changed_extent(self, before, after) -> int:
        return sum(
            int(int(left) != int(right))
            for left, right in zip(before, after, strict=True)
        )

    def cognitive_target_reached(self, target=None, outcome_uid=None) -> bool:
        del target, outcome_uid
        return bool(self._boundary.positive)
