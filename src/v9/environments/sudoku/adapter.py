from __future__ import annotations

from random import Random
from typing import Any

from v9.environments.base import StructuralAdapter
from v9.environments.contract import BoundaryEvent, BoundaryScope, WithinActionFrame, WithinActionTrace
from v9.environments.schemas import ActionSchema, EnvironmentIdentity, ObservationSchema

SIDE = 9
BOX = 3
ACTION_COUNT = SIDE * SIDE * SIDE


def encode_action(row: int, column: int, digit: int) -> int:
    if not 0 <= int(row) < SIDE or not 0 <= int(column) < SIDE or not 1 <= int(digit) <= SIDE:
        raise ValueError("invalid Sudoku action coordinates")
    return (int(row) * SIDE + int(column)) * SIDE + int(digit) - 1


def decode_action(action: int) -> tuple[int, int, int]:
    if not 0 <= int(action) < ACTION_COUNT:
        raise ValueError("Sudoku action is out of range")
    cell, digit = divmod(int(action), SIDE)
    row, column = divmod(cell, SIDE)
    return row, column, digit + 1


def _solution(seed: int) -> tuple[int, ...]:
    rng = Random(seed)
    groups = list(range(BOX))
    rows = [group * BOX + row for group in rng.sample(groups, BOX) for row in rng.sample(groups, BOX)]
    columns = [group * BOX + col for group in rng.sample(groups, BOX) for col in rng.sample(groups, BOX)]
    digits = rng.sample(list(range(1, SIDE + 1)), SIDE)
    return tuple(digits[(BOX * (row % BOX) + row // BOX + column) % SIDE] for row in rows for column in columns)


def _available(board: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    actions: list[int] = []
    for index, value in enumerate(board):
        if value:
            continue
        row, column = divmod(index, SIDE)
        used = {board[row * SIDE + col] for col in range(SIDE)} | {board[r * SIDE + column] for r in range(SIDE)}
        br, bc = (row // BOX) * BOX, (column // BOX) * BOX
        used |= {board[(br + dr) * SIDE + bc + dc] for dr in range(BOX) for dc in range(BOX)}
        actions.extend(encode_action(row, column, digit) for digit in range(1, SIDE + 1) if digit not in used)
    return tuple(actions)


class SudokuAdapter(StructuralAdapter):
    def __init__(self, *, seed: int = 0, clues: int = 36) -> None:
        if not 17 <= int(clues) < 81:
            raise ValueError("Sudoku clues must be in [17, 80]")
        self.seed, self.clues, self._episode = int(seed), int(clues), 0
        self._identity = EnvironmentIdentity("puzzle", "Sudoku-v0", f"clues={clues}", f"seed={seed}")
        self._observation_schema = ObservationSchema("grid", "sudoku9x9")
        self._action_schema = ActionSchema("placement", f"n={ACTION_COUNT}")
        self._boundary = BoundaryEvent()
        self._last_trace = None
        self._board: list[int] = []
        self.reset()

    def reset(self) -> tuple[int, ...]:
        seed = self.seed + self._episode * 104729
        self._episode += 1
        solution = _solution(seed)
        givens = set(Random(seed ^ 0x5A17).sample(range(81), self.clues))
        self._board = [solution[index] if index in givens else 0 for index in range(81)]
        self._boundary = BoundaryEvent()
        self._last_trace = None
        return self.observe()

    def observe(self) -> tuple[int, ...]:
        return tuple(self._board)

    def available_actions(self) -> tuple[int, ...]:
        return () if not self._boundary.continuation else _available(self._board)

    def step(self, native_action: Any) -> tuple[int, ...]:
        before = self.observe()
        action = int(native_action)
        if action not in set(self.available_actions()):
            raise ValueError("Sudoku action is not locally legal")
        row, column, digit = decode_action(action)
        self._board[row * SIDE + column] = digit
        available = _available(self._board)
        if not any(value == 0 for value in self._board):
            self._boundary = BoundaryEvent(BoundaryScope.EPISODE, 1, False)
        elif not available:
            self._boundary = BoundaryEvent(BoundaryScope.EPISODE, -1, False)
        else:
            self._boundary = BoundaryEvent()
        self._last_trace = WithinActionTrace(before, (WithinActionFrame(self.observe(), 0),), self.observe())
        return self.observe()

