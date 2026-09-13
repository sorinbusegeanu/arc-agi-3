from __future__ import annotations

from random import Random
from typing import Any

from v9.environments.base import StructuralAdapter
from v9.environments.contract import BoundaryEvent, BoundaryScope, WithinActionFrame, WithinActionTrace
from v9.environments.schemas import ActionSchema, EnvironmentIdentity, ObservationSchema


WALL = 1
FLOOR = 0
GOAL = 2
BOX = 3
BOX_ON_GOAL = 4
PLAYER = 5
PLAYER_ON_GOAL = 6

DIRECTIONS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}


class SokobanAdapter(StructuralAdapter):
    """Small deterministic native Sokoban family for curriculum training."""

    def __init__(self, *, environment_name: str = "Sokoban-small-v0", seed: int = 0) -> None:
        self.environment_name = str(environment_name)
        self.seed = int(seed)
        self._episode = 0
        self._identity = EnvironmentIdentity("sokoban", self.environment_name, "native-v9", f"seed={seed}")
        self._observation_schema = ObservationSchema("grid", "native-sokoban")
        self._action_schema = ActionSchema("discrete", "n=4")
        self._boundary = BoundaryEvent()
        self._last_trace = None
        self._grid: list[list[int]] = []
        self._player = (1, 1)
        self.reset()

    def _layout(self) -> list[str]:
        layouts = {
            "Sokoban-small-v0": [
                "#####",
                "#   #",
                "# $ #",
                "# .@#",
                "#####",
            ],
            "Sokoban-small-v1": [
                "######",
                "#    #",
                "# $  #",
                "#  #.#",
                "#  @ #",
                "######",
            ],
            "Sokoban-v0": [
                "#######",
                "#     #",
                "# .$  #",
                "#  #  #",
                "#  $ .#",
                "#  @  #",
                "#######",
            ],
            "Sokoban-v1": [
                "#######",
                "# .   #",
                "# $$# #",
                "#   . #",
                "#  @  #",
                "#######",
            ],
        }
        if self.environment_name in layouts:
            return layouts[self.environment_name]
        return layouts["Sokoban-small-v0"]

    def reset(self) -> tuple[tuple[int, ...], ...]:
        lines = self._layout()
        self._grid = []
        for row, line in enumerate(lines):
            cells: list[int] = []
            for col, char in enumerate(line):
                value = FLOOR
                if char == "#":
                    value = WALL
                elif char == ".":
                    value = GOAL
                elif char == "$":
                    value = BOX
                elif char == "*":
                    value = BOX_ON_GOAL
                elif char == "@":
                    value = PLAYER
                    self._player = (row, col)
                elif char == "+":
                    value = PLAYER_ON_GOAL
                    self._player = (row, col)
                cells.append(value)
            self._grid.append(cells)
        self._episode += 1
        self._boundary = BoundaryEvent()
        self._last_trace = None
        return self.observe()

    def observe(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(row) for row in self._grid)

    def available_actions(self) -> tuple[int, ...]:
        return () if not self._boundary.continuation else (0, 1, 2, 3)

    @staticmethod
    def _is_box(value: int) -> bool:
        return value in {BOX, BOX_ON_GOAL}

    @staticmethod
    def _is_goal(value: int) -> bool:
        return value in {GOAL, BOX_ON_GOAL, PLAYER_ON_GOAL}

    def _leave_player(self, row: int, col: int) -> None:
        self._grid[row][col] = GOAL if self._grid[row][col] == PLAYER_ON_GOAL else FLOOR

    def _place_player(self, row: int, col: int) -> None:
        self._grid[row][col] = PLAYER_ON_GOAL if self._grid[row][col] == GOAL else PLAYER
        self._player = (row, col)

    def _move_box(self, src: tuple[int, int], dst: tuple[int, int]) -> None:
        sr, sc = src
        dr, dc = dst
        source_goal = self._grid[sr][sc] == BOX_ON_GOAL
        target_goal = self._grid[dr][dc] == GOAL
        self._grid[sr][sc] = GOAL if source_goal else FLOOR
        self._grid[dr][dc] = BOX_ON_GOAL if target_goal else BOX

    def _solved(self) -> bool:
        return all(value != BOX for row in self._grid for value in row)

    def step(self, native_action: Any) -> tuple[tuple[int, ...], ...]:
        action = int(native_action)
        if action not in DIRECTIONS or action not in self.available_actions():
            raise ValueError("invalid Sokoban action")
        before = self.observe()
        pr, pc = self._player
        dr, dc = DIRECTIONS[action]
        nr, nc = pr + dr, pc + dc
        target = self._grid[nr][nc]
        moved = False

        if target != WALL:
            if self._is_box(target):
                br, bc = nr + dr, nc + dc
                beyond = self._grid[br][bc]
                if beyond in {FLOOR, GOAL}:
                    self._move_box((nr, nc), (br, bc))
                    self._leave_player(pr, pc)
                    self._place_player(nr, nc)
                    moved = True
            elif target in {FLOOR, GOAL}:
                self._leave_player(pr, pc)
                self._place_player(nr, nc)
                moved = True

        solved = self._solved()
        self._boundary = BoundaryEvent(BoundaryScope.EPISODE, 1, False) if solved else BoundaryEvent()
        after = self.observe()
        self._last_trace = WithinActionTrace(before, (WithinActionFrame(after, 0),), after)
        return after
