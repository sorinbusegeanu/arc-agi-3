from __future__ import annotations


def build_visit_heatmap(episodes: list[dict], *, width: int, height: int) -> list[list[int]]:
    grid = [[0 for _ in range(width)] for _ in range(height)]
    for episode in episodes:
        for step in episode.get("steps", []):
            cell = step.get("avatar_cell")
            if not isinstance(cell, (list, tuple)) or len(cell) != 2:
                continue
            x, y = int(cell[0]), int(cell[1])
            if 0 <= y < height and 0 <= x < width:
                grid[y][x] += 1
    return grid

