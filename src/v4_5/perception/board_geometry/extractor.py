from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from v4.agentContract.types import V4Observation
from v4_5.contracts.boardState import BoardGeometrySummary


@dataclass(frozen=True)
class BoardGeometryResult:
    summary: BoardGeometrySummary
    window_size: int
    frames: tuple[tuple[tuple[int, ...], ...], ...]


class BoardGeometryExtractor:
    module_name = "BoardGeometryExtractor"

    def extract(self, observations: tuple[Any, Any, Any]) -> BoardGeometryResult:
        if len(observations) != 3:
            raise ValueError("board perception requires exactly the last 3 observations")
        frames = tuple(self._extract_frame(item) for item in observations)
        widths = {len(frame[0]) for frame in frames if frame and frame[0]}
        heights = {len(frame) for frame in frames}
        if len(widths) != 1 or len(heights) != 1:
            raise ValueError("board perception observations must share frame geometry")
        width = next(iter(widths), 0)
        height = next(iter(heights), 0)
        return BoardGeometryResult(
            summary=BoardGeometrySummary(frame_width=width, frame_height=height),
            window_size=3,
            frames=frames,
        )

    def _extract_frame(self, observation: Any) -> tuple[tuple[int, ...], ...]:
        if isinstance(observation, V4Observation):
            return tuple(tuple(int(cell) for cell in row) for row in observation.frame[0])
        if isinstance(observation, dict):
            frame = observation.get("frame")
            if isinstance(frame, tuple) and frame and isinstance(frame[0], tuple) and frame and frame[0] and isinstance(frame[0][0], tuple):
                return tuple(tuple(int(cell) for cell in row) for row in frame[0])
            if isinstance(frame, tuple):
                return tuple(tuple(int(cell) for cell in row) for row in frame)
            raw = observation.get("raw_observation_payload", {})
            pre_frame = raw.get("pre_frame")
            if isinstance(pre_frame, tuple):
                return tuple(tuple(int(cell) for cell in row) for row in pre_frame)
        raise ValueError("unsupported observation shape for board perception")

