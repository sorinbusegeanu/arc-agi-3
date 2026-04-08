from __future__ import annotations

from typing import Any

from v4.agentContract.types import V4Observation
from v4_5.contracts.boardPerceptionReport import BoardPerceptionReport
from v4_5.perception.board_builder import DeterministicBoardBuilder
from v4_5.perception.board_fusion import BoardPerceptionFusion, LearnedBoardSupplementStub
from v4_5.perception.board_geometry import BoardGeometryExtractor


class BoardPerceptionService:
    module_name = "BoardPerceptionService"

    def __init__(
        self,
        *,
        geometry_extractor: BoardGeometryExtractor | None = None,
        builder: DeterministicBoardBuilder | None = None,
        fusion: BoardPerceptionFusion | None = None,
        learned_stub: LearnedBoardSupplementStub | None = None,
    ) -> None:
        self.geometry_extractor = geometry_extractor or BoardGeometryExtractor()
        self.builder = builder or DeterministicBoardBuilder()
        self.fusion = fusion or BoardPerceptionFusion()
        self.learned_stub = learned_stub or LearnedBoardSupplementStub()

    def build_report(self, *, observations: tuple[Any, Any, Any], round_id: str) -> BoardPerceptionReport:
        geometry_result = self.geometry_extractor.extract(observations)
        board_state = self.builder.build(geometry_result=geometry_result, round_id=round_id)
        learned_output = self.learned_stub.infer(observations=observations, geometry_result=geometry_result)
        return self.fusion.fuse(board_state=board_state, round_id=round_id, learned_output=learned_output)

    def observation_window_for_discovery(
        self,
        *,
        observation: Any | None,
        bootstrap_capture_bundle: Any | None = None,
        parsed_state: Any | None = None,
    ) -> tuple[Any, Any, Any]:
        if bootstrap_capture_bundle is not None and getattr(bootstrap_capture_bundle, "step_records", ()):
            frames = []
            first = bootstrap_capture_bundle.step_records[0]
            if getattr(first, "pre_observation_ref", None) is not None:
                frames.append({"frame": getattr(first, "pre_observation_ref")})
            for record in bootstrap_capture_bundle.step_records:
                if getattr(record, "post_observation_ref", None) is not None:
                    frames.append({"frame": getattr(record, "post_observation_ref")})
            if len(frames) >= 3:
                return tuple(frames[-3:])  # type: ignore[return-value]
        previous = getattr(parsed_state, "previous_observation", None)
        current = observation
        if isinstance(current, V4Observation):
            if isinstance(previous, V4Observation):
                return (previous, current, current)
            return (current, current, current)
        if current is not None:
            return (current, current, current)
        raise ValueError("board perception requires observations")

