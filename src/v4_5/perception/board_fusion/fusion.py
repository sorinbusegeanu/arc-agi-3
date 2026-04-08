from __future__ import annotations

from v4_5.contracts.boardPerceptionReport import BoardPerceptionReport
from v4_5.contracts.boardState import BoardState


class BoardPerceptionFusion:
    module_name = "BoardPerceptionFusion"

    def fuse(
        self,
        *,
        board_state: BoardState,
        round_id: str,
        learned_output=None,
    ) -> BoardPerceptionReport:
        del learned_output
        return BoardPerceptionReport(
            schema_version=board_state.schema_version,
            module_name=self.module_name,
            round_id=round_id,
            board_state=board_state,
            rationale_codes=("DETERMINISTIC_ONLY",),
            gaps=board_state.gaps,
            advisory_only=True,
        )

