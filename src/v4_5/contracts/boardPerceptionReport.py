from __future__ import annotations

from dataclasses import dataclass

from v4_5.contracts.boardState import BoardState


@dataclass(frozen=True)
class BoardPerceptionReport:
    schema_version: str
    module_name: str
    round_id: str
    board_state: BoardState
    rationale_codes: tuple[str, ...] = ()
    gaps: tuple[str, ...] = ()
    advisory_only: bool = True

