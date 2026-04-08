from __future__ import annotations

from dataclasses import asdict
from typing import Any

from v4_5.runtime.types import LiveGameResult, LiveLevelSummary, LiveStepRecord


class ResultBuilder:
    def build_raw_result(
        self,
        *,
        game_id: str,
        attempted: bool,
        stop_reason: str,
        steps_executed: int,
        failure_reason: str | None,
        levels_completed_start: int,
        levels_completed_end: int,
        win_levels: int,
        step_records: tuple[LiveStepRecord, ...],
        level_summaries: tuple[LiveLevelSummary, ...] = (),
        video_path: str | None = None,
    ) -> dict[str, Any]:
        result = LiveGameResult(
            game_id=game_id,
            attempted=attempted,
            stop_reason=stop_reason,
            steps_executed=steps_executed,
            failure_reason=failure_reason,
            levels_completed_start=levels_completed_start,
            levels_completed_end=levels_completed_end,
            win_levels=win_levels,
            step_records=step_records,
            level_summaries=level_summaries,
            video_path=video_path,
        )
        payload = result.to_dict()
        payload["level_summaries"] = [asdict(item) for item in level_summaries]
        return payload
