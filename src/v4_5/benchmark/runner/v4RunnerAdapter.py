from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vlm_v2.frame_writer import write_frame_png
from vlm_v2.video_builder import build_episode_video
from tests.v4.live_regression._helpers import run_case
from tests.v4.live_regression.catalog import LIVE_REGRESSION_CATALOG, LiveRegressionCase
from tests.v4.live_regression._helpers import map_fail_reason


def _catalog_by_game_id() -> dict[str, LiveRegressionCase]:
    cases = {case.game: case for case in LIVE_REGRESSION_CATALOG}
    tutorial_cases = (
        LiveRegressionCase(game="ez01", family="ez01", level=0, seed=0, track="movement", max_steps=25),
        LiveRegressionCase(game="ez02", family="ez02", level=0, seed=0, track="movement", max_steps=25),
        LiveRegressionCase(game="ez03", family="ez03", level=0, seed=0, track="movement", max_steps=25),
        LiveRegressionCase(game="ez04", family="ez04", level=0, seed=0, track="movement", max_steps=25),
        LiveRegressionCase(game="tt01", family="tt01", level=0, seed=0, track="movement", max_steps=12),
    )
    for case in tutorial_cases:
        cases.setdefault(case.game, case)
    return cases


@dataclass
class V4RunnerAdapter:
    """Legacy-only adapter retained temporarily for reference; unused by active v4.5 execution."""
    cases_by_game_id: dict[str, LiveRegressionCase] | None = None

    def __post_init__(self) -> None:
        if self.cases_by_game_id is None:
            self.cases_by_game_id = _catalog_by_game_id()

    def run_game(
        self,
        game_id: str,
        *,
        max_steps: int | None = None,
        seed: int | None = None,
        capture_video: bool = False,
        video_dir: str | None = None,
        render_terminal: bool = False,
    ) -> dict[str, Any]:
        case = self.cases_by_game_id.get(game_id)
        if case is None:
            raise KeyError(f"unknown benchmark game_id: {game_id}")
        artifacts = run_case(case, max_steps=max_steps, render_mode="terminal" if render_terminal else None)
        try:
            ledger_records = tuple(artifacts.ledger.records())
            final_observation = getattr(artifacts.session, "current_observation", None)
            initial_level = 0
            if ledger_records and getattr(ledger_records[0], "transition_record", None) is not None:
                initial_level = int(ledger_records[0].transition_record.pre_observation.levels_completed)
            elif final_observation is not None:
                initial_level = int(getattr(final_observation, "levels_completed", 0) or 0)
            fail_reason = map_fail_reason(
                case=case,
                summary=artifacts.summary,
                ledger_records=ledger_records,
                diagnostics=artifacts.diagnostics,
            )
            result = {
                "game_id": case.game,
                "family": case.family,
                "attempted": True,
                "stop_reason": str(getattr(artifacts.summary, "stop_reason", "")),
                "steps_executed": int(getattr(artifacts.summary, "steps_executed", 0)),
                "failure_reason": None if fail_reason == "win" else fail_reason,
                "levels_completed_start": initial_level,
                "levels_completed_end": int(getattr(final_observation, "levels_completed", initial_level) or initial_level),
                "win_levels": int(getattr(final_observation, "win_levels", 0) or 0),
                "step_records": [
                    {
                        "step_index": int(getattr(record, "step_index", 0) or 0),
                        "action_executed": getattr(record, "executed_action", None) is not None,
                        "pre_levels_completed": int(record.transition_record.pre_observation.levels_completed) if getattr(record, "transition_record", None) is not None else None,
                        "post_levels_completed": int(record.transition_record.post_observation.levels_completed) if getattr(record, "transition_record", None) is not None else None,
                        "levels_completed_delta": int(getattr(record.step_result, "levels_completed_delta", 0) or 0) if getattr(record, "step_result", None) is not None else 0,
                        "terminal_status": (
                            str(getattr(terminal_signal, "status"))
                            if (terminal_signal := getattr(getattr(record, "step_result", None), "terminal_signal", None)) is not None
                            else "non_terminal"
                        ),
                        "failure_bucket": getattr(record, "failure_bucket", None),
                        "action_legal": bool(getattr(getattr(record, "transition_record", None), "action_legal", True)),
                    }
                    for record in ledger_records
                    if getattr(record, "executed_action", None) is not None
                ],
            }
            if seed is not None:
                result["seed"] = int(seed)
            if capture_video and video_dir:
                result["video_path"] = self._capture_video(
                    session=artifacts.session,
                    ledger_records=ledger_records,
                    video_dir=video_dir,
                )
            return result
        finally:
            close = getattr(artifacts.session, "close", None)
            if callable(close):
                close()

    def _capture_video(self, *, session: object, ledger_records: tuple[object, ...], video_dir: str) -> str:
        frame_dir = Path(video_dir) / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_index = 0
        initial_observation = getattr(session, "initial_observation", None)
        if initial_observation is None and ledger_records:
            transition = getattr(ledger_records[0], "transition_record", None)
            if transition is not None:
                initial_observation = transition.pre_observation
        if initial_observation is not None:
            grid = _observation_grid(initial_observation)
            if grid is not None:
                write_frame_png(grid, str(frame_dir), frame_index)
                frame_index += 1
        for record in ledger_records:
            transition = getattr(record, "transition_record", None)
            if transition is None:
                continue
            grid = _observation_grid(transition.post_observation)
            if grid is None:
                continue
            write_frame_png(grid, str(frame_dir), frame_index)
            frame_index += 1
        if frame_index == 0:
            raise RuntimeError(f"no observable frames available for {video_dir}")
        return build_episode_video(str(frame_dir), fps=2, output_name="episode.mp4")


def _observation_grid(observation: object) -> Any | None:
    frames = getattr(observation, "frame", None)
    if isinstance(frames, tuple) and frames:
        return frames[-1]
    if isinstance(frames, list) and frames:
        return frames[-1]
    return None
