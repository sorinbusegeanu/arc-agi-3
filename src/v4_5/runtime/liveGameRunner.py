from __future__ import annotations
from pathlib import Path
from typing import Any, Callable

from v4_5.adapters.actionAdapter import ActionAdapter, ActionTranslationContext
from v4_5.contracts.bootstrapMediaTypes import BootstrapStepRecord
from v4_5.logging import BoundAgentLogger
from v4_5.orchestrator.context import OrchestratorContext
from v4_5.orchestrator.controller import V45Controller
from v4_5.runtime.resultBuilder import ResultBuilder
from v4_5.runtime.sessionAdapter import SessionAdapter
from v4_5.runtime.stopEvaluator import StopEvaluator
from v4_5.runtime.types import ExecutedPrefixResult, LiveLevelSummary, LiveStepRecord


class V45GameRunner:
    def __init__(
        self,
        *,
        controller: V45Controller,
        session_adapter: SessionAdapter,
        action_adapter: ActionAdapter | None = None,
        stop_evaluator: StopEvaluator | None = None,
        result_builder: ResultBuilder | None = None,
        logger: BoundAgentLogger | None = None,
        max_actions_per_level: int | None = None,
    ) -> None:
        self.controller = controller
        self.session_adapter = session_adapter
        self.action_adapter = action_adapter or ActionAdapter()
        self.stop_evaluator = stop_evaluator or StopEvaluator()
        self.result_builder = result_builder or ResultBuilder()
        self.logger = logger
        self.max_actions_per_level = max_actions_per_level

    def run_game(
        self,
        game_id: str,
        *,
        max_steps: int | None = None,
        max_levels: int | None = None,
        seed: int | None = None,
        capture_video: bool = False,
        video_dir: str | None = None,
        render_terminal: bool = False,
        env_factory: Callable[[], Any] | None = None,
    ) -> dict[str, Any]:
        session = self.session_adapter.create_session(
            game_id,
            seed=0 if seed is None else int(seed),
            render_terminal=render_terminal,
            env_factory=env_factory,
        )
        step_records: list[LiveStepRecord] = []
        level_summaries: list[LiveLevelSummary] = []
        last_executed: ExecutedPrefixResult | None = None
        no_decision_rounds = 0
        stop_reason = "continue"
        failure_reason = None
        start_snapshot = self.session_adapter.get_authoritative_observation(session)
        current_snapshot = start_snapshot
        current_level_start = start_snapshot.levels_completed
        current_level_step_start = 0
        context = OrchestratorContext(
            env_id=game_id,
            level_id=start_snapshot.level_id,
            round_id=f"{game_id}:r0000",
            observation=start_snapshot.observation,
            unseen_level=True,
            force_bootstrap=True,
        )
        try:
            if self.logger is not None:
                self.logger.info(game_id, "starting game run")
            round_index = 0
            while True:
                context.round_id = f"{game_id}:r{round_index:04d}"
                if self.logger is not None and context.unseen_level:
                    self.logger.info(game_id, "starting level", round_id=context.round_id, level_index=context.level_id)
                context.level_id = current_snapshot.level_id
                context.observation = current_snapshot.observation
                context.live_snapshot = current_snapshot
                context.execution_committed = False
                context.last_committed_prefix = ()
                context.last_executed_prefix_result = last_executed
                context.unseen_level = round_index == 0 or bool(last_executed and last_executed.level_transition)
                context.force_bootstrap = context.unseen_level
                context.reports["bootstrap_capture_executor"] = lambda sequence_name, actions, step_offset: self._execute_bootstrap_sequence(
                    session=session,
                    sequence_name=sequence_name,
                    actions=actions,
                    step_offset=step_offset,
                    step_records=step_records,
                    round_id=context.round_id,
                )
                context.reports["bootstrap_avatar_detector"] = lambda: self._detect_authoritative_avatar(session)
                context = self.controller.run_cycle(context)
                current_snapshot = self.session_adapter.get_authoritative_observation(session)
                discovery = context.reports.get("discovery")
                if getattr(discovery, "stop_reason", None) == "no_artifacts_identified":
                    no_decision_rounds += 1
                    step_records.append(
                        LiveStepRecord(
                            round_id=context.round_id,
                            step_index=len(step_records),
                            selected_prefix=(),
                            executed_prefix=(),
                            action_executed=False,
                            action_count=0,
                            pre_levels_completed=current_snapshot.levels_completed,
                            post_levels_completed=current_snapshot.levels_completed,
                            levels_completed_delta=0,
                            terminal_status=current_snapshot.terminal_status,
                            stop_reason="no_artifacts_identified",
                            failure_reason="no_artifacts_identified",
                            action_legal=True,
                            observed_effects=(),
                        )
                    )
                    failure_reason = "no_artifacts_identified"
                    stop_reason = "no_artifacts_identified"
                    self.controller.orchestrator_agent.stop_run(context, status="no_artifacts_identified")
                    break
                selected_prefix = tuple(context.last_committed_prefix)
                executed_result = None
                if selected_prefix:
                    if self.logger is not None:
                        self.logger.info(game_id, "executing selected prefix", round_id=context.round_id, level_index=context.level_id)
                    translated = self.action_adapter.translate_prefix(
                        selected_prefix,
                        ActionTranslationContext(
                            available_action_ids=current_snapshot.observation.available_actions,
                            coordinate_action_id=session.environment_metadata.coordinate_action_id,
                            coordinate_bounds=session.environment_metadata.coordinate_bounds,
                        ),
                    )
                    executed_result = self.session_adapter.execute_action_prefix(session, translated, selected_prefix)
                    current_snapshot = self.session_adapter.get_authoritative_observation(session)
                    no_decision_rounds = 0
                    action_legal = all(item.action_legal for item in executed_result.step_results)
                    step_records.append(
                        LiveStepRecord(
                            round_id=context.round_id,
                            step_index=len(step_records),
                            selected_prefix=selected_prefix,
                            executed_prefix=executed_result.prefix,
                            action_executed=True,
                            action_count=executed_result.steps_executed,
                            pre_levels_completed=executed_result.pre_levels_completed,
                            post_levels_completed=executed_result.post_levels_completed,
                            levels_completed_delta=executed_result.levels_completed_delta,
                            terminal_status=executed_result.terminal_status,
                            failure_reason=executed_result.failure_reason,
                            action_legal=action_legal,
                            observed_effects=executed_result.observed_effects,
                        )
                    )
                    if executed_result.level_transition:
                        if self.logger is not None:
                            self.logger.info(game_id, "finishing level", round_id=context.round_id, level_index=context.level_id)
                            self.logger.info(game_id, "promoting level memory to validated", round_id=context.round_id, level_index=context.level_id)
                        self.controller.discovery_agent.level_memory_service.promote_level_memory_to_validated(game_id, context.level_id)
                        level_summaries.append(
                            LiveLevelSummary(
                                level_index=current_level_start,
                                started_step_index=current_level_step_start,
                                ended_step_index=len(step_records) - 1,
                                completed=True,
                                terminal_status=executed_result.terminal_status,
                            )
                        )
                        current_level_start = current_snapshot.levels_completed
                        current_level_step_start = len(step_records)
                else:
                    no_decision_rounds += 1
                    step_records.append(
                        LiveStepRecord(
                            round_id=context.round_id,
                            step_index=len(step_records),
                            selected_prefix=selected_prefix,
                            executed_prefix=(),
                            action_executed=False,
                            action_count=0,
                            pre_levels_completed=current_snapshot.levels_completed,
                            post_levels_completed=current_snapshot.levels_completed,
                            levels_completed_delta=0,
                            terminal_status=current_snapshot.terminal_status,
                            action_legal=True,
                            observed_effects=(),
                        )
                    )
                last_executed = executed_result
                stop = self.stop_evaluator.evaluate(
                    snapshot=current_snapshot,
                    executed_prefix_result=executed_result,
                    steps_executed=sum(record.action_count for record in step_records if record.action_executed),
                    max_steps=max_steps,
                    level_steps_executed=sum(
                        record.action_count for record in step_records[current_level_step_start:] if record.action_executed
                    ),
                    max_actions_per_level=self.max_actions_per_level,
                    max_levels=max_levels,
                    no_decision_rounds=no_decision_rounds,
                )
                if stop.should_stop:
                    stop_reason = stop.stop_reason
                    if stop_reason == "terminal_win":
                        if self.logger is not None:
                            self.logger.info(game_id, "promoting level memory to validated", round_id=context.round_id, level_index=context.level_id)
                        self.controller.discovery_agent.level_memory_service.promote_level_memory_to_validated(game_id, context.level_id)
                    if stop.terminal_failure:
                        failure_reason = "terminal_failure"
                    elif stop.no_decision_dead_end:
                        failure_reason = "no_executable_prefix"
                    self.controller.orchestrator_agent.stop_run(context, status=stop_reason)
                    break
                round_index += 1
            if current_level_step_start < len(step_records):
                level_summaries.append(
                    LiveLevelSummary(
                        level_index=current_level_start,
                        started_step_index=current_level_step_start,
                        ended_step_index=len(step_records) - 1,
                        completed=bool(stop_reason == "terminal_win"),
                        terminal_status=current_snapshot.terminal_status,
                    )
                )
            video_path = None
            if capture_video and video_dir:
                video_path = str((Path(video_dir) / "episode.mp4").resolve())
            if self.logger is not None:
                self.logger.info(game_id, "finishing game")
            return self.result_builder.build_raw_result(
                game_id=game_id,
                attempted=True,
                stop_reason=stop_reason,
                steps_executed=sum(record.action_count for record in step_records if record.action_executed),
                failure_reason=failure_reason,
                levels_completed_start=start_snapshot.levels_completed,
                levels_completed_end=current_snapshot.levels_completed,
                win_levels=current_snapshot.win_levels,
                step_records=tuple(step_records),
                level_summaries=tuple(level_summaries),
                video_path=video_path,
            )
        finally:
            self.session_adapter.close_session(session)

    def _execute_bootstrap_sequence(
        self,
        *,
        session,
        sequence_name: str,
        actions: tuple[str, ...],
        step_offset: int,
        step_records: list[LiveStepRecord],
        round_id: str,
    ) -> tuple[BootstrapStepRecord, ...]:
        bootstrap_records: list[BootstrapStepRecord] = []
        for action in actions:
            current_snapshot = self.session_adapter.get_authoritative_observation(session)
            try:
                translated = self.action_adapter.translate_token(
                    action,
                    ActionTranslationContext(
                        available_action_ids=current_snapshot.observation.available_actions,
                        coordinate_action_id=session.environment_metadata.coordinate_action_id,
                        coordinate_bounds=session.environment_metadata.coordinate_bounds,
                    ),
                )
            except ValueError:
                bootstrap_records.append(
                    BootstrapStepRecord(
                        schema_version="v4.5",
                        action=action,
                        status="invalid",
                        invalid_action=True,
                        blocked_action=False,
                        step_index=step_offset + len(bootstrap_records),
                        raw_observation_ref=None,
                        sequence_name=sequence_name,
                    )
                )
                break
            pre_observation = self.session_adapter.get_current_observation(session)
            executed = self.session_adapter.execute_action_prefix(session, (translated,), (action,))
            post_observation = self.session_adapter.get_current_observation(session)
            blocked = any(not item.action_legal for item in executed.step_results)
            bootstrap_records.append(
                BootstrapStepRecord(
                    schema_version="v4.5",
                    action=action,
                    status="blocked" if blocked else "executed",
                    invalid_action=False,
                    blocked_action=blocked,
                    step_index=step_offset + len(bootstrap_records),
                    raw_observation_ref=self.session_adapter.state_adapter.extract_frame_plane(post_observation.frame),
                    sequence_name=sequence_name,
                    pre_observation_ref=self.session_adapter.state_adapter.extract_frame_plane(pre_observation.frame),
                    post_observation_ref=self.session_adapter.state_adapter.extract_frame_plane(post_observation.frame),
                )
            )
            step_records.append(
                LiveStepRecord(
                    round_id=f"{round_id}:{sequence_name}",
                    step_index=len(step_records),
                    selected_prefix=(action,),
                    executed_prefix=(action,),
                    action_executed=True,
                    action_count=executed.steps_executed,
                    pre_levels_completed=executed.pre_levels_completed,
                    post_levels_completed=executed.post_levels_completed,
                    levels_completed_delta=executed.levels_completed_delta,
                    terminal_status=executed.terminal_status,
                    failure_reason=executed.failure_reason,
                    action_legal=all(item.action_legal for item in executed.step_results),
                    observed_effects=executed.observed_effects,
                )
            )
            if executed.terminal_status in {"success", "failure"} or executed.level_transition:
                break
        return tuple(bootstrap_records)

    def _detect_authoritative_avatar(self, session) -> object | None:
        game = getattr(session.env_session.env, "_game", None)
        player = getattr(game, "_player", None)
        x = getattr(player, "_x", None)
        y = getattr(player, "_y", None)
        pixels = getattr(player, "pixels", None)
        if not isinstance(x, int) or not isinstance(y, int):
            return None
        width = 1
        height = 1
        shape = getattr(pixels, "shape", None)
        if isinstance(shape, tuple) and len(shape) >= 2:
            height = max(1, int(shape[0]))
            width = max(1, int(shape[1]))
        bbox = (x, y, x + width - 1, y + height - 1)
        center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        return type(
            "AuthoritativeAvatarDetection",
            (),
            {
                "avatar_bbox": bbox,
                "avatar_position": center,
                "support_actions": ("authoritative_runtime",),
                "used_fallback": False,
            },
        )()
