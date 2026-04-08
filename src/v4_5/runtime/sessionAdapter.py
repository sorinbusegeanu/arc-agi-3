from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from v4.agentContract.environmentMetadata import V4EnvironmentMetadata
from v4.agentContract.types import V4Action, V4Observation
from v4.runtime.envSession import EnvSessionV4
from v4.state.stateParser import StateParserV4
from v4_5.adapters.actionAdapter import ActionAdapter
from v4_5.adapters.controlProfileAdapter import ControlProfileAdapter
from v4_5.adapters.stateAdapter import StateAdapter
from v4_5.runtime.types import ExecutedPrefixResult, LiveObservationSnapshot


@dataclass
class LiveGameSession:
    game_id: str
    env_session: EnvSessionV4
    state_parser: StateParserV4
    state_adapter: StateAdapter
    action_adapter: ActionAdapter
    initial_observation: V4Observation
    previous_observation: V4Observation | None = None
    action_history: list[V4Action] = field(default_factory=list)

    @property
    def environment_metadata(self) -> V4EnvironmentMetadata:
        return self.env_session.environment_metadata


class SessionAdapter:
    reused_modules = ("src/v4/runtime/*", "src/v4/state/*", "src/v4/agentContract/*")

    def __init__(
        self,
        *,
        state_parser: StateParserV4 | None = None,
        state_adapter: StateAdapter | None = None,
        action_adapter: ActionAdapter | None = None,
        control_profile_adapter: ControlProfileAdapter | None = None,
    ) -> None:
        self.state_parser = state_parser or StateParserV4()
        self.action_adapter = action_adapter or ActionAdapter()
        self.state_adapter = state_adapter or StateAdapter()
        self.control_profile_adapter = control_profile_adapter or ControlProfileAdapter(action_adapter=self.action_adapter)

    def create_session(
        self,
        game_id: str,
        *,
        seed: int = 0,
        render_terminal: bool = False,
        env_root: str | None = None,
        env_factory: Callable[[], Any] | None = None,
    ) -> LiveGameSession:
        env_session = EnvSessionV4(
            env_id=game_id,
            env_root=env_root,
            seed=seed,
            render_mode="terminal" if render_terminal else None,
            env_factory=env_factory,
        )
        initial_observation = env_session.reset()
        return LiveGameSession(
            game_id=game_id,
            env_session=env_session,
            state_parser=self.state_parser,
            state_adapter=self.state_adapter,
            action_adapter=self.action_adapter,
            initial_observation=initial_observation,
        )

    def close_session(self, session: LiveGameSession) -> None:
        session.env_session.close()

    def get_current_observation(self, session: LiveGameSession) -> V4Observation:
        observation = session.env_session.current_observation
        if observation is None:
            raise ValueError("live session has no current observation")
        return observation

    def get_authoritative_observation(self, session: LiveGameSession) -> LiveObservationSnapshot:
        observation = self.get_current_observation(session)
        level_id = self._authoritative_level_id(session, observation)
        parsed_state = session.state_parser.build_parsed_state(
            current_observation=observation,
            previous_observation=session.previous_observation,
            environment_metadata=session.environment_metadata,
            local_memory_snapshot=None,
            step_index=session.env_session.step_index,
        )
        summary = session.state_adapter.summarize_observation(observation)
        control_profile = self.control_profile_adapter.build_profile(
            environment_metadata=session.environment_metadata,
            enabled_action_ids=tuple(observation.available_actions),
        )
        return LiveObservationSnapshot(
            game_id=session.game_id,
            level_id=level_id,
            step_index=session.env_session.step_index,
            observation=observation,
            parsed_state=parsed_state,
            game_control_profile=control_profile,
            levels_completed=int(observation.levels_completed),
            win_levels=int(observation.win_levels),
            terminal_status=self.detect_terminal_status(session),
            summary=summary,
        )

    def get_available_action_set(self, session: LiveGameSession) -> tuple[str, ...]:
        observation = self.get_current_observation(session)
        return self.action_adapter.available_primitive_actions(tuple(observation.available_actions))

    def execute_primitive_action(self, session: LiveGameSession, action: V4Action):
        pre = self.get_current_observation(session)
        transition, step_result = session.env_session.step(action)
        session.previous_observation = pre
        session.action_history.append(action)
        return transition, step_result

    def execute_action_prefix(self, session: LiveGameSession, prefix_actions: tuple[V4Action, ...], prefix_tokens: tuple[str, ...] | None = None) -> ExecutedPrefixResult:
        pre_observation = self.get_current_observation(session)
        transitions = []
        step_results = []
        for action in prefix_actions:
            transition, step_result = self.execute_primitive_action(session, action)
            transitions.append(transition)
            step_results.append(step_result)
            if step_result.terminal_signal.is_terminal or step_result.levels_completed_delta:
                break
        post_observation = self.get_current_observation(session)
        observed_effects: list[str] = []
        if transitions:
            observed_effects.append("state_change")
        if any(not item.action_legal for item in step_results):
            observed_effects.append("blocked")
        if post_observation.levels_completed > pre_observation.levels_completed:
            observed_effects.extend(("level_transition", "effect"))
        if prefix_actions and prefix_actions[0].action_id == 6:
            observed_effects.append("contact")
        terminal_status = self.detect_terminal_status(session)
        if terminal_status == "success":
            observed_effects.extend(("success", "effect"))
        elif terminal_status == "failure":
            observed_effects.append("failure")
        return ExecutedPrefixResult(
            prefix=tuple(prefix_tokens or tuple(action.action_name for action in prefix_actions)),
            primitive_actions=tuple(prefix_actions),
            transitions=tuple(transitions),
            step_results=tuple(step_results),
            steps_executed=len(step_results),
            pre_levels_completed=int(pre_observation.levels_completed),
            post_levels_completed=int(post_observation.levels_completed),
            levels_completed_delta=int(post_observation.levels_completed - pre_observation.levels_completed),
            terminal_status=terminal_status,
            terminal_success=terminal_status == "success",
            terminal_failure=terminal_status == "failure",
            level_transition=bool(post_observation.levels_completed > pre_observation.levels_completed),
            game_completion=bool(post_observation.levels_completed >= post_observation.win_levels and post_observation.win_levels > 0),
            observed_effects=tuple(dict.fromkeys(observed_effects)),
            failure_reason="terminal_failure" if terminal_status == "failure" else None,
        )

    def detect_terminal_status(self, session: LiveGameSession) -> str:
        observation = self.get_current_observation(session)
        lowered = str(observation.state).lower()
        if any(token in lowered for token in ("win", "success", "complete")):
            return "success"
        if any(token in lowered for token in ("lose", "fail", "dead", "game_over")):
            return "failure"
        return "non_terminal"

    def detect_level_transition(self, session: LiveGameSession, executed_prefix_result: ExecutedPrefixResult | None = None) -> bool:
        if executed_prefix_result is not None:
            return bool(executed_prefix_result.level_transition)
        observation = self.get_current_observation(session)
        previous = session.previous_observation
        return bool(previous is not None and observation.levels_completed > previous.levels_completed)

    def detect_levels_completed(self, session: LiveGameSession) -> int:
        return int(self.get_current_observation(session).levels_completed)

    def expose_action_history(self, session: LiveGameSession) -> tuple[V4Action, ...]:
        return tuple(session.action_history)

    def _authoritative_level_id(self, session: LiveGameSession, observation: V4Observation) -> str:
        raw_level_id = observation.raw_payload.get("level_id")
        if isinstance(raw_level_id, str) and raw_level_id:
            return raw_level_id
        env_level_id = getattr(session.env_session.env, "level_id", None) or getattr(session.env_session.env, "current_level_id", None)
        if isinstance(env_level_id, str) and env_level_id:
            return env_level_id
        game = getattr(session.env_session.env, "_game", None)
        game_level_index = getattr(game, "level_index", None)
        if isinstance(game_level_index, int):
            return f"L{game_level_index}"
        current_level_index = getattr(game, "_current_level_index", None)
        if isinstance(current_level_index, int):
            return f"L{current_level_index}"
        raise ValueError("authoritative level_id missing from runtime session")
