from __future__ import annotations

import json
import copy
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import v11.live as live_module
from v11.arcade_env import ArcadeGameEnv
from v11.live import (
    _active_level_transitions, _live_context, _parse_content, _perform_game_over_post_mortem,
    _perform_review,
    _perceptual_state_equivalence, _review_events,
    _verified_route_summaries, run_live,
)
from v11.analysis import DecisionValidationError
from v11.tools.game import GameTools
from v11.tools.game import ActionError
from v11.tools.history import History
from v11.tools.memory import (
    GameMemory, game_memory_dir, profile_memory_path, save_game_memory, shared_skill_path,
)


def test_perceptual_equivalence_is_sparse_exploration_signal_only() -> None:
    base = np.zeros((64, 64), dtype=np.uint8)
    flicker = base.copy()
    flicker[0, 0] = 1
    moved_object = base.copy()
    moved_object[10:14, 10:14] = 2

    signal = _perceptual_state_equivalence(base, flicker)
    assert signal["equivalent"] is True
    assert signal["certainty"] == "low"
    assert "hidden state is unknown" in signal["reason"]
    assert _perceptual_state_equivalence(base, moved_object)["equivalent"] is False
    assert _perceptual_state_equivalence(np.zeros((8, 8)), np.zeros((8, 8)))["certainty"] == "unreliable"




def test_action_decision_parser_accepts_one_fenced_json_object_with_prose() -> None:
    payload = {"actions": [{"action": "up", "args": {}}]}
    content = "The next move should be cautious.\n```json\n" + json.dumps(payload) + "\n```\nProceeding now."
    assert _parse_content(content) == payload
    with pytest.raises(DecisionValidationError):
        _parse_content("```json\nnot json\n```")
    with pytest.raises(DecisionValidationError):
        _parse_content("```json\n{}\n```\n```json\n{}\n```")


def test_perceptual_state_similarity_is_logged_and_never_resets_game(tmp_path: Path) -> None:
    class HiddenStateEnv(_FakeEnv):
        def __init__(self):
            super().__init__(change_every_action=False)
            self.frame = np.zeros((64, 64), dtype=np.int64)
            self.up_count = 0

        def act(self, action, args):
            self.actions.append((action, dict(args)))
            if action == "ACTION1":
                self.up_count += 1
                self.frame[0, :self.up_count] = self.up_count
            elif action == "ACTION2":
                self.frame[:] = 0
            return {}

    env = HiddenStateEnv()
    client = _FakeClient([_decision([_action("up"), _action("down"), _action("up")])])
    run_dir = tmp_path / "perceptual-signal"
    metrics = _run_live_without_starting_routine(
        env, run_dir, client, game_id="fixture", max_actions=3, plan_horizon=3,
    )

    rows = [json.loads(line) for line in (run_dir / "state_equivalence.jsonl").read_text().splitlines()]
    assert any(row["status"] == "candidate" for row in rows)
    false_rows = [row for row in rows if row["status"] == "validated_false"]
    assert false_rows and "different exact target states" in false_rows[0]["reason"]
    assert all(row["safety_effect"].startswith("none;") for row in rows)
    assert metrics["actions"] == 3
    assert env.resets == 0


@pytest.fixture(autouse=True)
def _isolated_memory_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("V11_STATE_DIR", str(tmp_path / "state"))


def _run_live_without_starting_routine(*args, **kwargs):
    """Keep legacy execution tests focused; startup behavior has dedicated tests below."""
    kwargs["starting_routine"] = False
    return run_live(*args, **kwargs)


def _decision(actions=None, *, reset: bool = False) -> dict:
    if reset:
        return {"actions": [{"action": "RESET", "args": {"justification": "A controlled restart is required."}}]}
    return {"actions": list(actions or [])}


def _action(action="up"):
    return {"action": action, "args": {}}


def _reasoned(decision, *, hypothesis=None, prediction=None):
    return {
        "state_assessment": {
            "player": {"location": "center", "certainty": "inferred", "evidence": "visible in CURRENT SCREEN"},
            "target_candidates": [], "obstacles": [], "other_relevant_changes": [],
        },
        "hypothesis": hypothesis,
        "trajectory_rationale": "Test one legal action from the current screen.",
        "prediction": prediction or {
            "visible_change": "unknown", "environment_outcome": "unknown", "description": "unknown",
        },
        "actions": list(decision.get("actions", [])),
    }


_VALID_GAME_MEMORY = "# Game Memory\n\n- A cue is visible in this game."
_VALID_GLOBAL_MEMORY = "# Global Memory\n\n- Compare each object before and after an action."
_VALID_REVIEW = json.dumps({
    "game_memory_markdown": _VALID_GAME_MEMORY,
    "profile_memory_candidates": [],
})


class _FakeClient:
    model = "mock-qwen"

    def __init__(
        self, decisions, *, request_attempts=1, review_responses=None,
        post_mortem_responses=None, review_errors=None, reasoning_responses=None,
        default_decision=None,
    ):
        self.decisions = list(decisions)
        self.calls = []
        self.system_prompts = []
        self.response_formats = []
        self.tool_definitions = []
        self.request_attempts = request_attempts
        self.review_responses = list(review_responses or [])
        self.post_mortem_responses = list(post_mortem_responses or [])
        self.review_errors = list(review_errors or [])
        self.reasoning_responses = list(reasoning_responses or [])
        self.default_decision = default_decision
        self.message_calls = []
        self.response_contents = []

    def complete(
        self, system_prompt, user_prompt, image_paths, image_labels,
        response_format="json_object", *, messages=None, conversation_reset_count=None, tools=None,
    ):
        if messages is not None:
            self.message_calls.append({
                "messages": copy.deepcopy(messages),
                "conversation_reset_count": conversation_reset_count,
            })
            last_user = next((message for message in reversed(messages) if message.get("role") == "user"), {})
            content = last_user.get("content", "")
            if isinstance(content, list):
                user_prompt = next((
                    str(block.get("text", "")) for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ), "")
            elif isinstance(content, str):
                user_prompt = content
        self.calls.append((user_prompt, list(image_paths), list(image_labels)))
        self.system_prompts.append(system_prompt)
        self.response_formats.append(response_format)
        self.tool_definitions.append(copy.deepcopy(tools))
        if system_prompt.startswith("# V11 game memory and profile candidate review"):
            if self.review_errors:
                raise self.review_errors.pop(0)
            content = self.review_responses.pop(0) if self.review_responses else _VALID_REVIEW
        elif system_prompt.startswith("# V11 game-over post-mortem"):
            content = self.post_mortem_responses.pop(0) if self.post_mortem_responses else "Post-mortem unavailable."
        elif system_prompt.startswith("# V10 uncertain action assessment"):
            content = self.reasoning_responses.pop(0) if self.reasoning_responses else json.dumps({
                "hypothesis_being_tested": "The action may change the visible state.",
                "expected_visible_change": "A visible object or position changes.",
                "actual_observed_change": "The before and after screens differ.",
                "hypothesis_remains_viable": "yes: a visible change occurred.",
            })
        else:
            decision = self.decisions.pop(0) if self.decisions else self.default_decision
            if decision is None:
                decision = _decision([])
            if isinstance(decision, dict) and set(decision) == {"actions"}:
                decision = _reasoned(decision)
            content = json.dumps(decision)
        self.response_contents.append(content)
        return {
            "content": content, "model": self.model,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "latency_seconds": 0.01, "request_attempts": self.request_attempts,
            "finish_reason": "stop",
        }


def _message_user_text(client: _FakeClient, call_index: int) -> str:
    messages = client.message_calls[call_index]["messages"]
    user = next(message for message in reversed(messages) if message.get("role") == "user")
    content = user.get("content")
    if isinstance(content, str):
        return content
    return next(
        str(block.get("text", "")) for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _continued_context(prompt: str) -> dict:
    marker = "CURRENT SCREEN is the actual latest environment state.\n"
    return json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1])[0]


class _FakeEnv:
    def __init__(self, *, change_every_action=True):
        self.frame = np.zeros((3, 4), dtype=np.int64)
        self.actions = []
        self.resets = 0
        self.change_every_action = change_every_action

    def observe(self):
        return {
            "frame": self.frame.copy(),
            "legal_actions": [
                {"id": "ACTION1", "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}},
                {"id": "ACTION2", "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}},
            ],
            "game_state": "NOT_FINISHED", "level": 0, "levels_completed": 0,
            "win_levels": 5, "terminal": False,
        }

    def act(self, action, args):
        self.actions.append((action, dict(args)))
        if self.change_every_action:
            self.frame[0, len(self.actions) % self.frame.shape[1]] = len(self.actions)
        return {}

    def reset(self):
        self.resets += 1
        self.frame[:] = 0
        return self.observe()


class _OscillatingEnv(_FakeEnv):
    """Visibly alternates between two states to exercise cycle detection."""

    def __init__(self):
        super().__init__(change_every_action=False)
        self.toggle = False

    def act(self, action, args):
        self.actions.append((action, dict(args)))
        self.toggle = not self.toggle
        self.frame[:] = int(self.toggle)
        return {}


def test_applicable_successful_strategy_is_in_first_action_prompt(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    saved = "# Game Memory\n\n## Successful strategies\n- On level zero, repeat up until progress changes."
    store_dir = game_memory_dir(state_dir, "fixture")
    legacy = GameMemory(tmp_path / "seed", store_dir=store_dir, run_id="seed", game_id="fixture")
    legacy.apply_review_changes([{
        "operation": "upsert", "pattern_type": "strategy", "status": "confirmed",
        "claim": "LEDGER_ONLY_STRATEGY_MUST_NOT_BE_IN_ACTION_PROMPT",
        "goal": "Complete level zero.", "preconditions": [],
        "action_policy": "This old structured policy is superseded.", "success_conditions": [],
    }], game_id="fixture", current_step=0, covered_event_count=1)
    save_game_memory(store_dir, saved)
    tools = GameTools(_FakeEnv(), tmp_path / "prompt-run")
    current = tools.observe()
    inputs = _live_context(
        tools, current, "fixture", 1, shared_skill_path(),
        game_memory_dir(state_dir, "fixture"), "prompt-run", 0,
    )
    assert inputs["context"]["memory"] == saved + "\n"
    assert "repeat up until progress changes" in inputs["user_prompt"]
    assert "applicable_successful_strategies" not in inputs["context"]
    assert inputs["user_prompt"].count("repeat up until progress changes") == 1
    assert "LEDGER_ONLY_STRATEGY_MUST_NOT_BE_IN_ACTION_PROMPT" not in inputs["user_prompt"]


def test_live_action_prompt_keeps_reasoning_in_skill_and_dynamic_constraints_in_user_prompt(tmp_path: Path) -> None:
    tools = GameTools(_FakeEnv(), tmp_path / "prompt-ownership")
    current = tools.observe()
    inputs = _live_context(
        tools, current, "fixture", 3, shared_skill_path(), tmp_path / "state/fixture", "run", 0,
    )

    skill = inputs["system_prompt"]
    prompt = inputs["user_prompt"]
    for section in ("Observe", "Hypothesize", "Evaluate", "Experiment", "Act"):
        assert f"**{section}:**" in skill
    assert "completing the level" in skill and "understanding it" in skill
    assert "When starting an unfamiliar game or level, scan the current screen" in skill
    assert "test one additional instance to see whether the effect generalizes" in skill
    assert "do not run a blind checklist or infer a mechanic from appearance alone" in skill
    assert "Level index 0 (0 completed)" in prompt
    assert "final CURRENT SCREEN as authoritative" in prompt
    assert "up to 3 legal actions" in prompt
    assert "current_state.legal_actions" in prompt
    assert "RESET restarts the whole game" in prompt
    assert "Required response contract (`v10.reasoned_action/1`):" in prompt
    from v11.analysis import reasoned_action_contract_description
    contract_text = prompt.split("Required response contract (`v10.reasoned_action/1`):\n", 1)[1].split("\nString limits:", 1)[0]
    assert json.loads(contract_text) == reasoned_action_contract_description()
    assert "player location 120/evidence 180" in prompt
    assert "Prefer a demonstrated strategy when its stated conditions match" not in prompt
    assert "Avoid repeating an action that had no visible effect" not in prompt


def test_review_consolidates_scoped_hypotheses_and_feasible_experiment(tmp_path: Path) -> None:
    env = _FakeEnv()
    run_dir = tmp_path / "review-investigation"
    tools = GameTools(env, run_dir)
    tools.observe()
    tools.act("ACTION1")
    game_memory = (
        "## Established mechanics\n- On the first level, the player moves one visible region per UP input.\n\n"
        "## Failed hypotheses\n- On the first level, entering the green region from above does not produce the expected completion.\n\n"
        "## Unresolved hypotheses\n- On the first level, completion may depend on an adjacent trigger rather than overlap.\n\n"
        "## Next experiment\n- On the first level, test one legal approach from the side to distinguish overlap from adjacency; purpose: identify the completion condition."
    )
    global_memory = "## Exploration method\n- Compare the same object's position before and after one action."
    client = _FakeClient([], review_responses=[json.dumps({
        "game_memory_markdown": game_memory,
        "global_memory_markdown": global_memory,
    })])
    store_dir = game_memory_dir(tmp_path / "state", "fixture")

    result = _perform_review(
        tools, client, game_id="fixture", run_id="investigation",
        memory_store_dir=store_dir, trigger_reasons=["stagnation"],
    )

    assert result["row"]["status"] == "saved"
    system_prompt = client.system_prompts[0]
    review_prompt = client.calls[0][0]
    for heading in ("Established mechanics", "Failed hypotheses", "Unresolved hypotheses", "Next experiment"):
        assert heading in system_prompt
    for instruction in (
        "most important unresolved question", "competing explanations",
        "best distinguishes those explanations", "purpose in game-specific memory",
    ):
        assert instruction in review_prompt
    assert "remains valid after a run restarts" in system_prompt
    assert "Do not create a “Game status,” “Current state,” or “Level progress” section" in system_prompt
    assert "Merge repetition" in system_prompt
    assert "do not prove a wall" in system_prompt
    assert "Legal actions available from the current active screen" in review_prompt
    assert '"id":"up"' in review_prompt
    assert "exactly `game_memory_markdown` and `profile_memory_candidates`" in system_prompt
    from v11.live import _validate_review_payload
    assert _validate_review_payload({
        "game_memory_markdown": game_memory,
        "global_memory_markdown": global_memory,
    }) == (game_memory, [])
    with pytest.raises(ValueError, match="exactly game_memory_markdown and profile_memory_candidates"):
        _validate_review_payload({
            "game_memory_markdown": game_memory,
            "global_memory_markdown": global_memory,
            "extra": "unexpected",
        })

    saved = (store_dir / "MEMORY.md").read_text(encoding="utf-8")
    assert saved == game_memory + "\n"
    assert not (tmp_path / "state/global/MEMORY.md").exists()
    action_context = _live_context(
        tools, tools.observe(), "fixture", 2, shared_skill_path(), store_dir, "next-run", 0,
    )
    assert action_context["context"]["memory"] == saved
    assert "test one legal approach from the side" in action_context["user_prompt"]


def test_action_prompt_scopes_prior_level_mechanics_to_their_conditions(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    store_dir = game_memory_dir(state_dir, "fixture")
    save_game_memory(
        store_dir,
        "# Game Memory\n\n- Gray is the goal on levels zero through two.",
    )
    tools = GameTools(_FakeEnv(), tmp_path / "scope-prompt")
    current = tools.observe()
    current["level"] = 3
    current["levels_completed"] = 3

    inputs = _live_context(
        tools, current, "fixture", 4, shared_skill_path(), store_dir, "run", 0,
    )

    assert "Level index 3 (3 completed)" in inputs["user_prompt"]
    assert "Memory applies only to its stated level/conditions" in inputs["user_prompt"]
    assert "Gray is the goal on levels zero through two" in inputs["user_prompt"]


def test_normal_and_bootstrap_prompts_keep_only_applicable_reset_constraints(tmp_path: Path) -> None:
    tools = GameTools(_FakeEnv(), tmp_path / "reset-question")
    current = tools.observe()
    normal = _live_context(
        tools, current, "fixture", 2, shared_skill_path(), tmp_path / "state/fixture", "run", 0,
    )
    assert "RESET restarts the whole game" in normal["user_prompt"]
    assert "return it alone with the required justification" in normal["user_prompt"]
    assert '"actions"' in normal["user_prompt"]

    bootstrap = _live_context(
        tools, current, "fixture", 2, shared_skill_path(), tmp_path / "state/fixture", "run", 0,
        starting_routine=True,
    )
    assert "do not RESET" in bootstrap["user_prompt"]
    assert "RESET\",\"parameters" not in bootstrap["user_prompt"]




def test_review_memory_accepts_fixed_level_counts_but_rejects_transient_counts() -> None:
    from v11.live import _review_memory_quality_errors

    assert not _review_memory_quality_errors("## Mechanics\n- Each level has four fixed markers.")
    assert "game_memory_markdown contains a transient item count" in _review_memory_quality_errors(
        "## Hypothesis\n- Four markers remain."
    )
    assert "game_memory_markdown infers a general obstacle from a state-specific no-op" in _review_memory_quality_errors(
        "## Mechanics\n- A rightward action produced zero pixel change, indicating a wall blocks movement.",
    )


def _seed_game_mechanic(state_dir: Path, game_id: str = "fixture") -> None:
    save_game_memory(
        game_memory_dir(state_dir, game_id),
        "# Game Memory\n\n- A prior observed mechanic.",
    )


def test_no_memory_uses_up_left_down_right_bootstrap_without_extra_call_when_budget_ends(tmp_path: Path) -> None:
    class FourDirections(_FakeEnv):
        def observe(self):
            result = super().observe()
            result["legal_actions"] += [
                {"id": action, "parameters": {
                    "type": "object", "properties": {}, "required": [], "additionalProperties": False,
                }}
                for action in ("ACTION3", "ACTION4")
            ]
            return result

    env = FourDirections()
    client = _FakeClient([])
    metrics = run_live(env, tmp_path / "run", client, game_id="fixture", max_actions=4)

    assert [action for action, _ in env.actions] == ["ACTION1", "ACTION3", "ACTION2", "ACTION4"]
    assert metrics["model_calls"] == 0
    assert env.resets == 1
    assert metrics["resets"] == 1
    rows = [json.loads(line) for line in (tmp_path / "run/decisions.jsonl").read_text().splitlines()]
    plan = next(row for row in rows if row.get("schema") == "v10.live_decision/2")
    assert plan["source"] == "bootstrap_probe"
    assert [item["action"] for item in plan["decision"]["actions"]] == ["up", "left", "down", "right"]


def test_bootstrap_is_reset_before_single_startup_plan_from_first_screen(tmp_path: Path) -> None:
    class FourDirections(_FakeEnv):
        def observe(self):
            result = super().observe()
            result["legal_actions"].extend([
                {"id": action, "parameters": {
                    "type": "object", "properties": {}, "required": [], "additionalProperties": False,
                }}
                for action in ("ACTION3", "ACTION4")
            ])
            return result

    env = FourDirections()
    client = _FakeClient([_decision([_action("left")])])
    metrics = run_live(env, tmp_path / "run", client, game_id="fixture", max_actions=5)

    assert env.resets == 1
    assert [action for action, _ in env.actions] == ["ACTION1", "ACTION3", "ACTION2", "ACTION4", "ACTION3"]
    assert metrics["model_calls"] == 1
    prompt, images, labels = client.calls[0]
    assert "bootstrap observations" in prompt
    assert "game was then reset" in prompt
    assert "must start from this current screen" in prompt
    assert "short exploratory action sequence" in prompt
    # The bootstrap probe and restored start form one ordered experiment:
    # include its initial screen, all four probe outcomes, and restored start.
    assert len(images) >= 5
    assert len(labels) == len(images)
    assert "CURRENT SCREEN" in labels[-1]
    action_screen_positions = [
        next(index for index, label in enumerate(labels) if f"Action {number} (" in label)
        for number in range(1, 5)
    ]
    assert action_screen_positions == sorted(action_screen_positions)
    assert "initial state before Action 1" in labels[0]
    assert "restored initial screen" in labels[-1]
    assert '"bootstrap_observations"' in prompt
    assert "1 up:" in prompt and "2 left:" in prompt and "3 down:" in prompt and "4 right:" in prompt
    decisions = [
        json.loads(line) for line in (tmp_path / "run/decisions.jsonl").read_text().splitlines()
    ]
    reset = next(row for row in decisions if row.get("schema") == "v10.live_bootstrap_reset/1")
    assert reset["status"] == "reset"
    plans = [row for row in decisions if row.get("schema") == "v10.live_decision/2"]
    assert plans[0]["source"] == "bootstrap_probe"
    assert plans[-1]["source"] == "bootstrap_model_start"


def test_each_new_level_bootstraps_then_resets_replays_prior_route_and_calls_model(
    tmp_path: Path,
) -> None:
    class TwoLevelEnv:
        def __init__(self):
            self.frame = np.zeros((3, 4), dtype=np.int64)
            self.level = 0
            self.levels_completed = 0
            self.resets = 0
            self.actions = []
            self.terminal = False

        def observe(self):
            return {
                "frame": self.frame.copy(),
                "legal_actions": [{"id": f"ACTION{i}", "parameters": {
                    "type": "object", "properties": {}, "required": [],
                    "additionalProperties": False,
                }} for i in range(1, 5)],
                "game_state": "WIN" if self.terminal else "NOT_FINISHED",
                "level": self.level, "levels_completed": self.levels_completed,
                "win_levels": 2, "terminal": self.terminal,
            }

        def act(self, action, args):
            self.actions.append((action, self.level))
            self.frame[0, 1] += 1
            if self.level == 0 and action == "ACTION1" and self.resets >= 1:
                self.level = 1
                self.levels_completed = 1
                self.frame[:] = 10
            elif self.level == 1 and action == "ACTION2" and self.resets >= 2:
                self.levels_completed = 2
                self.terminal = True
            return {}

        def reset(self):
            self.resets += 1
            self.frame[:] = 0
            self.level = 0
            self.levels_completed = 0
            self.terminal = False
            return self.observe()

    env = TwoLevelEnv()
    client = _FakeClient([
        _decision([_action("up")]),
        _decision([_action("down")]),
    ])
    metrics = run_live(
        env, tmp_path / "per-level-bootstrap", client, game_id="fixture",
        max_actions=11,
    )

    assert env.resets == 2
    assert metrics["resets"] == 2
    assert metrics["won"] is True
    assert metrics["available_level_solutions"] == 2
    # Initial probe, solve level 0, level-1 probe, replay level 0, solve level 1.
    assert [action for action, _level in env.actions] == [
        "ACTION1", "ACTION3", "ACTION2", "ACTION4", "ACTION1",
        "ACTION1", "ACTION3", "ACTION2", "ACTION4", "ACTION1", "ACTION2",
    ]
    records = [json.loads(line) for line in (tmp_path / "per-level-bootstrap/decisions.jsonl").read_text().splitlines()]
    probes = [row for row in records if row.get("source") == "bootstrap_probe"]
    replays = [row for row in records if row.get("source") == "bootstrap_replay"]
    model_starts = [row for row in records if row.get("source") == "bootstrap_model_start"]
    resets = [row for row in records if row.get("schema") == "v10.live_bootstrap_reset/1"]
    assert len(probes) == 2
    assert len(replays) == 1
    assert len(model_starts) == 2
    assert resets[1]["return_target_level_key"] == '{"level":1,"levels_completed":1}'

    level_one_prompts = []
    for prompt, images, labels in client.calls:
        if any("Bootstrap for level index 1" in label for label in labels):
            level_one_prompts.append((prompt, images, labels))
    assert len(level_one_prompts) == 1
    prompt, images, labels = level_one_prompts[0]
    context = json.loads(prompt.split("Visible context:\n", 1)[1])
    assert context["current_state"]["level"] == 1
    assert "solved routes replayed" in prompt
    assert len(images) >= 5
    assert labels[-1].startswith("Bootstrap for level index 1")
    assert "CURRENT SCREEN" in labels[-1]
    assert '"bootstrap_observations"' in prompt
    # Replaying the verified level-0 route during restoration did not review it.
    assert [row["through_action"] for row in _review_rows(tmp_path / "per-level-bootstrap")] == [5, 11]
    assert metrics["review_skipped_cached_success_count"] == 1


def test_saved_levels_replay_without_reviews_or_bootstraps_until_first_unsolved_level(
    tmp_path: Path,
) -> None:
    from v11.tools.solutions import LevelSolutionStore

    class ThreeLevelReplayEnv:
        def __init__(self):
            self.frame = np.zeros((3, 4), dtype=np.int64)
            self.level = 0
            self.levels_completed = 0
            self.resets = 0
            self.actions = []
            self.level0_progress = 0

        def observe(self):
            return {
                "frame": self.frame.copy(),
                "legal_actions": [{"id": f"ACTION{i}", "parameters": {
                    "type": "object", "properties": {}, "required": [],
                    "additionalProperties": False,
                }} for i in range(1, 5)],
                "game_state": "NOT_FINISHED", "level": self.level,
                "levels_completed": self.levels_completed, "win_levels": 3,
                "terminal": False,
            }

        def act(self, action, args):
            self.actions.append((action, self.level))
            if self.level == 0 and action == "ACTION1":
                self.level0_progress += 1
                self.frame[0, 0] += 1
                if self.level0_progress == 10:
                    self.level = 1
                    self.levels_completed = 1
                    self.frame[:] = 1
            elif self.level == 1 and action == "ACTION2":
                self.level = 2
                self.levels_completed = 2
                self.frame[:] = 2
            else:
                self.frame[0, 0] += 1
            return {}

        def reset(self):
            self.resets += 1
            self.frame[:] = 0
            self.level = 0
            self.levels_completed = 0
            self.level0_progress = 0
            return self.observe()

    state_dir = tmp_path / "state"
    store_dir = game_memory_dir(state_dir, "fixture")
    store = LevelSolutionStore(store_dir / "level_solutions.jsonl", "fixture")
    seed = ThreeLevelReplayEnv()
    start0 = seed.observe()
    level0_edges = []
    for _ in range(10):
        before = seed.observe()
        seed.act("ACTION1", {})
        edge = store.record_transition(before, "ACTION1", {}, seed.observe(), run_id="saved-routes")
        level0_edges.append(edge)
    start1 = seed.observe()
    seed.act("ACTION2", {})
    start2 = seed.observe()
    edge1 = store.record_transition(start1, "ACTION2", {}, start2, run_id="saved-routes")
    store.record_successful_trajectory(
        start0, level0_edges, run_id="saved-routes", success="level_completed",
    )
    store.record_successful_trajectory(
        start1, [edge1], run_id="saved-routes", success="level_completed",
    )

    env = ThreeLevelReplayEnv()
    client = _FakeClient([_decision([_action("down")])])
    metrics = run_live(
        env, tmp_path / "replay-to-unsolved", client, game_id="fixture",
        max_actions=27, state_dir=state_dir,
    )

    assert env.actions == [
        *[("ACTION1", 0)] * 10, ("ACTION2", 1),
        ("ACTION1", 2), ("ACTION3", 2), ("ACTION2", 2), ("ACTION4", 2),
        *[("ACTION1", 0)] * 10, ("ACTION2", 1), ("ACTION2", 2),
    ]
    assert env.resets == metrics["resets"] == 1
    assert metrics["actions"] == 27
    assert metrics["cached_solution_actions"] == 22
    assert metrics["review_count"] == 0
    assert metrics["review_skipped_cached_success_count"] == 4
    assert metrics["model_calls"] == 1
    assert len(client.calls) == 1
    assert _review_rows(tmp_path / "replay-to-unsolved") == []

    records = [
        json.loads(line)
        for line in (tmp_path / "replay-to-unsolved/decisions.jsonl").read_text().splitlines()
    ]
    assert len([row for row in records if row.get("source") == "bootstrap_probe"]) == 1
    assert sum(len(row.get("cached_path_edge_ids", [])) for row in records) == 22
    assert len([row for row in records if row.get("schema") == "v10.live_bootstrap_reset/1"]) == 1
    prompt, images, labels = client.calls[0]
    assert "The bootstrap observations below are a short exploratory action sequence" in prompt
    assert '"level":2' in prompt
    assert '"bootstrap_observations"' in prompt
    assert len(images) >= 5
    assert "CURRENT SCREEN" in labels[-1]


def test_standard_start_routine_respects_plan_horizon_and_action_budget(tmp_path: Path) -> None:
    class FourDirections(_FakeEnv):
        def observe(self):
            result = super().observe()
            result["legal_actions"].extend([
                {"id": action, "parameters": {
                    "type": "object", "properties": {}, "required": [], "additionalProperties": False,
                }}
                for action in ("ACTION3", "ACTION4")
            ])
            return result

    env = FourDirections()
    metrics = run_live(
        env, tmp_path / "run", _FakeClient([]), game_id="fixture",
        max_actions=2, plan_horizon=1,
    )
    plans = [
        json.loads(line) for line in (tmp_path / "run/decisions.jsonl").read_text().splitlines()
        if json.loads(line).get("schema") == "v10.live_decision/2"
    ]
    assert [action for action, _ in env.actions] == ["ACTION1", "ACTION3"]
    assert all(len(plan["decision"]["actions"]) == 1 for plan in plans)
    assert metrics["actions"] == 2
    assert metrics["model_calls"] == 0


def test_bootstrap_startup_model_call_uses_memory_and_valid_directions(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    _seed_game_mechanic(state_dir)
    client = _FakeClient([_decision([_action("down"), _action("up")])])
    env = _FakeEnv()

    metrics = run_live(
        env, tmp_path / "run", client, game_id="fixture", max_actions=4,
        state_dir=state_dir,
    )

    assert [action for action, _ in env.actions] == ["ACTION1", "ACTION2", "ACTION2", "ACTION1"]
    assert env.resets == 1
    assert metrics["model_calls"] == 1
    assert "game was then reset" in client.calls[0][0]
    assert "A prior observed mechanic." in client.calls[0][0]
    rows = [json.loads(line) for line in (tmp_path / "run/decisions.jsonl").read_text().splitlines()]
    plans = [row for row in rows if row.get("schema") == "v10.live_decision/2"]
    assert plans[-1]["source"] == "bootstrap_model_start"


def test_bootstrap_model_may_choose_legal_wait_action(tmp_path: Path) -> None:
    class WaitEnv(_FakeEnv):
        def observe(self):
            result = super().observe()
            result["legal_actions"].append({
                "id":"WAIT", "parameters":{
                    "type":"object", "properties":{}, "required":[], "additionalProperties":False,
                },
            })
            return result

    env = WaitEnv()
    client = _FakeClient([_decision([_action("WAIT")])])
    metrics = run_live(env, tmp_path / "run", client, game_id="fixture", max_actions=3)

    assert [action for action, _ in env.actions] == ["ACTION1", "ACTION2", "WAIT"]
    assert metrics["actions"] == 3
    rows = [json.loads(line) for line in (tmp_path / "run/decisions.jsonl").read_text().splitlines()]
    plans = [row for row in rows if row.get("schema") == "v10.live_decision/2"]
    assert plans[-1]["source"] == "bootstrap_model_start"
    assert plans[-1]["decision"]["actions"] == [{"action":"WAIT", "args":{}}]


def test_invalid_memory_startup_answer_reasks_model_instead_of_moving_by_fallback(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    _seed_game_mechanic(state_dir)
    client = _FakeClient([_decision([]), _decision([]), _decision([_action("up")])])
    env = _FakeEnv()

    metrics = run_live(env, tmp_path / "run", client, game_id="fixture", max_actions=3, state_dir=state_dir)

    assert [action for action, _ in env.actions] == ["ACTION1", "ACTION2", "ACTION1"]
    assert env.resets == 1
    assert metrics["model_calls"] == 3
    rows = [json.loads(line) for line in (tmp_path / "run/decisions.jsonl").read_text().splitlines()]
    plans = [row for row in rows if row.get("schema") == "v10.live_decision/2"]
    assert plans[-1]["source"] == "bootstrap_model_start"
    assert not any(row.get("source") == "bootstrap_start_fallback" for row in plans)
    rejections = [row for row in rows if row.get("schema") == "v10.live_bootstrap_plan/1"]
    assert len(rejections) == 1
    assert rejections[0]["status"] == "rejected"
    assert rejections[0]["validation_error"] == "bootstrap plan must contain legal directional or wait actions"


def test_three_invalid_startup_plans_use_directional_fallback_then_resume_model_loop(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    _seed_game_mechanic(state_dir)
    client = _FakeClient([_decision([])] * 6 + [_decision([_action("up")])])
    env = _FakeEnv()

    metrics = run_live(env, tmp_path / "run", client, game_id="fixture", max_actions=4, state_dir=state_dir)

    assert len(env.actions) == metrics["actions"] == 4
    assert [action for action, _ in env.actions[:2]] == ["ACTION1", "ACTION2"]
    assert env.actions[2][0] in {"ACTION1", "ACTION2"}
    assert env.actions[3][0] == "ACTION1"
    assert env.resets == 1
    assert metrics["model_calls"] == 7
    assert metrics["stop_reason"] == "action_budget_exhausted"
    rows = [json.loads(line) for line in (tmp_path / "run/decisions.jsonl").read_text().splitlines()]
    rejections = [row for row in rows if row.get("schema") == "v10.live_bootstrap_plan/1"]
    assert [row["attempt"] for row in rejections] == [1, 2, 3]
    fallback = next(row for row in rows if row.get("schema") == "v10.live_fallback/2" and row.get("status") == "executed")
    assert fallback["cause"] == "bootstrap_plan_invalid"
    assert fallback["action"] in {"up", "down"}
    accepted = [row for row in rows if row.get("schema") == "v10.live_decision/2" and row.get("status") == "accepted"][-1]
    assert accepted["source"] == "model"


def test_recent_history_is_independent_of_plan_horizon(tmp_path: Path) -> None:
    client = _FakeClient([_decision([_action()]), _decision([])])
    _run_live_without_starting_routine(_FakeEnv(), tmp_path / "run", client, game_id="fixture", max_actions=2, plan_horizon=1)

    context = _continued_context(_message_user_text(client, 1))
    assert context["state"]["plan_horizon"] == 1
    observations = context["new_observations_since_previous_action_request"]
    assert [row["action"] for row in observations] == ["up"]
    assert "recent_transitions" not in context


def test_action_calls_append_only_observations_and_exact_assistant_responses(tmp_path: Path) -> None:
    decisions = [_decision([_action("up")]) for _ in range(3)]
    client = _FakeClient(decisions)
    _run_live_without_starting_routine(
        _FakeEnv(), tmp_path / "conversation", client, game_id="fixture",
        max_actions=3, plan_horizon=1,
    )

    assert len(client.message_calls) == 3
    messages = [call["messages"] for call in client.message_calls]
    assert len(messages[0]) == 2
    assert messages[1][:-1] == [*messages[0], {
        "role": "assistant", "content": client.response_contents[0],
    }]
    assert messages[2][:-1] == [*messages[1], {
        "role": "assistant", "content": client.response_contents[1],
    }]
    assert messages[1][-2] == {"role": "assistant", "content": client.response_contents[0]}
    assert messages[2][-2] == {"role": "assistant", "content": client.response_contents[1]}

    second_update = _message_user_text(client, 1)
    third_update = _message_user_text(client, 2)
    second_context = _continued_context(second_update)
    third_context = _continued_context(third_update)
    assert [event["action"] for event in second_context["new_observations_since_previous_action_request"]] == ["up"]
    assert [event["action"] for event in third_context["new_observations_since_previous_action_request"]] == ["up"]
    assert second_context["new_observations_since_previous_action_request"][0]["action_number"] == 1
    assert third_context["new_observations_since_previous_action_request"][0]["action_number"] == 2
    assert "CURRENT SCREEN" in [
        block["text"] for block in messages[1][-1]["content"] if block.get("type") == "text"
    ][-1]
    assert "CURRENT SCREEN" in [
        block["text"] for block in messages[2][-1]["content"] if block.get("type") == "text"
    ][-1]


def test_memory_revision_updates_existing_action_conversation_without_reset(tmp_path: Path) -> None:
    client = _FakeClient([
        _decision([_action("up")] * 8),
        _decision([_action("up")] * 2),
        _decision([_action("up")]),
    ])
    metrics = _run_live_without_starting_routine(
        _FakeEnv(), tmp_path / "memory-update", client, game_id="fixture",
        max_actions=11, plan_horizon=8,
    )

    assert metrics["review_count"] == 1
    assert len(client.message_calls) == 3
    first, second, third = [call["messages"] for call in client.message_calls]
    assert second[:-1] == [*first, {"role": "assistant", "content": client.response_contents[0]}]
    assert third[:-1] == [*second, {"role": "assistant", "content": client.response_contents[1]}]
    assert client.message_calls[0]["conversation_reset_count"] == 1
    assert client.message_calls[1]["conversation_reset_count"] == 1
    assert client.message_calls[2]["conversation_reset_count"] == 1
    updated_user = _message_user_text(client, 2)
    assert "Approved profile MEMORY.md" in updated_user
    assert _VALID_GAME_MEMORY in updated_user
    assert "(empty)" in updated_user


def test_action_conversation_without_token_usage_compacts_after_24_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(live_module, "REVIEW_ACTION_THRESHOLD", 1000)
    class NoUsageClient(_FakeClient):
        def complete(self, *args, **kwargs):
            result = super().complete(*args, **kwargs)
            result["usage"]["prompt_tokens"] = None
            return result

    client = NoUsageClient([_decision([_action("up")]) for _ in range(25)])
    _run_live_without_starting_routine(
        _FakeEnv(), tmp_path / "24-turn-conversation", client, game_id="fixture",
        max_actions=25, plan_horizon=1,
    )

    assert len(client.message_calls) == 25
    assert len(client.message_calls[23]["messages"]) == 48
    assert client.message_calls[23]["conversation_reset_count"] == 1
    assert len(client.message_calls[24]["messages"]) == 2
    assert client.message_calls[24]["conversation_reset_count"] == 2
    compacted_prompt, compacted_images, compacted_labels = client.calls[24]
    assert '"recent_transitions"' in compacted_prompt
    assert len(compacted_images) == len(compacted_labels) == 9
    assert "CURRENT SCREEN" in compacted_labels[-1]


def test_default_plan_horizon_runs_eight_actions(tmp_path: Path) -> None:
    client = _FakeClient([_decision([_action()] * 8)])
    metrics = _run_live_without_starting_routine(
        _FakeEnv(), tmp_path / "horizon-eight", client,
        game_id="fixture", max_actions=8,
    )

    assert metrics["actions"] == 8
    assert len(client.calls) == 1
    assert "up to 8 legal actions" in client.calls[0][0]


def test_action_context_uses_eight_recent_transitions_and_all_their_screens(tmp_path: Path) -> None:
    tools = GameTools(_FakeEnv(), tmp_path / "short-context")
    current = tools.observe()
    for _ in range(10):
        current = tools.act("ACTION1")["observation"]

    transitions = _active_level_transitions(
        tools.history.records(), current, tools.reset_generation,
    )
    assert len(transitions) == 8
    inputs = _live_context(
        tools, current, "fixture", 4, shared_skill_path(), tmp_path / "state/fixture",
        "run", 0,
    )
    assert len(inputs["context"]["recent_transitions"]) == 8
    assert "CURRENT SCREEN" in inputs["image_labels"][-1]
    assert "final CURRENT SCREEN as authoritative" in inputs["user_prompt"]
    assert inputs["image_count"] == len(inputs["image_paths"]) == 9
    assert len(inputs["context"]["screen_sequence"]) == 9
    assert inputs["prompt_characters"] <= 4_000
    prompt_context = json.loads(inputs["user_prompt"].split("Visible context:\n", 1)[1])
    summaries = prompt_context["recent_transitions"]
    assert len(summaries) == 8
    assert all(len(summary) <= 200 for summary in summaries)
    assert [summary.split(":", 1)[0].split()[0] for summary in summaries] == [
        str(index) for index in range(1, 9)
    ]
    assert all("changed_pixels" not in summary for summary in summaries)
    assert any("visible" in summary for summary in summaries)


def test_action_context_includes_full_saved_game_and_global_memory(tmp_path: Path) -> None:
    tools = GameTools(_FakeEnv(), tmp_path / "full-memory-run")
    current = tools.observe()
    memory_store_dir = tmp_path / "state/games/fixture"
    memory_store_dir.mkdir(parents=True)
    game_memory = "# Game memory\n\n" + "\n".join(
        f"- Game finding {index}: " + ("g" * 90) for index in range(12)
    ) + "\n- FINAL_GAME_FINDING_MARKER"
    global_memory = "# Global memory\n\n" + "\n".join(
        f"- Global method {index}: " + ("m" * 90) for index in range(12)
    ) + "\n- FINAL_GLOBAL_FINDING_MARKER"
    (memory_store_dir / "MEMORY.md").write_text(game_memory, encoding="utf-8")
    global_path = profile_memory_path(tmp_path / "state", "generic")
    global_path.parent.mkdir(parents=True)
    global_path.write_text(global_memory, encoding="utf-8")

    inputs = _live_context(
        tools, current, "fixture", 8, shared_skill_path(), memory_store_dir,
        "full-memory", 0,
    )
    prompt_context = json.loads(inputs["user_prompt"].split("Visible context:\n", 1)[1])

    assert len(game_memory) > 850
    assert len(global_memory) > 350
    assert prompt_context["memory"] == game_memory
    assert prompt_context["profile_memory"] == global_memory


def test_compact_action_prompt_preserves_noop_and_failed_action_details(tmp_path: Path) -> None:
    tools = GameTools(_FakeEnv(change_every_action=False), tmp_path / "compact-action-failures")
    current = tools.observe()
    tools.act("ACTION1")
    tools.history.append({
        "type": "audit", "kind": "action_error", "action": "ACTION2",
        "error": "RuntimeError: environment rejected input", "reset_generation": 0,
    })
    inputs = _live_context(
        tools, tools.observe(), "fixture", 4, shared_skill_path(),
        tmp_path / "state/fixture", "run", 0,
    )
    context = json.loads(inputs["user_prompt"].split("Visible context:\n", 1)[1])
    assert context["ineffective_attempts_at_current_screen"] == [{"action": "up", "args": {}}]
    assert context["recent_action_errors"] == ["down: RuntimeError: environment rejected input"]
    assert "no visible change" in context["recent_transitions"][0]


def test_compact_action_prompt_marks_recorded_unexpected_outcome(tmp_path: Path) -> None:
    tools = GameTools(_FakeEnv(), tmp_path / "unexpected-outcome")
    tools.observe()
    result = tools.act("ACTION1")
    (tools.run_dir / "reasoning.jsonl").write_text(json.dumps({
        "action_number": 1, "level": 0, "reset_generation": 0,
        "comparison": "contradicted",
        "prediction": {"description": "player should move toward the target"},
        "actual_observed_change": {"visible_change_any_frame": False, "environment_outcome": "no_terminal"},
    }) + "\n", encoding="utf-8")
    inputs = _live_context(
        tools, result["observation"], "fixture", 4, shared_skill_path(),
        tmp_path / "state/fixture", "run", 0,
    )
    context = json.loads(inputs["user_prompt"].split("Visible context:\n", 1)[1])
    assert "unexpected: predicted player should move toward the target" in context["recent_transitions"][0]


def test_noop_knowledge_survives_beyond_recent_visual_history(tmp_path: Path) -> None:
    class VariedFrames(_FakeEnv):
        def __init__(self):
            super().__init__(change_every_action=False)

        def act(self, action, args):
            self.actions.append((action, dict(args)))
            count = len(self.actions)
            if count == 1:
                return {}  # up is ineffective from the initial all-zero screen
            self.frame[:] = 0 if count == 11 else count
            return {}

    env = VariedFrames()
    tools = GameTools(env, tmp_path / "persistent-noop")
    current = tools.observe()
    tools.act("ACTION1")
    for _ in range(10):
        current = tools.act("ACTION2")["observation"]

    transitions = _active_level_transitions(
        tools.history.records(), current, tools.reset_generation,
    )
    assert len(transitions) == 8
    inputs = _live_context(
        tools, current, "fixture", 4, shared_skill_path(), tmp_path / "state/fixture",
        "run", 0,
    )
    assert inputs["context"]["ineffective_actions_at_current_screen"] == ["up"]
    assert len(inputs["context"]["recent_transitions"]) == 8


def test_known_noop_later_in_plan_is_blocked_using_full_level_history(tmp_path: Path) -> None:
    class ReturnToOldStateEnv(_FakeEnv):
        def __init__(self):
            super().__init__(change_every_action=False)
            self.state = 0

        def observe(self):
            result = super().observe()
            result["legal_actions"].append({
                "id": "ACTION3", "parameters": {
                    "type": "object", "properties": {}, "required": [],
                    "additionalProperties": False,
                },
            })
            result["frame"][:] = self.state
            self.frame[:] = self.state
            return result

        def act(self, action, args):
            self.actions.append((action, dict(args)))
            if self.state == 0 and action == "ACTION1":
                return {}  # Record an old up no-op at state zero.
            if action == "ACTION2":
                self.state += 1
            elif action == "ACTION3" and self.state == 9:
                self.state = 0
            self.frame[:] = self.state
            return {}

    client = _FakeClient([
        _decision([_action("up")]),
        _decision([_action("down")] * 4),
        _decision([_action("down")] * 4),
        _decision([_action("down")]),
        _decision([_action("left"), _action("up")]),
    ])
    env = ReturnToOldStateEnv()
    metrics = _run_live_without_starting_routine(
        env, tmp_path / "old-noop-in-plan", client,
        game_id="fixture", max_actions=20, plan_horizon=4,
    )

    assert metrics["actions"] == 11
    assert [action for action, _args in env.actions] == [
        "ACTION1", *(["ACTION2"] * 9), "ACTION3",
    ]
    rows = [json.loads(line) for line in (tmp_path / "old-noop-in-plan/decisions.jsonl").read_text().splitlines()]
    assert any(
        row.get("interrupt_reason") == "known_ineffective_action"
        for row in rows if row.get("schema") == "v10.live_plan_result/2"
    )


def test_context_reports_grounded_native_pixel_regions_and_motion(tmp_path: Path) -> None:
    class MovingRegionEnv:
        def __init__(self):
            self.frame = np.zeros((8, 8), dtype=np.int64)
            self.frame[2:4, 1:3] = 9
            self.moved = False

        def observe(self):
            return {
                "frame": self.frame.copy(),
                "legal_actions": [{"id": "ACTION4", "parameters": {
                    "type": "object", "properties": {}, "required": [],
                    "additionalProperties": False,
                }}],
                "game_state": "NOT_FINISHED", "level": 0, "levels_completed": 0,
                "win_levels": 5, "terminal": False,
            }

        def act(self, action, args):
            self.frame[:] = 0
            self.frame[2:4, 3:5] = 9
            self.moved = True
            return {}

    env = MovingRegionEnv()
    tools = GameTools(env, tmp_path / "region-motion")
    current = tools.observe()
    tools.act("ACTION4")
    current = tools.observe()
    inputs = _live_context(
        tools, current, "fixture", 1, shared_skill_path(), tmp_path / "state/fixture",
        "run", 0,
    )
    assert inputs["context"]["visible_regions_current_screen"][9] == [
        {"x_min": 3, "y_min": 2, "x_max": 4, "y_max": 3, "pixels": 4},
    ]
    motion = inputs["context"]["recent_transitions"][0]["observed_pixel_region_motion"]
    assert motion == [{
        "pixel_class": 9,
        "from_bbox_native_pixels": [1, 2, 2, 3],
        "to_bbox_native_pixels": [3, 2, 4, 3],
        "shift_native_pixels": {"dx": 2, "dy": 0},
    }]


def test_review_prompt_classifies_repeated_same_state_noops_as_failures(tmp_path: Path) -> None:
    env = _FakeEnv(change_every_action=False)
    tools = GameTools(env, tmp_path / "review-noop")
    tools.observe()
    tools.act("ACTION1")
    client = _FakeClient([], review_responses=[_VALID_REVIEW])

    result = _perform_review(
        tools, client, game_id="fixture", run_id="review-run",
        memory_store_dir=game_memory_dir(tmp_path / "state", "fixture"),
        trigger_reasons=["stagnation"],
    )

    assert result["row"]["status"] == "saved"
    prompt = client.calls[0][0]
    assert "no pixel change from the same visible state is a failed/ineffective approach" in prompt
    assert '"unchanged_count":1' in prompt
    assert "remove or correct that wording" in prompt
    assert "failed or ineffective approaches" in client.system_prompts[0]
    assert "Scope object roles, goal identity, and completion rules" in client.system_prompts[0]


def test_known_ineffective_action_is_rejected_and_plan_reasks_after_noop(tmp_path: Path) -> None:
    class ThreeActionNoopEnv(_FakeEnv):
        def observe(self):
            result = super().observe()
            result["legal_actions"].append({
                "id": "ACTION3", "parameters": {
                    "type": "object", "properties": {}, "required": [],
                    "additionalProperties": False,
                },
            })
            return result

    env = ThreeActionNoopEnv(change_every_action=False)
    client = _FakeClient([
        _decision([_action("up"), _action("down")]),
        _decision([_action("up"), _action("down")]),
        _decision([_action("left")]),
    ])

    metrics = _run_live_without_starting_routine(
        env, tmp_path / "ineffective-action", client, game_id="fixture", max_actions=2,
    )

    assert [action for action, _args in env.actions] == ["ACTION1", "ACTION3"]
    assert metrics["actions"] == 2
    repair_context = _continued_context(_message_user_text(client, 1))
    assert repair_context["state"]["ineffective_attempts_at_current_screen"] == [
        {"action": "up", "args": {}}
    ]
    assert "first action 'up' already caused no visible change" in client.message_calls[2]["messages"][-1]["content"][0]["text"]


def test_all_ineffective_legal_actions_trigger_unbudgeted_reset(tmp_path: Path) -> None:
    env = _FakeEnv(change_every_action=False)
    client = _FakeClient([], default_decision={"invalid": True})

    metrics = _run_live_without_starting_routine(
        env, tmp_path / "automatic-recovery", client, game_id="fixture", max_actions=3,
    )

    assert env.resets == 1
    assert metrics["resets"] == 1
    assert metrics["actions"] == 3
    decisions = [
        json.loads(line) for line in (tmp_path / "automatic-recovery/decisions.jsonl").read_text().splitlines()
    ]
    assert any(row.get("status") == "auto_reset" for row in decisions)


def test_stagnation_guidance_remains_and_interval_review_runs_at_threshold(tmp_path: Path) -> None:
    client = _FakeClient(
        [], default_decision=_decision([_action()]),
    )
    metrics = _run_live_without_starting_routine(
        _OscillatingEnv(), tmp_path / "run", client,
        game_id="fixture", max_actions=16,
    )

    assert metrics["review_count"] == 1
    assert _review_rows(tmp_path / "run")[0]["through_action"] == 10
    action_contexts = [
        _continued_context(_message_user_text(client, index))
        for index in range(1, len(client.message_calls))
    ]
    recovery_context = next(
        ctx for ctx in action_contexts if ctx["state"]["recovery_guidance"]
    )
    assert recovery_context["state"]["consecutive_repeated_or_no_progress_actions"] >= 2
    assert "RESET alone" in recovery_context["state"]["recovery_guidance"]
    assert recovery_context["state"]["plan_horizon"] == 1


def test_stagnation_does_not_call_the_reviewer(tmp_path: Path) -> None:
    client = _FakeClient(
        [], default_decision=_decision([_action()]),
        review_errors=[RuntimeError("temporary reviewer failure")],
    )
    metrics = _run_live_without_starting_routine(
        _OscillatingEnv(), tmp_path / "run", client,
        game_id="fixture", max_actions=9,
    )
    assert metrics["review_count"] == 0
    assert _review_rows(tmp_path / "run") == []
    assert len(client.review_errors) == 1
    assert str(client.review_errors[0]) == "temporary reviewer failure"


def test_keyboard_interrupt_writes_interrupted_metrics(tmp_path: Path) -> None:
    class InterruptingClient(_FakeClient):
        def __init__(self):
            super().__init__([_decision([_action()])])
            self.calls_made = 0

        def complete(self, *args, **kwargs):
            self.calls_made += 1
            if self.calls_made == 1:
                return super().complete(*args, **kwargs)
            raise KeyboardInterrupt

    metrics = _run_live_without_starting_routine(
        _FakeEnv(), tmp_path / "run", InterruptingClient(),
        game_id="fixture", max_actions=4,
    )
    assert metrics["stop_reason"] == "interrupted"
    assert metrics["actions"] == 1
    assert len([row for row in History(tmp_path / "run").records() if row.get("type") == "transition"]) == 1
    assert not (tmp_path / "run/reviews.jsonl").exists()


def test_live_executes_action_plan_without_prediction_interrupts(tmp_path: Path) -> None:
    env = _FakeEnv()
    first = _decision([_action(), _action("down")])
    second = _decision([])
    client = _FakeClient([first, second])

    metrics = _run_live_without_starting_routine(env, tmp_path / "run", client, game_id="fixture", max_actions=5)

    assert env.actions == [("ACTION1", {}), ("ACTION2", {})]
    assert metrics["actions"] == 2
    assert metrics["stop_reason"] == "model_returned_no_actions"
    rows = [json.loads(line) for line in (tmp_path / "run" / "decisions.jsonl").read_text().splitlines()]
    plan_results = [row for row in rows if row["schema"] == "v10.live_plan_result/2"]
    assert plan_results[0]["interrupt_reason"] is None
    transitions = [row for row in History(tmp_path / "run").records() if row.get("type") == "transition"]
    assert len(transitions) == 2
    assert all("prediction_evidence_id" not in row and "prediction" not in row for row in transitions)
    assert transitions[0]["delta_image_id"]


def test_live_run_clears_existing_run_directory_contents(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    stale_dir = run_dir / "old-output"
    stale_dir.mkdir(parents=True)
    (stale_dir / "stale.txt").write_text("previous run", encoding="utf-8")

    metrics = _run_live_without_starting_routine(
        _FakeEnv(), run_dir, _FakeClient([_decision([])]), game_id="fixture", max_actions=3,
    )

    assert not stale_dir.exists()
    assert (run_dir / "metrics.json").exists()
    assert metrics["actions"] == 0


def test_live_executes_reset_action_without_a_reset_budget(tmp_path: Path) -> None:
    env = _FakeEnv()
    client = _FakeClient([_decision(reset=True), _decision([])])

    metrics = _run_live_without_starting_routine(env, tmp_path / "run", client, game_id="fixture", max_actions=3)

    assert env.resets == 1
    assert metrics["stop_reason"] == "model_returned_no_actions"
    assert metrics["actions"] == 0


def test_live_allows_more_than_ten_model_requested_resets(tmp_path: Path) -> None:
    env = _FakeEnv()
    client = _FakeClient([_decision(reset=True) for _ in range(12)] + [_decision([_action()])])

    metrics = _run_live_without_starting_routine(
        env, tmp_path / "unlimited-resets", client, game_id="fixture", max_actions=1,
    )

    assert env.resets == 12
    assert metrics["resets"] == 12
    assert "reset_budget" not in metrics
    first_prompt = client.calls[0][0]
    first_context = json.loads(first_prompt.split("Visible context:\n", 1)[1])
    assert any(action["id"] == "RESET" for action in first_context["current_state"]["legal_actions"])
    assert "resets_remaining" not in first_context["budgets"]


def test_live_metrics_include_strata_json_recovery_attempt(tmp_path: Path) -> None:
    metrics = _run_live_without_starting_routine(
        _FakeEnv(), tmp_path / "run", _FakeClient([_decision([])], request_attempts=2),
        game_id="fixture", max_actions=3,
    )

    assert metrics["model_calls"] == 2
    assert metrics["stop_reason"] == "model_returned_no_actions"


def test_live_action_only_responses_create_no_prediction_audits(tmp_path: Path) -> None:
    client = _FakeClient([_decision([_action(), _action("down")]), _decision([])])
    env = _FakeEnv()

    metrics = _run_live_without_starting_routine(
        env, tmp_path / "run", client, game_id="fixture", max_actions=2, plan_horizon=2,
    )

    assert metrics["actions"] == 2
    prediction_rows = [
        row for row in History(tmp_path / "run").records()
        if row.get("type") == "audit" and row.get("kind") == "prediction"
    ]
    assert prediction_rows == []


def test_confirmed_pattern_prediction_is_audited_and_triggers_review_on_conflict(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    prior = GameMemory(
        tmp_path / "prior", store_dir=state_dir / "games/fixture",
        run_id="prior-run", game_id="fixture",
    )
    prior.apply_review_changes([{
        "operation": "upsert", "pattern_type": "mechanic", "status": "confirmed", "claim": "Up changes 999 pixels.",
        "applicability_conditions": ["Any level."],
        "state_selector": {},
        "expected_observations": [{
            "action": "up", "state_selector": {},
            "measurable_prediction": {"all": [
                {"path": "measurements.changed_pixels", "op": "eq", "value": 999},
            ]},
        }],
    }], game_id="fixture", current_step=2, covered_event_count=2)
    pattern_id = prior.patterns("fixture")[0]["id"]
    client = _FakeClient(
        [_decision([_action()])], review_responses=[_VALID_REVIEW],
    )

    metrics = _run_live_without_starting_routine(
        _FakeEnv(), tmp_path / "run", client, game_id="fixture", max_actions=1,
        state_dir=state_dir,
    )

    audits = [
        row for row in History(tmp_path / "run").records()
        if row.get("kind") == "pattern_prediction"
    ]
    assert len(audits) == 1
    assert audits[0]["pattern_id"] == pattern_id
    assert audits[0]["verdict"] == "failed"
    assert metrics["pattern_predictions"] == 1
    assert metrics["pattern_prediction_contradictions"] == 1
    assert metrics["review_count"] == 0
    assert _review_rows(tmp_path / "run") == []
    updated = GameMemory(
        tmp_path / "run", store_dir=state_dir / "games/fixture",
        run_id=json.loads((tmp_path / "run/run_manifest.json").read_text())["run_id"], game_id="fixture",
    ).patterns("fixture")[0]
    assert updated["status"] == "contradicted"
    assert "contradicting_evidence" not in updated


def test_live_repairs_response_that_includes_prediction_fields(tmp_path: Path) -> None:
    env = _FakeEnv()
    wrapped = _decision([_action()])
    wrapped["actions"][0]["measurable_prediction"] = {"changed_pixels": 1}
    client = _FakeClient([wrapped, _decision([_action()])])

    metrics = _run_live_without_starting_routine(env, tmp_path / "run", client, game_id="fixture", max_actions=1)

    assert env.actions == [("ACTION1", {})]
    assert metrics["actions"] == 1
    rows = [json.loads(line) for line in (tmp_path / "run" / "decisions.jsonl").read_text().splitlines()]
    accepted = next(row for row in rows if row.get("status") == "accepted")
    assert accepted["decision"]["actions"] == [{"action": "up", "args": {}}]
    assert accepted["repair_attempted"] is True


def test_live_repairs_response_with_extra_fields(tmp_path: Path) -> None:
    invalid = _decision([_action()])
    invalid["actions"][0]["measurable_prediction"] = {"pixel_diff": {"invented": 1}}
    client = _FakeClient([invalid, _decision([_action()]), _decision([])])

    metrics = _run_live_without_starting_routine(_FakeEnv(), tmp_path / "run", client, game_id="fixture", max_actions=1)

    assert metrics["actions"] == 1
    assert len(client.calls) == 2
    repair_prompt = client.calls[1][0]
    assert "Validation error:" in repair_prompt
    initial_messages = client.message_calls[0]["messages"]
    repair_messages = client.message_calls[1]["messages"]
    assert repair_messages[:-1] == initial_messages[:-1]
    assert repair_messages[-1]["content"][1:] == initial_messages[-1]["content"][1:]
    assert "may contain only action and args" in repair_prompt
    assert "Do not add prose." in repair_prompt
    rows = [json.loads(line) for line in (tmp_path / "run" / "decisions.jsonl").read_text().splitlines()]
    accepted = next(row for row in rows if row.get("status") == "accepted")
    assert accepted["repair_attempted"] is True
    assert "actions[0] may contain only action and args" in accepted["initial_validation_error"]


def test_live_executes_random_fallback_after_invalid_decision_and_repair(tmp_path: Path) -> None:
    invalid = _decision([_action()])
    invalid["actions"][0]["measurable_prediction"] = {"pixel_diff": {"invented": 1}}
    still_invalid = _decision([_action()])
    still_invalid["actions"][0]["measurable_prediction"] = {"pixel_diff": {"invented": 1}}
    env = _FakeEnv()

    metrics = _run_live_without_starting_routine(
        env, tmp_path / "run", _FakeClient([invalid, still_invalid]), game_id="fixture", max_actions=1,
    )

    assert metrics["actions"] == 1
    assert env.actions[0][0] in {"ACTION1", "ACTION2"}
    assert metrics["stop_reason"] == "action_budget_exhausted"
    rows = [json.loads(line) for line in (tmp_path / "run" / "decisions.jsonl").read_text().splitlines()]
    fallback = next(row for row in rows if row.get("schema") == "v10.live_fallback/2")
    assert fallback["status"] == "executed"
    assert fallback["cause"] == "invalid_decision"
    assert "prediction_verdict" not in fallback


@pytest.mark.parametrize("repair_action", ["up", "NOT_LEGAL"])
def test_live_rejected_duplicate_candidate_cannot_execute_when_repair_fails(
    tmp_path: Path, repair_action: str,
) -> None:
    duplicate = _decision([_action("up"), _action("down")])
    duplicate["actions"][1]["prediction_id"] = duplicate["actions"][0]["prediction_id"] = "P_DUP"
    repair = _decision([_action(repair_action), _action("down")])
    if repair_action == "up":
        repair["actions"][1]["prediction_id"] = repair["actions"][0]["prediction_id"] = "P_DUP"
    env = _FakeEnv()
    metrics = _run_live_without_starting_routine(
        env, tmp_path / "run", _FakeClient([duplicate, repair]),
        game_id="fixture", max_actions=1,
    )
    rows = [json.loads(line) for line in (tmp_path / "run" / "decisions.jsonl").read_text().splitlines()]
    rejected = next(row for row in rows if row.get("status") == "rejected")
    fallback = next(row for row in rows if row.get("schema") == "v10.live_fallback/2" and row.get("status") == "executed")
    expected_error = (
        "actions[0] may contain only action and args" if repair_action == "up"
        else "actions[0] has illegal action: NOT_LEGAL"
    )
    assert rejected["validation_error"] == expected_error
    assert fallback["cause"] == "invalid_decision"
    assert len(env.actions) == metrics["actions"] == 1
    assert env.actions[0][0] in {"ACTION1", "ACTION2"}


@pytest.mark.parametrize("bad_value", [
    float("nan"), float("inf"), {"nested": [float("-inf")]},
    {"nested": {1: "non-string key"}}, ("tuple", "is not JSON data"),
])
def test_game_tools_reject_non_finite_args_before_environment_action(
    tmp_path: Path, bad_value: object,
) -> None:
    class ArgsEnv(_FakeEnv):
        def observe(self):
            result = super().observe()
            result["legal_actions"] = [{
                "id": "ARG_ACTION", "parameters": {
                    "type": "object", "properties": {"value": {}},
                    "required": ["value"], "additionalProperties": False,
                },
            }]
            return result

    env = ArgsEnv()
    tools = GameTools(env, tmp_path / "run")
    tools.observe()
    with pytest.raises(ActionError, match="finite JSON data"):
        tools.act("ARG_ACTION", {"value": bad_value})
    assert env.actions == []


def test_live_resumes_model_loop_after_invalid_decision_fallback(tmp_path: Path) -> None:
    invalid = _decision([_action()])
    invalid["actions"][0]["measurable_prediction"] = {"pixel_diff": {"invented": 1}}
    still_invalid = _decision([_action()])
    still_invalid["actions"][0]["measurable_prediction"] = {"pixel_diff": {"invented": 1}}
    client = _FakeClient([invalid, still_invalid, _decision([])])

    metrics = _run_live_without_starting_routine(_FakeEnv(), tmp_path / "run", client, game_id="fixture", max_actions=2)

    assert metrics["actions"] == 1
    assert len(client.calls) == 3
    assert metrics["stop_reason"] == "model_returned_no_actions"


def test_game_tools_observe_is_idempotent_between_actions(tmp_path: Path) -> None:
    env = _FakeEnv()
    tools = GameTools(env, tmp_path / "run")

    first = tools.observe()
    again = tools.observe()

    assert first["frame_id"] == again["frame_id"]
    assert first["evidence_id"] == again["evidence_id"]
    assert len(tools.frames.entries()) == 2  # one grid and one rendered PNG
    assert len(History(tmp_path / "run").records()) == 1


def test_arcade_adapter_maps_native_action_ids_and_observation(monkeypatch) -> None:
    class _Action:
        def __init__(self, action_id):
            self.value = action_id
            self.name = f"ACTION{action_id}"

    class _GameAction:
        @staticmethod
        def from_id(action_id):
            return _Action(action_id)

    monkeypatch.setitem(sys.modules, "arcengine", types.SimpleNamespace(GameAction=_GameAction))

    class _NativeEnv:
        def __init__(self):
            self.last_action = None
            self.raw = types.SimpleNamespace(
                frame=np.zeros((1, 3, 4), dtype=np.int64), available_actions=[1, 2],
                state=types.SimpleNamespace(name="NOT_FINISHED"), levels_completed=0, win_levels=5,
            )

        def reset(self):
            return self.raw

        def step(self, action, data, reasoning):
            self.last_action = (action.value, data, reasoning)
            return self.raw

    native = _NativeEnv()
    adapter = ArcadeGameEnv(
        "ez01", seed=0,
        env_factory=lambda game_id, **kwargs: native,
    )

    observation = adapter.observe()
    assert observation["frame"].shape == (3, 4)
    assert [item["id"] for item in observation["legal_actions"]] == ["ACTION1", "ACTION2"]
    adapter.act(1, {})
    assert native.last_action == (1, {"game_id": "ez01"}, None)


def test_live_context_has_no_future_records_and_includes_current_images(tmp_path: Path) -> None:
    env = _FakeEnv()
    client = _FakeClient([_decision([])])

    metrics = _run_live_without_starting_routine(env, tmp_path / "run", client, game_id="fixture", max_actions=3)

    assert metrics["stop_reason"] == "model_returned_no_actions"
    prompt, image_paths, labels = client.calls[0]
    system_prompt = client.system_prompts[0]
    assert "current_evidence_id" not in prompt
    assert "initial_state_evidence_id" not in prompt
    assert "measurable_prediction" not in prompt
    assert "memory_patch" not in prompt
    assert "up to 8 legal actions" in prompt
    assert '"id":"up"' in prompt
    assert '"id":"down"' in prompt
    assert "ACTION1" not in prompt
    assert len(image_paths) == 1
    assert len(labels) == 1 and "CURRENT SCREEN" in labels[0]
    assert '"level":0' in prompt
    assert '"win_levels":5' in prompt
    assert "first action only" in prompt
    assert "memory contains conflicting claims" in system_prompt
    assert "use `null` only when no hypothesis is being tested" in system_prompt
    assert "discovery functions are available during action selection" in system_prompt


def test_action_conversation_appends_action_outcomes_and_current_screen(tmp_path: Path) -> None:
    class IntermediateFramesEnv(_FakeEnv):
        def observe(self):
            result = super().observe()
            result["legal_actions"].extend([
                {"id": action, "parameters": {
                    "type": "object", "properties": {}, "required": [], "additionalProperties": False,
                }}
                for action in ("ACTION3", "ACTION4")
            ])
            return result

        def act(self, action, args):
            super().act(action, args)
            value = len(self.actions) * 10
            return {"frames": [
                np.full(self.frame.shape, value + 1, dtype=np.int64),
                np.full(self.frame.shape, value + 2, dtype=np.int64),
            ]}

    client = _FakeClient([
        _decision([_action("up"), _action("left"), _action("down"), _action("right")]),
        _decision([]),
    ])
    _run_live_without_starting_routine(
        IntermediateFramesEnv(), tmp_path / "run", client,
        game_id="fixture", max_actions=5, plan_horizon=4,
    )

    first_user = client.message_calls[0]["messages"][-1]["content"]
    assert len([block for block in first_user if block.get("type") == "image_url"]) == 1
    assert "CURRENT SCREEN" in first_user[1]["text"]
    update_user = client.message_calls[1]["messages"][-1]["content"]
    update_prompt = next(block["text"] for block in update_user if block.get("type") == "text")
    context = _continued_context(update_prompt)
    events = context["new_observations_since_previous_action_request"]
    assert [row["action"] for row in events] == ["up", "left", "down", "right"]
    assert [row["intermediate_screens"] for row in events] == [2, 2, 2, 2]
    image_blocks = [block for block in update_user if block.get("type") == "image_url"]
    assert len(image_blocks) == 13  # Initial, 2 intermediate + result per action; current is deduplicated.
    image_labels = [block["text"] for block in update_user if block.get("type") == "text"][1:]
    assert "initial state before Action 1 (up)" in image_labels[0]
    for action_number in range(1, 5):
        assert sum(f"produced by Action {action_number} (" in label for label in image_labels) == 3
    assert "CURRENT SCREEN" in image_labels[-1]
    history = [json.loads(line) for line in (tmp_path / "run/evidence.jsonl").read_text().splitlines()]
    transitions = [row for row in history if row.get("type") == "transition"]
    assert sum(len(row.get("result", {}).get("intermediate_frame_ids", [])) for row in transitions) == 8


def test_action_context_scopes_history_and_screens_to_active_level(tmp_path: Path) -> None:
    class TwoLevelEnv(_FakeEnv):
        def __init__(self):
            super().__init__()
            self.level = 0
            self.levels_completed = 0

        def observe(self):
            result = super().observe()
            result["legal_actions"].extend([
                {"id": action, "parameters": {
                    "type": "object", "properties": {}, "required": [], "additionalProperties": False,
                }}
                for action in ("ACTION3", "ACTION4")
            ])
            result["level"] = self.level
            result["levels_completed"] = self.levels_completed
            return result

        def act(self, action, args):
            self.actions.append((action, dict(args)))
            if len(self.actions) == 1:
                self.level = 1
                self.levels_completed = 1
                self.frame[0, 0] = 10  # New level's starting frame.
            else:
                self.frame[0, len(self.actions) % self.frame.shape[1]] = len(self.actions)
            return {}

    client = _FakeClient([
        _decision([_action("up")]),
        _decision([_action("right")]),
        _decision([]),
    ])
    _run_live_without_starting_routine(
        TwoLevelEnv(), tmp_path / "run", client, game_id="fixture",
        max_actions=3, plan_horizon=1,
    )

    assert len(client.message_calls) == 3

    # The first action in level 1 completed level 0. Its old-level pre-frame
    # and action must not appear in level 1's exploration context.
    second_prompt = _message_user_text(client, 1)
    second_context = json.loads(second_prompt.split("Visible context:\n", 1)[1])
    assert second_context["current_state"]["level"] == 1
    assert second_context["current_state"]["levels_completed"] == 1
    assert second_context["recent_transitions"] == []
    assert len([block for block in client.message_calls[1]["messages"][-1]["content"] if block.get("type") == "image_url"]) == 1
    assert "earlier levels are omitted" in second_prompt

    # Once level 1 has an action, its own start, transition, and current image
    # are aligned; the level 0 completion action remains excluded.
    third_prompt = _message_user_text(client, 2)
    third_context = _continued_context(third_prompt)
    events = third_context["new_observations_since_previous_action_request"]
    assert [row["action"] for row in events] == ["right"]
    assert all(row["level_after"] == 1 for row in events)
    assert [row["action"] for row in third_context["new_observations_since_previous_action_request"]] == ["right"]
    final_labels = [
        block["text"] for block in client.message_calls[2]["messages"][-1]["content"]
        if block.get("type") == "text"
    ]
    assert "CURRENT SCREEN" in final_labels[-1]


class _LevelCompleteEnv(_FakeEnv):
    def __init__(self, *, terminal=False, start_completed=0):
        super().__init__()
        self.terminal_after_action = terminal
        self.start_completed = start_completed

    def observe(self):
        result = super().observe()
        completed = self.start_completed + len(self.actions)
        result["levels_completed"] = completed
        result["level"] = completed
        result["terminal"] = self.terminal_after_action and bool(self.actions)
        result["game_state"] = "WIN" if result["terminal"] else "NOT_FINISHED"
        return result


def test_review_context_marks_level_boundary_and_verified_scope(tmp_path: Path) -> None:
    env = _LevelCompleteEnv()
    client = _FakeClient([_decision([_action()])], review_responses=[_VALID_REVIEW])
    _run_live_without_starting_routine(
        env, tmp_path / "run", client, game_id="fixture", max_actions=1,
    )

    review_prompt = next(
        prompt for prompt, _, _ in client.calls
        if "Consolidate memory for game fixture" in prompt
    )
    assert "active level index 1 (zero-based), displayed as level 2" in review_prompt
    assert "active level may be the newly started level after the completion boundary" in review_prompt
    assert "only for the level scopes present in verified routes" in review_prompt
    assert "completed-level review: prioritize transition-supported mechanics and strategies" in review_prompt
    events = _review_events(History(tmp_path / "run").records())
    transition = next(event for event in events if event["type"] == "transition")
    assert transition["level_index_before_action_zero_based"] == 0
    assert transition["level_index_after_action_zero_based"] == 1
    assert transition["crossed_level_boundary"] is True


def test_cached_level_success_skips_reviews_on_replayed_route(tmp_path: Path) -> None:
    from v11.tools.solutions import LevelSolutionStore

    state_dir = tmp_path / "state"
    store_dir = game_memory_dir(state_dir, "fixture")
    save_game_memory(store_dir, "# Game Memory\n\n- Use verified routes on matching starts.")

    start = _LevelCompleteEnv().observe()
    finish_frame = start["frame"].copy()
    finish_frame[0, 1] = 1
    finish = {
        **start, "frame": finish_frame, "level": 1, "levels_completed": 1,
        "terminal": False, "game_state": "NOT_FINISHED",
    }
    store = LevelSolutionStore(store_dir / "level_solutions.jsonl", "fixture")
    edge = store.record_transition(start, "ACTION1", {}, finish, run_id="previous")
    store.record_successful_trajectory(start, [edge], run_id="previous", success="level_completed")

    client = _FakeClient([])
    metrics = _run_live_without_starting_routine(
        _LevelCompleteEnv(), tmp_path / "cached-level", client, game_id="fixture",
        max_actions=1, state_dir=state_dir,
    )
    assert metrics["cached_solution_actions"] == 1
    assert metrics["review_count"] == 0
    assert metrics["review_skipped_cached_success_count"] == 1
    assert client.calls == []
    assert _review_rows(tmp_path / "cached-level") == []

    # A cached route that wins the game also skips model reviews.
    win_start = _LevelCompleteEnv(terminal=True, start_completed=4).observe()
    win_frame = win_start["frame"].copy()
    win_frame[0, 1] = 1
    win_finish = {
        **win_start, "frame": win_frame, "level": 5, "levels_completed": 5,
        "terminal": True, "game_state": "WIN",
    }
    win_store = LevelSolutionStore(store_dir / "level_solutions.jsonl", "fixture")
    win_edge = win_store.record_transition(win_start, "ACTION1", {}, win_finish, run_id="previous-win")
    win_store.record_successful_trajectory(win_start, [win_edge], run_id="previous-win", success="win")
    win_client = _FakeClient([])
    win_metrics = _run_live_without_starting_routine(
        _LevelCompleteEnv(terminal=True, start_completed=4), tmp_path / "cached-win",
        win_client, game_id="fixture", max_actions=1, state_dir=state_dir,
    )
    assert win_metrics["won"] is True
    assert win_metrics["review_count"] == 0
    assert win_metrics["review_count"] == 0
    assert win_metrics["review_skipped_cached_success_count"] == 1
    assert win_client.calls == []
    assert _review_rows(tmp_path / "cached-win") == []


def test_uncertain_reasoned_action_is_joined_to_outcome_without_extra_model_call(tmp_path: Path) -> None:
    client = _FakeClient([_reasoned(
        _decision([_action("up")]),
        hypothesis={"claim": "The action changes the visible board.", "evidence": "a visible player-like region"},
        prediction={"visible_change": "yes", "environment_outcome": "no_terminal", "description": "some pixels change"},
    )])
    metrics = _run_live_without_starting_routine(
        _FakeEnv(), tmp_path / "uncertain", client,
        game_id="fixture", max_actions=1, plan_horizon=1,
    )

    rows = [json.loads(line) for line in (tmp_path / "uncertain/reasoning.jsonl").read_text().splitlines()]
    assert metrics["reasoning_records"] == 1
    assert metrics["model_calls"] == 1
    assert len(rows) == 1
    assert rows[0]["comparison"] == "supported"
    assert rows[0]["comparison_scope"] == "first_executed_action"
    assert rows[0]["comparison_basis"] == ["visible_change", "environment_outcome"]
    assert rows[0]["comparison_check_results"] == {
        "visible_change": True, "environment_outcome": True,
    }
    assert rows[0]["prediction_description_assessed"] is False
    assert rows[0]["hypothesis_status"] == "unresolved"
    assert rows[0]["planned_first_action"] == {"action": "up", "args": {}}
    assert rows[0]["source_screen_number"] == 1
    assert rows[0]["produced_screen_numbers"] == [2]
    decision = next(
        json.loads(line) for line in (tmp_path / "uncertain/decisions.jsonl").read_text().splitlines()
        if json.loads(line).get("schema") == "v10.live_decision/2"
    )
    assert decision["decision"]["actions"] == [{"action": "up", "args": {}}]
    assert decision["decision_schema"] == "v10.reasoned_action/1"
    assert len(client.calls) == 1


def test_prediction_mismatch_is_logged_but_does_not_interrupt_plan(tmp_path: Path) -> None:
    client = _FakeClient([_reasoned(
        _decision([_action("up"), _action("down")]),
        hypothesis={"claim": "The action does not alter the board.", "evidence": "same visible region"},
        prediction={
            "visible_change": "no", "environment_outcome": "no_terminal",
            "description": "After the full up then down plan, the board remains unchanged.",
        },
    )])
    env = _FakeEnv()
    metrics = _run_live_without_starting_routine(
        env, tmp_path / "prediction-mismatch", client,
        game_id="fixture", max_actions=2, plan_horizon=2,
    )
    row = json.loads((tmp_path / "prediction-mismatch/reasoning.jsonl").read_text().splitlines()[0])
    result = next(
        json.loads(line) for line in (tmp_path / "prediction-mismatch/decisions.jsonl").read_text().splitlines()
        if json.loads(line).get("schema") == "v10.live_plan_result/2"
    )
    assert row["comparison"] == "contradicted"
    assert row["comparison_scope"] == "first_executed_action"
    assert row["prediction_description_assessed"] is False
    assert row["comparison_check_results"] == {
        "visible_change": False, "environment_outcome": True,
    }
    assert metrics["actions"] == 2
    assert len(result["executed"]) == 2
    assert row["remaining_plan_aborted_due_to_prediction"] is False


def _review_rows(run_dir: Path) -> list[dict]:
    path = run_dir / "reviews.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_review_saves_model_markdown_and_next_action_reads_it_verbatim(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    client = _FakeClient(
        [_decision([_action(), _action("down")]), _decision([])],
        review_responses=[_VALID_REVIEW],
    )
    env = _LevelCompleteEnv()

    metrics = _run_live_without_starting_routine(env, run_dir, client, game_id="fixture", max_actions=5)

    assert env.actions == [("ACTION1", {})]
    assert metrics["actions"] == 1
    assert metrics["review_count"] == 1
    assert client.response_formats == [None, "json_object", None]
    assert client.tool_definitions[0]
    assert client.tool_definitions[1] is None
    assert client.tool_definitions[2]
    assert "A cue is visible in this game." in client.calls[2][0]
    assert client.calls[2][0].count("A cue is visible in this game.") == 1
    assert "applicable_successful_strategies" not in client.calls[2][0]
    assert len(client.message_calls) == 2
    assert len(client.message_calls[0]["messages"]) == 2
    assert len(client.message_calls[1]["messages"]) == 2
    assert client.message_calls[1]["messages"][0] == client.message_calls[0]["messages"][0]
    assert "A cue is visible in this game." in client.message_calls[1]["messages"][1]["content"][0]["text"]
    view = (tmp_path / "state/games/fixture/MEMORY.md").read_text()
    assert view == _VALID_GAME_MEMORY + "\n"
    row = _review_rows(run_dir)[0]
    assert row["status"] == "saved"
    assert row["trigger_reasons"] == ["level_completed"]
    assert row["covered_transition_count"] == 1


def test_run_start_actions_and_state_recording_do_not_rewrite_saved_markdown(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    store_dir = game_memory_dir(state_dir, "fixture")
    authored = "# Game Memory\n\n- Keep this wording exactly as reviewed."
    save_game_memory(store_dir, authored)
    memory = GameMemory(tmp_path / "seed", store_dir=store_dir, run_id="seed", game_id="fixture")
    memory.record_run_state("fixture", {
        "level": 0, "levels_completed": 0, "game_state": "NOT_FINISHED", "terminal": False,
    }, step=0)
    client = _FakeClient([_decision([_action()])])
    run_live(
        _FakeEnv(), tmp_path / "run", client, game_id="fixture", max_actions=1,
        state_dir=state_dir, starting_routine=False,
    )
    assert (store_dir / "MEMORY.md").read_text() == authored + "\n"

def test_ez01_fixture_review_writes_model_markdown_without_state_or_evidence_noise(tmp_path: Path) -> None:
    fixture = json.loads((Path(__file__).parent / "fixtures/v10/ez01_memory_history.json").read_text())
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    history = History(run_dir)
    for row in fixture:
        history.append(row)
    tools = GameTools(_FakeEnv(), run_dir)
    game_id = "ez01"
    run_id = "ez01-fixture-run"
    memory_text = "# Game Memory\n\n- Repeated up completed the final level and produced WIN."
    client = _FakeClient([], review_responses=[json.dumps({
        "game_memory_markdown": memory_text,
        "global_memory_markdown": "# Global Memory\n\n- Compare objects across actions.",
    })])

    result = _perform_review(
        tools, client, game_id=game_id, run_id=run_id,
        memory_store_dir=tmp_path / "state/games/ez01", trigger_reasons=["win"],
    )

    assert result["row"]["status"] == "saved"
    saved = (tmp_path / "state/games/ez01/MEMORY.md").read_text()
    assert saved == memory_text + "\n"
    assert "T26" not in saved and "confidence" not in saved
    assert "Current progress belongs only in observation context" in client.calls[0][0]
    assert "An environment-confirmed level completion or WIN triggered this review" in client.calls[0][0]
    assert "145" in client.calls[0][0]  # raw evidence is review input, not saved prose


def test_equivalent_verified_routes_keep_level_scope_and_stay_out_of_review_prose(tmp_path: Path) -> None:
    from v11.tools.solutions import LevelSolutionStore

    state_root = tmp_path / "state"
    store_dir = game_memory_dir(state_root, "fixture")
    store = LevelSolutionStore(store_dir / "level_solutions.jsonl", "fixture")
    legal = [{"id": "ACTION1", "parameters": {
        "type": "object", "properties": {}, "required": [], "additionalProperties": False,
    }}]
    for level in (0, 1):
        start = {
            "frame": np.full((2, 2), level, dtype=np.int64), "legal_actions": legal,
            "game_state": "NOT_FINISHED", "level": level, "levels_completed": level,
            "terminal": False,
        }
        finish = {
            **start, "frame": np.full((2, 2), level + 10, dtype=np.int64),
            "level": level + 1, "levels_completed": level + 1,
        }
        edge = store.record_transition(start, "ACTION1", {}, finish, run_id=f"r{level}")
        store.record_successful_trajectory(
            start, [edge, {**edge, "source_state": edge["target_state"], "target_state": edge["target_state"]}],
            run_id=f"r{level}", success="level_completed",
    )
    game_text = "# Game Memory\n\n- The same approach completed levels zero and one from their matching starts."
    client = _FakeClient([], review_responses=[json.dumps({
        "game_memory_markdown": game_text,
        "global_memory_markdown": _VALID_GLOBAL_MEMORY,
    })])
    tools = GameTools(_FakeEnv(), tmp_path / "run")

    result = _perform_review(
        tools, client, game_id="fixture", run_id="review", memory_store_dir=store_dir,
        trigger_reasons=["level_completed"],
    )

    prompt = client.calls[0][0]
    assert result["row"]["verified_routes_supplied"] == 2
    assert '"level":0' in prompt and '"level":1' in prompt
    assert '"actions":["up","up"]' not in prompt
    assert '"step_count":2' not in prompt
    assert (store_dir / "MEMORY.md").read_text() == game_text + "\n"
    assert '["up","up"]' not in (store_dir / "MEMORY.md").read_text()


def test_review_route_summaries_omit_routes_but_store_retains_them_for_planning(tmp_path: Path) -> None:
    from v11.tools.solutions import LevelSolutionStore

    store = LevelSolutionStore(tmp_path / "level_solutions.jsonl", "fixture")
    start = {
        "frame": np.zeros((2, 2), dtype=np.int64),
        "legal_actions": [{"id": "ACTION1"}],
        "game_state": "NOT_FINISHED", "level": 2, "levels_completed": 2,
        "terminal": False,
    }
    finish = {
        **start, "frame": np.ones((2, 2), dtype=np.int64),
        "game_state": "WIN", "terminal": True,
    }
    edge = store.record_transition(start, "ACTION1", {}, finish, run_id="successful")
    store.record_successful_trajectory(start, [edge], run_id="successful", success="win")

    summaries = _verified_route_summaries(store)
    assert summaries == [{
        "scope": {"level": 2, "levels_completed": 2},
        "environment_recorded_success": "win",
    }]
    assert "actions" not in summaries[0]
    assert "step_count" not in summaries[0]

    replay = store.shortest_known_solution(start)
    assert replay is not None
    assert [step["action"] for step in replay] == ["ACTION1"]

def test_incremental_review_prompt_contains_only_uncovered_observations(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    tools = GameTools(_FakeEnv(), run_dir)
    tools.history.append({
        "type": "transition", "action": "ACTION1", "args": {}, "measurements": {"changed_pixels": 1},
        "level": 0, "levels_completed": 0, "game_state": "NOT_FINISHED", "terminal": False,
    })
    client = _FakeClient([], review_responses=[_VALID_REVIEW, _VALID_REVIEW])
    store_dir = tmp_path / "state/games/fixture"

    first = _perform_review(
        tools, client, game_id="fixture", run_id="run-one", memory_store_dir=store_dir,
        trigger_reasons=["test_review"],
    )
    assert first["row"]["status"] == "saved"
    tools.history.append({
        "type": "transition", "action": "ACTION2", "args": {}, "measurements": {"changed_pixels": 2},
        "level": 0, "levels_completed": 0, "game_state": "NOT_FINISHED", "terminal": False,
    })

    second = _perform_review(
        tools, client, game_id="fixture", run_id="run-one", memory_store_dir=store_dir,
        trigger_reasons=["test_review"],
    )

    assert second["row"]["status"] == "saved"
    assert '"changed_pixels":2' in client.calls[1][0]
    assert '"changed_pixels":1' not in client.calls[1][0]
    assert "No level completion or WIN triggered this review" in client.calls[1][0]


def test_review_repairs_empty_markdown_then_checkpoints(tmp_path: Path) -> None:
    tools = GameTools(_FakeEnv(), tmp_path / "review-run")
    tools.observe()
    tools.act("ACTION1", {})
    invalid = json.dumps({"game_memory_markdown": "  ", "global_memory_markdown": _VALID_GLOBAL_MEMORY})
    client = _FakeClient([], review_responses=[invalid, _VALID_REVIEW])

    result = _perform_review(
        tools, client, game_id="fixture", run_id="review-repair",
        memory_store_dir=tmp_path / "state/games/fixture", trigger_reasons=["level_completed"],
    )

    assert len(client.calls) == 2
    assert "prior review response was invalid" in client.calls[1][0]
    assert result["row"]["status"] == "saved"
    assert result["row"]["request_attempts"] == 2
    assert result["row"]["review_checkpoint_saved"] is True
    memory = GameMemory(
        tmp_path / "review-run", store_dir=tmp_path / "state/games/fixture",
        run_id="review-repair", game_id="fixture",
    )
    assert memory.view_path.read_text() == _VALID_GAME_MEMORY + "\n"
    assert memory.reviewed_event_count("fixture") == 2


def test_review_repairs_transient_progress_and_global_blockage_overclaim(tmp_path: Path) -> None:
    tools = GameTools(_FakeEnv(), tmp_path / "quality-review")
    tools.observe()
    invalid = json.dumps({
        "game_memory_markdown": (
            "## Game status\n- All levels completed; game state is WIN.\n\n"
            "## Level 2 progress\n- Four yellows collected at actions 10 and 20. "
            "The player recently moved left.\n- A rightward action produced zero pixel change, indicating a wall blocks movement."
        ),
        "global_memory_markdown": "ignored during migration",
    })
    client = _FakeClient([], review_responses=[invalid, _VALID_REVIEW])
    store_dir = tmp_path / "state/games/fixture"

    result = _perform_review(
        tools, client, game_id="fixture", run_id="quality-review",
        memory_store_dir=store_dir, trigger_reasons=["test_review"],
    )

    assert result["row"]["status"] == "saved"
    assert result["row"]["request_attempts"] == 2
    repair_prompt = client.calls[1][0]
    assert "transient status/progress section" in repair_prompt
    assert "run-specific completion status" in repair_prompt
    assert "action-numbered progress" in repair_prompt
    assert "transient item count" in repair_prompt
    assert "Describe no-visible-change only for that exact action and state" in repair_prompt
    assert "current collected/remaining counts" in repair_prompt
    assert (store_dir / "MEMORY.md").read_text() == _VALID_GAME_MEMORY + "\n"
    assert not (tmp_path / "state/global/MEMORY.md").exists()



def test_action_set_review_uses_strict_nine_action_threshold(tmp_path: Path) -> None:
    client = _FakeClient([
        _decision([_action()] * 8),
        _decision([_action()]),
        _decision([_action()] * 4),
        _decision([_action()]),
    ])

    metrics = _run_live_without_starting_routine(
        _FakeEnv(), tmp_path / "run", client, game_id="fixture",
        max_actions=14, plan_horizon=8,
    )

    assert metrics["actions"] == 14
    assert metrics["review_action_threshold"] == 9
    rows = _review_rows(tmp_path / "run")
    assert metrics["review_count"] == 1
    assert rows[0]["through_action"] == 13
    assert rows[0]["trigger_reasons"] == ["action_set_interval"]


def test_reviews_recur_at_action_set_ends_after_more_than_nine_steps(tmp_path: Path) -> None:
    client = _FakeClient([], default_decision=_decision([_action()] * 4))
    metrics = _run_live_without_starting_routine(
        _FakeEnv(), tmp_path / "run", client, game_id="fixture", max_actions=45,
    )

    assert metrics["actions"] == 45
    rows = _review_rows(tmp_path / "run")
    assert metrics["review_count"] == 3
    assert [row["through_action"] for row in rows] == [12, 24, 36]
    assert all(row["trigger_reasons"] == ["action_set_interval"] for row in rows)


def test_active_level_review_attaches_every_intermediate_screen_in_order(tmp_path: Path) -> None:
    class IntermediateFrameEnv(_FakeEnv):
        def act(self, action, args):
            super().act(action, args)
            intermediate = self.frame.copy()
            intermediate[2, 3] = 99 + len(self.actions)
            return {"frames": [intermediate]}

    run_dir = tmp_path / "all-review-screens"
    tools = GameTools(IntermediateFrameEnv(), run_dir)
    tools.observe()
    tools.act("ACTION1", {})
    tools.act("ACTION2", {})
    client = _FakeClient([])

    result = _perform_review(
        tools, client, game_id="fixture", run_id="all-screens",
        memory_store_dir=tmp_path / "state/games/fixture",
        trigger_reasons=["test_review"],
    )

    assert result["row"]["status"] == "saved"
    _prompt, image_paths, image_labels = client.calls[0]
    assert len(image_paths) == 5  # Initial, intermediate/after for both actions; current is the last screen.
    assert len(image_labels) == 5
    assert "initial state before Action 1 (up)" in image_labels[0]
    assert "intermediate 1/1 during Action 1 (up)" in image_labels[1]
    assert "result of Action 1 (up)" in image_labels[2]
    assert "intermediate 1/1 during Action 2 (down)" in image_labels[3]
    assert "result of Action 2 (down)" in image_labels[4] and "[CURRENT SCREEN]" in image_labels[4]


def test_level_completion_review_attaches_the_complete_solved_level(tmp_path: Path) -> None:
    class CompleteOnTenthAction(_FakeEnv):
        def observe(self):
            result = super().observe()
            if len(self.actions) >= 10:
                result["level"] = 1
                result["levels_completed"] = 1
            return result

    tools = GameTools(CompleteOnTenthAction(), tmp_path / "complete-level-screens")
    tools.observe()
    for index in range(10):
        tools.act("ACTION1" if index % 2 == 0 else "ACTION2", {})
    client = _FakeClient([])

    result = _perform_review(
        tools, client, game_id="fixture", run_id="complete-level",
        memory_store_dir=tmp_path / "state/games/fixture",
        trigger_reasons=["level_completed"],
    )

    assert result["row"]["status"] == "saved"
    _prompt, image_paths, image_labels = client.calls[0]
    assert len(image_paths) == 11  # Starting screen, every post-action screen, current screen deduplicated.
    assert "initial state before Action 1 (up)" in image_labels[0]
    assert "result of Action 10 (down)" in image_labels[-1] and "[CURRENT SCREEN]" in image_labels[-1]
    history_text = client.calls[0][0].split(
        "Complete ordered action history for the level that just completed. "
        "Use this full sequence to identify its demonstrated strategy and observed outcomes. "
        "Screens are numbered chronologically. The full action history below names each source screen and every screen produced. "
        "Attachments contain at most the latest 20 screens, in chronological order; older screens may appear in the history without an attached image. "
        "The current screen is included when available. The next action starts from the preceding action's final produced screen.\n", 1,
    )[1].split("\n\nReview constraint:", 1)[0]
    history = json.loads(history_text)
    assert len(history) == 10
    assert [row["sequence_index"] for row in history] == list(range(1, 11))
    assert history[0]["starts_from_screen_number"] == 1
    assert history[0]["produces_screen_numbers"] == [2]
    assert history[1]["starts_from_screen_number"] == 2
    assert history[-1]["starts_from_screen_number"] == 10


def test_model_selected_reset_restarts_startup_bootstrap_flow(tmp_path: Path) -> None:
    class FourDirections(_FakeEnv):
        def observe(self):
            result = super().observe()
            result["legal_actions"].extend([
                {"id": action, "parameters": {
                    "type": "object", "properties": {}, "required": [],
                    "additionalProperties": False,
                }}
                for action in ("ACTION3", "ACTION4")
            ])
            return result

    env = FourDirections()
    client = _FakeClient([
        _decision([_action()]),  # Initial bootstrap-informed plan.
        _decision(reset=True),  # Choose to restart after seeing an ineffective attempt.
        _decision([_action()]),  # Plan from the post-reset bootstrap screens.
    ])

    metrics = run_live(
        env, tmp_path / "reset-bootstrap", client,
        game_id="fixture", max_actions=11, plan_horizon=8,
    )

    assert metrics["resets"] == 3  # Initial bootstrap, requested reset, second bootstrap reset.
    assert env.actions == [(action, {}) for action in (
        "ACTION1", "ACTION3", "ACTION2", "ACTION4", "ACTION1",
        "ACTION1", "ACTION3", "ACTION2", "ACTION4", "ACTION1",
    )]
    bootstrap_prompts = [prompt for prompt, _paths, _labels in client.calls if "bootstrap" in prompt.casefold()]
    assert len(bootstrap_prompts) == 2
    assert "Bootstrap action outcomes" in bootstrap_prompts[1]
    messages = [call["messages"] for call in client.message_calls]
    assert messages[2][:-1] == [*messages[1], {
        "role": "assistant", "content": client.response_contents[1],
    }]
    reset_bootstrap_turn = messages[2][-1]
    assert len([block for block in reset_bootstrap_turn["content"] if block.get("type") == "image_url"]) >= 5
    labels = [block["text"] for block in reset_bootstrap_turn["content"] if block.get("type") == "text"]
    assert all(any(f"Action {number} (" in label for label in labels) for number in range(1, 5))
    assert "CURRENT SCREEN" in labels[-1]


def test_budget_expiration_does_not_trigger_review_or_assessment(tmp_path: Path) -> None:
    client = _FakeClient(
        [_decision([_action()] * 4)],
        review_responses=[_VALID_REVIEW],
    )

    metrics = _run_live_without_starting_routine(
        _FakeEnv(), tmp_path / "run", client, game_id="fixture", max_actions=2,
    )

    assert metrics["actions"] == 2
    assert metrics["review_count"] == 0
    assert len(client.calls) == 1
    plan_rows = [
        json.loads(line) for line in (tmp_path / "run/decisions.jsonl").read_text().splitlines()
        if json.loads(line).get("schema") == "v10.live_plan_result/2"
    ]
    assert plan_rows[-1]["interrupt_reason"] == "action_budget_exhausted"
    assert _review_rows(tmp_path / "run") == []


def test_level_completion_triggers_review(tmp_path: Path) -> None:
    client = _FakeClient(
        [_decision([_action()] * 4)], review_responses=[_VALID_REVIEW],
    )
    metrics = _run_live_without_starting_routine(
        _LevelCompleteEnv(), tmp_path / "run", client, game_id="fixture",
        max_actions=1,
    )

    rows = _review_rows(tmp_path / "run")
    assert metrics["review_count"] == 1
    assert len(client.response_formats) == 2
    assert rows[0]["trigger_reasons"] == ["level_completed"]


def test_interval_review_can_precede_later_level_completion_review(tmp_path: Path) -> None:
    class CompletesAtTwenty(_FakeEnv):
        def observe(self):
            result = super().observe()
            completed = int(len(self.actions) >= 20)
            result["levels_completed"] = completed
            result["level"] = completed
            return result

    client = _FakeClient([], default_decision=_decision([_action()] * 4))
    metrics = _run_live_without_starting_routine(
        CompletesAtTwenty(), tmp_path / "run", client, game_id="fixture",
        max_actions=21,
    )

    assert metrics["actions"] == 21
    rows = _review_rows(tmp_path / "run")
    assert len(rows) == 2
    assert [row["through_action"] for row in rows] == [12, 20]
    assert [row["trigger_reasons"] for row in rows] == [
        ["action_set_interval"], ["level_completed"],
    ]


def test_completion_review_wins_when_completion_crosses_interval_threshold(
    tmp_path: Path,
) -> None:
    class CompletesOnTenthAction(_FakeEnv):
        def observe(self):
            result = super().observe()
            if len(self.actions) >= 10:
                result["levels_completed"] = 1
                result["level"] = 1
            return result

    client = _FakeClient([], default_decision=_decision([_action()] * 4))
    metrics = _run_live_without_starting_routine(
        CompletesOnTenthAction(), tmp_path / "run", client, game_id="fixture",
        max_actions=11, plan_horizon=4,
    )

    rows = _review_rows(tmp_path / "run")
    assert metrics["review_count"] == 1
    assert rows[0]["through_action"] == 10
    assert rows[0]["trigger_reasons"] == ["level_completed"]


def test_failed_level_completion_review_preserves_memory_and_continues(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    store_dir = game_memory_dir(state_dir, "fixture")
    before = "# Game Memory\n\n- Prior durable finding."
    save_game_memory(store_dir, before)
    client = _FakeClient(
        [_decision([_action()] * 4), _decision([_action()] * 2)],
        review_errors=[RuntimeError("review endpoint unavailable")],
    )

    metrics = _run_live_without_starting_routine(
        _LevelCompleteEnv(), tmp_path / "run", client, game_id="fixture", max_actions=2,
        state_dir=state_dir,
    )

    assert metrics["actions"] == 2
    assert metrics["review_count"] == 2
    assert [row["status"] for row in _review_rows(tmp_path / "run")] == ["failed", "saved"]
    assert "review endpoint unavailable" in _review_rows(tmp_path / "run")[0]["error"]
    assert (store_dir / "MEMORY.md").read_text() == _VALID_GAME_MEMORY + "\n"
    assert "Prior durable finding." in client.calls[1][0]

def test_malformed_review_preserves_existing_memory(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    store_dir = game_memory_dir(state_dir, "fixture")
    before = "# Game Memory\n\n- Prior durable finding."
    save_game_memory(store_dir, before)
    client = _FakeClient(
        [_decision([_action()])],
        review_responses=["not json", "still not json"],
    )

    metrics = _run_live_without_starting_routine(
        _LevelCompleteEnv(), tmp_path / "run", client, game_id="fixture",
        max_actions=1, state_dir=state_dir,
    )

    assert metrics["review_count"] == 1
    row = _review_rows(tmp_path / "run")[0]
    assert row["status"] == "failed"
    assert "not valid JSON" in row["error"]
    assert row["review_checkpoint_saved"] is False
    assert len(row["repair_history"]) == 2
    assert (store_dir / "MEMORY.md").read_text() == before + "\n"

def test_terminal_level_completion_is_reviewed_without_resuming_play(tmp_path: Path) -> None:
    client = _FakeClient(
        [_decision([_action()] * 4)], review_responses=[_VALID_REVIEW],
    )
    env = _LevelCompleteEnv(terminal=True, start_completed=4)

    metrics = _run_live_without_starting_routine(env, tmp_path / "run", client, game_id="fixture", max_actions=10)

    assert env.actions == [("ACTION1", {})]
    assert env.resets == 0
    assert metrics["stop_reason"] == "win"
    assert metrics["review_count"] == 1
    assert metrics["review_count"] == 1
    assert len(client.calls) == 2
    assert client.system_prompts[-1].startswith("# V11 game memory and profile candidate review")
    assert _review_rows(tmp_path / "run")[0]["status"] == "saved"
    assert _review_rows(tmp_path / "run")[0]["trigger_reasons"] == ["level_completed"]


def test_game_over_with_budget_remaining_resets_and_repeats_startup_bootstrap(
    tmp_path: Path,
) -> None:
    class FourDirections(_FakeEnv):
        def __init__(self):
            super().__init__()
            self.actions_since_reset = 0
            self.did_game_over = False
            self.game_over = False

        def observe(self):
            result = super().observe()
            result["legal_actions"].extend([
                {"id": action, "parameters": {
                    "type": "object", "properties": {}, "required": [],
                    "additionalProperties": False,
                }}
                for action in ("ACTION3", "ACTION4")
            ])
            result["game_state"] = "GAME_OVER" if self.game_over else "NOT_FINISHED"
            result["terminal"] = self.game_over
            return result

        def act(self, action, args):
            super().act(action, args)
            self.actions_since_reset += 1
            if self.resets == 1 and self.actions_since_reset == 1 and not self.did_game_over:
                self.game_over = True
                self.did_game_over = True
            return {}

        def reset(self):
            self.resets += 1
            self.actions_since_reset = 0
            self.game_over = False
            self.frame[:] = 0
            return self.observe()

    env = FourDirections()
    client = _FakeClient([
        _decision([_action("left")]),  # First attempt ends in GAME_OVER.
        _decision([_action("up")]),    # Bootstrap-informed plan after retry bootstrap.
        _decision([_action("down")]),
    ], post_mortem_responses=[
        "## What happened\nThe last left action ended the attempt.\n\n"
        "## Choices to change\nDo not repeat the ineffective path.\n\n"
        "## Retry experiment\nTest one different action and inspect the result."
    ])

    metrics = run_live(env, tmp_path / "game-over-retry", client, game_id="fixture", max_actions=11)

    assert metrics["actions"] == 11
    assert metrics["resets"] == 3  # Initial bootstrap, game-over restart, retry bootstrap.
    assert metrics["game_over_restarts"] == 1
    assert metrics["stop_reason"] == "action_budget_exhausted"
    assert env.did_game_over is True
    assert [action for action, _ in env.actions] == [
        "ACTION1", "ACTION3", "ACTION2", "ACTION4", "ACTION3",
        "ACTION1", "ACTION3", "ACTION2", "ACTION4", "ACTION1", "ACTION2",
    ]

    records = [
        json.loads(line)
        for line in (tmp_path / "game-over-retry" / "decisions.jsonl").read_text().splitlines()
    ]
    restarts = [row for row in records if row.get("schema") == "v10.live_game_over_restart/1"]
    bootstrap_resets = [row for row in records if row.get("schema") == "v10.live_bootstrap_reset/1"]
    assert len(restarts) == 1
    assert restarts[0]["actions_remaining"] == 6
    assert restarts[0]["bootstrap_pending"] is True
    assert len(bootstrap_resets) == 2
    assert _review_rows(tmp_path / "game-over-retry") == []
    assert metrics["review_count"] == 0
    post_mortem_rows = [
        json.loads(line)
        for line in (tmp_path / "game-over-retry" / "post_mortems.jsonl").read_text().splitlines()
    ]
    assert post_mortem_rows[0]["status"] == "saved"
    assert post_mortem_rows[0]["action_sequence"] == [{
        "action_number": 1, "action": "left", "level_index": 0,
        "levels_completed": 0, "resulting_game_state": "GAME_OVER",
        "starts_from_screen_number": 1, "produces_screen_numbers": [2],
        "args": {}, "measurements": {"changed_pixels": 1, "color_change_count": 1},
    }]

    bootstrap_prompts = [
        (prompt, paths, labels)
        for prompt, paths, labels in client.calls
        if "bootstrap observations below" in prompt or any("Bootstrap Screen" in label for label in labels)
    ]
    assert len(bootstrap_prompts) == 2
    retry_prompt, retry_images, retry_labels = bootstrap_prompts[1]
    assert "Bootstrap action outcomes" in retry_prompt
    assert "restored CURRENT SCREEN" in retry_prompt
    assert "New post-mortem findings" in retry_prompt
    assert "Do not repeat the ineffective path" in retry_prompt
    assert len(retry_images) >= 5
    assert "CURRENT SCREEN" in retry_labels[-1]


def test_post_mortem_sends_every_action_and_screen_in_order(tmp_path: Path) -> None:
    class FourDirections(_FakeEnv):
        def observe(self):
            result = super().observe()
            result["legal_actions"].extend([
                {"id": action, "parameters": {
                    "type": "object", "properties": {}, "required": [],
                    "additionalProperties": False,
                }}
                for action in ("ACTION3", "ACTION4")
            ])
            return result

        def act(self, action, args):
            super().act(action, args)
            intermediate = self.frame.copy()
            intermediate[2, 3] = 99 + len(self.actions)
            return {"frames": [intermediate]}

    tools = GameTools(FourDirections(), tmp_path / "post-mortem-images")
    tools.observe()
    for index in range(12):
        tools.act(f"ACTION{index % 4 + 1}", {})
    client = _FakeClient([], post_mortem_responses=["The sequence shows no supported cause."])

    result = _perform_game_over_post_mortem(
        tools, client, game_id="fixture", guidance_memory="", profile_memory="",
    )
    assert client.tool_definitions == [None]

    prompt, paths, labels = client.calls[0]
    assert result["row"]["status"] == "saved"
    assert len(result["row"]["action_sequence"]) == 12
    assert len(paths) == len(labels) == result["row"]["image_count"] == 25
    expected_labels = ["Post-mortem Screen 1: initial state before Action 1 (up)"]
    directions = ("up", "down", "left", "right")
    for index in range(12):
        direction = directions[index % len(directions)]
        expected_labels.extend([
            f"Post-mortem Screen {index * 2 + 2}: intermediate 1/1 during Action {index + 1} ({direction}); "
            f"produced by Action {index + 1} ({direction}) from Screen {index * 2 + 1}",
            f"Post-mortem Screen {index * 2 + 3}: result of Action {index + 1} ({direction}); "
            f"produced by Action {index + 1} ({direction}) from Screen {index * 2 + 1}",
        ])
    expected_labels[-1] += " [CURRENT SCREEN]"
    assert labels == expected_labels
    action_json = prompt.split("Complete active-level action sequence:\n", 1)[1].split("\n\nGame MEMORY.md", 1)[0]
    assert len(json.loads(action_json)) == 12


def test_game_over_at_action_budget_does_not_restart(tmp_path: Path) -> None:
    class TerminalAfterOneAction(_FakeEnv):
        def __init__(self):
            super().__init__()
            self.game_over = False

        def observe(self):
            result = super().observe()
            result["game_state"] = "GAME_OVER" if self.game_over else "NOT_FINISHED"
            result["terminal"] = self.game_over
            return result

        def act(self, action, args):
            super().act(action, args)
            self.game_over = True
            return {}

    env = TerminalAfterOneAction()
    metrics = _run_live_without_starting_routine(
        env, tmp_path / "game-over-budget", _FakeClient([_decision([_action()])]),
        game_id="fixture", max_actions=1,
    )

    assert metrics["actions"] == 1
    assert metrics["resets"] == 0
    assert metrics["game_over_restarts"] == 0
    assert metrics["stop_reason"] == "game_over"
    assert _review_rows(tmp_path / "game-over-budget") == []
    assert metrics["review_count"] == 0


def test_win_without_level_counter_change_is_treated_as_level_completion(tmp_path: Path) -> None:
    class WinWithoutLevelChange(_FakeEnv):
        def observe(self):
            result = super().observe()
            result["terminal"] = bool(self.actions)
            result["game_state"] = "WIN" if result["terminal"] else "NOT_FINISHED"
            return result

    client = _FakeClient([_decision([_action()])], review_responses=[_VALID_REVIEW])
    metrics = _run_live_without_starting_routine(WinWithoutLevelChange(), tmp_path / "run", client, game_id="fixture", max_actions=2)

    assert metrics["stop_reason"] == "win"
    assert _review_rows(tmp_path / "run")[0]["trigger_reasons"] == ["level_completed"]






def test_stagnation_never_triggers_a_review(tmp_path: Path) -> None:
    client = _FakeClient(
        [], default_decision=_decision([_action()] * 4),
    )
    metrics = _run_live_without_starting_routine(
        _OscillatingEnv(), tmp_path / "run", client,
        game_id="fixture", max_actions=8,
    )

    assert metrics["review_count"] == 0
    assert _review_rows(tmp_path / "run") == []


def test_repeated_state_cycle_interrupts_remaining_model_plan(tmp_path: Path) -> None:
    client = _FakeClient([_decision([_action()] * 8)])
    metrics = _run_live_without_starting_routine(
        _OscillatingEnv(), tmp_path / "cycle-interruption", client,
        game_id="fixture", max_actions=3, plan_horizon=8,
    )

    rows = [json.loads(line) for line in (tmp_path / "cycle-interruption/decisions.jsonl").read_text().splitlines()]
    interrupted = next(row for row in rows if row.get("interrupt_reason") == "repeated_state_stagnation")
    assert len(interrupted["executed"]) == 3
    assert metrics["actions"] == 3
    assert metrics["interrupted_plans"] == 1


def test_changed_transition_resets_stagnation_streak(tmp_path: Path) -> None:
    class OneChangedAction(_FakeEnv):
        def __init__(self):
            super().__init__(change_every_action=False)

        def observe(self):
            result = super().observe()
            result["legal_actions"].append({
                "id": "ACTION3", "parameters": {
                    "type": "object", "properties": {}, "required": [],
                    "additionalProperties": False,
                },
            })
            return result

        def act(self, action, args):
            super().act(action, args)
            if action == "ACTION3":
                self.frame[0, 0] = 9
            return {}

    client = _FakeClient(
        [
            _decision([_action("up")]),
            _decision([_action("down")]),
            _decision([_action("left")]),
            _decision([_action("up")]),
            _decision([_action("down")]),
        ],
        review_responses=[_VALID_REVIEW],
    )
    metrics = _run_live_without_starting_routine(
        OneChangedAction(), tmp_path / "run", client,
        game_id="fixture", max_actions=5, plan_horizon=4,
    )

    assert metrics["review_count"] == 0
    after_change = _continued_context(_message_user_text(client, 3))
    assert after_change["state"]["consecutive_repeated_or_no_progress_actions"] == 0


def test_stale_stagnation_states_from_previous_level_do_not_retrigger(tmp_path: Path) -> None:
    class NoopsThenNewLevel(_FakeEnv):
        def __init__(self):
            super().__init__(change_every_action=False)
            self.level = 0
            self.levels_completed = 0

        def observe(self):
            result = super().observe()
            result["level"] = self.level
            result["levels_completed"] = self.levels_completed
            return result

        def act(self, action, args):
            self.actions.append((action, dict(args)))
            self.frame[:] = 0 if int(self.frame[0, 0]) else 1
            if len(self.actions) == 3:
                self.level = 1
                self.levels_completed = 1
                self.frame[0, 0] = 3
            return {}

    client = _FakeClient(
        [
            _decision([_action("up"), _action("up"), _action("up")]),
            _decision([_action("up")] * 3),
        ],
        review_responses=[_VALID_REVIEW],
    )
    metrics = _run_live_without_starting_routine(
        NoopsThenNewLevel(), tmp_path / "run", client,
        game_id="fixture", max_actions=6, plan_horizon=4,
    )

    assert metrics["review_count"] == 1
    assert [row["trigger_reasons"] for row in _review_rows(tmp_path / "run")] == [["level_completed"]]
    assert _review_rows(tmp_path / "run")[0]["through_action"] == 3



def test_live_direction_words_map_to_ids_and_logs_use_direction_names(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    class FourActionEnv(_FakeEnv):
        def observe(self):
            result = super().observe()
            result["legal_actions"] += [
                {"id": "ACTION3", "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}},
                {"id": "ACTION4", "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}},
            ]
            return result

    env = FourActionEnv()
    directions = _decision([_action("right"), _action("left"), _action("down"), _action("up")])
    client = _FakeClient([directions])

    metrics = _run_live_without_starting_routine(env, tmp_path / "run", client, game_id="fixture", max_actions=4)

    assert env.actions == [
        ("ACTION4", {}), ("ACTION3", {}), ("ACTION2", {}), ("ACTION1", {}),
    ]
    prompt = client.calls[0][0]
    for direction in ("up", "down", "left", "right"):
        assert f'"id":"{direction}"' in prompt
    assert "ACTION1" not in prompt and "ACTION2" not in prompt
    rows = [json.loads(line) for line in (tmp_path / "run" / "decisions.jsonl").read_text().splitlines()]
    accepted = next(row for row in rows if row.get("status") == "accepted")
    assert [item["action"] for item in accepted["decision"]["actions"]] == [
        "right", "left", "down", "up",
    ]
    plan_result = next(row for row in rows if row.get("schema") == "v10.live_plan_result/2")
    assert [item["action"] for item in plan_result["executed"]] == [
        "right", "left", "down", "up",
    ]
    transitions = [row for row in History(tmp_path / "run").records() if row.get("type") == "transition"]
    assert [row["action"] for row in transitions] == [
        "right", "left", "down", "up",
    ]
    output = capsys.readouterr().err
    for direction in ("up", "down", "left", "right"):
        assert direction in output
    assert "planned=['ACTION" not in output
    assert "taking action 1: ACTION4" not in output
    assert metrics["actions"] == 4


class _RegressionWinningEnv(_FakeEnv):
    def __init__(self, actions_to_win=1):
        super().__init__()
        self.actions_to_win = actions_to_win
        self.attempt_actions = 0

    def observe(self):
        result = super().observe()
        won = self.attempt_actions >= self.actions_to_win
        result.update(game_state="WIN" if won else "NOT_FINISHED", terminal=won,
                      level=int(won), levels_completed=int(won))
        return result

    def act(self, action, args):
        super().act(action, args)
        self.attempt_actions += 1
        return {}

    def reset(self):
        super().reset()
        self.attempt_actions = 0
        return self.observe()


def test_native_adapter_preserves_temporal_frames_and_validates_click_coordinates(tmp_path, monkeypatch):
    from v11.analysis import validate_model_decision, DecisionValidationError

    class Native:
        def reset(self):
            return types.SimpleNamespace(frame=np.zeros((1, 8, 8), dtype=np.int64),
                available_actions=[6], state="NOT_FINISHED", levels_completed=0, win_levels=1)

        def step(self, action, data, reasoning):
            self.data = data
            return types.SimpleNamespace(frame=np.stack([np.full((8, 8), n) for n in (1, 2, 3)]),
                available_actions=[6], state="NOT_FINISHED", levels_completed=0, win_levels=1)

    class Actions:
        @staticmethod
        def from_id(value):
            return types.SimpleNamespace(name=f"ACTION{value}", value=value)

    monkeypatch.setitem(sys.modules, "arcengine", types.SimpleNamespace(GameAction=Actions))
    native = Native()
    adapter = ArcadeGameEnv("fixture", env_factory=lambda *args, **kwargs: native)
    tools = GameTools(adapter, tmp_path / "native")
    start = tools.observe()
    decision = validate_model_decision({"actions": [{"action": "click", "args": {"x": 7, "y": 4}}]}, start["legal_actions"])
    with pytest.raises(DecisionValidationError, match="maximum"):
        validate_model_decision({"actions": [{"action": "click", "args": {"x": 8, "y": 0}}]}, start["legal_actions"])
    result = tools.act(**{"action": decision["actions"][0]["action"], "args": decision["actions"][0]["args"]})
    assert native.data["x"] == 7
    assert len(result["result"]["intermediate_frame_ids"]) == 2
    assert [int(tools.frames.get(frame)[0, 0]) for frame in result["result"]["intermediate_frame_ids"]] == [1, 2]
    assert int(result["observation"]["frame"][0, 0]) == 3
    inputs = _live_context(tools, result["observation"], "fixture", 8, shared_skill_path(),
                           tmp_path / "state/games/fixture", "r", 5)
    assert len(inputs["image_paths"]) == 4  # Initial, two intermediate, and current result screens.
    assert "CURRENT SCREEN" in inputs["image_labels"][-1]


def test_bootstrap_win_is_preserved_and_saved_without_reset(tmp_path):
    env = _RegressionWinningEnv()
    metrics = run_live(env, tmp_path / "probe-win", _FakeClient([]), game_id="fixture", max_actions=1)
    assert metrics["won"] is True
    assert metrics["stop_reason"] == "win"
    assert metrics["available_level_solutions"] == 1
    assert env.resets == 0


def test_live_winning_log_with_bootstrap_reset_imports_directions_idempotently(tmp_path):
    from v11.tools.solutions import LevelSolutionStore
    env = _RegressionWinningEnv(actions_to_win=3)
    client = _FakeClient([_decision([_action()] * 3)])
    run_dir = tmp_path / "winning-log"
    metrics = run_live(env, run_dir, client, game_id="fixture", max_actions=5)
    assert metrics["won"]
    assert env.resets == 1
    store = LevelSolutionStore(tmp_path / "recovered.jsonl", "fixture")
    assert len(store.import_successful_run(run_dir)) == 1
    assert [step["action"] for step in store.solutions()[0]["steps"]] == ["ACTION1"] * 3
    assert store.import_successful_run(run_dir) == []
    assert len(store.solutions()) == 1


def test_diverged_route_at_same_screen_is_suspended_and_model_runs(tmp_path):
    from v11.tools.solutions import LevelSolutionStore
    state = tmp_path / "state"
    env = _FakeEnv(change_every_action=False)
    start = env.observe()
    target = {**start, "frame": np.ones_like(start["frame"]), "level": 1, "levels_completed": 1}
    store = LevelSolutionStore(game_memory_dir(state, "fixture") / "level_solutions.jsonl", "fixture")
    edge = store.record_transition(start, "ACTION1", {}, target, run_id="old")
    store.record_successful_trajectory(start, [edge], run_id="old", success="level_completed")
    client = _FakeClient([_decision([_action("down")])])
    metrics = run_live(env, tmp_path / "diverged", client, game_id="fixture", max_actions=2,
                       state_dir=state, starting_routine=False)
    assert metrics["cached_solution_actions"] == 1
    assert metrics["model_calls"] == 1
    assert [action for action, _ in env.actions] == ["ACTION1", "ACTION2"]


def test_failed_click_does_not_block_other_coordinates(tmp_path, capsys):
    class ClickEnv(_FakeEnv):
        def observe(self):
            result = super().observe()
            result["legal_actions"] = [{"id": "ACTION6", "parameters": {
                "type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                "required": ["x", "y"], "additionalProperties": False}}]
            return result

        def act(self, action, args):
            self.actions.append((action, args))
            if args["x"]:
                self.frame[0, 0] = 1
            return {}

    client = _FakeClient([_decision([{"action": "click", "args": {"x": x, "y": 0}}]) for x in (0, 1)])
    env = ClickEnv()
    metrics = run_live(env, tmp_path / "clicks", client, game_id="fixture", max_actions=2, starting_routine=False)
    assert [args["x"] for _, args in env.actions] == [0, 1]
    assert metrics["model_calls"] == 2
    assert "Refer to an attempted click as `click(x=12,y=8)`" in client.system_prompts[0]
    assert "the legal action name `click`" in client.system_prompts[0]
    assert "ACTION6" not in client.system_prompts[0]
    assert "ACTION6" not in client.calls[0][0]
    assert '"id":"click"' in client.calls[0][0]
    context = _continued_context(_message_user_text(client, 1))
    assert context["state"]["ineffective_attempts_at_current_screen"] == [
        {"action": "click(x=0,y=0)"}
    ]
    output = capsys.readouterr().err
    assert "planned=['click(x=0,y=0)']" in output
    assert "taking action 1: click(x=0,y=0)" in output
    assert "action 1: click(x=0,y=0) ->" in output
    assert "ACTION6" not in output
    evidence = [row for row in History(tmp_path / "clicks").records() if row.get("type") == "transition"]
    assert evidence[0]["action"] == "ACTION6"
    assert evidence[0]["args"] == {"x": 0, "y": 0}
    assert evidence[0]["display_action"] == "click(x=0,y=0)"
    accepted = next(
        row for row in (json.loads(line) for line in (tmp_path / "clicks" / "decisions.jsonl").read_text().splitlines())
        if row.get("status") == "accepted"
    )
    assert accepted["decision"]["actions"] == [{"action": "click", "args": {"x": 0, "y": 0}}]
    assert accepted["display_actions"] == ["click(x=0,y=0)"]


def test_reset_memory_cli_does_not_reimport_historical_route_or_ledger(tmp_path, monkeypatch):
    from v11 import cli, arcade_env
    from v11.runlog import write_json
    root = tmp_path / "state"
    old = tmp_path / "old-run"
    tools = GameTools(_RegressionWinningEnv(), old)
    tools.observe()
    tools.act("ACTION1")  # Old canonical evidence remains importable.
    write_json(old / "run_manifest.json", {"game_id": "fixture", "run_id": "old"})
    write_json(old / "metrics.json", {"won": True, "actions": 1, "levels_completed": 1})
    (old / "memory.jsonl").write_text(json.dumps({"id": "old-marker", "type": "mechanic", "claim": "stale"}) + "\n")
    save_game_memory(game_memory_dir(root, "fixture"), "Old memory")
    env = _RegressionWinningEnv()
    env.close = lambda: None
    monkeypatch.setattr(arcade_env, "ArcadeGameEnv", lambda *args, **kwargs: env)
    monkeypatch.setattr(cli, "StrataClient", lambda **kwargs: _FakeClient([]))
    assert cli.main(["live", "--game-id", "fixture", "--run-dir", str(old), "--state-dir", str(root),
                     "--max-actions", "1", "--reset-memory"]) == 0
    metrics = json.loads((old / "metrics.json").read_text())
    manifest = json.loads((old / "run_manifest.json").read_text())
    assert metrics["cached_solution_actions"] == 0
    assert manifest["configuration"]["import_historical_routes"] is False
    assert "old-marker" not in (game_memory_dir(root, "fixture") / "memory.jsonl").read_text()


def test_bootstrap_restoration_rejects_different_start_screen(tmp_path):
    from v11.tools.solutions import LevelSolutionStore
    class DifferentReturn(_FakeEnv):
        def __init__(self):
            super().__init__()
            self.level = 0

        def observe(self):
            result = super().observe()
            result.update(level=self.level, levels_completed=self.level)
            return result

        def act(self, action, args):
            self.actions.append((action, args))
            if self.level == 0:
                self.level = 1
                self.frame[:] = 20 if self.resets == 0 else 10
            else:
                self.frame[0, 0] += 1
            return {}

        def reset(self):
            super().reset()
            self.level = 0
            return self.observe()

    env = DifferentReturn()
    state = tmp_path / "state"
    start = env.observe()
    target = {**start, "frame": np.full_like(start["frame"], 10), "level": 1, "levels_completed": 1}
    store = LevelSolutionStore(game_memory_dir(state, "fixture") / "level_solutions.jsonl", "fixture")
    edge = store.record_transition(start, "ACTION1", {}, target, run_id="old")
    store.record_successful_trajectory(start, [edge], run_id="old", success="level_completed")
    client = _FakeClient([_decision([])])
    metrics = run_live(env, tmp_path / "mismatch", client, game_id="fixture", max_actions=8, state_dir=state)
    action_prompts = [prompt for prompt, _, _ in client.calls if "Visible context:\n" in prompt]
    assert len(action_prompts) == 1
    assert "bootstrap observations below" not in action_prompts[0]
    context = json.loads(action_prompts[0].split("Visible context:\n", 1)[1])
    assert "bootstrap_observations" not in context
    assert context["current_state"]["level"] == 1
    assert metrics["stop_reason"] == "model_returned_no_actions"


def test_review_archives_previous_and_new_prose_and_manifest_records_settings(tmp_path):
    state = tmp_path / "state"
    game = game_memory_dir(state, "fixture")
    save_game_memory(game, "Previous game lesson")
    profile_path = profile_memory_path(state, "generic")
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text("Previous reusable method\n", encoding="utf-8")
    client = _FakeClient([_decision([_action()])])
    client.max_output_tokens = 4096
    client.reasoning_effort = "medium"
    client.timeout_seconds = 300
    run = tmp_path / "archive"
    metrics = run_live(_RegressionWinningEnv(), run, client, game_id="fixture", max_actions=1,
                       starting_routine=False, state_dir=state)
    assert metrics["won"]
    rows = [row for row in GameMemory(run, store_dir=game).ledger() if row.get("op") == "markdown"]
    assert [row["markdown"] for row in rows if row["scope"] == "game"] == ["Previous game lesson\n", _VALID_GAME_MEMORY + "\n"]
    assert not [row for row in rows if row["scope"] == "profile"]
    assert profile_path.read_text(encoding="utf-8") == "Previous reusable method\n"
    manifest = json.loads((run / "run_manifest.json").read_text())
    assert manifest["configuration"]["max_output_tokens"] == 4096
    assert manifest["configuration"]["reasoning_effort"] == "medium"
    assert manifest["starting_memory_sha256"]["game"] == next(row["sha256"] for row in rows if row["scope"] == "game")
    profile_requests = [json.loads(line) for line in (run / "profile_requests.jsonl").read_text().splitlines()]
    assert {row["phase"] for row in profile_requests} == {"action_selection", "memory_review"}
    assert all(row["profile_id"] == "generic" for row in profile_requests)
    assert all(row["argument_schema_fingerprint"] for row in profile_requests)


def test_run_directory_cannot_clear_persistent_state(tmp_path):
    from v11.live import _prepare_live_run_directory
    root = tmp_path / "skills"
    root.mkdir()
    sentinel = root / "SKILL.md"
    sentinel.write_text("keep")
    with pytest.raises(ValueError, match="overlaps"):
        _prepare_live_run_directory(root, (root, sentinel))
    assert sentinel.read_text() == "keep"


def test_single_action_experiment_can_discover_hidden_effect_after_repeated_input(tmp_path):
    class DelayedWin(_RegressionWinningEnv):
        def act(self, action, args):
            self.actions.append((action, args))
            self.attempt_actions += 1
            return {}  # Only the third input changes terminal/progress metadata.

    env = DelayedWin(actions_to_win=3)
    client = _FakeClient([_decision([_action()])] * 3)
    metrics = run_live(env, tmp_path / "hidden-effect", client, game_id="fixture",
                       max_actions=3, starting_routine=False)
    assert metrics["won"]
    assert metrics["actions"] == 3
    assert env.resets == 0


def test_click_capable_startup_uses_one_model_guided_native_click_per_turn(tmp_path: Path) -> None:
    class ClickBootstrapEnv(_FakeEnv):
        def observe(self):
            result = super().observe()
            result["legal_actions"].append({
                "id": "ACTION6", "parameters": {
                    "type": "object", "properties": {
                        "x": {"type": "integer", "minimum": 0, "maximum": 3},
                        "y": {"type": "integer", "minimum": 0, "maximum": 2},
                    }, "required": ["x", "y"], "additionalProperties": False,
                },
            })
            return result

        def act(self, action, args):
            self.actions.append((action, dict(args)))
            self.frame[args["y"], args["x"]] += 1
            return {}

    env = ClickBootstrapEnv()
    client = _FakeClient([
        _decision([{"action": "click", "args": {"x": 2, "y": 1}}]),
        _decision([{"action": "click", "args": {"x": 1, "y": 1}}]),
    ])
    metrics = run_live(env, tmp_path / "click-bootstrap", client, game_id="fixture", max_actions=2)

    assert env.actions == [("ACTION6", {"x": 2, "y": 1}), ("ACTION6", {"x": 1, "y": 1})]
    assert metrics["click_actions_executed"] == 2
    assert metrics["movement_actions_executed"] == 0
    assert metrics["bootstrap_actions_by_type"]["click"] == 2
    assert env.resets == 0  # The four-action probe cap was not reached.
    first_context_text = client.calls[0][0]
    assert '"coordinate_system":"native_zero_based_xy"' in first_context_text
    assert "click_candidates" in first_context_text
    assert "choose an informative legal action" in first_context_text.lower()


def test_game_and_profile_memory_persist_in_separate_scopes(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    game_text = "# Game Memory\n\n- A fixture strategy works on level zero."
    profile_text = "# Approved methods\n\n- Compare each outcome after one action."
    save_game_memory(game_memory_dir(state_dir, "ez01"), game_text)
    profile_path = profile_memory_path(state_dir, "generic")
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text(profile_text + "\n", encoding="utf-8")
    client = _FakeClient([_decision([])])
    _run_live_without_starting_routine(
        _FakeEnv(), tmp_path / "ez01-next", client, game_id="ez01", max_actions=1, state_dir=state_dir,
    )
    prompt = client.calls[0][0]
    assert "A fixture strategy works on level zero." in prompt
    assert "Compare each outcome after one action." in prompt
    assert (game_memory_dir(state_dir, "ez01") / "MEMORY.md").read_text() == game_text + "\n"
    assert profile_path.read_text() == profile_text + "\n"
