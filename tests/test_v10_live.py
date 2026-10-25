from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from v10.arcade_env import ArcadeGameEnv
from v10.live import _live_context, _perform_review, _review_frame_selection, run_live
from v10.tools.game import GameTools
from v10.tools.game import ActionError
from v10.tools.history import History
from v10.tools.memory import (
    GameMemory, game_memory_dir, global_memory_path, save_game_memory, shared_skill_path,
)


@pytest.fixture(autouse=True)
def _isolated_memory_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("V10_STATE_DIR", str(tmp_path / "state"))


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


_VALID_GAME_MEMORY = "# Game Memory\n\n- A cue is visible in this game."
_VALID_GLOBAL_MEMORY = "# Global Memory\n\n- Compare each object before and after an action."
_VALID_REVIEW = json.dumps({
    "game_memory_markdown": _VALID_GAME_MEMORY,
    "global_memory_markdown": _VALID_GLOBAL_MEMORY,
})


class _FakeClient:
    model = "mock-qwen"

    def __init__(
        self, decisions, *, request_attempts=1, review_responses=None,
        review_errors=None, default_decision=None,
    ):
        self.decisions = list(decisions)
        self.calls = []
        self.response_formats = []
        self.request_attempts = request_attempts
        self.review_responses = list(review_responses or [])
        self.review_errors = list(review_errors or [])
        self.default_decision = default_decision

    def complete(self, system_prompt, user_prompt, image_paths, image_labels, response_format="json_object"):
        self.calls.append((user_prompt, list(image_paths), list(image_labels)))
        self.response_formats.append(response_format)
        if system_prompt.startswith("# V10 memory consolidation"):
            if self.review_errors:
                raise self.review_errors.pop(0)
            content = self.review_responses.pop(0) if self.review_responses else _VALID_REVIEW
        else:
            decision = self.decisions.pop(0) if self.decisions else self.default_decision
            if decision is None:
                decision = _decision([])
            content = json.dumps(decision)
        return {
            "content": content, "model": self.model,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "latency_seconds": 0.01, "request_attempts": self.request_attempts,
            "finish_reason": "stop",
        }


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
    assert "must start from this initial screen" in prompt
    assert len(images) == 3
    assert labels == [
        "Bootstrap initial frame before reset", "Bootstrap final frame before reset",
        "Current actual game frame",
    ]
    decisions = [
        json.loads(line) for line in (tmp_path / "run/decisions.jsonl").read_text().splitlines()
    ]
    reset = next(row for row in decisions if row.get("schema") == "v10.live_bootstrap_reset/1")
    assert reset["status"] == "reset"
    plans = [row for row in decisions if row.get("schema") == "v10.live_decision/2"]
    assert plans[0]["source"] == "bootstrap_probe"
    assert plans[-1]["source"] == "bootstrap_model_start"


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


def test_invalid_memory_startup_answer_falls_back_to_standard_routine(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    _seed_game_mechanic(state_dir)
    client = _FakeClient([_decision([]), _decision([])])
    env = _FakeEnv()

    run_live(env, tmp_path / "run", client, game_id="fixture", max_actions=4, state_dir=state_dir)

    assert [action for action, _ in env.actions] == ["ACTION1", "ACTION2", "ACTION1", "ACTION2"]
    assert env.resets == 1
    rows = [json.loads(line) for line in (tmp_path / "run/decisions.jsonl").read_text().splitlines()]
    plans = [row for row in rows if row.get("schema") == "v10.live_decision/2"]
    assert plans[-1]["source"] == "bootstrap_start_fallback"
    assert plans[-1]["initial_validation_error"] == "bootstrap plan must contain legal lowercase directional actions"


def test_recent_history_is_independent_of_plan_horizon(tmp_path: Path) -> None:
    client = _FakeClient([_decision([_action()]), _decision([])])
    _run_live_without_starting_routine(_FakeEnv(), tmp_path / "run", client, game_id="fixture", max_actions=2, plan_horizon=1)

    context = json.loads(client.calls[1][0].split("Visible context:\n", 1)[1])
    assert context["budgets"]["plan_horizon"] == 1
    assert len(context["recent_transitions"]) == 1


def test_stagnation_reviews_require_fresh_actions_before_retrigger(tmp_path: Path) -> None:
    client = _FakeClient(
        [], default_decision=_decision([_action()] * 4),
        review_responses=[_VALID_REVIEW, _VALID_REVIEW],
    )
    metrics = _run_live_without_starting_routine(
        _FakeEnv(change_every_action=False), tmp_path / "run", client,
        game_id="fixture", max_actions=7, review_interval=50,
    )

    rows = _review_rows(tmp_path / "run")
    assert metrics["review_count"] == 2
    assert [row["through_action"] for row in rows] == [3, 6]


def test_stagnation_retriggers_after_failed_review_and_fresh_actions(tmp_path: Path) -> None:
    client = _FakeClient(
        [], default_decision=_decision([_action()] * 4),
        review_errors=[RuntimeError("temporary reviewer failure")],
        review_responses=[_VALID_REVIEW],
    )
    metrics = _run_live_without_starting_routine(
        _FakeEnv(change_every_action=False), tmp_path / "run", client,
        game_id="fixture", max_actions=6, review_interval=50,
    )
    rows = _review_rows(tmp_path / "run")
    assert metrics["review_count"] == 2
    assert [row["status"] for row in rows] == ["failed", "saved"]
    assert [row["through_action"] for row in rows] == [3, 6]


def test_review_image_selection_includes_movement_blocked_interaction_and_progress(tmp_path: Path) -> None:
    from v10.frames import FrameStore

    frames = FrameStore(tmp_path)
    rows = []
    for index in range(4):
        before = frames.put(np.full((2, 2), index, dtype=np.int64), source="test")
        after = frames.put(np.full((2, 2), index + 10, dtype=np.int64), source="test")
        before_image = frames.put_image(np.zeros((2, 2, 3), dtype=np.uint8), f"B{index}", kind="image")
        after_image = frames.put_image(np.ones((2, 2, 3), dtype=np.uint8), f"A{index}", kind="image")
        rows.append({
            "pre_frame_id": before, "post_frame_id": after,
            "pre_frame_image_id": before_image, "post_frame_image_id": after_image,
            "levels_completed": 1 if index == 3 else 0, "terminal": False,
            "measurements": {
                "changed_pixels": [10, 0, 8, 20][index],
                "color_transitions": ([{}, {}, {}] if index == 2 else []),
            },
        })
    _, labels = _review_frame_selection(rows, frames, max_images=8)
    selected = " ".join(labels)
    for index in range(1, 5):
        assert f"transition {index}" in selected


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


def test_live_executes_reset_action_only_when_budgeted(tmp_path: Path) -> None:
    env = _FakeEnv()
    client = _FakeClient([_decision(reset=True), _decision([])])

    metrics = _run_live_without_starting_routine(env, tmp_path / "run", client, game_id="fixture", max_actions=3, max_resets=1)

    assert env.resets == 1
    assert metrics["stop_reason"] == "model_returned_no_actions"
    assert metrics["actions"] == 0


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
    assert _review_rows(tmp_path / "run")[0]["trigger_reasons"] == ["unexpected_transition"]
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
    assert "Your previous JSON failed validation" in repair_prompt
    assert "predictions" in repair_prompt
    assert '"actions"' in repair_prompt
    assert "memory updates" in repair_prompt
    rows = [json.loads(line) for line in (tmp_path / "run" / "decisions.jsonl").read_text().splitlines()]
    accepted = next(row for row in rows if row.get("status") == "accepted")
    assert accepted["repair_attempted"] is True
    assert "only action and args" in accepted["initial_validation_error"]


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
    assert "current_evidence_id" not in prompt
    assert "initial_state_evidence_id" not in prompt
    assert "measurable_prediction" not in prompt
    assert "memory_patch" not in prompt
    assert "up to 4 legal actions" in prompt
    assert '"id":"up"' in prompt
    assert '"id":"down"' in prompt
    assert "ACTION1" not in prompt
    assert len(image_paths) == 1
    assert labels == ["Current actual game frame"]


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
    assert client.response_formats == ["json_object", "json_object", "json_object"]
    assert "A cue is visible in this game." in client.calls[2][0]
    assert client.calls[2][0].count("A cue is visible in this game.") == 1
    assert "applicable_successful_strategies" not in client.calls[2][0]
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
    assert "145" in client.calls[0][0]  # raw evidence is review input, not saved prose


def test_equivalent_verified_routes_keep_level_scope_and_stay_out_of_markdown(tmp_path: Path) -> None:
    from v10.tools.solutions import LevelSolutionStore

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
    assert '"actions":["up","up"]' in prompt
    assert (store_dir / "MEMORY.md").read_text() == game_text + "\n"
    assert '["up","up"]' not in (store_dir / "MEMORY.md").read_text()

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
        trigger_reasons=["action_interval"],
    )
    assert first["row"]["status"] == "saved"
    tools.history.append({
        "type": "transition", "action": "ACTION2", "args": {}, "measurements": {"changed_pixels": 2},
        "level": 0, "levels_completed": 0, "game_state": "NOT_FINISHED", "terminal": False,
    })

    second = _perform_review(
        tools, client, game_id="fixture", run_id="run-one", memory_store_dir=store_dir,
        trigger_reasons=["action_interval"],
    )

    assert second["row"]["status"] == "saved"
    assert '"changed_pixels":2' in client.calls[1][0]
    assert '"changed_pixels":1' not in client.calls[1][0]


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


@pytest.mark.parametrize(
    ("bad_payload", "game_limit", "global_limit"),
    [
        ('{"game_memory_markdown":" ","global_memory_markdown":"# Global Memory"}', 6000, 3000),
        ('{"game_memory_markdown":"' + "g" * 40 + '","global_memory_markdown":"# Global Memory"}', 32, 3000),
        ('{"game_memory_markdown":"# Game Memory","global_memory_markdown":"' + "g" * 40 + '"}', 6000, 32),
    ],
)
def test_rejected_markdown_review_preserves_both_previous_files(
    tmp_path: Path, bad_payload: str, game_limit: int, global_limit: int,
) -> None:
    state_dir = tmp_path / "state"
    store_dir = game_memory_dir(state_dir, "fixture")
    previous_game = "# Game Memory\n\n- Keep prior game guidance."
    previous_global = "# Global Memory\n\n- Keep prior global guidance."
    save_game_memory(store_dir, previous_game)
    from v10.tools.memory import save_global_memory
    save_global_memory(state_dir, previous_global)
    client = _FakeClient([], review_responses=[bad_payload, bad_payload])
    result = _perform_review(
        GameTools(_FakeEnv(), tmp_path / "run"), client, game_id="fixture", run_id="r1",
        memory_store_dir=store_dir, trigger_reasons=["action_interval"],
        game_memory_max_chars=game_limit, global_memory_max_chars=global_limit,
    )
    assert result["row"]["status"] == "failed"
    assert (store_dir / "MEMORY.md").read_text() == previous_game + "\n"
    assert global_memory_path(state_dir).read_text() == previous_global + "\n"
    assert result["row"]["review_checkpoint_saved"] is False

def test_review_interval_defaults_to_50_and_runs_after_fiftieth_action(tmp_path: Path) -> None:
    client = _FakeClient(
        [], default_decision=_decision([_action()] * 4),
        review_responses=[_VALID_REVIEW],
    )

    metrics = _run_live_without_starting_routine(_FakeEnv(), tmp_path / "run", client, game_id="fixture", max_actions=50)

    assert metrics["actions"] == 50
    assert metrics["review_interval"] == 50
    assert metrics["review_count"] == 1
    review_row = _review_rows(tmp_path / "run")[0]
    assert review_row["trigger_reasons"] == ["action_interval"]
    assert review_row["covered_transition_count"] == 50
    assert len(review_row["selected_image_labels"]) <= 8


def test_configured_review_interval_interrupts_remaining_plan(tmp_path: Path) -> None:
    client = _FakeClient(
        [_decision([_action()] * 4)],
        review_responses=[_VALID_REVIEW],
    )

    metrics = _run_live_without_starting_routine(
        _FakeEnv(), tmp_path / "run", client, game_id="fixture", max_actions=2,
        review_interval=2,
    )

    assert metrics["actions"] == 2
    assert metrics["review_count"] == 1
    assert len(client.calls) == 2
    plan_rows = [
        json.loads(line) for line in (tmp_path / "run/decisions.jsonl").read_text().splitlines()
        if json.loads(line).get("schema") == "v10.live_plan_result/2"
    ]
    assert plan_rows[-1]["interrupt_reason"] == "review"
    assert _review_rows(tmp_path / "run")[0]["trigger_reasons"] == ["action_interval"]


def test_simultaneous_review_triggers_are_coalesced(tmp_path: Path) -> None:
    client = _FakeClient(
        [_decision([_action()] * 4)], review_responses=[_VALID_REVIEW],
    )
    metrics = _run_live_without_starting_routine(
        _LevelCompleteEnv(), tmp_path / "run", client, game_id="fixture",
        max_actions=1, review_interval=1,
    )

    rows = _review_rows(tmp_path / "run")
    assert metrics["review_count"] == 1
    assert len(client.response_formats) == 2
    assert rows[0]["trigger_reasons"] == ["level_completed", "action_interval"]


def test_failed_review_preserves_memory_resets_interval_and_continues(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    store_dir = game_memory_dir(state_dir, "fixture")
    before = "# Game Memory\n\n- Prior durable finding."
    save_game_memory(store_dir, before)
    client = _FakeClient(
        [_decision([_action()] * 4), _decision([_action()] * 2)],
        review_errors=[RuntimeError("review endpoint unavailable")],
    )

    metrics = _run_live_without_starting_routine(
        _FakeEnv(), tmp_path / "run", client, game_id="fixture", max_actions=3,
        review_interval=2, state_dir=state_dir,
    )

    assert metrics["actions"] == 3
    assert metrics["review_count"] == 1
    assert len(_review_rows(tmp_path / "run")) == 1
    assert _review_rows(tmp_path / "run")[0]["status"] == "failed"
    assert "review endpoint unavailable" in _review_rows(tmp_path / "run")[0]["error"]
    assert (store_dir / "MEMORY.md").read_text() == before + "\n"
    assert "Prior durable finding." in client.calls[2][0]

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
    assert len(client.calls) == 2
    assert _review_rows(tmp_path / "run")[0]["status"] == "saved"
    assert _review_rows(tmp_path / "run")[0]["trigger_reasons"] == ["level_completed", "win"]


def test_win_without_level_counter_change_triggers_review(tmp_path: Path) -> None:
    class WinWithoutLevelChange(_FakeEnv):
        def observe(self):
            result = super().observe()
            result["terminal"] = bool(self.actions)
            result["game_state"] = "WIN" if result["terminal"] else "NOT_FINISHED"
            return result

    client = _FakeClient([_decision([_action()])], review_responses=[_VALID_REVIEW])
    metrics = _run_live_without_starting_routine(WinWithoutLevelChange(), tmp_path / "run", client, game_id="fixture", max_actions=2)

    assert metrics["stop_reason"] == "win"
    assert _review_rows(tmp_path / "run")[0]["trigger_reasons"] == ["win"]


def test_three_consecutive_unchanged_actions_trigger_stagnation_review(tmp_path: Path) -> None:
    client = _FakeClient(
        [_decision([_action()] * 4), _decision([])], review_responses=[_VALID_REVIEW],
    )
    metrics = _run_live_without_starting_routine(
        _FakeEnv(change_every_action=False), tmp_path / "run", client,
        game_id="fixture", max_actions=4, review_interval=50,
    )

    assert metrics["review_count"] == 1
    assert _review_rows(tmp_path / "run")[0]["trigger_reasons"] == ["stagnation"]


def test_changed_transition_resets_stagnation_streak(tmp_path: Path) -> None:
    class OneChangedAction(_FakeEnv):
        def __init__(self):
            super().__init__(change_every_action=False)

        def act(self, action, args):
            super().act(action, args)
            if action == "ACTION2":
                self.frame[0, 0] = 9
            return {}

    client = _FakeClient(
        [
            _decision([_action("up"), _action("down"), _action("up"), _action("up")]),
            _decision([_action("up")]),
        ],
        review_responses=[_VALID_REVIEW],
    )
    metrics = _run_live_without_starting_routine(
        OneChangedAction(), tmp_path / "run", client,
        game_id="fixture", max_actions=5, plan_horizon=4, review_interval=50,
    )

    assert metrics["review_count"] == 1
    assert _review_rows(tmp_path / "run")[0]["through_action"] == 3
    assert _review_rows(tmp_path / "run")[0]["trigger_reasons"] == ["stagnation"]


def test_global_and_game_markdown_persist_across_sessions_without_duplicates(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    reusable_lesson = "Track each object's motion separately when several move at once."
    game_text = "# Game Memory\n\n- A fixture strategy works on level zero."
    global_text = f"# Global Memory\n\n- {reusable_lesson}"
    review = json.dumps({
        "game_memory_markdown": game_text,
        "global_memory_markdown": global_text,
    })
    first_client = _FakeClient([_decision([_action()])], review_responses=[review])
    first_metrics = _run_live_without_starting_routine(
        _LevelCompleteEnv(), tmp_path / "ez01-run", first_client,
        game_id="ez01", max_actions=1, state_dir=state_dir,
    )
    second_client = _FakeClient([_decision([])])
    different_start = _FakeEnv()
    different_start.frame[:] = 7
    # A new run uses the already persisted Markdown verbatim in its first prompt.
    second_metrics = _run_live_without_starting_routine(
        different_start, tmp_path / "ez01-next", second_client,
        game_id="ez01", max_actions=1, state_dir=state_dir,
    )

    assert first_metrics["review_count"] == 1
    assert second_metrics["game_id"] == "ez01"
    assert "A fixture strategy works on level zero." in second_client.calls[0][0]
    assert reusable_lesson in second_client.calls[0][0]
    assert second_client.calls[0][0].count(reusable_lesson) == 1
    assert (game_memory_dir(state_dir, "ez01") / "MEMORY.md").read_text() == game_text + "\n"
    assert global_memory_path(state_dir).read_text() == global_text + "\n"

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
