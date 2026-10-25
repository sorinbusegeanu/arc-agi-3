from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from v10.arcade_env import ArcadeGameEnv
from v10.live import run_live
from v10.tools.game import GameTools
from v10.tools.game import ActionError
from v10.tools.history import History


@pytest.fixture(autouse=True)
def _isolated_memory_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("V10_STATE_DIR", str(tmp_path / "state"))


def _decision(actions=None, *, reset: bool = False) -> dict:
    if reset:
        return {"actions": [{"action": "RESET", "args": {"justification": "A controlled restart is required."}}]}
    return {"actions": list(actions or [])}


def _action(action="ACTION1"):
    return {"action": action, "args": {}}


class _FakeClient:
    model = "mock-qwen"

    def __init__(self, decisions, *, request_attempts=1):
        self.decisions = list(decisions)
        self.calls = []
        self.request_attempts = request_attempts

    def complete(self, system_prompt, user_prompt, image_paths, image_labels):
        self.calls.append((user_prompt, list(image_paths), list(image_labels)))
        decision = self.decisions.pop(0)
        return {
            "content": json.dumps(decision), "model": self.model,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "latency_seconds": 0.01, "request_attempts": self.request_attempts,
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


def test_live_executes_action_plan_without_prediction_interrupts(tmp_path: Path) -> None:
    env = _FakeEnv()
    first = _decision([_action(), _action("ACTION2")])
    second = _decision([])
    client = _FakeClient([first, second])

    metrics = run_live(env, tmp_path / "run", client, game_id="fixture", max_actions=5)

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

    metrics = run_live(
        _FakeEnv(), run_dir, _FakeClient([_decision([])]), game_id="fixture", max_actions=3,
    )

    assert not stale_dir.exists()
    assert (run_dir / "metrics.json").exists()
    assert metrics["actions"] == 0


def test_live_executes_reset_action_only_when_budgeted(tmp_path: Path) -> None:
    env = _FakeEnv()
    client = _FakeClient([_decision(reset=True), _decision([])])

    metrics = run_live(env, tmp_path / "run", client, game_id="fixture", max_actions=3, max_resets=1)

    assert env.resets == 1
    assert metrics["stop_reason"] == "model_returned_no_actions"
    assert metrics["actions"] == 0


def test_live_metrics_include_strata_json_recovery_attempt(tmp_path: Path) -> None:
    metrics = run_live(
        _FakeEnv(), tmp_path / "run", _FakeClient([_decision([])], request_attempts=2),
        game_id="fixture", max_actions=3,
    )

    assert metrics["model_calls"] == 2
    assert metrics["stop_reason"] == "model_returned_no_actions"


def test_live_action_only_responses_create_no_prediction_audits(tmp_path: Path) -> None:
    client = _FakeClient([_decision([_action(), _action("ACTION2")]), _decision([])])
    env = _FakeEnv()

    metrics = run_live(
        env, tmp_path / "run", client, game_id="fixture", max_actions=2, plan_horizon=2,
    )

    assert metrics["actions"] == 2
    prediction_rows = [
        row for row in History(tmp_path / "run").records()
        if row.get("type") == "audit" and row.get("kind") == "prediction"
    ]
    assert prediction_rows == []


def test_live_repairs_response_that_includes_prediction_fields(tmp_path: Path) -> None:
    env = _FakeEnv()
    wrapped = _decision([_action()])
    wrapped["actions"][0]["measurable_prediction"] = {"changed_pixels": 1}
    client = _FakeClient([wrapped, _decision([_action()])])

    metrics = run_live(env, tmp_path / "run", client, game_id="fixture", max_actions=1)

    assert env.actions == [("ACTION1", {})]
    assert metrics["actions"] == 1
    rows = [json.loads(line) for line in (tmp_path / "run" / "decisions.jsonl").read_text().splitlines()]
    accepted = next(row for row in rows if row.get("status") == "accepted")
    assert accepted["decision"]["actions"] == [{"action": "ACTION1", "args": {}}]
    assert accepted["repair_attempted"] is True


def test_live_repairs_response_with_extra_fields(tmp_path: Path) -> None:
    invalid = _decision([_action()])
    invalid["actions"][0]["measurable_prediction"] = {"pixel_diff": {"invented": 1}}
    client = _FakeClient([invalid, _decision([_action()]), _decision([])])

    metrics = run_live(_FakeEnv(), tmp_path / "run", client, game_id="fixture", max_actions=1)

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

    metrics = run_live(
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


@pytest.mark.parametrize("repair_action", ["ACTION1", "NOT_LEGAL"])
def test_live_rejected_duplicate_candidate_cannot_execute_when_repair_fails(
    tmp_path: Path, repair_action: str,
) -> None:
    duplicate = _decision([_action("ACTION1"), _action("ACTION2")])
    duplicate["actions"][1]["prediction_id"] = duplicate["actions"][0]["prediction_id"] = "P_DUP"
    repair = _decision([_action(repair_action), _action("ACTION2")])
    if repair_action == "ACTION1":
        repair["actions"][1]["prediction_id"] = repair["actions"][0]["prediction_id"] = "P_DUP"
    env = _FakeEnv()
    metrics = run_live(
        env, tmp_path / "run", _FakeClient([duplicate, repair]),
        game_id="fixture", max_actions=1,
    )
    rows = [json.loads(line) for line in (tmp_path / "run" / "decisions.jsonl").read_text().splitlines()]
    rejected = next(row for row in rows if row.get("status") == "rejected")
    fallback = next(row for row in rows if row.get("schema") == "v10.live_fallback/2" and row.get("status") == "executed")
    expected_error = (
        "actions[0] may contain only action and args" if repair_action == "ACTION1"
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

    metrics = run_live(_FakeEnv(), tmp_path / "run", client, game_id="fixture", max_actions=2)

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

    metrics = run_live(env, tmp_path / "run", client, game_id="fixture", max_actions=3)

    assert metrics["stop_reason"] == "model_returned_no_actions"
    prompt, image_paths, labels = client.calls[0]
    assert "current_evidence_id" not in prompt
    assert "initial_state_evidence_id" not in prompt
    assert "measurable_prediction" not in prompt
    assert "memory_patch" not in prompt
    assert "up to 4 legal actions" in prompt
    assert len(image_paths) == 1
    assert labels == ["Current actual game frame"]
