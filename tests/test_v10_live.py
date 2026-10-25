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
from v10.tools.history import History


@pytest.fixture(autouse=True)
def _isolated_memory_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("V10_STATE_DIR", str(tmp_path / "state"))


def _decision(actions=None, *, reset: bool = False) -> dict:
    return {
        "observations": [], "patterns": [], "hypotheses": [], "prediction_audit": [], "goals": [],
        "priority_question": "What changes after the next legal action?",
        "decision": {
            "mode": "explore", "continue_or_reset": "reset" if reset else "continue",
            "reset_justification": "A controlled restart is required." if reset else None,
            "actions": list(actions or []),
        },
        "memory_patch": [],
    }


def _action(action="ACTION1", prediction="P1", changed_pixels=1):
    return {
        "action": action, "args": {}, "prediction_id": prediction,
        "expected_observation": "The frame changes.",
        "measurable_prediction": {"changed_pixels": changed_pixels}, "stop_if": [],
    }


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


def test_live_loop_interrupts_remaining_actions_after_prediction_failure(tmp_path: Path) -> None:
    env = _FakeEnv()
    first = _decision([_action(changed_pixels=0), _action("ACTION2", "P2")])
    second = _decision([])
    client = _FakeClient([first, second])

    metrics = run_live(env, tmp_path / "run", client, game_id="fixture", max_actions=5)

    assert env.actions == [("ACTION1", {})]
    assert metrics["actions"] == 1
    assert metrics["stop_reason"] == "model_returned_no_actions"
    rows = [json.loads(line) for line in (tmp_path / "run" / "decisions.jsonl").read_text().splitlines()]
    plan_results = [row for row in rows if row["schema"] == "v10.live_plan_result/1"]
    assert plan_results[0]["interrupt_reason"] == "prediction_mismatch"
    transitions = [row for row in History(tmp_path / "run").records() if row.get("type") == "transition"]
    assert len(transitions) == 1
    assert transitions[0]["prediction_evidence_id"]
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


def test_live_loop_rejects_reset_when_reset_budget_is_zero(tmp_path: Path) -> None:
    env = _FakeEnv()
    client = _FakeClient([_decision(reset=True)])

    metrics = run_live(env, tmp_path / "run", client, game_id="fixture", max_actions=3, max_resets=0)

    assert env.resets == 0
    assert metrics["stop_reason"] == "reset_budget_exhausted"
    assert metrics["actions"] == 0


def test_live_metrics_include_strata_json_recovery_attempt(tmp_path: Path) -> None:
    metrics = run_live(
        _FakeEnv(), tmp_path / "run", _FakeClient([_decision([])], request_attempts=2),
        game_id="fixture", max_actions=3,
    )

    assert metrics["model_calls"] == 2
    assert metrics["stop_reason"] == "model_returned_no_actions"


def test_live_smoke_accepts_wrapped_pixel_diff_prediction(tmp_path: Path) -> None:
    env = _FakeEnv()
    wrapped = _decision([_action()])
    wrapped["decision"]["actions"][0]["measurable_prediction"] = {
        "pixel_diff": {"changed_pixels": 1},
    }
    client = _FakeClient([wrapped, _decision([])])

    metrics = run_live(env, tmp_path / "run", client, game_id="fixture", max_actions=3)

    assert env.actions == [("ACTION1", {})]
    assert metrics["actions"] == 1
    rows = [json.loads(line) for line in (tmp_path / "run" / "decisions.jsonl").read_text().splitlines()]
    accepted = next(row for row in rows if row.get("status") == "accepted")
    assert accepted["decision"]["decision"]["actions"][0]["measurable_prediction"] == {
        "all": [{"path": "measurements.changed_pixels", "op": "eq", "value": 1}],
    }


def test_live_repairs_invalid_measurement_schema_with_targeted_guidance(tmp_path: Path) -> None:
    invalid = _decision([_action()])
    invalid["decision"]["actions"][0]["measurable_prediction"] = {"pixel_diff": {"invented": 1}}
    client = _FakeClient([invalid, _decision([_action()]), _decision([])])

    metrics = run_live(_FakeEnv(), tmp_path / "run", client, game_id="fixture", max_actions=1)

    assert metrics["actions"] == 1
    assert len(client.calls) == 2
    repair_prompt = client.calls[1][0]
    assert "Your previous JSON failed validation" in repair_prompt
    assert "measurements.changed_pixels" in repair_prompt
    assert "not citable evidence" in repair_prompt
    rows = [json.loads(line) for line in (tmp_path / "run" / "decisions.jsonl").read_text().splitlines()]
    accepted = next(row for row in rows if row.get("status") == "accepted")
    assert accepted["repair_attempted"] is True
    assert "unsupported" in accepted["initial_validation_error"]


def test_live_executes_random_fallback_after_invalid_decision_and_repair(tmp_path: Path) -> None:
    invalid = _decision([_action()])
    invalid["decision"]["actions"][0]["measurable_prediction"] = {"pixel_diff": {"invented": 1}}
    still_invalid = _decision([_action()])
    still_invalid["decision"]["actions"][0]["measurable_prediction"] = {"pixel_diff": {"invented": 1}}
    env = _FakeEnv()

    metrics = run_live(
        env, tmp_path / "run", _FakeClient([invalid, still_invalid]), game_id="fixture", max_actions=1,
    )

    assert metrics["actions"] == 1
    assert env.actions[0][0] in {"ACTION1", "ACTION2"}
    assert metrics["stop_reason"] == "action_budget_exhausted"
    rows = [json.loads(line) for line in (tmp_path / "run" / "decisions.jsonl").read_text().splitlines()]
    fallback = next(row for row in rows if row.get("schema") == "v10.live_fallback/1")
    assert fallback["status"] == "executed"
    assert fallback["cause"] == "invalid_decision"
    assert fallback["prediction_verdict"] == "inconclusive"


def test_live_resumes_model_loop_after_invalid_decision_fallback(tmp_path: Path) -> None:
    invalid = _decision([_action()])
    invalid["decision"]["actions"][0]["measurable_prediction"] = {"pixel_diff": {"invented": 1}}
    still_invalid = _decision([_action()])
    still_invalid["decision"]["actions"][0]["measurable_prediction"] = {"pixel_diff": {"invented": 1}}
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
    assert '"current_evidence_id":"T1"' in prompt
    assert len(image_paths) == 2
    assert labels == ["Initial state", "Current state"]
