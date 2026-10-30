from __future__ import annotations

import json

import numpy as np
import pytest

from v11.live import run_live
from v11.tools.discovery import Discovery
from v11.tools.game import ActionBudget, GameTools


class Env:
    def __init__(self):
        self.frame = np.zeros((8, 8), dtype=np.int64)
        self.actions = []

    def observe(self):
        return {"frame": self.frame.copy(), "legal_actions": ["up", "right"], "level": 1,
                "levels_completed": 0, "terminal": False, "game_state": "PLAYING"}

    def act(self, action, args):
        self.actions.append(action)
        self.frame[2, 2] += 1
        return {}

    def fork(self):
        return Env()


class CanonicalMultiLevelEnv:
    ACTIONS = [
        {"id": f"ACTION{index}", "parameters": {"type": "object", "properties": {},
                                                   "required": [], "additionalProperties": False}}
        for index in range(1, 5)
    ]

    def __init__(self):
        self.level = 0
        self.levels_completed = 0
        self.frame = np.zeros((8, 8), dtype=np.int64)

    def observe(self):
        return {"frame": self.frame.copy(), "legal_actions": self.ACTIONS, "level": self.level,
                "levels_completed": self.levels_completed, "terminal": False, "game_state": "PLAYING"}

    def act(self, action, args):
        if self.level == 0 and action == "ACTION4":
            self.level = 1
            self.levels_completed = 1
        self.frame[1 if self.level else 0, 1] += 1
        return {}

    def reset(self):
        self.level = 0
        self.levels_completed = 0
        self.frame.fill(0)
        return self.observe()

    def fork(self):
        return CanonicalMultiLevelEnv()


def test_inspection_region_history_and_invalid_region(tmp_path):
    from v11.tools.discovery import DISCOVERY_TOOLS

    inspect_schema = next(item["function"] for item in DISCOVERY_TOOLS
                          if item["function"]["name"] == "inspect_region")
    assert "20x20" in inspect_schema["description"]
    assert "20 rows by 20 columns" in inspect_schema["parameters"]["properties"]["region"]["description"]
    env = Env()
    env.frame = np.zeros((24, 24), dtype=np.int64)
    game = GameTools(env, tmp_path, max_actions=3)
    state = game.observe()
    game.act("up")
    discovery = Discovery(game, "r", "g", tmp_path / "memory")
    result = discovery.dispatch({"request_id":"D1", "tool":"inspect_region",
                                 "arguments":{"evidence_id":state["evidence_id"], "region":[0,0,20,20]}})
    assert len(result["data"]["pixels"]) == 20
    assert all(len(row) == 20 for row in result["data"]["pixels"])
    transition_id = game.history.records()[-1]["evidence_id"]
    transition = discovery.dispatch({"request_id":"D2", "tool":"inspect_region",
                                     "arguments":{"evidence_id":transition_id, "region":[0,0,20,20]}})
    assert len(transition["data"]["before"]) == 20
    assert len(transition["data"]["after"]) == 20
    assert transition["data"]["measurements"]["total_pixels"] == 400
    invalid = discovery.dispatch({"request_id":"D3", "tool":"inspect_region",
                                  "arguments":{"evidence_id":state["evidence_id"], "region":[0,0,21,20]}})
    assert invalid["status"] == "rejected"
    assert "at most 20 rows by 20 columns" in invalid["error"]["message"]
    invalid_width = discovery.dispatch({"request_id":"D4", "tool":"inspect_region",
                                        "arguments":{"evidence_id":state["evidence_id"], "region":[0,0,20,21]}})
    assert invalid_width["status"] == "rejected"
    assert "at most 20 rows by 20 columns" in invalid_width["error"]["message"]
    missing = discovery.dispatch({"request_id":"D5", "tool":"inspect_region",
                                  "arguments":{"evidence_id":state["evidence_id"]}})
    assert missing["status"] == "rejected"
    assert "region is required" in missing["error"]["message"]
    found = discovery.dispatch({"request_id":"D6", "tool":"search_history", "arguments":{"action":"up","limit":10}})
    assert found["data"]["count"] == 1


def test_experiment_runs_in_isolated_branch_and_retry_returns_scored_result(tmp_path):
    env = Env()
    game = GameTools(env, tmp_path, max_actions=3)
    state = game.observe()
    discovery = Discovery(game, "r", "g", tmp_path / "memory")
    request = {"request_id":"D1", "tool":"test_hypotheses", "arguments":{
        "base_evidence_id":state["evidence_id"], "action":{"action":"up","args":{}},
        "hypotheses":[
            {"id":"H1", "prediction":{"all":[{"path":"measurements.changed_pixels","op":"gt","value":0}]}},
            {"id":"H2", "prediction":{"all":[{"path":"measurements.changed_pixels","op":"eq","value":0}]}},
        ]}}
    first = discovery.dispatch(request)
    retry = discovery.dispatch(request)
    assert first == retry
    assert env.actions == []
    assert first["data"]["cases"][0]["status"] == "completed"
    assert [h["verdict"] for h in first["data"]["cases"][0]["hypotheses"]] == ["matched", "failed"]
    assert game.action_budget.used == 0
    assert discovery.branch_test_actions == 1
    rows = [json.loads(line) for line in (tmp_path / "discovery.jsonl").read_text().splitlines()]
    assert any(row["status"] == "started" for row in rows)
    assert any(row.get("event") == "branch_started" for row in rows)


def test_completed_branch_outcome_is_reused_and_branch_frames_are_scoped(tmp_path):
    game = GameTools(Env(), tmp_path, max_actions=3)
    state = game.observe()
    game.env.frame[2, 2] = 7
    live_collision = game.act("up")
    discovery = Discovery(game, "r", "g", tmp_path / "memory")
    changed = {"all": [{"path": "measurements.changed_pixels", "op": "gt", "value": 0}]}
    unchanged = {"all": [{"path": "measurements.changed_pixels", "op": "eq", "value": 0}]}
    first = discovery.dispatch({"request_id": "first", "tool": "test_hypotheses", "arguments": {
        "base_evidence_id": state["evidence_id"], "cases": [
            {"case_id": "one", "action": {"action": "up"}, "hypotheses": [{"id": "yes", "prediction": changed}]},
            {"case_id": "same", "action": {"action": "up", "args": {}}, "hypotheses": [{"id": "no", "prediction": unchanged}]},
        ]}})
    cases = first["data"]["cases"]
    assert cases[0]["status"] == cases[1]["status"] == "completed"
    assert cases[1]["reused"] is True
    assert cases[0]["branch_id"] == cases[1]["branch_id"]
    assert cases[1]["hypotheses"][0]["verdict"] == "failed"
    assert first["evidence_refs"] == [{"branch_id": cases[0]["branch_id"], "evidence_id": cases[0]["evidence_id"]},
                                      {"branch_id": cases[1]["branch_id"], "evidence_id": cases[1]["evidence_id"]}]
    assert discovery.branch_test_actions == 1
    branch_inspection = discovery.dispatch({"request_id": "branch-read", "tool": "inspect_region", "arguments": {
        "evidence_id": cases[0]["evidence_id"], "branch_id": cases[0]["branch_id"], "region": [2, 2, 3, 3]}})
    assert branch_inspection["data"]["after"] == [[1]]
    assert live_collision["evidence"]["evidence_id"] == cases[0]["evidence_id"]
    live_inspection = discovery.dispatch({"request_id": "live-read", "tool": "inspect_region", "arguments": {
        "evidence_id": live_collision["evidence"]["evidence_id"], "region": [2, 2, 3, 3]}})
    assert live_inspection["data"]["after"] == [[8]]
    invalid = discovery.dispatch({"request_id": "bad-branch", "tool": "inspect_region", "arguments": {
        "evidence_id": cases[0]["evidence_id"], "branch_id": "badbranch", "region": [0, 0, 1, 1]}})
    assert invalid["status"] == "rejected"
    assert "invalid branch_id" in invalid["error"]["message"]
    missing_branch = discovery.dispatch({"request_id": "missing-branch", "tool": "inspect_region", "arguments": {
        "evidence_id": live_collision["evidence"]["evidence_id"], "branch_id": "a" * 16, "region": [2, 2, 3, 3]}})
    assert missing_branch["status"] == "rejected"
    assert "unknown branch evidence_id" in missing_branch["error"]["message"]
    second_discovery = Discovery(game, "r", "g", tmp_path / "memory")
    second = second_discovery.dispatch({"request_id": "second", "tool": "test_hypotheses", "arguments": {
        "base_evidence_id": state["evidence_id"], "action": {"action": "up"},
        "hypotheses": [{"prediction": unchanged}]}})
    assert second["status"] == "completed", second.get("error")
    assert second["data"]["cases"][0]["reused"] is True
    assert second_discovery.branch_test_actions == 0
    assert second_discovery.branch_reuses == 2


def test_hypothesis_prediction_contract_is_advertised_and_matches_runtime(tmp_path):
    from v11.tools.discovery import DISCOVERY_TOOLS

    tool = next(item["function"] for item in DISCOVERY_TOOLS
                if item["function"]["name"] == "test_hypotheses")
    parameters = tool["parameters"]["properties"]
    case_hypothesis = parameters["cases"]["items"]["properties"]["hypotheses"]["items"]
    assert "hypotheses" not in parameters and "action" not in parameters
    assert parameters["cases"]["maxItems"] == 10
    for hypothesis in (case_hypothesis,):
        prediction = hypothesis["properties"]["prediction"]
        assert prediction["required"] == ["all"]
        assert prediction["additionalProperties"] is False
        predicate = prediction["properties"]["all"]["items"]
        assert predicate["required"] == ["path", "op", "value"]
        assert predicate["properties"]["path"]["enum"] == [
            "measurements.changed_pixels", "measurements.change_ratio",
            "measurements.click_effect.board_changed_pixels", "game_state",
            "terminal", "level", "levels_completed",
        ]
        assert predicate["properties"]["op"]["enum"] == [
            "eq", "ne", "gt", "gte", "lt", "lte", "between",
        ]

    game = GameTools(Env(), tmp_path, max_actions=3)
    state = game.observe()
    discovery = Discovery(game, "r", "g", tmp_path / "memory")
    rejected = discovery.dispatch({"request_id": "bad-format", "tool": "test_hypotheses", "arguments": {
        "base_evidence_id": state["evidence_id"], "action": {"action": "up"},
        "hypotheses": [
            {"id": "changed", "prediction": {"type": "pixel_diff", "description": "pixels change"}},
            {"id": "unchanged", "prediction": {"type": "pixel_diff", "description": "no pixels change"}},
        ],
    }})
    assert rejected["status"] == "rejected"
    assert "hypothesis prediction must use" in rejected["error"]["message"]

    completed = discovery.dispatch({"request_id": "good-format", "tool": "test_hypotheses", "arguments": {
        "base_evidence_id": state["evidence_id"], "action": {"action": "up"},
        "hypotheses": [
            {"id": "changed", "prediction": {"all": [
                {"path": "measurements.changed_pixels", "op": "gt", "value": 0}]}},
            {"id": "unchanged", "prediction": {"all": [
                {"path": "measurements.changed_pixels", "op": "eq", "value": 0}]}},
        ],
    }})
    assert completed["status"] == "completed"


def test_hypothesis_branch_does_not_consume_live_action_cap(tmp_path):
    budget = ActionBudget(1, used=1)
    game = GameTools(Env(), tmp_path, max_actions=1, action_budget=budget)
    state = game.observe()
    discovery = Discovery(game, "r", "g", tmp_path / "memory")
    request = {"request_id": "D1", "tool": "test_hypotheses", "arguments": {
        "base_evidence_id": state["evidence_id"], "action": {"action": "up"},
        "hypotheses": [
            {"prediction": {"all": [{"path": "measurements.changed_pixels", "op": "gt", "value": 0}]}},
            {"prediction": {"all": [{"path": "measurements.changed_pixels", "op": "eq", "value": 0}]}},
        ],
    }}
    result = discovery.dispatch(request)
    assert result["status"] == "completed"
    assert budget.used == 1
    assert result["data"]["branch_test_actions"] == 1


def test_multiple_candidate_cases_use_independent_branches_from_same_base(tmp_path):
    game = GameTools(Env(), tmp_path, max_actions=4)
    state = game.observe()
    discovery = Discovery(game, "r", "g", tmp_path / "memory")
    hypotheses = [
        {"id": "changed", "prediction": {"all": [{"path": "measurements.changed_pixels", "op": "gt", "value": 0}]}},
        {"id": "unchanged", "prediction": {"all": [{"path": "measurements.changed_pixels", "op": "eq", "value": 0}]}},
    ]
    result = discovery.dispatch({"request_id": "D1", "tool": "test_hypotheses", "arguments": {
        "base_evidence_id": state["evidence_id"],
        "cases": [
            {"case_id": f"candidate-{index}",
             "action": {"action": "up" if index % 2 else "right"},
             "hypotheses": hypotheses}
            for index in range(10)
        ],
    }})
    assert all(case["status"] == "completed" for case in result["data"]["cases"])
    assert len(result["data"]["cases"]) == 10
    assert {case["case_id"] for case in result["data"]["cases"]} == {f"candidate-{index}" for index in range(10)}
    assert game.action_budget.used == 0
    assert discovery.branch_test_actions == 2
    assert discovery.branch_replay_actions == 0
    assert game.env.actions == []
    repeated = discovery.dispatch({"request_id": "D2", "tool": "test_hypotheses", "arguments": {
        "base_evidence_id": state["evidence_id"], "action": {"action": "up"},
        "hypotheses": hypotheses,
    }})
    assert repeated["status"] == "completed"
    assert discovery.branch_test_actions == 2
    assert repeated["data"]["cases"][0]["reused"] is True


def test_replay_mismatch_does_not_execute_candidate(tmp_path):
    class MismatchedEnv(Env):
        def fork(self):
            branch = Env()
            branch.frame[0, 0] = 9
            return branch

    game = GameTools(MismatchedEnv(), tmp_path, max_actions=2)
    state = game.observe()
    discovery = Discovery(game, "r", "g", tmp_path / "memory")
    result = discovery.dispatch({"request_id": "D1", "tool": "test_hypotheses", "arguments": {
        "base_evidence_id": state["evidence_id"], "action": {"action": "up"},
        "hypotheses": [
            {"prediction": {"all": [{"path": "measurements.changed_pixels", "op": "gt", "value": 0}]}},
            {"prediction": {"all": [{"path": "measurements.changed_pixels", "op": "eq", "value": 0}]}},
        ],
    }})
    assert result["data"]["cases"][0]["status"] == "replay_mismatch"
    assert game.action_budget.used == 0
    assert game.env.actions == []


def test_hypothesis_replays_from_level_zero_through_reset_to_target_level(tmp_path):
    game = GameTools(CanonicalMultiLevelEnv(), tmp_path, max_actions=4)
    game.observe()
    game.reset("exercise the recorded reset before level progression")
    reached_level = game.act("ACTION4", action_name="right")
    assert reached_level["evidence"]["post_level"] == 1
    discovery = Discovery(game, "r", "g", tmp_path / "memory")
    hypotheses = [
        {"id": "level-one", "prediction": {"all": [{"path": "level", "op": "eq", "value": 1}]}},
        {"id": "completed-first-level", "prediction": {"all": [{"path": "levels_completed", "op": "eq", "value": 1}]}},
    ]

    result = discovery.dispatch({"request_id": "level-test", "tool": "test_hypotheses", "arguments": {
        "base_evidence_id": reached_level["evidence"]["evidence_id"],
        "action": {"action": "down"},
        "hypotheses": hypotheses,
    }})

    case = result["data"]["cases"][0]
    assert case["status"] == "completed"
    assert case["action"]["action"] == "ACTION2"
    assert all(item["verdict"] == "matched" for item in case["hypotheses"])
    assert result["data"]["branch_resets"] == 1
    assert result["data"]["branch_replay_actions"] == 1
    assert result["data"]["branch_test_actions"] == 1

    branch_records = [json.loads(line) for line in (
        tmp_path / "discovery_branches" / case["branch_id"] / "evidence.jsonl"
    ).read_text().splitlines()]
    transitions = [row for row in branch_records if row.get("type") == "transition"]
    assert [(row["pre_level"], row["post_level"]) for row in transitions] == [(0, 1), (1, 1)]
    assert transitions[1]["action"] == "ACTION2"


def test_request_id_cannot_be_reused_with_changed_payload(tmp_path):
    game = GameTools(Env(), tmp_path)
    state = game.observe()
    discovery = Discovery(game, "r", "g", tmp_path / "memory")
    req = {"request_id":"D1", "tool":"inspect_region", "arguments":{"evidence_id":state["evidence_id"], "region":[0,0,1,1]}}
    discovery.dispatch(req)
    req["arguments"] = {"evidence_id":state["evidence_id"], "region":[0,0,2,1]}
    with pytest.raises(ValueError, match="changed payload"):
        discovery.dispatch(req)


def test_reads_are_not_count_limited_and_proposals_queue_for_review(tmp_path):
    game = GameTools(Env(), tmp_path)
    state = game.observe()
    discovery = Discovery(game, "r", "g", tmp_path / "memory")
    for request_id in ("D1", "D2"):
        result = discovery.dispatch({"request_id":request_id, "tool":"inspect_region",
                                     "arguments":{"evidence_id":state["evidence_id"], "region":[0,0,1,1]}})
        assert result["status"] == "completed"
    additional_read = discovery.dispatch({"request_id":"D3", "tool":"search_history", "arguments":{}})
    assert additional_read["status"] == "completed"

    queued = discovery.dispatch({"request_id":"D4", "tool":"game_memory", "arguments":{
        "operation":"propose", "claim":"The test board changes after moving up.",
        "conditions":["level 1"], "supporting_evidence":[state["evidence_id"]],
        "contradicting_evidence":[state["evidence_id"]],
    }})
    assert queued["data"]["status"] == "queued"
    assert queued["data"]["proposal"]["supporting_evidence"] == [state["evidence_id"]]
    assert queued["data"]["proposal"]["contradicting_evidence"] == []
    another = discovery.dispatch({"request_id":"D5", "tool":"game_memory", "arguments":{
        "operation":"propose", "claim":"The test board also changes after moving right.",
        "conditions":["level 1"], "supporting_evidence":[state["evidence_id"]],
    }})
    assert another["data"]["status"] == "queued"


def test_live_tool_call_continues_to_normal_decision_and_shared_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("V11_STATE_DIR", str(tmp_path / "state"))

    class Client:
        model = "test"
        def __init__(self): self.calls = []
        def complete(self, *args, **kwargs):
            self.system_prompt = args[0]
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return {"content": "", "tool_calls": [{"id":"D1", "type":"function", "function": {
                    "name":"test_hypotheses", "arguments": json.dumps({
                        "base_evidence_id":"T1", "action":{"action":"up", "args":{}},
                        "hypotheses":[
                            {"id":"H1", "prediction":{"all":[{"path":"measurements.changed_pixels","op":"gt","value":0}]}},
                            {"id":"H2", "prediction":{"all":[{"path":"measurements.changed_pixels","op":"eq","value":0}]}},
                        ],
                    }),
                }}], "usage":{}, "latency_seconds":0.0, "request_attempts":1, "model":"test"}
            return {"content": json.dumps({
                "state_assessment": {"player": {"location": "unknown", "certainty": "unknown", "evidence": "unknown"}, "target_candidates": [], "obstacles": [], "other_relevant_changes": []},
                "hypothesis": None, "trajectory_rationale": "Use the measured action.",
                "prediction": {"visible_change": "unknown", "environment_outcome": "unknown", "description": "unknown"},
                "actions": [{"action": "right", "args": {}}],
            }), "usage":{}, "latency_seconds":0.0, "request_attempts":1, "model":"test"}

    env = Env()
    run_dir = tmp_path / "live"
    client = Client()
    metrics = run_live(env, run_dir, client, game_id="fixture", max_actions=1,
                       starting_routine=False)
    assert env.actions == ["right"]
    assert metrics["actions"] == 1
    assert metrics["action_budget_used"] == 1
    assert metrics["action_budget_remaining"] == 0
    assert metrics["discovery_test_actions"] == 1
    assert "## Discovery tools" in client.system_prompt
    prompt_text = "\n".join(
        str(block.get("text", ""))
        for message in client.calls[0]["messages"] if message.get("role") == "user"
        for block in (message.get("content") if isinstance(message.get("content"), list) else [{"text": message.get("content", "")}])
        if isinstance(block, dict)
    )
    assert '"evidence_id":"T1"' in prompt_text
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["configuration"]["discovery_tools"] is True
    decisions = [json.loads(line) for line in (run_dir / "decisions.jsonl").read_text().splitlines()]
    assert any(row.get("source") == "model" for row in decisions)
    events = [json.loads(line) for line in (run_dir / "discovery.jsonl").read_text().splitlines()]
    assert any(row.get("status") == "experiment_scored" for row in events)


@pytest.mark.parametrize("raw_arguments", ["{bad json", json.dumps({"evidence_id": "missing"})])
def test_live_native_tool_errors_are_returned_and_logged(tmp_path, monkeypatch, raw_arguments, capsys):
    monkeypatch.setenv("V11_STATE_DIR", str(tmp_path / "state"))

    class Client:
        model = "test"
        def __init__(self): self.calls = []
        def complete(self, *args, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return {"content": "", "tool_calls": [{"id": "read-1", "type": "function", "function": {
                    "name": "inspect_region", "arguments": raw_arguments,
                }}], "usage": {}, "latency_seconds": 0.0, "request_attempts": 1, "model": "test"}
            return {"content": json.dumps({
                "state_assessment": {"player": {"location": "unknown", "certainty": "unknown", "evidence": "unknown"}, "target_candidates": [], "obstacles": [], "other_relevant_changes": []},
                "hypothesis": None, "trajectory_rationale": "Test a legal move.",
                "prediction": {"visible_change": "unknown", "environment_outcome": "unknown", "description": "unknown"},
                "actions": [{"action": "up", "args": {}}],
            }), "usage": {}, "latency_seconds": 0.0, "request_attempts": 1, "model": "test"}

    client = Client()
    env = Env()
    run_live(env, tmp_path / "continued", client, game_id="fixture", max_actions=1, starting_routine=False)

    assert client.calls[0]["tools"]
    assert client.calls[0]["response_format"] is None
    history = client.calls[1]["messages"]
    assistant_call = next(message for message in history if message.get("role") == "assistant" and message.get("tool_calls"))
    tool_result = next(message for message in history if message.get("role") == "tool")
    assert assistant_call["tool_calls"][0]["id"] == "read-1"
    assert tool_result["tool_call_id"] == "read-1"
    assert history.index(assistant_call) < history.index(tool_result)
    assert env.actions == ["up"]
    result = json.loads(history[history.index(tool_result)]["content"])
    assert result["status"] == "rejected"
    assert any(json.loads(line).get("status") == "rejected" for line in (tmp_path / "continued/discovery.jsonl").read_text().splitlines())
    stdout = capsys.readouterr().out
    assert "[v11 tool] call: inspect_region (id=read-1)" in stdout
    assert "[v11 tool] result: inspect_region status=rejected" in stdout


def test_discovery_tools_are_in_shared_skill():
    from pathlib import Path
    skill = (Path(__file__).resolve().parents[1] / "skills/SKILL.md").read_text()
    assert "discovery functions are available during action selection" in skill
    assert '"tool":"inspect_region"' in skill or "`inspect_region`" in skill


def test_live_caps_planning_at_three_tool_rounds_then_requires_action(tmp_path, monkeypatch):
    monkeypatch.setenv("V11_STATE_DIR", str(tmp_path / "state"))

    action_decision = {
        "state_assessment": {"player": {"location": "unknown", "certainty": "unknown", "evidence": "unknown"},
                             "target_candidates": [], "obstacles": [], "other_relevant_changes": []},
        "hypothesis": None, "trajectory_rationale": "Use a legal move.",
        "prediction": {"visible_change": "unknown", "environment_outcome": "unknown", "description": "unknown"},
        "actions": [{"action": "right", "args": {}}],
    }
    class Client:
        model = "test"

        def __init__(self):
            self.calls = []

        def complete(self, *args, **kwargs):
            self.calls.append(kwargs)
            completion = len(self.calls)
            if completion <= 3:
                return {"content": "", "tool_calls": [{
                    "id": f"D{completion}", "type": "function", "function": {
                        "name": "search_history",
                        "arguments": json.dumps({}),
                    },
                }], "usage": {}, "latency_seconds": 0.0, "request_attempts": 1, "model": "test"}
            return {"content": json.dumps(action_decision), "usage": {}, "latency_seconds": 0.0,
                    "request_attempts": 1, "model": "test"}

    client = Client()
    env = Env()
    run_dir = tmp_path / "ten-completions"
    metrics = run_live(env, run_dir, client, game_id="fixture", max_actions=1, starting_routine=False)

    assert len(client.calls) == 4
    assert all(call["tools"] and call["response_format"] is None for call in client.calls[:3])
    assert client.calls[3]["tools"] is None
    assert client.calls[3]["response_format"] == "json_object"
    assert any(
        message.get("role") == "user"
        and "three tool-call planning rounds have been used" in message.get("content", "")
        for message in client.calls[3]["messages"]
    )
    assert metrics["model_calls"] == 4
    assert metrics["actions"] == 1
    assert metrics["discovery_test_actions"] == 0
    assert env.actions == ["right"]
    rows = [json.loads(line) for line in (run_dir / "discovery.jsonl").read_text().splitlines()]
    assert sum(row.get("status") == "experiment_scored" for row in rows) == 0


def test_four_hypothesis_cases_force_action_completion(tmp_path, monkeypatch):
    monkeypatch.setenv("V11_STATE_DIR", str(tmp_path / "state"))
    action_decision = {
        "state_assessment": {"player": {"location": "unknown", "certainty": "unknown", "evidence": "unknown"},
                             "target_candidates": [], "obstacles": [], "other_relevant_changes": []},
        "hypothesis": None, "trajectory_rationale": "Use a legal move.",
        "prediction": {"visible_change": "unknown", "environment_outcome": "unknown", "description": "unknown"},
        "actions": [{"action": "right", "args": {}}],
    }
    cases = [{
        "case_id": f"case-{index}", "action": {"action": ["up", "right", "left", "down"][index - 1], "args": {}},
        "hypotheses": [{"prediction": {"all": [{"path": "measurements.changed_pixels", "op": "gte", "value": index}]}}],
    } for index in range(1, 5)]

    class Client:
        model = "test"

        def __init__(self):
            self.calls = []

        def complete(self, *args, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return {"content": "", "tool_calls": [{"id": "four-tests", "type": "function", "function": {
                    "name": "test_hypotheses", "arguments": json.dumps({"base_evidence_id": "T1", "cases": cases}),
                }}], "usage": {}, "latency_seconds": 0.0, "request_attempts": 1, "model": "test"}
            return {"content": json.dumps(action_decision), "usage": {}, "latency_seconds": 0.0,
                    "request_attempts": 1, "model": "test"}

    class FourActionEnv(Env):
        def observe(self):
            result = super().observe()
            result["legal_actions"] = ["up", "right", "left", "down"]
            return result

    client = Client()
    run_dir = tmp_path / "four-tests"
    metrics = run_live(FourActionEnv(), run_dir, client, game_id="fixture", max_actions=1, starting_routine=False)
    assert client.calls[0]["tools"]
    assert client.calls[1]["tools"] is None
    assert any(message.get("role") == "user" and "ACTION DECISION REQUIRED" in message.get("content", "")
               for message in client.calls[1]["messages"])
    assert metrics["discovery_test_actions"] == 4
    assert metrics["model_calls"] == 2


def test_same_hypothesis_test_reuses_outcome_from_same_evidence_state(tmp_path, monkeypatch):
    monkeypatch.setenv("V11_STATE_DIR", str(tmp_path / "state"))
    decision = {
        "state_assessment": {"player": {"location": "unknown", "certainty": "unknown", "evidence": "unknown"},
                             "target_candidates": [], "obstacles": [], "other_relevant_changes": []},
        "hypothesis": None, "trajectory_rationale": "Use a legal move.",
        "prediction": {"visible_change": "unknown", "environment_outcome": "unknown", "description": "unknown"},
        "actions": [{"action": "right", "args": {}}],
    }
    arguments = {"base_evidence_id": "T1", "cases": [{"action": {"action": "up"}, "hypotheses": [
        {"prediction": {"all": [{"path": "measurements.changed_pixels", "op": "gt", "value": 0}]}}
    ]}]}

    class Client:
        model = "test"

        def __init__(self):
            self.count = 0

        def complete(self, *args, **kwargs):
            self.count += 1
            if self.count <= 2:
                return {"content": "", "tool_calls": [{"id": f"dup-{self.count}", "type": "function", "function": {
                    "name": "test_hypotheses", "arguments": json.dumps(arguments),
                }}], "usage": {}, "latency_seconds": 0.0, "request_attempts": 1, "model": "test"}
            return {"content": json.dumps(decision), "usage": {}, "latency_seconds": 0.0,
                    "request_attempts": 1, "model": "test"}

    run_dir = tmp_path / "duplicate-test"
    metrics = run_live(Env(), run_dir, Client(), game_id="fixture", max_actions=1, starting_routine=False)
    assert metrics["discovery_test_actions"] == 1
    rows = [json.loads(line) for line in (run_dir / "discovery.jsonl").read_text().splitlines()]
    assert any(row.get("event") == "branch_reused" for row in rows)
    assert metrics["discovery_branch_reuses"] == 1


def test_click_discovery_answers_use_click_label_and_preserve_canonical_branch_evidence(tmp_path):
    class ClickEnv:
        legal_actions = [{"id": "ACTION6", "parameters": {
            "type": "object", "properties": {
                "x": {"type": "integer", "minimum": 0, "maximum": 7},
                "y": {"type": "integer", "minimum": 0, "maximum": 7},
            }, "required": ["x", "y"], "additionalProperties": False,
        }}]

        def __init__(self):
            self.frame = np.zeros((8, 8), dtype=np.int64)

        def observe(self):
            return {"frame": self.frame.copy(), "legal_actions": self.legal_actions,
                    "level": 0, "levels_completed": 0, "terminal": False, "game_state": "PLAYING"}

        def act(self, action, args):
            self.frame[args["y"], args["x"]] = 1
            return {}

        def fork(self):
            return ClickEnv()

    game = GameTools(ClickEnv(), tmp_path, max_actions=3)
    state = game.observe()
    discovery = Discovery(game, "run", "fixture", tmp_path / "memory")
    coordinates = [(index, 3) for index in range(8)] + [(0, 4), (1, 4)]
    cases = [{
        "case_id": f"click-site-{index}",
        "action": {"action": "click", "args": {"x": x, "y": y}},
        "hypotheses": [{"id": f"changed-{index}", "prediction": {"all": [
            {"path": "measurements.changed_pixels", "op": "gt", "value": 0},
        ]}}],
    } for index, (x, y) in enumerate(coordinates)]
    result = discovery.dispatch({"request_id": "click-test", "tool": "test_hypotheses", "arguments": {
        "base_evidence_id": state["evidence_id"], "cases": cases,
    }})
    assert len(result["data"]["cases"]) == 10
    assert result["data"]["executed_branches"] == 10
    assert discovery.branch_test_actions == 10
    assert game.action_count == 0
    assert len({(case["action"]["args"]["x"], case["action"]["args"]["y"]) for case in result["data"]["cases"]}) == 10
    case = result["data"]["cases"][2]
    assert case["action"] == {"action": "click", "args": {"x": 2, "y": 3}}
    assert case["display_action"] == "click(x=2,y=3)"
    assert case["actual"]["action"] == "click"
    assert case["actual"]["display_action"] == "click(x=2,y=3)"
    branch_evidence = json.loads((tmp_path / "discovery_branches" / case["branch_id"] / "evidence.jsonl").read_text().splitlines()[-1])
    assert branch_evidence["action"] == "ACTION6"
    assert branch_evidence["args"] == {"x": 2, "y": 3}
    assert branch_evidence["display_action"] == "click(x=2,y=3)"

    game.act("ACTION6", {"x": 4, "y": 3})
    history = discovery.dispatch({"request_id": "click-history", "tool": "search_history", "arguments": {
        "action": "click",
    }})
    assert history["data"]["evidence"][-1]["action"] == "click"
    assert history["data"]["evidence"][-1]["display_action"] == "click(x=4,y=3)"
