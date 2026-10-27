from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from v10.analysis import (
    DecisionValidationError,
    build_analysis_input,
    run_analysis,
    score_recorded_plan,
    validate_model_decision,
    validate_reasoned_action,
)
from v10 import cli as cli_module
from v10.cli import build_parser
from v10.frames import FrameStore, png_bytes
from v10.evaluation import run_ablation
from v10.replay import replay_trace
from v10.strata_client import StrataClient, StrataRequestError
from v10.tools.game import evaluate_prediction
from v10.tools.history import History


@pytest.fixture(autouse=True)
def _isolated_memory_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("V10_STATE_DIR", str(tmp_path / "state"))


def _write_trace(path: Path) -> None:
    frame = np.zeros((2, 3), dtype=np.int64)
    with path.open("w", encoding="utf-8") as handle:
        for index in range(7):
            if index:
                frame = frame.copy()
                frame[0, 0] = index % 2
            action_id = "RESET" if index == 0 else f"ACTION{1 if index == 6 else (index % 2) + 1}"
            row = {
                "timestamp": f"2026-01-01T00:00:{index:02d}Z",
                "data": {
                    "game_id": "fixture",
                    "state": "NOT_FINISHED",
                    "levels_completed": 0,
                    "win_levels": 1,
                    "action_input": {
                        "id": action_id,
                        "data": {} if index == 0 else {"game_id": "fixture"},
                        "reasoning": f"HISTORICAL_REASONING_{index}" + (" FUTURE_ONLY" if index == 5 else ""),
                    },
                    "guid": "fixture-guid",
                    "full_reset": index == 0,
                    "available_actions": [1, 2],
                    "frame": frame.tolist(),
                },
            }
            handle.write(json.dumps(row) + "\n")


def _valid_decision(action: str = "down", args: dict | None = None) -> dict:
    return {"actions": [{"action": action, "args": args or {}}]}


def _reasoned_decision(action: str = "down", args: dict | None = None) -> dict:
    return {
        "state_assessment": {
            "player": {"location": "unknown", "certainty": "unknown", "evidence": "not visually identifiable"},
            "target_candidates": [], "obstacles": [], "other_relevant_changes": [],
        },
        "hypothesis": None,
        "trajectory_rationale": "Test one legal action and compare the next screen.",
        "prediction": {"visible_change": "unknown", "environment_outcome": "unknown", "description": "unknown"},
        "actions": [{"action": action, "args": args or {}}],
    }


class _FakeClient:
    def __init__(self, outputs: list[str] | None = None, decision: dict | None = None):
        self.outputs = list(outputs or [])
        self.decision = decision or _reasoned_decision()
        self.calls = []

    def complete(self, system_prompt, user_prompt, image_paths, image_labels):
        self.calls.append((system_prompt, user_prompt, list(image_paths), list(image_labels)))
        content = self.outputs.pop(0) if self.outputs else json.dumps(self.decision)
        return {
            "content": content,
            "model": "mock-qwen",
            "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
            "latency_seconds": 0.01,
        }


def test_analysis_context_contains_only_prefix_and_selected_images(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace)
    run_dir = tmp_path / "run"
    replay_trace(trace, run_dir)

    context = build_analysis_input(run_dir, through_action=4)

    assert len(context["prefix_transitions"]) == 4
    assert len(context["image_paths"]) == 2
    assert "replay_records" not in context
    assert "FUTURE_ONLY" not in context["user_prompt"]
    assert "HISTORICAL_REASONING" not in context["user_prompt"]
    assert all("reasoning" not in row for row in context["prefix_transitions"])
    visible_ids = set(context["visible_evidence_ids"])
    assert context["initial_state_evidence_id"] in visible_ids
    assert all(row["evidence_id"] in visible_ids for row in context["prefix_transitions"])
    assert all(frame["frame_id"] not in visible_ids for frame in FrameStore(run_dir).entries())
    assert "citable_evidence_ids" not in context["context"]
    assert "visible_prediction_ids" not in context["context"]


def test_context_reader_stops_before_malformed_future_evidence(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace)
    run_dir = tmp_path / "run"
    replay_trace(trace, run_dir)
    with (run_dir / "evidence.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("malformed future evidence\n")

    context = build_analysis_input(run_dir, through_action=4)

    assert len(context["prefix_transitions"]) == 4


def test_ablation_profiles_select_expected_memory_and_delta_images(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace)
    run_dir = tmp_path / "run"
    replay_trace(trace, run_dir)

    pixels = build_analysis_input(run_dir, conditioning="pixels_actions", horizon=2)
    with_memory = build_analysis_input(run_dir, conditioning="pixels_actions_memory", horizon=2)
    assert len(pixels["image_paths"]) == len(with_memory["image_paths"]) == 2
    assert pixels["context"]["memory"] is None
    assert with_memory["context"]["memory"] is not None


def test_ablation_runs_matched_profiles_without_writing_analysis_decisions(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace)
    run_dir = tmp_path / "run"
    replay_trace(trace, run_dir)

    result = run_ablation(run_dir, _FakeClient(), through_action=4, horizon=2)

    assert [arm["conditioning"] for arm in result["arms"]] == [
        profile for profile in ("pixels_actions", "pixels_actions_memory")
        for _ in (False, True)
    ]
    assert [arm["include_prior_decisions"] for arm in result["arms"]] == [False, True] * 2
    assert all(arm["status"] == "ok" for arm in result["arms"])
    assert not (run_dir / "decisions.jsonl").exists()
    assert (run_dir / "ablation_runs.jsonl").exists()


def test_run_analysis_scores_matching_recorded_action_after_model_call(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace)
    run_dir = tmp_path / "run"
    replay_trace(trace, run_dir)
    client = _FakeClient()

    result = run_analysis(run_dir, client, through_action=4)

    assert result["action_matches"] == [{
        "plan_index": 0,
        "planned_action": "ACTION2",
        "status": "matched",
        "recorded_evidence_id": "T6",
        "recorded_action": "ACTION2",
    }]
    assert "FUTURE_ONLY" not in client.calls[0][1]
    assert '"actions"' in client.calls[0][1]
    assert "memory_patch" not in client.calls[0][1]
    assert "hypotheses_item" not in client.calls[0][1]
    assert result["usage"]["total_tokens"] == 17
    assert (run_dir / "decisions.jsonl").exists()


def test_analysis_retries_once_for_invalid_json_and_never_scores_before_validity(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace)
    run_dir = tmp_path / "run"
    replay_trace(trace, run_dir)
    client = _FakeClient(outputs=["not-json", json.dumps(_reasoned_decision())])

    result = run_analysis(run_dir, client, through_action=4)

    assert result["repair_attempted"] is True
    assert result["initial_validation_error"].startswith("response is not valid JSON")
    assert len(client.calls) == 2
    assert "FUTURE_ONLY" not in client.calls[1][1]


def test_analysis_repairs_legacy_action_only_shape_instead_of_migrating_it(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace)
    run_dir = tmp_path / "run"
    replay_trace(trace, run_dir)
    client = _FakeClient(outputs=[json.dumps(_valid_decision()), json.dumps(_reasoned_decision())])

    result = run_analysis(run_dir, client, through_action=4)

    assert result["repair_attempted"] is True
    assert "reasoned action must contain exactly" in result["initial_validation_error"]
    assert len(client.calls) == 2
    assert "corrected JSON object using exactly the required structured fields" in client.calls[1][1]


def test_action_only_response_rejects_unrequested_analysis_and_memory_fields(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace)
    run_dir = tmp_path / "run"
    replay_trace(trace, run_dir)
    context = build_analysis_input(run_dir, through_action=4)
    decision = _valid_decision()
    decision["memory_patch"] = {"upsert": "legacy malformed patch ignored by action selector"}
    decision["observations"] = [{"id": "O1", "claim": "legacy", "evidence_ids": ["F0"]}]

    with pytest.raises(DecisionValidationError, match="only an actions list"):
        validate_model_decision(decision, context["legal_action_specs"])


def test_action_response_needs_only_action_and_args(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace)
    run_dir = tmp_path / "run"
    replay_trace(trace, run_dir)
    context = build_analysis_input(run_dir, through_action=4)

    normalized = validate_model_decision(
        {"actions": [{"action": "down", "args": {}}]},
        context["legal_action_specs"],
    )

    assert normalized["actions"][0]["action"] == "ACTION2"
    assert normalized == {"actions": [{"action": "ACTION2", "args": {}}]}
    with pytest.raises(DecisionValidationError, match="directional name 'down'"):
        validate_model_decision(
            {"actions": [{"action": "ACTION2", "args": {}}]},
            context["legal_action_specs"],
        )


def test_action_response_respects_requested_plan_horizon(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace)
    run_dir = tmp_path / "run"
    replay_trace(trace, run_dir)
    context = build_analysis_input(run_dir, through_action=4)

    with pytest.raises(DecisionValidationError, match="0 to 1 actions"):
        validate_model_decision(
            {"actions": [{"action": "up", "args": {}}, {"action": "down", "args": {}}]},
            context["legal_action_specs"], max_actions=1,
        )


def test_reasoned_action_contract_requires_categorical_assessment_and_single_action_list(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace)
    run_dir = tmp_path / "run"
    replay_trace(trace, run_dir)
    context = build_analysis_input(run_dir, through_action=4)
    payload = {
        "state_assessment": {
            "player": {"location": "center", "certainty": "inferred", "evidence": "visible in current screen"},
            "target_candidates": [{
                "description": "distinct object", "location": "upper area",
                "relevance": "may be a goal", "certainty": "inferred",
            }],
            "obstacles": [{
                "description": "boundary-like element", "location": "left edge",
                "status": "suspected_blocked", "evidence": "one attempted move had no visible effect",
            }],
            "other_relevant_changes": [],
        },
        "hypothesis": {
            "claim": "Moving down may affect the boundary-like element.",
            "evidence": "the current screen shows it beside the player",
            "uncertainty": "tentative",
        },
        "trajectory_rationale": "Test one legal action to distinguish movement from blockage.",
        "prediction": {"visible_change": "yes", "environment_outcome": "no_terminal", "description": "the visible board changes"},
        "actions": [{"action": "down", "args": {}}],
    }

    normalized = validate_reasoned_action(payload, context["legal_action_specs"], max_actions=1)

    assert normalized["actions"][0]["action"] == "ACTION2"
    assert normalized["hypothesis"]["uncertainty"] == "tentative"
    payload["state_assessment"]["player"]["certainty"] = 0.9
    with pytest.raises(DecisionValidationError, match="certainty must be"):
        validate_reasoned_action(payload, context["legal_action_specs"], max_actions=1)
    with pytest.raises(DecisionValidationError, match="exactly"):
        validate_reasoned_action({"actions": [{"action": "down", "args": {}}]}, context["legal_action_specs"])


def test_offline_analysis_uses_same_reasoned_contract_and_records_full_decision(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace)
    run_dir = tmp_path / "run"
    replay_trace(trace, run_dir)
    client = _FakeClient()

    result = run_analysis(run_dir, client, through_action=4)

    assert result["decision_schema"] == "v10.reasoned_action/1"
    assert result["decision"]["state_assessment"]["player"]["certainty"] == "unknown"
    assert result["decision"]["actions"] == [{"action": "ACTION2", "args": {}}]
    assert "Required response shape (`v10.reasoned_action/1`)" in client.calls[0][1]


def test_model_memory_context_is_capped_at_five_entries(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace)
    run_dir = tmp_path / "run"
    replay_trace(trace, run_dir)

    with pytest.raises(ValueError, match="between 0 and 5"):
        build_analysis_input(run_dir, through_action=4, memory_context_entries=6)


def test_model_decision_rejects_malformed_enum_types_cleanly(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace)
    run_dir = tmp_path / "run"
    replay_trace(trace, run_dir)
    context = build_analysis_input(run_dir, through_action=4)
    decision = _valid_decision()
    decision["actions"][0]["action"] = ["ACTION2"]

    with pytest.raises(DecisionValidationError, match="illegal action"):
        validate_model_decision(
            decision, context["legal_action_specs"],
        )


def test_scoring_stops_at_first_action_divergence() -> None:
    decision = _valid_decision(action="ACTION1")
    plan = decision["actions"]
    plan.append({**plan[0], "prediction_id": "PLAN_2"})
    future = [{
        "action": "ACTION2", "args": {}, "evidence_id": "T10",
        "measurements": {"changed_pixels": 1},
    }]

    matches = score_recorded_plan(decision, future)

    assert matches[0]["status"] == "mismatch"
    assert matches[1]["status"] == "unscorable"
    assert matches[1]["reason"] == "prior action diverged or interrupted the trace"


def test_scoring_matches_actions_without_claiming_policy_quality() -> None:
    decision = _valid_decision(action="ACTION2")
    first = decision["actions"][0]
    second = {**first, "action": "ACTION1"}
    decision["actions"].append(second)
    future = [
        {"action": "ACTION2", "args": {}, "evidence_id": "T10", "measurements": {"changed_pixels": 2}},
        {"action": "ACTION1", "args": {}, "evidence_id": "T11", "measurements": {"changed_pixels": 1}},
    ]

    matches = score_recorded_plan(decision, future)

    assert [item["status"] for item in matches] == ["matched", "matched"]
    assert all("policy" not in item and "prediction" not in item for item in matches)


def test_prediction_type_mismatch_is_inconclusive_not_failed() -> None:
    result = evaluate_prediction(
        {"changed_pixels": "zero or more"}, {"measurements": {"changed_pixels": 2}},
    )
    assert result["verdict"] == "inconclusive"
    assert result["reason"].startswith("invalid prediction:")


def test_measurement_scoring_requires_every_predicted_field() -> None:
    assert evaluate_prediction(
        {"changed_pixels": 1, "missing": 2}, {"measurements": {"changed_pixels": 1}},
    )["verdict"] == "inconclusive"
    assert evaluate_prediction(
        {"candidate_displacement": {"best": {"dy": -2}}},
        {"measurements": {"candidate_displacement": {"best": {"dy": -2, "dx": 0}}}},
    )["verdict"] == "matched"
    assert evaluate_prediction(None, {"measurements": {"changed_pixels": 1}}, "unobservable")["verdict"] == "inconclusive"


def test_structured_measurement_comparisons_and_game_outcomes() -> None:
    actual = {
        "measurements": {"changed_pixels": 3}, "levels_completed": 2, "terminal": False,
    }
    assert evaluate_prediction({"all": [
        {"path": "measurements.changed_pixels", "op": "between", "value": [3, 5]},
        {"path": "levels_completed", "op": "gte", "value": 2},
        {"path": "terminal", "op": "eq", "value": False},
    ]}, actual)["verdict"] == "matched"
    assert evaluate_prediction({"all": [
        {"path": "measurements.changed_pixels", "op": "gt", "value": 3},
    ]}, actual)["verdict"] == "failed"
    assert evaluate_prediction({"all": [
        {"path": "measurements.changed_pixels", "op": "eq", "value": 3.0},
    ]}, actual)["verdict"] == "matched"


def test_prediction_wrong_actual_type_is_inconclusive() -> None:
    result = evaluate_prediction(
        {"all": [{"path": "measurements.changed_pixels", "op": "gt", "value": 0}]},
        {"measurements": {"changed_pixels": "many"}},
    )
    assert result["verdict"] == "inconclusive"
    assert result["unavailable_paths"] == ["measurements.changed_pixels"]


def test_strata_client_sends_labeled_images_and_parses_usage(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(png_bytes(np.zeros((1, 1, 3), dtype=np.uint8)))
    response_body = json.dumps({
        "model": "qwen-test",
        "choices": [{"message": {"content": "{}"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }).encode()

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return response_body

    def fake_urlopen(request, timeout):
        assert request.full_url == "http://strata.example/v1/chat/completions"
        assert request.get_header("Authorization") == "Bearer test-key"
        payload = json.loads(request.data)
        assert payload["reasoning_effort"] == "medium"
        assert payload["reasoning_budget_tokens"] == 128
        assert payload["messages"][1]["content"][1]["text"] == "initial frame"
        assert payload["messages"][1]["content"][2]["image_url"]["url"].startswith("data:image/png;base64,")
        return FakeResponse()

    client = StrataClient(
        "http://strata.example", "qwen-test", api_key="test-key", reasoning_budget_tokens=128,
        log_request=True, log_response=True, log_dir=tmp_path / "run",
    )
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = client.complete("system", "user", [image_path], ["initial frame"])
    assert result["model"] == "qwen-test"
    assert result["usage"]["total_tokens"] == 12
    request_record = json.loads((tmp_path / "run" / "model_requests.jsonl").read_text().splitlines()[0])
    response_record = json.loads((tmp_path / "run" / "model_responses.jsonl").read_text().splitlines()[0])
    assert request_record["payload"]["messages"][0]["content"] == "system"
    assert request_record["payload"]["messages"][1]["content"][0]["text"] == "user"
    assert request_record["system_prompt_characters"] == len("system")
    assert request_record["user_text_characters"] == len("user") + len("initial frame")
    assert request_record["prompt_characters"] == len("system") + len("user") + len("initial frame")
    assert request_record["user_prompt_characters"] == len("user")
    assert request_record["image_count"] == 1
    assert "test-key" not in json.dumps(request_record)
    assert response_record["body"] == response_body.decode()


def test_strata_client_preserves_multimodal_conversation_and_logs_cache_usage(tmp_path: Path) -> None:
    from v10.strata_client import user_message

    image_path = tmp_path / "conversation.png"
    image_path.write_bytes(png_bytes(np.zeros((1, 1, 3), dtype=np.uint8)))
    initial = [
        {"role": "system", "content": "stable system"},
        user_message("first observation", [image_path], ["screen 0"]),
        {"role": "assistant", "content": '{"actions":[{"action":"up","args":{}}]}'},
    ]
    calls = []

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 2,
                          "prompt_tokens_details": {"cached_tokens": 72}},
                "timings": {"prompt_ms": 14.5},
            }).encode()

    def fake_urlopen(request, timeout):
        calls.append(json.loads(request.data)["messages"])
        return FakeResponse()

    client = StrataClient(
        "http://strata.example", "qwen-test", log_request=True, log_response=True,
        log_dir=tmp_path / "conversation-run",
    )
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        first = client.complete("", "", messages=initial, conversation_reset_count=1)
        second_messages = [*initial, {"role": "user", "content": "actual outcome and current screen"}]
        client.complete("", "", messages=second_messages, conversation_reset_count=1)
        third_messages = [*second_messages, {"role": "assistant", "content": "second accepted response"},
                          {"role": "user", "content": "next actual outcome"}]
        client.complete("", "", messages=third_messages, conversation_reset_count=1)

    assert calls[0] == initial
    assert calls[1][:-1] == calls[0]
    assert calls[2][:-2] == calls[1]
    assert calls[0][1]["content"][2] == initial[1]["content"][2]
    assert first["usage"]["cached_prompt_tokens"] == 72
    assert first["usage"]["newly_processed_prompt_tokens"] == 28
    assert first["usage"]["prompt_processing_latency_ms"] == 14.5
    request_rows = [json.loads(row) for row in (tmp_path / "conversation-run/model_requests.jsonl").read_text().splitlines()]
    response_rows = [json.loads(row) for row in (tmp_path / "conversation-run/model_responses.jsonl").read_text().splitlines()]
    assert [row["conversation_message_count"] for row in request_rows] == [3, 4, 6]
    assert [row["conversation_reset_count"] for row in request_rows] == [1, 1, 1]
    assert response_rows[0]["cached_prompt_tokens"] == 72
    assert response_rows[0]["newly_processed_prompt_tokens"] == 28
    assert response_rows[0]["prompt_processing_latency_ms"] == 14.5


def test_model_logging_cli_flags_default_off_and_enable_independently() -> None:
    parser = build_parser()
    defaults = parser.parse_args(["live", "--game-id", "ez01", "--run-dir", "run"])
    enabled = parser.parse_args([
        "live", "--game-id", "ez01", "--run-dir", "run", "--log-request", "--log-response",
    ])

    assert defaults.log_request is False
    assert defaults.log_response is False
    assert defaults.reset_memory is False
    assert defaults.global_memory_max_chars == 6000
    assert defaults.plan_horizon == 8
    assert defaults.max_output_tokens == 6144
    with pytest.raises(SystemExit):
        parser.parse_args([
            "live", "--game-id", "ez01", "--run-dir", "run", "--review-interval", "20",
        ])
    assert enabled.log_request is True
    assert enabled.log_response is True
    reset = parser.parse_args([
        "live", "--game-id", "ez01", "--run-dir", "run", "--reset-memory",
    ])
    assert reset.reset_memory is True
    configured_memory = parser.parse_args([
        "live", "--game-id", "ez01", "--run-dir", "run", "--global-memory-max-chars", "7200",
    ])
    assert configured_memory.global_memory_max_chars == 7200
    configured_horizon = parser.parse_args([
        "live", "--game-id", "ez01", "--run-dir", "run", "--plan-horizon", "8",
    ])
    assert configured_horizon.plan_horizon == 8


def test_cli_reports_v10_patch_version_and_context_override(capsys) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as version_exit:
        parser.parse_args(["--version"])
    assert version_exit.value.code == 0
    assert capsys.readouterr().out.strip() == "v10 10.0.3"

    configured = parser.parse_args([
        "live", "--game-id", "ez01", "--run-dir", "run", "--context-window-tokens", "32768",
    ])
    assert configured.context_window_tokens == 32768


def test_strata_context_window_is_discovered_from_model_metadata(monkeypatch) -> None:
    body = json.dumps({"data": [{"id": "qwen-test", "meta": {"n_ctx": 32768}}]}).encode()

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return body

    def fake_urlopen(request, timeout):
        assert request.full_url == "http://strata.example/v1/models"
        return FakeResponse()

    client = StrataClient("http://strata.example", "qwen-test")
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        assert client.discover_context_window_tokens() == 32768
    assert client.context_window_tokens == 32768


def test_live_reset_memory_flag_clears_persistent_memory_before_run(tmp_path: Path, monkeypatch) -> None:
    from v10 import arcade_env

    state_dir = tmp_path / "skills"
    (state_dir / "games/ez01").mkdir(parents=True)
    (state_dir / "games/ez01/MEMORY.md").write_text("game knowledge")
    (state_dir / "global").mkdir()
    (state_dir / "global/MEMORY.md").write_text("global knowledge")
    (state_dir / "global/notes.txt").write_text("keep")

    class FakeClient:
        def __init__(self, **kwargs):
            pass

    class FakeEnv:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

    def fake_run_live(env, run_dir, client, **kwargs):
        assert not (state_dir / "games").exists()
        assert not (state_dir / "global/MEMORY.md").exists()
        assert (state_dir / "global/notes.txt").read_text() == "keep"
        return {"started": True}

    monkeypatch.setattr(cli_module, "StrataClient", FakeClient)
    monkeypatch.setattr(cli_module, "run_live", fake_run_live)
    monkeypatch.setattr(arcade_env, "ArcadeGameEnv", FakeEnv)

    result = cli_module.main([
        "live", "--game-id", "ez01", "--run-dir", str(tmp_path / "run"),
        "--state-dir", str(state_dir), "--reset-memory",
    ])

    assert result == 0


def test_strata_client_falls_back_when_strata_rejects_json_mode_output(tmp_path: Path) -> None:
    response_body = json.dumps({
        "model": "qwen-test",
        "choices": [{"message": {"content": json.dumps(_valid_decision())}}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35},
    }).encode()
    failure_body = json.dumps({
        "error": {"code": "structured_output_failed", "message": "model did not return valid JSON"},
    }).encode()
    payloads = []

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return response_body

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data)
        payloads.append(payload)
        if len(payloads) == 1:
            raise urllib.error.HTTPError(
                request.full_url, 502, "Bad Gateway", {}, io.BytesIO(failure_body),
            )
        return FakeResponse()

    client = StrataClient(
        "http://strata.example", "qwen-test", api_key="test-key",
        log_request=True, log_response=True, log_dir=tmp_path,
    )
    conversation = [
        {"role": "system", "content": "stable system"},
        {"role": "user", "content": "return JSON"},
    ]
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = client.complete("system", "return JSON", (), (), messages=conversation)

    assert result["request_attempts"] == 2
    assert "response_format" in payloads[0]
    assert "response_format" not in payloads[1]
    retry_content = payloads[1]["messages"][1]["content"]
    retry_text = retry_content if isinstance(retry_content, str) else "\n".join(
        block["text"] for block in retry_content if isinstance(block, dict) and block.get("type") == "text"
    )
    assert "previous generation did not produce valid JSON" in retry_text
    assert result["submitted_messages"] == payloads[1]["messages"]
    logged_requests = [json.loads(line) for line in (tmp_path / "model_requests.jsonl").read_text().splitlines()]
    logged_responses = [json.loads(line) for line in (tmp_path / "model_responses.jsonl").read_text().splitlines()]
    assert len(logged_requests) == len(logged_responses) == 2
    assert [row["status"] for row in logged_responses] == [502, 200]
    assert result["request_ids"] == [row["request_id"] for row in logged_requests]
    assert all(row["payload_bytes"] > 0 and row["image_count"] == 0 for row in logged_requests)
    assert all(row["latency_seconds"] >= 0 and row["timestamp_utc"] for row in logged_responses)


def test_strata_client_can_omit_json_response_format_for_review_markdown(monkeypatch) -> None:
    payloads = []

    class FakeResponse:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self):
            return json.dumps({"choices": [{"message": {"content": "## Patterns\\nObserved."}}]}).encode()

    def fake_urlopen(request, **kwargs):
        payloads.append(json.loads(request.data))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = StrataClient("http://strata.example", "qwen-test")

    response = client.complete("review", "return Markdown", response_format=None)

    assert "response_format" not in payloads[0]
    assert response["content"] == "## Patterns\\nObserved."


@pytest.mark.parametrize("body", [
    '{"choices":[{"message":null}]}',
    '{"choices":[null]}',
    '{"choices":[]}',
    '[]',
])
def test_strata_malformed_envelope_raises_request_error(monkeypatch, body: str) -> None:
    class FakeResponse:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self): return body.encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: FakeResponse())
    client = StrataClient("http://strata.example", "qwen-test")
    with pytest.raises(StrataRequestError, match="invalid Strata response"):
        client.complete("system", "user")


def test_offline_memory_limit_changes_the_actual_prompt(tmp_path):
    from v10.tools.memory import game_memory_dir, save_game_memory
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace)
    run = tmp_path / "replay"
    replay_trace(trace, run)
    state = tmp_path / "state"
    save_game_memory(game_memory_dir(state, "fixture"), "First observation.\n\nSecond observation.\n\nThird observation.")
    none = build_analysis_input(run, through_action=2, state_dir=state, memory_context_entries=0)
    one = build_analysis_input(run, through_action=2, state_dir=state, memory_context_entries=1)
    all_memory = build_analysis_input(run, through_action=2, state_dir=state, memory_context_entries=5)
    assert none["context"]["memory"] == ""
    assert one["context"]["memory"] == "First observation.\n"
    assert "Third observation." in all_memory["context"]["memory"]
    assert len({row["user_prompt"] for row in (none, one, all_memory)}) == 3
