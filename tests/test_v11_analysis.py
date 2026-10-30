from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from v11.analysis import (
    DecisionValidationError,
    build_analysis_input,
    run_analysis,
    score_recorded_plan,
    validate_model_decision,
    validate_reasoned_action,
)
from v11 import cli as cli_module
from v11.cli import build_parser
from v11.frames import FrameStore, png_bytes
from v11.evaluation import run_ablation
from v11.replay import replay_trace
from v11.strata_client import (
    DEFAULT_STRATA_BASE_URL,
    DEFAULT_STRATA_MODEL,
    DEFAULT_VLLM_BASE_URL,
    DEFAULT_VLLM_MODEL,
    StrataClient,
    StrataRequestError,
)
from v11.tools.game import evaluate_prediction
from v11.tools.history import History


@pytest.fixture(autouse=True)
def _isolated_memory_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("V11_STATE_DIR", str(tmp_path / "state"))


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

    client = _FakeClient()
    client.backend = "vllm"
    result = run_ablation(run_dir, client, through_action=4, horizon=2)

    assert [arm["conditioning"] for arm in result["arms"]] == [
        profile for profile in ("pixels_actions", "pixels_actions_memory")
        for _ in (False, True)
    ]
    assert [arm["include_prior_decisions"] for arm in result["arms"]] == [False, True] * 2
    assert all(arm["status"] == "ok" for arm in result["arms"])
    assert result["backend"] == "vllm"
    assert all(arm["backend"] == "vllm" for arm in result["arms"])
    assert all(arm["profile_id"] == "generic" and arm["action_signature"] == "ACTION1,ACTION2" for arm in result["arms"])
    assert all(arm["prompt_revision"] and arm["memory_revision"] for arm in result["arms"])
    assert not (run_dir / "decisions.jsonl").exists()
    assert (run_dir / "ablation_runs.jsonl").exists()


def test_run_analysis_scores_matching_recorded_action_after_model_call(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace)
    run_dir = tmp_path / "run"
    replay_trace(trace, run_dir)
    client = _FakeClient()
    client.backend = "vllm"

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
    assert result["backend"] == "vllm"
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
        },
        "trajectory_rationale": "Test one legal action to distinguish movement from blockage.",
        "prediction": {"visible_change": "yes", "environment_outcome": "no_terminal", "description": "the visible board changes"},
        "actions": [{"action": "down", "args": {}}],
    }

    normalized = validate_reasoned_action(payload, context["legal_action_specs"], max_actions=1)

    assert normalized["actions"][0]["action"] == "ACTION2"
    assert normalized["hypothesis"] == {
        "claim": "Moving down may affect the boundary-like element.",
        "evidence": "the current screen shows it beside the player",
    }
    payload["prediction"]["environment_outcome"] = "NOT_FINISHED"
    normalized = validate_reasoned_action(payload, context["legal_action_specs"], max_actions=1)
    assert normalized["prediction"]["environment_outcome"] == "no_terminal"
    payload["prediction"]["environment_outcome"] = "no_terminal"
    payload["hypothesis"]["uncertainty"] = "unknown"
    with pytest.raises(DecisionValidationError, match="hypothesis must contain claim and evidence"):
        validate_reasoned_action(payload, context["legal_action_specs"], max_actions=1)
    del payload["hypothesis"]["uncertainty"]
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


def test_vllm_client_preserves_messages_and_uses_openai_compatible_fields(monkeypatch, tmp_path: Path) -> None:
    payloads = []
    history = [
        {"role": "system", "content": "unchanged system prompt"},
        {"role": "user", "content": [
            {"type": "text", "text": "unchanged user prompt"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
        ]},
    ]
    tools = [{"type": "function", "function": {"name": "inspect_region", "parameters": {"type": "object"}}}]

    class FakeResponse:
        status = 200

        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self):
            return json.dumps({
                "model": "custom/served-model",
                "choices": [{"message": {"content": None, "tool_calls": [
                    {"id": "call-vllm", "type": "function", "function": {"name": "inspect_region", "arguments": "{}"}},
                ]}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
            }).encode()

    def fake_urlopen(request, **kwargs):
        assert request.full_url == "http://vllm.example/v1/chat/completions"
        payloads.append(json.loads(request.data))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = StrataClient(
        "http://vllm.example", "custom/served-model", backend="vllm",
        reasoning_effort="high", reasoning_budget_tokens=128,
        log_request=True, log_response=True, log_dir=tmp_path,
    )

    result = client.complete("ignored", "ignored", messages=history, tools=tools)

    payload = payloads[0]
    assert payload["model"] == "custom/served-model"
    assert payload["messages"] == history
    assert payload["tools"] == tools
    assert payload["tool_choice"] == "auto"
    assert payload["parallel_tool_calls"] is False
    assert payload["max_tokens"] == 6144
    assert "reasoning_effort" not in payload
    assert "reasoning_budget_tokens" not in payload
    assert result["backend"] == "vllm"
    assert result["tool_calls"][0]["id"] == "call-vllm"

    client.complete("ignored", "ignored", messages=history, response_format="json_object")
    assert payloads[1]["messages"] == history
    assert payloads[1]["response_format"] == {"type": "json_object"}
    request_record = json.loads((tmp_path / "model_requests.jsonl").read_text().splitlines()[0])
    response_record = json.loads((tmp_path / "model_responses.jsonl").read_text().splitlines()[0])
    assert request_record["backend"] == response_record["backend"] == "vllm"


def test_vllm_client_discovers_max_model_len(monkeypatch) -> None:
    body = json.dumps({"data": [{"id": "custom/served-model", "max_model_len": 131072}]}).encode()

    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self): return body

    def fake_urlopen(request, timeout):
        assert request.full_url == "http://vllm.example/v1/models"
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = StrataClient("http://vllm.example", "custom/served-model", backend="vllm")

    assert client.discover_context_window_tokens() == 131072


def test_vllm_client_uses_vllm_api_key_environment(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_API_KEY", "vllm-key")
    monkeypatch.setenv("STRATA_API_KEY", "strata-key")

    client = StrataClient("http://vllm.example", "custom/served-model", backend="vllm")

    assert client.api_key == "vllm-key"


def test_vllm_client_surfaces_endpoint_errors(monkeypatch) -> None:
    def reject_request(request, **kwargs):
        raise urllib.error.HTTPError(
            request.full_url, 400, "Bad Request", {}, io.BytesIO(b'{"error":"unsupported model"}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", reject_request)
    client = StrataClient("http://vllm.example", "custom/text-model", backend="vllm")

    with pytest.raises(StrataRequestError, match="vLLM HTTP 400.*unsupported model"):
        client.complete("system", "user")


def test_strata_client_preserves_multimodal_conversation_and_logs_cache_usage(tmp_path: Path) -> None:
    from v11.strata_client import user_message

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
    assert defaults.profile_memory_max_chars == 4500
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
        "live", "--game-id", "ez01", "--run-dir", "run", "--profile-memory-max-chars", "4200",
    ])
    assert configured_memory.profile_memory_max_chars == 4200
    with pytest.raises(SystemExit):
        parser.parse_args([
            "live", "--game-id", "ez01", "--run-dir", "run", "--profile-memory-max-chars", "4501",
        ])
    configured_horizon = parser.parse_args([
        "live", "--game-id", "ez01", "--run-dir", "run", "--plan-horizon", "8",
    ])
    assert configured_horizon.plan_horizon == 8


def test_cli_reports_v10_patch_version_and_context_override(capsys) -> None:
    parser = build_parser()
    defaults = parser.parse_args(["live", "--game-id", "ez01", "--run-dir", "run"])
    assert defaults.discovery_tools is True
    assert defaults.vllm is False
    assert cli_module._selected_model_backend(defaults) == (
        "strata", DEFAULT_STRATA_BASE_URL, DEFAULT_STRATA_MODEL,
    )
    disabled = parser.parse_args(["live", "--game-id", "ez01", "--run-dir", "run", "--no-discovery-tools"])
    assert disabled.discovery_tools is False
    with pytest.raises(SystemExit) as version_exit:
        parser.parse_args(["--version"])
    assert version_exit.value.code == 0
    assert capsys.readouterr().out.strip() == "v11 11.1.6"

    configured = parser.parse_args([
        "live", "--game-id", "ez01", "--run-dir", "run", "--context-window-tokens", "32768",
        "--discovery-tools",
    ])
    assert configured.context_window_tokens == 32768
    assert configured.discovery_tools is True

    vllm = parser.parse_args([
        "live", "--game-id", "ez01", "--run-dir", "run", "--vllm",
    ])
    assert cli_module._selected_model_backend(vllm) == (
        "vllm", DEFAULT_VLLM_BASE_URL, DEFAULT_VLLM_MODEL,
    )
    override = parser.parse_args([
        "live", "--game-id", "ez01", "--run-dir", "run", "--vllm",
        "--base-url", "http://custom-vllm:9000/v1", "--model", "org/any-model",
    ])
    assert cli_module._selected_model_backend(override) == (
        "vllm", "http://custom-vllm:9000/v1", "org/any-model",
    )


@pytest.mark.parametrize("command,required", [
    ("analyze", ["--run-dir", "run"]),
    ("ablate", ["--run-dir", "run"]),
    ("optimize", ["--run-dir", "run"]),
    ("live", ["--game-id", "ez01", "--run-dir", "run"]),
])
def test_model_backed_cli_commands_accept_vllm_and_model(command: str, required: list[str]) -> None:
    args = build_parser().parse_args([command, *required, "--vllm", "--model", "Qwen/custom"])

    assert args.vllm is True
    assert args.model == "Qwen/custom"


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
    from v11 import arcade_env

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


def test_live_cli_passes_vllm_backend_and_model_to_client(tmp_path: Path, monkeypatch) -> None:
    from v11 import arcade_env

    received = {}

    class CapturingClient:
        def __init__(self, **kwargs):
            received.update(kwargs)

    class FakeEnv:
        def __init__(self, *args, **kwargs): pass
        def close(self): pass

    monkeypatch.setattr(cli_module, "StrataClient", CapturingClient)
    monkeypatch.setattr(cli_module, "run_live", lambda *args, **kwargs: {"started": True})
    monkeypatch.setattr(arcade_env, "ArcadeGameEnv", FakeEnv)

    result = cli_module.main([
        "live", "--game-id", "ez01", "--run-dir", str(tmp_path / "run"),
        "--vllm", "--base-url", "http://vllm.example:8000/v1", "--model", "org/served-model",
    ])

    assert result == 0
    assert received["backend"] == "vllm"
    assert received["base_url"] == "http://vllm.example:8000/v1"
    assert received["model"] == "org/served-model"


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


def test_strata_client_sends_native_tools_and_parses_tool_calls(monkeypatch) -> None:
    payloads = []

    class FakeResponse:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self):
            return json.dumps({"choices": [{"message": {"content": None, "tool_calls": [
                {"id": "call-1", "type": "function", "function": {"name": "inspect_region", "arguments": "{}"}},
            ]}, "finish_reason": "tool_calls"}]}).encode()

    def fake_urlopen(request, **kwargs):
        payloads.append(json.loads(request.data))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = StrataClient("http://strata.example", "qwen-test")
    history = [{"role": "system", "content": "sys"}, {"role": "user", "content": "act"}]
    schema = [{"type": "function", "function": {"name": "inspect_region", "parameters": {"type": "object"}}}]

    response = client.complete("sys", "act", messages=history, tools=schema)

    assert "response_format" not in payloads[0]
    assert payloads[0]["tools"] == schema
    assert payloads[0]["tool_choice"] == "auto"
    assert payloads[0]["parallel_tool_calls"] is False
    assert response["tool_calls"][0]["id"] == "call-1"
    assert response["finish_reason"] == "tool_calls"


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
    from v11.tools.memory import GameMemory, game_memory_dir, save_game_memory
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace)
    run = tmp_path / "replay"
    replay_trace(trace, run)
    state = tmp_path / "state"
    store_dir = game_memory_dir(state, "fixture")
    saved = "First observation.\n\nSecond observation.\n\nThird observation."
    save_game_memory(store_dir, saved)
    GameMemory(run, store_dir=store_dir, game_id="fixture").record_markdown_revision("game", saved, step=2)
    none = build_analysis_input(run, through_action=2, state_dir=state, memory_context_entries=0)
    one = build_analysis_input(run, through_action=2, state_dir=state, memory_context_entries=1)
    all_memory = build_analysis_input(run, through_action=2, state_dir=state, memory_context_entries=5)
    assert none["context"]["memory"] == ""
    assert one["context"]["memory"] == "First observation.\n"
    assert "Third observation." in all_memory["context"]["memory"]
    assert len({row["user_prompt"] for row in (none, one, all_memory)}) == 3


def _write_click_profile_trace(path: Path, *, change_after_prefix: bool = False) -> None:
    rows = []
    for index in range(4):
        if index == 0:
            action_id, available = "RESET", ["ACTION6"]
        else:
            action_id = "ACTION6"
            available = ["ACTION6", "ACTION5"] if change_after_prefix and index >= 3 else ["ACTION6"]
        rows.append({
            "timestamp": f"2028-01-01T00:00:0{index}Z",
            "data": {
                "timestamp": f"2028-01-01T00:00:0{index}Z",
                "game_id": "click-fixture", "state": "NOT_FINISHED",
                "levels_completed": 0, "win_levels": 1, "guid": "click-fixture",
                "full_reset": index == 0,
                "action_input": {"id": action_id, "data": {} if index == 0 else {"x": index, "y": index}},
                "available_actions": available,
                "frame": [[index, 0], [0, 0]],
            },
        })
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_offline_profile_prompt_matches_live_capability_composition(tmp_path: Path) -> None:
    from v11.live import _live_context
    from v11.tools.memory import game_memory_dir, shared_skill_path
    from v11.tools.game import GameTools

    trace = tmp_path / "click.jsonl"
    _write_click_profile_trace(trace)
    run = tmp_path / "run"
    replay_trace(trace, run)
    offline = build_analysis_input(run, through_action=2, state_dir=tmp_path / "state")

    class ClickEnv:
        legal_actions = [{"id": "ACTION6", "parameters": {
            "type": "object", "properties": {"x": {"type": "integer", "minimum": 0, "maximum": 1},
                                                 "y": {"type": "integer", "minimum": 0, "maximum": 1}},
            "required": ["x", "y"], "additionalProperties": False,
        }}]
        def observe(self):
            return {"frame": np.zeros((2, 2), dtype=np.uint8), "legal_actions": self.legal_actions,
                    "game_state": "NOT_FINISHED", "level": 0, "levels_completed": 0,
                    "win_levels": 1, "terminal": False}
        def act(self, action, args): return {}

    tools = GameTools(ClickEnv(), tmp_path / "live")
    current = tools.observe()
    live = _live_context(
        tools, current, "click-fixture", 2, shared_skill_path(),
        game_memory_dir(tmp_path / "state", "click-fixture"), "live", 5,
    )
    capability = (Path(__file__).parents[1] / "skills/v11/profile_capabilities/a6_click_only.md").read_text().strip()
    discrete = (Path(__file__).parents[1] / "skills/v11/profile_capabilities/discrete_control.md").read_text().strip()
    assert offline["profile_id"] == live["context"]["action_profile"]["profile_id"] == "a6"
    assert capability in offline["system_prompt"] and capability in live["system_prompt"]
    assert "Keep these observations separate" in live["system_prompt"]
    assert "hypothesis.evidence 180 characters" in live["system_prompt"]
    assert "movement" not in offline["system_prompt"].casefold()
    assert "movement" not in live["system_prompt"].casefold()
    assert discrete not in offline["system_prompt"] and discrete not in live["system_prompt"]
    assert offline["system_prompt"] == live["system_prompt"]


def test_offline_movement_profile_excludes_click_capability(tmp_path: Path) -> None:
    trace = tmp_path / "movement.jsonl"
    _write_trace(trace)
    run = tmp_path / "run"
    replay_trace(trace, run)
    offline = build_analysis_input(run, through_action=2)
    capability_dir = Path(__file__).parents[1] / "skills/v11/profile_capabilities"
    click = (capability_dir / "coordinate_action6.md").read_text().strip()
    movement = (capability_dir / "discrete_control.md").read_text().strip()
    assert movement in offline["system_prompt"]
    assert click not in offline["system_prompt"]


def test_offline_prefix_isolates_actions_and_game_memory_revisions(tmp_path: Path) -> None:
    from v11.tools.memory import GameMemory, game_memory_dir

    trace = tmp_path / "prefix.jsonl"
    _write_click_profile_trace(trace, change_after_prefix=True)
    run = tmp_path / "run"
    replay_trace(trace, run)
    state = tmp_path / "state"
    store_dir = game_memory_dir(state, "click-fixture")
    memory = GameMemory(run, store_dir=store_dir, game_id="click-fixture")
    memory.record_markdown_revision("game", "Known at action two.", step=2)
    memory.record_markdown_revision("game", "Known only later.", step=3)

    early = build_analysis_input(run, through_action=2, state_dir=state)
    later = build_analysis_input(run, through_action=3, state_dir=state)
    assert len(early["prefix_transitions"]) == 2
    assert len(later["prefix_transitions"]) == 3
    assert early["context"]["memory"] == "Known at action two.\n"
    assert later["context"]["memory"] == "Known only later.\n"
    assert early["context"]["current_state"]["legal_actions"] == [{
        "id": "click", "parameters": {"type": "object", "properties": {
            "x": {"type": "integer", "minimum": 0, "maximum": 1},
            "y": {"type": "integer", "minimum": 0, "maximum": 1}},
            "required": ["x", "y"], "additionalProperties": False},
    }]
    assert early["profile_id"] == "a6"
    assert later["profile_id"] == "a56"
    assert "Known only later." not in early["user_prompt"]
    assert "ACTION5" not in early["user_prompt"]
    assert early["memory_revision"]["game"]["step"] == 2


def test_offline_profile_memory_is_scoped_and_prefix_timestamped(tmp_path: Path) -> None:
    from v11.tools.memory import archive_profile_memory_revision, profile_memory_path

    trace = tmp_path / "profile-memory.jsonl"
    _write_click_profile_trace(trace)
    run = tmp_path / "run"
    replay_trace(trace, run)
    state = tmp_path / "state"
    archive_profile_memory_revision(state, "a6", "# Earlier click method\n\n- Compare one outcome.", timestamp_utc="2027-01-01T00:00:00+00:00")
    archive_profile_memory_revision(state, "a6", "# Later click method\n\n- Learned after this prefix.", timestamp_utc="2029-01-01T00:00:00+00:00")
    archive_profile_memory_revision(state, "a5", "# Other profile\n\n- Must stay isolated.", timestamp_utc="2027-01-01T00:00:00+00:00")
    profile_memory_path(state, "a6").write_text("# Current unversioned\n\n- Must not leak.", encoding="utf-8")

    offline = build_analysis_input(run, through_action=2, state_dir=state)
    assert offline["context"]["profile_memory"] == "# Earlier click method\n\n- Compare one outcome.\n"
    assert "Must stay isolated" not in offline["user_prompt"]
    assert "Learned after this prefix" not in offline["user_prompt"]
    assert "Current unversioned" not in offline["user_prompt"]
    assert offline["memory_revision"]["profile"]["status"] == "historical_snapshot"


def test_pixels_actions_omits_both_saved_memory_scopes_and_analysis_records_revisions(tmp_path: Path) -> None:
    from v11.tools.memory import GameMemory, archive_profile_memory_revision, game_memory_dir

    trace = tmp_path / "memory-ablation.jsonl"
    _write_click_profile_trace(trace)
    run = tmp_path / "run"
    replay_trace(trace, run)
    state = tmp_path / "state"
    GameMemory(run, store_dir=game_memory_dir(state, "click-fixture")).record_markdown_revision(
        "game", "Game-only saved knowledge.", step=2,
    )
    archive_profile_memory_revision(state, "a6", "# Approved profile method\n\n- Reuse verified mechanics.", timestamp_utc="2027-01-01T00:00:00+00:00")
    no_memory = build_analysis_input(run, through_action=2, state_dir=state, conditioning="pixels_actions")
    with_memory = build_analysis_input(run, through_action=2, state_dir=state, conditioning="pixels_actions_memory")
    assert no_memory["context"]["memory"] is None
    assert no_memory["context"]["profile_memory"] is None
    assert with_memory["context"]["memory"]
    assert with_memory["context"]["profile_memory"].startswith("# Approved profile method")

    result = run_analysis(
        run, _FakeClient(decision=_reasoned_decision("click", {"x": 1, "y": 1})),
        through_action=2, state_dir=state, write_decision=False,
    )
    assert result["profile_id"] == "a6"
    assert result["action_signature"] == "ACTION6"
    assert result["profile_prompt_revision"]
    assert result["prompt_revision"]
    assert result["memory_revision"]["game"]["step"] == 2
    assert result["memory_revision"]["profile"]["status"] == "historical_snapshot"
