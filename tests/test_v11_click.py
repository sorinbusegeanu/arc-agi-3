from __future__ import annotations

import json

import numpy as np
import pytest

from v11.analysis import (
    DecisionValidationError, model_action_aliases, reasoned_action_contract_description, validate_model_decision,
)
from v11.live import (
    _action_conversation_observation, _action_observation_events, _click_outcome_ledger_entry,
    _click_prediction_mismatch,
    _action_transition_summary, _compact_transition_measurements, _ineffective_action_keys, _live_context,
    _review_events, _transition_is_noop,
)
from v11.tools.game import GameTools
from v11.tools.memory import game_memory_dir, shared_skill_path
from v11.tools.vision import click_effect, propose_click_candidates


def test_click_candidate_proposals_are_native_bounded_and_state_local() -> None:
    frame = np.zeros((20, 30), dtype=np.uint8)
    frame[3, 4] = 2
    frame[10:13, 20:23] = 3
    result = propose_click_candidates(frame, "F00007", max_candidates=12)
    assert result["schema"] == "v11.click_candidates/1"
    assert result["source_frame_id"] == "F00007"
    assert result["frame_size"] == {"width": 30, "height": 20}
    assert result["coordinate_system"] == "native_zero_based_xy"
    assert 1 <= len(result["items"]) <= 12
    assert any(item["basis"] == "distinct_pixel_region" for item in result["items"])
    for item in result["items"]:
        assert 0 <= item["x"] < 30 and 0 <= item["y"] < 20
    first = result["items"][0]
    without_first = propose_click_candidates(
        frame, "F00008", previous_attempts=[(first["x"], first["y"])],
    )
    assert all((row["x"], row["y"]) != (first["x"], first["y"]) for row in without_first["items"])


def test_click_alias_is_available_only_when_action6_is_legal() -> None:
    click_spec = {
        "id": "ACTION6", "parameters": {
            "type": "object", "properties": {
                "x": {"type": "integer", "minimum": 0, "maximum": 4},
                "y": {"type": "integer", "minimum": 0, "maximum": 3},
            }, "required": ["x", "y"], "additionalProperties": False,
        },
    }
    assert model_action_aliases([click_spec]) == {"click": "ACTION6"}
    assert validate_model_decision({"actions": [{"action": "click", "args": {"x": 2, "y": 1}}]}, [click_spec]) == {
        "actions": [{"action": "ACTION6", "args": {"x": 2, "y": 1}}],
    }
    assert model_action_aliases([]) == {}
    with pytest.raises(DecisionValidationError, match="model-facing name 'click'") as exc_info:
        validate_model_decision({"actions": [{"action": "ACTION6", "args": {"x": 2, "y": 1}}]}, [click_spec])
    assert "ACTION6" not in str(exc_info.value)


def test_game_tools_records_local_remote_click_effect(tmp_path) -> None:
    class Env:
        def __init__(self):
            self.frame = np.zeros((8, 9), dtype=np.uint8)
            self.legal_actions = [{
                "id": "ACTION6", "parameters": {
                    "type": "object", "properties": {
                        "x": {"type": "integer", "minimum": 0, "maximum": 8},
                        "y": {"type": "integer", "minimum": 0, "maximum": 7},
                    }, "required": ["x", "y"], "additionalProperties": False,
                },
            }]

        def observe(self):
            return {"frame": self.frame.copy(), "legal_actions": self.legal_actions, "level": 0,
                    "levels_completed": 0, "terminal": False, "game_state": "NOT_FINISHED"}

        def act(self, action, args):
            self.frame[args["y"], args["x"]] = 1
            self.frame[0, 0] = 2
            return {}

    tools = GameTools(Env(), tmp_path / "click-run", max_actions=2)
    tools.observe()
    result = tools.act("ACTION6", {"x": 4, "y": 3})
    effect = result["measurements"]["click_effect"]
    assert effect["schema"] == "v11.click_effect/1"
    assert (effect["x"], effect["y"]) == (4, 3)
    assert effect["local_changed_pixels"] == 1
    assert effect["remote_changed_pixels"] == 1
    assert effect["total_changed_pixels"] == 2
    assert result["evidence"]["args"] == {"x": 4, "y": 3}


def test_click_effect_distinguishes_local_from_remote_without_progress_claims() -> None:
    before = np.zeros((5, 5), dtype=np.uint8)
    after = before.copy()
    after[2, 2] = 1
    after[0, 0] = 1
    effect = click_effect(before, after, 2, 2, radius=0, level_changed=True)
    assert effect["local_changed_pixels"] == 1
    assert effect["remote_changed_pixels"] == 1
    assert effect["level_changed"] is True
    assert "player" not in effect


def test_click_effect_separates_ui_pixels_from_board_pixels_and_requires_matching_mask() -> None:
    before = np.zeros((8, 9), dtype=np.uint8)
    after = before.copy()
    after[0, 0] = 1
    after[4, 5] = 2
    mask = {"frame_shape": [8, 9], "excluded_regions": [{"name": "hud", "bounds": [0, 0, 1, 9]}]}
    measured = click_effect(before, after, 5, 4, ui_mask=mask)
    assert measured["total_changed_pixels"] == 2
    assert measured["board_measurement_status"] == "available"
    assert measured["board_changed_pixels"] == 1
    assert measured["excluded_changed_pixels"] == 1

    ui_frame = before.copy()
    ui_frame[0, 2] = 1
    ui_only = click_effect(before, ui_frame, 5, 4, ui_mask=mask)
    assert ui_only["board_changed_pixels"] == 0
    assert ui_only["excluded_changed_pixels"] == 1

    unavailable = click_effect(before, after, 5, 4)
    assert unavailable["board_measurement_status"] == "mask_unavailable"
    assert unavailable["board_changed_pixels"] is None
    assert not _transition_is_noop({"changed_pixels": 0, "click_effect": unavailable}, "ACTION6")
    mismatch = click_effect(before, after, 5, 4, ui_mask={**mask, "frame_shape": [64, 64]})
    assert mismatch["board_measurement_status"] == "frame_shape_mismatch"
    assert mismatch["board_changed_pixels"] is None


def test_click_prediction_comparison_ignores_raw_pixel_counts_without_board_measurement() -> None:
    prediction = {"board_change": "yes", "environment_outcome": "unknown", "description": "board changes"}
    assert _click_prediction_mismatch(
        prediction, {}, {}, {"changed_pixels": 0, "click_effect": {"board_measurement_status": "mask_unavailable"}},
    ) is None
    assert _click_prediction_mismatch(
        prediction, {}, {}, {"changed_pixels": 18, "click_effect": {
            "board_measurement_status": "available", "board_changed_pixels": 0,
        }},
    ) == "predicted a board change, but the configured board region was unchanged"


def test_ui_only_click_change_is_not_reported_as_board_effect_and_noop_matching_ignores_ui(tmp_path) -> None:
    class Env:
        def __init__(self):
            self.frame = np.zeros((64, 64), dtype=np.uint8)
            self.legal_actions = [{"id": "ACTION6", "parameters": {
                "type": "object", "properties": {
                    "x": {"type": "integer", "minimum": 0, "maximum": 63},
                    "y": {"type": "integer", "minimum": 0, "maximum": 63},
                }, "required": ["x", "y"], "additionalProperties": False,
            }}]

        def observe(self):
            return {"frame": self.frame.copy(), "legal_actions": self.legal_actions,
                    "game_state": "NOT_FINISHED", "level": 0, "levels_completed": 0, "terminal": False}

        def act(self, action, args):
            self.frame[1, 54] = 1
            return {}

    tools = GameTools(Env(), tmp_path / "ui-only", game_id="sy01")
    tools.observe()
    result = tools.act("ACTION6", {"x": 54, "y": 1})
    effect = result["evidence"]["measurements"]["click_effect"]
    assert effect["board_measurement_status"] == "available"
    assert effect["board_changed_pixels"] == 0
    assert effect["excluded_changed_pixels"] == 1
    assert "no board change; excluded UI changed" in _action_transition_summary(result["evidence"], 1)
    assert _ineffective_action_keys(
        [result["evidence"]], result["observation"], tools.frames, tools.click_ui_mask,
    ) == {("ACTION6", '{"x": 54, "y": 1}')}


def test_click_coordinates_and_distinct_targets_survive_compact_summaries() -> None:
    first = {"action": "ACTION6", "args": {"x": 12, "y": 8}, "measurements": {"changed_pixels": 2}}
    second = {"action": "ACTION6", "args": {"x": 2, "y": 3}, "measurements": {"changed_pixels": 2}}
    assert _action_transition_summary(first, 1) == "1 click(x=12,y=8): visible change"
    assert _action_transition_summary(second, 2) == "2 click(x=2,y=3): visible change"
    assert _action_transition_summary(first, 1) != _action_transition_summary(second, 2)
    assert _action_transition_summary({"action": "ACTION5", "args": {"mode": "toggle"}}, 3).startswith(
        '3 ACTION5(args={"mode":"toggle"}):'
    )


def test_compact_click_effect_keeps_existing_local_remote_measurements_only() -> None:
    compact = _compact_transition_measurements({
        "changed_pixels": 7,
        "click_effect": {
            "schema": "v11.click_effect/1", "x": 12, "y": 8, "radius": 2,
            "local_changed_pixels": 3, "remote_changed_pixels": 4,
            "total_changed_pixels": 7, "level_changed": True, "terminal_changed": False,
            "board_measurement_status": "available", "board_changed_pixels": 5,
            "excluded_changed_pixels": 2,
        },
        "raw_pixels": [[1, 2, 3]],
    })
    assert compact == {
        "changed_pixels": 7,
        "click_effect": {
            "local_changed_pixels": 3, "remote_changed_pixels": 4,
            "board_measurement_status": "available", "board_changed_pixels": 5,
            "excluded_changed_pixels": 2,
            "total_changed_pixels": 7, "level_changed": True, "terminal_changed": False,
        },
    }


def test_click_args_and_effect_survive_continued_bootstrap_review_and_reinitialized_context(tmp_path) -> None:
    class ClickEnv:
        def __init__(self):
            self.frame = np.zeros((12, 16), dtype=np.uint8)
            self.legal_actions = [{"id": "ACTION6", "parameters": {
                "type": "object", "properties": {
                    "x": {"type": "integer", "minimum": 0, "maximum": 15},
                    "y": {"type": "integer", "minimum": 0, "maximum": 11},
                }, "required": ["x", "y"], "additionalProperties": False,
            }}]

        def observe(self):
            return {"frame": self.frame.copy(), "legal_actions": self.legal_actions,
                    "game_state": "NOT_FINISHED", "level": 0, "levels_completed": 0,
                    "win_levels": 1, "terminal": False}

        def act(self, action, args):
            self.frame[args["y"], args["x"]] = 1
            self.frame[0, 0] = int(args["x"] + args["y"])
            return {}

    tools = GameTools(ClickEnv(), tmp_path / "click-context", max_actions=4)
    tools.observe()
    first = tools.act("ACTION6", {"x": 12, "y": 8})
    second = tools.act("ACTION6", {"x": 2, "y": 3})
    current = second["observation"]
    records = tools.history.records()

    events = _action_observation_events(records)
    click_events = [row for row in events if row.get("action") == "click"]
    assert [row["observed_result"].split(";", 1)[0] for row in click_events] == [
        "1 click(x=12,y=8): board effect unknown", "2 click(x=2,y=3): board effect unknown",
    ]
    assert [row["measurements"]["click_effect"]["local_changed_pixels"] for row in click_events] == [1, 1]
    assert [row["measurements"]["click_effect"]["remote_changed_pixels"] for row in click_events] == [1, 1]

    continued, _, _ = _action_conversation_observation(
        tools, current, records, horizon=2,
        previous_plan={
            "actions": [
                {"action": "ACTION6", "args": {"x": 12, "y": 8}},
                {"action": "ACTION6", "args": {"x": 2, "y": 3}},
            ],
            "action_predictions": [
                {"board_change": "yes", "environment_outcome": "no_terminal", "description": "first predicted"},
                {"board_change": "no", "environment_outcome": "no_terminal", "description": "second predicted"},
            ],
        },
    )
    continued_payload, _ = json.JSONDecoder().raw_decode(
        continued.split("CURRENT SCREEN is the actual latest environment state.\n", 1)[1]
    )
    continued_events = continued_payload["new_observations_since_previous_action_request"]
    assert [row["click"] for row in continued_events] == ["click(x=12,y=8)", "click(x=2,y=3)"]
    assert [row["observed"]["board_measurement_status"] for row in continued_events] == [
        "mask_unavailable", "mask_unavailable",
    ]
    assert [row["prediction"]["description"] for row in continued_events] == [
        "first predicted", "second predicted",
    ]
    bootstrap, _, _ = _action_conversation_observation(
        tools, current, records, horizon=1, bootstrap_transitions=[records[1], records[2]],
    )
    assert "click(x=12,y=8)" in bootstrap and "click(x=2,y=3)" in bootstrap
    assert "local_changed_pixels" in bootstrap and "remote_changed_pixels" in bootstrap

    transition_rows = [row for row in records if row.get("type") == "transition"]
    evidence_by_id = {row.get("evidence_id"): row for row in records if row.get("evidence_id") is not None}
    ledger = [
        _click_outcome_ledger_entry(row, evidence_by_id.get(row.get("pre_evidence_id"), {}))
        for row in transition_rows
    ]
    reinitialized = _live_context(
        tools, current, "click-fixture", 1, shared_skill_path(),
        game_memory_dir(tmp_path / "state", "click-fixture"), "reinit", 0,
        click_screen_transitions=transition_rows, click_history_ledger=ledger,
    )
    compacted_rows = reinitialized["context"]["recent_transitions"]
    assert [row["args"] for row in compacted_rows] == [{"x": 12, "y": 8}, {"x": 2, "y": 3}]
    assert [row["measurements"]["click_effect"]["total_changed_pixels"] for row in compacted_rows] == [2, 2]
    visible_context = json.loads(reinitialized["user_prompt"].split("Visible context:\n", 1)[1])
    assert "click(x=12,y=8)" == visible_context["click_history_ledger"][0]["click"]
    assert visible_context["click_history_ledger"][0]["observed"]["board_measurement_status"] == "mask_unavailable"

    review_events = [row for row in _review_events(records) if row.get("type") == "transition"]
    assert [row["args"] for row in review_events] == [{"x": 12, "y": 8}, {"x": 2, "y": 3}]
    assert review_events[0]["measurements"]["click_effect"]["remote_changed_pixels"] == 1
    # Keep a reference to the first transition's observation for the assertions above.
    assert first["evidence"]["args"] == {"x": 12, "y": 8}


def test_click_only_contract_has_required_coordinates_and_no_assumed_player_or_obstacles() -> None:
    click_spec = {"id": "ACTION6", "parameters": {
        "type": "object", "properties": {
            "x": {"type": "integer", "minimum": 0, "maximum": 15},
            "y": {"type": "integer", "minimum": 0, "maximum": 11},
        }, "required": ["x", "y"], "additionalProperties": False,
    }}
    contract = reasoned_action_contract_description([click_spec])
    assert contract["actions"] == [{"action": "click", "args": {"x": 0, "y": 0}}]
    assert contract["schema"] == "v11.click_action/3"
    assert "state_assessment" not in contract
    valid = {
        **contract,
        "hypothesis": None,
        "trajectory_rationale": "Test this legal coordinate.",
        "predictions": [{"board_change": "unknown", "environment_outcome": "unknown", "description": "unknown"}],
    }
    from v11.analysis import validate_reasoned_action
    assert validate_reasoned_action(valid, [click_spec])["actions"] == [
        {"action": "ACTION6", "args": {"x": 0, "y": 0}},
    ]
    invalid = {**valid, "hypothesis": {"claim": "A claim", "evidence": "x" * 181}}
    with pytest.raises(DecisionValidationError, match="hypothesis.evidence exceeds the 180-character limit"):
        validate_reasoned_action(invalid, [click_spec])


def test_click_batch_requires_and_preserves_one_prediction_per_action() -> None:
    click_spec = {"id": "ACTION6", "parameters": {
        "type": "object", "properties": {
            "x": {"type": "integer", "minimum": 0, "maximum": 15},
            "y": {"type": "integer", "minimum": 0, "maximum": 11},
        }, "required": ["x", "y"], "additionalProperties": False,
    }}
    from v11.analysis import validate_reasoned_action

    payload = {
        "schema": "v11.click_action/3",
        "actions": [
            {"action": "click", "args": {"x": 2, "y": 3}},
            {"action": "click", "args": {"x": 7, "y": 4}},
        ],
        "hypothesis": {"claim": "Each region toggles independently", "evidence": "Two marked shapes are visible"},
        "trajectory_rationale": "Test both independent regions in sequence.",
        "predictions": [
            {"board_change": "yes", "environment_outcome": "no_terminal", "description": "first region toggles"},
            {"board_change": "no", "environment_outcome": "no_terminal", "description": "second region remains unchanged"},
        ],
    }
    normalized = validate_reasoned_action(payload, [click_spec], max_actions=3)
    assert [item["board_change"] for item in normalized["action_predictions"]] == ["yes", "no"]
    payload["predictions"].pop()
    with pytest.raises(DecisionValidationError, match="one prediction per action"):
        validate_reasoned_action(payload, [click_spec], max_actions=3)


@pytest.mark.parametrize(
    ("actions", "click_example"),
    [
        (["ACTION6"], True),
        (["ACTION6", "RESET"], True),
        (["ACTION1", "ACTION2", "ACTION3", "ACTION4", "RESET"], False),
        (["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION6", "RESET"], False),
    ],
)
def test_click_only_contract_detection_ignores_reset(actions, click_example) -> None:
    specs = []
    for action in actions:
        if action == "ACTION6":
            params = {"type": "object", "properties": {
                "x": {"type": "integer", "minimum": 0, "maximum": 15},
                "y": {"type": "integer", "minimum": 0, "maximum": 11},
            }, "required": ["x", "y"], "additionalProperties": False}
        elif action == "RESET":
            params = {"type": "object", "properties": {
                "justification": {"type": "string"},
            }, "required": ["justification"], "additionalProperties": False}
        else:
            params = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
        specs.append({"id": action, "parameters": params})
    contract = reasoned_action_contract_description(specs)
    if click_example:
        assert contract["schema"] == "v11.click_action/3"
        assert "state_assessment" not in contract
    else:
        assert contract["state_assessment"]["player"]["location"] == "visual region or unknown"
        assert contract["state_assessment"]["target_candidates"]
        assert contract["state_assessment"]["obstacles"]
    assert contract["actions"][0]["action"] != "RESET"
    if click_example:
        assert contract["actions"] == [{"action": "click", "args": {"x": 0, "y": 0}}]


def test_live_click_profile_with_reset_uses_click_contract_in_full_prompt(tmp_path) -> None:
    class ClickResetEnv:
        legal_actions = [{"id": "ACTION6", "parameters": {
            "type": "object", "properties": {
                "x": {"type": "integer", "minimum": 0, "maximum": 15},
                "y": {"type": "integer", "minimum": 0, "maximum": 11},
            }, "required": ["x", "y"], "additionalProperties": False,
        }}]

        def observe(self):
            return {"frame": np.zeros((12, 16), dtype=np.uint8), "legal_actions": self.legal_actions,
                    "game_state": "NOT_FINISHED", "level": 0, "levels_completed": 0,
                    "win_levels": 1, "terminal": False}

        def act(self, action, args):
            return {}

        def reset(self):
            return self.observe()

    tools = GameTools(ClickResetEnv(), tmp_path / "live-click-reset")
    current = tools.observe()
    inputs = _live_context(
        tools, current, "click-reset", 2, shared_skill_path(),
        game_memory_dir(tmp_path / "state", "click-reset"), "run", 5,
    )
    prompt = inputs["user_prompt"]
    contract_text = prompt.split("Required response contract (`v11.click_action/3`):\n", 1)[1].split("\nString limits:", 1)[0]
    contract = json.loads(contract_text)
    assert {item["id"] for item in inputs["context"]["current_state"]["legal_actions"]} == {"ACTION6", "RESET"}
    assert contract["actions"] == [{"action": "click", "args": {"x": 0, "y": 0}}]
    assert contract["schema"] == "v11.click_action/3"
    assert "hypothesis.evidence 180" in prompt
    assert "state_assessment" not in contract
    profile_prompt = inputs["system_prompt"]
    for instruction in (
        "Observe → Model → Compare → Click → Verify",
        "Each candidate runs independently in a disposable fork",
        "one prediction per live click",
        "`click_candidates` are optional suggestions",
        "Empty and background-colored cells can be valid targets",
    ):
        assert instruction in profile_prompt
