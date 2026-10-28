from __future__ import annotations

import numpy as np

from v11.analysis import model_action_aliases, validate_model_decision
from v11.tools.game import GameTools
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
