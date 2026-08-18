import os
import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from oss_v1.adaptor import ArcEngineAdaptor
from oss_v1.agent import OSSAgent, grid_to_hex

# The game used for tests – small and deterministic.
GAME_ID = "tt02"
MAX_STEPS = 3


def test_adaptor_basic():
    adaptor = ArcEngineAdaptor(GAME_ID, seed=0)
    grid = adaptor.reset()
    assert isinstance(grid, (list, tuple)) or hasattr(grid, 'shape'), "reset should return a numpy array"
    # Ensure shape is at least 2D
    if hasattr(grid, 'shape'):
        assert len(grid.shape) == 2
    else:
        grid = list(grid)
        assert isinstance(grid[0], (list, tuple))
    actions = adaptor.available_actions()
    assert isinstance(actions, list) and len(actions) > 0
    # Step through the first action to confirm no exception
    step_grid = adaptor.step(actions[0])
    assert hasattr(step_grid, 'shape')

@patch("oss_v1.agent.requests.post")
@patch("time.sleep", lambda x: None)
def test_agent_run_with_mock_llm(mock_post):
    # Mock LLM to always return action 1
    mock_resp = {
        "choices": [
            {"message": {"content": "1"}}
        ]
    }
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = mock_resp

    agent = OSSAgent(GAME_ID, seed=0, max_steps=MAX_STEPS)
    actions = agent.run_episode()
    # Should have executed MAX_STEPS actions
    assert len(actions) == MAX_STEPS
    for act_id, x, y in actions:
        assert act_id == 1
    # Log file should exist and contain the correct number of rows (excluding header)
    log_path = Path("runs/oss_v1") / f"{GAME_ID}_log.csv"
    assert log_path.exists()
    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # number of data rows should equal MAX_STEPS
    assert len(rows) == MAX_STEPS

@patch("oss_v1.agent.requests.post")
@patch("time.sleep", lambda x: None)
@patch("random.choice", lambda seq: seq[0])  # deterministic fallback to first available action
def test_agent_retry_and_fallback(mock_post):
    # First two calls return bad output, third call (not used) returns a valid action.
    responses = ["BAD", "INVALID", "3"]
    def side_effect(url, json, timeout):
        idx = len(responses) - 1
        resp_text = responses.pop()
        mock_resp = {
            "choices": [
                {"message": {"content": resp_text}}
            ]
        }
        m = type("_", (), {})()
        m.status_code = 200
        m.json = lambda: mock_resp
        return m
    mock_post.side_effect = side_effect

    agent = OSSAgent(GAME_ID, seed=0, max_steps=1)
    actions = agent.run_episode()
    # Because we have two invalid attempts, fallback to random action from available_actions.
    assert len(actions) == 1
    act_id, x, y = actions[0]
    # Our mock_choice returned first available action which should be the smallest valid id.
    assert act_id in agent.adaptor.available_actions()
    # The log file should record failure (FAILED) as llm_output
    log_path = Path("runs/oss_v1") / f"{GAME_ID}_log.csv"
    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)
    assert row["llm_output"] == "FAILED"

# End of tests
