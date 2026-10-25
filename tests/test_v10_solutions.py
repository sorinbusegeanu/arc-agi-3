from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from v10.live import run_live
from v10.tools.solutions import LevelSolutionStore, state_fingerprint


def _observation(frame: int, *, level: int = 0, completed: int = 0) -> dict:
    return {
        "frame": np.full((2, 2), frame, dtype=np.int64),
        "legal_actions": ["left", "right"], "game_state": "NOT_FINISHED",
        "level": level, "levels_completed": completed, "terminal": False,
    }


def test_solution_store_finds_shortest_unambiguous_success_path(tmp_path: Path) -> None:
    store = LevelSolutionStore(tmp_path / "solutions.jsonl", "fixture")
    start, middle, end = _observation(0), _observation(1), _observation(2)
    goal = {**_observation(3, level=1, completed=1), "game_state": "WIN", "terminal": True}

    store.record_transition(start, "right", {}, middle, run_id="r1")
    store.record_transition(middle, "left", {}, end, run_id="r1")
    store.record_transition(end, "right", {}, goal, run_id="r1")
    store.record_successful_trajectory(start, store.edges(), run_id="r1", success="win")

    path = store.shortest_known_solution(start)
    assert [edge["action"] for edge in path] == ["right", "left", "right"]
    assert len(store.solutions()) == 1
    assert state_fingerprint(start) == store.edges()[0]["source_state"]


def test_solution_store_ignores_ambiguous_action_edges_and_deduplicates(tmp_path: Path) -> None:
    store = LevelSolutionStore(tmp_path / "solutions.jsonl", "fixture")
    start, first_target, other_target = _observation(0), _observation(1), _observation(2)
    goal = {**_observation(3, level=1, completed=1), "game_state": "WIN", "terminal": True}
    edge = store.record_transition(start, "left", {}, first_target, run_id="r1")
    duplicate = store.record_transition(start, "left", {}, first_target, run_id="r2")
    store.record_transition(start, "left", {}, other_target, run_id="r2")
    store.record_transition(first_target, "right", {}, goal, run_id="r1")

    assert edge["id"] == duplicate["id"]
    assert len(store.edges()) == 3
    assert store.shortest_known_solution(start) is None


class _CompletingEnv:
    def __init__(self):
        self.frame = np.zeros((2, 2), dtype=np.int64)
        self.actions = []

    def act(self, action, args):
        self.actions.append(action)
        self.frame[0, 0] += 1
        return {}

    def reset(self):
        self.frame[:] = 0
        return self.observe()

    def observe(self):
        observation = {
            "frame": self.frame.copy(),
            "legal_actions": [{
                "id": "up", "parameters": {
                    "type": "object", "properties": {}, "required": [],
                    "additionalProperties": False,
                },
            }],
            "game_state": "NOT_FINISHED", "level": 0, "levels_completed": 0,
            "win_levels": 1, "terminal": False,
        }
        if self.actions:
            observation["level"] = 1
            observation["levels_completed"] = 1
        return observation


class _Client:
    model = "test"

    def __init__(self):
        self.calls = []

    def complete(self, system_prompt, user_prompt, image_paths, image_labels, response_format="json_object"):
        self.calls.append(system_prompt)
        if system_prompt.startswith("# V10 incremental memory review"):
            content = '{"changes": [], "global_memory_markdown": "# Global Memory\\n"}'
        else:
            content = '{"actions":[{"action":"up","args":{}}]}'
        return {
            "content": content, "model": self.model,
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "latency_seconds": 0.01, "request_attempts": 1, "finish_reason": "stop",
        }


def test_new_run_replays_successful_level_path_without_action_model_call(tmp_path: Path, monkeypatch) -> None:
    state_dir = tmp_path / "state"
    monkeypatch.setenv("V10_STATE_DIR", str(state_dir))
    first_env = _CompletingEnv()
    first_client = _Client()
    first = run_live(first_env, tmp_path / "first", first_client, game_id="fixture", max_actions=1)
    assert first["actions"] == 1
    assert first["available_level_solutions"] == 1

    second_env = _CompletingEnv()
    second_client = _Client()
    second = run_live(second_env, tmp_path / "second", second_client, game_id="fixture", max_actions=1)

    rows = [json.loads(line) for line in (tmp_path / "second/decisions.jsonl").read_text().splitlines()]
    plan = next(row for row in rows if row.get("schema") == "v10.live_decision/2")
    assert second_env.actions == ["up"]
    assert second["cached_solution_actions"] == 1
    assert second["model_calls"] == 1  # review call only; no action-decision call
    assert plan["source"] == "verified_level_path"
    assert len(second_client.calls) == 1  # completion review only
