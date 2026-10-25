from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from v10.live import run_live
from v10.tools.memory import save_game_memory
from v10.tools.solutions import LevelSolutionStore, state_fingerprint


def _run_live_without_starting_routine(*args, **kwargs):
    """Keep route tests focused on route reuse, independently of startup selection."""
    kwargs["starting_routine"] = False
    return run_live(*args, **kwargs)


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


def test_exact_start_replays_shortest_successful_trajectory_only(tmp_path: Path) -> None:
    store = LevelSolutionStore(tmp_path / "solutions.jsonl", "fixture")
    start = {**_observation(0), "legal_actions": ["left", "right", "up"]}
    first = _observation(1)
    second = _observation(2)
    third = _observation(3)
    goal = {**_observation(9, level=1, completed=1), "game_state": "WIN", "terminal": True}

    long_edges = [
        store.record_transition(start, "left", {}, first, run_id="long"),
        store.record_transition(first, "right", {}, second, run_id="long"),
        store.record_transition(second, "left", {}, goal, run_id="long"),
    ]
    store.record_successful_trajectory(start, long_edges, run_id="long", success="win")
    short_edges = [
        store.record_transition(start, "right", {}, third, run_id="short"),
        store.record_transition(third, "left", {}, goal, run_id="short"),
    ]
    store.record_successful_trajectory(start, short_edges, run_id="short", success="win")
    # An observed completion edge is not itself a saved successful trajectory.
    store.record_transition(start, "up", {}, goal, run_id="edge-only")
    overlong_path = [
        store.record_transition(start, "up", {}, _observation(4), run_id="longer-again"),
        store.record_transition(_observation(4), "right", {}, _observation(5), run_id="longer-again"),
        store.record_transition(_observation(5), "left", {}, _observation(6), run_id="longer-again"),
        store.record_transition(_observation(6), "right", {}, goal, run_id="longer-again"),
    ]
    retained = store.record_successful_trajectory(
        start, overlong_path, run_id="longer-again", success="win",
    )

    selected = store.shortest_known_solution(start)
    assert selected is not None
    assert [edge["action"] for edge in selected] == ["right", "left"]
    assert retained is not None and len(retained["steps"]) == 2
    assert len(store.solutions()) == 1
    assert len(store.solutions()[0]["steps"]) == 2
    assert sum(
        json.loads(line).get("kind") == "solution"
        for line in store.path.read_text().splitlines()
    ) == 2


def test_successful_run_logs_are_discovered_and_imported_idempotently(tmp_path: Path) -> None:
    from shutil import copytree
    from v10.live import _import_prior_winning_runs

    source = Path(__file__).resolve().parents[1] / "runs/v10/ez01_20"
    runs_root = tmp_path / "runs"
    copytree(source, runs_root / "ez01_legacy")
    store = LevelSolutionStore(tmp_path / "skills/games/ez01/level_solutions.jsonl", "ez01")

    first_imports = _import_prior_winning_runs(
        store, "ez01", tmp_path / "new-run", runs_root=runs_root,
    )
    second_imports = _import_prior_winning_runs(
        store, "ez01", tmp_path / "new-run", runs_root=runs_root,
    )

    assert first_imports == 5
    assert second_imports == 0
    assert len(store.solutions()) == 5


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


def test_import_historical_ez01_win_is_complete_and_idempotent(tmp_path: Path) -> None:
    recorded_run = Path(__file__).resolve().parents[1] / "runs/v10/ez01_20"
    store = LevelSolutionStore(tmp_path / "solutions.jsonl", "ez01")

    imported = store.import_successful_run(recorded_run)
    assert len(imported) == 5
    assert len(store.edges()) == 25
    assert len(store.solutions()) == 5
    first = store.shortest_known_solution(_recorded_start(recorded_run))
    assert first is not None
    assert [edge["action"] for edge in first] == ["ACTION1", "ACTION1"]

    store.import_successful_run(recorded_run)
    assert len(store.edges()) == 25
    assert len(store.solutions()) == 5


def _recorded_start(run_dir: Path) -> dict:
    from v10.frames import FrameStore
    from v10.runlog import read_jsonl

    state = next(row for row in read_jsonl(run_dir / "evidence.jsonl") if row.get("type") == "state")
    return {
        "frame": FrameStore(run_dir).get(state["frame_id"]),
        "game_id": "ez01", "frame_id": state["frame_id"],
        "legal_actions": state["legal_actions"], "game_state": state["game_state"],
        "level": state["level"], "levels_completed": state["levels_completed"],
        "terminal": state["terminal"],
    }


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
                "id": "ACTION1", "parameters": {
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
        if system_prompt.startswith("# V10 memory consolidation"):
            content = json.dumps({
                "game_memory_markdown": "# Game Memory\n\n- The verified route completes level zero.",
                "global_memory_markdown": "# Global Memory\n\n- Compare objects before and after actions.",
            })
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
    first = _run_live_without_starting_routine(first_env, tmp_path / "first", first_client, game_id="fixture", max_actions=1)
    assert first["actions"] == 1
    assert first["available_level_solutions"] == 1

    second_env = _CompletingEnv()
    second_client = _Client()
    second = _run_live_without_starting_routine(second_env, tmp_path / "second", second_client, game_id="fixture", max_actions=1)

    rows = [json.loads(line) for line in (tmp_path / "second/decisions.jsonl").read_text().splitlines()]
    plan = next(row for row in rows if row.get("schema") == "v10.live_decision/2")
    assert second_env.actions == ["ACTION1"]
    assert second["cached_solution_actions"] == 1
    assert second["model_calls"] == 1  # review call only; no action-decision call
    assert plan["source"] == "verified_level_path"
    assert len(second_client.calls) == 1  # completion review only
    assert plan["decision"]["actions"] == [{"action": "up", "args": {}}]
    assert "verified route" in (state_dir / "games/fixture/MEMORY.md").read_text()


def test_startup_skips_bootstrap_when_exact_route_exists_for_current_level(tmp_path: Path, monkeypatch) -> None:
    state_dir = tmp_path / "state"
    monkeypatch.setenv("V10_STATE_DIR", str(state_dir))
    first = _run_live_without_starting_routine(
        _CompletingEnv(), tmp_path / "learn", _Client(),
        game_id="fixture", max_actions=1, state_dir=state_dir,
    )
    assert first["available_level_solutions"] == 1

    env = _CompletingEnv()
    client = _Client()
    metrics = run_live(
        env, tmp_path / "replay", client, game_id="fixture",
        max_actions=1, state_dir=state_dir,
    )

    rows = [json.loads(line) for line in (tmp_path / "replay/decisions.jsonl").read_text().splitlines()]
    plan = next(row for row in rows if row.get("schema") == "v10.live_decision/2")
    assert plan["source"] == "verified_level_path"
    assert env.actions == ["ACTION1"]
    assert metrics["cached_solution_actions"] == 1
    assert metrics["resets"] == 0
    assert len(client.calls) == 1  # completion review only, no model action plan
    assert client.calls[0].startswith("# V10 memory consolidation")
    assert not any(row.get("schema") == "v10.live_bootstrap_reset/1" for row in rows)


def test_level_route_with_different_start_skips_bootstrap_and_guides_model(tmp_path: Path, monkeypatch) -> None:
    from v10.tools.memory import game_memory_dir

    state_dir = tmp_path / "state"
    monkeypatch.setenv("V10_STATE_DIR", str(state_dir))
    saved_start = {
        **_observation(0, level=2, completed=2), "game_id": "fixture",
        "legal_actions": ["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
    }
    saved_end = {
        **_observation(1, level=3, completed=3), "game_id": "fixture",
        "legal_actions": ["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
    }
    store = LevelSolutionStore(game_memory_dir(state_dir, "fixture") / "level_solutions.jsonl", "fixture")
    edge = store.record_transition(saved_start, "ACTION1", {}, saved_end, run_id="seed")
    store.record_successful_trajectory(saved_start, [edge], run_id="seed", success="level_completed")

    class DifferentStartEnv:
        def __init__(self):
            self.frame = np.full((2, 2), 77, dtype=np.int64)
            self.actions = []
            self.resets = 0

        def observe(self):
            return {
                "frame": self.frame.copy(),
                "legal_actions": [{"id": f"ACTION{i}", "parameters": {
                    "type": "object", "properties": {}, "required": [], "additionalProperties": False,
                }} for i in range(1, 5)],
                "game_state": "NOT_FINISHED", "level": 2, "levels_completed": 2,
                "win_levels": 5, "terminal": False,
            }

        def act(self, action, args):
            self.actions.append(action)
            self.frame[0, 0] += 1
            return {}

        def reset(self):
            self.resets += 1
            self.frame[:] = 0
            return self.observe()

    class DownClient(_Client):
        def __init__(self):
            super().__init__()
            self.user_prompts = []

        def complete(self, system_prompt, user_prompt, image_paths, image_labels, response_format="json_object"):
            self.user_prompts.append(user_prompt)
            if system_prompt.startswith("# V10 memory consolidation"):
                return super().complete(system_prompt, user_prompt, image_paths, image_labels, response_format)
            return {
                "content": '{"actions":[{"action":"down","args":{}}]}',
                "model": self.model, "usage": {}, "latency_seconds": 0,
                "request_attempts": 1, "finish_reason": "stop",
            }

    env = DifferentStartEnv()
    client = DownClient()
    metrics = run_live(
        env, tmp_path / "different-start", client, game_id="fixture",
        max_actions=1, state_dir=state_dir,
    )

    rows = [json.loads(line) for line in (tmp_path / "different-start/decisions.jsonl").read_text().splitlines()]
    plan = next(row for row in rows if row.get("schema") == "v10.live_decision/2")
    prompt_context = json.loads(client.user_prompts[0].split("Visible context:\n", 1)[1])
    guidance = prompt_context["saved_route_guidance"][0]
    assert env.resets == 0
    assert env.actions == ["ACTION2"]
    assert plan["source"] == "model"
    assert metrics["cached_solution_actions"] == 0
    assert guidance["actions"] == ["up"]
    assert guidance["applies_to_current_starting_screen"] is False
    assert "starting screen differs" in guidance["guidance"]
    assert not any(row.get("source") == "bootstrap_probe" for row in rows)


def test_cached_route_stops_on_divergence_then_asks_model(tmp_path: Path, monkeypatch) -> None:
    from v10.tools.memory import game_memory_dir

    state_dir = tmp_path / "state"
    monkeypatch.setenv("V10_STATE_DIR", str(state_dir))
    start = _CompletingEnv().observe()
    wrong_target = {
        **start, "frame": np.full((2, 2), 99, dtype=np.int64),
        "level": 1, "levels_completed": 1,
    }
    store = LevelSolutionStore(game_memory_dir(state_dir, "fixture") / "level_solutions.jsonl", "fixture")
    edge = store.record_transition(start, "ACTION1", {}, wrong_target, run_id="seed")
    store.record_successful_trajectory(start, [edge], run_id="seed", success="level_completed")
    metrics = _run_live_without_starting_routine(
        _CompletingEnv(), tmp_path / "diverging", _Client(),
        game_id="fixture", max_actions=2, state_dir=state_dir,
    )

    rows = [json.loads(line) for line in (tmp_path / "diverging/decisions.jsonl").read_text().splitlines()]
    plans = [row for row in rows if row.get("schema") == "v10.live_decision/2"]
    assert [row["source"] for row in plans] == ["verified_level_path", "model"]
    assert metrics["cached_solution_actions"] == 1
    assert metrics["actions"] == 2


def test_illegal_cached_action_is_rejected_and_model_is_queried(tmp_path: Path, monkeypatch) -> None:
    from v10.tools.memory import game_memory_dir

    state_dir = tmp_path / "state"
    monkeypatch.setenv("V10_STATE_DIR", str(state_dir))
    start = _CompletingEnv().observe()
    illegal_target = {**start, "frame": np.ones((2, 2), dtype=np.int64), "level": 1, "levels_completed": 1}
    store = LevelSolutionStore(game_memory_dir(state_dir, "fixture") / "level_solutions.jsonl", "fixture")
    edge = store.record_transition(start, "ACTION2", {}, illegal_target, run_id="seed")
    store.record_successful_trajectory(start, [edge], run_id="seed", success="level_completed")

    _run_live_without_starting_routine(
        _CompletingEnv(), tmp_path / "illegal", _Client(),
        game_id="fixture", max_actions=1, state_dir=state_dir,
    )
    rows = [json.loads(line) for line in (tmp_path / "illegal/decisions.jsonl").read_text().splitlines()]
    plan = next(row for row in rows if row.get("schema") == "v10.live_decision/2")
    assert plan["source"] == "model"
    assert "cached path is no longer valid" in plan["cached_path_error"]


def test_bootstrap_model_startup_uses_memory_before_normal_route_reuse(tmp_path: Path, monkeypatch) -> None:
    from v10.tools.memory import GameMemory, game_memory_dir

    class TwoStepWinEnv:
        def __init__(self):
            self.frame = np.zeros((2, 2), dtype=np.int64)
            self.actions = []

        def observe(self):
            completed = 1 if self.actions else 0
            won = len(self.actions) >= 2
            return {
                "frame": self.frame.copy(),
                "legal_actions": [{"id": "ACTION1", "parameters": {
                    "type": "object", "properties": {}, "required": [], "additionalProperties": False,
                }}],
                "game_state": "WIN" if won else "NOT_FINISHED", "level": completed,
                "levels_completed": completed, "win_levels": 1,
                "terminal": won,
            }

        def act(self, action, args):
            self.actions.append(action)
            self.frame[0, 0] += 1
            return {}

        def reset(self):
            self.frame[:] = 0
            self.actions.clear()
            return self.observe()

    state_dir = tmp_path / "state"
    monkeypatch.setenv("V10_STATE_DIR", str(state_dir))
    save_game_memory(
        game_memory_dir(state_dir, "fixture"),
        "# Game Memory\n\n- This game has a previously observed mechanic.",
    )
    route_env = TwoStepWinEnv()
    route_env.act("ACTION1", {})
    route_start = route_env.observe()
    route_env.act("ACTION1", {})
    win = route_env.observe()
    store = LevelSolutionStore(game_memory_dir(state_dir, "fixture") / "level_solutions.jsonl", "fixture")
    edge = store.record_transition(route_start, "ACTION1", {}, win, run_id="seed")
    store.record_successful_trajectory(route_start, [edge], run_id="seed", success="win")

    env = TwoStepWinEnv()
    metrics = run_live(
        env, tmp_path / "run", _Client(), game_id="fixture", max_actions=2,
        state_dir=state_dir,
    )
    rows = [json.loads(line) for line in (tmp_path / "run/decisions.jsonl").read_text().splitlines()]
    plans = [row for row in rows if row.get("schema") == "v10.live_decision/2"]

    assert [row["source"] for row in plans] == ["bootstrap_probe", "bootstrap_model_start"]
    assert metrics["cached_solution_actions"] == 0
    assert metrics["actions"] == 2
    assert metrics["resets"] == 1


def test_successful_route_is_saved_and_replayed_even_when_review_fails(tmp_path: Path, monkeypatch) -> None:
    from v10.tools.memory import game_memory_dir

    state_dir = tmp_path / "state"
    monkeypatch.setenv("V10_STATE_DIR", str(state_dir))

    class InvalidReviewClient(_Client):
        def complete(self, system_prompt, user_prompt, image_paths, image_labels, response_format="json_object"):
            if system_prompt.startswith("# V10 memory consolidation"):
                self.calls.append(system_prompt)
                return {"content": "not JSON", "model": self.model, "usage": {},
                        "latency_seconds": 0.0, "request_attempts": 1, "finish_reason": "stop"}
            return super().complete(system_prompt, user_prompt, image_paths, image_labels, response_format)

    failed = _run_live_without_starting_routine(
        _CompletingEnv(), tmp_path / "failed-review", InvalidReviewClient(),
        game_id="fixture", max_actions=1, state_dir=state_dir,
    )
    store = LevelSolutionStore(game_memory_dir(state_dir, "fixture") / "level_solutions.jsonl", "fixture")
    assert failed["available_level_solutions"] == 1
    assert store.solutions()[0]["success"] == "level_completed"
    assert failed["stop_reason"] == "action_budget_exhausted"

    env = _CompletingEnv()
    replayed = _run_live_without_starting_routine(
        env, tmp_path / "replayed", _Client(), game_id="fixture", max_actions=1,
        state_dir=state_dir,
    )
    assert env.actions == ["ACTION1"]
    assert replayed["cached_solution_actions"] == 1
