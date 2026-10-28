from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from v11 import cli as cli_module
from v11.cli import build_parser


def _live_args(*args: str):
    return build_parser().parse_args(["live", *args])


def test_live_parser_accepts_game_list_and_defaults_to_one_worker() -> None:
    args = _live_args("--game-list", "ez01, ez02", "--run-dir", "runs/batch")

    assert args.game_id is None
    assert args.game_list == ["ez01", "ez02"]
    assert args.parallel_games == 1
    assert args.profile_memory_max_chars == 4500


def test_profile_memory_cli_flag_and_legacy_alias_are_equivalent() -> None:
    parser = cli_module.build_parser()
    current = parser.parse_args([
        "live", "--game-id", "ez01", "--run-dir", "run", "--profile-memory-max-chars", "4000",
    ])
    legacy = parser.parse_args([
        "live", "--game-id", "ez01", "--run-dir", "run", "--global-memory-max-chars", "4000",
    ])
    assert current.profile_memory_max_chars == legacy.profile_memory_max_chars == 4000


def test_live_parser_rejects_both_game_selectors() -> None:
    with pytest.raises(SystemExit):
        _live_args(
            "--game-id", "ez01", "--game-list", "ez02", "--run-dir", "runs/batch",
        )


@pytest.mark.parametrize("game_list", ["", "ez01,ez01", "../ez01", "ez01,../ez02"])
def test_live_parser_rejects_invalid_game_lists(game_list: str) -> None:
    with pytest.raises(SystemExit):
        _live_args("--game-list", game_list, "--run-dir", "runs/batch")


@pytest.mark.parametrize("parallel_games", ["0", "-1"])
def test_live_parser_requires_positive_parallel_games(parallel_games: str) -> None:
    with pytest.raises(SystemExit):
        _live_args(
            "--game-list", "ez01,ez02", "--parallel-games", parallel_games,
            "--run-dir", "runs/batch",
        )


def test_batch_continues_after_failure_and_writes_per_game_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    active = 0
    max_active = 0
    active_lock = threading.Lock()
    initial_workers = threading.Barrier(2)
    started = 0
    clients = []
    envs = []
    reset_calls = []
    run_arguments = []

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            clients.append(self)

    class FakeEnv:
        def __init__(self, game_id, *, seed, env_root):
            self.game_id = game_id
            self.seed = seed
            self.closed = False
            envs.append(self)

        def close(self):
            self.closed = True

    def fake_run_live(env, run_dir, client, **kwargs):
        nonlocal active, max_active, started
        with active_lock:
            active += 1
            max_active = max(max_active, active)
            started += 1
            wait_for_pair = started <= 2
        if wait_for_pair:
            initial_workers.wait(timeout=2)
        time.sleep(0.02)
        run_arguments.append(dict(kwargs))
        with active_lock:
            active -= 1
        if env.game_id == "ez02":
            raise RuntimeError("fixture failed")
        return {"game_id": env.game_id, "won": True, "run_dir": str(run_dir)}

    monkeypatch.setattr(cli_module, "StrataClient", FakeClient)
    monkeypatch.setattr(cli_module, "run_live", fake_run_live)
    monkeypatch.setattr(cli_module, "reset_persistent_memory", lambda state_dir: reset_calls.append(state_dir))
    monkeypatch.setattr("v11.arcade_env.ArcadeGameEnv", FakeEnv)

    result_code = cli_module.main([
        "live", "--game-list", "ez01,ez02,ez03", "--run-dir", str(tmp_path / "batch"),
        "--state-dir", str(tmp_path / "state"), "--parallel-games", "2", "--seed", "17",
        "--reset-memory", "--vllm", "--base-url", "http://vllm", "--model", "test-model",
    ])

    assert result_code == 1
    assert max_active == 2
    assert len(clients) == len(envs) == 3
    assert len({id(client) for client in clients}) == 3
    assert all(client.kwargs["backend"] == "vllm" for client in clients)
    assert all(client.kwargs["model"] == "test-model" for client in clients)
    assert all(client.kwargs["log_dir"].parent == tmp_path / "batch" for client in clients)
    assert all(env.seed == 17 and env.closed for env in envs)
    assert len(reset_calls) == 1
    assert all("global_memory_lock" not in kwargs for kwargs in run_arguments)

    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "partial_failure"
    assert output["parallel_games"] == 2
    assert [row["game_id"] for row in output["games"]] == ["ez01", "ez02", "ez03"]
    assert [row["status"] for row in output["games"]] == ["completed", "failed", "completed"]
    assert output["games"][1]["error"] == "fixture failed"


def test_batch_caps_worker_count_to_number_of_games(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeClient:
        def __init__(self, **kwargs):
            pass

    class FakeEnv:
        def __init__(self, game_id, **kwargs):
            self.game_id = game_id

        def close(self):
            pass

    monkeypatch.setattr(cli_module, "StrataClient", FakeClient)
    monkeypatch.setattr(cli_module, "run_live", lambda env, run_dir, client, **kwargs: {"won": False})
    monkeypatch.setattr("v11.arcade_env.ArcadeGameEnv", FakeEnv)

    result_code = cli_module.main([
        "live", "--game-list", "ez01,ez02", "--run-dir", str(tmp_path / "batch"),
        "--parallel-games", "20",
    ])

    output = json.loads(capsys.readouterr().out)
    assert result_code == 0
    assert output["parallel_games"] == 2
    assert len(output["games"]) == 2
