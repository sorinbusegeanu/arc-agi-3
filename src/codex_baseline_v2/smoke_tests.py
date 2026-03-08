from __future__ import annotations

import json
import tempfile
from typing import List

from codex_baseline_v2.adapters.trajectory_import import import_legacy_trajectories
from codex_baseline_v2.analyst.analyst import analyze_episodes
from codex_baseline_v2.controller.controller import select_instruction
from codex_baseline_v2.executor.executor import execute_instruction_offline
from codex_baseline_v2.memory.store import load_blackboard, save_blackboard
from codex_baseline_v2.shared.config import V2Config
from codex_baseline_v2.shared.schemas import SCHEMA_VERSION
from codex_baseline_v2.shared.storage import StoragePathsV2
from codex_baseline_v2.trajectory_analysis.analyzer import analyze_trajectories


def _make_grid(color: int) -> List[List[int]]:
    return [[color for _ in range(5)] for _ in range(5)]


def _make_payload() -> dict:
    g0 = _make_grid(1)
    g1 = _make_grid(1)
    g1[2][2] = 2
    return {
        "schema_version": "TRAJECTORY_BATCH_V1",
        "episodes": [
            {
                "game_id": "test_game",
                "seed": 1,
                "steps": [
                    {"step_idx": 0, "reward": 0.0, "done": False, "grid_stack_t": [g0], "action_id": 0},
                    {"step_idx": 1, "reward": 0.0, "done": True, "grid_stack_t": [g1], "action_id": 1},
                ],
                "done": True,
                "win": False,
            }
        ],
    }


def test_analyst_static() -> None:
    payload = _make_payload()
    episodes = import_legacy_trajectories(payload)
    cfg = V2Config().analyst
    analyzed = analyze_episodes(episodes, cfg)
    assert analyzed[0].steps[0].observation_summary is not None


def test_avatar_detection() -> None:
    payload = _make_payload()
    episodes = import_legacy_trajectories(payload)
    cfg = V2Config().analyst
    analyzed = analyze_episodes(episodes, cfg)
    avatars = analyzed[0].steps[1].observation_summary.avatar_candidates
    assert isinstance(avatars, list)


def test_trajectory_analysis() -> None:
    payload = _make_payload()
    episodes = import_legacy_trajectories(payload)
    analyzed = analyze_episodes(episodes, V2Config().analyst)
    blackboard = analyze_trajectories(analyzed, V2Config().trajectory_analysis, round_id=0)
    assert blackboard.schema_version == SCHEMA_VERSION


def test_memory_cycle() -> None:
    payload = _make_payload()
    episodes = import_legacy_trajectories(payload)
    analyzed = analyze_episodes(episodes, V2Config().analyst)
    blackboard = analyze_trajectories(analyzed, V2Config().trajectory_analysis, round_id=0)
    with tempfile.TemporaryDirectory() as tmp:
        cfg = V2Config()
        mem_cfg = cfg.memory
        storage = StoragePathsV2(tmp)
        save_blackboard(mem_cfg, storage, blackboard)
        loaded = load_blackboard(storage, blackboard.game_id)
        assert loaded is not None


def test_controller_and_executor() -> None:
    payload = _make_payload()
    episodes = import_legacy_trajectories(payload)
    cfg = V2Config()
    analyzed = analyze_episodes(episodes, cfg.analyst)
    blackboard = analyze_trajectories(analyzed, cfg.trajectory_analysis, round_id=0)
    instruction = select_instruction(blackboard, cfg.controller, cfg.scoring, 1)
    outcome = execute_instruction_offline(analyzed, instruction, cfg.executor)
    assert outcome.schema_version == SCHEMA_VERSION


def test_end_to_end_dry_run() -> None:
    payload = _make_payload()
    episodes = import_legacy_trajectories(payload)
    cfg = V2Config()
    analyzed = analyze_episodes(episodes, cfg.analyst)
    blackboard = analyze_trajectories(analyzed, cfg.trajectory_analysis, round_id=0)
    instruction = select_instruction(blackboard, cfg.controller, cfg.scoring, 1)
    outcome = execute_instruction_offline(analyzed, instruction, cfg.executor)
    assert outcome.actions is not None


def run_all() -> None:
    test_analyst_static()
    test_avatar_detection()
    test_trajectory_analysis()
    test_memory_cycle()
    test_controller_and_executor()
    test_end_to_end_dry_run()
    print("smoke_tests_passed")


if __name__ == "__main__":
    run_all()
