from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from v6.evaluation.h12_efficiency_emergence import evaluate_h12_efficiency_emergence
from v6.main import V6Config, V6System
from v6.memory.trajectory_efficiency import (
    TrajectoryEfficiencyStore,
    TrajectoryEfficiencyTracker,
    TrajectoryStep,
    compact_state_hash,
    save_best_known_solution_lengths,
)


def _step(
    interaction_id: int,
    *,
    before: str,
    after: str,
    future_option_delta: float | None = None,
    repeated_state: bool = False,
    repeated_context_action: bool = False,
    no_effect_action: bool = False,
) -> TrajectoryStep:
    return TrajectoryStep(
        interaction_id=interaction_id,
        global_step=interaction_id,
        state_hash_before=before,
        state_hash_after=after,
        future_option_delta=future_option_delta,
        repeated_state=repeated_state,
        repeated_context_action=repeated_context_action,
        no_effect_action=no_effect_action,
        outcome_state=None,
        level_completed_event=False,
        action_cost=1.0,
        memory_fitness_base=0.3,
        memory_replay_priority_base=0.2,
    )


def test_h12_shorter_successful_trajectory_has_higher_normalized_efficiency() -> None:
    tracker = TrajectoryEfficiencyTracker(best_known_solution_lengths={"g1|l1": 4})
    short = tracker.finalize_trajectory(
        trajectory_id="t1",
        game_id="g1",
        level_id="l1",
        sampler="s",
        seed=0,
        epoch=1,
        outcome_class="WIN",
        terminal=True,
        steps=[_step(1, before="a", after="b"), _step(2, before="b", after="c"), _step(3, before="c", after="d"), _step(4, before="d", after="e")],
    )
    long = tracker.finalize_trajectory(
        trajectory_id="t2",
        game_id="g1",
        level_id="l1",
        sampler="s",
        seed=0,
        epoch=1,
        outcome_class="WIN",
        terminal=True,
        steps=[_step(i, before=str(i), after=str(i + 1)) for i in range(1, 7)],
    )
    assert short.normalized_solve_efficiency == 1.0
    assert long.normalized_solve_efficiency is not None
    assert short.normalized_solve_efficiency > long.normalized_solve_efficiency


def test_h12_failed_short_trajectory_gets_no_bonus_when_future_options_worse() -> None:
    tracker = TrajectoryEfficiencyTracker()
    tracker.group_counts["g1|l1|GAME_OVER|state:s2"] = 1
    record = tracker.finalize_trajectory(
        trajectory_id="tfail",
        game_id="g1",
        level_id="l1",
        sampler="s",
        seed=0,
        epoch=1,
        outcome_class="GAME_OVER",
        terminal=True,
        steps=[_step(1, before="s1", after="s2", future_option_delta=-0.5, no_effect_action=True)],
    )
    assert record.efficiency_active is False
    assert record.efficiency_memory_bonus == 0.0
    assert record.efficiency_replay_bonus == 0.0


def test_h12_no_comparable_group_means_efficiency_inactive() -> None:
    tracker = TrajectoryEfficiencyTracker()
    record = tracker.finalize_trajectory(
        trajectory_id="t0",
        game_id="g1",
        level_id="l1",
        sampler="s",
        seed=0,
        epoch=1,
        outcome_class="WIN",
        terminal=True,
        steps=[_step(1, before="a", after="b"), _step(2, before="b", after="c")],
    )
    assert record.efficiency_active is False
    assert record.efficiency_score is None


def test_h12_loop_and_blocked_ratios_are_computed() -> None:
    tracker = TrajectoryEfficiencyTracker()
    tracker.group_counts["g1|l1|WIN|success"] = 1
    record = tracker.finalize_trajectory(
        trajectory_id="tloop",
        game_id="g1",
        level_id="l1",
        sampler="s",
        seed=0,
        epoch=1,
        outcome_class="WIN",
        terminal=True,
        steps=[
            _step(1, before="a", after="b"),
            _step(2, before="b", after="a", repeated_state=True),
            _step(3, before="a", after="a", repeated_state=True, no_effect_action=True),
        ],
    )
    assert record.loop_ratio > 0.0
    assert record.repeated_state_ratio > 0.0
    assert record.blocked_action_ratio > 0.0


def test_h12_best_known_solution_lengths_persist(tmp_path) -> None:
    path = tmp_path / "efficiency" / "best_known_solution_lengths.json"
    save_best_known_solution_lengths(path, {"g1|l1": 5})
    assert path.exists()
    assert "g1|l1" in path.read_text(encoding="utf-8")


class _TinyEnv:
    def __init__(self) -> None:
        self.game_id = "tt01"
        self.seed = 0
        self._obs = np.array([[0]], dtype=int)
        self._step = 0
        self.last_outcome_state = "NOT_FINISHED"
        self.last_outcome_polarity = "neutral"
        self.level_completed_event = False
        self.last_step_was_reset_boundary = False
        self.last_terminal_state = None

    def observe(self) -> np.ndarray:
        return self._obs.copy()

    def step(self, action: int) -> np.ndarray:
        self._step += 1
        self._obs = np.array([[self._step]], dtype=int)
        self.level_completed_event = self._step >= 2
        self.last_outcome_state = "NOT_FINISHED"
        self.last_outcome_polarity = "positive" if self.level_completed_event else "neutral"
        self.last_step_was_reset_boundary = False
        self.last_terminal_state = None
        return self._obs.copy()

    def available_actions(self) -> list[int]:
        return [0]


class _Sampler:
    name = "unit_sampler"

    def choose_action(self, system, actions):
        return int(actions[0])

    def record_result(self, **kwargs):
        return None


def test_h12_v6system_integration_writes_trajectory_efficiency_and_bonus_fields(tmp_path) -> None:
    db_path = tmp_path / "runtime.sqlite"
    system = V6System(
        _TinyEnv(),
        V6Config(database_path=str(db_path), context_length=1, max_context_depth=1, memory_output_dir=str(tmp_path / "memory")),
        action_sampler=_Sampler(),
    )
    system.run(steps=2)
    trajectory_rows = system.connection.execute(
        "SELECT trajectory_id, efficiency_active, efficiency_memory_bonus FROM trajectory_efficiency"
    ).fetchall()
    assert trajectory_rows
    columns = {row[1] for row in system.connection.execute("PRAGMA table_info(interactions)").fetchall()}
    assert "trajectory_efficiency_score" in columns
    assert "state_hash_before" in columns
    interaction_row = system.connection.execute(
        "SELECT state_hash_before, state_hash_after, trajectory_outcome_class FROM interactions ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert interaction_row[0]
    assert interaction_row[1]
    assert interaction_row[2] == "LEVEL_COMPLETE"


def test_h12_report_returns_insufficient_evidence_without_comparable_successes(tmp_path) -> None:
    db_path = tmp_path / "seed_0.sqlite"
    with sqlite3.connect(db_path) as conn:
        store = TrajectoryEfficiencyStore(conn)
        store.upsert(
            tracker_record := TrajectoryEfficiencyTracker().finalize_trajectory(
                trajectory_id="t1",
                game_id="g1",
                level_id="l1",
                sampler="s",
                seed=0,
                epoch=1,
                outcome_class="GAME_OVER",
                terminal=True,
                steps=[_step(1, before="a", after="b", future_option_delta=-0.1)],
            )
        )
        assert tracker_record.success is False
    report = evaluate_h12_efficiency_emergence(run_dir=tmp_path, memory_dir=None, output_dir=tmp_path / "reports" / "h12")
    assert report["decision"] == "INSUFFICIENT_EVIDENCE"


def test_h12_state_hash_helper_is_deterministic() -> None:
    grid = np.array([[1, 2], [3, 4]], dtype=int)
    assert compact_state_hash(grid) == compact_state_hash(grid.copy())
