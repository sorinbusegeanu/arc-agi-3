from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from v7.environment.arc_adapter import ArcGridEnvironment
from v7.environment.cognition import LocalCognitionOverlay
from v7.environment.encoding import (
    SupportedPredictionTracker,
    grid_signature,
    structural_grid_signature,
    transformation_family_signature,
    transition_signature,
)
from v7.environment.runner import ArcGameRunConfig, run_arc_game


def test_grid_and_transition_signatures_are_deterministic_and_sqlite_safe() -> None:
    before = np.array([[0, 1], [2, 3]], dtype=np.int64)
    after = np.array([[0, 1], [2, 4]], dtype=np.int64)
    assert grid_signature(before) == grid_signature(before.copy())
    assert transition_signature(before, after) == transition_signature(before.copy(), after.copy())
    assert transition_signature(before, before) != transition_signature(before, after)
    assert 0 <= grid_signature(before) < 2**63
    assert 0 <= transition_signature(before, after) < 2**63


def test_structural_context_and_transformation_family_generalize() -> None:
    a = np.array([[0, 1, 0], [0, 0, 0]], dtype=np.int64)
    b = np.array([[0, 7, 0], [0, 0, 0]], dtype=np.int64)
    assert structural_grid_signature(a) == structural_grid_signature(b)

    before_a = np.zeros((4, 4), dtype=np.int64)
    after_a = before_a.copy()
    before_a[0, 0] = 1
    after_a[0, 0] = 0
    after_a[0, 1] = 1
    before_b = np.zeros((4, 4), dtype=np.int64)
    after_b = before_b.copy()
    before_b[2, 1] = 7
    after_b[2, 1] = 0
    after_b[2, 2] = 7
    assert transformation_family_signature(before_a, after_a) == transformation_family_signature(before_b, after_b)


def test_worker_local_overlay_learns_before_canonical_commit() -> None:
    overlay = LocalCognitionOverlay(prediction_min_support=1)
    context = overlay.build_context(structural_signature=10, exact_signature=20)
    overlay.record_step(
        contexts=context.signatures,
        next_contexts=(),
        action_id=1,
        outcome_signature=100,
        terminal_polarity=-1,
        prediction_error=0.0,
        future_option_delta=-2.0,
        changed=False,
    )
    stats = overlay.stats_for(context.signatures[0], 1)
    assert stats.failure_risk == 1.0
    assert stats.no_change_ratio == 1.0


def test_prediction_error_is_inactive_until_supported_expectation_exists() -> None:
    tracker = SupportedPredictionTracker(minimum_support=2)
    assert tracker.prediction_error(10, 2, 100) == 0.0
    tracker.observe(10, 2, 100)
    assert tracker.prediction_error(10, 2, 101) == 0.0
    tracker.observe(10, 2, 100)
    assert tracker.prediction_error(10, 2, 100) == 0.0
    assert tracker.prediction_error(10, 2, 101) == 1.0


class _FakeEngine:
    def __init__(self) -> None:
        self.index = 0
        self.rows = [
            SimpleNamespace(frame=np.array([[0, 0], [0, 0]]), state="NOT_FINISHED", levels_completed=0, available_actions=[1, 2]),
            SimpleNamespace(frame=np.array([[0, 1], [0, 0]]), state="NOT_FINISHED", levels_completed=0, available_actions=[1, 2]),
            SimpleNamespace(frame=np.array([[0, 1], [0, 2]]), state="WIN", levels_completed=1, available_actions=[1]),
        ]

    def reset(self):
        self.index = 0
        return self.rows[0]

    def step(self, _action):
        self.index = min(self.index + 1, len(self.rows) - 1)
        return self.rows[self.index]


def test_arc_adapter_accepts_injected_engine_and_tracks_outcome() -> None:
    engine = _FakeEngine()
    env = ArcGridEnvironment(game_id="fake", env_factory=lambda **_: engine)
    assert env.available_actions() == [1, 2]
    env.step(1)
    assert env.last_outcome_state == "NOT_FINISHED"
    env.step(1)
    assert env.last_outcome_state == "WIN"
    assert env.last_outcome_polarity == "positive"
    assert env.level_completed_event is True


class _FakeAdapter:
    def __init__(self, **_kwargs) -> None:
        self.grid = np.array([[0]], dtype=np.int64)
        self.last_outcome_state = "NOT_FINISHED"
        self.last_outcome_polarity = "neutral"
        self.last_levels_completed = 0
        self.reset_count = 0

    def observe(self):
        return self.grid.copy()

    def available_actions(self):
        return [1, 2]

    def step(self, action):
        self.grid = np.array([[int(action)]], dtype=np.int64)
        return self.grid.copy()


def test_game_runner_ingests_transition_loop(tmp_path) -> None:
    result = run_arc_game(
        tmp_path,
        ArcGameRunConfig(game_id="fake", steps=4, commit_every=2, epsilon=0.0, restore=False),
        env_factory=_FakeAdapter,
    )
    assert result.steps == 4
    assert result.memories > 0
    assert result.generation >= 2
    assert (tmp_path / "state.sqlite").exists()
