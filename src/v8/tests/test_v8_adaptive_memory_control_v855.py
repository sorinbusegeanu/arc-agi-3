from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import v8
from v8 import adaptive_learning_allocation_v819 as v819
from v8 import adaptive_memory_control_v855 as control
from v8 import sampling_portfolio_v831 as portfolio
from v8 import sampling_progress_control_v829 as v829
from v8.model import MemoryUid, stable_u64
from v8.publication import PlannedAction


class _Rng:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def random(self) -> float:
        return self.value


class _View:
    def __init__(self, *, reliability=0.9, attempts=5.0, draw=0.0, source_games=()) -> None:
        self._behavior_rng = _Rng(draw)
        self.strategy_uid = MemoryUid(7, 11)
        self.outcome_uid = MemoryUid(6, 13)
        self._node_by_uid = {
            self.strategy_uid: SimpleNamespace(
                strategy_reliability=float(reliability),
                attempt_weight=float(attempts),
            )
        }
        self._source_games = frozenset(int(value) for value in source_games)

    def source_games(self, uid):
        del uid
        return self._source_games


class AdaptiveMemoryControlV855Tests(unittest.TestCase):
    def setUp(self) -> None:
        control._reset_adaptive_memory_control_v855()
        v829._PROGRESS_ACTION.clear()
        v829._NO_PROGRESS.clear()
        v829._CONTROL_STATE.game_id = "g1"
        v829._CONTROL_STATE.level = 0
        v829._CONTROL_STATE.context = None
        v829._CONTROL_STATE.selection_source = "UNKNOWN"
        v829._CONTROL_STATE.planned_actions = frozenset()
        portfolio._PORTFOLIO_STATE.mode = "SEQUENCE"
        self.prior_mode = os.environ.get(v819._SAMPLING_MODE_ENV)
        os.environ[v819._SAMPLING_MODE_ENV] = v819.SamplingMode.DISCOVERY.value

    def tearDown(self) -> None:
        if self.prior_mode is None:
            os.environ.pop(v819._SAMPLING_MODE_ENV, None)
        else:
            os.environ[v819._SAMPLING_MODE_ENV] = self.prior_mode
        for name in ("game_id", "level", "context", "selection_source", "planned_actions"):
            try:
                delattr(v829._CONTROL_STATE, name)
            except AttributeError:
                pass
        try:
            delattr(portfolio._PORTFOLIO_STATE, "mode")
        except AttributeError:
            pass
        control._reset_adaptive_memory_control_v855()
        v829._PROGRESS_ACTION.clear()
        v829._NO_PROGRESS.clear()

    @staticmethod
    def _plan(view: _View) -> PlannedAction:
        return PlannedAction(2, view.outcome_uid, view.strategy_uid, 1.0, False)

    def test_probability_keeps_exploration_floor_and_failure_backoff(self) -> None:
        cold = control.adaptive_m7_probability_v855(reliability=1.0, warm=False)
        warm = control.adaptive_m7_probability_v855(reliability=1.0, warm=True)
        failed = control.adaptive_m7_probability_v855(
            reliability=1.0,
            warm=True,
            failures=4,
        )
        probe = control.adaptive_m7_probability_v855(
            reliability=0.5,
            warm=False,
            probationary=True,
        )
        self.assertGreaterEqual(cold.exploration_probability, 0.25)
        self.assertGreaterEqual(warm.exploration_probability, 0.15)
        self.assertGreater(failed.exploration_probability, warm.exploration_probability)
        self.assertLessEqual(probe.memory_probability, 0.10)

    def test_strong_exact_m7_can_act_outside_memory_slot(self) -> None:
        view = _View(draw=0.10)
        plan = self._plan(view)
        with patch.object(control, "_exact_m7_candidates", return_value=((plan,), False)):
            rows = control._plan_chain_v855(view, 123, (1, 2, 3))
        self.assertEqual(rows, (plan,))
        self.assertEqual(v829._CONTROL_STATE.selection_source, "M7_ADAPTIVE")

    def test_exploration_floor_can_override_strong_m7(self) -> None:
        view = _View(draw=0.99)
        plan = self._plan(view)
        with patch.object(control, "_exact_m7_candidates", return_value=((plan,), False)):
            rows = control._plan_chain_v855(view, 123, (1, 2, 3))
        self.assertEqual(rows, ())
        telemetry = control.adaptive_memory_control_telemetry_v855("g1")
        self.assertEqual(telemetry["eligible"], 1.0)
        self.assertEqual(telemetry["exploration_floor"], 1.0)

    def test_random_slot_remains_unconditional_exploration(self) -> None:
        portfolio._PORTFOLIO_STATE.mode = "RANDOM"
        view = _View(draw=0.0)
        plan = self._plan(view)
        with patch.object(control, "_exact_m7_candidates", return_value=((plan,), False)) as exact:
            rows = control._plan_chain_v855(view, 123, (1, 2, 3))
        self.assertEqual(rows, ())
        exact.assert_not_called()

    def test_observed_progress_replay_stays_authoritative(self) -> None:
        state = ("g1", 0, 123)
        v829._PROGRESS_ACTION[state] = 3
        view = _View(draw=0.0)
        plan = self._plan(view)
        with patch.object(control, "_exact_m7_candidates", return_value=((plan,), False)) as exact:
            rows = control._plan_chain_v855(view, 123, (1, 2, 3))
        self.assertEqual(rows, ())
        self.assertEqual(v829._CONTROL_STATE.selection_source, "PROGRESS_REPLAY")
        exact.assert_not_called()

    def test_foreign_only_m7_remains_for_explicit_transfer_mode(self) -> None:
        foreign = stable_u64("other", person=b"v8-game")
        view = _View(draw=0.0, source_games=(foreign,))
        plan = self._plan(view)
        with patch.object(control, "_exact_m7_candidates", return_value=((plan,), False)):
            rows = control._plan_chain_v855(view, 123, (1, 2, 3))
        self.assertEqual(rows, ())

    def test_repeated_failure_reduces_memory_probability(self) -> None:
        clean = control.adaptive_m7_probability_v855(
            reliability=0.9,
            warm=True,
            failures=0,
        )
        failed = control.adaptive_m7_probability_v855(
            reliability=0.9,
            warm=True,
            failures=5,
        )
        self.assertLess(failed.memory_probability, clean.memory_probability)

    def test_arbitration_starvation_release_is_bounded(self) -> None:
        view = _View(draw=0.99)
        plan = self._plan(view)
        stats = control._stats("g1")
        stats["consecutive_exploration"] = float(
            control._MAX_CONSECUTIVE_ARBITRATION_EXPLORATION
        )
        with patch.object(control, "_exact_m7_candidates", return_value=((plan,), False)):
            rows = control._plan_chain_v855(view, 123, (1, 2, 3))
        self.assertEqual(rows, (plan,))
        self.assertEqual(control._stats("g1")["starvation_release"], 1.0)

    def test_non_discovery_modes_preserve_existing_policy(self) -> None:
        os.environ[v819._SAMPLING_MODE_ENV] = v819.SamplingMode.TRANSFER.value
        view = _View(draw=0.0)
        sentinel = (object(),)
        with patch.object(control, "_BASE_PLAN_CHAIN", return_value=sentinel) as base:
            rows = control._plan_chain_v855(view, 123, (1, 2, 3))
        self.assertIs(rows, sentinel)
        base.assert_called_once()


if __name__ == "__main__":
    unittest.main()
