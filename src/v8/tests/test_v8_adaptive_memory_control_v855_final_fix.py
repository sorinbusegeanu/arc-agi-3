from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import v8  # noqa: F401 - install production runtime stack
from v8 import adaptive_learning_allocation_v819 as v819
from v8 import adaptive_memory_control_v855_fixups as fixups
from v8 import learning_performance_repair_v824 as v824
from v8 import sampling_portfolio_v831 as portfolio
from v8 import sampling_progress_control_v829 as v829
from v8.model import MemoryUid, stable_u64
from v8.publication import PlannedAction


class _SequenceRng:
    def __init__(self, *values: float) -> None:
        self.values = list(float(value) for value in values)
        self.calls = 0

    def random(self) -> float:
        self.calls += 1
        if self.values:
            return self.values.pop(0)
        return 0.0


class _View:
    def __init__(self, *, rng=None) -> None:
        self._behavior_rng = rng or _SequenceRng(0.0)
        self.strategy_uid = MemoryUid(7, 101)
        self.outcome_uid = MemoryUid(6, 102)
        self._node_by_uid = {
            self.strategy_uid: SimpleNamespace(
                strategy_reliability=1.0,
                attempt_weight=8.0,
            )
        }
        self._provenance = frozenset(
            {int(stable_u64("g1", person=b"v8-game"))}
        )

    def source_games(self, uid):
        del uid
        return self._provenance


def _plan(view: _View) -> PlannedAction:
    return PlannedAction(2, view.outcome_uid, view.strategy_uid, 1.0, False)


class AdaptiveMemoryControlV855FinalFixTests(unittest.TestCase):
    def setUp(self) -> None:
        v829._TESTED.clear()
        v829._PROGRESS_ACTION.clear()
        v829._NO_PROGRESS.clear()
        v829._CONTROL_STATE.game_id = "g1"
        v829._CONTROL_STATE.level = 0
        v829._CONTROL_STATE.context = None
        v829._CONTROL_STATE.selection_source = "UNKNOWN"
        v829._CONTROL_STATE.planned_actions = frozenset()
        portfolio._PORTFOLIO_STATE.mode = "MEMORY"
        self.prior_mode = os.environ.get(v819._SAMPLING_MODE_ENV)
        os.environ[v819._SAMPLING_MODE_ENV] = v819.SamplingMode.DISCOVERY.value

    def tearDown(self) -> None:
        if self.prior_mode is None:
            os.environ.pop(v819._SAMPLING_MODE_ENV, None)
        else:
            os.environ[v819._SAMPLING_MODE_ENV] = self.prior_mode
        v829._TESTED.clear()
        v829._PROGRESS_ACTION.clear()
        v829._NO_PROGRESS.clear()
        try:
            delattr(portfolio._PORTFOLIO_STATE, "mode")
        except AttributeError:
            pass

    def test_full_coverage_does_not_give_m7_a_second_draw_after_exploration_wins(self) -> None:
        state = ("g1", 0, 123)
        v829._TESTED[state] = {1, 2, 3}
        rng = _SequenceRng(0.99, 0.0)
        view = _View(rng=rng)
        plan = _plan(view)
        with patch.object(fixups, "_adaptive_candidates", return_value=((plan,), False)):
            rows = tuple(v824._BASE_PLAN_CHAIN(view, 123, (1, 2, 3)))
        self.assertEqual(rows, ())
        self.assertEqual(rng.calls, 1)
        self.assertEqual(v829._CONTROL_STATE.selection_source, "DISCOVERY")

    def test_full_coverage_composite_m7_is_not_refiltered_by_generic_no_progress(self) -> None:
        state = ("g1", 0, 123)
        v829._TESTED[state] = {1, 2, 3}
        v829._NO_PROGRESS[(*state, 2)] = 100
        v829._NO_PROGRESS[(*state, 1)] = 0
        view = _View(rng=_SequenceRng(0.0))
        plan = _plan(view)
        with (
            patch.object(fixups, "_adaptive_candidates", return_value=((plan,), False)),
            patch.object(fixups, "_arm_composite"),
        ):
            rows = tuple(v824._BASE_PLAN_CHAIN(view, 123, (1, 2, 3)))
        self.assertEqual(rows, (plan,))
        self.assertEqual(v829._CONTROL_STATE.selection_source, "M7_ADAPTIVE")

    def test_random_mode_remains_outside_adaptive_m7_authority(self) -> None:
        state = ("g1", 0, 123)
        v829._TESTED[state] = {1, 2, 3}
        portfolio._PORTFOLIO_STATE.mode = "RANDOM"
        view = _View(rng=_SequenceRng(0.0))
        plan = _plan(view)
        with patch.object(fixups, "_adaptive_candidates", return_value=((plan,), False)) as candidates:
            rows = tuple(v824._BASE_PLAN_CHAIN(view, 123, (1, 2, 3)))
        self.assertEqual(rows, ())
        candidates.assert_not_called()


if __name__ == "__main__":
    unittest.main()
