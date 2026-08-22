from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import v8  # noqa: F401 - install current runtime stack
from v7.environment.arc_adapter import ArcGridEnvironment
from v7.game_sets import LEARNING_GAMES, resolve_game_selector
from v8 import adaptive_learning_allocation_v819 as v819
from v8 import adaptive_memory_control_v855 as v855
from v8 import adaptive_memory_control_v855_fixups as fixups
from v8 import behavior_recovery as behavior
from v8 import learning_blockers_v055 as blockers
from v8 import learning_performance_repair_v824 as v824
from v8 import performance_memory_v854 as performance
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
    def __init__(
        self,
        *,
        game_id: str = "g1",
        draw: float = 0.99,
        reliability: float = 0.9,
        attempts: float = 5.0,
        provenance: tuple[int, ...] | None = None,
    ) -> None:
        self._behavior_rng = _Rng(draw)
        self.strategy_uid = MemoryUid(7, 11)
        self.outcome_uid = MemoryUid(6, 13)
        self._node_by_uid = {
            self.strategy_uid: SimpleNamespace(
                strategy_reliability=float(reliability),
                attempt_weight=float(attempts),
            )
        }
        if provenance is None:
            provenance = (int(stable_u64(game_id, person=b"v8-game")),)
        self._provenance = frozenset(int(value) for value in provenance)

    def source_games(self, uid):
        del uid
        return self._provenance


def _plan(view: _View) -> PlannedAction:
    return PlannedAction(2, view.outcome_uid, view.strategy_uid, 1.0, False)


class AdaptiveMemoryControlV855ReviewFixTests(unittest.TestCase):
    def setUp(self) -> None:
        v855._reset_adaptive_memory_control_v855()
        v829._TESTED.clear()
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
        for name in (
            "game_id",
            "level",
            "context",
            "selection_source",
            "planned_actions",
        ):
            try:
                delattr(v829._CONTROL_STATE, name)
            except AttributeError:
                pass
        try:
            delattr(portfolio._PORTFOLIO_STATE, "mode")
        except AttributeError:
            pass
        v855._reset_adaptive_memory_control_v855()
        v829._TESTED.clear()
        v829._PROGRESS_ACTION.clear()
        v829._NO_PROGRESS.clear()

    def test_production_chain_can_use_strong_m7_before_local_coverage_finishes(self) -> None:
        state = ("g1", 0, 123)
        v829._TESTED[state] = {1}
        view = _View(draw=0.0)
        plan = _plan(view)
        with patch.object(v855, "_exact_m7_candidates", return_value=((plan,), False)):
            rows = tuple(v824._BASE_PLAN_CHAIN(view, 123, (1, 2, 3)))
        self.assertEqual(rows, (plan,))
        self.assertEqual(v829._CONTROL_STATE.selection_source, "M7_ADAPTIVE")
        self.assertIs(v824._BASE_PLAN_CHAIN, v829._plan_chain_v829)

    def test_memory_slot_can_use_strong_m7_before_local_coverage_finishes(self) -> None:
        state = ("g1", 0, 123)
        v829._TESTED[state] = {1}
        portfolio._PORTFOLIO_STATE.mode = "MEMORY"
        view = _View(draw=0.0)
        plan = _plan(view)
        with patch.object(v855, "_exact_m7_candidates", return_value=((plan,), False)):
            rows = tuple(v824._BASE_PLAN_CHAIN(view, 123, (1, 2, 3)))
        self.assertEqual(rows, (plan,))
        self.assertEqual(v829._CONTROL_STATE.selection_source, "M7_ADAPTIVE")

    def test_random_slot_preserves_exploration_before_local_coverage_finishes(self) -> None:
        state = ("g1", 0, 123)
        v829._TESTED[state] = {1}
        portfolio._PORTFOLIO_STATE.mode = "RANDOM"
        view = _View(draw=0.0)
        plan = _plan(view)
        with patch.object(v855, "_exact_m7_candidates", return_value=((plan,), False)):
            rows = tuple(v824._BASE_PLAN_CHAIN(view, 123, (1, 2, 3)))
        self.assertEqual(rows, ())
        self.assertNotEqual(v829._CONTROL_STATE.selection_source, "M7_ADAPTIVE")

    def test_starvation_release_is_scoped_to_context_and_strategy(self) -> None:
        view = _View(draw=0.99)
        plan = _plan(view)
        with patch.object(v855, "_exact_m7_candidates", return_value=((plan,), False)):
            for _ in range(v855._MAX_CONSECUTIVE_ARBITRATION_EXPLORATION):
                self.assertEqual(
                    tuple(v855._plan_chain_v855(view, 111, (1, 2, 3))),
                    (),
                )
            self.assertEqual(
                tuple(v855._plan_chain_v855(view, 222, (1, 2, 3))),
                (),
            )
            released = tuple(v855._plan_chain_v855(view, 111, (1, 2, 3)))
        self.assertEqual(released, (plan,))
        self.assertEqual(v855.adaptive_memory_control_telemetry_v855("g1")["starvation_release"], 1.0)

    def test_missing_provenance_cannot_autonomously_control_discovery(self) -> None:
        view = _View(draw=0.0, provenance=())
        plan = _plan(view)
        with patch.object(v855, "_exact_m7_candidates", return_value=((plan,), False)):
            rows = tuple(v855._plan_chain_v855(view, 123, (1, 2, 3)))
        self.assertEqual(rows, ())
        self.assertFalse(
            fixups._same_world_v855_fixup(
                view,
                view.strategy_uid,
                int(stable_u64("g1", person=b"v8-game")),
            )
        )

    def test_unrelated_no_progress_does_not_penalize_m7_strategy(self) -> None:
        state = ("g1", 0, 123)
        v829._NO_PROGRESS[(*state, 2)] = 100
        view = _View(draw=0.60, reliability=1.0, attempts=8.0)
        plan = _plan(view)
        with patch.object(v855, "_exact_m7_candidates", return_value=((plan,), False)):
            rows = tuple(v855._plan_chain_v855(view, 123, (1, 2, 3)))
        self.assertEqual(rows, (plan,))
        self.assertEqual(fixups._strategy_failure_evidence(view._node_by_uid[view.strategy_uid]), 0)
        self.assertEqual(v855.adaptive_memory_control_telemetry_v855("g1")["failure_backoff"], 0.0)

    def test_stagnation_escape_blocks_fresh_adaptive_m7(self) -> None:
        view = _View(draw=0.0)
        view._v055_escape_budget = 4
        plan = _plan(view)
        with patch.object(v855, "_exact_m7_candidates", return_value=((plan,), False)):
            rows = tuple(v855._plan_chain_v855(view, 123, (1, 2, 3)))
        self.assertEqual(rows, ())
        self.assertNotEqual(v829._CONTROL_STATE.selection_source, "M7_ADAPTIVE")

    def test_composite_m7_participates_in_adaptive_arbitration_and_arms_remainder(self) -> None:
        view = _View(draw=0.0)
        composite_uid = MemoryUid(7, 99)
        composite_outcome = MemoryUid(6, 77)
        composite_node = SimpleNamespace(
            strategy_reliability=0.95,
            attempt_weight=8.0,
        )
        view._node_by_uid[composite_uid] = composite_node
        composite_plan = PlannedAction(2, composite_outcome, composite_uid, 2.0, False)
        first = SimpleNamespace(key_parts=(123, 2, 0, 456))
        second = SimpleNamespace(key_parts=(456, 3, 0, 789))

        with (
            patch.object(v855, "_exact_m7_candidates", return_value=((), False)),
            patch.object(blockers, "_composite_plans", return_value=(composite_plan,)),
            patch.object(behavior, "strategy_can_control", return_value=True),
            patch.object(behavior, "_strategy_can_probe", return_value=False),
            patch.object(blockers, "is_composite_strategy", return_value=True),
            patch.object(blockers, "_path_for_composite", return_value=(first, second)),
        ):
            rows = tuple(v855._plan_chain_v855(view, 123, (1, 2, 3)))

        self.assertEqual(rows, (composite_plan,))
        self.assertEqual(
            view._v055_active_sequence,
            (composite_uid, composite_outcome, (second,)),
        )
        self.assertEqual(v829._CONTROL_STATE.selection_source, "M7_ADAPTIVE")

    def test_active_composite_sequence_continues_even_during_escape_window(self) -> None:
        view = _View(draw=0.99)
        strategy_uid = MemoryUid(7, 99)
        outcome_uid = MemoryUid(6, 77)
        node = SimpleNamespace(
            strategy_reliability=0.95,
            attempt_weight=8.0,
            expected_primary_valence=0.0,
            primary_valence_confidence=0.0,
        )
        outcome = SimpleNamespace(
            expected_primary_valence=0.0,
            primary_valence_confidence=0.0,
        )
        view._node_by_uid[strategy_uid] = node
        view._node_by_uid[outcome_uid] = outcome
        next_row = SimpleNamespace(key_parts=(456, 3, 0, 789))
        view._v055_active_sequence = (strategy_uid, outcome_uid, (next_row,))
        view._v055_escape_budget = 4

        rows = tuple(v855._plan_chain_v855(view, 456, (1, 2, 3)))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].action_id, 3)
        self.assertEqual(rows[0].strategy_uid, strategy_uid)
        self.assertEqual(view._v055_active_sequence, (strategy_uid, outcome_uid, ()))

    def test_learning_preset_resolves_against_real_local_registry_scan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for game_id in LEARNING_GAMES:
                metadata = root / game_id / "default" / "metadata.json"
                metadata.parent.mkdir(parents=True, exist_ok=True)
                metadata.write_text("{}", encoding="utf-8")
            self.assertEqual(resolve_game_selector("learning", env_root=tmp), LEARNING_GAMES)

    def test_fi01_patch_runs_through_real_arc_adapter_constructor(self) -> None:
        class Raw:
            frame = np.zeros((3, 3), dtype=np.int64)
            state = SimpleNamespace(value="NOT_FINISHED")
            levels_completed = 0
            available_actions = (1,)

        class Game:
            def _spread(self):
                return None

        class Engine:
            def __init__(self, game):
                self._game = game
                self.environment_info = SimpleNamespace(game_id="fi01")

            def reset(self):
                return Raw()

            def step(self, action):
                del action
                return Raw()

        game = Game()

        def factory(**kwargs):
            self.assertEqual(kwargs["env_id"], "fi01")
            return Engine(game)

        env = ArcGridEnvironment(game_id="fi01", env_factory=factory)
        self.assertIs(env.env._game, game)
        self.assertTrue(hasattr(game, "_v854_original_spread"))
        self.assertIs(game._spread.__func__, performance._spread_fi01_v854)


if __name__ == "__main__":
    unittest.main()
