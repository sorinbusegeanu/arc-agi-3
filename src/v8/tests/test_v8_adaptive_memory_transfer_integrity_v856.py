from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import v8  # noqa: F401 - install production runtime stack
from v8 import adaptive_learning_allocation_v819 as v819
from v8 import adaptive_memory_control_v855 as v855
from v8 import adaptive_memory_transfer_integrity_v856 as v856
from v8 import behavior_recovery as behavior
from v8 import sampling_portfolio_v831 as portfolio
from v8 import sampling_progress_control_v829 as v829
from v8.actor_read_view_v851 import ActorReadView
from v8.arena import EdgeRecord
from v8.model import MemoryLevel, MemoryType, MemoryUid, RelationType, stable_u64
from v8.publication import PlannedAction


class _Arena:
    def __init__(self, rows) -> None:
        self.rows = tuple(rows)
        self.count = len(self.rows)
        self.sequence = 2

    def read(self, index: int):
        return self.rows[int(index)]


class _TransferCutView:
    _load_needed_low = staticmethod(ActorReadView._load_needed_low)
    _load_provenance_games = staticmethod(ActorReadView._load_provenance_games)

    def __init__(self, *, nodes, edges, initial_nodes, dependencies) -> None:
        self._strategy_version = (2, 2)
        self._node_by_uid = {row.uid: row for row in initial_nodes}
        self._behavior_strategy_dependencies = dependencies
        self._nodes = (_Arena(nodes),)
        self._edges = (_Arena(edges),)
        self._v851_compact_edges = ()
        self._v851_compact_nodes = tuple(initial_nodes)
        self._source_games_direct = {}
        self._source_games_cache = {}
        self._parents = {}
        self.published = 0

    def _publish_compact_cut(self) -> None:
        self.published += 1


class AdaptiveMemoryTransferIntegrityV856Tests(unittest.TestCase):
    def setUp(self) -> None:
        v856._RECENT_FAILURES.clear()
        v856._TARGET_TRANSFER_FAILURES.clear()
        v829._PROGRESS_ACTION.clear()
        v829._CONTROL_STATE.game_id = "g1"
        v829._CONTROL_STATE.level = 0
        v829._CONTROL_STATE.context = 123
        portfolio._PORTFOLIO_STATE.mode = "SEQUENCE"
        self.prior_mode = os.environ.get(v819._SAMPLING_MODE_ENV)
        os.environ[v819._SAMPLING_MODE_ENV] = v819.SamplingMode.DISCOVERY.value
        self.prior_view = behavior._CURRENT_ACTOR_VIEW

    def tearDown(self) -> None:
        behavior._CURRENT_ACTOR_VIEW = self.prior_view
        if self.prior_mode is None:
            os.environ.pop(v819._SAMPLING_MODE_ENV, None)
        else:
            os.environ[v819._SAMPLING_MODE_ENV] = self.prior_mode
        v856._RECENT_FAILURES.clear()
        v856._TARGET_TRANSFER_FAILURES.clear()
        v829._PROGRESS_ACTION.clear()
        for name in ("game_id", "level", "context"):
            try:
                delattr(v829._CONTROL_STATE, name)
            except AttributeError:
                pass
        try:
            delattr(portfolio._PORTFOLIO_STATE, "mode")
        except AttributeError:
            pass

    def test_combined_floor_is_25_percent_cold_and_15_percent_warm(self) -> None:
        self.assertAlmostEqual(v856.combined_exploration_floor_v856(warm=False), 0.25)
        self.assertAlmostEqual(v856.combined_exploration_floor_v856(warm=True), 0.15)
        cold = v855.adaptive_m7_probability_v855(reliability=1.0, warm=False)
        warm = v855.adaptive_m7_probability_v855(reliability=1.0, warm=True)
        self.assertAlmostEqual(
            v856._PORTFOLIO_RANDOM_FLOOR
            + (1.0 - v856._PORTFOLIO_RANDOM_FLOOR) * cold.exploration_probability,
            0.25,
        )
        self.assertAlmostEqual(
            v856._PORTFOLIO_RANDOM_FLOOR
            + (1.0 - v856._PORTFOLIO_RANDOM_FLOOR) * warm.exploration_probability,
            0.15,
        )

    def test_forced_exploration_clears_stale_m7_credit_state(self) -> None:
        stale = PlannedAction(2, MemoryUid(6, 1), MemoryUid(7, 1), 1.0, False)
        view = SimpleNamespace(_behavior_last_plans=(stale,))
        behavior._CURRENT_ACTOR_VIEW = view
        with patch.object(v856, "_BASE_FORCED_DELEGATE", return_value=3):
            action = v856._forced_delegate_v856(SimpleNamespace(), level=0, context=1, actions=(1, 2, 3), history=())
        self.assertEqual(action, 3)
        self.assertEqual(view._behavior_last_plans, ())

    def test_active_composite_crosses_random_portfolio_slot(self) -> None:
        portfolio._PORTFOLIO_STATE.mode = "RANDOM"
        strategy_uid = MemoryUid(7, 9)
        outcome_uid = MemoryUid(6, 9)
        plan = PlannedAction(3, outcome_uid, strategy_uid, 1.0, False)
        view = SimpleNamespace(
            _behavior_last_plans=(),
            _v055_active_sequence=(strategy_uid, outcome_uid, (object(),)),
        )
        with (
            patch("v8.adaptive_memory_control_v855_fixups._continue_composite", return_value=(plan,)) as continued,
            patch.object(v856, "_BASE_PLAN_CHAIN", return_value=()) as base,
        ):
            rows = v856._plan_chain_v856(view, 123, (1, 2, 3))
        self.assertEqual(rows, (plan,))
        self.assertEqual(view._behavior_last_plans, (plan,))
        continued.assert_called_once()
        base.assert_not_called()

    def test_recent_failure_is_strategy_context_specific_not_lifetime_reliability(self) -> None:
        uid = MemoryUid(7, 10)
        node = SimpleNamespace(uid=uid, attempt_weight=100.0, strategy_reliability=0.51)
        self.assertEqual(v856._strategy_failure_evidence_v856(node), 0)
        v856._record_strategy_result("g1", 123, uid, success=False, foreign=False)
        self.assertEqual(v856._strategy_failure_evidence_v856(node), 1)
        v829._CONTROL_STATE.context = 999
        self.assertEqual(v856._strategy_failure_evidence_v856(node), 0)
        v829._CONTROL_STATE.context = 123
        v856._record_strategy_result("g1", 123, uid, success=True, foreign=False)
        self.assertEqual(v856._strategy_failure_evidence_v856(node), 0)

    def test_sampler_cross_game_m7_publishes_plan_for_learning_credit(self) -> None:
        strategy_uid = MemoryUid(7, 11)
        outcome_uid = MemoryUid(6, 12)
        strategy = SimpleNamespace(
            uid=strategy_uid,
            key_parts=(2, outcome_uid.hi, outcome_uid.lo, 99),
            strategy_reliability=0.9,
        )
        view = SimpleNamespace(_node_by_uid={strategy_uid: strategy}, _behavior_last_plans=())
        behavior._CURRENT_ACTOR_VIEW = view
        with patch.object(v856, "_BASE_CROSS_GAME", return_value=(2, "M7", strategy_uid)):
            selected = v856._cross_game_v856(SimpleNamespace(game_id="g1"), (1, 2, 3))
        self.assertEqual(selected, (2, "M7", strategy_uid))
        self.assertEqual(len(view._behavior_last_plans), 1)
        self.assertEqual(view._behavior_last_plans[0].strategy_uid, strategy_uid)
        self.assertEqual(view._behavior_last_plans[0].outcome_uid, outcome_uid)

    def test_repeated_target_failure_temporarily_blocks_same_foreign_strategy(self) -> None:
        uid = MemoryUid(7, 13)
        for _ in range(v856._TRANSFER_BACKOFF_LIMIT):
            v856._record_strategy_result("target", 123, uid, success=False, foreign=True)
        with patch.object(
            v856,
            "_BASE_TRANSFER_INDEX",
            return_value={2: ((3.0, uid, "M7"),)},
        ):
            rows = v856._transfer_index_v856(SimpleNamespace(), "target")
        self.assertEqual(rows, {})

    def test_compact_actor_cut_restores_normalized_grounding_and_exact_provenance(self) -> None:
        source_uid = MemoryUid(1, 101)
        target_uid = MemoryUid(1, 102)
        normalized_uid = MemoryUid(1, 103)
        strategy_uid = MemoryUid(7, 104)
        source = SimpleNamespace(
            uid=source_uid,
            level=int(MemoryLevel.M1),
            memory_type=int(MemoryType.CONTINGENCY),
            key_parts=(10, 2, 20, 11),
        )
        target = SimpleNamespace(
            uid=target_uid,
            level=int(MemoryLevel.M1),
            memory_type=int(MemoryType.CONTINGENCY),
            key_parts=(30, 5, 40, 31),
        )
        normalized = SimpleNamespace(
            uid=normalized_uid,
            level=int(MemoryLevel.M1),
            memory_type=int(MemoryType.CONTINGENCY),
            key_parts=(999,),
        )
        strategy = SimpleNamespace(
            uid=strategy_uid,
            level=int(MemoryLevel.M7),
            memory_type=int(MemoryType.STRATEGY),
            key_parts=(2, 6, 7, 8),
        )
        foreign_hash = int(stable_u64("foreign", person=b"v8-game"))
        target_hash = int(stable_u64("target", person=b"v8-game"))
        edges = (
            EdgeRecord(normalized_uid, int(RelationType.EXPLAINS), source_uid, 1, 1),
            EdgeRecord(normalized_uid, int(RelationType.EXPLAINS), target_uid, 1, 1),
            EdgeRecord(source_uid, int(RelationType.GAME_PROVENANCE), MemoryUid(0, foreign_hash), 1, 1),
            EdgeRecord(target_uid, int(RelationType.GAME_PROVENANCE), MemoryUid(0, target_hash), 1, 1),
        )
        view = _TransferCutView(
            nodes=(source, target, normalized, strategy),
            edges=edges,
            initial_nodes=(source, strategy),
            dependencies={strategy_uid: {source_uid}},
        )
        v856._augment_actor_transfer_cut(view)
        self.assertIn(normalized_uid, view._node_by_uid)
        self.assertIn(target_uid, view._node_by_uid)
        compact = {
            (edge.source_uid, int(edge.relation_type), edge.target_uid)
            for edge in view._v851_compact_edges
        }
        self.assertIn((normalized_uid, int(RelationType.EXPLAINS), source_uid), compact)
        self.assertIn((normalized_uid, int(RelationType.EXPLAINS), target_uid), compact)
        self.assertIn(
            (target_uid, int(RelationType.GAME_PROVENANCE), MemoryUid(0, target_hash)),
            compact,
        )
        self.assertEqual(view._source_games_direct[target_uid], {target_hash})
        self.assertGreater(view.published, 0)


if __name__ == "__main__":
    unittest.main()
