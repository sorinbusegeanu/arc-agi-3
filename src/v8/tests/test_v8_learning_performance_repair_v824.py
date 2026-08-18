from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import v8
from v8 import adaptive_allocator_breadth_v840 as breadth
from v8 import adaptive_learning_allocation_v819 as v819
from v8 import adaptive_learning_allocation_v819_performance_fix as perf
from v8 import final_save_lifecycle_v812 as lifecycle
from v8 import learning_control_continuity_v826 as v826
from v8 import learning_performance_repair_v824 as repair
from v8 import runtime_repair_v822 as v822
from v8 import trajectory_optimizer_v814 as optimizer
from v8.model import MemoryLevel, MemoryType, MemoryUid, stable_u64
from v8.publication import LiveReadView, PlannedAction


class LearningPerformanceRepairV824Tests(unittest.TestCase):
    def test_final_install_authorities(self) -> None:
        self.assertIs(
            perf.__dict__["_v823_initial_unsolved_lease_steps"],
            breadth._initial_breadth_lease_steps_v840,
        )
        self.assertIs(
            breadth._BASE_UNSOLVED_LEASE_STEPS,
            v826.episode_aligned_unsolved_lease_steps_v826,
        )
        self.assertIs(v819._service_submit_v819, repair._prewin_submit_v824)
        self.assertIs(LiveReadView.plan_candidates, v826._plan_candidates_v826)
        self.assertIs(v822._BASE_PLAN_CANDIDATES, v826._plan_candidates_v826)
        self.assertIs(v826._BASE_PLAN_CANDIDATES, repair._plan_candidates_v824)
        self.assertEqual(lifecycle._LIFECYCLE_GENERATION_SPAN, 64)
        self.assertIs(v822._BASE_LIFECYCLE_WORKER, repair._lifecycle_worker_v824)
        self.assertEqual(repair._LIFECYCLE_MIN_INTERVAL_SECONDS, 60.0)

    def test_every_unsolved_lease_is_bounded(self) -> None:
        for initial in (False, True):
            self.assertEqual(
                repair.unsolved_lease_steps_v824(
                    available=100000,
                    base_steps=10000,
                    initial_probe=initial,
                    worker_count=10,
                    game_count=10,
                ),
                2048,
            )

    def test_discovery_probe_suppresses_existing_plan(self) -> None:
        prior = getattr(v822._PROBE_STATE, "before_plan", False)
        try:
            v822._PROBE_STATE.before_plan = True
            with patch.object(repair, "_BASE_PLAN_CHAIN", Mock(return_value=(object(),))) as base:
                self.assertEqual(repair._plan_candidates_v824(object(), 1, (1, 2)), ())
            base.assert_not_called()
        finally:
            v822._PROBE_STATE.before_plan = prior

    def test_transfer_filters_same_game_only_strategy(self) -> None:
        foreign = MemoryUid.from_key(MemoryLevel.M7, MemoryType.STRATEGY, (1, 2, 3))
        local = MemoryUid.from_key(MemoryLevel.M7, MemoryType.STRATEGY, (4, 5, 6))
        outcome = MemoryUid.from_key(MemoryLevel.M6, MemoryType.OUTCOME, (7, 8, 9))
        rows = (
            PlannedAction(1, outcome, local, 2.0, False),
            PlannedAction(2, outcome, foreign, 1.0, False),
        )
        game = "ic01"
        current_hash = stable_u64(game, person=b"v8-game")
        other_hash = stable_u64("other", person=b"v8-game")
        view = SimpleNamespace(
            source_games=lambda uid: (
                frozenset((current_hash,)) if uid == local else frozenset((other_hash,))
            )
        )
        prior_mode = os.environ.get(v819._SAMPLING_MODE_ENV)
        prior_source = optimizer._CAPTURE_SOURCE_ID
        try:
            os.environ[v819._SAMPLING_MODE_ENV] = v819.SamplingMode.TRANSFER.value
            optimizer._CAPTURE_SOURCE_ID = game
            with patch.object(repair, "_BASE_PLAN_CHAIN", Mock(return_value=rows)):
                selected = repair._plan_candidates_v824(view, 1, (1, 2))
            self.assertEqual(tuple(row.strategy_uid for row in selected), (foreign,))
        finally:
            optimizer._CAPTURE_SOURCE_ID = prior_source
            if prior_mode is None:
                os.environ.pop(v819._SAMPLING_MODE_ENV, None)
            else:
                os.environ[v819._SAMPLING_MODE_ENV] = prior_mode

    def test_transfer_label_requires_foreign_parent_strategy(self) -> None:
        strategy = MemoryUid.from_key(MemoryLevel.M7, MemoryType.STRATEGY, (10, 11, 12))
        anchor = optimizer.ReplayAnchor("ic01", 0, (), None)
        target = optimizer.TrajectoryTarget(1, "LEVEL")
        row = optimizer.SuccessfulTrajectory(
            "t", anchor, target, (1,), strategy, MemoryUid.zero(), 0
        )
        key = repair._foreign_key("ic01", strategy)
        prior_mode = os.environ.get(v819._SAMPLING_MODE_ENV)
        try:
            os.environ[v819._SAMPLING_MODE_ENV] = v819.SamplingMode.TRANSFER.value
            repair._FOREIGN_TRANSFER_STRATEGIES.discard(key)
            self.assertEqual(
                repair._success_to_dict_v824(row)["frontier_source"], "SAMPLER"
            )
            repair._FOREIGN_TRANSFER_STRATEGIES.add(key)
            self.assertEqual(
                repair._success_to_dict_v824(row)["frontier_source"], "TRANSFER"
            )
        finally:
            repair._FOREIGN_TRANSFER_STRATEGIES.discard(key)
            if prior_mode is None:
                os.environ.pop(v819._SAMPLING_MODE_ENV, None)
            else:
                os.environ[v819._SAMPLING_MODE_ENV] = prior_mode


if __name__ == "__main__":
    unittest.main()
