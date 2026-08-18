from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import v8  # noqa: F401 - install chronological runtime stack
from v8 import adaptive_learning_allocation_v819 as v819
from v8 import restart_causal_progress_v844 as v844
from v8 import sampling_portfolio_v831 as portfolio
from v8 import sampling_progress_control_v829 as v829
from v8 import trajectory_optimizer_v814 as optimizer
from v8.model import MemoryLevel, MemoryType, MemoryUid


_ACTIONS = (1, 2, 3, 4)


def _validated(
    *,
    variant: str,
    seed: int,
    actions=(1, 1),
    cost_parent: int = 4,
    target_level: int = 5,
    terminal: str = "WIN",
    strategy_uid: MemoryUid | None = None,
    outcome_uid: MemoryUid | None = None,
):
    anchor = optimizer.ReplayAnchor("ez01", int(seed), (), None)
    target = optimizer.TrajectoryTarget(int(target_level), str(terminal))
    strategy = strategy_uid or MemoryUid.from_key(
        MemoryLevel.M7,
        MemoryType.STRATEGY,
        (99, int(seed)),
    )
    outcome = outcome_uid or MemoryUid.from_key(
        MemoryLevel.M6,
        MemoryType.OUTCOME,
        (77, 1),
    )
    return optimizer.ValidatedTrajectory(
        str(variant),
        anchor,
        target,
        tuple(int(value) for value in actions),
        strategy,
        outcome,
        MemoryUid.zero(),
        int(cost_parent),
        "VALIDATE_SOURCE",
        2,
        2,
    )


class _EmptyLifecycleView:
    def __init__(self) -> None:
        self._node_by_uid = {}
        self._v814_variants = ()
        self._v814_next_refresh = 0.0
        self._v827_variant_lifecycle_mode = None

    def _refresh_strategy_cache(self) -> None:
        return None


class RestartCausalProgressV844Tests(unittest.TestCase):
    def setUp(self) -> None:
        v829._reset_sampling_state_v829()
        v844._CAUSAL_PROGRESS_STEPS.clear()
        v829._CONTROL_STATE.game_id = "ez01"
        v829._CONTROL_STATE.level = 0
        v829._CONTROL_STATE.context = None
        v829._CONTROL_STATE.selection_source = "UNKNOWN"
        v829._CONTROL_STATE.planned_actions = frozenset()
        portfolio._set_mode(None)

    def tearDown(self) -> None:
        portfolio._set_mode(None)
        for name in ("game_id", "level", "context", "selection_source", "planned_actions"):
            try:
                delattr(v829._CONTROL_STATE, name)
            except AttributeError:
                pass

    def test_semantic_trajectory_identity_is_seed_neutral(self) -> None:
        target = optimizer.TrajectoryTarget(5, "WIN")
        first = optimizer.ReplayAnchor("ez01", 11, (1, 1), None)
        second = optimizer.ReplayAnchor("ez01", 999, (1, 1), None)
        self.assertEqual(
            optimizer._anchor_hash(first, target),
            optimizer._anchor_hash(second, target),
        )

    def test_persisted_validated_trajectory_reuses_across_actor_seed(self) -> None:
        row = _validated(variant="seed-7", seed=7, actions=(1,) * 25, cost_parent=25)
        selected = optimizer.select_validated_variant(
            (row,),
            source_id="ez01",
            seed=123456,
            action_history=(),
        )
        self.assertIs(selected, row)
        self.assertEqual(int(selected.anchor.seed), 7)  # provenance is retained

    def test_newer_validated_sidecar_survives_older_snapshot_restore(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            service = optimizer.TrajectoryOptimizationService(
                Path(root),
                validator=lambda candidate: None,
            )
            old = _validated(
                variant="snapshot-old",
                seed=7,
                actions=(1,) * 25,
                cost_parent=25,
            )
            newer = _validated(
                variant="sidecar-new",
                seed=99,
                actions=(1,) * 20,
                cost_parent=25,
            )
            service.validated_path.write_text(
                json.dumps({"version": 1, "validated": [newer.to_dict()]}),
                encoding="utf-8",
            )
            service.load_state(
                {
                    "version": 1,
                    "validated": [old.to_dict()],
                    "seen_sources": [],
                    "attempted": [],
                    "metrics": {},
                }
            )
            rows = tuple(service._validated.values())
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].variant_id, "sidecar-new")
            published = optimizer._load_validated_rows(service.validated_path)
            self.assertEqual(tuple(row.variant_id for row in published), ("sidecar-new",))

    def test_durable_win_reconciles_solved_state_without_runtime_win_marker(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            service = optimizer.TrajectoryOptimizationService(
                Path(root),
                validator=lambda candidate: None,
            )
            row = _validated(
                variant="complete",
                seed=7,
                actions=(1,) * 25,
                cost_parent=25,
            )
            service._validated[optimizer._frontier_key(row.anchor, row.target)] = row
            service._publish_validated()

            coordinator = v819.AdaptiveLearningCoordinator()
            coordinator._v827_read_view = _EmptyLifecycleView()
            runtime = SimpleNamespace(
                _v814_trajectory_optimizer=service,
                _v819_adaptive_learning=coordinator,
                generation=10,
            )
            v844._reconcile_durable_competence_v844(runtime)

            self.assertEqual(
                coordinator.game_state("ez01"),
                v819.GameLearningState.SOLVED_OPTIMIZING,
            )
            self.assertEqual(coordinator.choose_mode("ez01"), v819.SamplingMode.VERIFY)
            self.assertIn("ez01", coordinator._v844_durable_complete_games)

    def test_verify_can_replay_missing_canonical_complete_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            row = _validated(
                variant="complete",
                seed=7,
                actions=(1,) * 25,
                cost_parent=25,
            )
            path = Path(root) / "validated.json"
            path.write_text(
                json.dumps({"version": 1, "validated": [row.to_dict()]}),
                encoding="utf-8",
            )
            view = _EmptyLifecycleView()
            prior_root = os.environ.get(optimizer._TRAJECTORY_ROOT_ENV)
            prior_mode = os.environ.get(v819._SAMPLING_MODE_ENV)
            prior_source = optimizer._CAPTURE_SOURCE_ID
            try:
                os.environ[optimizer._TRAJECTORY_ROOT_ENV] = root
                os.environ[v819._SAMPLING_MODE_ENV] = v819.SamplingMode.VERIFY.value
                optimizer._CAPTURE_SOURCE_ID = "ez01"
                optimizer._refresh_view_variants(view)
                self.assertEqual(
                    tuple(value.variant_id for value in view._v814_variants),
                    ("complete",),
                )
            finally:
                optimizer._CAPTURE_SOURCE_ID = prior_source
                if prior_root is None:
                    os.environ.pop(optimizer._TRAJECTORY_ROOT_ENV, None)
                else:
                    os.environ[optimizer._TRAJECTORY_ROOT_ENV] = prior_root
                if prior_mode is None:
                    os.environ.pop(v819._SAMPLING_MODE_ENV, None)
                else:
                    os.environ[v819._SAMPLING_MODE_ENV] = prior_mode

    def test_unproven_motion_does_not_auto_promote_to_action_persistence(self) -> None:
        sampler = portfolio.PortfolioSampler("ez01", seed=1)
        sampler.begin_lease(1)
        row = sampler._frontier(level=0, context=10, actions=_ACTIONS, history=())
        row.next_index = 2  # depth-1 ACTION3 / LEFT
        portfolio._set_mode("SEQUENCE")
        action = sampler.discovery_action(
            level=0,
            context=10,
            actions=_ACTIONS,
            history=(),
        )
        self.assertEqual(action, 3)
        sampler.observe_transition(
            before_level=0,
            before_context=10,
            action=3,
            after_level=0,
            after_context=11,
            after_actions=_ACTIONS,
            history_after=(3,),
            changed_cells=2,
            terminal_state="NOT_FINISHED",
            terminal_polarity=0,
            level_advanced=False,
            prediction_error=0.0,
            future_delta=0.0,
        )
        self.assertIsNone(getattr(sampler, "_v832_persist_action", None))

    def test_same_game_success_becomes_cross_level_causal_continuation(self) -> None:
        from v8 import decision_point_sampling_v821 as sampling

        sampler = portfolio.PortfolioSampler("ez01", seed=2)
        sampler.begin_lease(2)
        sampler.base.transfer_action = 1
        sampler.base.transfer_from_level = 0
        sampler.base.current = sampling.Intervention("TRANSFER", (1, 20), 1, ())
        v829._NO_PROGRESS[("ez01", 1, 20, 1)] = 1
        sampler.observe_transition(
            before_level=1,
            before_context=20,
            action=1,
            after_level=1,
            after_context=21,
            after_actions=_ACTIONS,
            history_after=(1,),
            changed_cells=2,
            terminal_state="NOT_FINISHED",
            terminal_polarity=0,
            level_advanced=False,
            prediction_error=0.0,
            future_delta=0.0,
        )
        self.assertEqual(getattr(sampler, "_v844_causal_action", None), 1)
        self.assertEqual(v829._NO_PROGRESS[("ez01", 1, 20, 1)], 0)

        action = sampler.forced_action(
            level=1,
            context=21,
            actions=_ACTIONS,
            history=(1,),
        )
        self.assertEqual(action, 1)
        self.assertEqual(sampler.base.current.kind, "CAUSAL_PROGRESS")
        sampler.observe_transition(
            before_level=1,
            before_context=21,
            action=1,
            after_level=2,
            after_context=30,
            after_actions=_ACTIONS,
            history_after=(1, 1),
            changed_cells=2,
            terminal_state="NOT_FINISHED",
            terminal_polarity=1,
            level_advanced=True,
            prediction_error=0.0,
            future_delta=0.0,
        )
        self.assertIsNone(sampler.base.verification)
        self.assertIsNone(sampler.base.pending_reset)
        self.assertEqual(
            sampler.forced_action(
                level=2,
                context=30,
                actions=_ACTIONS,
                history=(1, 1),
            ),
            1,
        )

    def test_ez01_cold_contract_discovers_then_completes_25_up_actions(self) -> None:
        """Regression for ez01: distances are 2,4,6,6,7 and only ACTION1 progresses."""

        sampler = portfolio.PortfolioSampler("ez01", seed=3)
        sampler.begin_lease(3)

        # Empty-memory sequence search reaches (ACTION1,ACTION1) as the fifth
        # bounded candidate: four depth-1 probes, then the first depth-2 prefix.
        row = sampler._frontier(level=0, context=100, actions=_ACTIONS, history=())
        row.next_index = 4
        portfolio._set_mode("SEQUENCE")
        first = sampler.discovery_action(
            level=0,
            context=100,
            actions=_ACTIONS,
            history=(),
        )
        self.assertEqual(first, 1)
        sampler.observe_transition(
            before_level=0,
            before_context=100,
            action=1,
            after_level=0,
            after_context=101,
            after_actions=_ACTIONS,
            history_after=(1,),
            changed_cells=2,
            terminal_state="NOT_FINISHED",
            terminal_polarity=0,
            level_advanced=False,
            prediction_error=0.0,
            future_delta=0.0,
        )
        second = sampler.forced_action(
            level=0,
            context=101,
            actions=_ACTIONS,
            history=(1,),
        )
        self.assertEqual(second, 1)
        sampler.observe_transition(
            before_level=0,
            before_context=101,
            action=1,
            after_level=1,
            after_context=200,
            after_actions=_ACTIONS,
            history_after=(1, 1),
            changed_cells=2,
            terminal_state="NOT_FINISHED",
            terminal_polarity=1,
            level_advanced=True,
            prediction_error=0.0,
            future_delta=0.0,
        )
        self.assertEqual(sampler.base.transfer_action, 1)

        # The next level probes the demonstrated action once; v8.44 then keeps the
        # causal macro alive across all remaining level boundaries.
        portfolio._set_mode("PROGRESS")
        action = sampler.discovery_action(
            level=1,
            context=200,
            actions=_ACTIONS,
            history=(1, 1),
        )
        self.assertEqual(action, 1)

        total_actions = 2
        level = 1
        context = 200
        remaining_by_level = [4, 6, 6, 7]
        history = [1, 1]
        for level_index, distance in enumerate(remaining_by_level):
            for step in range(distance):
                if not (level_index == 0 and step == 0):
                    action = sampler.forced_action(
                        level=level,
                        context=context,
                        actions=_ACTIONS,
                        history=tuple(history),
                    )
                    self.assertEqual(action, 1)
                total_actions += 1
                history.append(1)
                final_step = step == distance - 1
                final_game = level_index == len(remaining_by_level) - 1 and final_step
                next_level = level + 1 if final_step else level
                next_context = context + 1
                sampler.observe_transition(
                    before_level=level,
                    before_context=context,
                    action=1,
                    after_level=next_level,
                    after_context=next_context,
                    after_actions=_ACTIONS,
                    history_after=tuple(history),
                    changed_cells=2,
                    terminal_state="WIN" if final_game else "NOT_FINISHED",
                    terminal_polarity=1 if final_step else 0,
                    level_advanced=final_step,
                    prediction_error=0.0,
                    future_delta=0.0,
                )
                context = next_context
                level = next_level

        self.assertEqual(total_actions, 25)
        self.assertEqual(level, 5)
        self.assertIsNone(getattr(sampler, "_v844_causal_action", None))
        self.assertGreater(v844.causal_progress_telemetry_v844("ez01")["steps"], 0)


if __name__ == "__main__":
    unittest.main()
