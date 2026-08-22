from __future__ import annotations

import unittest

import v8
from v8 import click_exploration_v848 as v848
from v8 import sampling_evidence_frontier_v847_fixups as v847_fixups
from v8 import sampling_persistence_v832 as repair
from v8 import sampling_portfolio_v831 as portfolio
from v8 import sampling_progress_control_v829 as v829


class SamplingPersistenceV832Tests(unittest.TestCase):
    def setUp(self):
        v829._reset_sampling_state_v829()
        v829._CONTROL_STATE.game_id = "ez02"
        v829._CONTROL_STATE.level = 0
        v829._CONTROL_STATE.context = None
        v829._CONTROL_STATE.selection_source = "UNKNOWN"
        v829._CONTROL_STATE.planned_actions = frozenset()
        portfolio._set_mode(None)

    def tearDown(self):
        portfolio._set_mode(None)
        for name in ("game_id", "level", "context", "selection_source", "planned_actions"):
            try:
                delattr(v829._CONTROL_STATE, name)
            except AttributeError:
                pass

    def _observe(
        self,
        sampler,
        *,
        before_level: int,
        before_context: int,
        action: int,
        after_level: int,
        after_context: int,
        level_advanced: bool = False,
        terminal_state: str = "NOT_FINISHED",
        changed_cells: int = 2,
        history_after=(),
    ) -> None:
        sampler.observe_transition(
            before_level=before_level,
            before_context=before_context,
            action=action,
            after_level=after_level,
            after_context=after_context,
            after_actions=(1, 2, 3, 4),
            history_after=tuple(history_after),
            changed_cells=changed_cells,
            terminal_state=terminal_state,
            terminal_polarity=0,
            level_advanced=level_advanced,
            prediction_error=0.0,
            future_delta=0.0,
        )

    def test_unproven_productive_singleton_does_not_auto_persist(self):
        sampler = portfolio.PortfolioSampler("ez02", seed=1)
        sampler.begin_lease(1)
        portfolio._set_mode("SEQUENCE")
        action = sampler.discovery_action(
            level=0,
            context=10,
            actions=(1, 2, 3, 4),
            history=(),
        )
        self.assertIn(action, (1, 2, 3, 4))
        self._observe(
            sampler,
            before_level=0,
            before_context=10,
            action=action,
            after_level=0,
            after_context=11,
            history_after=(action,),
        )

        self.assertIsNone(getattr(sampler, "_v832_persist_action", None))
        self.assertFalse(hasattr(repair, "_MAX_ACTION_PERSISTENCE"))

    def test_level_progress_ends_rollout_and_exposes_transfer_probe(self):
        sampler = portfolio.PortfolioSampler("ez02", seed=2)
        sampler.begin_lease(2)
        repair._arm_persistence_v832(sampler, 3, None)
        action = sampler.forced_action(
            level=0,
            context=20,
            actions=(1, 2, 3, 4),
            history=(3,),
        )
        self.assertEqual(action, 3)
        self._observe(
            sampler,
            before_level=0,
            before_context=20,
            action=3,
            after_level=1,
            after_context=100,
            level_advanced=True,
            history_after=(3, 3),
        )

        self.assertIsNone(getattr(sampler, "_v832_persist_action", None))
        self.assertEqual(sampler.base.transfer_action, 3)
        self.assertEqual(sampler.base.transfer_from_level, 0)
        self.assertGreaterEqual(sampler.base.points[(1, 100)].priority, 6)

        portfolio._set_mode("TRANSFER")
        action = sampler.discovery_action(
            level=1,
            context=100,
            actions=(1, 2, 3, 4),
            history=(3, 3),
        )
        self.assertEqual(action, 3)
        self.assertEqual(v829._CONTROL_STATE.selection_source, "TRANSFER_PROBE")

    def test_persistence_stops_on_stationary_noop(self):
        sampler = portfolio.PortfolioSampler("ez02", seed=3)
        sampler.begin_lease(3)
        repair._arm_persistence_v832(sampler, 3, None)
        action = sampler.forced_action(
            level=0,
            context=30,
            actions=(1, 2, 3, 4),
            history=(),
        )
        self.assertEqual(action, 3)
        self._observe(
            sampler,
            before_level=0,
            before_context=30,
            action=3,
            after_level=0,
            after_context=30,
            changed_cells=0,
            history_after=(3,),
        )
        self.assertIsNone(getattr(sampler, "_v832_persist_action", None))

    def test_large_step_counter_does_not_stop_productive_rollout(self):
        sampler = portfolio.PortfolioSampler("ez02", seed=4)
        sampler.begin_lease(4)
        repair._arm_persistence_v832(sampler, 3, None)
        sampler._v832_persist_steps = 1_000_000
        action = sampler.forced_action(
            level=0,
            context=40,
            actions=(1, 2, 3, 4),
            history=(),
        )
        self.assertEqual(action, 3)
        self.assertEqual(v829._CONTROL_STATE.selection_source, "ACTION_PERSISTENCE")

    def test_install_keeps_v832_public_authority_and_composes_v847_below_it(self):
        self.assertIs(portfolio.PortfolioSampler.begin_lease, repair._begin_lease_v832)
        self.assertIs(portfolio.PortfolioSampler.on_external_reset, repair._on_external_reset_v832)
        self.assertIs(portfolio.PortfolioSampler.forced_action, repair._forced_action_v832)
        self.assertIs(portfolio.PortfolioSampler.observe_transition, repair._observe_transition_v832)
        self.assertIs(repair._BASE_ON_EXTERNAL_RESET, v847_fixups._lower_reset_v847)
        self.assertIs(repair._BASE_FORCED_ACTION, v848._sampler_forced_action_v848)
        self.assertIs(v848._BASE_SAMPLER_FORCED_ACTION, v847_fixups._lower_forced_v847)
        self.assertIs(repair._BASE_OBSERVE_TRANSITION, v848._sampler_observe_transition_v848)
        self.assertIs(v848._BASE_SAMPLER_OBSERVE_TRANSITION, v847_fixups._lower_observe_v847)


if __name__ == "__main__":
    unittest.main()
