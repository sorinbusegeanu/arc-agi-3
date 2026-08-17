from __future__ import annotations

import unittest

import v8
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

    def test_productive_singleton_extends_beyond_depth_four(self):
        sampler = portfolio.PortfolioSampler("ez02", seed=1)
        sampler.begin_lease(1)
        row = sampler._frontier(level=0, context=10, actions=(1, 2, 3, 4), history=())
        row.next_index = 2  # depth-1 candidate ACTION3 / LEFT
        portfolio._set_mode("SEQUENCE")
        action = sampler.discovery_action(
            level=0,
            context=10,
            actions=(1, 2, 3, 4),
            history=(),
        )
        self.assertEqual(action, 3)
        self._observe(
            sampler,
            before_level=0,
            before_context=10,
            action=3,
            after_level=0,
            after_context=11,
            history_after=(3,),
        )

        # v8.31 stopped arbitrary sequence enumeration at depth four.  Productive
        # action persistence must continue the same action well beyond that bound.
        for step in range(2, 8):
            action = sampler.forced_action(
                level=0,
                context=10 + step - 1,
                actions=(1, 2, 3, 4),
                history=(3,) * (step - 1),
            )
            self.assertEqual(action, 3)
            self.assertEqual(v829._CONTROL_STATE.selection_source, "ACTION_PERSISTENCE")
            self._observe(
                sampler,
                before_level=0,
                before_context=10 + step - 1,
                action=3,
                after_level=0,
                after_context=10 + step,
                history_after=(3,) * step,
            )

    def test_level_progress_carries_productive_action_to_next_level(self):
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

        action = sampler.forced_action(
            level=1,
            context=100,
            actions=(1, 2, 3, 4),
            history=(3, 3),
        )
        self.assertEqual(action, 3)
        self.assertEqual(v829._CONTROL_STATE.selection_source, "ACTION_PERSISTENCE")

    def test_persistence_stops_on_noop(self):
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

    def test_persistence_is_hard_bounded(self):
        sampler = portfolio.PortfolioSampler("ez02", seed=4)
        sampler.begin_lease(4)
        repair._arm_persistence_v832(sampler, 3, None)
        sampler._v832_persist_steps = repair._MAX_ACTION_PERSISTENCE
        action = sampler.forced_action(
            level=0,
            context=40,
            actions=(1, 2, 3, 4),
            history=(),
        )
        self.assertIsNone(getattr(sampler, "_v832_persist_action", None))
        # The normal portfolio may choose another action; only persistence must stop.
        self.assertNotEqual(v829._CONTROL_STATE.selection_source, "ACTION_PERSISTENCE")

    def test_install_patches_portfolio_sampler(self):
        self.assertIs(portfolio.PortfolioSampler.begin_lease, repair._begin_lease_v832)
        self.assertIs(portfolio.PortfolioSampler.on_external_reset, repair._on_external_reset_v832)
        self.assertIs(portfolio.PortfolioSampler.forced_action, repair._forced_action_v832)
        self.assertIs(portfolio.PortfolioSampler.observe_transition, repair._observe_transition_v832)


if __name__ == "__main__":
    unittest.main()
