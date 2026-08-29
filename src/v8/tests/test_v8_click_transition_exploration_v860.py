from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import v8
from v8 import click_exploration_v848 as click
from v8 import click_transition_exploration_v860 as v860
from v8 import sampling_evidence_frontier_v847 as frontier
from v8 import sampling_evidence_frontier_v847_fixups as frontier_fixups
from v8 import sampling_persistence_v832 as persistence
from v8.decision_point_sampling_v821 import Intervention
from v8.learning_blockers_v055 import pack_action_choice
from v8.sampling_portfolio_v831 import PortfolioSampler


class ClickTransitionExplorationV860Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.action = pack_action_choice(6, 4, 7)
        self.sampler = PortfolioSampler("gp03-v860-fixture", seed=0)
        self.sampler.begin_lease(0)

    def _observe(self, before: int, after: int, *, changed: int = 1, kind: str = "CLICK_SCAN") -> None:
        self.sampler.base.current = Intervention(kind, (0, before), self.action, ())
        kwargs = dict(
            before_level=0,
            before_context=before,
            action=self.action,
            after_level=0,
            after_context=after,
            after_actions=(self.action,),
            history_after=(self.action,),
            changed_cells=changed,
            terminal_state="",
            terminal_polarity=0,
            level_advanced=False,
            prediction_error=0.0,
            future_delta=0.0,
        )
        with patch.object(v860, "_BASE_OBSERVE_TRANSITION", return_value=None):
            v860._observe_transition_v860(self.sampler, **kwargs)

    def test_productive_scan_immediately_queues_same_coordinate(self) -> None:
        self._observe(10, 11)
        self.assertEqual(self.sampler._v860_pending_action, self.action)
        self.assertEqual(
            self.sampler._v860_seen_transitions,
            {(self.action, 10, 11)},
        )

        with patch.object(v860, "_BASE_FORCED_ACTION", return_value=None):
            selected = v860._forced_action_v860(
                self.sampler,
                level=0,
                context=11,
                actions=(self.action,),
                history=(self.action,),
            )
        self.assertEqual(selected, self.action)
        self.assertEqual(self.sampler.base.current.kind, "CLICK_CHARACTERIZE")

    def test_pending_characterization_prevents_frontier_reset(self) -> None:
        self._observe(20, 21)
        self.sampler.pending_sequence = ((1,), (0, 20), (self.action,))
        self.sampler.base.pending_reset = ((1,), (0, 20))
        with patch.object(v860, "_BASE_PREPARE_STEP", return_value=True) as delegated:
            self.assertFalse(v860._prepare_step_v860(self.sampler, SimpleNamespace()))
        delegated.assert_not_called()
        self.assertIsNone(self.sampler.pending_sequence)
        self.assertIsNone(self.sampler.base.pending_reset)

    def test_no_change_ends_characterization_and_returns_to_broad_search(self) -> None:
        self.sampler._v860_pending_action = self.action
        self._observe(30, 30, changed=0, kind="CLICK_CHARACTERIZE")
        self.assertIsNone(self.sampler._v860_pending_action)
        self.assertEqual(len(self.sampler._v860_seen_transitions), 0)

    def test_state_transition_cycle_stops_without_spending_another_click(self) -> None:
        self._observe(40, 41)
        self.assertEqual(self.sampler._v860_pending_action, self.action)
        self.sampler._v860_pending_action = None

        self._observe(41, 40, kind="CLICK_CHARACTERIZE")
        self.assertEqual(self.sampler._v860_pending_action, self.action)
        self.sampler._v860_pending_action = None

        self._observe(40, 41, kind="CLICK_CHARACTERIZE")
        self.assertIsNone(self.sampler._v860_pending_action)
        self.assertEqual(len(self.sampler._v860_seen_transitions), 2)

    def test_characterization_repeat_cap_is_bounded_and_configurable(self) -> None:
        prior = os.environ.get(v860._REPEAT_CAP_ENV)
        os.environ[v860._REPEAT_CAP_ENV] = "2"
        try:
            self._observe(50, 51)
            self.assertEqual(self.sampler._v860_pending_action, self.action)
            self.sampler._v860_pending_action = None
            self._observe(51, 52, kind="CLICK_CHARACTERIZE")
            self.assertIsNone(self.sampler._v860_pending_action)
            self.assertEqual(self.sampler._v860_repeat_depth[self.action], 2)
        finally:
            if prior is None:
                os.environ.pop(v860._REPEAT_CAP_ENV, None)
            else:
                os.environ[v860._REPEAT_CAP_ENV] = prior

    def test_level_progress_is_never_consumed_by_local_characterization(self) -> None:
        self.sampler.base.current = Intervention("CLICK_SCAN", (0, 60), self.action, ())
        kwargs = dict(
            before_level=0,
            before_context=60,
            action=self.action,
            after_level=1,
            after_context=61,
            after_actions=(self.action,),
            history_after=(self.action,),
            changed_cells=1,
            terminal_state="",
            terminal_polarity=1,
            level_advanced=True,
            prediction_error=1.0,
            future_delta=0.0,
        )
        with patch.object(v860, "_BASE_OBSERVE_TRANSITION", return_value=None):
            v860._observe_transition_v860(self.sampler, **kwargs)
        self.assertIsNone(self.sampler._v860_pending_action)

    def test_nonclick_transition_does_not_activate_characterization(self) -> None:
        movement = 1
        self.sampler.base.current = Intervention("CLICK_SCAN", (0, 70), movement, ())
        kwargs = dict(
            before_level=0,
            before_context=70,
            action=movement,
            after_level=0,
            after_context=71,
            after_actions=(movement,),
            history_after=(movement,),
            changed_cells=3,
            terminal_state="",
            terminal_polarity=0,
            level_advanced=False,
            prediction_error=0.0,
            future_delta=0.0,
        )
        with patch.object(v860, "_BASE_OBSERVE_TRANSITION", return_value=None):
            v860._observe_transition_v860(self.sampler, **kwargs)
        self.assertIsNone(self.sampler._v860_pending_action)

    def test_runtime_stack_preserves_historical_authorities_and_installs_v860_deep(self) -> None:
        self.assertIs(PortfolioSampler.begin_lease, persistence._begin_lease_v832)
        self.assertIs(PortfolioSampler.prepare_step, frontier._prepare_step_v847)
        self.assertIs(PortfolioSampler.forced_action, persistence._forced_action_v832)
        self.assertIs(PortfolioSampler.observe_transition, persistence._observe_transition_v832)

        self.assertIs(persistence._BASE_BEGIN_LEASE, click._sampler_begin_lease_v848)
        self.assertIs(frontier._BASE_PREPARE_STEP, click._sampler_prepare_step_v848)
        self.assertIs(persistence._BASE_FORCED_ACTION, click._sampler_forced_action_v848)
        self.assertIs(persistence._BASE_OBSERVE_TRANSITION, click._sampler_observe_transition_v848)

        self.assertIs(click._BASE_SAMPLER_BEGIN_LEASE, v860._begin_lease_v860)
        self.assertIs(click._BASE_SAMPLER_PREPARE_STEP, v860._prepare_step_v860)
        self.assertIs(click._BASE_SAMPLER_FORCED_ACTION, frontier_fixups._lower_forced_v847)
        self.assertIs(click._BASE_SAMPLER_OBSERVE_TRANSITION, frontier_fixups._lower_observe_v847)
        self.assertIs(frontier_fixups._BASE_LOWER_FORCED, v860._forced_action_v860)
        self.assertIs(frontier_fixups._BASE_LOWER_OBSERVE, v860._observe_transition_v860)


if __name__ == "__main__":
    unittest.main()
