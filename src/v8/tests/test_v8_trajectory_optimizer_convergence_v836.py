from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace
from unittest import mock

import v8  # noqa: F401 - installs the chronological runtime stack
from v8 import trajectory_inspection_v819 as inspection
from v8 import trajectory_optimizer_convergence_v836 as convergence
from v8 import trajectory_optimizer_v814 as optimizer
from v8 import trajectory_optimizer_v818 as v818
from v8 import trajectory_target_minimization_v820 as v820
from v8.environment_contract import BoundaryEvent, BoundaryScope


def full_source(actions, *, game="ez02", levels=5):
    values = tuple(int(value) for value in actions)
    anchor = optimizer.ReplayAnchor(game, 0, (), None)
    target = optimizer.TrajectoryTarget(levels, "WIN")
    return optimizer.SuccessfulTrajectory(
        optimizer._trajectory_id(anchor, target, values),
        anchor,
        target,
        values,
    )


def candidate(source, actions=None, kind="DELETE_SEGMENT"):
    values = tuple(source.actions if actions is None else actions)
    return optimizer.TrajectoryCandidate(
        optimizer._candidate_id(source, kind, values),
        source,
        kind,
        values,
        0,
        max(0, int(source.cost) - len(values)),
    )


class TrajectoryOptimizerConvergenceV836Tests(unittest.TestCase):
    def test_full_win_tries_action_kind_projection_before_generic_deletions(self):
        source = full_source((3, 2, 3, 1, 3, 4, 3, 2, 3, 1))
        rows = v820._generate_v820(
            source,
            optimizer.TrajectoryOptimizerConfig(max_candidates_per_round=16),
        )
        self.assertTrue(rows)
        self.assertEqual(rows[0].edit_kind, convergence._PROJECT_ACTION_KIND)
        self.assertEqual(rows[0].actions, (3, 3, 3, 3, 3))

    def test_non_full_win_keeps_existing_generator_semantics(self):
        values = (3, 2, 3, 1, 3, 4)
        anchor = optimizer.ReplayAnchor("ez02", 0, (), None)
        target = optimizer.TrajectoryTarget(2, "LEVEL")
        source = optimizer.SuccessfulTrajectory(
            optimizer._trajectory_id(anchor, target, values),
            anchor,
            target,
            values,
        )
        rows = v820._generate_v820(
            source,
            optimizer.TrajectoryOptimizerConfig(max_candidates_per_round=16),
        )
        self.assertTrue(rows)
        self.assertNotEqual(rows[0].edit_kind, convergence._PROJECT_ACTION_KIND)

    def test_final_optimizer_has_no_fixed_round_horizon(self):
        self.assertIs(v818._optimizer_loop_v818, convergence._optimizer_loop_v836)
        self.assertNotIn(
            "max_optimization_rounds",
            inspect.getsource(convergence._optimizer_loop_v836),
        )

    def test_full_win_replay_recovers_level_boundaries(self):
        source = full_source((3, 3, 3, 3, 3, 3), levels=3)
        row = candidate(source)

        class FakeEnv:
            def __init__(self):
                self.steps = 0
                self.last_levels_completed = 0
                self.last_outcome_state = ""

            def available_actions(self):
                return (3,)

            def step(self, action):
                self.steps += 1
                if self.steps in (2, 4, 6):
                    self.last_levels_completed += 1
                if self.steps == 6:
                    self.last_outcome_state = "WIN"
                return None

            def cognitive_subepisode_index(self):
                return self.last_levels_completed

            def cognitive_boundary_event(self):
                if self.last_outcome_state == "WIN":
                    return BoundaryEvent(BoundaryScope.EPISODE, +1, False)
                return BoundaryEvent(BoundaryScope.NONE, 0, True)

        class FakeReplayValidator:
            def __init__(self, _service, _game):
                pass

            def _environment(self, _seed, _root):
                return FakeEnv()

        service = SimpleNamespace(_v818_prefix_for=lambda _candidate: ())
        with mock.patch.object(v818, "_GameReplayValidator", FakeReplayValidator):
            levels = convergence._replay_full_win_levels(service, row)
        self.assertEqual(levels, ((3, 3), (3, 3), (3, 3)))

    def test_full_win_optimized_result_is_published_with_replayed_levels(self):
        source = full_source((3, 3, 3, 3, 3, 3), levels=3)
        row = candidate(
            source,
            (3, 3, 3, 3, 3, 3),
            convergence._PROJECT_ACTION_KIND,
        )
        validated = SimpleNamespace(variant_id="optimized-a3", attempts=2, successes=2)
        result = SimpleNamespace(attempts=2, successes=2)
        captured = []
        service = SimpleNamespace(_log=lambda *_args, **_kwargs: None)

        with mock.patch.object(
            convergence,
            "_BASE_PUBLISH_OPTIMIZED_SOLUTION",
            return_value=False,
        ), mock.patch.object(
            convergence,
            "_replay_full_win_levels",
            return_value=((3, 3), (3, 3), (3, 3)),
        ), mock.patch.object(
            inspection,
            "_consider_best_solution",
            side_effect=lambda _service, record: captured.append(record) or True,
        ):
            self.assertTrue(
                convergence._publish_optimized_solution_v836(
                    service,
                    row,
                    result,
                    validated,
                )
            )

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["source"], "optimized")
        self.assertEqual(captured[0]["total_cost"], 6)
        self.assertEqual(
            tuple(tuple(level["actions"]) for level in captured[0]["levels"]),
            ((3, 3), (3, 3), (3, 3)),
        )


if __name__ == "__main__":
    unittest.main()
