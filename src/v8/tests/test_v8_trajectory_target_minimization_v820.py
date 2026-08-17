from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import v8
from v8 import trajectory_target_minimization_v820 as v820
from v8 import trajectory_optimizer_v814 as optimizer
from v8.model import MemoryUid


def source(actions=tuple(range(1, 11)), *, prefix=(), game="world", level=1, terminal="LEVEL"):
    anchor = optimizer.ReplayAnchor(game, 0, tuple(prefix), None)
    target = optimizer.TrajectoryTarget(level, terminal)
    values = tuple(int(value) for value in actions)
    return optimizer.SuccessfulTrajectory(
        optimizer._trajectory_id(anchor, target, values),
        anchor,
        target,
        values,
        MemoryUid.zero(),
        MemoryUid.zero(),
        0,
    )


def result(success, lengths=(), *, attempts=2, successes=None, prefix=()):
    count = int(success) if successes is None else int(successes)
    return v820.V820ValidationResult(
        bool(success),
        sum(int(value) for value in lengths),
        "target_preserved" if success else "target_not_reached",
        "LEVEL" if success else "",
        1 if success else 0,
        int(attempts),
        count,
        tuple(prefix),
        11 if success else 0,
        1 if success else 0,
        22 if success else 0,
        tuple(int(value) for value in lengths),
    )


class FakeValidator:
    def __init__(self, fn):
        self.fn = fn
        self.calls = []

    def validate(self, candidate):
        self.calls.append((candidate.edit_kind, tuple(candidate.actions)))
        return self.fn(candidate)


class PrefixCanonicalizationTests(unittest.TestCase):
    def test_two_seed_success_lengths_store_max_required_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            service = optimizer.TrajectoryOptimizationService(Path(root), validator=lambda _c: None)
            row = source(tuple(range(100)))
            candidate = v820._candidate(optimizer, row, "DELETE_SEGMENT", tuple(range(80)), 80, 20)

            def validate(c):
                if len(c.actions) == 80:
                    return result(True, (3, 5), successes=2)
                if len(c.actions) == 5:
                    return result(True, (5, 5), successes=2)
                return result(False)

            accepted, validated = v820._process_candidate(service, FakeValidator(validate), candidate)
            self.assertTrue(accepted)
            self.assertIsNotNone(validated)
            self.assertEqual(tuple(validated.actions), tuple(range(5)))
            self.assertEqual(validated.cost, 5)
            self.assertEqual(validated.edit_kind, v820._TRUNCATE_SUCCESS_PREFIX)

    def test_failed_shortened_revalidation_keeps_longer_validated_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            service = optimizer.TrajectoryOptimizationService(Path(root), validator=lambda _c: None)
            row = source(tuple(range(10)))
            candidate = v820._candidate(optimizer, row, "DELETE_SEGMENT", tuple(range(8)), 8, 2)

            def validate(c):
                if len(c.actions) == 8:
                    return result(True, (3,), attempts=1, successes=1)
                return result(False, attempts=1, successes=0)

            accepted, validated = v820._process_candidate(service, FakeValidator(validate), candidate)
            self.assertTrue(accepted)
            self.assertEqual(validated.cost, 8)
            self.assertEqual(validated.edit_kind, "DELETE_SEGMENT")

    def test_generic_validation_without_prefix_lengths_is_not_silently_truncated(self) -> None:
        raw = SimpleNamespace(
            success=True,
            actions_executed=3,
            reason="ok",
            attempts=1,
            successes=1,
        )
        converted = v820._validation_result_from_legacy(raw)
        self.assertEqual(converted.successful_prefix_lengths, ())


class DirectActionTests(unittest.TestCase):
    def test_singleton_success_immediately_establishes_cost_one_frontier(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            callbacks = []
            service = optimizer.TrajectoryOptimizationService(
                Path(root),
                validator=lambda _c: None,
                on_validation=lambda candidate, _result, validated: callbacks.append(
                    (candidate.edit_kind, candidate.cost, validated is not None)
                ),
            )
            row = source((4, 4, 3, 2, 1, 4, 3, 2))
            marker = v820._candidate(optimizer, row, v820._TARGET_MINIMIZE, row.actions)

            validator = FakeValidator(
                lambda c: result(
                    tuple(c.actions) == (2,),
                    (1, 1) if tuple(c.actions) == (2,) else (),
                    successes=2 if tuple(c.actions) == (2,) else 0,
                )
            )
            with patch.object(v820, "_available_actions_at_anchor", return_value=(1, 2, 3, 4)):
                v820._process_target_minimize(service, validator, marker)

            rows = tuple(service._validated.values())
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].actions, (2,))
            self.assertEqual(rows[0].cost, 1)
            self.assertEqual(
                [call for call in validator.calls if call[0] == v820._DIRECT_ACTION],
                [
                    (v820._DIRECT_ACTION, (1,)),
                    (v820._DIRECT_ACTION, (2,)),
                ],
            )
            self.assertTrue(any(item[:2] == (v820._DIRECT_ACTION, 1) for item in callbacks))

    def test_available_actions_are_read_after_exact_optimized_anchor_prefix(self) -> None:
        structural_click = (1 << 30) | (123 << 8) | 6

        class FakeEnv:
            def __init__(self):
                self.position = 0
                self.last_outcome_state = "NOT_FINISHED"

            def available_actions(self):
                if self.position == 0:
                    return [7]
                if self.position == 1:
                    return [8]
                return [1, structural_click]

            def step(self, action):
                expected = (7, 8)[self.position]
                if int(action) != expected:
                    raise AssertionError((action, expected))
                self.position += 1

        row = source((9, 9), prefix=(99, 99, 99), level=2)
        candidate = v820._candidate(optimizer, row, v820._TARGET_MINIMIZE, row.actions)
        service = SimpleNamespace(_v818_prefix_for=lambda _candidate: (7, 8))

        class FakeAnchorValidator:
            def __init__(self):
                self.service = service

            def _environment(self, _seed, _root):
                return FakeEnv()

            @staticmethod
            def _target_reached(_env, _target):
                return False

        actions = v820._available_actions_at_anchor(FakeAnchorValidator(), candidate)
        self.assertEqual(actions, (1, structural_click))


class SearchOrderingTests(unittest.TestCase):
    def test_ddmin_precedes_local_segment_and_single_action_cleanup(self) -> None:
        row = source(tuple(range(1, 101)))
        candidates = v820._generate_v820(
            row,
            optimizer.TrajectoryOptimizerConfig(max_candidates_per_round=24),
        )
        kinds = [candidate.edit_kind for candidate in candidates]
        self.assertTrue(kinds)
        self.assertEqual(kinds[0], v820._DELTA_DELETE)
        self.assertTrue(any(candidate.removed_length >= 50 for candidate in candidates))
        self.assertIn("DELETE_SEGMENT", kinds)
        if "DELETE_ACTION" in kinds:
            self.assertLess(kinds.index("DELETE_SEGMENT"), kinds.index("DELETE_ACTION"))
        self.assertLess(kinds.index(v820._DELTA_DELETE), kinds.index("DELETE_SEGMENT"))

    def test_target_marker_is_production_only_and_custom_validators_keep_v818_generator(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            service = optimizer.TrajectoryOptimizationService(Path(root), validator=lambda _c: None)
            self.assertFalse(v820._is_arc_validator(service))
            row = source((1, 2, 1, 2, 1, 2))
            legacy = v820._BASE_GENERATE_V818(row, service.config)
            self.assertTrue(any(candidate.edit_kind == "REDUCE_REPEAT" for candidate in legacy))


class ValidationResultTests(unittest.TestCase):
    def test_real_validator_reports_each_successful_seed_stopping_point(self) -> None:
        from v8 import trajectory_optimizer_v818 as v818
        from v8.trajectory_validation_v814 import validate_arc_candidate

        with tempfile.TemporaryDirectory() as root:
            service = optimizer.TrajectoryOptimizationService(Path(root), validator=validate_arc_candidate)
            row = source((1, 2, 3, 4, 5))
            candidate = v820._candidate(optimizer, row, "DELETE_SEGMENT", row.actions[:-1], 4, 1)
            validator = object.__new__(v818._GameReplayValidator)
            validator.service = service
            validator.game_id = "world"
            validator._envs = {}
            trials = iter(
                [
                    (True, 3, "target_preserved", 10, 1, 20, 0),
                    (True, 4, "target_preserved", 11, 1, 21, 0),
                ]
            )
            with patch.object(validator, "_trial", side_effect=lambda *_args, **_kwargs: next(trials)):
                output = v820._game_validate_v820(validator, candidate)
            self.assertTrue(output.success)
            self.assertEqual(output.successful_prefix_lengths, (3, 4))
            self.assertEqual(output.actions_executed, 7)


if __name__ == "__main__":
    unittest.main()
