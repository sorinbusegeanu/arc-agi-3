from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import v8
from v8.model import MemoryUid
from v8.trajectory_optimizer_v814 import (
    ReplayAnchor,
    SuccessfulTrajectory,
    TrajectoryCandidate,
    TrajectoryOptimizationService,
    TrajectoryTarget,
)


def _source(actions, *, trajectory_id="source") -> SuccessfulTrajectory:
    return SuccessfulTrajectory(
        trajectory_id,
        ReplayAnchor("world", 1, (), None),
        TrajectoryTarget(1, "LEVEL"),
        tuple(actions),
        MemoryUid.zero(),
        MemoryUid.zero(),
        0,
    )


def _candidate(source, candidate_id, actions) -> TrajectoryCandidate:
    return TrajectoryCandidate(
        candidate_id,
        source,
        "DELETE_ACTION",
        tuple(actions),
        0,
        max(1, source.cost - len(tuple(actions))),
    )


class TrajectoryOptimizerStdoutV816Tests(unittest.TestCase):
    def test_first_frontier_improvement_reports_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            service = TrajectoryOptimizationService(Path(root), validator=lambda _candidate: None)
            service._validations = 1
            source = _source((1, 2, 3, 4))
            candidate = _candidate(source, "candidate-1", (1, 2, 4))
            output = io.StringIO()
            with redirect_stdout(output):
                service._accept(candidate)
            line = output.getvalue()
            self.assertIn("trajectory optimization complete", line)
            self.assertIn("rounds=1", line)
            self.assertIn("validations=1", line)
            self.assertIn("cost=4->3", line)
            self.assertIn("saved=1", line)
            self.assertIn("best=4->3", line)

    def test_successes_are_aggregated_and_throttled_for_five_minutes(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            service = TrajectoryOptimizationService(Path(root), validator=lambda _candidate: None)

            first = _candidate(_source((1, 2, 3, 4), trajectory_id="source-1"), "candidate-1", (1, 2, 4))
            service._validations = 1
            with redirect_stdout(io.StringIO()):
                service._accept(first)

            second_source = _source((1, 2, 4), trajectory_id="source-2")
            second = _candidate(second_source, "candidate-2", (1, 4))
            service._validations = 5
            output = io.StringIO()
            with redirect_stdout(output):
                service._accept(second)
            self.assertEqual(output.getvalue(), "")

            service._v816_last_report_monotonic -= 301.0
            with redirect_stdout(output):
                emitted = service._v816_emit_success_report_if_due()
            self.assertTrue(emitted)
            line = output.getvalue()
            self.assertIn("rounds=1", line)
            self.assertIn("validations=4", line)
            self.assertIn("cost=3->2", line)
            self.assertIn("saved=1", line)
            self.assertIn("best=3->2", line)

    def test_non_frontier_success_does_not_report_as_completed_round(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            service = TrajectoryOptimizationService(Path(root), validator=lambda _candidate: None)
            best = _candidate(_source((1, 2, 3, 4), trajectory_id="source-1"), "best", (1, 4))
            with redirect_stdout(io.StringIO()):
                service._accept(best)

            service._v816_last_report_monotonic -= 301.0
            worse = _candidate(_source((1, 2, 3, 4), trajectory_id="source-2"), "worse", (1, 2, 4))
            output = io.StringIO()
            with redirect_stdout(output):
                returned = service._accept(worse)
                emitted = service._v816_emit_success_report_if_due()
            self.assertEqual(returned.variant_id, "best")
            self.assertFalse(emitted)
            self.assertEqual(output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
