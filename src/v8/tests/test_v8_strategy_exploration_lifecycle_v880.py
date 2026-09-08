from __future__ import annotations

import unittest
from types import SimpleNamespace

from v8.model import CognitiveState, MemoryLevel, MemoryType, MemoryUid, ValidationState
from v8.strategy_exploration_lifecycle_v880 import _advance_probationary_m7


class StrategyExplorationLifecycleV880Tests(unittest.TestCase):
    def test_probationary_m7_gets_bounded_lifecycle_attention(self) -> None:
        rows = tuple(
            SimpleNamespace(
                uid=MemoryUid(7, index + 1),
                level=int(MemoryLevel.M7),
                memory_type=int(MemoryType.STRATEGY),
                cognitive_state=int(CognitiveState.PROBATION),
                validation_state=int(ValidationState.STRUCTURAL),
                updated_watermark=10 + index,
            )
            for index in range(20)
        )

        class ReadView:
            def node_records(self, *, level=None):
                self.requested_level = level
                return rows

        class Lifecycle:
            def __init__(self):
                self.seen = []

            def decide(self, row):
                self.seen.append(row.uid)
                return SimpleNamespace(
                    cognitive_state=int(CognitiveState.ACTIVE),
                    validation_state=int(ValidationState.STRUCTURAL),
                )

        class Supervisor:
            candidate_budget = 16

            def __init__(self):
                self.read_view = ReadView()
                self.lifecycle = Lifecycle()
                self.submitted = []

            def _fresh(self, *_args):
                return True

            def _existing_proposal(self, row, **kwargs):
                return (row.uid, kwargs)

            def _submit(self, proposal):
                self.submitted.append(proposal)

        supervisor = Supervisor()
        advanced = _advance_probationary_m7(supervisor)

        self.assertEqual(advanced, 4)
        self.assertEqual(len(supervisor.lifecycle.seen), 4)
        self.assertEqual(len(supervisor.submitted), 4)
        self.assertEqual(supervisor.read_view.requested_level, MemoryLevel.M7)

    def test_efficiency_gate_is_not_emitted_or_relaxed_here(self) -> None:
        row = SimpleNamespace(
            uid=MemoryUid(7, 1),
            level=int(MemoryLevel.M7),
            memory_type=int(MemoryType.STRATEGY),
            cognitive_state=int(CognitiveState.PROBATION),
            validation_state=int(ValidationState.STRUCTURAL),
            updated_watermark=1,
        )

        class ReadView:
            def node_records(self, *, level=None):
                return (row,)

        class Supervisor:
            candidate_budget = 8
            read_view = ReadView()
            lifecycle = SimpleNamespace(decide=lambda _row: None)

            def _fresh(self, *_args):
                raise AssertionError("no evidence/freshness call when lifecycle does not promote")

        self.assertEqual(_advance_probationary_m7(Supervisor()), 0)


if __name__ == "__main__":
    unittest.main()
