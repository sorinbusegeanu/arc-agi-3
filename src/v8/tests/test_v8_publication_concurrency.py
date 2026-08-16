from __future__ import annotations

import unittest

from v8.publication import LiveReadView


class _CrossMutatingArena:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._sequence = 0
        self.other: _CrossMutatingArena | None = None

    @property
    def sequence(self) -> int:
        return self._sequence

    def records(self):
        if self.other is not None:
            self.other._sequence += 2
        return ()


class _CountingArena:
    def __init__(self) -> None:
        self._sequence = 0
        self.record_reads = 0

    @property
    def sequence(self) -> int:
        return self._sequence

    def records(self):
        self.record_reads += 1
        return ()


class PublicationConcurrencyRegressionTests(unittest.TestCase):
    def test_negative_refresh_interval_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            LiveReadView((), refresh_interval_seconds=-0.1)

    def test_bounded_refresh_does_not_rescan_unchanged_memory_per_action(self) -> None:
        nodes = _CountingArena()
        view = LiveReadView((), refresh_interval_seconds=60.0)
        view._nodes = (nodes,)

        self.assertEqual(view.outcome_distribution(10, 2), {})
        self.assertEqual(nodes.record_reads, 1)

        nodes._sequence = 2
        self.assertEqual(view.outcome_distribution(10, 2), {})
        self.assertEqual(nodes.record_reads, 1)

        view._next_strategy_refresh = 0.0
        self.assertEqual(view.outcome_distribution(10, 2), {})
        self.assertEqual(nodes.record_reads, 2)

    def test_explicit_refresh_mode_rechecks_memory_only_after_invalidation(self) -> None:
        nodes = _CountingArena()
        view = LiveReadView((), refresh_interval_seconds=None)
        view._nodes = (nodes,)

        self.assertEqual(view.outcome_distribution(10, 2), {})
        self.assertEqual(nodes.record_reads, 1)

        nodes._sequence = 2
        self.assertEqual(view.outcome_distribution(10, 2), {})
        self.assertEqual(nodes.record_reads, 1)

        view.invalidate_strategy_cache()
        self.assertEqual(view.outcome_distribution(10, 2), {})
        self.assertEqual(nodes.record_reads, 2)

    def test_strategy_cache_does_not_require_global_shard_quiescence(self) -> None:
        nodes = _CrossMutatingArena("nodes")
        edges = _CrossMutatingArena("edges")
        nodes.other = edges
        edges.other = nodes

        view = LiveReadView(())
        view._nodes = (nodes,)
        view._edges = (edges,)

        self.assertEqual(view.plan_candidates(123, (1, 2)), ())
        self.assertEqual(view.plan_candidates(123, (1, 2)), ())
        self.assertTrue(view._strategy_version)

    def test_edge_read_uses_last_coherent_snapshot_while_writer_is_active(self) -> None:
        edges = _CrossMutatingArena("edges")
        view = LiveReadView(())
        view._edges = (edges,)

        self.assertEqual(view.edge_records(), ())
        edges._sequence = 1
        self.assertEqual(view.edge_records(), ())


if __name__ == "__main__":
    unittest.main()
