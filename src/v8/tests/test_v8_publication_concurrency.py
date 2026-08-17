from __future__ import annotations

import unittest

from v8.arena import EdgeRecord, SharedEdgeArena
from v8.model import MemoryUid, RelationType
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


class _SnapshotCapableArena:
    kind = "edges"

    def __init__(self) -> None:
        self.sequence = 0
        self.snapshot_reads = 0

    def snapshot_records(self, *, timeout: float):
        self.snapshot_reads += 1
        return (("coherent",), 2)

    def records(self):
        raise AssertionError("live records must not be decoded under the seqlock")


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

    def test_snapshot_capable_arena_decodes_after_coherent_copy(self) -> None:
        edges = _SnapshotCapableArena()
        view = LiveReadView(())
        view._edges = (edges,)

        self.assertEqual(view.edge_records(), ("coherent",))
        self.assertEqual(edges.snapshot_reads, 1)

    def test_shared_edge_snapshot_records_are_detached_from_later_writes(self) -> None:
        arena = SharedEdgeArena(capacity=1)
        try:
            first = EdgeRecord(
                MemoryUid(1, 2),
                int(RelationType.EXPLAINS),
                MemoryUid(3, 4),
                1,
                5,
            )
            second = EdgeRecord(
                MemoryUid(6, 7),
                int(RelationType.EXPLAINS),
                MemoryUid(8, 9),
                2,
                10,
            )
            arena.begin_write()
            arena.write(0, first)
            arena.end_write(count=1)

            rows, sequence = arena.snapshot_records()

            arena.begin_write()
            arena.write(0, second)
            arena.end_write(count=1)
            self.assertEqual(rows, (first,))
            self.assertEqual(sequence, 2)
            self.assertEqual(arena.read(0), second)
        finally:
            arena.dispose()


if __name__ == "__main__":
    unittest.main()
