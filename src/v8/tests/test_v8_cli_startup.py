from __future__ import annotations

import unittest
from pathlib import Path

from v8.cli import _graph_load_line


class StartupGraphLineTests(unittest.TestCase):
    def test_snapshot_source_and_node_count_are_shown(self) -> None:
        line = _graph_load_line(
            snapshot_path=Path("runs/v8/continuous/snapshots/snapshot-00000000000000000042"),
            restore_enabled=True,
            nodes=24903,
        )
        self.assertEqual(
            line,
            "graph source=runs/v8/continuous/snapshots/snapshot-00000000000000000042 nodes=24903",
        )
        self.assertNotIn("\n", line)

    def test_empty_graph_without_snapshot_is_explicit(self) -> None:
        self.assertEqual(
            _graph_load_line(snapshot_path=None, restore_enabled=True, nodes=0),
            "graph source=empty(no-snapshot) nodes=0",
        )

    def test_no_restore_is_explicit(self) -> None:
        self.assertEqual(
            _graph_load_line(snapshot_path=None, restore_enabled=False, nodes=0),
            "graph source=empty(--no-restore) nodes=0",
        )


if __name__ == "__main__":
    unittest.main()
