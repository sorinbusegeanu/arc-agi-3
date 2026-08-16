from __future__ import annotations

import unittest
from types import SimpleNamespace

from v8.action_targeting_v810_fixups import _cached_legacy_exact_click_actions
from v8.learning_blockers_v055 import pack_action_choice
from v8.model import MemoryLevel


class _CountingNodes(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scans = 0

    def values(self):
        self.scans += 1
        return super().values()


class LegacyClickIndexTests(unittest.TestCase):
    def test_legacy_click_nodes_are_scanned_once_per_graph_version(self):
        context = 1234
        click_a = pack_action_choice(6, 5, 7)
        click_b = pack_action_choice(6, 8, 9)
        nodes = _CountingNodes(
            {
                1: SimpleNamespace(
                    level=int(MemoryLevel.M1),
                    key_parts=(context, click_a),
                    support_count=4,
                    expected_primary_valence=0.8,
                    primary_valence_confidence=0.9,
                ),
                2: SimpleNamespace(
                    level=int(MemoryLevel.M1),
                    key_parts=(context, 1),
                    support_count=20,
                    expected_primary_valence=1.0,
                    primary_valence_confidence=1.0,
                ),
            }
        )
        view = SimpleNamespace(
            _strategy_version=(2, 4),
            _node_by_uid=nodes,
            _refresh_strategy_cache=lambda: None,
        )

        self.assertEqual(_cached_legacy_exact_click_actions(view, context), (click_a,))
        self.assertEqual(_cached_legacy_exact_click_actions(view, context), (click_a,))
        self.assertEqual(nodes.scans, 1)

        nodes[3] = SimpleNamespace(
            level=int(MemoryLevel.M1),
            key_parts=(context, click_b),
            support_count=8,
            expected_primary_valence=0.9,
            primary_valence_confidence=0.9,
        )
        view._strategy_version = (4, 4)
        self.assertEqual(
            _cached_legacy_exact_click_actions(view, context),
            (click_b, click_a),
        )
        self.assertEqual(nodes.scans, 2)


if __name__ == "__main__":
    unittest.main()
