from __future__ import annotations

import unittest

import v8  # noqa: F401 - installs v8.8 fixups
from v8 import learning_fixes_v088 as learning


class TransferCutRefreshTests(unittest.TestCase):
    def test_transfer_cut_refreshes_before_candidate_discovery(self) -> None:
        node_arena = object()
        edge_arena = object()
        stale_node = object()
        fresh_node = object()
        fresh_edge = object()

        class View:
            def __init__(self) -> None:
                self._nodes = (node_arena,)
                self._edges = (edge_arena,)
                self._strategy_version = (0, 0)
                self._record_cache = {
                    id(node_arena): ((stale_node,), 0),
                    id(edge_arena): ((), 0),
                }
                self._node_by_uid = {"stale": stale_node}
                self.invalidations = 0
                self.refreshes = 0

            def invalidate_strategy_cache(self) -> None:
                self.invalidations += 1

            def _refresh_strategy_cache(self) -> None:
                self.refreshes += 1
                self._strategy_version = (2, 4)
                self._record_cache = {
                    id(node_arena): ((fresh_node,), 2),
                    id(edge_arena): ((fresh_edge,), 4),
                }
                self._node_by_uid = {"fresh": fresh_node}

        view = View()
        cut = learning._coherent_cached_transfer_cut(view)

        self.assertEqual(view.invalidations, 1)
        self.assertEqual(view.refreshes, 1)
        self.assertEqual(cut, ((fresh_node,), (fresh_edge,)))


if __name__ == "__main__":
    unittest.main()
