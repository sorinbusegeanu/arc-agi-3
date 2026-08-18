from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from v8 import read_view_cache_v839 as cache


class ReadViewCacheV839Tests(unittest.TestCase):
    def test_unchanged_stable_arena_reuses_coherent_cut(self) -> None:
        arena = SimpleNamespace(sequence=4)
        cached = (("row",), 4)
        view = SimpleNamespace(_record_cache={id(arena): cached})
        base = Mock(side_effect=AssertionError("cache miss"))

        with patch.object(cache, "_BASE_STABLE_RECORDS_WITH_VERSION", base):
            result = cache._stable_records_with_version_v839(view, arena)

        self.assertIs(result, cached)
        base.assert_not_called()

    def test_inflight_writer_reuses_last_coherent_cut(self) -> None:
        arena = SimpleNamespace(sequence=5)
        cached = (("row",), 4)
        view = SimpleNamespace(_record_cache={id(arena): cached})
        base = Mock(side_effect=AssertionError("cache miss"))

        with patch.object(cache, "_BASE_STABLE_RECORDS_WITH_VERSION", base):
            result = cache._stable_records_with_version_v839(view, arena)

        self.assertIs(result, cached)
        base.assert_not_called()

    def test_changed_stable_sequence_refreshes_cut(self) -> None:
        arena = SimpleNamespace(sequence=6)
        cached = (("old",), 4)
        refreshed = (("new",), 6)
        view = SimpleNamespace(_record_cache={id(arena): cached})
        base = Mock(return_value=refreshed)

        with patch.object(cache, "_BASE_STABLE_RECORDS_WITH_VERSION", base):
            result = cache._stable_records_with_version_v839(view, arena)

        self.assertIs(result, refreshed)
        base.assert_called_once_with(view, arena, timeout=1.0)


if __name__ == "__main__":
    unittest.main()
