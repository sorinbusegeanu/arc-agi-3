from __future__ import annotations

import unittest
from types import SimpleNamespace

from v8 import actor_compact_scan_resilience_v867 as v867
from v8.actor_read_view_v851 import ActorReadView


class ActorCompactScanResilienceV867Tests(unittest.TestCase):
    def test_runtime_stack_installs_v867_as_public_actor_refresh(self):
        self.assertTrue(v867._INSTALLED)
        self.assertIs(ActorReadView._refresh_strategy_cache, v867._actor_refresh_strategy_cache_v867)

    def test_unstable_arena_does_not_terminate_actor_and_is_backed_off(self):
        original = v867._BASE_ACTOR_REFRESH
        calls = []
        dummy = SimpleNamespace(
            _strategy_cache_stale=True,
            _v867_refresh_retry_at=0.0,
        )

        def unstable(instance):
            calls.append(instance)
            raise RuntimeError("actor compact node scan could not obtain stable arena")

        try:
            v867._BASE_ACTOR_REFRESH = unstable
            v867._actor_refresh_strategy_cache_v867(dummy)
            self.assertTrue(dummy._strategy_cache_stale)
            self.assertGreater(dummy._v867_refresh_retry_at, 0.0)
            v867._actor_refresh_strategy_cache_v867(dummy)
            self.assertEqual(calls, [dummy])
        finally:
            v867._BASE_ACTOR_REFRESH = original

    def test_refresh_retries_after_backoff_and_recovers(self):
        original = v867._BASE_ACTOR_REFRESH
        calls = []
        dummy = SimpleNamespace(
            _strategy_cache_stale=True,
            _v867_refresh_retry_at=0.0,
        )

        def recovered(instance):
            calls.append(instance)
            instance._strategy_cache_stale = False
            return None

        try:
            v867._BASE_ACTOR_REFRESH = recovered
            v867._actor_refresh_strategy_cache_v867(dummy)
        finally:
            v867._BASE_ACTOR_REFRESH = original
        self.assertEqual(calls, [dummy])
        self.assertFalse(dummy._strategy_cache_stale)
        self.assertEqual(dummy._v867_refresh_retry_at, 0.0)

    def test_unrelated_runtime_error_is_not_hidden(self):
        original = v867._BASE_ACTOR_REFRESH
        dummy = SimpleNamespace(_v867_refresh_retry_at=0.0)
        try:
            v867._BASE_ACTOR_REFRESH = lambda instance: (_ for _ in ()).throw(
                RuntimeError("corrupt compact graph")
            )
            with self.assertRaisesRegex(RuntimeError, "corrupt compact graph"):
                v867._actor_refresh_strategy_cache_v867(dummy)
        finally:
            v867._BASE_ACTOR_REFRESH = original


if __name__ == "__main__":
    unittest.main()
