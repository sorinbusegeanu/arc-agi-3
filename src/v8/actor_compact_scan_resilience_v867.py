from __future__ import annotations

"""v8.67 keep actors alive when a compact graph scan races active writers."""

import time


_INSTALLED = False
_BASE_ACTOR_REFRESH = None
_RETRY_SECONDS = 0.25
_STABLE_ARENA_ERROR = "could not obtain stable arena"


def _actor_refresh_strategy_cache_v867(self) -> None:
    """Retry an unavailable coherent cut later instead of terminating the actor.

    v8.52 intentionally keeps a coherent compact graph cut for the actor lifetime.
    A restored/late-starting actor can still reach the initial compact scan while
    canonical writers are already active.  Large arenas may then lose every seqlock
    attempt.  That is temporary publication contention, not actor failure.
    """
    now = time.monotonic()
    if now < float(getattr(self, "_v867_refresh_retry_at", 0.0)):
        return
    try:
        result = _BASE_ACTOR_REFRESH(self)
    except RuntimeError as exc:
        if _STABLE_ARENA_ERROR not in str(exc):
            raise
        self._v867_refresh_retry_at = now + _RETRY_SECONDS
        self._strategy_cache_stale = True
        return None
    self._v867_refresh_retry_at = 0.0
    return result


def install_actor_compact_scan_resilience_v867() -> None:
    global _INSTALLED, _BASE_ACTOR_REFRESH
    if _INSTALLED:
        return

    from v8.actor_read_view_v851 import ActorReadView

    _BASE_ACTOR_REFRESH = ActorReadView._refresh_strategy_cache
    ActorReadView._refresh_strategy_cache = _actor_refresh_strategy_cache_v867
    _INSTALLED = True
