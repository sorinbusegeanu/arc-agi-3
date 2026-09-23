from __future__ import annotations

from typing import Any


def install_actor_policy_cache(runtime_cls: type) -> None:
    """Avoid rebuilding the same immutable actor policy for every child launch."""
    if getattr(runtime_cls, "_actor_policy_cache_installed", False):
        return

    original = runtime_cls.actor_policy_snapshot

    def actor_policy_snapshot(self: Any):
        # Canonical commit is atomic from the actors' perspective. While the
        # commit thread owns the authoritative runtime cut, keep serving the last
        # completed immutable policy snapshot so the coordinator remains free to
        # drain publication/IPC queues instead of waiting on the runtime lock.
        if bool(getattr(self, "_canonical_commit_inflight", False)):
            previous = getattr(self, "_actor_policy_snapshot_last", None)
            if previous is not None:
                self._actor_policy_snapshot_inflight_hits = int(
                    getattr(self, "_actor_policy_snapshot_inflight_hits", 0)
                ) + 1
                self.set_telemetry_gauge(
                    "actor_policy_snapshot_inflight_hits",
                    self._actor_policy_snapshot_inflight_hits,
                )
                return previous

        graph_generation = int(self.graph.generation)
        policy_generation = int(getattr(self, "_actor_policy_generation", 0))
        model_version = str(getattr(self.unified_telemetry, "model_version", ""))
        key = (graph_generation, policy_generation, model_version)
        cached = getattr(self, "_actor_policy_snapshot_cache", None)
        if cached is not None and cached[0] == key:
            self.set_telemetry_gauge(
                "actor_policy_snapshot_cache_hits",
                int(getattr(self, "_actor_policy_snapshot_cache_hits", 0)) + 1,
            )
            self._actor_policy_snapshot_cache_hits = int(getattr(self, "_actor_policy_snapshot_cache_hits", 0)) + 1
            self._actor_policy_snapshot_last = cached[1]
            return cached[1]
        snapshot = original(self)
        self._actor_policy_snapshot_cache = (key, snapshot)
        self._actor_policy_snapshot_last = snapshot
        self._actor_policy_snapshot_cache_misses = int(getattr(self, "_actor_policy_snapshot_cache_misses", 0)) + 1
        self.set_telemetry_gauge("actor_policy_snapshot_cache_misses", self._actor_policy_snapshot_cache_misses)
        return snapshot

    runtime_cls.actor_policy_snapshot = actor_policy_snapshot
    runtime_cls._actor_policy_cache_installed = True
