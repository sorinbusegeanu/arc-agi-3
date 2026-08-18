from __future__ import annotations

"""v8.39 coherent read-index reuse for graph-heavy parent maintenance."""

import threading

from v8.model import MemoryUid, RelationType


_INSTALLED = False
_BASE_STABLE_RECORDS_WITH_VERSION = None
_BASE_NODE_RECORDS = None
_INDEX_LOCK = threading.RLock()


def _stable_records_with_version_v839(self, arena, *, timeout: float = 1.0):
    cached = self._record_cache.get(id(arena))
    if cached is not None:
        sequence = int(arena.sequence)
        # Cached records are a coherent bounded-staleness cut. Reuse them while
        # a newer write is in flight or when no write occurred since the cut.
        if sequence & 1 or sequence == int(cached[1]):
            return cached
    return _BASE_STABLE_RECORDS_WITH_VERSION(self, arena, timeout=timeout)


def _cached_node_versions(self) -> tuple[int, ...] | None:
    values = []
    for arena in self._nodes:
        cached = self._record_cache.get(id(arena))
        if cached is None:
            return None
        values.append(int(cached[1]))
    return tuple(values)


def _node_records_v839(self, *, level=None):
    with _INDEX_LOCK:
        wanted = None if level is None else int(level)
        versions = _cached_node_versions(self)
        cached_version = getattr(self, "_v839_node_query_version", None)
        cache = getattr(self, "_v839_node_query_cache", None)
        if versions is not None and versions == cached_version and isinstance(cache, dict):
            current = tuple(int(arena.sequence) for arena in self._nodes)
            if all(
                (sequence & 1) or sequence == version
                for sequence, version in zip(current, versions, strict=True)
            ):
                hit = cache.get(wanted)
                if hit is not None:
                    return hit

        rows = tuple(_BASE_NODE_RECORDS(self, level=level))
        stable_versions = _cached_node_versions(self)
        if stable_versions is not None:
            if stable_versions != getattr(self, "_v839_node_query_version", None):
                self._v839_node_query_version = stable_versions
                self._v839_node_query_cache = {}
            self._v839_node_query_cache[wanted] = rows
        return rows


def _refresh_provenance_index(self) -> None:
    from v8 import publication as publication_module

    cuts = tuple(self._stable_records_with_version(arena) for arena in self._edges)
    versions = tuple(int(version) for _rows, version in cuts)
    if getattr(self, "_v839_provenance_version", None) == versions:
        return

    direct: dict[MemoryUid, set[int]] = {}
    parents: dict[MemoryUid, set[MemoryUid]] = {}
    lineage = set(int(value) for value in publication_module._LINEAGE_RELATIONS)
    for rows, _version in cuts:
        for edge in rows:
            relation = int(edge.relation_type)
            if (
                relation == int(RelationType.GAME_PROVENANCE)
                and int(edge.target_uid.hi) == 0
            ):
                direct.setdefault(edge.source_uid, set()).add(int(edge.target_uid.lo))
            elif relation in lineage:
                parents.setdefault(edge.source_uid, set()).add(edge.target_uid)

    self._v839_provenance_version = versions
    self._v839_direct_games = direct
    self._v839_provenance_parents = parents
    self._v839_source_games_cache = {}


def _source_games_v839(
    self,
    uid: MemoryUid,
    *,
    max_depth: int = 8,
) -> frozenset[int]:
    with _INDEX_LOCK:
        _refresh_provenance_index(self)
        depth = max(0, int(max_depth))
        cache = self._v839_source_games_cache
        key = (uid, depth)
        cached = cache.get(key)
        if cached is not None:
            return cached

        direct = self._v839_direct_games
        parents = self._v839_provenance_parents
        games = set(direct.get(uid, ()))
        frontier = {uid}
        visited = {uid}
        for _ in range(depth):
            following: set[MemoryUid] = set()
            for current in frontier:
                for parent in parents.get(current, ()):
                    games.update(direct.get(parent, ()))
                    if parent not in visited:
                        visited.add(parent)
                        following.add(parent)
            if not following:
                break
            frontier = following

        result = frozenset(games)
        cache[key] = result
        return result


def install_read_view_cache_v839() -> None:
    global _INSTALLED, _BASE_STABLE_RECORDS_WITH_VERSION, _BASE_NODE_RECORDS
    if _INSTALLED:
        return

    from v8.publication import LiveReadView

    _BASE_STABLE_RECORDS_WITH_VERSION = LiveReadView._stable_records_with_version
    _BASE_NODE_RECORDS = LiveReadView.node_records
    LiveReadView._stable_records_with_version = _stable_records_with_version_v839
    LiveReadView.node_records = _node_records_v839
    LiveReadView.source_games = _source_games_v839
    _INSTALLED = True
