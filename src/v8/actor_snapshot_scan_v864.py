from __future__ import annotations

"""v8.64: scan actor compact memory from coherent byte cuts.

A restored non-empty graph can be mutated continuously by live writers. v8.51 decoded
records directly from the shared arena and then required the arena sequence to remain
unchanged for the whole Python scan. Large restored arenas made that stability window
long enough to fail repeatedly. This layer takes the coherent byte copy first and lets
the existing compact filters decode that immutable cut outside the live seqlock.
"""

from v8.arena import _HEADER


_INSTALLED = False
_BASE_SCAN_NODE_ARENA = None
_BASE_SCAN_EDGE_ARENA = None
_BASE_LOAD_NEEDED_LOW = None
_BASE_LOAD_PROVENANCE_GAMES = None


class _FrozenArenaCut:
    __slots__ = ("_arena", "_payload", "count", "sequence")

    def __init__(self, arena, *, timeout: float = 2.0) -> None:
        self._arena = arena
        self._payload = arena.snapshot_bytes(retries=1, timeout=timeout)
        count, sequence = _HEADER.unpack_from(self._payload, 0)
        self.count = int(count)
        self.sequence = int(sequence)

    def read(self, index: int):
        return self._arena._read_from_buffer(self._payload, int(index))


def _freeze(arena):
    # Keep lightweight test/dummy arenas compatible. Production shared arenas expose
    # both methods and therefore always use the coherent byte-cut path.
    if callable(getattr(arena, "snapshot_bytes", None)) and callable(
        getattr(arena, "_read_from_buffer", None)
    ):
        return _FrozenArenaCut(arena)
    return arena


def _scan_node_arena_v864(arena):
    return _BASE_SCAN_NODE_ARENA(_freeze(arena))


def _scan_edge_arena_v864(arena, *, lineage_uids, active_uids, node_by_uid):
    return _BASE_SCAN_EDGE_ARENA(
        _freeze(arena),
        lineage_uids=lineage_uids,
        active_uids=active_uids,
        node_by_uid=node_by_uid,
    )


def _load_needed_low_v864(nodes, needed):
    return _BASE_LOAD_NEEDED_LOW(tuple(_freeze(arena) for arena in nodes), needed)


def _load_provenance_games_v864(edges, relevant):
    return _BASE_LOAD_PROVENANCE_GAMES(tuple(_freeze(arena) for arena in edges), relevant)


def install_actor_snapshot_scan_v864() -> None:
    global _INSTALLED, _BASE_SCAN_NODE_ARENA, _BASE_SCAN_EDGE_ARENA
    global _BASE_LOAD_NEEDED_LOW, _BASE_LOAD_PROVENANCE_GAMES
    if _INSTALLED:
        return

    from v8.actor_read_view_v851 import ActorReadView

    _BASE_SCAN_NODE_ARENA = ActorReadView._scan_node_arena
    _BASE_SCAN_EDGE_ARENA = ActorReadView._scan_edge_arena
    _BASE_LOAD_NEEDED_LOW = ActorReadView._load_needed_low
    _BASE_LOAD_PROVENANCE_GAMES = ActorReadView._load_provenance_games

    ActorReadView._scan_node_arena = staticmethod(_scan_node_arena_v864)
    ActorReadView._scan_edge_arena = staticmethod(_scan_edge_arena_v864)
    ActorReadView._load_needed_low = staticmethod(_load_needed_low_v864)
    ActorReadView._load_provenance_games = staticmethod(_load_provenance_games_v864)
    _INSTALLED = True
