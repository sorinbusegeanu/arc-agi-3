from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
from collections import OrderedDict
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from threading import RLock
from types import MappingProxyType
from typing import Any, Callable, Hashable, Iterable, Iterator, Mapping
from pathlib import Path

from .canonical_transaction import CanonicalCollection, TransactionOverlay, is_tombstone


MAX_CHUNK_ENTRIES = 8192
MAX_CHUNK_BYTES = 64 * 1024 * 1024


class FrozenMapping(Mapping[Hashable, object]):
    def __init__(self, rows: Iterable[tuple[Hashable, object]]) -> None:
        self._items = tuple(rows)

    def __getitem__(self, key: Hashable) -> object:
        for existing, value in self._items:
            if existing == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[Hashable]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return FrozenMapping(
            (key, _freeze(item)) for key, item in sorted(value.items(), key=lambda row: repr(row[0]))
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


def _canonical(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _canonical(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda row: repr(row[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_canonical(item) for item in value), key=repr)
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "hex") and callable(getattr(value, "hex")):
        return {"type": type(value).__name__, "hex": value.hex()}
    return {"type": type(value).__name__, "repr": repr(value)}


def canonical_value(value: object) -> object:
    """Return the deterministic JSON-safe representation used by chunks/WAL."""
    return _canonical(value)


def _encoded(value: object) -> bytes:
    return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


@dataclass(frozen=True, order=True, slots=True)
class ChunkId:
    value: str

    def __post_init__(self) -> None:
        if len(self.value) != 64 or any(character not in "0123456789abcdef" for character in self.value):
            raise ValueError("ChunkId must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class CanonicalChunk:
    chunk_id: ChunkId
    collection: CanonicalCollection
    schema_version: int
    entries: tuple[tuple[Hashable, object], ...]
    encoded_bytes: int
    generation_created: int
    checksum: str

    def __post_init__(self) -> None:
        if len(self.entries) > MAX_CHUNK_ENTRIES:
            raise ValueError("canonical chunk entry ceiling exceeded")
        if self.encoded_bytes > MAX_CHUNK_BYTES:
            raise ValueError("canonical chunk byte ceiling exceeded")
        if self.chunk_id.value != self.checksum:
            raise ValueError("chunk identity and checksum must match")

    @property
    def mapping(self) -> Mapping[Hashable, object]:
        return MappingProxyType(dict(self.entries))


class GraphChunk(CanonicalChunk):
    pass


class PayloadChunk(CanonicalChunk):
    pass


class LevelIndexChunk(CanonicalChunk):
    pass


class SignatureIndexChunk(CanonicalChunk):
    pass


class GroundingIndexChunk(CanonicalChunk):
    pass


class ProvenanceChunk(CanonicalChunk):
    pass


class LifecycleChunk(CanonicalChunk):
    pass


_CHUNK_TYPES = {
    CanonicalCollection.GRAPH: GraphChunk,
    CanonicalCollection.PAYLOAD: PayloadChunk,
    CanonicalCollection.LEVEL_INDEX: LevelIndexChunk,
    CanonicalCollection.SIGNATURE_INDEX: SignatureIndexChunk,
    CanonicalCollection.GROUNDING_INDEX: GroundingIndexChunk,
    CanonicalCollection.PROVENANCE_INDEX: ProvenanceChunk,
    CanonicalCollection.LIFECYCLE_INDEX: LifecycleChunk,
    CanonicalCollection.POLICY_SOURCE_INDEX: CanonicalChunk,
}


@dataclass(frozen=True, slots=True)
class CanonicalStateHandle:
    canonical_applied_lsn: int
    graph_root: tuple[ChunkId, ...] = ()
    payload_root: tuple[ChunkId, ...] = ()
    level_index_root: tuple[ChunkId, ...] = ()
    signature_index_root: tuple[ChunkId, ...] = ()
    grounding_index_root: tuple[ChunkId, ...] = ()
    provenance_index_root: tuple[ChunkId, ...] = ()
    lifecycle_index_root: tuple[ChunkId, ...] = ()
    policy_source_index_root: tuple[ChunkId, ...] = ()
    schema_versions: tuple[tuple[str, int], ...] = ()
    generation: int = 0
    checksum: str = ""

    def roots(self, collection: CanonicalCollection | str) -> tuple[ChunkId, ...]:
        selected = collection if isinstance(collection, CanonicalCollection) else CanonicalCollection(str(collection))
        return getattr(self, f"{selected.value}_root")

    @property
    def all_chunk_ids(self) -> tuple[ChunkId, ...]:
        return tuple(chunk_id for collection in CanonicalCollection for chunk_id in self.roots(collection))

    def __post_init__(self) -> None:
        if min(self.canonical_applied_lsn, self.generation) < 0:
            raise ValueError("canonical handle frontiers must be non-negative")
        expected = _handle_checksum(self)
        if self.checksum and self.checksum != expected:
            raise ValueError("CanonicalStateHandle checksum mismatch")
        if not self.checksum:
            object.__setattr__(self, "checksum", expected)


def _handle_checksum(handle: CanonicalStateHandle) -> str:
    payload = {
        "canonical_applied_lsn": handle.canonical_applied_lsn,
        "roots": {
            collection.value: [chunk.value for chunk in handle.roots(collection)]
            for collection in CanonicalCollection
        },
        "schema_versions": handle.schema_versions,
        "generation": handle.generation,
    }
    return hashlib.sha256(_encoded(payload)).hexdigest()


class PinnedCanonicalHandle(AbstractContextManager[CanonicalStateHandle]):
    def __init__(self, store: "CanonicalStore", handle: CanonicalStateHandle) -> None:
        self._store = store
        self.handle = handle
        self._released = False

    def __enter__(self) -> CanonicalStateHandle:
        return self.handle

    def __exit__(self, *_args: object) -> None:
        self.release()

    def release(self) -> None:
        if not self._released:
            self._store.release(self.handle)
            self._released = True


class CanonicalStore:
    """Immutable bounded chunks with atomic handle publication and bounded GC."""

    def __init__(
        self,
        *,
        schema_versions: Mapping[str, int] | None = None,
        max_chunk_entries: int = MAX_CHUNK_ENTRIES,
        max_chunk_bytes: int = MAX_CHUNK_BYTES,
        orphan_queue_limit: int = 65_536,
        chunk_directory: str | Path | None = None,
        resident_chunk_limit: int | None = None,
    ) -> None:
        if not 0 < max_chunk_entries <= MAX_CHUNK_ENTRIES:
            raise ValueError("max_chunk_entries is outside the supported bound")
        if not 0 < max_chunk_bytes <= MAX_CHUNK_BYTES:
            raise ValueError("max_chunk_bytes is outside the supported bound")
        if orphan_queue_limit <= 0:
            raise ValueError("orphan_queue_limit must be positive")
        if resident_chunk_limit is not None and resident_chunk_limit < 0:
            raise ValueError("resident_chunk_limit must be non-negative")
        self.max_chunk_entries = int(max_chunk_entries)
        self.max_chunk_bytes = int(max_chunk_bytes)
        self.orphan_queue_limit = int(orphan_queue_limit)
        self.chunk_directory = None if chunk_directory is None else Path(chunk_directory)
        if self.chunk_directory is not None:
            self.chunk_directory.mkdir(parents=True, exist_ok=True)
        self.resident_chunk_limit = None if resident_chunk_limit is None else int(resident_chunk_limit)
        self._lock = RLock()
        self._chunks: OrderedDict[ChunkId, CanonicalChunk] = OrderedDict()
        self._chunk_references: dict[ChunkId, int] = {}
        self._pin_counts: dict[str, int] = {}
        self._pinned_handles: dict[str, CanonicalStateHandle] = {}
        self._orphans: list[ChunkId] = []
        self._current = CanonicalStateHandle(
            0,
            schema_versions=tuple(sorted((str(key), int(value)) for key, value in (schema_versions or {}).items())),
        )
        self._retain_chunks(self._current)

    @property
    def current_handle(self) -> CanonicalStateHandle:
        with self._lock:
            return self._current

    @property
    def chunk_count(self) -> int:
        with self._lock:
            return len(self._chunk_references)

    @property
    def orphan_count(self) -> int:
        with self._lock:
            return len(self._orphans)

    def chunk(self, chunk_id: ChunkId) -> CanonicalChunk:
        with self._lock:
            cached = self._chunks.get(chunk_id)
            if cached is not None:
                self._chunks.move_to_end(chunk_id)
                return cached
            if self.chunk_directory is None:
                raise KeyError(chunk_id)
            path = self.chunk_directory / f"{chunk_id.value}.pkl"
            chunk = pickle.loads(path.read_bytes())
            if not isinstance(chunk, CanonicalChunk) or chunk.chunk_id != chunk_id:
                raise RuntimeError("persisted canonical chunk identity mismatch")
            self._chunks[chunk_id] = chunk
            self._trim_resident_chunks()
            return chunk

    def _trim_resident_chunks(self) -> None:
        if self.resident_chunk_limit is None:
            return
        while len(self._chunks) > self.resident_chunk_limit:
            self._chunks.popitem(last=False)

    def _store_chunk(self, chunk: CanonicalChunk) -> None:
        if chunk.chunk_id in self._chunk_references:
            return
        if self.chunk_directory is not None:
            target = self.chunk_directory / f"{chunk.chunk_id.value}.pkl"
            if not target.exists():
                temporary = target.with_suffix(".tmp")
                temporary.write_bytes(pickle.dumps(chunk, protocol=5))
                os.replace(temporary, target)
        self._chunks[chunk.chunk_id] = chunk
        self._chunk_references[chunk.chunk_id] = 0
        self._trim_resident_chunks()

    def enable_disk_backing(
        self, chunk_directory: str | Path, *, resident_chunk_limit: int = 8
    ) -> None:
        """Persist existing chunks and retain only a bounded resident cache."""
        if resident_chunk_limit < 0:
            raise ValueError("resident_chunk_limit must be non-negative")
        with self._lock:
            self.chunk_directory = Path(chunk_directory)
            self.chunk_directory.mkdir(parents=True, exist_ok=True)
            for chunk_id, chunk in tuple(self._chunks.items()):
                target = self.chunk_directory / f"{chunk_id.value}.pkl"
                if not target.exists():
                    temporary = target.with_suffix(".tmp")
                    temporary.write_bytes(pickle.dumps(chunk, protocol=5))
                    os.replace(temporary, target)
            self.resident_chunk_limit = int(resident_chunk_limit)
            self._trim_resident_chunks()

    def close(self) -> None:
        """Release the bounded resident cache; durable chunks remain on disk."""
        with self._lock:
            self._chunks.clear()

    def pin(self, handle: CanonicalStateHandle | None = None) -> PinnedCanonicalHandle:
        with self._lock:
            selected = self._current if handle is None else handle
            count = self._pin_counts.get(selected.checksum, 0)
            self._pin_counts[selected.checksum] = count + 1
            self._pinned_handles[selected.checksum] = selected
            if count == 0:
                self._retain_chunks(selected)
            return PinnedCanonicalHandle(self, selected)

    def release(self, handle: CanonicalStateHandle) -> None:
        with self._lock:
            count = self._pin_counts.get(handle.checksum, 0)
            if count <= 0:
                raise RuntimeError("canonical handle is not pinned")
            if count == 1:
                self._pin_counts.pop(handle.checksum, None)
                self._pinned_handles.pop(handle.checksum, None)
                self._release_chunks(handle)
            else:
                self._pin_counts[handle.checksum] = count - 1

    def begin_overlay(
        self,
        target_lsn: int,
        *,
        max_entries: int = MAX_CHUNK_ENTRIES,
        max_bytes: int = MAX_CHUNK_BYTES,
    ) -> TransactionOverlay:
        with self._lock:
            if target_lsn <= self._current.canonical_applied_lsn:
                raise ValueError("overlay target LSN must advance the canonical frontier")
            return TransactionOverlay(self._current, target_lsn, max_entries=max_entries, max_bytes=max_bytes)

    def read_from_handle(
        self,
        handle: object,
        collection: CanonicalCollection | str,
        key: Hashable,
        default: Any = None,
    ) -> Any:
        if not isinstance(handle, CanonicalStateHandle):
            raise TypeError("read requires a CanonicalStateHandle")
        selected = collection if isinstance(collection, CanonicalCollection) else CanonicalCollection(str(collection))
        with self._lock:
            for chunk_id in handle.roots(selected):
                try:
                    chunk = self.chunk(chunk_id)
                except (KeyError, FileNotFoundError):
                    raise RuntimeError(f"canonical handle references missing chunk {chunk_id.value}")
                for existing_key, value in chunk.entries:
                    if existing_key == key:
                        return value
        return default

    def collection(self, handle: CanonicalStateHandle, collection: CanonicalCollection | str) -> Mapping[Hashable, object]:
        selected = collection if isinstance(collection, CanonicalCollection) else CanonicalCollection(str(collection))
        with self._lock:
            result: dict[Hashable, object] = {}
            for chunk_id in handle.roots(selected):
                result.update(self.chunk(chunk_id).entries)
            return MappingProxyType(result)

    def _make_chunks(
        self,
        collection: CanonicalCollection,
        entries: Iterable[tuple[Hashable, object]],
        generation: int,
    ) -> tuple[ChunkId, ...]:
        result: list[ChunkId] = []
        pending: list[tuple[Hashable, object]] = []
        pending_bytes = 2

        def finish() -> None:
            nonlocal pending, pending_bytes
            if not pending:
                return
            ordered = tuple((key, _freeze(value)) for key, value in sorted(pending, key=lambda row: repr(row[0])))
            payload = _encoded({"collection": collection.value, "schema_version": 1, "entries": ordered})
            checksum = hashlib.sha256(payload).hexdigest()
            chunk_id = ChunkId(checksum)
            if chunk_id not in self._chunk_references:
                chunk_type = _CHUNK_TYPES[collection]
                self._store_chunk(chunk_type(
                    chunk_id, collection, 1, ordered, len(payload), generation, checksum
                ))
            result.append(chunk_id)
            pending = []
            pending_bytes = 2

        for key, value in sorted(entries, key=lambda row: repr(row[0])):
            entry_bytes = len(_encoded((key, value)))
            if entry_bytes > self.max_chunk_bytes:
                raise OverflowError("one canonical entry exceeds the chunk byte ceiling")
            if pending and (
                len(pending) >= self.max_chunk_entries
                or pending_bytes + entry_bytes > self.max_chunk_bytes
            ):
                finish()
            pending.append((key, value))
            pending_bytes += entry_bytes
        finish()
        return tuple(result)

    def finalize_overlay(self, overlay: TransactionOverlay) -> CanonicalStateHandle:
        with self._lock:
            if overlay.closed:
                raise RuntimeError("cannot finalize a closed overlay")
            if overlay.base_handle is not self._current:
                raise RuntimeError("overlay base is no longer the current canonical handle")
            roots = {collection: list(self._current.roots(collection)) for collection in CanonicalCollection}
            changes_by_collection: dict[CanonicalCollection, dict[Hashable, object]] = {}
            for collection, key, value in overlay.iter_changes():
                changes_by_collection.setdefault(collection, {})[key] = value

            generation = self._current.generation + 1
            for collection, changes in changes_by_collection.items():
                affected: dict[ChunkId, dict[Hashable, object]] = {}
                inserts = dict(changes)
                untouched: list[ChunkId] = []
                for chunk_id in roots[collection]:
                    chunk_rows = dict(self.chunk(chunk_id).entries)
                    overlap = set(chunk_rows).intersection(changes)
                    if not overlap:
                        untouched.append(chunk_id)
                        continue
                    for key in overlap:
                        value = inserts.pop(key)
                        if is_tombstone(value):
                            chunk_rows.pop(key, None)
                        else:
                            chunk_rows[key] = value
                    affected[chunk_id] = chunk_rows
                new_ids = list(untouched)
                for rows in affected.values():
                    new_ids.extend(self._make_chunks(collection, rows.items(), generation))
                new_rows = ((key, value) for key, value in inserts.items() if not is_tombstone(value))
                new_ids.extend(self._make_chunks(collection, new_rows, generation))
                roots[collection] = sorted(set(new_ids), key=lambda chunk_id: chunk_id.value)

            handle = CanonicalStateHandle(
                canonical_applied_lsn=overlay.target_lsn,
                graph_root=tuple(roots[CanonicalCollection.GRAPH]),
                payload_root=tuple(roots[CanonicalCollection.PAYLOAD]),
                level_index_root=tuple(roots[CanonicalCollection.LEVEL_INDEX]),
                signature_index_root=tuple(roots[CanonicalCollection.SIGNATURE_INDEX]),
                grounding_index_root=tuple(roots[CanonicalCollection.GROUNDING_INDEX]),
                provenance_index_root=tuple(roots[CanonicalCollection.PROVENANCE_INDEX]),
                lifecycle_index_root=tuple(roots[CanonicalCollection.LIFECYCLE_INDEX]),
                policy_source_index_root=tuple(roots[CanonicalCollection.POLICY_SOURCE_INDEX]),
                schema_versions=self._current.schema_versions,
                generation=generation,
            )
            old = self._current
            self._retain_chunks(handle)
            self._current = handle
            self._release_chunks(old)
            overlay.close()
            return handle

    def abort_overlay(self, overlay: TransactionOverlay) -> None:
        overlay.abort()

    def _retain_chunks(self, handle: CanonicalStateHandle) -> None:
        for chunk_id in handle.all_chunk_ids:
            self._chunk_references[chunk_id] = self._chunk_references.get(chunk_id, 0) + 1

    def _release_chunks(self, handle: CanonicalStateHandle) -> None:
        for chunk_id in handle.all_chunk_ids:
            count = self._chunk_references.get(chunk_id, 0) - 1
            self._chunk_references[chunk_id] = count
            if count == 0 and chunk_id not in self._orphans:
                if len(self._orphans) >= self.orphan_queue_limit:
                    raise RuntimeError("canonical orphan queue ceiling exceeded")
                self._orphans.append(chunk_id)

    def collect_garbage(self, *, max_chunks: int = 128) -> int:
        if max_chunks <= 0:
            raise ValueError("max_chunks must be positive")
        with self._lock:
            removed = 0
            retained: list[ChunkId] = []
            for chunk_id in self._orphans:
                if removed >= max_chunks:
                    retained.append(chunk_id)
                elif self._chunk_references.get(chunk_id, 0) == 0:
                    self._chunks.pop(chunk_id, None)
                    self._chunk_references.pop(chunk_id, None)
                    if self.chunk_directory is not None:
                        (self.chunk_directory / f"{chunk_id.value}.pkl").unlink(missing_ok=True)
                    removed += 1
                else:
                    retained.append(chunk_id)
            self._orphans = retained
            return removed

    def write_snapshot(
        self,
        path: str | Path,
        *,
        scientific_identity: Mapping[str, str] | None = None,
        crash_hook: Callable[[str], None] | None = None,
    ) -> Path:
        target = Path(path)
        temporary = target.with_name(f".{target.name}.tmp")
        previous = target.with_name(f".{target.name}.previous")
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        with self.pin() as handle:
            with self._lock:
                chunks = tuple(self.chunk(chunk_id) for chunk_id in handle.all_chunk_ids)
            state_payload = pickle.dumps({"handle": handle, "chunks": chunks}, protocol=5)
            state_checksum = hashlib.sha256(state_payload).hexdigest()
            state_path = temporary / "canonical-state.pkl"
            with state_path.open("wb") as output:
                output.write(state_payload)
                output.flush()
                os.fsync(output.fileno())
            if crash_hook:
                crash_hook("state_fsynced")
            manifest = {
                "snapshot_applied_lsn": handle.canonical_applied_lsn,
                "canonical_handle_checksum": handle.checksum,
                "canonical_generation": handle.generation,
                "state_checksum": state_checksum,
                "scientific_identity": dict(sorted((scientific_identity or {}).items())),
            }
            manifest_path = temporary / "manifest.json"
            with manifest_path.open("wb") as output:
                output.write(json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n")
                output.flush()
                os.fsync(output.fileno())
            if crash_hook:
                crash_hook("manifest_fsynced")
            complete = temporary / "COMPLETE"
            with complete.open("wb") as output:
                output.write(handle.checksum.encode() + b"\n")
                output.flush()
                os.fsync(output.fileno())
            if crash_hook:
                crash_hook("complete_fsynced")
        if previous.exists() and target.exists():
            shutil.rmtree(previous)
        if target.exists():
            os.replace(target, previous)
            if crash_hook:
                crash_hook("previous_preserved")
        os.replace(temporary, target)
        if crash_hook:
            crash_hook("snapshot_published")
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if crash_hook:
            crash_hook("directory_fsynced")
        if previous.exists():
            shutil.rmtree(previous)
        return target

    @staticmethod
    def _restorable_snapshot_path(path: str | Path) -> Path:
        target = Path(path)
        if (target / "COMPLETE").is_file():
            return target
        previous = target.with_name(f".{target.name}.previous")
        if (previous / "COMPLETE").is_file():
            return previous
        return target

    @classmethod
    def from_snapshot(
        cls,
        path: str | Path,
        *,
        expected_scientific_identity: Mapping[str, str] | None = None,
    ) -> "CanonicalStore":
        target = cls._restorable_snapshot_path(path)
        if not (target / "COMPLETE").is_file():
            raise ValueError("canonical snapshot is incomplete")
        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        expected = dict(sorted((expected_scientific_identity or {}).items()))
        if expected_scientific_identity is not None and manifest.get("scientific_identity") != expected:
            raise ValueError("canonical snapshot scientific identity mismatch")
        state_payload = (target / "canonical-state.pkl").read_bytes()
        if hashlib.sha256(state_payload).hexdigest() != manifest["state_checksum"]:
            raise ValueError("canonical snapshot state checksum mismatch")
        state = pickle.loads(state_payload)
        handle = state["handle"]
        chunks = tuple(state["chunks"])
        if not isinstance(handle, CanonicalStateHandle) or handle.checksum != manifest["canonical_handle_checksum"]:
            raise ValueError("canonical snapshot handle mismatch")
        store = cls(schema_versions=dict(handle.schema_versions))
        with store._lock:
            store._chunks = OrderedDict()
            store._chunk_references = {}
            for chunk in chunks:
                if not isinstance(chunk, CanonicalChunk):
                    raise ValueError("canonical snapshot contains an invalid chunk")
                payload = _encoded({"collection": chunk.collection.value, "schema_version": chunk.schema_version, "entries": chunk.entries})
                if hashlib.sha256(payload).hexdigest() != chunk.chunk_id.value:
                    raise ValueError("canonical snapshot chunk checksum mismatch")
                store._store_chunk(chunk)
            if set(handle.all_chunk_ids) != set(store._chunk_references):
                raise ValueError("canonical snapshot root/chunk reachability mismatch")
            store._current = handle
            store._retain_chunks(handle)
        return store


__all__ = [
    "canonical_value",
    "CanonicalChunk",
    "CanonicalStateHandle",
    "CanonicalStore",
    "ChunkId",
    "GraphChunk",
    "GroundingIndexChunk",
    "LevelIndexChunk",
    "LifecycleChunk",
    "MAX_CHUNK_BYTES",
    "MAX_CHUNK_ENTRIES",
    "PayloadChunk",
    "PinnedCanonicalHandle",
    "ProvenanceChunk",
    "SignatureIndexChunk",
]
