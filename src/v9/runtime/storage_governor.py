from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Mapping

from .canonical_store import CanonicalStateHandle, CanonicalStore, PinnedCanonicalHandle


class DurableStorageClass(str, Enum):
    WAL = "wal"
    CANONICAL_CHUNK = "canonical_chunks"
    SNAPSHOT = "snapshots"
    TRAINING_EVIDENCE = "training_evidence"
    MODEL_CHECKPOINT = "model_checkpoints"
    OPTIMIZER_CHECKPOINT = "optimizer_checkpoints"
    EVALUATION_ARTIFACT = "evaluation_artifacts"
    SIGNATURE_INDEX = "signature_index"
    POLICY_VIEW_MANIFEST = "policy_view_manifests"


class StoragePressureState(str, Enum):
    NORMAL = "NORMAL"
    SOFT_PRESSURE = "SOFT_PRESSURE"
    HARD_PRESSURE_DRAIN = "HARD_PRESSURE_DRAIN"


@dataclass(frozen=True, slots=True)
class StorageObject:
    object_id: str
    storage_class: DurableStorageClass
    path: Path
    encoded_bytes: int
    references: frozenset[str] = frozenset()
    reclaimable: bool = False


@dataclass(frozen=True, slots=True)
class StorageGovernorStatus:
    state: StoragePressureState
    bytes_by_class: tuple[tuple[str, int], ...]
    aggregate_bytes: int
    aggregate_ceiling: int
    filesystem_free_bytes: int
    filesystem_free_fraction: float
    actor_admission_allowed: bool
    checkpoint_creation_allowed: bool


class StorageGovernor:
    def __init__(
        self,
        root: str | Path,
        *,
        class_ceilings: Mapping[DurableStorageClass | str, int],
        aggregate_ceiling: int,
        soft_fraction: float = 0.85,
        minimum_free_bytes: int = 2 * 1024 * 1024 * 1024,
        minimum_free_fraction: float = 0.05,
        max_objects: int = 1_000_000,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.class_ceilings = {
            key if isinstance(key, DurableStorageClass) else DurableStorageClass(str(key)): int(value)
            for key, value in class_ceilings.items()
        }
        if set(self.class_ceilings) != set(DurableStorageClass):
            raise ValueError("storage governor requires a ceiling for every durable class")
        if min(self.class_ceilings.values()) <= 0 or aggregate_ceiling <= 0 or max_objects <= 0:
            raise ValueError("durable storage ceilings must be positive")
        if not 0 < soft_fraction < 1 or minimum_free_bytes < 0 or not 0 <= minimum_free_fraction < 1:
            raise ValueError("invalid filesystem pressure thresholds")
        self.aggregate_ceiling = int(aggregate_ceiling)
        self.soft_fraction = float(soft_fraction)
        self.minimum_free_bytes = int(minimum_free_bytes)
        self.minimum_free_fraction = float(minimum_free_fraction)
        self.max_objects = int(max_objects)
        self._objects: dict[str, StorageObject] = {}
        self._lock = RLock()

    @staticmethod
    def _classify(relative: Path) -> DurableStorageClass:
        parts = relative.parts
        suffix = relative.suffix.lower()
        if relative == Path("canonical/commit.wal") or relative.name.startswith("commit.wal"):
            return DurableStorageClass.WAL
        if parts[:2] == ("canonical", "chunks"):
            return DurableStorageClass.CANONICAL_CHUNK
        if "snapshots" in parts or relative.name.startswith("snapshot-"):
            return DurableStorageClass.SNAPSHOT
        if parts[:2] == ("hgt", "training_evidence"):
            return DurableStorageClass.TRAINING_EVIDENCE
        if "optimizer" in relative.name.lower():
            return DurableStorageClass.OPTIMIZER_CHECKPOINT
        if suffix in {".pt", ".pth", ".safetensors"}:
            return DurableStorageClass.MODEL_CHECKPOINT
        if parts and parts[0] == "indexes":
            return DurableStorageClass.SIGNATURE_INDEX
        if "cut" in relative.name.lower() or "view" in relative.name.lower():
            return DurableStorageClass.POLICY_VIEW_MANIFEST
        return DurableStorageClass.EVALUATION_ARTIFACT

    def reconcile_filesystem(self) -> None:
        """Rebuild bounded accounting from durable files after restart or writes."""
        with self._lock:
            rows: dict[str, StorageObject] = {}
            for path in sorted(self.root.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(self.root)
                object_id = str(relative)
                previous = self._objects.get(object_id)
                rows[object_id] = StorageObject(
                    object_id,
                    self._classify(relative),
                    path,
                    int(path.stat().st_size),
                    frozenset() if previous is None else previous.references,
                    False if previous is None else previous.reclaimable,
                )
                if len(rows) > self.max_objects:
                    raise OverflowError("durable storage object-count ceiling exceeded")
            self._objects = rows

    def register(self, row: StorageObject) -> None:
        path = row.path.resolve()
        if path != self.root and self.root not in path.parents:
            raise ValueError("durable object path is outside the governed root")
        if row.encoded_bytes < 0:
            raise ValueError("durable object bytes must be non-negative")
        with self._lock:
            existing = self._objects.get(row.object_id)
            if existing is not None and existing != row:
                raise ValueError("durable object identity collision")
            if existing is None and len(self._objects) >= self.max_objects:
                raise OverflowError("durable storage object-count ceiling exceeded")
            projected = self.bytes_by_class()
            projected[row.storage_class] += row.encoded_bytes - (0 if existing is None else existing.encoded_bytes)
            if projected[row.storage_class] > self.class_ceilings[row.storage_class]:
                raise OverflowError(f"{row.storage_class.value} durable byte ceiling exceeded")
            if sum(projected.values()) > self.aggregate_ceiling:
                raise OverflowError("aggregate durable byte ceiling exceeded")
            self._objects[row.object_id] = row

    def add_reference(self, object_id: str, reference: str) -> None:
        with self._lock:
            row = self._objects[object_id]
            self._objects[object_id] = StorageObject(row.object_id, row.storage_class, row.path, row.encoded_bytes, row.references | {reference}, row.reclaimable)

    def release_reference(self, object_id: str, reference: str) -> None:
        with self._lock:
            row = self._objects[object_id]
            self._objects[object_id] = StorageObject(row.object_id, row.storage_class, row.path, row.encoded_bytes, row.references - {reference}, row.reclaimable)

    def mark_reclaimable(self, object_id: str) -> None:
        with self._lock:
            row = self._objects[object_id]
            self._objects[object_id] = StorageObject(row.object_id, row.storage_class, row.path, row.encoded_bytes, row.references, True)

    def bytes_by_class(self) -> dict[DurableStorageClass, int]:
        result = {kind: 0 for kind in DurableStorageClass}
        for row in self._objects.values():
            result[row.storage_class] += row.encoded_bytes
        return result

    def status(self) -> StorageGovernorStatus:
        with self._lock:
            totals = self.bytes_by_class()
            aggregate = sum(totals.values())
            usage = shutil.disk_usage(self.root)
            free_fraction = usage.free / max(1, usage.total)
            hard = (
                aggregate >= self.aggregate_ceiling
                or any(totals[kind] >= self.class_ceilings[kind] for kind in DurableStorageClass)
                or usage.free <= self.minimum_free_bytes
                or free_fraction <= self.minimum_free_fraction
            )
            soft = aggregate >= int(self.aggregate_ceiling * self.soft_fraction) or any(
                totals[kind] >= int(self.class_ceilings[kind] * self.soft_fraction)
                for kind in DurableStorageClass
            )
            state = StoragePressureState.HARD_PRESSURE_DRAIN if hard else StoragePressureState.SOFT_PRESSURE if soft else StoragePressureState.NORMAL
            return StorageGovernorStatus(
                state,
                tuple((kind.value, totals[kind]) for kind in DurableStorageClass),
                aggregate,
                self.aggregate_ceiling,
                usage.free,
                free_fraction,
                state is not StoragePressureState.HARD_PRESSURE_DRAIN,
                state is StoragePressureState.NORMAL,
            )

    def collect(self, *, max_objects: int = 128) -> tuple[str, ...]:
        if max_objects <= 0:
            raise ValueError("storage GC budget must be positive")
        with self._lock:
            candidates = sorted(
                (
                    row for row in self._objects.values()
                    if row.reclaimable and not row.references
                ),
                key=lambda row: (row.storage_class.value, row.object_id),
            )[:max_objects]
            removed = []
            for row in candidates:
                path = row.path.resolve()
                if path != self.root and self.root not in path.parents:
                    raise RuntimeError("refusing to reclaim an object outside the governed root")
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink(missing_ok=True)
                self._objects.pop(row.object_id, None)
                removed.append(row.object_id)
            return tuple(removed)


class CanonicalSnapshotPin:
    def __init__(self, store: CanonicalStore, handle: CanonicalStateHandle | None = None) -> None:
        self._pin: PinnedCanonicalHandle = store.pin(handle)
        self.handle = self._pin.handle

    def manifest(self) -> dict[str, object]:
        return {
            "snapshot_applied_lsn": self.handle.canonical_applied_lsn,
            "canonical_handle_checksum": self.handle.checksum,
            "canonical_generation": self.handle.generation,
            "roots": [chunk.value for chunk in self.handle.all_chunk_ids],
            "schema_versions": dict(self.handle.schema_versions),
        }

    def close(self) -> None:
        self._pin.release()

    def __enter__(self) -> "CanonicalSnapshotPin":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


__all__ = [
    "CanonicalSnapshotPin",
    "DurableStorageClass",
    "StorageGovernor",
    "StorageGovernorStatus",
    "StorageObject",
    "StoragePressureState",
]
