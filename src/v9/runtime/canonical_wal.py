from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Iterable, Mapping

from .canonical_store import CanonicalStateHandle, CanonicalStore
from .canonical_transaction import TransactionOverlay


_HEADER_MAGIC = b"V9WAL716"
_FRAME_MAGIC = b"FRM1"
_FOOTER_MAGIC = b"CMT1"
_VERSION = 1
_HEADER = struct.Struct(">8sIQQIQ32s")
_FRAME = struct.Struct(">4sQQQ32s")
_FOOTER = struct.Struct(">4sQQ32s")


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _checksum(value: bytes) -> bytes:
    return hashlib.sha256(value).digest()


@dataclass(frozen=True, slots=True)
class WalGroupHeader:
    group_id: int
    previous_lsn: int
    frame_count: int
    payload_bytes: int
    checksum: str


@dataclass(frozen=True, slots=True)
class CanonicalCommitFrame:
    wal_lsn: int
    wal_tx_id: str
    previous_lsn: int
    mutations: tuple[Mapping[str, Any], ...] = ()
    scientific_identity: Mapping[str, Any] | None = None
    producer_causal_ranges: tuple[tuple[int, int, int], ...] = ()
    training_evidence_records: tuple[Mapping[str, Any], ...] = ()
    work_metadata: Mapping[str, int] | None = None
    checksum: str = ""

    def payload(self) -> dict[str, object]:
        return {
            "wal_lsn": self.wal_lsn,
            "wal_tx_id": self.wal_tx_id,
            "previous_lsn": self.previous_lsn,
            "mutations": list(self.mutations),
            "scientific_identity": self.scientific_identity,
            "producer_causal_ranges": [list(row) for row in self.producer_causal_ranges],
            "training_evidence_records": list(self.training_evidence_records),
            "work_metadata": self.work_metadata,
        }

    def with_checksum(self) -> "CanonicalCommitFrame":
        digest = hashlib.sha256(_json_bytes(self.payload())).hexdigest()
        if self.checksum and self.checksum != digest:
            raise ValueError("canonical WAL frame checksum mismatch")
        return self if self.checksum else replace(self, checksum=digest)


@dataclass(frozen=True, slots=True)
class WalGroupCommitFooter:
    group_id: int
    last_lsn: int
    group_checksum: str


@dataclass(frozen=True, slots=True)
class WALRecoveryResult:
    frames: tuple[CanonicalCommitFrame, ...]
    groups: int
    wal_durable_lsn: int
    valid_bytes: int
    truncated_bytes: int


@dataclass(frozen=True, slots=True)
class PersistenceFrontiers:
    wal_durable_lsn: int = 0
    canonical_applied_lsn: int = 0
    snapshot_applied_lsn: int = 0
    hgt_checkpoint_lsn: int = 0

    def __post_init__(self) -> None:
        values = (
            self.wal_durable_lsn,
            self.canonical_applied_lsn,
            self.snapshot_applied_lsn,
            self.hgt_checkpoint_lsn,
        )
        if min(values) < 0:
            raise ValueError("persistence frontiers must be non-negative")
        if not self.wal_durable_lsn >= self.canonical_applied_lsn >= self.snapshot_applied_lsn:
            raise ValueError("WAL/canonical/snapshot frontier invariant violated")
        if self.hgt_checkpoint_lsn > self.wal_durable_lsn:
            raise ValueError("HGT checkpoint cannot exceed durable WAL")


@dataclass(frozen=True, slots=True)
class CanonicalRecoveryResult:
    store: CanonicalStore
    wal_recovery: WALRecoveryResult
    replayed_lsns: tuple[int, ...]
    frontiers: PersistenceFrontiers


def _decode_frame(payload: bytes, checksum: bytes, lsn: int, previous_lsn: int) -> CanonicalCommitFrame:
    if _checksum(payload) != checksum:
        raise ValueError("canonical WAL transaction-frame checksum mismatch")
    raw = json.loads(payload.decode("utf-8"))
    if int(raw["wal_lsn"]) != lsn or int(raw["previous_lsn"]) != previous_lsn:
        raise ValueError("canonical WAL frame header/payload mismatch")
    return CanonicalCommitFrame(
        wal_lsn=lsn,
        wal_tx_id=str(raw["wal_tx_id"]),
        previous_lsn=previous_lsn,
        mutations=tuple(dict(row) for row in raw.get("mutations", ())),
        scientific_identity=None if raw.get("scientific_identity") is None else dict(raw["scientific_identity"]),
        producer_causal_ranges=tuple(tuple(int(value) for value in row) for row in raw.get("producer_causal_ranges", ())),
        training_evidence_records=tuple(dict(row) for row in raw.get("training_evidence_records", ())),
        work_metadata=None if raw.get("work_metadata") is None else {str(key): int(value) for key, value in dict(raw["work_metadata"]).items()},
        checksum=checksum.hex(),
    )


class CanonicalCommitWAL:
    """Framed group-commit redo log; visibility advances only after fsync."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_group_frames: int = 1024,
        max_pending_bytes: int = 64 * 1024 * 1024,
        max_durable_consumers: int = 256,
        fsync: Callable[[int], None] = os.fsync,
    ) -> None:
        if min(max_group_frames, max_pending_bytes, max_durable_consumers) <= 0:
            raise ValueError("WAL group bounds must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._consumer_path = self.path.with_name(f"{self.path.name}.consumers.json")
        self.max_group_frames = int(max_group_frames)
        self.max_pending_bytes = int(max_pending_bytes)
        self.max_durable_consumers = int(max_durable_consumers)
        self._fsync = fsync
        self._lock = RLock()
        self._durable_consumers: dict[str, int] = {}
        self._group_id = 0
        self._durable_lsn = 0
        self._poisoned = False
        self._pending_bytes = 0
        self._pending_started_at = 0.0
        recovery = self.recover(truncate=True)
        self._durable_lsn = recovery.wal_durable_lsn
        self._group_id = recovery.groups
        self._durable_consumers = self._load_durable_consumers()

    @property
    def wal_durable_lsn(self) -> int:
        with self._lock:
            return self._durable_lsn

    @property
    def append_available(self) -> bool:
        """Whether this process can safely append without reopening the WAL."""
        with self._lock:
            return not self._poisoned

    @property
    def pending_bytes(self) -> int:
        return int(self._pending_bytes)

    @property
    def pending_age_seconds(self) -> float:
        started = float(self._pending_started_at)
        return 0.0 if started <= 0.0 else max(0.0, time.monotonic() - started)

    @staticmethod
    def _write_all(handle: Any, payload: bytes) -> None:
        view = memoryview(payload)
        while view:
            written = handle.write(view)
            if written is None or written <= 0:
                raise OSError("canonical WAL write made no progress")
            view = view[written:]

    @staticmethod
    def _encode_group(
        frames: tuple[CanonicalCommitFrame, ...], *, group_id: int
    ) -> bytes:
        if not frames:
            raise ValueError("cannot encode an empty WAL group")
        previous = int(frames[0].previous_lsn)
        payloads = tuple(_json_bytes(frame.payload()) for frame in frames)
        for frame in frames:
            if frame.previous_lsn != previous or frame.wal_lsn != previous + 1:
                raise ValueError("WAL group frames are not contiguous")
            previous = frame.wal_lsn
        total_payload = sum(len(payload) for payload in payloads)
        header_core = struct.pack(
            ">8sIQQIQ",
            _HEADER_MAGIC,
            _VERSION,
            int(group_id),
            int(frames[0].previous_lsn),
            len(frames),
            total_payload,
        )
        encoded_header = _HEADER.pack(
            _HEADER_MAGIC,
            _VERSION,
            int(group_id),
            int(frames[0].previous_lsn),
            len(frames),
            total_payload,
            _checksum(header_core),
        )
        encoded_frames = tuple(
            _FRAME.pack(
                _FRAME_MAGIC,
                frame.wal_lsn,
                frame.previous_lsn,
                len(payload),
                bytes.fromhex(frame.checksum),
            )
            + payload
            for frame, payload in zip(frames, payloads)
        )
        body = encoded_header + b"".join(encoded_frames)
        return body + _FOOTER.pack(
            _FOOTER_MAGIC,
            int(group_id),
            frames[-1].wal_lsn,
            _checksum(body),
        )

    def _prepare_frames(
        self, rows: Iterable[CanonicalCommitFrame | Mapping[str, Any]]
    ) -> tuple[CanonicalCommitFrame, ...]:
        result: list[CanonicalCommitFrame] = []
        previous = self._durable_lsn
        for index, row in enumerate(rows, start=1):
            if isinstance(row, CanonicalCommitFrame):
                tx_id = row.wal_tx_id
                values = {
                    "mutations": row.mutations,
                    "scientific_identity": row.scientific_identity,
                    "producer_causal_ranges": row.producer_causal_ranges,
                    "training_evidence_records": row.training_evidence_records,
                    "work_metadata": row.work_metadata,
                }
            else:
                values = dict(row)
                tx_id = str(values.pop("wal_tx_id", values.pop("transaction_id", "")))
            if not tx_id:
                raise ValueError("every WAL transaction requires wal_tx_id")
            frame = CanonicalCommitFrame(
                wal_lsn=self._durable_lsn + index,
                wal_tx_id=tx_id,
                previous_lsn=previous,
                mutations=tuple(values.get("mutations", ())),
                scientific_identity=values.get("scientific_identity"),
                producer_causal_ranges=tuple(tuple(row) for row in values.get("producer_causal_ranges", ())),
                training_evidence_records=tuple(values.get("training_evidence_records", ())),
                work_metadata=values.get("work_metadata"),
            ).with_checksum()
            result.append(frame)
            previous = frame.wal_lsn
        if not result:
            raise ValueError("cannot append an empty WAL group")
        if len(result) > self.max_group_frames:
            raise OverflowError("WAL group frame ceiling exceeded")
        return tuple(result)

    def append_group(
        self, rows: Iterable[CanonicalCommitFrame | Mapping[str, Any]]
    ) -> tuple[CanonicalCommitFrame, ...]:
        with self._lock:
            if self._poisoned:
                raise RuntimeError(
                    "canonical WAL append state is indeterminate; reopen and recover the WAL"
                )
            frames = self._prepare_frames(rows)
            payloads = tuple(_json_bytes(frame.payload()) for frame in frames)
            total_payload = sum(len(payload) for payload in payloads)
            if total_payload > self.max_pending_bytes:
                raise OverflowError("WAL group pending-byte ceiling exceeded")
            group_id = self._group_id + 1
            encoded_group = self._encode_group(frames, group_id=group_id)
            self._pending_bytes = total_payload
            self._pending_started_at = time.monotonic()
            try:
                with self.path.open("ab", buffering=0) as handle:
                    self._write_all(handle, encoded_group)
                    handle.flush()
                    self._fsync(handle.fileno())
            except BaseException:
                # Once bytes may have reached the file, their durability is
                # unknowable to this process.  Continuing from the old LSN can
                # create a second branch with the same previous_lsn.  Recovery
                # in a newly opened WAL is the only safe way to resume.
                self._poisoned = True
                raise
            finally:
                self._pending_bytes = 0
                self._pending_started_at = 0.0
            # The durable frontier moves only after the complete group fsync returns.
            self._durable_lsn = frames[-1].wal_lsn
            self._group_id = group_id
            return frames

    def recover(self, *, truncate: bool = True) -> WALRecoveryResult:
        with self._lock:
            if not self.path.exists():
                return WALRecoveryResult((), 0, 0, 0, 0)
            file_size = int(self.path.stat().st_size)
            valid_end = 0
            frames: list[CanonicalCommitFrame] = []
            groups = 0
            expected_previous = 0

            with self.path.open("rb", buffering=0) as handle:
                while True:
                    group_start = int(handle.tell())
                    encoded_header = handle.read(_HEADER.size)
                    if not encoded_header:
                        break
                    if len(encoded_header) != _HEADER.size:
                        break
                    try:
                        magic, version, group_id, previous_lsn, frame_count, payload_bytes, header_checksum = _HEADER.unpack(encoded_header)
                        if groups == 0 and group_id == 1:
                            expected_previous = int(previous_lsn)
                        if (
                            magic != _HEADER_MAGIC
                            or version != _VERSION
                            or group_id != groups + 1
                            or previous_lsn != expected_previous
                        ):
                            break
                        core = struct.pack(
                            ">8sIQQIQ",
                            magic,
                            version,
                            group_id,
                            previous_lsn,
                            frame_count,
                            payload_bytes,
                        )
                        if (
                            _checksum(core) != header_checksum
                            or not 0 < frame_count <= self.max_group_frames
                            or payload_bytes > self.max_pending_bytes
                        ):
                            break

                        group_hasher = hashlib.sha256()
                        group_hasher.update(encoded_header)
                        group_frames: list[CanonicalCommitFrame] = []
                        actual_payload = 0
                        frame_previous = previous_lsn

                        for _ in range(frame_count):
                            encoded_frame = handle.read(_FRAME.size)
                            if len(encoded_frame) != _FRAME.size:
                                raise ValueError("torn frame header")
                            group_hasher.update(encoded_frame)
                            frame_magic, lsn, prior, length, checksum = _FRAME.unpack(encoded_frame)
                            if (
                                frame_magic != _FRAME_MAGIC
                                or prior != frame_previous
                                or lsn != frame_previous + 1
                                or length > self.max_pending_bytes
                            ):
                                raise ValueError("invalid frame header")
                            payload = handle.read(length)
                            if len(payload) != length:
                                raise ValueError("torn frame payload")
                            group_hasher.update(payload)
                            group_frames.append(_decode_frame(payload, checksum, lsn, prior))
                            actual_payload += length
                            frame_previous = lsn

                        if actual_payload != payload_bytes:
                            raise ValueError("invalid group payload size")
                        encoded_footer = handle.read(_FOOTER.size)
                        if len(encoded_footer) != _FOOTER.size:
                            raise ValueError("torn footer")
                        footer_magic, footer_group, last_lsn, group_checksum = _FOOTER.unpack(encoded_footer)
                        if (
                            footer_magic != _FOOTER_MAGIC
                            or footer_group != group_id
                            or last_lsn != group_frames[-1].wal_lsn
                            or group_hasher.digest() != group_checksum
                        ):
                            raise ValueError("invalid WAL group footer")
                    except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
                        break

                    frames.extend(group_frames)
                    groups += 1
                    expected_previous = group_frames[-1].wal_lsn
                    valid_end = int(handle.tell())
                    if valid_end <= group_start:
                        raise RuntimeError("WAL recovery made no forward progress")

            truncated = file_size - valid_end
            if truncate and truncated:
                with self.path.open("r+b") as handle:
                    handle.truncate(valid_end)
                    handle.flush()
                    self._fsync(handle.fileno())
            return WALRecoveryResult(
                tuple(frames), groups, expected_previous, valid_end, truncated
            )

    def commit_overlay(
        self,
        store: CanonicalStore,
        overlay: TransactionOverlay,
        *,
        transaction_id: str,
        frame_payload: Mapping[str, Any] | None = None,
        after_durable: Callable[[CanonicalCommitFrame], None] | None = None,
    ) -> CanonicalStateHandle:
        if overlay.base_handle is not store.current_handle:
            raise RuntimeError("WAL commit overlay is not based on the current canonical handle")
        values = dict(frame_payload or {})
        values["wal_tx_id"] = transaction_id
        frames = self.append_group((values,))
        durable = frames[0]
        if durable.wal_lsn != overlay.target_lsn:
            raise RuntimeError("overlay target LSN does not match durable WAL LSN")
        if after_durable is not None:
            after_durable(durable)
        return store.finalize_overlay(overlay)

    def _load_durable_consumers(self) -> dict[str, int]:
        if not self._consumer_path.exists():
            return {}
        raw = json.loads(self._consumer_path.read_text(encoding="utf-8"))
        if int(raw.get("schema_version", 0)) != 1:
            raise RuntimeError("unsupported WAL durable-consumer registry schema")
        consumers = {
            str(name): int(checkpoint)
            for name, checkpoint in dict(raw.get("consumers", {})).items()
        }
        if len(consumers) > self.max_durable_consumers:
            raise RuntimeError("WAL durable-consumer registry exceeds configured bound")
        if any(
            not name or not 0 <= checkpoint <= self._durable_lsn
            for name, checkpoint in consumers.items()
        ):
            raise RuntimeError("WAL durable-consumer registry has an invalid checkpoint")
        return consumers

    def _persist_durable_consumers(self, consumers: Mapping[str, int]) -> None:
        payload = _json_bytes(
            {"schema_version": 1, "consumers": dict(sorted(consumers.items()))}
        ) + b"\n"
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.path.parent,
                prefix=f".{self._consumer_path.name}.",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                self._write_all(handle, payload)
                handle.flush()
                self._fsync(handle.fileno())
            os.replace(temporary_path, self._consumer_path)
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                self._fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def register_durable_consumer(self, name: str, checkpoint_lsn: int = 0) -> None:
        if not name or not 0 <= checkpoint_lsn <= self.wal_durable_lsn:
            raise ValueError("invalid WAL durable-consumer checkpoint")
        with self._lock:
            if name not in self._durable_consumers and len(self._durable_consumers) >= self.max_durable_consumers:
                raise OverflowError("WAL durable-consumer count ceiling exceeded")
            previous = self._durable_consumers.get(name)
            if previous is not None:
                if checkpoint_lsn == 0:
                    return
                if checkpoint_lsn < previous:
                    raise ValueError("durable-consumer checkpoint cannot move backward")
            updated = dict(self._durable_consumers)
            updated[name] = int(checkpoint_lsn)
            self._persist_durable_consumers(updated)
            self._durable_consumers = updated

    def update_durable_consumer(self, name: str, checkpoint_lsn: int) -> None:
        with self._lock:
            previous = self._durable_consumers.get(name)
            if previous is None:
                raise KeyError(name)
            if not previous <= checkpoint_lsn <= self._durable_lsn:
                raise ValueError("durable-consumer checkpoints must advance within durable WAL")
            updated = dict(self._durable_consumers)
            updated[name] = int(checkpoint_lsn)
            self._persist_durable_consumers(updated)
            self._durable_consumers = updated

    @property
    def wal_reclaim_lsn(self) -> int:
        with self._lock:
            return min(self._durable_consumers.values(), default=0)

    def durable_consumer_checkpoint(self, name: str) -> int:
        with self._lock:
            if name not in self._durable_consumers:
                raise KeyError(name)
            return int(self._durable_consumers[name])

    def reclaim_prefix(self) -> int:
        """Atomically remove frames no registered durable consumer still needs."""
        with self._lock:
            if self._poisoned:
                raise RuntimeError("cannot reclaim a poisoned WAL before reopening it")
            reclaim_lsn = self.wal_reclaim_lsn
            if reclaim_lsn <= 0:
                return 0
            recovery = self.recover(truncate=True)
            frames = recovery.frames
            if len(frames) <= 1:
                return 0
            removable = 0
            while removable < len(frames) - 1 and frames[removable].wal_lsn <= reclaim_lsn:
                removable += 1
            if removable == 0:
                return 0
            retained = frames[removable:]

            temporary_path: Path | None = None
            replaced = False
            written_groups = 0
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=self.path.parent,
                    prefix=f".{self.path.name}.reclaim-",
                    delete=False,
                ) as handle:
                    temporary_path = Path(handle.name)
                    pending: list[CanonicalCommitFrame] = []
                    pending_bytes = 0

                    def flush_group() -> None:
                        nonlocal pending, pending_bytes, written_groups
                        if not pending:
                            return
                        written_groups += 1
                        encoded = self._encode_group(
                            tuple(pending), group_id=written_groups
                        )
                        self._write_all(handle, encoded)
                        pending = []
                        pending_bytes = 0

                    for frame in retained:
                        payload_bytes = len(_json_bytes(frame.payload()))
                        if pending and (
                            len(pending) >= self.max_group_frames
                            or pending_bytes + payload_bytes > self.max_pending_bytes
                        ):
                            flush_group()
                        pending.append(frame)
                        pending_bytes += payload_bytes
                    flush_group()
                    handle.flush()
                    self._fsync(handle.fileno())

                os.replace(temporary_path, self.path)
                replaced = True
                directory_fd = os.open(self.path.parent, os.O_RDONLY)
                try:
                    self._fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except BaseException:
                if replaced:
                    self._poisoned = True
                raise
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

            self._group_id = written_groups
            self._durable_lsn = retained[-1].wal_lsn
            return removable


def recover_canonical_store(
    *,
    snapshot_path: str | Path,
    wal: CanonicalCommitWAL,
    expected_scientific_identity: Mapping[str, str] | None = None,
) -> CanonicalRecoveryResult:
    store = CanonicalStore.from_snapshot(
        snapshot_path, expected_scientific_identity=expected_scientific_identity
    )
    recovery = wal.recover(truncate=True)
    snapshot_applied_lsn = store.current_handle.canonical_applied_lsn
    replayed = []
    for frame in recovery.frames:
        if frame.wal_lsn <= store.current_handle.canonical_applied_lsn:
            continue
        overlay = store.begin_overlay(frame.wal_lsn)
        for mutation in frame.mutations:
            collection = str(mutation["collection"])
            key = mutation["key"]
            operation = str(mutation.get("operation", "put"))
            if operation == "delete":
                overlay.delete(collection, key)
            elif operation == "put":
                overlay.put(collection, key, mutation.get("value"))
            else:
                raise ValueError(f"unknown recovered canonical mutation: {operation}")
        store.finalize_overlay(overlay)
        replayed.append(frame.wal_lsn)
    frontiers = PersistenceFrontiers(
        recovery.wal_durable_lsn,
        store.current_handle.canonical_applied_lsn,
        snapshot_applied_lsn,
        0,
    )
    return CanonicalRecoveryResult(store, recovery, tuple(replayed), frontiers)


__all__ = [
    "CanonicalCommitFrame",
    "CanonicalCommitWAL",
    "CanonicalRecoveryResult",
    "PersistenceFrontiers",
    "WALRecoveryResult",
    "WalGroupCommitFooter",
    "WalGroupHeader",
    "recover_canonical_store",
]
