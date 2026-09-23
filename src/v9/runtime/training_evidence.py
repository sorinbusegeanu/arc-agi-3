from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


class TrainingEvidenceKind(str, Enum):
    INTERACTION = "interaction"
    TRANSITION_CONSEQUENCE = "transition_consequence"
    MEMORY_FORMATION = "memory_formation"
    TRANSFER = "transfer"
    GROUNDING = "grounding"
    REASONING_TRACE = "reasoning_trace"
    DELAYED_OUTCOME = "delayed_outcome"
    CONSOLIDATION = "consolidation"
    DERIVATION = "derivation"
    INVARIANCE = "invariance"
    STRATEGY_RANKING = "strategy_ranking"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


@dataclass(frozen=True, order=True, slots=True)
class TrainingEvidenceId:
    value: str

    def __post_init__(self) -> None:
        if len(self.value) != 64:
            raise ValueError("TrainingEvidenceId must be a SHA-256 digest")


@dataclass(frozen=True, slots=True)
class TrainingEvidenceRecord:
    evidence_id: TrainingEvidenceId
    kind: TrainingEvidenceKind
    source_wal_lsn: int
    scientific_provenance: tuple[tuple[str, str], ...]
    schema_versions: tuple[tuple[str, int], ...]
    label_payload: Mapping[str, Any]
    quality_millionths: int = 1_000_000
    weight_millionths: int = 1_000_000
    checksum: str = ""

    @classmethod
    def create(
        cls,
        *,
        kind: TrainingEvidenceKind | str,
        source_wal_lsn: int,
        scientific_provenance: Mapping[str, object],
        schema_versions: Mapping[str, int],
        label_payload: Mapping[str, Any],
        quality_millionths: int = 1_000_000,
        weight_millionths: int = 1_000_000,
    ) -> "TrainingEvidenceRecord":
        selected = kind if isinstance(kind, TrainingEvidenceKind) else TrainingEvidenceKind(str(kind))
        identity_payload = {
            "kind": selected.value,
            "scientific_provenance": sorted((str(key), str(value)) for key, value in scientific_provenance.items()),
            "schema_versions": sorted((str(key), int(value)) for key, value in schema_versions.items()),
            "label_payload": label_payload,
        }
        evidence_id = TrainingEvidenceId(hashlib.sha256(_canonical(identity_payload)).hexdigest())
        record_payload = {
            **identity_payload,
            "evidence_id": evidence_id.value,
            "source_wal_lsn": int(source_wal_lsn),
            "quality_millionths": int(quality_millionths),
            "weight_millionths": int(weight_millionths),
        }
        checksum = hashlib.sha256(_canonical(record_payload)).hexdigest()
        return cls(
            evidence_id,
            selected,
            int(source_wal_lsn),
            tuple(identity_payload["scientific_provenance"]),
            tuple(identity_payload["schema_versions"]),
            dict(label_payload),
            int(quality_millionths),
            int(weight_millionths),
            checksum,
        )

    def __post_init__(self) -> None:
        if self.source_wal_lsn < 0 or not 0 <= self.quality_millionths <= 1_000_000 or self.weight_millionths < 0:
            raise ValueError("invalid training evidence frontier/quality/weight")
        payload = self.as_dict(include_checksum=False)
        expected = hashlib.sha256(_canonical(payload)).hexdigest()
        if self.checksum and self.checksum != expected:
            raise ValueError("TrainingEvidenceRecord checksum mismatch")
        if not self.checksum:
            object.__setattr__(self, "checksum", expected)

    def as_dict(self, *, include_checksum: bool = True) -> dict[str, object]:
        payload = {
            "evidence_id": self.evidence_id.value,
            "kind": self.kind.value,
            "source_wal_lsn": self.source_wal_lsn,
            "scientific_provenance": [list(row) for row in self.scientific_provenance],
            "schema_versions": [list(row) for row in self.schema_versions],
            "label_payload": dict(self.label_payload),
            "quality_millionths": self.quality_millionths,
            "weight_millionths": self.weight_millionths,
        }
        if include_checksum:
            payload["checksum"] = self.checksum
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TrainingEvidenceRecord":
        return cls(
            TrainingEvidenceId(str(raw["evidence_id"])),
            TrainingEvidenceKind(str(raw["kind"])),
            int(raw["source_wal_lsn"]),
            tuple((str(key), str(value)) for key, value in raw["scientific_provenance"]),
            tuple((str(key), int(value)) for key, value in raw["schema_versions"]),
            dict(raw["label_payload"]),
            int(raw.get("quality_millionths", 1_000_000)),
            int(raw.get("weight_millionths", 1_000_000)),
            str(raw.get("checksum", "")),
        )


@dataclass(frozen=True, slots=True)
class TrainingEvidenceSegment:
    segment_id: str
    path: str
    start_lsn: int
    end_lsn: int
    records: int
    encoded_bytes: int
    checksum: str


@dataclass(frozen=True, slots=True)
class TrainingEvidenceManifest:
    segments: tuple[TrainingEvidenceSegment, ...] = ()
    hgt_checkpoint_lsn: int = 0
    generation: int = 0
    checksum: str = ""

    def __post_init__(self) -> None:
        previous = 0
        for segment in self.segments:
            if segment.start_lsn != previous + 1 or segment.end_lsn < segment.start_lsn:
                raise ValueError("training evidence segments must cover contiguous WAL ranges")
            previous = segment.end_lsn
        if self.hgt_checkpoint_lsn != previous:
            raise ValueError("HGT checkpoint must equal the manifested contiguous WAL frontier")
        payload = {
            "segments": [asdict(segment) for segment in self.segments],
            "hgt_checkpoint_lsn": self.hgt_checkpoint_lsn,
            "generation": self.generation,
        }
        expected = hashlib.sha256(_canonical(payload)).hexdigest()
        if self.checksum and self.checksum != expected:
            raise ValueError("TrainingEvidenceManifest checksum mismatch")
        if not self.checksum:
            object.__setattr__(self, "checksum", expected)


class TrainingEvidenceMaterializer:
    def __init__(self, root: str | Path, *, max_segment_bytes: int = 64 * 1024 * 1024) -> None:
        if max_segment_bytes <= 0:
            raise ValueError("training evidence segment bound must be positive")
        self.root = Path(root)
        self.segments_root = self.root / "segments"
        self.manifest_path = self.root / "manifest.json"
        self.segments_root.mkdir(parents=True, exist_ok=True)
        self.max_segment_bytes = int(max_segment_bytes)
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> TrainingEvidenceManifest:
        if not self.manifest_path.exists():
            return TrainingEvidenceManifest()
        raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return TrainingEvidenceManifest(
            tuple(TrainingEvidenceSegment(**row) for row in raw.get("segments", ())),
            int(raw.get("hgt_checkpoint_lsn", 0)),
            int(raw.get("generation", 0)),
            str(raw.get("checksum", "")),
        )

    def materialize(
        self,
        *,
        start_lsn: int,
        end_lsn: int,
        records: Iterable[TrainingEvidenceRecord],
        wal_durable_lsn: int,
        crash_hook: Callable[[str], None] | None = None,
    ) -> TrainingEvidenceManifest:
        if start_lsn != self.manifest.hgt_checkpoint_lsn + 1 or not start_lsn <= end_lsn <= wal_durable_lsn:
            raise ValueError("training evidence range is not the next durable contiguous WAL range")
        rows = tuple(sorted(records, key=lambda row: (row.source_wal_lsn, row.evidence_id.value)))
        if any(not start_lsn <= row.source_wal_lsn <= end_lsn for row in rows):
            raise ValueError("training evidence record lies outside materialized WAL range")
        payload = b"".join(_canonical(row.as_dict()) + b"\n" for row in rows)
        if len(payload) > self.max_segment_bytes:
            raise OverflowError("training evidence segment byte ceiling exceeded")
        checksum = hashlib.sha256(payload).hexdigest()
        segment_id = hashlib.sha256(_canonical([start_lsn, end_lsn, checksum])).hexdigest()
        name = f"segment-{start_lsn:020d}-{end_lsn:020d}-{segment_id[:12]}.jsonl"
        final = self.segments_root / name
        temporary = final.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if crash_hook:
            crash_hook("segment_fsynced")
        os.replace(temporary, final)
        if crash_hook:
            crash_hook("segment_renamed")
        segment_directory_fd = os.open(self.segments_root, os.O_RDONLY)
        try:
            os.fsync(segment_directory_fd)
        finally:
            os.close(segment_directory_fd)
        if crash_hook:
            crash_hook("segment_directory_fsynced")
        segment = TrainingEvidenceSegment(segment_id, f"segments/{name}", start_lsn, end_lsn, len(rows), len(payload), checksum)
        manifest = TrainingEvidenceManifest(
            self.manifest.segments + (segment,), end_lsn, self.manifest.generation + 1
        )
        manifest_payload = {
            "segments": [asdict(row) for row in manifest.segments],
            "hgt_checkpoint_lsn": manifest.hgt_checkpoint_lsn,
            "generation": manifest.generation,
            "checksum": manifest.checksum,
        }
        manifest_tmp = self.manifest_path.with_suffix(".tmp")
        with manifest_tmp.open("wb") as handle:
            handle.write(json.dumps(manifest_payload, indent=2, sort_keys=True).encode() + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        if crash_hook:
            crash_hook("manifest_fsynced")
        os.replace(manifest_tmp, self.manifest_path)
        directory_fd = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        self.manifest = manifest
        return manifest

    def records(self) -> tuple[TrainingEvidenceRecord, ...]:
        result = []
        for segment in self.manifest.segments:
            path = self.root / segment.path
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != segment.checksum:
                raise RuntimeError("training evidence segment checksum mismatch")
            result.extend(TrainingEvidenceRecord.from_dict(json.loads(line)) for line in payload.splitlines() if line)
        return tuple(result)

    def reclaim_orphans(self) -> tuple[Path, ...]:
        referenced = {str((self.root / segment.path).resolve()) for segment in self.manifest.segments}
        removed = []
        for path in self.segments_root.glob("segment-*.jsonl"):
            if str(path.resolve()) not in referenced:
                path.unlink()
                removed.append(path)
        for path in self.segments_root.glob("*.tmp"):
            path.unlink()
            removed.append(path)
        return tuple(removed)


__all__ = [
    "TrainingEvidenceId",
    "TrainingEvidenceKind",
    "TrainingEvidenceManifest",
    "TrainingEvidenceMaterializer",
    "TrainingEvidenceRecord",
    "TrainingEvidenceSegment",
]
