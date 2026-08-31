from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from v8.model import stable_u64


@dataclass(frozen=True, slots=True, order=True)
class ObjectRef:
    kind: str
    uid: int


@dataclass(frozen=True, slots=True)
class ReadDependency:
    ref: ObjectRef
    version: int


@dataclass(frozen=True, slots=True)
class StateWrite:
    ref: ObjectRef
    value: int


@dataclass(frozen=True, slots=True)
class StateMutationProposal:
    proposal_uid: int
    base_graph_generation: int
    mutation_kind: str
    target_partition_ids: tuple[int, ...]
    read_set: tuple[ReadDependency, ...]
    evidence_refs: tuple[int, ...]
    causal_watermark: int
    deterministic_ordering_key: tuple[int, int, int]
    writes: tuple[StateWrite, ...]

    @classmethod
    def build(
        cls,
        mutation_kind: str,
        *,
        base_graph_generation: int,
        target_partition_ids: Iterable[int],
        read_set: Iterable[ReadDependency],
        evidence_refs: Iterable[int],
        causal_watermark: int,
        writes: Iterable[StateWrite],
        event_type_priority: int = 0,
    ) -> "StateMutationProposal":
        partitions = tuple(sorted({int(v) for v in target_partition_ids}))
        reads = tuple(sorted(tuple(read_set), key=lambda row: row.ref))
        evidence = tuple(sorted({int(v) for v in evidence_refs}))
        write_rows = tuple(sorted(tuple(writes), key=lambda row: row.ref))
        uid = stable_u64(
            str(mutation_kind), int(causal_watermark), int(event_type_priority),
            *(f"r:{r.ref.kind}:{r.ref.uid}:{r.version}" for r in reads),
            *(f"w:{w.ref.kind}:{w.ref.uid}:{w.value}" for w in write_rows),
            person=b"v9-state-mutation",
        )
        return cls(
            uid, int(base_graph_generation), str(mutation_kind), partitions, reads, evidence,
            int(causal_watermark), (int(causal_watermark), int(event_type_priority), int(uid)), write_rows,
        )


@dataclass(frozen=True, slots=True)
class MutationResult:
    proposal_uid: int
    accepted: bool
    reason: str
    graph_generation: int


class VersionedMutationStore:
    """Dependency-local version authority for non-commutative v9 state."""

    STATE_VERSION = 1

    def __init__(self) -> None:
        self.graph_generation = 0
        self.object_versions: dict[ObjectRef, int] = {}
        self.values: dict[ObjectRef, int] = {}
        self.proposal_count = 0
        self.stateful_proposal_count = 0
        self.stale_proposal_count = 0
        self.read_set_conflict_count = 0
        self.cross_partition_transaction_count = 0

    def version(self, ref: ObjectRef) -> int:
        return int(self.object_versions.get(ref, 0))

    def read(self, ref: ObjectRef, default: int = 0) -> tuple[int, int]:
        return int(self.values.get(ref, default)), self.version(ref)

    def validate_read_set(self, read_set: Iterable[ReadDependency]) -> bool:
        return all(self.version(row.ref) == int(row.version) for row in read_set)

    def apply_stateful(self, proposals: Iterable[StateMutationProposal]) -> tuple[MutationResult, ...]:
        output: list[MutationResult] = []
        for proposal in sorted(tuple(proposals), key=lambda row: row.deterministic_ordering_key):
            self.proposal_count += 1
            self.stateful_proposal_count += 1
            if len(proposal.target_partition_ids) > 1:
                self.cross_partition_transaction_count += 1
            if any(part < 0 for part in proposal.target_partition_ids):
                output.append(MutationResult(proposal.proposal_uid, False, "INVALID_PARTITION", self.graph_generation))
                continue
            if not self.validate_read_set(proposal.read_set):
                self.stale_proposal_count += 1
                self.read_set_conflict_count += 1
                output.append(MutationResult(proposal.proposal_uid, False, "STALE_READ_SET", self.graph_generation))
                continue
            # Prepare all writes first, then swap both dictionaries. This is the
            # all-or-nothing boundary for cross-partition auxiliary state.
            new_values = dict(self.values)
            new_versions = dict(self.object_versions)
            for write in proposal.writes:
                new_values[write.ref] = int(write.value)
                new_versions[write.ref] = int(new_versions.get(write.ref, 0)) + 1
            self.values = new_values
            self.object_versions = new_versions
            self.graph_generation += 1
            output.append(MutationResult(proposal.proposal_uid, True, "ACCEPTED", self.graph_generation))
        return tuple(output)

    def telemetry(self) -> dict[str, int]:
        return {
            "proposal_count": self.proposal_count,
            "stateful_proposal_count": self.stateful_proposal_count,
            "stale_proposal_count": self.stale_proposal_count,
            "read_set_conflict_count": self.read_set_conflict_count,
            "cross_partition_transaction_count": self.cross_partition_transaction_count,
            "graph_generation": self.graph_generation,
        }

    def state_dict(self) -> dict[str, object]:
        rows = []
        for ref in sorted(set(self.object_versions) | set(self.values)):
            rows.append({
                "kind": ref.kind, "uid": ref.uid, "version": self.version(ref),
                "value": int(self.values.get(ref, 0)),
            })
        return {
            "version": self.STATE_VERSION,
            "graph_generation": self.graph_generation,
            "objects": rows,
            "telemetry": self.telemetry(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "VersionedMutationStore":
        if int(state.get("version", 0)) != cls.STATE_VERSION:
            raise ValueError("unsupported versioned mutation state")
        obj = cls()
        obj.graph_generation = int(state.get("graph_generation", 0))
        for raw in state.get("objects", []):
            if not isinstance(raw, dict):
                continue
            ref = ObjectRef(str(raw["kind"]), int(raw["uid"]))
            obj.object_versions[ref] = int(raw.get("version", 0))
            obj.values[ref] = int(raw.get("value", 0))
        telemetry = state.get("telemetry", {})
        if isinstance(telemetry, dict):
            for name in (
                "proposal_count", "stateful_proposal_count", "stale_proposal_count",
                "read_set_conflict_count", "cross_partition_transaction_count",
            ):
                setattr(obj, name, int(telemetry.get(name, 0)))
        return obj
