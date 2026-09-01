from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from v8.model import stable_u64


@dataclass(frozen=True, order=True, slots=True)
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
    deterministic_ordering_key: tuple[int, int]
    writes: tuple[StateWrite, ...]

    @classmethod
    def build(cls, mutation_kind: str, *, base_graph_generation: int, target_partition_ids: Iterable[int], read_set: Iterable[ReadDependency], evidence_refs: Iterable[int], causal_watermark: int, writes: Iterable[StateWrite]) -> "StateMutationProposal":
        partitions = tuple(sorted({int(x) for x in target_partition_ids}))
        reads = tuple(sorted(tuple(read_set), key=lambda x: x.ref))
        evidence = tuple(sorted({int(x) for x in evidence_refs}))
        write_rows = tuple(sorted(tuple(writes), key=lambda x: x.ref))
        uid = stable_u64(mutation_kind, int(causal_watermark), *(f"{d.ref.kind}:{d.ref.uid}:{d.version}" for d in reads), *(f"{w.ref.kind}:{w.ref.uid}:{w.value}" for w in write_rows), person=b"v9-mutation")
        return cls(uid, int(base_graph_generation), mutation_kind, partitions, reads, evidence, int(causal_watermark), (int(causal_watermark), int(uid)), write_rows)


@dataclass(frozen=True, slots=True)
class MutationResult:
    proposal_uid: int
    accepted: bool
    reason: str
    graph_generation: int


class VersionedMutationStore:
    STATE_VERSION = 1

    def __init__(self) -> None:
        self.graph_generation = 0
        self.object_versions: dict[ObjectRef, int] = {}
        self.values: dict[ObjectRef, int] = {}
        self.additive_counts: dict[ObjectRef, int] = {}

    def version(self, ref: ObjectRef) -> int:
        return int(self.object_versions.get(ref, 0))

    def read(self, ref: ObjectRef, default: int = 0) -> tuple[int, int]:
        return int(self.values.get(ref, default)), self.version(ref)

    def apply_additive(self, ref: ObjectRef, evidence_count: int = 1) -> None:
        amount = int(evidence_count)
        if amount < 0:
            raise ValueError("additive evidence cannot be negative")
        self.additive_counts[ref] = int(self.additive_counts.get(ref, 0)) + amount
        self.object_versions[ref] = self.version(ref) + 1
        self.graph_generation += 1

    def validate_read_set(self, read_set: Iterable[ReadDependency]) -> bool:
        return all(self.version(dep.ref) == int(dep.version) for dep in read_set)

    def apply_stateful(self, proposals: Iterable[StateMutationProposal]) -> tuple[MutationResult, ...]:
        results: list[MutationResult] = []
        for proposal in sorted(tuple(proposals), key=lambda x: x.deterministic_ordering_key):
            if not self.validate_read_set(proposal.read_set):
                results.append(MutationResult(proposal.proposal_uid, False, "STALE_READ_SET", self.graph_generation))
                continue
            if any(int(partition) < 0 for partition in proposal.target_partition_ids):
                results.append(MutationResult(proposal.proposal_uid, False, "INVALID_PARTITION", self.graph_generation))
                continue
            new_values = dict(self.values)
            new_versions = dict(self.object_versions)
            for write in proposal.writes:
                new_values[write.ref] = int(write.value)
                new_versions[write.ref] = int(new_versions.get(write.ref, 0)) + 1
            self.values = new_values
            self.object_versions = new_versions
            self.graph_generation += 1
            results.append(MutationResult(proposal.proposal_uid, True, "ACCEPTED", self.graph_generation))
        return tuple(results)

    def state_dict(self) -> dict[str, object]:
        refs = sorted(set(self.object_versions) | set(self.values) | set(self.additive_counts))
        return {"version": self.STATE_VERSION, "graph_generation": int(self.graph_generation), "objects": [{"kind": ref.kind, "uid": int(ref.uid), "object_version": self.version(ref), "value": int(self.values.get(ref, 0)), "additive_count": int(self.additive_counts.get(ref, 0))} for ref in refs]}

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "VersionedMutationStore":
        if int(state.get("version", 0)) != cls.STATE_VERSION:
            raise ValueError("unsupported versioned mutation state")
        store = cls()
        store.graph_generation = int(state.get("graph_generation", 0))
        rows = state.get("objects", [])
        if not isinstance(rows, list):
            raise ValueError("invalid versioned object list")
        for raw in rows:
            if not isinstance(raw, dict):
                continue
            ref = ObjectRef(str(raw["kind"]), int(raw["uid"]))
            store.object_versions[ref] = int(raw.get("object_version", 0))
            store.values[ref] = int(raw.get("value", 0))
            count = int(raw.get("additive_count", 0))
            if count:
                store.additive_counts[ref] = count
        return store
