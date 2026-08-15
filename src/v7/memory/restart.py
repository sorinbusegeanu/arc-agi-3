from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from v7.derivation.dependencies import DependencyMutation, MemoryDependencyGraph
from v7.memory.canonical import CanonicalMemoryKey, CanonicalMemoryRegistry
from v7.memory.durable_store import DurableGenerationStore
from v7.memory.generation import GenerationId
from v7.memory.ids import MemoryId, MemoryLevel
from v7.memory.indexes.cognition import ActionAggregateDelta, CognitionIndexBuilder, ContingencyIndexMutation, RoleConceptIndexMutation, RoleIndexMutation
from v7.memory.models import MemoryNode, MemoryScore
from v7.memory.read_view import MemoryReadView
from v7.memory.state import CognitiveState, GateId, GateValidationState
from v7.memory.writer import CanonicalMemoryWriter


@dataclass(frozen=True, slots=True)
class RestartSnapshot:
    generation_id: int
    payload: Mapping[str, Any]


class RuntimeSnapshotStore:
    """Persist and restore the complete writer-owned runtime frontier."""

    def __init__(self, durable_store: DurableGenerationStore) -> None:
        self.durable_store = durable_store
        connection = durable_store.connection
        connection.execute("CREATE TABLE IF NOT EXISTS runtime_snapshots (generation_id INTEGER PRIMARY KEY, payload_json TEXT NOT NULL)")
        connection.commit()

    def persist(self, writer: CanonicalMemoryWriter) -> RestartSnapshot:
        payload = export_writer_state(writer)
        generation_id = int(writer.published_view.generation_id)
        with self.durable_store.connection:
            self.durable_store.connection.execute("INSERT OR REPLACE INTO runtime_snapshots(generation_id,payload_json) VALUES (?,?)", (generation_id, json.dumps(payload, separators=(",", ":"), sort_keys=True)))
        return RestartSnapshot(generation_id, payload)

    def latest(self) -> RestartSnapshot | None:
        row = self.durable_store.connection.execute("SELECT generation_id,payload_json FROM runtime_snapshots ORDER BY generation_id DESC LIMIT 1").fetchone()
        return None if row is None else RestartSnapshot(int(row[0]), json.loads(row[1]))

    def restore(self) -> CanonicalMemoryWriter:
        snapshot = self.latest()
        return CanonicalMemoryWriter() if snapshot is None else restore_writer_state(snapshot.payload)


def export_writer_state(writer: CanonicalMemoryWriter) -> dict[str, Any]:
    view = writer.published_view
    registry = getattr(writer, "_canonical_registry")
    dependencies = getattr(writer, "_dependencies")
    edge_support = getattr(writer, "_edge_support")
    return {
        "generation": int(view.generation_id),
        "nodes": [
            [
                int(n.memory_id),
                int(n.level),
                n.type_id,
                int(n.created_generation),
                int(n.updated_generation),
                n.status_flags,
                n.support_count,
                int(n.cognitive_state),
                int(n.validation_state),
                int(n.gate_id),
            ]
            for n in view.nodes.values()
        ],
        "scores": [[int(s.memory_id), s.significance, s.prediction_error, s.learning_value, s.transfer_prior, s.explanatory_potential, s.future_option_delta] for s in view.scores.values()],
        "edges": [[int(source), relation, int(target), support] for (source, relation, target), support in edge_support.items()],
        "canonical": [[int(key.level), key.type_id, list(key.parts), int(memory_id)] for key, memory_id in registry._ids_by_key.items()],
        "dependencies": [[int(source), int(dependencies._levels[source]), int(target), int(dependencies._levels[target])] for source, targets in dependencies._upstream_to_dependents.items() for target in targets],
        "dirty": [int(v) for v in dependencies._dirty],
        "cognition": {
            "contingencies": [[c, a, [int(v) for v in values]] for (c, a), values in view.cognition_indexes.contingency_by_context_action.items()],
            "roles_exact": [[c, a, int(f), [int(v) for v in values]] for (c, a, f), values in view.cognition_indexes.role_by_context_action_family.items()],
            "roles": [[c, a, [int(v) for v in values]] for (c, a), values in view.cognition_indexes.role_by_context_action.items()],
            "concepts": [[int(role), [int(v) for v in values]] for role, values in view.cognition_indexes.concepts_by_role.items()],
            "aggregates": [[action, agg.future_option_sum, agg.future_option_count, agg.positive_count, agg.negative_count, agg.failure_count, agg.contradiction_count] for action, agg in view.cognition_indexes.action_aggregates.items()],
        },
    }


def _restore_node(row) -> MemoryNode:
    cognitive = int(row[7]) if len(row) > 7 else int(CognitiveState.ACTIVE)
    validation = int(row[8]) if len(row) > 8 else int(GateValidationState.VALIDATED)
    gate_id = int(row[9]) if len(row) > 9 else int(GateId.NONE)
    return MemoryNode(
        MemoryId(int(row[0])),
        MemoryLevel(int(row[1])),
        int(row[2]),
        GenerationId(int(row[3])),
        GenerationId(int(row[4])),
        int(row[5]),
        int(row[6]),
        cognitive,
        validation,
        gate_id,
    )


def restore_writer_state(payload: Mapping[str, Any]) -> CanonicalMemoryWriter:
    generation = int(payload.get("generation", 0))
    writer = CanonicalMemoryWriter(initial_generation=generation)
    nodes = {
        MemoryId(int(row[0])): _restore_node(row)
        for row in payload.get("nodes", ())
    }
    scores = {MemoryId(int(r[0])): MemoryScore(MemoryId(int(r[0])), *(float(v) for v in r[1:7])) for r in payload.get("scores", ())}
    edge_support = {(MemoryId(int(r[0])), int(r[1]), MemoryId(int(r[2]))): int(r[3]) for r in payload.get("edges", ())}
    adjacency: dict[tuple[MemoryId, int], list[MemoryId]] = {}
    for (source, relation, target), support in edge_support.items():
        if support > 0:
            adjacency.setdefault((source, relation), []).append(target)
    setattr(writer, "_nodes", nodes)
    setattr(writer, "_scores", scores)
    setattr(writer, "_edge_support", edge_support)

    registry = CanonicalMemoryRegistry()
    for level, type_id, parts, memory_id in payload.get("canonical", ()):
        registry.bind_existing(CanonicalMemoryKey(MemoryLevel(int(level)), int(type_id), tuple(int(v) for v in parts)), MemoryId(int(memory_id)))
    for memory_id in nodes:
        registry.observe_existing_id(memory_id)
    setattr(writer, "_canonical_registry", registry)

    graph = MemoryDependencyGraph()
    for memory_id, node in nodes.items():
        graph.register_node(memory_id, node.level)
    deps = [DependencyMutation(MemoryId(int(r[0])), MemoryLevel(int(r[1])), MemoryId(int(r[2])), MemoryLevel(int(r[3]))) for r in payload.get("dependencies", ())]
    if deps:
        graph.apply_dependency_batch(deps)
    dirty = [MemoryId(int(v)) for v in payload.get("dirty", ())]
    if dirty:
        graph.mark_dirty(dirty)
    setattr(writer, "_dependencies", graph)

    builder = CognitionIndexBuilder()
    raw = payload.get("cognition", {})
    builder.apply_contingency_batch(ContingencyIndexMutation(int(c), int(a), MemoryId(int(v))) for c, a, values in raw.get("contingencies", ()) for v in values)
    builder.apply_role_batch(RoleIndexMutation(int(c), int(a), MemoryId(int(v)), MemoryId(int(f))) for c, a, f, values in raw.get("roles_exact", ()) for v in values)
    builder.apply_role_batch(RoleIndexMutation(int(c), int(a), MemoryId(int(v)), None) for c, a, values in raw.get("roles", ()) for v in values)
    builder.apply_role_concept_batch(RoleConceptIndexMutation(MemoryId(int(role)), MemoryId(int(v))) for role, values in raw.get("concepts", ()) for v in values)
    builder.apply_action_aggregate_batch(ActionAggregateDelta(int(r[0]), float(r[1]), int(r[2]), int(r[3]), int(r[4]), int(r[5]), int(r[6])) for r in raw.get("aggregates", ()))
    setattr(writer, "_cognition_indexes", builder)
    setattr(
        writer,
        "_published_view",
        MemoryReadView.freeze(
            generation_id=GenerationId(generation),
            nodes=nodes,
            scores=scores,
            adjacency=adjacency,
            cognition_indexes=builder.freeze(),
        ),
    )
    return writer
