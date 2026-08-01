"""Immutable, process-local memory indexes for sampling workers.

The compact-memory SQLite files remain the source of truth.  This module only
materializes the read-side indexes needed by the hot action-selection path.
"""

from __future__ import annotations

import json
import pickle
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
from hashlib import sha1

from v6.memory.query_engine import MemoryActionScore, MemoryPrediction
from v6.memory.substrate import action_node_id
from v6.contingency.contingency_learner import Contingency
from v6.memory.compact_memory import stable_family_int_id
from v6.memory.compact_memory_restore import _context_signature_from_canonical_key


@dataclass(frozen=True)
class WorkerMemorySnapshot:
    version: int
    version_metadata: Mapping[str, Any]
    stable_contingencies_by_context_action: Mapping[tuple[tuple, int], tuple[dict, ...]]
    family_scores_by_context_action: Mapping[tuple[tuple, int], tuple[dict, ...]]
    replay_candidates_by_context_action: Mapping[tuple[tuple, int], tuple[dict, ...]]
    carrier_candidates_by_context: Mapping[str, tuple[dict, ...]]
    future_option_scores_by_context_action: Mapping[tuple[str, int], tuple[float, ...]]
    graph_adjacency: Mapping[str, tuple[dict, ...]]
    substrate_nodes: Mapping[str, dict]
    substrate_scores: Mapping[str, dict]
    substrate_edges_from: Mapping[tuple[str, str | None], tuple[dict, ...]]
    substrate_edges_to: Mapping[tuple[str, str | None], tuple[dict, ...]]
    snapshot_bytes: int
    restore_seconds: float
    graph_restore_seconds: float
    substrate_restore_seconds: float
    source_memory_dir: str | None = None

    @classmethod
    def from_system(
        cls,
        system: Any,
        *,
        memory_dir: str | Path | None = None,
        restore_seconds: float = 0.0,
        graph_restore_seconds: float = 0.0,
        substrate_restore_seconds: float = 0.0,
        version_metadata: Mapping[str, Any] | None = None,
    ) -> "WorkerMemorySnapshot":
        contingencies: dict[tuple[tuple, int], list[dict]] = {}
        learner = getattr(system, "contingency_learner", None)
        for contingency in (learner.stable_contingencies() if learner is not None else []):
            row = {
                "node_id": f"M1:contingency:{int(contingency.id)}",
                "family": int(contingency.transformation_family),
                "confidence": float(contingency.confidence),
                "context_level": int(contingency.context_level),
                "context_signature": tuple(contingency.context_signature),
                "action": int(contingency.action),
                "support_count": int(contingency.support_count),
            }
            contingencies.setdefault((tuple(contingency.context_signature), int(contingency.action)), []).append(row)
        for rows in contingencies.values():
            rows.sort(key=lambda item: (str(item["node_id"])))

        memory = getattr(system, "memory", None)
        nodes: dict[str, dict] = {}
        scores: dict[str, dict] = {}
        edges_from: dict[tuple[str, str | None], list[dict]] = {}
        edges_to: dict[tuple[str, str | None], list[dict]] = {}
        if memory is not None:
            for row in memory.query_nodes():
                nodes[str(row["node_id"])] = dict(row)
            connection = getattr(memory, "connection", None)
            if connection is not None:
                for row in connection.execute(
                    "SELECT node_id, future_option_delta, replay_priority, transfer_score FROM memory_scores"
                ).fetchall():
                    scores[str(row[0])] = {
                        "future_option_delta": None if row[1] is None else float(row[1]),
                        "replay_priority": None if row[2] is None else float(row[2]),
                        "transfer_score": None if row[3] is None else float(row[3]),
                    }
                edge_rows = connection.execute(
                    "SELECT source_node_id, target_node_id, edge_type, weight, support_count, evidence_json FROM memory_edges ORDER BY source_node_id, target_node_id, edge_type"
                ).fetchall()
                for row in edge_rows:
                    evidence = {}
                    try:
                        evidence = json.loads(str(row[5] or "{}"))
                    except (TypeError, ValueError, json.JSONDecodeError):
                        pass
                    edge = {
                        "source_node_id": str(row[0]), "target_node_id": str(row[1]), "edge_type": str(row[2]),
                        "weight": float(row[3] or 1.0), "support_count": int(row[4] or 0), "evidence": evidence,
                    }
                    edges_from.setdefault((str(row[0]), str(row[2])), []).append(edge)
                    edges_from.setdefault((str(row[0]), None), []).append(edge)
                    edges_to.setdefault((str(row[1]), str(row[2])), []).append(edge)
                    edges_to.setdefault((str(row[1]), None), []).append(edge)

        future_by_action: dict[tuple[str, int], list[float]] = {}
        for (target, edge_type), rows in edges_to.items():
            if edge_type != "takes_action":
                continue
            try:
                action = int(str(target).rsplit(":", 1)[-1])
            except ValueError:
                continue
            for edge in rows:
                score = scores.get(str(edge["source_node_id"]), {}).get("future_option_delta")
                if score is not None:
                    future_by_action.setdefault(("", action), []).append(float(score))

        graph_adjacency: dict[str, list[dict]] = {}
        graph = getattr(system, "graph", None)
        if graph is not None and hasattr(graph, "export_compact_rows"):
            exported = graph.export_compact_rows()
            for edge in exported.get("edges", []):
                graph_adjacency.setdefault(str(edge["source_node_id"]), []).append(dict(edge))

        metadata = dict(version_metadata or {})
        metadata.setdefault("schema_version", "worker-memory-snapshot-v1")
        metadata.setdefault("created_at", time.time())
        payload = {
            "contingencies": contingencies,
            "scores": scores,
            "edges_from": edges_from,
            "edges_to": edges_to,
            "nodes": nodes,
            "graph": graph_adjacency,
        }
        snapshot_bytes = len(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
        return cls(
            version=1,
            version_metadata=MappingProxyType(metadata),
            stable_contingencies_by_context_action=MappingProxyType({key: tuple(value) for key, value in contingencies.items()}),
            family_scores_by_context_action=MappingProxyType({}),
            replay_candidates_by_context_action=MappingProxyType({}),
            carrier_candidates_by_context=MappingProxyType({}),
            future_option_scores_by_context_action=MappingProxyType({key: tuple(value) for key, value in future_by_action.items()}),
            graph_adjacency=MappingProxyType({key: tuple(value) for key, value in graph_adjacency.items()}),
            substrate_nodes=MappingProxyType(nodes),
            substrate_scores=MappingProxyType(scores),
            substrate_edges_from=MappingProxyType({key: tuple(value) for key, value in edges_from.items()}),
            substrate_edges_to=MappingProxyType({key: tuple(value) for key, value in edges_to.items()}),
            snapshot_bytes=int(snapshot_bytes),
            restore_seconds=float(restore_seconds),
            graph_restore_seconds=float(graph_restore_seconds),
            substrate_restore_seconds=float(substrate_restore_seconds),
            source_memory_dir=None if memory_dir is None else str(memory_dir),
        )


def build_worker_memory_snapshot_from_directory(
    memory_dir: str | Path,
    *,
    include_graph: bool = True,
    include_substrate: bool = True,
    version_metadata: Mapping[str, Any] | None = None,
) -> WorkerMemorySnapshot:
    """Build a snapshot with bulk read-only SQLite scans, without V6System restore."""

    root = Path(memory_dir)
    started = time.perf_counter()
    state_path = root / "current_state.sqlite"
    graph_path = root / "graph.sqlite"
    state = sqlite3.connect(f"file:{state_path.resolve()}?mode=ro", uri=True, timeout=10.0)
    state.row_factory = sqlite3.Row

    class _Learner:
        def stable_contingencies(self) -> list[Contingency]:
            family_map = {
                str(row["canonical_signature"]): int(row["stable_family_id"])
                for row in state.execute("SELECT canonical_signature, stable_family_id FROM family_identity_map").fetchall()
            }
            output = []
            for index, row in enumerate(state.execute("SELECT canonical_key, context_level, action, effect_signature, support_count, stability_score FROM stable_contingencies").fetchall(), 1):
                effect = str(row["effect_signature"])
                output.append(Contingency(
                    id=index,
                    context_level=int(row["context_level"] or 0),
                    context_signature=_context_signature_from_canonical_key(str(row["canonical_key"])),
                    action=int(row["action"]),
                    transformation_family=int(family_map.get(effect, stable_family_int_id(effect))),
                    support_count=int(row["support_count"] or 0),
                    confidence=float(row["stability_score"] or 0.0),
                ))
            return output

    class _Memory:
        connection = state

        def query_nodes(self) -> list[dict]:
            if not include_substrate:
                return []
            rows = state.execute("SELECT node_id, memory_level, node_type, canonical_key, attrs_json FROM memory_nodes ORDER BY node_id").fetchall()
            output = []
            for row in rows:
                try:
                    attrs = json.loads(str(row[4] or "{}"))
                except (TypeError, ValueError, json.JSONDecodeError):
                    attrs = {}
                output.append({"node_id": str(row[0]), "memory_level": str(row[1]), "node_type": str(row[2]), "canonical_key": row[3], "attrs": attrs})
            return output

    class _Graph:
        def export_compact_rows(self) -> dict[str, list[dict]]:
            if not include_graph or not graph_path.exists():
                return {"nodes": [], "edges": []}
            graph = sqlite3.connect(f"file:{graph_path.resolve()}?mode=ro", uri=True, timeout=10.0)
            try:
                nodes = [{"node_id": str(row[0]), "node_type": str(row[1] or "Unknown"), "canonical_key": row[2], "support_count": int(row[3] or 0)} for row in graph.execute("SELECT node_id, node_type, canonical_key, support_count FROM graph_nodes").fetchall()]
                edges = [{"source_node_id": str(row[0]), "target_node_id": str(row[1]), "edge_type": str(row[2]), "weight": float(row[3] or 1.0), "support_count": int(row[4] or 0)} for row in graph.execute("SELECT source_node_id, target_node_id, edge_type, weight, support_count FROM graph_edges").fetchall()]
                return {"nodes": nodes, "edges": edges}
            finally:
                graph.close()

    class _System:
        contingency_learner = _Learner()
        memory = _Memory()
        graph = _Graph()

    try:
        snapshot = WorkerMemorySnapshot.from_system(
            _System(),
            memory_dir=root,
            restore_seconds=0.0,
            graph_restore_seconds=0.0,
            substrate_restore_seconds=0.0,
            version_metadata=version_metadata,
        )
        return WorkerMemorySnapshot(
            **{
                **snapshot.__dict__,
                "restore_seconds": time.perf_counter() - started,
            }
        )
    finally:
        state.close()


@dataclass
class WorkerMemoryOverlay:
    contingencies: dict[tuple[tuple, int], dict] = field(default_factory=dict)
    replay_candidates: dict[str, dict] = field(default_factory=dict)
    future_option_events: dict[str, dict] = field(default_factory=dict)
    graph_edges: dict[tuple[str, str, str], dict] = field(default_factory=dict)
    last_sequence: int = 0

    def apply_rows(self, rows: list[dict[str, Any]], sequence: int) -> int:
        applied = 0
        for row in rows:
            payload = dict(row.get("payload") or row)
            event_type = str(row.get("event_type") or payload.get("event_type") or "")
            if event_type == "stable_contingency":
                try:
                    key = (tuple(json.loads(str(payload.get("context_signature") or "[]"))), int(payload.get("action") or 0))
                    self.contingencies[key] = payload
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
            elif event_type == "high_priority_replay":
                self.replay_candidates[str(payload.get("interaction_id") or row.get("event_id"))] = payload
            elif event_type in {"future_option", "future_option_event"}:
                self.future_option_events[str(row.get("event_id") or payload.get("event_id"))] = payload
            elif event_type == "graph_edge":
                key = (str(payload.get("source_node_id")), str(payload.get("target_node_id")), str(payload.get("edge_type")))
                self.graph_edges[key] = payload
            applied += 1
        self.last_sequence = max(int(self.last_sequence), int(sequence))
        return applied


class SnapshotMemoryQueryEngine:
    """MemoryQueryEngine-compatible read path backed only by worker RAM."""

    def __init__(self, snapshot: WorkerMemorySnapshot) -> None:
        self.snapshot = snapshot
        self.overlay = WorkerMemoryOverlay()
        self.memory_query_count = 0
        self.memory_query_seconds = 0.0
        self.memory_action_rank_count = 0
        self.memory_action_rank_seconds = 0.0
        self.sqlite_queries_during_action_selection = 0
        self.cache_hit_count = 0
        self.cache_miss_count = 0

    def _best_context_signature(self, context_signatures: dict[int, tuple], action: int) -> tuple:
        if not context_signatures:
            return (int(action),)
        max_level = max(int(level) for level in context_signatures)
        return tuple(context_signatures.get(max_level, next(iter(context_signatures.values()))))

    def _contingency(self, contexts: dict[int, tuple], action: int) -> dict | None:
        for level in sorted(contexts, reverse=True):
            key = (tuple(contexts[level]), int(action))
            if key in self.overlay.contingencies:
                self.cache_hit_count += 1
                return self.overlay.contingencies[key]
            candidates = self.snapshot.stable_contingencies_by_context_action.get(key, ())
            if candidates:
                self.cache_hit_count += 1
                return max(candidates, key=lambda item: (float(item.get("confidence", 0.0)), int(item.get("support_count", 0)), -int(item.get("family", 0))))
        self.cache_miss_count += 1
        return None

    def predict_family(self, context_signatures: dict[int, tuple], action: int, *, record_query: bool = False) -> MemoryPrediction:
        started = time.perf_counter()
        self.memory_query_count += 1
        exact = self._contingency(context_signatures, action)
        if exact is None:
            result = MemoryPrediction(None, 0.0, "none", [])
        else:
            result = MemoryPrediction(
                predicted_family=int(exact.get("family", exact.get("transformation_family", 0))),
                confidence=float(exact.get("confidence", 0.0) or 0.0),
                source="memory_contingency",
                evidence_node_ids=[str(exact.get("node_id", ""))],
            )
        self.memory_query_seconds += time.perf_counter() - started
        return result

    def _future_and_failure(self, context: tuple, action: int) -> tuple[dict, dict]:
        deltas = list(self.snapshot.future_option_scores_by_context_action.get(("", int(action)), ()))
        for payload in self.overlay.future_option_events.values():
            if payload.get("action") is None or int(payload.get("action")) == int(action):
                if payload.get("option_delta") is not None:
                    deltas.append(float(payload["option_delta"]))
        positive = sum(1 for value in deltas if value > 0.0)
        future = {
            "expected_future_option_delta": sum(deltas) / len(deltas) if deltas else 0.0,
            "completion_likelihood": positive / len(deltas) if deltas else 0.0,
            "sources": ["future_option_memory"] if deltas else [],
        }
        contradiction = any(
            self._edges_from(str(edge.get("source_node_id")), "violates_prediction")
            for edge in self._edges_to(action_node_id(action), "takes_action")
        )
        failure = {
            "failure_risk": sum(1 for value in deltas if value < 0.0) / len(deltas) if deltas else 0.0,
            "contradiction_evidence": contradiction,
            "sources": ["failure_path_memory"] if deltas else [],
        }
        return future, failure

    def _edges_from(self, node_id: str, edge_type: str | None = None) -> tuple[dict, ...]:
        return self.snapshot.substrate_edges_from.get((str(node_id), edge_type), ())

    def _edges_to(self, node_id: str, edge_type: str | None = None) -> tuple[dict, ...]:
        return self.snapshot.substrate_edges_to.get((str(node_id), edge_type), ())

    def find_similar_roles(self, context_signature: str, action: int) -> list[dict]:
        context_node_id = "M0:context:" + sha1(str(context_signature).encode("utf-8")).hexdigest()[:20]
        target_action_node = action_node_id(action)
        matches: list[dict] = []
        for role in self.snapshot.substrate_nodes.values():
            if str(role.get("node_type")) != "FunctionalRoleMemory":
                continue
            attrs = dict(role.get("attrs") or {})
            score = float(attrs.get("transfer_score", 0.0) or 0.0)
            action_match = False
            context_match = False
            family_ids: list[int] = []
            for carrier_edge in self._edges_to(str(role.get("node_id")), "plays_role"):
                carrier_id = str(carrier_edge.get("source_node_id"))
                for family_edge in self._edges_from(carrier_id, "associated_with_family"):
                    family_id = str(family_edge.get("target_node_id"))
                    if family_id.startswith("M2:family:"):
                        try:
                            family_ids.append(int(family_id.rsplit(":", 1)[-1]))
                        except ValueError:
                            pass
                context_match = context_match or any(str(item.get("target_node_id")) == context_node_id for item in self._edges_from(carrier_id, "appears_in_context"))
                for interaction_edge in self._edges_from(carrier_id, "carried_by"):
                    action_match = action_match or any(str(item.get("target_node_id")) == target_action_node for item in self._edges_from(str(interaction_edge.get("target_node_id")), "takes_action"))
            if action_match:
                score += 0.25
            if context_match:
                score += 0.25
            if not action_match and not context_match:
                score *= 0.25
            score = max(0.0, min(1.0, score))
            if score > 0.0:
                matches.append({"node_id": role.get("node_id"), "score": score, "family_id": family_ids[0] if family_ids else None, "action_match": action_match, "context_match": context_match})
        return sorted(matches, key=lambda item: (-float(item["score"]), str(item["node_id"])))

    def find_concept_matches(self, context_signature: str, action: int) -> list[dict]:
        matches: list[dict] = []
        role_matches = self.find_similar_roles(context_signature, action)
        role_by_id = {str(item["node_id"]): item for item in role_matches}
        for concept in self.snapshot.substrate_nodes.values():
            if str(concept.get("node_type")) != "ConceptMemory":
                continue
            best_role_score = 0.0
            family_ids: list[int] = []
            for role_edge in self._edges_to(str(concept.get("node_id")), "transfers_to"):
                role_id = str(role_edge.get("source_node_id"))
                role_match = role_by_id.get(role_id)
                if role_match and role_match.get("action_match"):
                    best_role_score = max(best_role_score, float(role_match.get("score", 0.0)))
                for carrier_edge in self._edges_from(role_id, "abstracts_from"):
                    for family_edge in self._edges_from(str(carrier_edge.get("target_node_id")), "associated_with_family"):
                        family_id = str(family_edge.get("target_node_id"))
                        if family_id.startswith("M2:family:"):
                            try:
                                family_ids.append(int(family_id.rsplit(":", 1)[-1]))
                            except ValueError:
                                pass
            transfer_count = float(dict(concept.get("attrs") or {}).get("transfer_success_count", 0) or 0) / 3.0
            score = max(0.0, min(1.0, best_role_score * min(1.0, transfer_count)))
            if score > 0.0:
                matches.append({"node_id": concept.get("node_id"), "score": score, "family_id": family_ids[0] if family_ids else None})
        return sorted(matches, key=lambda item: (-float(item["score"]), str(item["node_id"])))

    def score_action(self, context_signatures: dict[int, tuple], action: int, available_actions: list[int], *, record_query: bool = False) -> MemoryActionScore:
        del available_actions, record_query
        started = time.perf_counter()
        prediction = self.predict_family(context_signatures, action)
        future, failure = self._future_and_failure(self._best_context_signature(context_signatures, action), action)
        best_context = json.dumps(list(self._best_context_signature(context_signatures, action)))
        role_matches = self.find_similar_roles(best_context, action)
        concept_matches = self.find_concept_matches(best_context, action)
        transfer_score = max(
            float(role_matches[0].get("score", 0.0)) if role_matches else 0.0,
            float(concept_matches[0].get("score", 0.0)) if concept_matches else 0.0,
        )
        contradiction_risk = 1.0 if failure.get("contradiction_evidence") else 0.0
        score = max(0.0, min(1.0, 0.30 * prediction.confidence + 0.25 * max(0.0, future["expected_future_option_delta"]) + 0.20 * future["completion_likelihood"] + 0.15 * transfer_score - 0.25 * failure["failure_risk"] - 0.10 * contradiction_risk))
        self.memory_query_seconds += time.perf_counter() - started
        return MemoryActionScore(
            action=int(action), score=score, predicted_family=prediction.predicted_family,
            expected_future_option_delta=float(future["expected_future_option_delta"]),
            failure_risk=float(failure["failure_risk"]), completion_likelihood=float(future["completion_likelihood"]),
            evidence_sources=[prediction.source, *future["sources"], *failure["sources"]],
        )

    def rank_actions(self, context_signatures_by_action: dict[int, dict[int, tuple]], available_actions: list[int]) -> list[MemoryActionScore]:
        started = time.perf_counter()
        self.memory_action_rank_count += 1
        scores = [self.score_action(context_signatures_by_action[int(action)], int(action), available_actions)
                  for action in sorted(int(item) for item in available_actions)
                  if int(action) in context_signatures_by_action]
        self.memory_action_rank_seconds += time.perf_counter() - started
        return sorted(scores, key=lambda item: (-float(item.score), int(item.action)))

    def rank_actions_with_shared_context(self, context_signatures: dict[int, tuple], available_actions: list[int]) -> list[MemoryActionScore]:
        return self.rank_actions({int(action): context_signatures for action in available_actions}, available_actions)

    def record_selected_action_query(self, *, context_signatures: dict[int, tuple], action: int, prediction: MemoryPrediction | None = None) -> None:
        del context_signatures, action, prediction

    def apply_live_overlay(self, overlay: WorkerMemoryOverlay) -> None:
        self.overlay = overlay

    def metrics(self) -> dict[str, Any]:
        return {
            "memory_query_count": int(self.memory_query_count),
            "memory_query_seconds": float(self.memory_query_seconds),
            "memory_query_mean_seconds": float(self.memory_query_seconds / self.memory_query_count) if self.memory_query_count else 0.0,
            "memory_action_rank_count": int(self.memory_action_rank_count),
            "memory_action_rank_seconds": float(self.memory_action_rank_seconds),
            "memory_cache_hit_count": int(self.cache_hit_count),
            "memory_cache_miss_count": int(self.cache_miss_count),
            "sqlite_queries_during_action_selection": int(self.sqlite_queries_during_action_selection),
            "last_applied_live_sequence": int(self.overlay.last_sequence),
        }
