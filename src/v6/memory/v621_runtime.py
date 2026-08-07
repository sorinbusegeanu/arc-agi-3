from __future__ import annotations

import io
import json
import math
import sqlite3
import time
from collections import defaultdict, deque
from dataclasses import replace
from hashlib import sha1
from pathlib import Path
from typing import Any

import numpy as np

from v6.future_options import FutureOptionDelta, FutureOptionEstimator, FutureOptionSet
from v6.memory.controller import MemoryController
from v6.memory.interaction_store import decode_array
from v6.memory.migrations.v621 import migrate_connection as migrate_v621
from v6.memory.promotion_engine import MemoryPromotionEngine
from v6.memory.query_engine import MemoryActionScore, MemoryPrediction, compute_memory_action_score
from v6.memory.substrate import (
    MemoryEdge,
    MemoryNode,
    MemoryScore,
    MemorySubstrate,
    concept_node_id,
    world_model_node_id,
)
from v6.memory.v62_runtime import (
    HierarchicalSignificanceEngine,
    V62MemoryQueryEngine,
)

POLICY_VERSION = "v621_evidence_policy_v1"
LIFECYCLE_VERSION = "v621_hierarchical_lifecycle_v1"
ABSTRACTION_VERSION = "v621_relational_abstraction_v1"


def _clamp01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return max(0.0, min(1.0, number))


def _json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _table_columns(
    connection: sqlite3.Connection,
    table: str,
) -> set[str]:
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    if exists is None:
        return set()
    return {
        str(row[1])
        for row in connection.execute(
            f'PRAGMA table_info("{table}")'
        ).fetchall()
    }


def _node_level(node_id: str) -> str:
    text = str(node_id)
    return text.split(":", 1)[0] if ":" in text else ""


class CachedAbstractionFutureOptionEstimator:
    """Incremental exact + structural reachability with no full DB rescan per step."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        fallback: FutureOptionEstimator | None = None,
        refresh_interval: int = 64,
    ) -> None:
        self.connection = connection
        self.fallback = fallback or FutureOptionEstimator()
        self.refresh_interval = max(1, int(refresh_interval))
        self._last_interaction_id = 0
        self._last_refresh_count = -1
        self._exact_adjacency: dict[str, list[tuple[int, str, bool]]] = defaultdict(list)
        self._struct_adjacency: dict[str, list[tuple[int, str, bool]]] = defaultdict(list)
        self._state_structural: dict[str, str] = {}
        self._transition_rows: list[tuple[str, int, str, bool]] = []
        self._seen_exact: set[tuple[str, int, str, bool]] = set()
        self.refresh_count = 0
        self.rows_loaded = 0

    def estimate_option_set(
        self,
        env_or_state: Any,
        *,
        depth: int = 1,
        available_actions: list[int] | tuple[int, ...] | None = None,
    ) -> FutureOptionSet:
        self._refresh_if_needed()
        depth = max(1, int(depth))
        actions = tuple(
            sorted(int(item) for item in (available_actions or ()))
        )
        state_hash = self._state_hash(env_or_state)
        structural = self._structural_signature(env_or_state)

        exact = self._reachable(
            self._exact_adjacency,
            state_hash,
            depth,
            prefix="exact",
        )
        abstract = self._reachable(
            self._struct_adjacency,
            structural,
            depth,
            prefix="struct",
        )
        reachable = exact | abstract

        if not reachable:
            return self.fallback.estimate_option_set(
                env_or_state,
                depth=depth,
                available_actions=list(actions),
            )

        first_actions = {
            int(item[0])
            for item in self._exact_adjacency.get(state_hash, ())
        }
        first_actions.update(
            int(item[0])
            for item in self._struct_adjacency.get(structural, ())
        )
        resolved_actions = tuple(
            sorted(first_actions or set(actions))
        )
        option_set_id = "fos:v621:" + sha1(
            _json(
                {
                    "state": state_hash,
                    "structural": structural,
                    "depth": depth,
                    "reachable": sorted(reachable),
                }
            ).encode("utf-8")
        ).hexdigest()[:20]
        return FutureOptionSet(
            option_set_id=option_set_id,
            state_signature=state_hash,
            available_actions=resolved_actions,
            reachable_signatures=tuple(sorted(reachable)),
            estimated_branching_factor=len(resolved_actions),
            depth=depth,
        )

    def compare(
        self,
        before: FutureOptionSet,
        after: FutureOptionSet,
        interaction_id: int,
    ) -> FutureOptionDelta:
        before_set = set(before.reachable_signatures)
        after_set = set(after.reachable_signatures)
        added = tuple(sorted(after_set - before_set))
        removed = tuple(sorted(before_set - after_set))
        preserved = tuple(sorted(before_set & after_set))
        return FutureOptionDelta(
            interaction_id=int(interaction_id),
            before_option_set_id=before.option_set_id,
            after_option_set_id=after.option_set_id,
            added_options=added,
            removed_options=removed,
            preserved_options=preserved,
            delta_score=float(len(added) - len(removed)),
        )

    def force_refresh(self) -> None:
        self._refresh_incremental()

    def _refresh_if_needed(self) -> None:
        columns = _table_columns(self.connection, "interactions")
        if "id" not in columns:
            return
        row = self.connection.execute(
            "SELECT COALESCE(MAX(id), 0) FROM interactions"
        ).fetchone()
        current = int(row[0] or 0)
        if current <= self._last_interaction_id:
            return
        if (
            self._last_refresh_count >= 0
            and current - self._last_interaction_id < self.refresh_interval
        ):
            return
        self._refresh_incremental()

    def _refresh_incremental(self) -> None:
        columns = _table_columns(self.connection, "interactions")
        required = {
            "id",
            "action",
            "state_hash_before",
            "state_hash_after",
        }
        if not required.issubset(columns):
            return
        outcome_expr = (
            "outcome_state"
            if "outcome_state" in columns
            else "NULL"
        )
        observation_before_expr = (
            "observation_before"
            if "observation_before" in columns
            else "NULL"
        )
        observation_after_expr = (
            "observation_after"
            if "observation_after" in columns
            else "NULL"
        )
        rows = self.connection.execute(
            f"""
            SELECT id, action, state_hash_before, state_hash_after,
                   {outcome_expr},
                   {observation_before_expr},
                   {observation_after_expr}
            FROM interactions
            WHERE id > ?
              AND state_hash_before IS NOT NULL
              AND state_hash_after IS NOT NULL
            ORDER BY id ASC
            """,
            (int(self._last_interaction_id),),
        ).fetchall()
        if not rows:
            return

        for row in rows:
            interaction_id = int(row[0])
            action = int(row[1])
            before_hash = str(row[2])
            after_hash = str(row[3])
            terminal = str(row[4] or "") in {"WIN", "GAME_OVER"}
            item = (before_hash, action, after_hash, terminal)
            if item not in self._seen_exact:
                self._seen_exact.add(item)
                self._exact_adjacency[before_hash].append(
                    (action, after_hash, terminal)
                )
                self._transition_rows.append(item)

            before_array = self._decode_observation(row[5])
            after_array = self._decode_observation(row[6])
            if before_array is not None:
                self._state_structural[before_hash] = (
                    self._structural_signature(before_array)
                )
            if after_array is not None:
                self._state_structural[after_hash] = (
                    self._structural_signature(after_array)
                )
            self._last_interaction_id = max(
                self._last_interaction_id,
                interaction_id,
            )

        self._rebuild_structural_graph()
        self._last_refresh_count = self._last_interaction_id
        self.rows_loaded += len(rows)
        self.refresh_count += 1

    def _rebuild_structural_graph(self) -> None:
        graph: dict[str, list[tuple[int, str, bool]]] = defaultdict(list)
        seen: set[tuple[str, int, str, bool]] = set()
        for before_hash, action, after_hash, terminal in self._transition_rows:
            before_struct = self._state_structural.get(before_hash)
            after_struct = self._state_structural.get(after_hash)
            if not before_struct or not after_struct:
                continue
            item = (
                before_struct,
                int(action),
                after_struct,
                bool(terminal),
            )
            if item in seen:
                continue
            seen.add(item)
            graph[before_struct].append(
                (int(action), after_struct, bool(terminal))
            )
        self._struct_adjacency = graph

    @staticmethod
    def _reachable(
        adjacency: dict[str, list[tuple[int, str, bool]]],
        root: str,
        depth: int,
        *,
        prefix: str,
    ) -> set[str]:
        if root not in adjacency:
            return set()
        reachable: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(root, 0)])
        visited = {root: 0}
        while queue:
            node, distance = queue.popleft()
            if distance >= depth:
                continue
            for action, target, terminal in adjacency.get(node, ()):
                reachable.add(
                    f"{prefix}:{node}|a{int(action)}|{target}"
                )
                if terminal:
                    continue
                next_distance = distance + 1
                old = visited.get(target)
                if old is None or next_distance < old:
                    visited[target] = next_distance
                    queue.append((target, next_distance))
        return reachable

    @staticmethod
    def _decode_observation(payload: Any) -> np.ndarray | None:
        if payload is None:
            return None
        if isinstance(payload, np.ndarray):
            return np.asarray(payload)
        if isinstance(payload, (bytes, bytearray, memoryview)):
            try:
                return decode_array(bytes(payload))
            except Exception:
                return None
        return None

    @staticmethod
    def _state_hash(value: Any) -> str:
        try:
            from v6.memory.trajectory_efficiency import compact_state_hash

            return str(compact_state_hash(value))
        except Exception:
            return "state:" + sha1(
                _json(value).encode("utf-8")
            ).hexdigest()[:24]

    @staticmethod
    def _structural_signature(value: Any) -> str:
        try:
            array = np.asarray(value, dtype=int)
        except Exception:
            return "struct:" + sha1(
                _json(value).encode("utf-8")
            ).hexdigest()[:24]
        if array.ndim == 0:
            array = array.reshape(1, 1)
        if array.ndim > 2:
            array = array.reshape(array.shape[-2], array.shape[-1])
        height, width = array.shape
        unique, counts = np.unique(array, return_counts=True)
        histogram = sorted(
            (int(count) for count in counts),
            reverse=True,
        )
        nonzero = array != 0
        row_occupancy = tuple(
            int(value)
            for value in nonzero.sum(axis=1).tolist()
        )
        col_occupancy = tuple(
            int(value)
            for value in nonzero.sum(axis=0).tolist()
        )
        payload = {
            "shape": [int(height), int(width)],
            "palette_size": int(len(unique)),
            "histogram": histogram,
            "row_occupancy": row_occupancy,
            "col_occupancy": col_occupancy,
        }
        return "struct:" + sha1(
            _json(payload).encode("utf-8")
        ).hexdigest()[:24]


class HierarchicalMemoryLifecycleEngine:
    """Retention/replay/forgetting policy for M1-M6 nodes."""

    def __init__(self, memory: MemorySubstrate) -> None:
        self.memory = memory

    def apply(
        self,
        *,
        step: int | None = None,
    ) -> dict[str, int]:
        migrate_v621(self.memory.connection)
        current_step = int(step or 0)
        summary = {
            "protected": 0,
            "active": 0,
            "dormant": 0,
            "forgotten": 0,
        }
        for level in ("M1", "M2", "M3", "M4", "M5", "M6"):
            for node in self.memory.query_nodes(memory_level=level):
                node_id = str(node["node_id"])
                attrs = dict(node.get("attrs") or {})
                promotion = str(
                    attrs.get("promotion_status", "candidate")
                )
                if promotion in {"rejected", "superseded"}:
                    state = "forgotten"
                    score = 0.0
                    reason = f"promotion_{promotion}"
                else:
                    row = self.memory.connection.execute(
                        """
                        SELECT COALESCE(hierarchical_score, isf_total, 0.0),
                               COALESCE(replay_priority, 0.0)
                        FROM memory_scores
                        WHERE node_id=?
                        """,
                        (node_id,),
                    ).fetchone()
                    score = (
                        max(float(row[0] or 0.0), float(row[1] or 0.0))
                        if row is not None
                        else 0.0
                    )
                    support = int(
                        attrs.get(
                            "support_count",
                            attrs.get(
                                "carrier_count",
                                attrs.get("transfer_tests", 0),
                            ),
                        )
                        or 0
                    )
                    first_seen = node.get("first_seen_step")
                    age = (
                        max(0, current_step - int(first_seen))
                        if first_seen is not None and current_step
                        else 0
                    )
                    if score >= 0.75:
                        state = "protected"
                        reason = "high_hierarchical_significance"
                    elif score >= 0.35:
                        state = "active"
                        reason = "useful_hierarchical_memory"
                    elif score >= 0.15 or support < 3 or age < 500:
                        state = "dormant"
                        reason = "low_current_value_retained_for_revalidation"
                    else:
                        state = "forgotten"
                        reason = "low_value_with_sufficient_evidence"

                retention = 0.0 if state == "forgotten" else _clamp01(score)
                forgetting = _clamp01(1.0 - retention)
                replay = (
                    0.0
                    if state == "forgotten"
                    else _clamp01(
                        score
                        * (
                            1.0
                            if state in {"protected", "active"}
                            else 0.5
                        )
                    )
                )
                self.memory.connection.execute(
                    """
                    UPDATE memory_scores
                    SET retention_status=?,
                        memory_state=?,
                        replay_priority=?,
                        retention_score=?,
                        forgetting_score=?,
                        forgetting_reason=?,
                        lifecycle_version=?
                    WHERE node_id=?
                    """,
                    (
                        state,
                        state,
                        replay,
                        retention,
                        forgetting,
                        None if state != "forgotten" else reason,
                        LIFECYCLE_VERSION,
                        node_id,
                    ),
                )
                self.memory.connection.execute(
                    """
                    UPDATE memory_nodes
                    SET status=?
                    WHERE node_id=?
                    """,
                    (state, node_id),
                )
                self.memory.connection.execute(
                    """
                    INSERT INTO memory_level_lifecycle_v621(
                        memory_id, memory_level, memory_state,
                        replay_priority, retention_score, forgetting_score,
                        last_replayed_step, last_transition_step,
                        reason, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
                    ON CONFLICT(memory_id) DO UPDATE SET
                        memory_level=excluded.memory_level,
                        memory_state=excluded.memory_state,
                        replay_priority=excluded.replay_priority,
                        retention_score=excluded.retention_score,
                        forgetting_score=excluded.forgetting_score,
                        last_transition_step=excluded.last_transition_step,
                        reason=excluded.reason,
                        updated_at=excluded.updated_at
                    """,
                    (
                        node_id,
                        level,
                        state,
                        replay,
                        retention,
                        forgetting,
                        step,
                        reason,
                        time.time(),
                    ),
                )
                summary[state] += 1
        self.memory.connection.commit()
        return summary

    def mark_replayed(
        self,
        memory_id: str,
        *,
        step: int | None = None,
    ) -> None:
        self.memory.connection.execute(
            """
            UPDATE memory_level_lifecycle_v621
            SET last_replayed_step=?, updated_at=?
            WHERE memory_id=?
            """,
            (step, time.time(), str(memory_id)),
        )
        self.memory.connection.execute(
            """
            UPDATE memory_scores
            SET last_replayed_epoch=COALESCE(?, last_replayed_epoch)
            WHERE node_id=?
            """,
            (step, str(memory_id)),
        )
        self.memory.connection.commit()


class V621AbstractionEngine:
    """Strict M4 transfer compressions and explicit relational M5 fragments."""

    def __init__(self, memory: MemorySubstrate) -> None:
        self.memory = memory
        migrate_v621(self.memory.connection)

    def run(
        self,
        *,
        step: int | None = None,
    ) -> dict[str, int]:
        concepts = self.promote_multi_role_concepts(step=step)
        models = self.promote_world_models(step=step)
        return {
            "concepts_created": concepts,
            "world_models_created": models,
        }

    def promote_multi_role_concepts(
        self,
        *,
        step: int | None = None,
    ) -> int:
        roles = [
            node
            for node in self.memory.query_nodes(
                memory_level="M3",
                node_type="FunctionalRoleMemory",
            )
            if str(
                node.get("attrs", {}).get(
                    "promotion_status",
                    "candidate",
                )
            )
            in {"accepted", "candidate"}
        ]
        descriptors = {
            str(role["node_id"]): self._role_descriptor(role)
            for role in roles
        }
        adjacency: dict[str, set[str]] = defaultdict(set)
        role_ids = sorted(descriptors)
        for index, left in enumerate(role_ids):
            for right in role_ids[index + 1 :]:
                if self._roles_compatible(
                    descriptors[left],
                    descriptors[right],
                ):
                    adjacency[left].add(right)
                    adjacency[right].add(left)

        components = self._components(role_ids, adjacency)
        created = 0
        strict_role_sets: list[set[str]] = []
        for component in components:
            if len(component) < 2:
                continue
            items = [descriptors[item] for item in sorted(component)]
            role_set = set(component)
            families = sorted(
                {
                    value
                    for item in items
                    for value in item["families"]
                }
            )
            contexts = sorted(
                {
                    value
                    for item in items
                    for value in item["contexts"]
                }
            )
            outcomes = sorted(
                {
                    value
                    for item in items
                    for value in item["outcomes"]
                }
            )
            games = sorted(
                {
                    value
                    for item in items
                    for value in item["games"]
                    if value
                }
            )
            effects = sorted(
                {
                    str(item["future_option_effect"] or "neutral")
                    for item in items
                }
            )
            overlap_score = self._component_overlap_score(items)
            if overlap_score <= 0.0:
                continue

            direct = self._direct_concept_transfer_for_roles(role_set)
            direct_tests = int(direct["tests"])
            direct_successes = int(direct["successes"])
            direct_rate = (
                direct_successes / direct_tests
                if direct_tests > 0
                else 0.0
            )
            cross_game = len(games) >= 2 or len(contexts) >= 2
            status = (
                "accepted"
                if direct_tests >= 2
                and direct_rate >= 0.5
                and cross_game
                else "candidate"
            )

            signature_payload = {
                "roles": sorted(role_set),
                "families": families,
                "effects": effects,
            }
            signature = sha1(
                _json(signature_payload).encode("utf-8")
            ).hexdigest()[:20]
            node_id = concept_node_id("v621:" + signature)
            attrs = {
                "source_roles": sorted(role_set),
                "source_families": families,
                "applicability_contexts": contexts,
                "predicted_outcomes": outcomes,
                "source_games": games,
                "future_option_effects": effects,
                "transfer_tests": direct_tests,
                "transfer_success_count": direct_successes,
                "transfer_failure_count": max(
                    0,
                    direct_tests - direct_successes,
                ),
                "transfer_score": direct_rate,
                "structural_overlap_score": overlap_score,
                "cross_game_evidence": cross_game,
                "explanatory_reach": len(families) + len(contexts),
                "compression_gain": max(
                    0.0,
                    (len(role_set) - 1) / len(role_set),
                ),
                "promotion_status": status,
                "concept_version": ABSTRACTION_VERSION,
                "validation_source": (
                    "direct_concept_transfer"
                    if direct_tests
                    else "structural_cross_role_candidate"
                ),
            }
            self.memory.upsert_node(
                MemoryNode(
                    node_id=node_id,
                    memory_level="M4",
                    node_type="ConceptMemory",
                    canonical_key=signature,
                    attrs=attrs,
                ),
                step=step,
            )
            for role_id in sorted(role_set):
                self.memory.upsert_edge(
                    MemoryEdge(
                        role_id,
                        node_id,
                        "transfers_to",
                        edge_source=ABSTRACTION_VERSION,
                    )
                )
                self.memory.upsert_edge(
                    MemoryEdge(
                        node_id,
                        role_id,
                        "derived_from",
                        edge_source=ABSTRACTION_VERSION,
                    )
                )
            strict_role_sets.append(role_set)
            created += 1

        if strict_role_sets:
            for concept in self.memory.query_nodes(
                memory_level="M4",
                node_type="ConceptMemory",
            ):
                attrs = dict(concept.get("attrs") or {})
                if attrs.get("concept_version") == ABSTRACTION_VERSION:
                    continue
                old_roles = {
                    str(value)
                    for value in attrs.get("source_roles", []) or []
                }
                if old_roles and any(
                    old_roles.issubset(strict)
                    for strict in strict_role_sets
                ):
                    attrs["promotion_status"] = "superseded"
                    attrs["superseded_reason"] = (
                        "replaced_by_v621_structurally_compatible_concept"
                    )
                    self.memory.update_node_support_and_attrs(
                        str(concept["node_id"]),
                        attrs,
                        support_increment=0,
                        step=step,
                    )
        return created

    def promote_world_models(
        self,
        *,
        step: int | None = None,
    ) -> int:
        concepts = [
            node
            for node in self.memory.query_nodes(
                memory_level="M4",
                node_type="ConceptMemory",
            )
            if str(
                node.get("attrs", {}).get(
                    "promotion_status",
                    "candidate",
                )
            )
            == "accepted"
            and node.get("attrs", {}).get("concept_version")
            == ABSTRACTION_VERSION
        ]
        if len(concepts) < 2:
            return 0

        descriptors = {
            str(node["node_id"]): self._concept_descriptor(node)
            for node in concepts
        }
        pair_relations: dict[
            tuple[str, str],
            list[tuple[str, int, float, dict[str, Any]]],
        ] = {}
        adjacency: dict[str, set[str]] = defaultdict(set)
        ids = sorted(descriptors)
        for index, left in enumerate(ids):
            for right in ids[index + 1 :]:
                relations = self._relations_between(
                    left,
                    right,
                    descriptors[left],
                    descriptors[right],
                )
                if not relations:
                    continue
                pair_relations[(left, right)] = relations
                adjacency[left].add(right)
                adjacency[right].add(left)

        created = 0
        for component in self._components(ids, adjacency):
            if len(component) < 2:
                continue
            component_relations: list[
                tuple[
                    str,
                    str,
                    str,
                    int,
                    float,
                    dict[str, Any],
                ]
            ] = []
            for left in sorted(component):
                for right in sorted(component):
                    if left >= right:
                        continue
                    for relation in pair_relations.get(
                        (left, right),
                        (),
                    ):
                        relation_type, support, confidence, evidence = relation
                        component_relations.append(
                            (
                                left,
                                right,
                                relation_type,
                                support,
                                confidence,
                                evidence,
                            )
                        )
            if len(component_relations) < 2:
                continue

            concept_ids = sorted(component)
            contexts = sorted(
                {
                    item
                    for concept_id in component
                    for item in descriptors[concept_id]["contexts"]
                }
            )
            outcomes = sorted(
                {
                    item
                    for concept_id in component
                    for item in descriptors[concept_id]["outcomes"]
                }
            )
            families = sorted(
                {
                    item
                    for concept_id in component
                    for item in descriptors[concept_id]["families"]
                }
            )
            predictive_types = {
                "precedes",
                "enables",
                "constrains",
                "shared_outcome",
            }
            predictive_count = sum(
                1
                for relation in component_relations
                if relation[2] in predictive_types
            )
            signature = sha1(
                _json(
                    {
                        "concepts": concept_ids,
                        "relations": [
                            [item[0], item[1], item[2]]
                            for item in component_relations
                        ],
                    }
                ).encode("utf-8")
            ).hexdigest()[:20]
            node_id = world_model_node_id("v621:" + signature)
            status = (
                "accepted"
                if predictive_count >= 1
                and contexts
                and (outcomes or families)
                else "candidate"
            )
            attrs = {
                "concept_ids": concept_ids,
                "supported_contexts": contexts,
                "predicted_outcomes": outcomes,
                "source_families": families,
                "relation_count": len(component_relations),
                "predictive_relation_count": predictive_count,
                "relation_types": sorted(
                    {item[2] for item in component_relations}
                ),
                "explanatory_reach": sum(
                    float(
                        descriptors[concept_id]["explanatory_reach"]
                    )
                    for concept_id in component
                ),
                "promotion_status": status,
                "world_model_version": ABSTRACTION_VERSION,
            }
            self.memory.upsert_node(
                MemoryNode(
                    node_id=node_id,
                    memory_level="M5",
                    node_type="WorldModelFragment",
                    canonical_key=signature,
                    attrs=attrs,
                ),
                step=step,
            )
            for concept_id in concept_ids:
                self.memory.upsert_edge(
                    MemoryEdge(
                        concept_id,
                        node_id,
                        "explains",
                        edge_source=ABSTRACTION_VERSION,
                    )
                )
                self.memory.upsert_edge(
                    MemoryEdge(
                        node_id,
                        concept_id,
                        "depends_on",
                        edge_source=ABSTRACTION_VERSION,
                    )
                )

            self.memory.connection.execute(
                "DELETE FROM world_model_relations_v621 WHERE model_id=?",
                (node_id,),
            )
            for (
                source,
                target,
                relation_type,
                support,
                confidence,
                evidence,
            ) in component_relations:
                relation_id = "wmrel:" + sha1(
                    f"{node_id}|{source}|{target}|{relation_type}".encode(
                        "utf-8"
                    )
                ).hexdigest()[:24]
                self.memory.connection.execute(
                    """
                    INSERT OR REPLACE INTO world_model_relations_v621(
                        relation_id, model_id,
                        source_concept_id, target_concept_id,
                        relation_type, support_count, confidence,
                        evidence_json, updated_step, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        relation_id,
                        node_id,
                        source,
                        target,
                        relation_type,
                        int(support),
                        float(confidence),
                        _json(evidence),
                        step,
                        time.time(),
                    ),
                )
            created += 1

        if created:
            for model in self.memory.query_nodes(
                memory_level="M5",
                node_type="WorldModelFragment",
            ):
                attrs = dict(model.get("attrs") or {})
                if attrs.get("world_model_version") == ABSTRACTION_VERSION:
                    continue
                attrs["promotion_status"] = "superseded"
                attrs["superseded_reason"] = (
                    "replaced_by_v621_relational_world_model"
                )
                self.memory.update_node_support_and_attrs(
                    str(model["node_id"]),
                    attrs,
                    support_increment=0,
                    step=step,
                )
        self.memory.connection.commit()
        return created

    def _role_descriptor(
        self,
        role: dict[str, Any],
    ) -> dict[str, Any]:
        role_id = str(role["node_id"])
        attrs = dict(role.get("attrs") or {})
        carriers = {
            str(edge["source_node_id"])
            for edge in self.memory.edges_to(role_id, "plays_role")
        }
        families: set[str] = set()
        contexts: set[str] = set()
        outcomes: set[str] = set()
        for carrier in carriers:
            for edge in self.memory.edges_from(carrier):
                edge_type = str(edge["edge_type"])
                target = str(edge["target_node_id"])
                if edge_type == "associated_with_family":
                    families.add(target)
                elif edge_type == "appears_in_context":
                    contexts.add(target)
                elif edge_type == "carried_by":
                    for interaction_edge in self.memory.edges_from(target):
                        if str(interaction_edge["edge_type"]) == "has_outcome":
                            outcomes.add(
                                str(interaction_edge["target_node_id"])
                            )
        profile = self._role_transfer_profile(
            str(
                attrs.get(
                    "role_signature",
                    role.get("canonical_key", ""),
                )
            )
        )
        return {
            "role_id": role_id,
            "carriers": carriers,
            "families": families,
            "contexts": contexts,
            "outcomes": outcomes,
            "future_option_effect": attrs.get(
                "future_option_effect"
            ),
            "tests": int(profile["tests"]),
            "successes": int(profile["successes"]),
            "games": set(profile["games"]),
        }

    def _role_transfer_profile(
        self,
        signature: str,
    ) -> dict[str, Any]:
        columns = _table_columns(
            self.memory.connection,
            "role_transfer_attempts",
        )
        if not columns or "role_signature" not in columns:
            return {"tests": 0, "successes": 0, "games": set()}
        success_col = (
            "reuse_success"
            if "reuse_success" in columns
            else "success"
            if "success" in columns
            else None
        )
        if success_col is None:
            return {"tests": 0, "successes": 0, "games": set()}
        game_columns = [
            candidate
            for candidate in (
                "target_game_key",
                "target_game",
                "source_game_key",
                "source_game",
                "game",
            )
            if candidate in columns
        ]
        select_games = (
            ", " + ", ".join(game_columns)
            if game_columns
            else ""
        )
        rows = self.memory.connection.execute(
            f"""
            SELECT COALESCE({success_col}, 0){select_games}
            FROM role_transfer_attempts
            WHERE role_signature=?
            """,
            (signature,),
        ).fetchall()
        games: set[str] = set()
        successes = 0
        for row in rows:
            successes += int(bool(row[0]))
            for value in row[1:]:
                if value not in (None, ""):
                    games.add(str(value))
        return {
            "tests": len(rows),
            "successes": successes,
            "games": games,
        }

    def _direct_concept_transfer_for_roles(
        self,
        role_ids: set[str],
    ) -> dict[str, int]:
        if not role_ids:
            return {"tests": 0, "successes": 0}
        concepts = [
            node
            for node in self.memory.query_nodes(
                memory_level="M4",
                node_type="ConceptMemory",
            )
            if set(
                str(value)
                for value in node.get("attrs", {}).get(
                    "source_roles",
                    [],
                )
            )
            == role_ids
        ]
        if not concepts:
            return {"tests": 0, "successes": 0}
        concept_ids = [
            str(node["node_id"])
            for node in concepts
        ]
        placeholders = ",".join("?" for _ in concept_ids)
        row = self.memory.connection.execute(
            f"""
            SELECT COUNT(*),
                   COALESCE(SUM(CASE WHEN success=1 THEN 1 ELSE 0 END), 0)
            FROM concept_transfer_attempts_v621
            WHERE concept_id IN ({placeholders})
            """,
            tuple(concept_ids),
        ).fetchone()
        return {
            "tests": int(row[0] or 0),
            "successes": int(row[1] or 0),
        }

    @staticmethod
    def _roles_compatible(
        left: dict[str, Any],
        right: dict[str, Any],
    ) -> bool:
        left_effect = str(
            left.get("future_option_effect") or "neutral"
        )
        right_effect = str(
            right.get("future_option_effect") or "neutral"
        )
        if left_effect != right_effect:
            return False
        structural_overlap = bool(
            (left["families"] & right["families"])
            or (left["contexts"] & right["contexts"])
            or (left["outcomes"] & right["outcomes"])
        )
        if not structural_overlap:
            return False
        if int(left["successes"]) <= 0 or int(right["successes"]) <= 0:
            return False
        games = set(left["games"]) | set(right["games"])
        contexts = set(left["contexts"]) | set(right["contexts"])
        return len(games) >= 2 or len(contexts) >= 2

    @staticmethod
    def _component_overlap_score(
        items: list[dict[str, Any]],
    ) -> float:
        if len(items) < 2:
            return 0.0
        pair_scores: list[float] = []
        for index, left in enumerate(items):
            for right in items[index + 1 :]:
                dimensions = (
                    bool(left["families"] & right["families"]),
                    bool(left["contexts"] & right["contexts"]),
                    bool(left["outcomes"] & right["outcomes"]),
                )
                pair_scores.append(
                    sum(int(value) for value in dimensions) / 3.0
                )
        return (
            sum(pair_scores) / len(pair_scores)
            if pair_scores
            else 0.0
        )

    @staticmethod
    def _components(
        nodes: list[str],
        adjacency: dict[str, set[str]],
    ) -> list[set[str]]:
        unseen = set(nodes)
        output: list[set[str]] = []
        while unseen:
            root = unseen.pop()
            component = {root}
            queue = [root]
            while queue:
                current = queue.pop()
                for neighbor in adjacency.get(current, set()):
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        component.add(neighbor)
                        queue.append(neighbor)
            output.append(component)
        return output

    @staticmethod
    def _concept_descriptor(
        concept: dict[str, Any],
    ) -> dict[str, Any]:
        attrs = dict(concept.get("attrs") or {})
        effects = attrs.get("future_option_effects")
        if not effects:
            single = attrs.get("future_option_effect")
            effects = [single] if single else []
        return {
            "contexts": set(
                str(value)
                for value in attrs.get(
                    "applicability_contexts",
                    attrs.get("supported_contexts", []),
                )
                or []
            ),
            "families": set(
                str(value)
                for value in attrs.get("source_families", []) or []
            ),
            "outcomes": set(
                str(value)
                for value in attrs.get("predicted_outcomes", []) or []
            ),
            "effects": set(
                str(value)
                for value in effects or []
            ),
            "explanatory_reach": float(
                attrs.get("explanatory_reach", 0.0) or 0.0
            ),
        }

    def _relations_between(
        self,
        left_id: str,
        right_id: str,
        left: dict[str, Any],
        right: dict[str, Any],
    ) -> list[tuple[str, int, float, dict[str, Any]]]:
        relations: list[
            tuple[str, int, float, dict[str, Any]]
        ] = []
        shared_contexts = left["contexts"] & right["contexts"]
        shared_families = left["families"] & right["families"]
        shared_outcomes = left["outcomes"] & right["outcomes"]
        if shared_contexts:
            relations.append(
                (
                    "co_context",
                    len(shared_contexts),
                    _clamp01(len(shared_contexts) / 3.0),
                    {"shared_contexts": sorted(shared_contexts)},
                )
            )
        if shared_families:
            relations.append(
                (
                    "shared_family",
                    len(shared_families),
                    _clamp01(len(shared_families) / 3.0),
                    {"shared_families": sorted(shared_families)},
                )
            )
        if shared_outcomes:
            relations.append(
                (
                    "shared_outcome",
                    len(shared_outcomes),
                    _clamp01(len(shared_outcomes) / 2.0),
                    {"shared_outcomes": sorted(shared_outcomes)},
                )
            )

        order = self._ordered_support(left_id, right_id)
        if order["support"] >= 2 and order["confidence"] >= 0.65:
            source = left_id if order["direction"] == "left_before" else right_id
            target = right_id if source == left_id else left_id
            relations.append(
                (
                    "precedes",
                    int(order["support"]),
                    float(order["confidence"]),
                    {
                        "source": source,
                        "target": target,
                    },
                )
            )
            source_desc = left if source == left_id else right
            if "positive" in source_desc["effects"]:
                relations.append(
                    (
                        "enables",
                        int(order["support"]),
                        float(order["confidence"]),
                        {
                            "source": source,
                            "target": target,
                            "basis": "positive_future_option_precedence",
                        },
                    )
                )
            if "negative" in source_desc["effects"]:
                relations.append(
                    (
                        "constrains",
                        int(order["support"]),
                        float(order["confidence"]),
                        {
                            "source": source,
                            "target": target,
                            "basis": "negative_future_option_precedence",
                        },
                    )
                )
        return relations

    def _ordered_support(
        self,
        left_id: str,
        right_id: str,
    ) -> dict[str, Any]:
        left_occurrences = self._concept_occurrences(left_id)
        right_occurrences = self._concept_occurrences(right_id)
        left_before = 0
        right_before = 0
        comparable = 0
        for left in left_occurrences:
            for right in right_occurrences:
                if left[0] != right[0] or left[1] != right[1]:
                    continue
                if left[2] == right[2]:
                    continue
                comparable += 1
                if left[2] < right[2]:
                    left_before += 1
                else:
                    right_before += 1
        if comparable <= 0:
            return {
                "support": 0,
                "confidence": 0.0,
                "direction": "unknown",
            }
        best = max(left_before, right_before)
        return {
            "support": comparable,
            "confidence": best / comparable,
            "direction": (
                "left_before"
                if left_before >= right_before
                else "right_before"
            ),
        }

    def _concept_occurrences(
        self,
        concept_id: str,
    ) -> list[tuple[str, int, int]]:
        concept = self.memory.get_node(concept_id)
        if concept is None:
            return []
        role_ids = [
            str(value)
            for value in concept.get("attrs", {}).get(
                "source_roles",
                [],
            )
        ]
        interaction_ids: set[int] = set()
        for role_id in role_ids:
            for carrier_edge in self.memory.edges_from(
                role_id,
                "abstracts_from",
            ):
                carrier_id = str(carrier_edge["target_node_id"])
                for interaction_edge in self.memory.edges_from(
                    carrier_id,
                    "carried_by",
                ):
                    target = str(interaction_edge["target_node_id"])
                    try:
                        interaction_ids.add(
                            int(target.rsplit(":", 1)[-1])
                        )
                    except ValueError:
                        continue
        if not interaction_ids:
            return []
        placeholders = ",".join("?" for _ in interaction_ids)
        columns = _table_columns(
            self.memory.connection,
            "interactions",
        )
        game_expr = "COALESCE(game_id, '')" if "game_id" in columns else "''"
        episode_expr = (
            "COALESCE(episode_id, 0)"
            if "episode_id" in columns
            else "0"
        )
        step_expr = (
            "COALESCE(global_step, id)"
            if "global_step" in columns
            else "id"
        )
        rows = self.memory.connection.execute(
            f"""
            SELECT {game_expr}, {episode_expr}, {step_expr}
            FROM interactions
            WHERE id IN ({placeholders})
            ORDER BY {step_expr}
            """,
            tuple(sorted(interaction_ids)),
        ).fetchall()
        return [
            (str(row[0]), int(row[1] or 0), int(row[2] or 0))
            for row in rows
        ]


class V621PromotionEngine:
    def __init__(
        self,
        memory: MemorySubstrate,
        *,
        base: MemoryPromotionEngine | None = None,
    ) -> None:
        self.memory = memory
        migrate_v621(self.memory.connection)
        self.base = base or MemoryPromotionEngine(memory)
        self.abstractions = V621AbstractionEngine(memory)
        self.significance = HierarchicalSignificanceEngine(memory)
        self.lifecycle = HierarchicalMemoryLifecycleEngine(memory)

    def run_all(
        self,
        step: int | None = None,
    ) -> dict[str, Any]:
        base_summary = self.base.run_all(step=step)
        lower = self._validate_levels(
            {"M1", "M2", "M3"},
            step=step,
        )
        abstractions = self.abstractions.run(step=step)
        upper = self._validate_levels(
            {"M4", "M5", "M6"},
            step=step,
        )
        scoring = self.significance.rescore_all(step=step)
        lifecycle = self.lifecycle.apply(step=step)
        self._sync_promotion_rows()
        return {
            **base_summary,
            "v621_validation_lower": lower,
            "v621_abstractions": abstractions,
            "v621_validation_upper": upper,
            "v621_hierarchical_scoring": scoring,
            "v621_hierarchical_lifecycle": lifecycle,
        }

    def _validate_levels(
        self,
        levels: set[str],
        *,
        step: int | None,
    ) -> dict[str, int]:
        summary = {
            "accepted": 0,
            "candidate": 0,
            "rejected": 0,
        }
        for level in sorted(levels):
            for node in self.memory.query_nodes(memory_level=level):
                attrs = dict(node.get("attrs") or {})
                dimensions, mandatory, required = (
                    self._evidence_dimensions(node)
                )
                count = sum(
                    1
                    for value in dimensions.values()
                    if bool(value)
                )
                all_mandatory = all(
                    bool(dimensions.get(name))
                    for name in mandatory
                )
                status = (
                    "accepted"
                    if all_mandatory and count >= required
                    else self._candidate_status(
                        node,
                        dimensions,
                        count,
                        required,
                    )
                )
                reason = (
                    None
                    if status == "accepted"
                    else (
                        f"requires {required} evidence dimensions "
                        f"and mandatory={sorted(mandatory)}"
                    )
                )
                attrs["promotion_status"] = status
                attrs["promotion_policy_version"] = POLICY_VERSION
                attrs["promotion_evidence_dimensions"] = dimensions
                attrs["promotion_evidence_dimension_count"] = count
                if reason:
                    attrs["promotion_rejection_reason"] = reason
                else:
                    attrs.pop("promotion_rejection_reason", None)
                self.memory.update_node_support_and_attrs(
                    str(node["node_id"]),
                    attrs,
                    support_increment=0,
                    step=step,
                )
                self.memory.connection.execute(
                    """
                    INSERT INTO memory_promotion_evidence_v62(
                        node_id, memory_level, node_type,
                        evidence_dimensions_json,
                        evidence_dimension_count,
                        required_dimension_count,
                        validation_status,
                        validation_reason,
                        updated_step,
                        updated_at,
                        policy_version,
                        validation_source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(node_id) DO UPDATE SET
                        evidence_dimensions_json=excluded.evidence_dimensions_json,
                        evidence_dimension_count=excluded.evidence_dimension_count,
                        required_dimension_count=excluded.required_dimension_count,
                        validation_status=excluded.validation_status,
                        validation_reason=excluded.validation_reason,
                        updated_step=excluded.updated_step,
                        updated_at=excluded.updated_at,
                        policy_version=excluded.policy_version,
                        validation_source=excluded.validation_source
                    """,
                    (
                        str(node["node_id"]),
                        str(level),
                        str(node["node_type"]),
                        _json(dimensions),
                        int(count),
                        int(required),
                        status,
                        reason,
                        step,
                        time.time(),
                        POLICY_VERSION,
                        "v621_runtime_validation",
                    ),
                )
                summary[status] += 1
        self.memory.connection.commit()
        return summary

    def _candidate_status(
        self,
        node: dict[str, Any],
        dimensions: dict[str, bool],
        count: int,
        required: int,
    ) -> str:
        level = str(node.get("memory_level"))
        attrs = dict(node.get("attrs") or {})
        if level == "M4":
            structural = (
                bool(dimensions.get("multi_role"))
                and bool(dimensions.get("structural_overlap"))
                and bool(dimensions.get("cross_game"))
            )
            return "candidate" if structural else "rejected"
        if level == "M5":
            structural = (
                bool(dimensions.get("multi_concept"))
                and bool(dimensions.get("relations"))
            )
            return "candidate" if structural else "rejected"
        if level == "M6":
            structural = all(
                bool(dimensions.get(name))
                for name in (
                    "successful",
                    "cost_known",
                    "equivalent_outcome",
                    "cost_advantage",
                )
            )
            return "candidate" if structural else "rejected"
        return "rejected"

    def _evidence_dimensions(
        self,
        node: dict[str, Any],
    ) -> tuple[dict[str, bool], set[str], int]:
        level = str(node.get("memory_level"))
        node_type = str(node.get("node_type"))
        node_id = str(node.get("node_id"))
        attrs = dict(node.get("attrs") or {})
        support = int(
            attrs.get(
                "support_count",
                attrs.get("carrier_count", 0),
            )
            or 0
        )
        incoming_lower = [
            edge
            for edge in self.memory.edges_to(node_id)
            if _node_level(str(edge["source_node_id"]))
            in {"M0", "M1", "M2", "M3", "M4"}
        ]
        lower_sources = {
            str(edge["source_node_id"])
            for edge in incoming_lower
        }
        prediction = float(
            attrs.get("prediction_lift", 0.0) or 0.0
        )
        compression = float(
            attrs.get("compression_gain", 0.0) or 0.0
        )
        explanatory = float(
            attrs.get("explanatory_reach", 0.0) or 0.0
        )

        if level == "M1":
            dims = {
                "support": support >= 3,
                "confidence": float(
                    attrs.get("confidence", 0.0) or 0.0
                )
                >= 0.6,
                "provenance": len(lower_sources) >= 3,
            }
            return dims, set(dims), 3

        if level == "M2":
            family_sources = {
                str(edge["source_node_id"])
                for edge in incoming_lower
                if _node_level(str(edge["source_node_id"])) == "M1"
            }
            dims = {
                "support": support >= 3 or len(family_sources) >= 2,
                "provenance": len(family_sources) >= 2,
                "prediction": prediction > 0.0,
                "compression": compression > 0.0,
                "breadth": len(family_sources) >= 3,
            }
            return dims, {"support", "provenance"}, 3

        if level == "M3" and node_type == "CarrierMemory":
            dims = {
                "support": support >= 3,
                "prediction": prediction > 0.0,
                "compression": compression > 0.0,
                "context_breadth": int(
                    attrs.get(
                        "distinct_context_count",
                        attrs.get(
                            "carrier_distinct_context_count",
                            0,
                        ),
                    )
                    or 0
                )
                >= 2,
            }
            return (
                dims,
                {"support", "prediction", "compression"},
                3,
            )

        if level == "M3":
            role_signature = str(
                attrs.get(
                    "role_signature",
                    node.get("canonical_key", ""),
                )
            )
            transfer_tests = self._role_transfer_tests(role_signature)
            future_effect = attrs.get("future_option_effect")
            dims = {
                "structural_support": int(
                    attrs.get("carrier_count", 0) or 0
                )
                >= 2,
                "direct_transfer": transfer_tests > 0,
                "future_option": future_effect
                in {"positive", "negative", "neutral"},
                "explanatory": explanatory > 0.0
                or int(
                    attrs.get("carried_interaction_count", 0)
                    or 0
                )
                > 0,
            }
            return dims, {"structural_support"}, 3

        if level == "M4":
            direct_tests, direct_successes = (
                self._concept_transfer_counts(node_id)
            )
            roles = attrs.get("source_roles", []) or []
            dims = {
                "multi_role": len(roles) >= 2,
                "structural_overlap": float(
                    attrs.get("structural_overlap_score", 0.0)
                    or 0.0
                )
                > 0.0,
                "cross_game": bool(
                    attrs.get("cross_game_evidence", False)
                ),
                "direct_transfer_tests": direct_tests >= 2,
                "direct_transfer_success": (
                    direct_successes / max(1, direct_tests)
                )
                >= 0.5,
                "compression": compression > 0.0,
            }
            return (
                dims,
                {
                    "multi_role",
                    "structural_overlap",
                    "cross_game",
                    "direct_transfer_tests",
                    "direct_transfer_success",
                },
                5,
            )

        if level == "M5":
            relation_count = int(
                attrs.get("relation_count", 0) or 0
            )
            predictive = int(
                attrs.get("predictive_relation_count", 0) or 0
            )
            dims = {
                "multi_concept": len(
                    attrs.get("concept_ids", []) or []
                )
                >= 2,
                "relations": relation_count >= 2,
                "predictive_relations": predictive >= 1,
                "contexts": len(
                    attrs.get("supported_contexts", []) or []
                )
                > 0,
                "outcomes_or_families": bool(
                    attrs.get("predicted_outcomes", [])
                    or attrs.get("source_families", [])
                ),
            }
            return (
                dims,
                {
                    "multi_concept",
                    "relations",
                    "predictive_relations",
                },
                4,
            )

        if level == "M6":
            cost = attrs.get(
                "cost",
                attrs.get("current_cost"),
            )
            best = attrs.get(
                "best_known_length",
                attrs.get("best_known_cost"),
            )
            equivalent = bool(
                attrs.get("outcome_signature")
                or attrs.get("effects")
                or attrs.get("comparable_outcome_group_id")
            )
            cost_advantage = False
            if cost is not None and best is not None:
                try:
                    cost_advantage = float(cost) <= float(best)
                except (TypeError, ValueError):
                    cost_advantage = False
            dims = {
                "successful": float(
                    attrs.get("success_rate", 0.0) or 0.0
                )
                > 0.0,
                "cost_known": cost is not None and best is not None,
                "equivalent_outcome": equivalent,
                "cost_advantage": cost_advantage,
                "reuse": int(attrs.get("reuse_count", 0) or 0) > 0,
            }
            return (
                dims,
                {
                    "successful",
                    "cost_known",
                    "equivalent_outcome",
                    "cost_advantage",
                    "reuse",
                },
                5,
            )

        return {}, set(), 99

    def _role_transfer_tests(self, signature: str) -> int:
        columns = _table_columns(
            self.memory.connection,
            "role_transfer_attempts",
        )
        if "role_signature" not in columns:
            return 0
        row = self.memory.connection.execute(
            """
            SELECT COUNT(*)
            FROM role_transfer_attempts
            WHERE role_signature=?
            """,
            (signature,),
        ).fetchone()
        return int(row[0] or 0)

    def _concept_transfer_counts(
        self,
        concept_id: str,
    ) -> tuple[int, int]:
        row = self.memory.connection.execute(
            """
            SELECT COUNT(*),
                   COALESCE(SUM(CASE WHEN success=1 THEN 1 ELSE 0 END), 0)
            FROM concept_transfer_attempts_v621
            WHERE concept_id=?
            """,
            (str(concept_id),),
        ).fetchone()
        return int(row[0] or 0), int(row[1] or 0)

    def _sync_promotion_rows(self) -> None:
        rows = self.memory.connection.execute(
            """
            SELECT node_id, evidence_dimension_count,
                   validation_status, validation_reason,
                   policy_version
            FROM memory_promotion_evidence_v62
            WHERE policy_version=?
            """,
            (POLICY_VERSION,),
        ).fetchall()
        for row in rows:
            self.memory.connection.execute(
                """
                UPDATE memory_promotions
                SET evidence_dimension_count=?,
                    validation_status=?,
                    validation_reason=?,
                    policy_version=?
                WHERE target_node_id=?
                """,
                (
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[0],
                ),
            )
        self.memory.connection.commit()


class V621MemoryQueryEngine(V62MemoryQueryEngine):
    def __init__(
        self,
        memory: MemorySubstrate,
        contingency_learner: Any = None,
        graph: Any = None,
    ) -> None:
        super().__init__(
            memory,
            contingency_learner=contingency_learner,
            graph=graph,
        )
        self.last_strategy_by_action: dict[int, str] = {}

    def find_similar_roles(
        self,
        context_signature: str,
        action: int,
    ) -> list[dict]:
        return [
            item
            for item in super().find_similar_roles(
                context_signature,
                action,
            )
            if self._usable(str(item["node_id"]))
        ]

    def find_concept_matches(
        self,
        context_signature: str,
        action: int,
    ) -> list[dict]:
        output: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in super().find_concept_matches(
            context_signature,
            action,
        ):
            node_id = str(item["node_id"])
            if not self._usable(node_id):
                continue
            node = self.memory.get_node(node_id)
            status = str(
                (node or {}).get("attrs", {}).get(
                    "promotion_status",
                    "candidate",
                )
            )
            adjusted = dict(item)
            if status == "candidate":
                adjusted["score"] = float(
                    adjusted.get("score", 0.0)
                ) * 0.6
            output.append(adjusted)
            seen.add(node_id)

        # Bootstrap direct concept validation without requiring concept
        # transfer successes to already exist. Candidate concepts receive
        # only a weak structural prior until direct attempts accumulate.
        role_matches = self.find_similar_roles(
            context_signature,
            action,
        )
        role_by_id = {
            str(item["node_id"]): item
            for item in role_matches
        }
        for concept in self.memory.query_nodes(
            memory_level="M4",
            node_type="ConceptMemory",
        ):
            node_id = str(concept["node_id"])
            if node_id in seen or not self._usable(node_id):
                continue
            attrs = dict(concept.get("attrs") or {})
            if attrs.get("concept_version") != ABSTRACTION_VERSION:
                continue
            if str(attrs.get("promotion_status")) != "candidate":
                continue
            source_roles = [
                str(value)
                for value in attrs.get("source_roles", []) or []
            ]
            matching = [
                role_by_id[role_id]
                for role_id in source_roles
                if role_id in role_by_id
                and role_by_id[role_id].get("family_id") is not None
            ]
            if not matching:
                continue
            best_role = max(
                matching,
                key=lambda item: float(item.get("score", 0.0)),
            )
            structural = _clamp01(
                attrs.get("structural_overlap_score", 0.0)
            )
            score = (
                float(best_role.get("score", 0.0))
                * max(0.1, structural)
                * 0.4
            )
            if score <= 0.0:
                continue
            output.append(
                {
                    "node_id": node_id,
                    "score": score,
                    "family_id": best_role.get("family_id"),
                    "bootstrap_candidate": True,
                }
            )
        return sorted(
            output,
            key=lambda item: (
                -float(item.get("score", 0.0)),
                str(item.get("node_id")),
            ),
        )

    def score_action(
        self,
        context_signatures: dict[int, tuple],
        action: int,
        available_actions: list[int],
        *,
        record_query: bool = False,
    ) -> MemoryActionScore:
        base = super().score_action(
            context_signatures,
            action,
            available_actions,
            record_query=record_query,
        )
        best_context = self._best_context_signature(
            context_signatures,
            action,
        )
        strategy_score = self._strategy_score(
            best_context,
            action,
        )
        world_score = self._world_model_score(
            best_context,
            action,
        )
        score = _clamp01(
            float(base.score)
            + 0.10 * strategy_score
            + 0.07 * world_score
        )
        sources = list(base.evidence_sources)
        if strategy_score > 0:
            sources.append("M6_strategy_memory_v621")
        if world_score > 0:
            sources.append("M5_world_model_memory_v621")
        return replace(
            base,
            score=score,
            evidence_sources=sources,
        )

    def _usable(self, node_id: str) -> bool:
        node = self.memory.get_node(str(node_id))
        if node is None:
            return False
        attrs = dict(node.get("attrs") or {})
        promotion = str(
            attrs.get("promotion_status", "candidate")
        )
        node_status = str(node.get("status", "active"))
        return (
            promotion not in {"rejected", "superseded"}
            and node_status != "forgotten"
        )

    def _strategy_score(
        self,
        context_signature: str,
        action: int,
    ) -> float:
        best_score = 0.0
        best_id: str | None = None
        for node in self.memory.query_nodes(
            memory_level="M6",
            node_type="EfficientStrategyMemory",
        ):
            node_id = str(node["node_id"])
            if not self._usable(node_id):
                continue
            attrs = dict(node.get("attrs") or {})
            sequence = attrs.get("action_sequence") or []
            if not sequence or int(sequence[0]) != int(action):
                continue
            context = str(attrs.get("context_key") or "")
            context_factor = (
                1.0
                if not context or context == context_signature
                else 0.5
            )
            success = _clamp01(attrs.get("success_rate"))
            cost = float(
                attrs.get("cost", len(sequence))
                or len(sequence)
                or 1
            )
            best_len = float(
                attrs.get("best_known_length", cost) or cost
            )
            efficiency = _clamp01(
                best_len / max(cost, 1e-9)
            )
            status = str(
                attrs.get("promotion_status", "candidate")
            )
            status_factor = 1.0 if status == "accepted" else 0.65
            score = (
                context_factor
                * status_factor
                * (0.7 * success + 0.3 * efficiency)
            )
            if score > best_score:
                best_score = score
                best_id = node_id
        if best_id is not None:
            self.last_strategy_by_action[int(action)] = best_id
        return best_score

    def _world_model_score(
        self,
        context_signature: str,
        action: int,
    ) -> float:
        role_ids = {
            str(item["node_id"])
            for item in self.find_similar_roles(
                context_signature,
                action,
            )
        }
        concept_ids: set[str] = set()
        for role_id in role_ids:
            for edge in self.memory.edges_from(
                role_id,
                "transfers_to",
            ):
                target = str(edge["target_node_id"])
                if self._usable(target):
                    concept_ids.add(target)
        if not concept_ids:
            return 0.0

        best = 0.0
        for model in self.memory.query_nodes(
            memory_level="M5",
            node_type="WorldModelFragment",
        ):
            model_id = str(model["node_id"])
            if not self._usable(model_id):
                continue
            attrs = dict(model.get("attrs") or {})
            overlap = concept_ids & {
                str(value)
                for value in attrs.get("concept_ids", []) or []
            }
            if not overlap:
                continue
            relations = self.memory.connection.execute(
                """
                SELECT relation_type, support_count, confidence
                FROM world_model_relations_v621
                WHERE model_id=?
                """,
                (model_id,),
            ).fetchall()
            if not relations:
                continue
            predictive = [
                row
                for row in relations
                if str(row[0])
                in {
                    "precedes",
                    "enables",
                    "constrains",
                    "shared_outcome",
                }
            ]
            if not predictive:
                continue
            relation_strength = sum(
                float(row[2] or 0.0)
                * min(1.0, float(row[1] or 0) / 3.0)
                for row in predictive
            ) / len(predictive)
            contexts = {
                str(value)
                for value in attrs.get("supported_contexts", []) or []
            }
            context_factor = (
                1.0
                if not contexts or context_signature in contexts
                else 0.5
            )
            status = str(
                attrs.get("promotion_status", "candidate")
            )
            status_factor = 1.0 if status == "accepted" else 0.6
            best = max(
                best,
                context_factor
                * status_factor
                * _clamp01(relation_strength),
            )
        return best


class V621SnapshotMemoryQueryEngine:
    """RAM-only v6.2.1 wrapper for an existing SnapshotMemoryQueryEngine."""

    def __init__(self, base_engine: Any) -> None:
        self.base = base_engine
        self.snapshot = base_engine.snapshot
        self.overlay = base_engine.overlay
        self.last_strategy_by_action: dict[int, str] = {}
        self._status_by_id: dict[str, tuple[str, str]] = {}
        self._world_models: dict[str, dict[str, Any]] = {}
        self._relations_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._strategies: dict[str, dict[str, Any]] = {}
        self._load_extension_indexes()

    @classmethod
    def from_existing(
        cls,
        engine: Any,
    ) -> "V621SnapshotMemoryQueryEngine":
        if isinstance(engine, cls):
            return engine
        return cls(engine)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def apply_live_overlay(self, overlay: Any) -> Any:
        return self.base.apply_live_overlay(overlay)

    def record_selected_action_query(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        method = getattr(
            self.base,
            "record_selected_action_query",
            None,
        )
        if method is None:
            return None
        return method(*args, **kwargs)

    def predict_family(
        self,
        context_signatures: dict[int, tuple],
        action: int,
        *,
        record_query: bool = False,
    ) -> MemoryPrediction:
        exact_method = getattr(self.base, "_exact_contingency", None)
        if exact_method is not None:
            exact = exact_method(context_signatures, action)
            if exact is not None:
                return MemoryPrediction(
                    predicted_family=int(
                        exact.get(
                            "family",
                            exact.get("transformation_family", 0),
                        )
                    ),
                    confidence=float(exact.get("confidence", 0.0) or 0.0),
                    source="memory_contingency",
                    evidence_node_ids=[
                        str(exact.get("node_id", ""))
                    ],
                )
        contingency_method = getattr(self.base, "_contingency", None)
        if contingency_method is not None:
            stable = contingency_method(context_signatures, action)
            if stable is not None:
                return MemoryPrediction(
                    predicted_family=int(
                        stable.get(
                            "family",
                            stable.get("transformation_family", 0),
                        )
                    ),
                    confidence=float(
                        stable.get("confidence", 0.0) or 0.0
                    ),
                    source="contingency_learner",
                    evidence_node_ids=[],
                )

        best_context = self._best_context_signature(
            context_signatures,
            action,
        )
        roles = self.find_similar_roles(best_context, action)
        concepts = self.find_concept_matches(
            best_context,
            action,
            role_matches=roles,
        )
        if roles and roles[0].get("family_id") is not None:
            return MemoryPrediction(
                predicted_family=int(roles[0]["family_id"]),
                confidence=float(roles[0].get("score", 0.0) or 0.0),
                source="role_match",
                evidence_node_ids=[str(roles[0]["node_id"])],
            )
        if concepts and concepts[0].get("family_id") is not None:
            return MemoryPrediction(
                predicted_family=int(concepts[0]["family_id"]),
                confidence=float(
                    concepts[0].get("score", 0.0) or 0.0
                ),
                source="concept_match",
                evidence_node_ids=[str(concepts[0]["node_id"])],
            )
        return MemoryPrediction(
            predicted_family=None,
            confidence=0.0,
            source="none",
            evidence_node_ids=[],
        )

    def find_similar_roles(
        self,
        context_signature: str,
        action: int,
    ) -> list[dict]:
        return [
            item
            for item in self.base.find_similar_roles(
                context_signature,
                action,
            )
            if self._usable(str(item["node_id"]))
        ]

    def find_concept_matches(
        self,
        context_signature: str,
        action: int,
        *,
        role_matches: list[dict] | None = None,
    ) -> list[dict]:
        method = self.base.find_concept_matches
        try:
            rows = method(
                context_signature,
                action,
                role_matches=role_matches,
            )
        except TypeError:
            rows = method(context_signature, action)
        output = []
        for item in rows:
            node_id = str(item["node_id"])
            if not self._usable(node_id):
                continue
            promotion = self._status_by_id.get(
                node_id,
                ("candidate", "active"),
            )[0]
            adjusted = dict(item)
            if promotion == "candidate":
                adjusted["score"] = float(
                    adjusted.get("score", 0.0)
                ) * 0.6
            output.append(adjusted)
        return sorted(
            output,
            key=lambda item: (
                -float(item.get("score", 0.0)),
                str(item.get("node_id")),
            ),
        )

    def score_action(
        self,
        context_signatures: dict[int, tuple],
        action: int,
        available_actions: list[int],
        *,
        record_query: bool = False,
    ) -> MemoryActionScore:
        prediction = self.predict_family(
            context_signatures,
            action,
            record_query=record_query,
        )
        best_context = self._best_context_signature(
            context_signatures,
            action,
        )
        future_failure = getattr(
            self.base,
            "_future_and_failure",
            None,
        )
        if future_failure is not None:
            context_tuple = (
                tuple(json.loads(best_context))
                if best_context.startswith("[")
                else (best_context,)
            )
            future, failure = future_failure(
                context_tuple,
                action,
            )
        else:
            future = {
                "expected_future_option_delta": 0.0,
                "completion_likelihood": 0.0,
                "sources": [],
            }
            failure = {
                "failure_risk": 0.0,
                "contradiction_evidence": False,
                "sources": [],
            }
        roles = self.find_similar_roles(best_context, action)
        concepts = self.find_concept_matches(
            best_context,
            action,
            role_matches=roles,
        )
        base_score = compute_memory_action_score(
            action=action,
            prediction=prediction,
            future_option_evidence=future,
            failure_evidence=failure,
            role_matches=roles,
            concept_matches=concepts,
        )
        strategy_score = self._strategy_score(
            best_context,
            action,
        )
        world_score = self._world_model_score(
            best_context,
            action,
        )
        score = _clamp01(
            float(base_score.score)
            + 0.10 * strategy_score
            + 0.07 * world_score
        )
        sources = list(base_score.evidence_sources)
        if strategy_score > 0:
            sources.append("M6_strategy_memory_v621")
        if world_score > 0:
            sources.append("M5_world_model_memory_v621")
        return replace(
            base_score,
            score=score,
            evidence_sources=sources,
        )

    def rank_actions(
        self,
        context_signatures_by_action: dict[int, dict[int, tuple]],
        available_actions: list[int],
    ) -> list[MemoryActionScore]:
        scores = [
            self.score_action(
                context_signatures_by_action[int(action)],
                int(action),
                available_actions,
            )
            for action in sorted(
                int(item) for item in available_actions
            )
            if int(action) in context_signatures_by_action
        ]
        return sorted(
            scores,
            key=lambda item: (
                -float(item.score),
                int(item.action),
            ),
        )

    def _load_extension_indexes(self) -> None:
        source_dir = getattr(
            self.snapshot,
            "source_memory_dir",
            None,
        )
        if not source_dir:
            return
        path = Path(source_dir) / "current_state.sqlite"
        if not path.exists():
            return
        connection = sqlite3.connect(
            f"file:{path.resolve()}?mode=ro",
            uri=True,
            timeout=10.0,
        )
        try:
            connection.row_factory = sqlite3.Row
            node_columns = _table_columns(
                connection,
                "memory_nodes",
            )
            if node_columns:
                rows = connection.execute(
                    """
                    SELECT node_id, memory_level, node_type,
                           attrs_json, status
                    FROM memory_nodes
                    WHERE memory_level IN ('M1','M2','M3','M4','M5','M6')
                    """
                ).fetchall()
                for row in rows:
                    try:
                        attrs = json.loads(
                            str(row["attrs_json"] or "{}")
                        )
                    except json.JSONDecodeError:
                        attrs = {}
                    node_id = str(row["node_id"])
                    promotion = str(
                        attrs.get(
                            "promotion_status",
                            "candidate",
                        )
                    )
                    node_status = str(row["status"] or "active")
                    self._status_by_id[node_id] = (
                        promotion,
                        node_status,
                    )
                    if str(row["memory_level"]) == "M5":
                        self._world_models[node_id] = attrs
                    elif str(row["memory_level"]) == "M6":
                        self._strategies[node_id] = attrs
            if _table_columns(
                connection,
                "world_model_relations_v621",
            ):
                for row in connection.execute(
                    """
                    SELECT model_id, relation_type,
                           support_count, confidence
                    FROM world_model_relations_v621
                    """
                ).fetchall():
                    self._relations_by_model[
                        str(row["model_id"])
                    ].append(
                        {
                            "relation_type": str(
                                row["relation_type"]
                            ),
                            "support_count": int(
                                row["support_count"] or 0
                            ),
                            "confidence": float(
                                row["confidence"] or 0.0
                            ),
                        }
                    )
        finally:
            connection.close()

    def _usable(self, node_id: str) -> bool:
        promotion, state = self._status_by_id.get(
            str(node_id),
            ("candidate", "active"),
        )
        return (
            promotion not in {"rejected", "superseded"}
            and state != "forgotten"
        )

    def _best_context_signature(
        self,
        context_signatures: dict[int, tuple],
        action: int,
    ) -> str:
        method = getattr(
            self.base,
            "_best_context_signature",
            None,
        )
        if method is not None:
            value = method(context_signatures, action)
            if isinstance(value, tuple):
                return json.dumps(list(value))
            return str(value)
        if not context_signatures:
            return json.dumps([int(action)])
        return json.dumps(
            list(
                context_signatures[
                    max(int(key) for key in context_signatures)
                ]
            )
        )

    def _strategy_score(
        self,
        context_signature: str,
        action: int,
    ) -> float:
        best = 0.0
        best_id: str | None = None
        for node_id, attrs in self._strategies.items():
            if not self._usable(node_id):
                continue
            sequence = attrs.get("action_sequence") or []
            if not sequence or int(sequence[0]) != int(action):
                continue
            context = str(attrs.get("context_key") or "")
            context_factor = (
                1.0
                if not context or context == context_signature
                else 0.5
            )
            success = _clamp01(attrs.get("success_rate"))
            cost = float(
                attrs.get("cost", len(sequence))
                or len(sequence)
                or 1
            )
            best_len = float(
                attrs.get("best_known_length", cost) or cost
            )
            efficiency = _clamp01(
                best_len / max(cost, 1e-9)
            )
            promotion = self._status_by_id.get(
                node_id,
                ("candidate", "active"),
            )[0]
            status_factor = (
                1.0 if promotion == "accepted" else 0.65
            )
            score = (
                context_factor
                * status_factor
                * (0.7 * success + 0.3 * efficiency)
            )
            if score > best:
                best = score
                best_id = node_id
        if best_id is not None:
            self.last_strategy_by_action[int(action)] = best_id
        return best

    def _world_model_score(
        self,
        context_signature: str,
        action: int,
    ) -> float:
        role_matches = self.find_similar_roles(
            context_signature,
            action,
        )
        role_ids = {
            str(item["node_id"])
            for item in role_matches
        }
        concept_ids: set[str] = set()
        concept_map = getattr(
            self.snapshot,
            "concept_ids_by_role",
            {},
        )
        for role_id in role_ids:
            concept_ids.update(
                str(value)
                for value in concept_map.get(role_id, ())
                if self._usable(str(value))
            )
        best = 0.0
        for model_id, attrs in self._world_models.items():
            if not self._usable(model_id):
                continue
            overlap = concept_ids & {
                str(value)
                for value in attrs.get("concept_ids", []) or []
            }
            if not overlap:
                continue
            relations = [
                row
                for row in self._relations_by_model.get(
                    model_id,
                    (),
                )
                if row["relation_type"]
                in {
                    "precedes",
                    "enables",
                    "constrains",
                    "shared_outcome",
                }
            ]
            if not relations:
                continue
            strength = sum(
                float(row["confidence"])
                * min(
                    1.0,
                    float(row["support_count"]) / 3.0,
                )
                for row in relations
            ) / len(relations)
            contexts = {
                str(value)
                for value in attrs.get(
                    "supported_contexts",
                    [],
                )
                or []
            }
            context_factor = (
                1.0
                if not contexts
                or context_signature in contexts
                else 0.5
            )
            promotion = self._status_by_id.get(
                model_id,
                ("candidate", "active"),
            )[0]
            status_factor = (
                1.0 if promotion == "accepted" else 0.6
            )
            best = max(
                best,
                context_factor
                * status_factor
                * _clamp01(strength),
            )
        return best


class V621MemoryController(MemoryController):
    """Canonical runtime facade for v6.2.1."""

    def __init__(
        self,
        memory: MemorySubstrate,
        *,
        contingency_learner: Any = None,
        graph: Any = None,
        query_engine: Any | None = None,
        promotion_engine: Any | None = None,
        context_head: Any | None = None,
        carrier_head: Any | None = None,
        lifecycle_head: Any | None = None,
        efficiency_head: Any | None = None,
    ) -> None:
        migrate_v621(memory.connection)
        query = V621MemoryQueryEngine(
            memory,
            contingency_learner=contingency_learner,
            graph=graph,
        )
        base_promotion = (
            promotion_engine
            if isinstance(
                promotion_engine,
                MemoryPromotionEngine,
            )
            else MemoryPromotionEngine(memory)
        )
        promotion = V621PromotionEngine(
            memory,
            base=base_promotion,
        )
        super().__init__(
            memory,
            contingency_learner=contingency_learner,
            graph=graph,
            query_engine=query,
            promotion_engine=promotion,
        )
        self.context_head = context_head
        self.carrier_head = carrier_head
        self.lifecycle_head = lifecycle_head
        self.efficiency_head = efficiency_head
        self.significance = promotion.significance
        self._last_ranked_scores: dict[int, MemoryActionScore] = {}

    def adapt_query_engine(self, engine: Any) -> Any:
        if isinstance(engine, V621MemoryQueryEngine):
            return engine
        if isinstance(engine, V621SnapshotMemoryQueryEngine):
            return engine
        if hasattr(engine, "snapshot") and hasattr(
            engine,
            "rank_actions",
        ):
            return V621SnapshotMemoryQueryEngine.from_existing(
                engine
            )
        return self.query_engine

    def predict(
        self,
        context_signatures: dict[int, tuple],
        action: int,
        *,
        record_query: bool = False,
    ) -> MemoryPrediction:
        return self.query_engine.predict_family(
            context_signatures,
            int(action),
            record_query=record_query,
        )

    def choose_action_candidates(
        self,
        context_signatures_by_action: dict[int, dict[int, tuple]],
        available_actions: list[int],
    ) -> list[MemoryActionScore]:
        ranked = self.query_engine.rank_actions(
            context_signatures_by_action,
            available_actions,
        )
        self._last_ranked_scores = {
            int(item.action): item
            for item in ranked
        }
        return ranked

    def choose_with_sampler_prior(
        self,
        *,
        context_signatures_by_action: dict[int, dict[int, tuple]],
        available_actions: list[int],
        sampler_action: int,
        override_margin: float = 0.15,
    ) -> int:
        ranked = self.choose_action_candidates(
            context_signatures_by_action,
            available_actions,
        )
        if not ranked:
            return int(sampler_action)
        top = ranked[0]
        baseline = self._last_ranked_scores.get(
            int(sampler_action)
        )
        baseline_score = (
            float(baseline.score)
            if baseline is not None
            else 0.0
        )
        if (
            int(top.action) != int(sampler_action)
            and float(top.score)
            >= baseline_score + max(0.0, float(override_margin))
        ):
            self._audit(
                "sampler_memory_override",
                owner_id=str(top.action),
                payload={
                    "sampler_action": int(sampler_action),
                    "memory_action": int(top.action),
                    "memory_score": float(top.score),
                    "sampler_action_memory_score": baseline_score,
                    "override_margin": float(override_margin),
                },
            )
            return int(top.action)
        return int(sampler_action)

    def current_isf_weights(self) -> dict[str, float]:
        return self.significance.current_isf_weights()

    def promote_candidates(
        self,
        *,
        step: int | None = None,
    ) -> dict[str, Any]:
        return self.promotion_engine.run_all(step=step)

    def record_prediction_outcome(
        self,
        *,
        prediction: MemoryPrediction | None,
        success: bool | None,
        game: str | None,
        context_key: str,
        context_signatures: dict[int, tuple] | None = None,
        action: int,
        actual_family: str | int | None,
        global_step: int | None,
    ) -> None:
        if actual_family is None:
            return

        attempts: dict[str, tuple[str | int | None, bool, str]] = {}
        if (
            prediction is not None
            and success is not None
            and prediction.source == "concept_match"
        ):
            for concept_id in prediction.evidence_node_ids:
                if str(concept_id).startswith("M4:"):
                    attempts[str(concept_id)] = (
                        prediction.predicted_family,
                        bool(success),
                        "runtime_selected_concept_prediction",
                    )

        if context_signatures:
            best_context = json.dumps(
                list(
                    context_signatures[
                        max(int(level) for level in context_signatures)
                    ]
                )
            )
            finder = getattr(
                self.query_engine,
                "find_concept_matches",
                None,
            )
            if finder is not None:
                try:
                    matches = finder(best_context, int(action))
                except TypeError:
                    matches = []
                for match in matches:
                    concept_id = str(match.get("node_id") or "")
                    family_id = match.get("family_id")
                    if not concept_id.startswith("M4:") or family_id is None:
                        continue
                    attempts.setdefault(
                        concept_id,
                        (
                            family_id,
                            str(family_id) == str(actual_family),
                            "runtime_counterfactual_concept_test",
                        ),
                    )

        for concept_id, (
            predicted_family,
            attempt_success,
            evidence_source,
        ) in attempts.items():
            attempt_id = "concept_transfer:" + sha1(
                _json(
                    {
                        "concept_id": concept_id,
                        "game": game,
                        "context": context_key,
                        "action": int(action),
                        "global_step": global_step,
                    }
                ).encode("utf-8")
            ).hexdigest()[:24]
            self.memory.connection.execute(
                """
                INSERT OR REPLACE INTO concept_transfer_attempts_v621(
                    attempt_id, concept_id, game, context_key,
                    action, predicted_family, actual_family,
                    success, evidence_source, global_step, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt_id,
                    concept_id,
                    None if game is None else str(game),
                    str(context_key),
                    int(action),
                    None
                    if predicted_family is None
                    else str(predicted_family),
                    str(actual_family),
                    int(bool(attempt_success)),
                    evidence_source,
                    global_step,
                    time.time(),
                ),
            )
        self.memory.connection.commit()

    def record_selected_action_outcome(
        self,
        *,
        action: int,
        success: bool | None,
        game: str | None,
        level_key: str | None,
        context_key: str | None,
        cost: float | None,
        epoch: int | None,
        global_step: int | None,
    ) -> None:
        if success is None:
            return
        strategy_map = getattr(
            self.query_engine,
            "last_strategy_by_action",
            {},
        )
        strategy_id = strategy_map.get(int(action))
        if not strategy_id:
            return
        self.record_strategy_reuse(
            strategy_id=str(strategy_id),
            game=game,
            level_key=level_key,
            context_key=context_key,
            success=bool(success),
            cost=cost,
            epoch=epoch,
            global_step=global_step,
        )

    def record_selected_action_query(
        self,
        *,
        context_signatures: dict[int, tuple],
        action: int,
        prediction: MemoryPrediction | None = None,
    ) -> None:
        method = getattr(
            self.query_engine,
            "record_selected_action_query",
            None,
        )
        if method is None:
            return
        method(
            context_signatures=context_signatures,
            action=action,
            prediction=prediction,
        )

    def record_prediction_result(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if self.context_head is None:
            return None
        return self.context_head.record_prediction_result(
            *args,
            **kwargs,
        )

    def should_expand_context(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        if self.context_head is None:
            return False
        return bool(
            self.context_head.should_expand_context(
                *args,
                **kwargs,
            )
        )

    def context_summary(self) -> dict[str, Any]:
        if self.context_head is None:
            return {}
        return dict(self.context_head.summary())

    def record_carrier_interaction(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if self.carrier_head is None:
            return None
        return self.carrier_head.record_interaction(
            *args,
            **kwargs,
        )

    def carrier_stats(
        self,
        signature: str,
    ) -> dict[str, Any]:
        if self.carrier_head is None:
            return {}
        return dict(
            self.carrier_head.stats_for_carrier(signature)
        )

    def import_carrier_candidate(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if self.carrier_head is None:
            return None
        return self.carrier_head.import_candidate(
            *args,
            **kwargs,
        )

    def register_interaction(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if self.lifecycle_head is None:
            raise RuntimeError(
                "v6.2.1 lifecycle head is not configured"
            )
        return self.lifecycle_head.register_interaction(
            *args,
            **kwargs,
        )

    @property
    def replay_candidates(self) -> Any:
        if self.lifecycle_head is None:
            return {}
        return self.lifecycle_head.replay_candidates

    @property
    def lifecycle_records(self) -> Any:
        if self.lifecycle_head is None:
            return {}
        return self.lifecycle_head.records

    def import_lifecycle_record(self, record: Any) -> None:
        if self.lifecycle_head is not None:
            self.lifecycle_head.import_record(record)

    def import_replay_candidate(self, candidate: Any) -> None:
        if self.lifecycle_head is not None:
            self.lifecycle_head.import_replay_candidate(candidate)

    def apply_post_factum_credit(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if self.lifecycle_head is None:
            return None
        return self.lifecycle_head.apply_post_factum_credit(
            *args,
            **kwargs,
        )

    def record_efficiency_interaction(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if self.efficiency_head is None:
            raise RuntimeError(
                "v6.2.1 efficiency head is not configured"
            )
        return self.efficiency_head.record_interaction(
            *args,
            **kwargs,
        )

    def query_replay_candidates(
        self,
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        rows = self.memory.connection.execute(
            """
            SELECT node_id, replay_priority, retention_status,
                   memory_state, forgetting_reason
            FROM memory_scores
            WHERE COALESCE(replay_priority, 0.0) > 0.0
              AND COALESCE(memory_state, 'active') != 'forgotten'
            ORDER BY replay_priority DESC, node_id ASC
            LIMIT ?
            """,
            (max(0, int(limit)),),
        ).fetchall()
        return [
            {
                "memory_id": str(row[0]),
                "replay_priority": float(row[1] or 0.0),
                "retention_status": row[2],
                "memory_state": row[3],
                "reason": row[4],
            }
            for row in rows
        ]

    def _audit(
        self,
        event_type: str,
        *,
        owner_id: str | None,
        payload: dict[str, Any],
        global_step: int | None = None,
    ) -> None:
        audit_id = "v621audit:" + sha1(
            _json(
                {
                    "event_type": event_type,
                    "owner_id": owner_id,
                    "payload": payload,
                    "time_bucket": int(time.time()),
                }
            ).encode("utf-8")
        ).hexdigest()[:24]
        self.memory.connection.execute(
            """
            INSERT OR REPLACE INTO memory_runtime_audit_v621(
                audit_id, event_type, owner_id,
                payload_json, global_step, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                audit_id,
                str(event_type),
                owner_id,
                _json(payload),
                global_step,
                time.time(),
            ),
        )
        self.memory.connection.commit()
