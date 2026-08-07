from __future__ import annotations

import json
import sqlite3
from hashlib import sha1
from pathlib import Path
from typing import Any, Iterable

from v6.memory.promotion_engine import MemoryPromotionEngine
from v6.memory.query_engine import MemoryQueryEngine
from v6.memory.substrate import (
    MemoryEdge,
    MemoryEvidence,
    MemoryLifecycleEvent,
    MemoryNode,
    MemoryScore,
    MemorySubstrate,
    strategy_node_id,
)


class MemoryController:
    """Single runtime facade over write, query, promotion and lifecycle APIs."""

    def __init__(
        self,
        memory: MemorySubstrate,
        *,
        contingency_learner: Any = None,
        graph: Any = None,
        query_engine: Any | None = None,
        promotion_engine: Any | None = None,
    ) -> None:
        self.memory = memory
        self.query_engine = query_engine or MemoryQueryEngine(
            memory,
            contingency_learner=contingency_learner,
            graph=graph,
        )
        self.promotion_engine = (
            promotion_engine or MemoryPromotionEngine(memory)
        )

    def observe_interaction(
        self,
        node: MemoryNode,
        *,
        evidence: Iterable[MemoryEvidence] = (),
        score: MemoryScore | None = None,
        edges: Iterable[MemoryEdge] = (),
        step: int | None = None,
    ) -> None:
        self.memory.upsert_node(node, step=step)
        for item in evidence:
            self.memory.add_evidence(item)
        for edge in edges:
            self.memory.upsert_edge(edge)
        if score is not None:
            self.memory.upsert_score(score, step=step)

    def predict(
        self,
        context_signatures: dict[int, tuple],
        action: int,
    ) -> Any:
        return self.query_engine.predict_family(
            context_signatures,
            int(action),
            record_query=True,
        )

    def choose_action_candidates(
        self,
        context_signatures_by_action: dict[int, dict[int, tuple]],
        available_actions: list[int],
    ) -> list[Any]:
        return self.query_engine.rank_actions(
            context_signatures_by_action,
            available_actions,
        )

    def score_replay(self, memory_id: str) -> float:
        row = self.memory.connection.execute(
            """
            SELECT replay_priority
            FROM memory_scores
            WHERE node_id=?
            """,
            (str(memory_id),),
        ).fetchone()
        return 0.0 if row is None or row[0] is None else float(row[0])

    def promote_candidates(
        self,
        *,
        step: int | None = None,
    ) -> dict[str, Any]:
        return self.promotion_engine.run_all(step=step)

    def retrieve_similar(
        self,
        context_signature: str,
        action: int,
    ) -> dict[str, list[dict[str, Any]]]:
        return {
            "roles": self.query_engine.find_similar_roles(
                context_signature, int(action)
            ),
            "concepts": self.query_engine.find_concept_matches(
                context_signature, int(action)
            ),
            "contingencies": self.query_similar_contingencies(
                context_signature, int(action)
            ),
        }

    def query_prediction(
        self,
        context_signatures: dict[int, tuple],
        action: int,
    ) -> Any:
        return self.predict(context_signatures, action)

    def query_action_priors(
        self,
        context_signatures_by_action: dict[int, dict[int, tuple]],
        available_actions: list[int],
    ) -> list[Any]:
        return self.choose_action_candidates(
            context_signatures_by_action,
            available_actions,
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

    def query_similar_contingencies(
        self,
        context_signature: str,
        action: int,
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        target_context = str(context_signature)
        output: list[dict[str, Any]] = []
        for node in self.memory.query_nodes(
            memory_level="M1",
            node_type="ContingencyMemory",
        ):
            attrs = dict(node.get("attrs") or {})
            if int(attrs.get("action", -1)) != int(action):
                continue
            candidate_context = str(
                attrs.get("context_signature") or ""
            )
            exact = candidate_context == target_context
            overlap = _context_overlap(
                candidate_context,
                target_context,
            )
            output.append(
                {
                    **node,
                    "exact_context_match": exact,
                    "context_overlap": overlap,
                }
            )
        output.sort(
            key=lambda item: (
                not bool(item["exact_context_match"]),
                -float(item["context_overlap"]),
                -float(
                    item.get("attrs", {}).get("confidence", 0.0) or 0.0
                ),
                str(item["node_id"]),
            )
        )
        return output[: max(0, int(limit))]

    def query_successful_trajectories(
        self,
        *,
        game: str | None = None,
        level_key: str | None = None,
        context_key: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT node_id, canonical_key, attrs_json
            FROM memory_nodes
            WHERE memory_level='M6'
              AND node_type='EfficientStrategyMemory'
              AND status IN ('active', 'protected')
        """
        rows = self.memory.connection.execute(query).fetchall()
        output: list[dict[str, Any]] = []
        for row in rows:
            try:
                attrs = json.loads(str(row[2] or "{}"))
            except json.JSONDecodeError:
                attrs = {}
            if game is not None and str(attrs.get("game")) != str(game):
                continue
            if (
                level_key is not None
                and str(attrs.get("level_key")) != str(level_key)
            ):
                continue
            if (
                context_key is not None
                and str(attrs.get("context_key")) != str(context_key)
            ):
                continue
            output.append(
                {
                    "strategy_id": str(row[0]),
                    "canonical_key": row[1],
                    **attrs,
                }
            )
        output.sort(
            key=lambda item: (
                -float(item.get("success_rate", 0.0) or 0.0),
                float(item.get("cost", float("inf")) or float("inf")),
                -int(item.get("reuse_count", 0) or 0),
                str(item["strategy_id"]),
            )
        )
        return output[: max(0, int(limit))]

    def query_contradictions(
        self,
        context_key: str | None = None,
        action: int | None = None,
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for edge in self.memory.connection.execute(
            """
            SELECT source_node_id, target_node_id, edge_type,
                   weight, support_count, evidence_json
            FROM memory_edges
            WHERE edge_type IN ('violates_prediction', 'contradicts')
            ORDER BY support_count DESC, weight DESC
            LIMIT ?
            """,
            (max(0, int(limit * 4)),),
        ).fetchall():
            try:
                evidence = json.loads(str(edge[5] or "{}"))
            except json.JSONDecodeError:
                evidence = {}
            if (
                context_key is not None
                and str(evidence.get("context_signature"))
                != str(context_key)
            ):
                continue
            if (
                action is not None
                and evidence.get("action") is not None
                and int(evidence.get("action")) != int(action)
            ):
                continue
            output.append(
                {
                    "source_node_id": str(edge[0]),
                    "target_node_id": str(edge[1]),
                    "edge_type": str(edge[2]),
                    "weight": float(edge[3] or 0.0),
                    "support_count": int(edge[4] or 0),
                    "evidence": evidence,
                }
            )
        return output[: max(0, int(limit))]

    def query_future_option_effects(
        self,
        context_signatures: dict[int, tuple],
        action: int,
    ) -> dict[str, Any]:
        best_context = json.dumps(
            list(context_signatures[max(context_signatures)])
        )
        return self.query_engine.find_future_option_evidence(
            best_context,
            int(action),
        )

    def query_strategies(
        self,
        *,
        game: str | None = None,
        level_key: str | None = None,
        context_key: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return self.query_successful_trajectories(
            game=game,
            level_key=level_key,
            context_key=context_key,
            limit=limit,
        )

    def record_contradiction(
        self,
        *,
        interaction_id: str,
        context_key: str,
        action_key: str,
        predicted_family: str | None,
        actual_family: str | None,
        confidence: float | None,
        epoch: int | None = None,
        global_step: int | None = None,
    ) -> str:
        payload = {
            "context_signature": context_key,
            "action": action_key,
            "predicted_family": predicted_family,
            "actual_family": actual_family,
            "confidence": confidence,
        }
        contradiction_id = (
            "M1:violation:"
            + sha1(
                json.dumps(payload, sort_keys=True).encode("utf-8")
            ).hexdigest()[:20]
        )
        self.memory.upsert_node(
            MemoryNode(
                node_id=contradiction_id,
                memory_level="M1",
                node_type="PredictionContradictionMemory",
                canonical_key=json.dumps(payload, sort_keys=True),
                attrs=payload,
                created_epoch=epoch,
                updated_epoch=epoch,
            ),
            step=global_step,
        )
        self.memory.upsert_edge(
            MemoryEdge(
                source_node_id=str(interaction_id),
                target_node_id=contradiction_id,
                edge_type="violates_prediction",
                weight=float(confidence or 1.0),
                edge_confidence=float(confidence or 1.0),
                edge_source="MemoryController.record_contradiction",
                evidence=payload,
            )
        )
        return contradiction_id

    def record_context_split(
        self,
        *,
        parent_context_key: str,
        child_context_key: str,
        action_key: str | None,
        contradiction_key: str | None,
        differentiating_features: list[Any],
        prediction_lift_before: float | None = None,
        prediction_lift_after: float | None = None,
        validation_status: str = "candidate",
        epoch: int | None = None,
        global_step: int | None = None,
    ) -> str:
        split_id = (
            "context_split:"
            + sha1(
                f"{parent_context_key}|{child_context_key}|{action_key}".encode(
                    "utf-8"
                )
            ).hexdigest()[:20]
        )
        self.memory.record_context_split(
            split_id=split_id,
            parent_context_key=parent_context_key,
            child_context_key=child_context_key,
            action_key=action_key,
            contradiction_key=contradiction_key,
            differentiating_features=differentiating_features,
            prediction_lift_before=prediction_lift_before,
            prediction_lift_after=prediction_lift_after,
            validation_status=validation_status,
            epoch=epoch,
            global_step=global_step,
        )
        return split_id

    def record_success_trajectory(
        self,
        *,
        game: str,
        level_key: str | None,
        context_key: str | None,
        action_sequence: list[int],
        preconditions: dict[str, Any],
        effects: dict[str, Any],
        cost: float,
        success_rate: float = 1.0,
        best_known_length: int | None = None,
        epoch: int | None = None,
        global_step: int | None = None,
    ) -> str:
        signature_payload = {
            "game": game,
            "level_key": level_key,
            "context_key": context_key,
            "action_sequence": list(action_sequence),
            "preconditions": preconditions,
            "effects": effects,
        }
        canonical = json.dumps(signature_payload, sort_keys=True)
        strategy_id = strategy_node_id(canonical)
        existing = self.memory.get_node(strategy_id)
        old_attrs = dict((existing or {}).get("attrs") or {})
        attrs = {
            **old_attrs,
            **signature_payload,
            "action_sequence_signature": sha1(
                json.dumps(action_sequence).encode("utf-8")
            ).hexdigest()[:20],
            "cost": float(cost),
            "success_rate": float(success_rate),
            "reuse_count": int(old_attrs.get("reuse_count", 0) or 0),
            "failure_count": int(old_attrs.get("failure_count", 0) or 0),
            "best_known_length": (
                int(best_known_length)
                if best_known_length is not None
                else len(action_sequence)
            ),
        }
        self.memory.upsert_node(
            MemoryNode(
                node_id=strategy_id,
                memory_level="M6",
                node_type="EfficientStrategyMemory",
                canonical_key=canonical,
                attrs=attrs,
                created_epoch=epoch,
                updated_epoch=epoch,
                status="active",
            ),
            step=global_step,
            support_increment=1,
        )
        return strategy_id

    def record_strategy_reuse(
        self,
        *,
        strategy_id: str,
        game: str | None,
        level_key: str | None,
        context_key: str | None,
        success: bool,
        cost: float | None,
        epoch: int | None,
        global_step: int | None,
    ) -> None:
        event_id = (
            "strategy_reuse:"
            + sha1(
                f"{strategy_id}|{game}|{level_key}|{global_step}".encode(
                    "utf-8"
                )
            ).hexdigest()[:20]
        )
        self.memory.record_strategy_reuse(
            reuse_event_id=event_id,
            strategy_id=strategy_id,
            game=game,
            level_key=level_key,
            context_key=context_key,
            success=success,
            cost=cost,
            epoch=epoch,
            global_step=global_step,
        )
        node = self.memory.get_node(strategy_id)
        if node is not None:
            attrs = dict(node.get("attrs") or {})
            key = "reuse_count" if success else "failure_count"
            attrs[key] = int(attrs.get(key, 0) or 0) + 1
            self.memory.update_node_support_and_attrs(
                strategy_id,
                attrs,
                step=global_step,
            )


def _context_overlap(left: str, right: str) -> float:
    try:
        left_items = set(json.loads(left))
        right_items = set(json.loads(right))
    except (TypeError, ValueError, json.JSONDecodeError):
        left_items = set(str(left).split("|"))
        right_items = set(str(right).split("|"))
    union = left_items | right_items
    return len(left_items & right_items) / len(union) if union else 0.0
