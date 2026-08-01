from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha1
from typing import Any

from v6.memory.substrate import MemoryEdge, MemoryNode, MemorySubstrate, action_node_id, family_node_id


@dataclass(frozen=True)
class MemoryPrediction:
    predicted_family: int | None
    confidence: float
    source: str
    evidence_node_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MemoryActionScore:
    action: int
    score: float
    predicted_family: int | None
    expected_future_option_delta: float | None
    failure_risk: float
    completion_likelihood: float
    evidence_sources: list[str] = field(default_factory=list)


def score_role_match(*, base_transfer_score: float, action_match: bool, context_match: bool) -> float:
    """Pure role scoring shared by SQLite and snapshot query paths."""
    score = float(base_transfer_score)
    if action_match:
        score += 0.25
    if context_match:
        score += 0.25
    if not action_match and not context_match:
        score *= 0.25
    return max(0.0, min(1.0, score))


def score_concept_match(*, best_role_score: float, transfer_success_count: float) -> float:
    return max(0.0, min(1.0, float(best_role_score) * min(1.0, float(transfer_success_count) / 3.0)))


def aggregate_future_option_evidence(deltas: list[float]) -> dict[str, Any]:
    positive = sum(1 for value in deltas if float(value) > 0.0)
    return {
        "expected_future_option_delta": sum(deltas) / len(deltas) if deltas else 0.0,
        "completion_likelihood": positive / len(deltas) if deltas else 0.0,
        "sources": ["future_option_memory"] if deltas else [],
    }


def compute_failure_path_evidence(*, deltas: list[float], evidence_count: int, contradiction: bool) -> dict[str, Any]:
    return {
        "failure_risk": sum(1 for value in deltas if float(value) < 0.0) / int(evidence_count) if evidence_count else 0.0,
        "contradiction_evidence": bool(contradiction),
        "sources": ["failure_path_memory"] if evidence_count else [],
    }


def compute_memory_action_score(
    *,
    action: int,
    prediction: MemoryPrediction,
    future_option_evidence: dict[str, Any],
    failure_evidence: dict[str, Any],
    role_matches: list[dict],
    concept_matches: list[dict],
) -> MemoryActionScore:
    """The canonical action-ranking formula; keep scientific weights centralized."""
    transfer_score = max(
        float(role_matches[0].get("score", 0.0)) if role_matches else 0.0,
        float(concept_matches[0].get("score", 0.0)) if concept_matches else 0.0,
    )
    future_gain = float(future_option_evidence.get("expected_future_option_delta", 0.0) or 0.0)
    failure_risk = float(failure_evidence.get("failure_risk", 0.0) or 0.0)
    completion_likelihood = float(future_option_evidence.get("completion_likelihood", 0.0) or 0.0)
    contradiction_risk = 1.0 if failure_evidence.get("contradiction_evidence") else 0.0
    score = (
        0.30 * float(prediction.confidence)
        + 0.25 * max(0.0, future_gain)
        + 0.20 * completion_likelihood
        + 0.15 * transfer_score
        - 0.25 * failure_risk
        - 0.10 * contradiction_risk
    )
    return MemoryActionScore(
        action=int(action),
        score=max(0.0, min(1.0, float(score))),
        predicted_family=prediction.predicted_family,
        expected_future_option_delta=future_gain,
        failure_risk=failure_risk,
        completion_likelihood=completion_likelihood,
        evidence_sources=[prediction.source, *future_option_evidence.get("sources", []), *failure_evidence.get("sources", [])],
    )


def order_memory_action_scores(scores: list[MemoryActionScore]) -> list[MemoryActionScore]:
    return sorted(scores, key=lambda item: (-float(item.score), int(item.action)))


class MemoryQueryEngine:
    def __init__(self, memory: MemorySubstrate, contingency_learner: Any = None, graph: Any = None) -> None:
        self.memory = memory
        self.contingency_learner = contingency_learner
        self.graph = graph

    def predict_family(
        self,
        context_signatures: dict[int, tuple],
        action: int,
        *,
        record_query: bool = False,
    ) -> MemoryPrediction:
        exact = self._exact_contingency_match(context_signatures, action)
        if exact is not None:
            prediction = MemoryPrediction(
                predicted_family=exact["family"],
                confidence=float(exact["confidence"]),
                source="memory_contingency",
                evidence_node_ids=[str(exact["node_id"])],
            )
            if record_query:
                self._record_query_event(action=action, prediction=prediction)
            return prediction
        if self.contingency_learner is not None and hasattr(self.contingency_learner, "best_stable_for_action"):
            stable = self.contingency_learner.best_stable_for_action(context_signatures, action)
            if stable is not None:
                prediction = MemoryPrediction(
                    predicted_family=int(stable.transformation_family),
                    confidence=float(stable.confidence),
                    source="contingency_learner",
                    evidence_node_ids=[],
                )
                if record_query:
                    self._record_query_event(action=action, prediction=prediction)
                return prediction
        best_context_signature = self._best_context_signature(context_signatures, action)
        role_matches = self.find_similar_roles(best_context_signature, action)
        if role_matches and role_matches[0].get("family_id") is not None:
            prediction = MemoryPrediction(
                predicted_family=role_matches[0].get("family_id"),
                confidence=float(role_matches[0].get("score", 0.4)),
                source="role_match",
                evidence_node_ids=[str(role_matches[0]["node_id"])],
            )
            if record_query:
                self._record_query_event(action=action, prediction=prediction)
            return prediction
        concept_matches = self.find_concept_matches(best_context_signature, action)
        if concept_matches and concept_matches[0].get("family_id") is not None:
            prediction = MemoryPrediction(
                predicted_family=concept_matches[0].get("family_id"),
                confidence=float(concept_matches[0].get("score", 0.35)),
                source="concept_match",
                evidence_node_ids=[str(concept_matches[0]["node_id"])],
            )
            if record_query:
                self._record_query_event(action=action, prediction=prediction)
            return prediction
        prediction = MemoryPrediction(predicted_family=None, confidence=0.0, source="none", evidence_node_ids=[])
        if record_query:
            self._record_query_event(action=action, prediction=prediction)
        return prediction

    def score_action(
        self,
        context_signatures: dict[int, tuple],
        action: int,
        available_actions: list[int],
        *,
        record_query: bool = False,
    ) -> MemoryActionScore:
        del available_actions
        prediction = self.predict_family(context_signatures, action, record_query=record_query)
        best_context_signature = self._best_context_signature(context_signatures, action)
        future = self.find_future_option_evidence(best_context_signature, action)
        failure = self.find_failure_path_evidence(best_context_signature, action)
        role_matches = self.find_similar_roles(best_context_signature, action)
        concept_matches = self.find_concept_matches(best_context_signature, action)
        return compute_memory_action_score(
            action=action,
            prediction=prediction,
            future_option_evidence=future,
            failure_evidence=failure,
            role_matches=role_matches,
            concept_matches=concept_matches,
        )

    def rank_actions(
        self,
        context_signatures_by_action: dict[int, dict[int, tuple]],
        available_actions: list[int],
    ) -> list[MemoryActionScore]:
        scores = [
            self.score_action(context_signatures_by_action[int(action)], int(action), available_actions, record_query=False)
            for action in sorted(int(item) for item in available_actions)
            if int(action) in context_signatures_by_action and context_signatures_by_action[int(action)] is not None
        ]
        return order_memory_action_scores(scores)

    def rank_actions_with_shared_context(
        self,
        context_signatures: dict[int, tuple],
        available_actions: list[int],
    ) -> list[MemoryActionScore]:
        scores = [
            self.score_action(context_signatures, int(action), available_actions, record_query=False)
            for action in sorted(int(item) for item in available_actions)
        ]
        return order_memory_action_scores(scores)

    def find_similar_roles(self, context_signature: str, action: int) -> list[dict]:
        matches: list[dict[str, Any]] = []
        context_node_id = self._context_node_id(context_signature)
        target_action_node = action_node_id(action)
        for role in self.memory.query_nodes(memory_level="M3", node_type="FunctionalRoleMemory"):
            base_transfer_score = float(role.get("attrs", {}).get("transfer_score", 0.0) or 0.0)
            carrier_edges = self.memory.edges_to(role["node_id"], "plays_role")
            action_match = False
            context_match = False
            family_ids: list[int] = []
            for edge in carrier_edges:
                carrier_id = str(edge["source_node_id"])
                for family_edge in self.memory.edges_from(carrier_id, "associated_with_family"):
                    if str(family_edge["target_node_id"]).startswith("M2:family:"):
                        family_ids.append(int(str(family_edge["target_node_id"]).split(":")[-1]))
                if any(str(item["target_node_id"]) == context_node_id for item in self.memory.edges_from(carrier_id, "appears_in_context")):
                    context_match = True
                for interaction_edge in self.memory.edges_from(carrier_id, "carried_by"):
                    interaction_id = str(interaction_edge["target_node_id"])
                    if any(str(item["target_node_id"]) == target_action_node for item in self.memory.edges_from(interaction_id, "takes_action")):
                        action_match = True
            score = score_role_match(
                base_transfer_score=base_transfer_score,
                action_match=action_match,
                context_match=context_match,
            )
            if score <= 0.0:
                continue
            matches.append(
                {
                    "node_id": role["node_id"],
                    "score": score,
                    "family_id": family_ids[0] if family_ids else None,
                    "action_match": action_match,
                    "context_match": context_match,
                }
            )
        return sorted(matches, key=lambda item: (-float(item["score"]), str(item["node_id"])))

    def find_concept_matches(self, context_signature: str, action: int) -> list[dict]:
        matches: list[dict[str, Any]] = []
        for concept in self.memory.query_nodes(memory_level="M4", node_type="ConceptMemory"):
            role_edges = self.memory.edges_to(concept["node_id"], "transfers_to")
            best_role_score = 0.0
            family_ids: list[int] = []
            for role_edge in role_edges:
                role_node_id_value = str(role_edge["source_node_id"])
                role_matches = [item for item in self.find_similar_roles(context_signature, action) if str(item["node_id"]) == role_node_id_value]
                if role_matches:
                    if bool(role_matches[0].get("action_match")):
                        best_role_score = max(best_role_score, float(role_matches[0].get("score", 0.0) or 0.0))
                for carrier_edge in self.memory.edges_from(role_node_id_value, "abstracts_from"):
                    for family_edge in self.memory.edges_from(carrier_edge["target_node_id"], "associated_with_family"):
                        if str(family_edge["target_node_id"]).startswith("M2:family:"):
                            family_ids.append(int(str(family_edge["target_node_id"]).split(":")[-1]))
            score = score_concept_match(
                best_role_score=best_role_score,
                transfer_success_count=float(concept.get("attrs", {}).get("transfer_success_count", 0) or 0),
            )
            if score <= 0.0:
                continue
            matches.append(
                {
                    "node_id": concept["node_id"],
                    "score": score,
                    "family_id": family_ids[0] if family_ids else None,
                }
            )
        return sorted(matches, key=lambda item: (-float(item["score"]), str(item["node_id"])))

    def record_selected_action_query(
        self,
        *,
        context_signatures: dict[int, tuple],
        action: int,
        prediction: MemoryPrediction | None = None,
    ) -> None:
        final_prediction = prediction or self.predict_family(context_signatures, action, record_query=False)
        self._record_query_event(action=action, prediction=final_prediction, selected=True)

    def find_future_option_evidence(self, context_signature: str, action: int) -> dict[str, Any]:
        del context_signature
        action_edges = self.memory.edges_to(action_node_id(action), "takes_action")
        deltas: list[float] = []
        positive = 0
        for edge in action_edges:
            score_row = self.memory.connection.execute(
                "SELECT future_option_delta FROM memory_scores WHERE node_id = ?",
                (str(edge["source_node_id"]),),
            ).fetchone()
            if score_row is None or score_row[0] is None:
                continue
            delta = float(score_row[0])
            deltas.append(delta)
            if delta > 0:
                positive += 1
        del positive
        return aggregate_future_option_evidence(deltas)

    def find_failure_path_evidence(self, context_signature: str, action: int) -> dict[str, Any]:
        del context_signature
        action_edges = self.memory.edges_to(action_node_id(action), "takes_action")
        deltas: list[float] = []
        contradiction = False
        total = 0
        for edge in action_edges:
            total += 1
            score_row = self.memory.connection.execute(
                "SELECT future_option_delta FROM memory_scores WHERE node_id = ?",
                (str(edge["source_node_id"]),),
            ).fetchone()
            if score_row is not None and score_row[0] is not None:
                deltas.append(float(score_row[0]))
            if self.memory.edges_from(str(edge["source_node_id"]), "violates_prediction"):
                contradiction = True
        # Preserve the legacy denominator: all action evidence, including
        # evidence rows without a future-option score.
        return compute_failure_path_evidence(
            deltas=deltas,
            evidence_count=total,
            contradiction=contradiction,
        )

    def _exact_contingency_match(self, context_signatures: dict[int, tuple], action: int) -> dict[str, Any] | None:
        targets = {json.dumps(list(signature)) for signature in context_signatures.values()}
        for node in self.memory.query_nodes(memory_level="M1", node_type="ContingencyMemory"):
            attrs = dict(node.get("attrs", {}))
            if str(attrs.get("context_signature")) not in targets:
                continue
            if int(attrs.get("action", -1)) != int(action):
                continue
            return {
                "node_id": node["node_id"],
                "family": int(attrs.get("transformation_family")),
                "confidence": float(attrs.get("confidence", 0.0) or 0.0),
            }
        return None

    def _record_query_event(self, *, action: int, prediction: MemoryPrediction, selected: bool = False) -> None:
        query_key = json.dumps({"action": int(action), "source": prediction.source, "family": prediction.predicted_family}, sort_keys=True)
        query_node_id = "M5:query:" + sha1(query_key.encode("utf-8")).hexdigest()[:20]
        self.memory.upsert_node(
            MemoryNode(
                node_id=query_node_id,
                memory_level="M5",
                node_type="MemoryQueryEvent",
                canonical_key=query_key,
                attrs={"action": int(action), "source": prediction.source},
            ),
        )
        self.memory.upsert_edge(MemoryEdge(query_node_id, action_node_id(action), "evaluated_action"))
        if selected:
            self.memory.upsert_edge(MemoryEdge(query_node_id, action_node_id(action), "selected_action"))
        if prediction.predicted_family is not None:
            self.memory.upsert_edge(MemoryEdge(query_node_id, family_node_id(prediction.predicted_family), "predicted_family"))
        for evidence_node_id in prediction.evidence_node_ids:
            self.memory.upsert_edge(MemoryEdge(query_node_id, evidence_node_id, "used_evidence"))

    def _best_context_signature(self, context_signatures: dict[int, tuple], action: int) -> str:
        if not context_signatures:
            return json.dumps([int(action)])
        max_level = max(int(level) for level in context_signatures)
        return json.dumps(list(context_signatures.get(max_level, next(iter(context_signatures.values())))))

    def _context_node_id(self, context_signature: str) -> str:
        return "M0:context:" + sha1(str(context_signature).encode("utf-8")).hexdigest()[:20]
