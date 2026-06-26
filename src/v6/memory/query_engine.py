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


class MemoryQueryEngine:
    def __init__(self, memory: MemorySubstrate, contingency_learner: Any = None, graph: Any = None) -> None:
        self.memory = memory
        self.contingency_learner = contingency_learner
        self.graph = graph

    def predict_family(self, context_signatures: dict[int, tuple], action: int) -> MemoryPrediction:
        exact = self._exact_contingency_match(context_signatures, action)
        if exact is not None:
            prediction = MemoryPrediction(
                predicted_family=exact["family"],
                confidence=float(exact["confidence"]),
                source="memory_contingency",
                evidence_node_ids=[str(exact["node_id"])],
            )
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
                self._record_query_event(action=action, prediction=prediction)
                return prediction
        role_matches = self.find_similar_roles(json.dumps(list(context_signatures.get(max(context_signatures), ()))), action)
        if role_matches and role_matches[0].get("family_id") is not None:
            prediction = MemoryPrediction(
                predicted_family=role_matches[0].get("family_id"),
                confidence=float(role_matches[0].get("score", 0.4)),
                source="role_match",
                evidence_node_ids=[str(role_matches[0]["node_id"])],
            )
            self._record_query_event(action=action, prediction=prediction)
            return prediction
        concept_matches = self.find_concept_matches(json.dumps(list(context_signatures.get(max(context_signatures), ()))), action)
        if concept_matches and concept_matches[0].get("family_id") is not None:
            prediction = MemoryPrediction(
                predicted_family=concept_matches[0].get("family_id"),
                confidence=float(concept_matches[0].get("score", 0.35)),
                source="concept_match",
                evidence_node_ids=[str(concept_matches[0]["node_id"])],
            )
            self._record_query_event(action=action, prediction=prediction)
            return prediction
        prediction = MemoryPrediction(predicted_family=None, confidence=0.0, source="none", evidence_node_ids=[])
        self._record_query_event(action=action, prediction=prediction)
        return prediction

    def score_action(self, context_signatures: dict[int, tuple], action: int, available_actions: list[int]) -> MemoryActionScore:
        prediction = self.predict_family(context_signatures, action)
        future = self.find_future_option_evidence(json.dumps(list(context_signatures.get(max(context_signatures), ()))), action)
        failure = self.find_failure_path_evidence(json.dumps(list(context_signatures.get(max(context_signatures), ()))), action)
        role_matches = self.find_similar_roles(json.dumps(list(context_signatures.get(max(context_signatures), ()))), action)
        concept_matches = self.find_concept_matches(json.dumps(list(context_signatures.get(max(context_signatures), ()))), action)
        role_or_concept_transfer_score = max(
            float(role_matches[0].get("score", 0.0)) if role_matches else 0.0,
            float(concept_matches[0].get("score", 0.0)) if concept_matches else 0.0,
        )
        contradiction_risk = 1.0 if failure.get("contradiction_evidence") else 0.0
        expected_future_option_gain = float(future.get("expected_future_option_delta", 0.0) or 0.0)
        failure_risk = float(failure.get("failure_risk", 0.0) or 0.0)
        completion_likelihood = float(future.get("completion_likelihood", 0.0) or 0.0)
        score = (
            0.30 * float(prediction.confidence)
            + 0.25 * max(0.0, expected_future_option_gain)
            + 0.20 * completion_likelihood
            + 0.15 * role_or_concept_transfer_score
            - 0.25 * failure_risk
            - 0.10 * contradiction_risk
        )
        score = max(0.0, min(1.0, float(score)))
        return MemoryActionScore(
            action=int(action),
            score=score,
            predicted_family=prediction.predicted_family,
            expected_future_option_delta=expected_future_option_gain,
            failure_risk=failure_risk,
            completion_likelihood=completion_likelihood,
            evidence_sources=[prediction.source, *future.get("sources", []), *failure.get("sources", [])],
        )

    def rank_actions(self, context_signatures: dict[int, tuple], available_actions: list[int]) -> list[MemoryActionScore]:
        scores = [self.score_action(context_signatures, int(action), available_actions) for action in sorted(int(item) for item in available_actions)]
        return sorted(scores, key=lambda item: (-float(item.score), int(item.action)))

    def find_similar_roles(self, context_signature: str, action: int) -> list[dict]:
        del context_signature, action
        matches: list[dict[str, Any]] = []
        for role in self.memory.query_nodes(memory_level="M3", node_type="FunctionalRoleMemory"):
            family_ids = [
                int(family_edge["target_node_id"].split(":")[-1])
                for edge in self.memory.edges_to(role["node_id"], "plays_role")
                for family_edge in self.memory.edges_from(edge["source_node_id"], "associated_with_family")
                if str(family_edge["target_node_id"]).startswith("M2:family:")
            ]
            matches.append(
                {
                    "node_id": role["node_id"],
                    "score": float(role.get("attrs", {}).get("transfer_score", 0.0) or 0.0),
                    "family_id": family_ids[0] if family_ids else None,
                }
            )
        return sorted(matches, key=lambda item: (-float(item["score"]), str(item["node_id"])))

    def find_concept_matches(self, context_signature: str, action: int) -> list[dict]:
        del context_signature, action
        matches: list[dict[str, Any]] = []
        for concept in self.memory.query_nodes(memory_level="M4", node_type="ConceptMemory"):
            role_edges = self.memory.edges_to(concept["node_id"], "transfers_to")
            family_ids: list[int] = []
            for role_edge in role_edges:
                for carrier_edge in self.memory.edges_from(role_edge["source_node_id"], "abstracts_from"):
                    for family_edge in self.memory.edges_from(carrier_edge["target_node_id"], "associated_with_family"):
                        if str(family_edge["target_node_id"]).startswith("M2:family:"):
                            family_ids.append(int(str(family_edge["target_node_id"]).split(":")[-1]))
            matches.append(
                {
                    "node_id": concept["node_id"],
                    "score": min(1.0, float(concept.get("attrs", {}).get("transfer_success_count", 0) or 0) / 3.0),
                    "family_id": family_ids[0] if family_ids else None,
                }
            )
        return sorted(matches, key=lambda item: (-float(item["score"]), str(item["node_id"])))

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
        mean_delta = (sum(deltas) / len(deltas)) if deltas else 0.0
        return {
            "expected_future_option_delta": mean_delta,
            "completion_likelihood": 0.0 if not deltas else float(positive) / float(len(deltas)),
            "sources": ["future_option_memory"] if deltas else [],
        }

    def find_failure_path_evidence(self, context_signature: str, action: int) -> dict[str, Any]:
        del context_signature
        action_edges = self.memory.edges_to(action_node_id(action), "takes_action")
        negatives = 0
        contradiction = False
        total = 0
        for edge in action_edges:
            total += 1
            score_row = self.memory.connection.execute(
                "SELECT future_option_delta FROM memory_scores WHERE node_id = ?",
                (str(edge["source_node_id"]),),
            ).fetchone()
            if score_row is not None and score_row[0] is not None and float(score_row[0]) < 0.0:
                negatives += 1
            if self.memory.edges_from(str(edge["source_node_id"]), "violates_prediction"):
                contradiction = True
        return {
            "failure_risk": 0.0 if total == 0 else float(negatives) / float(total),
            "contradiction_evidence": contradiction,
            "sources": ["failure_path_memory"] if total else [],
        }

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

    def _record_query_event(self, *, action: int, prediction: MemoryPrediction) -> None:
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
        self.memory.upsert_edge(MemoryEdge(query_node_id, action_node_id(action), "selected_action"))
        if prediction.predicted_family is not None:
            self.memory.upsert_edge(MemoryEdge(query_node_id, family_node_id(prediction.predicted_family), "predicted_family"))
        for evidence_node_id in prediction.evidence_node_ids:
            self.memory.upsert_edge(MemoryEdge(query_node_id, evidence_node_id, "used_evidence"))
