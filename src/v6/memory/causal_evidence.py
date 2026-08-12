from __future__ import annotations

import json
import math
import time
from hashlib import sha1
from typing import Any

_INSTALLED = False
_ORIGINAL_INSTALL_V63: Any = None


def _clamp01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return max(0.0, min(1.0, number))


def _previous_stage(memory: Any) -> str:
    try:
        row = memory.connection.execute(
            "SELECT value_json FROM memory_development_state WHERE key='current'"
        ).fetchone()
    except Exception:
        row = None
    if row is not None and row[0]:
        try:
            payload = json.loads(str(row[0]))
            stage = str(payload.get("next_stage") or payload.get("stage") or "")
            if stage:
                return stage
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    return "survival"


def _weights_for_stage(stage: str) -> dict[str, float]:
    weights = {
        "survival": (0.65, 0.0, 0.35, 0.0, 0.0),
        "movement_freedom": (0.25, 0.30, 0.30, 0.075, 0.075),
        "environmental_influence": (0.15, 0.20, 0.30, 0.20, 0.15),
        "graph_expansion": (0.10, 0.20, 0.25, 0.25, 0.20),
        "role_discovery": (0.10, 0.15, 0.20, 0.30, 0.25),
        "concept_transfer": (0.10, 0.10, 0.15, 0.30, 0.35),
    }.get(str(stage), (0.65, 0.0, 0.35, 0.0, 0.0))
    keys = (
        "survival_impact",
        "prediction_error",
        "learning_value",
        "transfer_potential",
        "explanatory_potential",
    )
    return dict(zip(keys, weights))


def _causal_current_isf_weights(self: Any) -> dict[str, float]:
    return _weights_for_stage(_previous_stage(self.memory))


def _available_step(node: dict[str, Any], attrs: dict[str, Any], score_step: int | None) -> int | None:
    candidates = (
        attrs.get("evidence_available_step"),
        attrs.get("realized_step"),
        node.get("last_seen_step"),
        node.get("first_seen_step"),
    )
    for value in candidates:
        if value is None:
            continue
        try:
            resolved = int(value)
        except (TypeError, ValueError):
            continue
        if score_step is None or resolved <= score_step:
            return resolved
    return score_step


def _causal_value(
    attrs: dict[str, Any],
    *,
    prospective_key: str,
    realized_key: str,
    score_step: int | None,
) -> tuple[Any, str]:
    realized = attrs.get(realized_key)
    realized_step = attrs.get(f"{realized_key}_step", attrs.get("realized_step"))
    if realized is not None:
        if score_step is None or realized_step is None:
            return realized, "realized"
        try:
            if int(realized_step) <= int(score_step):
                return realized, "realized"
        except (TypeError, ValueError):
            pass
    prospective = attrs.get(prospective_key)
    return prospective, "prospective" if prospective is not None else "none"


def _ensure_causal_schema(connection: Any) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS memory_score_causal_snapshots_v64 (
            node_id TEXT NOT NULL,
            score_step INTEGER,
            evidence_available_step INTEGER,
            evidence_kind TEXT NOT NULL,
            developmental_stage_snapshot TEXT NOT NULL,
            next_developmental_stage TEXT NOT NULL,
            score REAL NOT NULL,
            score_policy_version TEXT NOT NULL,
            evidence_json TEXT NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY(node_id, score_step, score_policy_version)
        );
        CREATE INDEX IF NOT EXISTS idx_memory_score_causal_v64_step
        ON memory_score_causal_snapshots_v64(score_step, developmental_stage_snapshot);
        """
    )


def _record_snapshot(
    connection: Any,
    *,
    node_id: str,
    score_step: int | None,
    evidence_available_step: int | None,
    evidence_kind: str,
    stage: str,
    next_stage: str,
    score: float,
    score_policy_version: str,
    evidence: dict[str, Any],
) -> None:
    connection.execute(
        """
        INSERT OR REPLACE INTO memory_score_causal_snapshots_v64(
            node_id, score_step, evidence_available_step, evidence_kind,
            developmental_stage_snapshot, next_developmental_stage,
            score, score_policy_version, evidence_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(node_id), score_step, evidence_available_step, str(evidence_kind),
            str(stage), str(next_stage), float(score), str(score_policy_version),
            json.dumps(evidence, sort_keys=True, default=str), time.time(),
        ),
    )


def _causal_rescore_all_v63(self: Any, *, step: int | None = None) -> dict[str, Any]:
    from v6.memory import v63_policy as policy
    from v6.memory.substrate import MemoryScore

    policy.migrate_v63(self.memory.connection)
    _ensure_causal_schema(self.memory.connection)
    score_step = None if step is None else int(step)
    stage = _previous_stage(self.memory)
    next_stage = str(self.development_stage())
    scored = 0

    for level in self.LEVELS[1:]:
        for node in self.memory.query_nodes(memory_level=level):
            attrs = dict(node.get("attrs") or {})
            if str(attrs.get("promotion_status", "")) == "rejected":
                continue
            source_scores = self._source_scores(str(node["node_id"]))
            source_isf = sum(source_scores) / len(source_scores) if source_scores else None
            explanatory_raw, explanatory_kind = _causal_value(
                attrs, prospective_key="explanatory_potential",
                realized_key="explanatory_reach", score_step=score_step,
            )
            explanatory = self._normalize_reach(explanatory_raw) if explanatory_raw is not None else None
            transfer_prior, transfer_empirical, _effective, transfer_status = policy.resolve_transfer_evidence(attrs)
            empirical_step = attrs.get("transfer_empirical_step", attrs.get("realized_step"))
            if transfer_empirical is not None and score_step is not None and empirical_step is not None:
                try:
                    if int(empirical_step) > score_step:
                        transfer_empirical = None
                        transfer_status = "prior_only" if transfer_prior is not None else "untested"
                except (TypeError, ValueError):
                    transfer_empirical = None
            transfer_effective = transfer_empirical if transfer_empirical is not None else transfer_prior
            support = attrs.get("support_count", attrs.get("carrier_count", attrs.get("transfer_tests")))
            recurrence = policy.bounded_support(support)
            efficiency = None
            if level == "M6" and bool(attrs.get("outcome_signature") or attrs.get("effects") or attrs.get("comparable_outcome_group_id")):
                raw_efficiency = attrs.get("normalized_solve_efficiency", attrs.get("efficiency_score"))
                if raw_efficiency is not None:
                    efficiency = policy.clamp01(raw_efficiency)
            fitness, components = policy.unified_memory_fitness(
                isf_score=source_isf, explanatory_reach=explanatory,
                transfer_prior=transfer_prior, transfer_empirical=transfer_empirical,
                recurrence_score=recurrence, efficiency_score=efficiency,
            )
            prediction = policy.clamp01(attrs.get("prediction_lift")) if attrs.get("prediction_lift") is not None else None
            compression = policy.clamp01(attrs.get("compression_gain")) if attrs.get("compression_gain") is not None else None
            future = None
            if attrs.get("future_option_delta") is not None:
                future = policy.clamp01(abs(float(attrs.get("future_option_delta") or 0.0)))
            elif attrs.get("future_option_effect") in {"positive", "negative"}:
                future = 0.5
            self.memory.upsert_score(MemoryScore(
                node_id=str(node["node_id"]), isf_total=fitness,
                prediction_lift=prediction, transfer_score=transfer_effective,
                explanatory_reach=explanatory, compression_gain=compression,
                future_option_delta=future, replay_priority=fitness,
            ), step=score_step)
            learning_value, learning_kind = _causal_value(
                attrs, prospective_key="learning_value",
                realized_key="learning_value_realized", score_step=score_step,
            )
            evidence_kind = "realized" if "realized" in {learning_kind, explanatory_kind, transfer_status} else "prospective"
            available_step = _available_step(node, attrs, score_step)
            evidence_payload = {
                "source_score_count": len(source_scores), "learning_value": learning_value,
                "explanatory_value": explanatory_raw, "transfer_prior": transfer_prior,
                "transfer_empirical_rate": transfer_empirical,
                "transfer_evidence_status": transfer_status,
                "score_components": components, "stage_weights": _weights_for_stage(stage),
            }
            self.memory.connection.execute(
                """
                UPDATE memory_scores
                SET hierarchical_score=?, developmental_stage=?, source_score_count=?,
                    score_version=?, transfer_prior=?, transfer_empirical_rate=?,
                    transfer_evidence_status=?, memory_fitness=?, recurrence_score=?,
                    efficiency_score=?, score_components_json=?, prospective_learning_value=?,
                    realized_learning_value=?, prospective_explanatory_potential=?,
                    realized_explanatory_reach=?, score_policy_version=?
                WHERE node_id=?
                """,
                (fitness, stage, len(source_scores), "v64_causal_evidence_v1",
                 transfer_prior, transfer_empirical, transfer_status, fitness,
                 recurrence, efficiency, json.dumps(components, sort_keys=True),
                 attrs.get("learning_value"), attrs.get("learning_value_realized"),
                 attrs.get("explanatory_potential"), attrs.get("explanatory_reach"),
                 "v64_causal_evidence_v1", str(node["node_id"])),
            )
            attrs.update({
                "hierarchical_isf": fitness, "memory_fitness": fitness,
                "developmental_stage": stage, "next_developmental_stage": next_stage,
                "hierarchical_score_version": "v64_causal_evidence_v1",
                "source_score_count": len(source_scores), "transfer_prior": transfer_prior,
                "transfer_empirical_rate": transfer_empirical,
                "transfer_evidence_status": transfer_status, "score_components": components,
                "score_step": score_step, "evidence_available_step": available_step,
                "evidence_kind": evidence_kind,
            })
            self.memory.update_node_support_and_attrs(str(node["node_id"]), attrs, support_increment=0, step=score_step)
            _record_snapshot(
                self.memory.connection, node_id=str(node["node_id"]), score_step=score_step,
                evidence_available_step=available_step, evidence_kind=evidence_kind,
                stage=stage, next_stage=next_stage, score=fitness,
                score_policy_version="v64_causal_evidence_v1", evidence=evidence_payload,
            )
            scored += 1

    self.memory.connection.execute(
        """
        INSERT INTO memory_development_state(key, value_json, updated_step, updated_at)
        VALUES ('current', ?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json,
            updated_step=excluded.updated_step, updated_at=excluded.updated_at
        """,
        (json.dumps({"stage": stage, "weights": _weights_for_stage(stage),
                     "next_stage": next_stage, "score_version": "v64_causal_evidence_v1"}, sort_keys=True),
         score_step, time.time()),
    )
    self.memory.connection.commit()
    return {"scored": scored, "developmental_stage": stage,
            "next_developmental_stage": next_stage,
            "score_policy_version": "v64_causal_evidence_v1"}


def _oe0_estimate_option_set(self: Any, env_or_state: Any, *, depth: int = 1, available_actions: Any = None) -> Any:
    from v6.future_options import FutureOptionSet
    if hasattr(env_or_state, "tolist"):
        state_signature = json.dumps(env_or_state.tolist(), separators=(",", ":"))
    else:
        state_signature = json.dumps(env_or_state, sort_keys=True, default=str, separators=(",", ":"))
    actions = tuple(sorted(int(item) for item in (available_actions or ())))
    immediate = tuple(f"oe0:{state_signature}|a{action}" for action in actions)
    option_set_id = "oe0:" + sha1(
        json.dumps({"state": state_signature, "actions": actions}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    return FutureOptionSet(option_set_id=option_set_id, state_signature=state_signature,
                           available_actions=actions, reachable_signatures=immediate,
                           estimated_branching_factor=len(actions), depth=0)


def _apply_runtime_patches() -> None:
    from v6.future_options import FutureOptionEstimator
    from v6.memory.v62_runtime import HierarchicalSignificanceEngine
    FutureOptionEstimator.estimate_option_set = _oe0_estimate_option_set
    HierarchicalSignificanceEngine.current_isf_weights = _causal_current_isf_weights
    HierarchicalSignificanceEngine.rescore_all = _causal_rescore_all_v63


def install_causal_evidence_policy() -> None:
    global _INSTALLED, _ORIGINAL_INSTALL_V63
    if _INSTALLED:
        return
    from v6.memory import v63_policy
    _ORIGINAL_INSTALL_V63 = v63_policy.install_v63_runtime_policy
    def install_v63_then_causal() -> None:
        _ORIGINAL_INSTALL_V63()
        _apply_runtime_patches()
    v63_policy.install_v63_runtime_policy = install_v63_then_causal
    _apply_runtime_patches()
    _INSTALLED = True
