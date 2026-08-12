from __future__ import annotations

import math
import sqlite3
import threading
from collections import defaultdict, deque
from typing import Any

import numpy as np

_PATCHED = False
_TLS = threading.local()
_ORIGINAL_COMPUTE_ISF: Any = None
_ORIGINAL_PREDICT_MULTI: Any = None
_ORIGINAL_PREDICT_SINGLE: Any = None
_ORIGINAL_QUERY_PREDICT: Any = None
_ORIGINAL_EXTRACT_DELTA: Any = None
_ORIGINAL_PROMOTION_STATE: Any = None
_ORIGINAL_ROLE_TO_CONCEPT: Any = None
_ORIGINAL_MULTI_ROLE_CONCEPTS: Any = None

POLICY_VERSION = "v042_cognitive_contract_v1"
DELTA_OPERATOR_VERSION = "arc_d_o_v042"
ISF_NORMALIZATION_VERSION = "isf_robust_rank_v042"


class RobustISFNormalizer:
    """Bounded rolling percentile normalizer for comparable ISF evidence.

    The normalizer is intentionally conservative during cold start: until a
    component has enough comparable samples, its already bounded raw value is
    used unchanged. Thereafter the component is winsorized to the rolling
    2nd/98th percentiles and converted to an empirical percentile rank.
    """

    COMPONENTS = (
        "survival_impact",
        "prediction_error",
        "learning_value",
        "transfer_potential",
        "explanatory_potential",
    )

    def __init__(self, *, window: int = 256, min_samples: int = 64) -> None:
        self.window = max(16, int(window))
        self.min_samples = max(8, int(min_samples))
        self._values: dict[tuple[str, tuple[tuple[str, float], ...]], deque[float]] = defaultdict(
            lambda: deque(maxlen=self.window)
        )

    @staticmethod
    def _clamp01(value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(number):
            return 0.0
        return max(0.0, min(1.0, number))

    def normalize(
        self,
        values: dict[str, float],
        *,
        weights: dict[str, float],
        active: dict[str, bool],
    ) -> dict[str, float]:
        weight_key = tuple(sorted((str(k), round(float(v), 6)) for k, v in weights.items()))
        output: dict[str, float] = {}
        for component in self.COMPONENTS:
            raw = self._clamp01(values.get(component, 0.0))
            if not bool(active.get(component, False)):
                output[component] = raw
                continue
            key = (component, weight_key)
            history = self._values[key]
            if len(history) < self.min_samples:
                normalized = raw
            else:
                ordered = sorted(history)
                low_index = max(0, int(round(0.02 * (len(ordered) - 1))))
                high_index = min(len(ordered) - 1, int(round(0.98 * (len(ordered) - 1))))
                low = float(ordered[low_index])
                high = float(ordered[high_index])
                clipped = max(low, min(high, raw))
                less = sum(1 for value in ordered if value < clipped)
                equal = sum(1 for value in ordered if value == clipped)
                normalized = (less + 0.5 * equal) / max(1, len(ordered))
            history.append(raw)
            output[component] = self._clamp01(normalized)
        return output


_NORMALIZER = RobustISFNormalizer()


def _set_expectation_context(*, supported: bool | None, source: str, support_count: int | None = None, stability: float | None = None) -> None:
    _TLS.expectation_supported = supported
    _TLS.expectation_source = str(source)
    _TLS.expectation_support_count = support_count
    _TLS.expectation_stability = stability
    _TLS.runtime_scoring = True


def _stable_prediction_for_multi(learner: Any, context_signatures: dict[int, tuple], action: int) -> Any:
    if learner is None or not hasattr(learner, "best_stable_for_action"):
        return None
    try:
        return learner.best_stable_for_action(context_signatures, action)
    except Exception:
        return None


def _predict_multi_scale_v042(self: Any, context_signatures: dict[int, tuple], action: int) -> Any:
    stable = _stable_prediction_for_multi(self.learner, context_signatures, action)
    result = _ORIGINAL_PREDICT_MULTI(self, context_signatures, action)
    supported = stable is not None and result is not None and int(result) == int(stable.transformation_family)
    _set_expectation_context(
        supported=supported,
        source="stable_contingency" if supported else "speculative_distribution",
        support_count=(None if stable is None else int(stable.support_count)),
        stability=(None if stable is None else float(stable.confidence)),
    )
    return result


def _predict_single_v042(self: Any, context_signature: tuple, action: int) -> Any:
    stable = _stable_prediction_for_multi(self.learner, {0: tuple(context_signature)}, action)
    result = _ORIGINAL_PREDICT_SINGLE(self, context_signature, action)
    supported = stable is not None and result is not None and int(result) == int(stable.transformation_family)
    _set_expectation_context(
        supported=supported,
        source="stable_contingency" if supported else "speculative_distribution",
        support_count=(None if stable is None else int(stable.support_count)),
        stability=(None if stable is None else float(stable.confidence)),
    )
    return result


def _query_predict_v042(self: Any, context_signatures: dict[int, tuple], action: int, *, record_query: bool = False) -> Any:
    prediction = _ORIGINAL_QUERY_PREDICT(
        self,
        context_signatures,
        action,
        record_query=record_query,
    )
    source = str(getattr(prediction, "source", "none") or "none")
    supported = source in {"memory_contingency", "contingency_learner"}
    support_count: int | None = None
    stability: float | None = None
    if supported:
        stable = _stable_prediction_for_multi(getattr(self, "contingency_learner", None), context_signatures, action)
        if stable is not None:
            support_count = int(stable.support_count)
            stability = float(stable.confidence)
        else:
            stability = float(getattr(prediction, "confidence", 0.0) or 0.0)
    object.__setattr__(prediction, "expectation_supported", supported)
    object.__setattr__(prediction, "expectation_source", source)
    object.__setattr__(prediction, "expectation_support_count", support_count)
    object.__setattr__(prediction, "expectation_stability", stability)
    object.__setattr__(
        prediction,
        "expectation_evidence_status",
        "supported" if supported else "speculative",
    )
    _set_expectation_context(
        supported=supported,
        source=source,
        support_count=support_count,
        stability=stability,
    )
    return prediction


def _weighted_total(values: dict[str, float], weights: dict[str, float], active: dict[str, bool]) -> float:
    active_weight = sum(
        float(weights.get(key, 0.0))
        for key in values
        if bool(active.get(key, False)) and float(weights.get(key, 0.0)) > 0.0
    )
    if active_weight <= 0.0:
        return 0.0
    total = sum(
        max(0.0, min(1.0, float(value))) * float(weights.get(key, 0.0))
        for key, value in values.items()
        if bool(active.get(key, False)) and float(weights.get(key, 0.0)) > 0.0
    ) / active_weight
    return max(0.0, min(1.0, float(total)))


def _compute_isf_v042(**kwargs: Any) -> Any:
    supported = getattr(_TLS, "expectation_supported", None)
    runtime_scoring = bool(getattr(_TLS, "runtime_scoring", False))
    call_kwargs = dict(kwargs)
    if supported is False:
        # Speculative prediction remains available to action ranking but is not
        # causally eligible to generate prediction-error evidence.
        call_kwargs["prediction_correct"] = None
        call_kwargs["prediction_confidence"] = None
    score = _ORIGINAL_COMPUTE_ISF(**call_kwargs)
    raw_components = {
        "survival_impact": float(score.survival_impact),
        "prediction_error": float(score.prediction_error),
        "learning_value": float(score.learning_value),
        "transfer_potential": float(score.transfer_potential),
        "explanatory_potential": float(score.explanatory_potential),
    }
    active = dict(score.component_active or {key: True for key in raw_components})
    normalized_components = dict(raw_components)
    if runtime_scoring:
        normalized_components = _NORMALIZER.normalize(
            raw_components,
            weights=dict(score.weights),
            active=active,
        )
        object.__setattr__(
            score,
            "total",
            _weighted_total(normalized_components, dict(score.weights), active),
        )
    object.__setattr__(score, "raw_components", raw_components)
    object.__setattr__(score, "normalized_components", normalized_components)
    object.__setattr__(score, "normalization_version", ISF_NORMALIZATION_VERSION)
    object.__setattr__(score, "expectation_supported", supported)
    object.__setattr__(score, "expectation_source", getattr(_TLS, "expectation_source", "direct"))
    object.__setattr__(score, "expectation_support_count", getattr(_TLS, "expectation_support_count", None))
    object.__setattr__(score, "expectation_stability", getattr(_TLS, "expectation_stability", None))
    object.__setattr__(
        score,
        "expectation_evidence_status",
        "supported" if supported is True else "speculative" if supported is False else "direct",
    )
    _TLS.expectation_supported = None
    _TLS.runtime_scoring = False
    return score


def _score_to_dict_v042(self: Any) -> dict[str, Any]:
    from dataclasses import asdict

    payload = asdict(self)
    for name in (
        "raw_components",
        "normalized_components",
        "normalization_version",
        "expectation_supported",
        "expectation_source",
        "expectation_support_count",
        "expectation_stability",
        "expectation_evidence_status",
    ):
        if hasattr(self, name):
            payload[name] = getattr(self, name)
    payload["policy_version"] = POLICY_VERSION
    return payload


def _categorical_displacement(before: np.ndarray, after: np.ndarray, changed_mask: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    weighted_before: list[tuple[float, float, int]] = []
    weighted_after: list[tuple[float, float, int]] = []
    shared_values = sorted(set(int(v) for v in np.unique(before[changed_mask])) & set(int(v) for v in np.unique(after[changed_mask])))
    for value in shared_values:
        before_positions = np.argwhere(changed_mask & (before == value))
        after_positions = np.argwhere(changed_mask & (after == value))
        count = min(len(before_positions), len(after_positions))
        if count <= 0:
            continue
        before_centroid = before_positions.mean(axis=0)
        after_centroid = after_positions.mean(axis=0)
        weighted_before.append((float(before_centroid[1]), float(before_centroid[0]), count))
        weighted_after.append((float(after_centroid[1]), float(after_centroid[0]), count))
    total = sum(item[2] for item in weighted_before)
    if total <= 0:
        positions = np.argwhere(changed_mask)
        if positions.size == 0:
            return (0.0, 0.0), (0.0, 0.0)
        y = float(np.mean(positions[:, 0]))
        x = float(np.mean(positions[:, 1]))
        return (x, y), (x, y)
    before_centroid = (
        sum(x * weight for x, _y, weight in weighted_before) / total,
        sum(y * weight for _x, y, weight in weighted_before) / total,
    )
    after_centroid = (
        sum(x * weight for x, _y, weight in weighted_after) / total,
        sum(y * weight for _x, y, weight in weighted_after) / total,
    )
    return before_centroid, after_centroid


def _component_count(positions: list[tuple[int, int]]) -> int:
    remaining = set(positions)
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            y, x = stack.pop()
            for neighbor in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return count


def _extract_delta_v042(before_grid: np.ndarray, after_grid: np.ndarray, delta_id: int = 0) -> Any:
    from v6.delta.delta_extractor import Delta

    before = np.asarray(before_grid, dtype=int)
    after = np.asarray(after_grid, dtype=int)
    if before.shape != after.shape:
        raise ValueError(f"before/after shape mismatch: {before.shape} != {after.shape}")
    if before.ndim != 2:
        raise ValueError(f"expected 2D ARC grids, got shape {before.shape}")
    changed_mask = before != after
    positions_array = np.argwhere(changed_mask)
    changed_positions = [(int(y), int(x)) for y, x in positions_array]
    before_values = set(int(value) for value in np.unique(before))
    after_values = set(int(value) for value in np.unique(after))
    before_centroid, after_centroid = _categorical_displacement(before, after, changed_mask)
    delta = Delta(
        id=int(delta_id),
        changed_cells=len(changed_positions),
        changed_positions=changed_positions,
        colors_added=sorted(after_values - before_values),
        colors_removed=sorted(before_values - after_values),
        centroid_before_x=float(before_centroid[0]),
        centroid_before_y=float(before_centroid[1]),
        centroid_after_x=float(after_centroid[0]),
        centroid_after_y=float(after_centroid[1]),
        dx=float(after_centroid[0] - before_centroid[0]),
        dy=float(after_centroid[1] - before_centroid[1]),
    )
    change_tuples = tuple(
        (int(y), int(x), int(before[y, x]), int(after[y, x]))
        for y, x in changed_positions
    )
    if changed_positions:
        ys = [item[0] for item in changed_positions]
        xs = [item[1] for item in changed_positions]
        bbox = (min(ys), min(xs), max(ys), max(xs))
    else:
        bbox = None
    object.__setattr__(delta, "change_tuples", change_tuples)
    object.__setattr__(delta, "changed_bbox", bbox)
    object.__setattr__(delta, "connected_component_count", _component_count(changed_positions))
    object.__setattr__(delta, "delta_operator_version", DELTA_OPERATOR_VERSION)
    return delta


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
    except sqlite3.Error:
        return set()
    if exists is None:
        return set()
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _demotion_evidence(conn: sqlite3.Connection, candidate_type: str, signature: str) -> tuple[int, float]:
    table = {
        "concept": "concept_candidates",
        "role": "role_candidates",
        "world_model": "world_model_components",
    }.get(candidate_type)
    if not table:
        return 0, 0.0
    columns = _table_columns(conn, table)
    if not columns:
        return 0, 0.0
    key = {
        "concept": "concept_signature",
        "role": "role_signature",
        "world_model": "component_signature",
    }[candidate_type]
    evidence_candidates = [
        name for name in (
            "validation_evidence_count",
            "observed_outcome_count",
            "support_count",
            "transfer_test_count",
            "transfer_tests",
        ) if name in columns
    ]
    scope_candidates = [
        name for name in ("cross_context_count", "cross_game_count", "linked_context_count", "linked_game_count")
        if name in columns
    ]
    select = [key, *evidence_candidates, *scope_candidates]
    try:
        row = conn.execute(
            f"SELECT {', '.join(select)} FROM {table} WHERE {key}=?",
            (signature,),
        ).fetchone()
    except sqlite3.Error:
        return 0, 0.0
    if row is None:
        return 0, 0.0
    mapping = dict(zip(select, row))
    evidence_volume = max([int(mapping.get(name) or 0) for name in evidence_candidates] or [0])
    scope = max([int(mapping.get(name) or 0) for name in scope_candidates] or [0])
    context_confidence = min(1.0, scope / 2.0) if scope_candidates else (1.0 if evidence_volume > 0 else 0.0)
    return evidence_volume, context_confidence


def _ensure_hysteresis_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS promotion_hysteresis_v042 (
            candidate_type TEXT NOT NULL,
            candidate_signature TEXT NOT NULL,
            lifecycle_status TEXT NOT NULL,
            probation_since_step INTEGER,
            evidence_volume INTEGER NOT NULL DEFAULT 0,
            context_confidence REAL NOT NULL DEFAULT 0.0,
            failure_windows INTEGER NOT NULL DEFAULT 0,
            demotion_suppressed_reason TEXT,
            reactivation_count INTEGER NOT NULL DEFAULT 0,
            updated_step INTEGER,
            PRIMARY KEY(candidate_type, candidate_signature)
        )
        """
    )


def _update_promotion_validation_state_v042(state_conn: sqlite3.Connection, **kwargs: Any) -> tuple[str, int, bool]:
    _ensure_hysteresis_table(state_conn)
    candidate_type = str(kwargs.get("candidate_type") or "")
    signature = str(kwargs.get("candidate_signature") or "")
    passed = bool(kwargs.get("passed"))
    previously_promoted = bool(kwargs.get("previously_promoted", False))
    updated_step = kwargs.get("updated_global_step")
    evidence_volume, context_confidence = _demotion_evidence(state_conn, candidate_type, signature)
    previous = state_conn.execute(
        "SELECT lifecycle_status, failure_windows, reactivation_count FROM promotion_hysteresis_v042 "
        "WHERE candidate_type=? AND candidate_signature=?",
        (candidate_type, signature),
    ).fetchone()
    previous_lifecycle = str(previous[0]) if previous is not None else "candidate"
    previous_windows = int(previous[1] or 0) if previous is not None else 0
    reactivation_count = int(previous[2] or 0) if previous is not None else 0

    call_kwargs = dict(kwargs)
    suppression_reason: str | None = None
    comparable_failure = bool(call_kwargs.get("count_failure", True))
    validation_result = str(call_kwargs.get("validation_result") or "")
    if not passed and comparable_failure:
        if evidence_volume < 2:
            suppression_reason = "insufficient_evidence_volume"
        elif context_confidence < 0.5:
            suppression_reason = "unresolved_context"
        elif any(token in validation_result for token in ("insufficient", "invalid", "unavailable", "noncomparable")):
            suppression_reason = "noncomparable_validation_context"
        if suppression_reason is not None:
            call_kwargs["count_failure"] = False
            call_kwargs["retain_previous_promotion"] = bool(previously_promoted or previous_lifecycle in {"validated", "reactivated", "probation"})

    status, failure_count, demoted = _ORIGINAL_PROMOTION_STATE(state_conn, **call_kwargs)
    failure_windows = previous_windows
    probation_since = None
    if passed:
        if previous_lifecycle == "demoted":
            lifecycle = "reactivated"
            reactivation_count += 1
        else:
            lifecycle = "validated"
        failure_windows = 0
    elif suppression_reason is not None:
        lifecycle = "probation" if bool(previously_promoted or previous_lifecycle in {"validated", "reactivated", "probation"}) else "candidate"
        probation_since = updated_step if lifecycle == "probation" else None
    elif demoted:
        lifecycle = "demoted"
        failure_windows = max(previous_windows + 1, int(failure_count))
    elif bool(previously_promoted or previous_lifecycle in {"validated", "reactivated", "probation"}):
        lifecycle = "probation"
        failure_windows = max(previous_windows + 1, int(failure_count))
        probation_since = updated_step
    else:
        lifecycle = "candidate"
        failure_windows = max(previous_windows, int(failure_count))

    state_conn.execute(
        """
        INSERT INTO promotion_hysteresis_v042(
            candidate_type, candidate_signature, lifecycle_status,
            probation_since_step, evidence_volume, context_confidence,
            failure_windows, demotion_suppressed_reason, reactivation_count,
            updated_step
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(candidate_type, candidate_signature) DO UPDATE SET
            lifecycle_status=excluded.lifecycle_status,
            probation_since_step=excluded.probation_since_step,
            evidence_volume=excluded.evidence_volume,
            context_confidence=excluded.context_confidence,
            failure_windows=excluded.failure_windows,
            demotion_suppressed_reason=excluded.demotion_suppressed_reason,
            reactivation_count=excluded.reactivation_count,
            updated_step=excluded.updated_step
        """,
        (
            candidate_type,
            signature,
            lifecycle,
            probation_since,
            int(evidence_volume),
            float(context_confidence),
            int(failure_windows),
            suppression_reason,
            int(reactivation_count),
            updated_step,
        ),
    )
    return status, failure_count, demoted


def _apply_concept_status(memory: Any, *, step: int | None = None) -> None:
    for node in memory.query_nodes(memory_level="M4", node_type="ConceptMemory"):
        attrs = dict(node.get("attrs") or {})
        tests = int(attrs.get("transfer_tests", 0) or 0)
        empirical = attrs.get("transfer_empirical_rate")
        success_count = int(attrs.get("transfer_success_count", 0) or 0)
        validated = tests >= 2 and empirical is not None and float(empirical) >= 0.5 and success_count > 0
        attrs["concept_status"] = "validated_transferable" if validated else "candidate"
        attrs["concept_status_version"] = POLICY_VERSION
        memory.update_node_support_and_attrs(
            str(node["node_id"]),
            attrs,
            support_increment=0,
            step=step,
        )


def _promote_role_to_concept_v042(self: Any, *, step: int | None = None) -> dict[str, Any]:
    result = _ORIGINAL_ROLE_TO_CONCEPT(self, step=step)
    _apply_concept_status(self.memory, step=step)
    return result


def _promote_multi_role_concepts_v042(self: Any, *, step: int | None = None) -> int:
    result = _ORIGINAL_MULTI_ROLE_CONCEPTS(self, step=step)
    _apply_concept_status(self.memory, step=step)
    return int(result)


def install_v042_policy() -> None:
    global _PATCHED
    global _ORIGINAL_COMPUTE_ISF
    global _ORIGINAL_PREDICT_MULTI
    global _ORIGINAL_PREDICT_SINGLE
    global _ORIGINAL_QUERY_PREDICT
    global _ORIGINAL_EXTRACT_DELTA
    global _ORIGINAL_PROMOTION_STATE
    global _ORIGINAL_ROLE_TO_CONCEPT
    global _ORIGINAL_MULTI_ROLE_CONCEPTS
    if _PATCHED:
        return

    from v6 import interaction_significance as isf_module
    from v6 import main as main_module
    from v6 import higher_order_substrate as higher_order
    from v6.delta import delta_extractor as delta_module
    from v6.interaction_significance import InteractionSignificanceScore
    from v6.prediction.predictor import Predictor
    from v6.memory.query_engine import MemoryQueryEngine
    from v6.memory.promotion_engine import MemoryPromotionEngine
    from v6.memory.v621_runtime import V621AbstractionEngine

    _ORIGINAL_COMPUTE_ISF = isf_module.compute_interaction_significance
    _ORIGINAL_PREDICT_MULTI = Predictor.predict_multi_scale
    _ORIGINAL_PREDICT_SINGLE = Predictor.predict
    _ORIGINAL_QUERY_PREDICT = MemoryQueryEngine.predict_family
    _ORIGINAL_EXTRACT_DELTA = delta_module.extract_delta
    _ORIGINAL_PROMOTION_STATE = higher_order._update_promotion_validation_state
    _ORIGINAL_ROLE_TO_CONCEPT = MemoryPromotionEngine.promote_role_to_concept
    _ORIGINAL_MULTI_ROLE_CONCEPTS = V621AbstractionEngine.promote_multi_role_concepts

    Predictor.predict_multi_scale = _predict_multi_scale_v042
    Predictor.predict = _predict_single_v042
    MemoryQueryEngine.predict_family = _query_predict_v042
    isf_module.compute_interaction_significance = _compute_isf_v042
    main_module.compute_interaction_significance = _compute_isf_v042
    InteractionSignificanceScore.to_dict = _score_to_dict_v042
    delta_module.extract_delta = _extract_delta_v042
    higher_order._update_promotion_validation_state = _update_promotion_validation_state_v042
    MemoryPromotionEngine.promote_role_to_concept = _promote_role_to_concept_v042
    V621AbstractionEngine.promote_multi_role_concepts = _promote_multi_role_concepts_v042
    _PATCHED = True
