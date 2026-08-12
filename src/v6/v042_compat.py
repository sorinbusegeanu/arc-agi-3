from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np

_APPLIED = False


def apply_v042_compatibility() -> None:
    global _APPLIED
    if _APPLIED:
        return

    from v6 import v042_policy as policy
    from v6.memory.query_engine import MemoryQueryEngine
    from v6.v042_policy import RobustISFNormalizer
    from v6 import higher_order_substrate as higher_order

    def normalizer_init(self: Any, *, window: int = 256, min_samples: int = 64) -> None:
        self.window = max(16, int(window))
        self.min_samples = max(2, int(min_samples))
        self._values = defaultdict(lambda: deque(maxlen=self.window))

    RobustISFNormalizer.__init__ = normalizer_init

    def categorical_displacement(
        before: np.ndarray,
        after: np.ndarray,
        changed_mask: np.ndarray,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        # Infer the dominant unchanged field value structurally and exclude it
        # from displacement matching. This avoids treating the background field
        # as an object while making no assumption about which categorical value
        # denotes background (it need not be ARC color 0).
        unchanged = (~changed_mask) & (before == after)
        dominant_field: int | None = None
        if np.any(unchanged):
            values, counts = np.unique(before[unchanged], return_counts=True)
            if len(values):
                dominant_field = int(values[int(np.argmax(counts))])

        before_changed = set(int(v) for v in np.unique(before[changed_mask]))
        after_changed = set(int(v) for v in np.unique(after[changed_mask]))
        shared_values = sorted(before_changed & after_changed)
        if dominant_field is not None:
            shared_values = [value for value in shared_values if value != dominant_field]

        weighted_before: list[tuple[float, float, int]] = []
        weighted_after: list[tuple[float, float, int]] = []
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

    policy._categorical_displacement = categorical_displacement

    def query_predict(
        self: Any,
        context_signatures: dict[int, tuple],
        action: int,
        *,
        record_query: bool = False,
    ) -> Any:
        prediction = policy._ORIGINAL_QUERY_PREDICT(
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
            stable = policy._stable_prediction_for_multi(
                getattr(self, "contingency_learner", None),
                context_signatures,
                action,
            )
            if stable is not None:
                raw_support = getattr(stable, "support_count", None)
                support_count = None if raw_support is None else int(raw_support)
                stability = float(getattr(stable, "confidence", getattr(prediction, "confidence", 0.0)) or 0.0)
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
        policy._set_expectation_context(
            supported=supported,
            source=source,
            support_count=support_count,
            stability=stability,
        )
        return prediction

    MemoryQueryEngine.predict_family = query_predict

    def comparable_promotion_state(state_conn: Any, **kwargs: Any) -> tuple[str, int, bool]:
        # The higher-order validator already computes population comparability
        # before setting count_failure=True and validation_result='failed'. Do
        # not override that established context judgment with a second proxy
        # based only on candidate link counts. Other insufficient/unresolved
        # results still go through the v0.4.2 suppression/probation gate.
        explicitly_comparable = (
            not bool(kwargs.get("passed"))
            and bool(kwargs.get("count_failure", True))
            and str(kwargs.get("validation_result") or "") == "failed"
        )
        if not explicitly_comparable:
            return policy._update_promotion_validation_state_v042(state_conn, **kwargs)

        policy._ensure_hysteresis_table(state_conn)
        candidate_type = str(kwargs.get("candidate_type") or "")
        signature = str(kwargs.get("candidate_signature") or "")
        updated_step = kwargs.get("updated_global_step")
        evidence_volume, context_confidence = policy._demotion_evidence(
            state_conn, candidate_type, signature
        )
        previous = state_conn.execute(
            "SELECT lifecycle_status, failure_windows, reactivation_count "
            "FROM promotion_hysteresis_v042 WHERE candidate_type=? AND candidate_signature=?",
            (candidate_type, signature),
        ).fetchone()
        previous_lifecycle = str(previous[0]) if previous is not None else "validated"
        previous_windows = int(previous[1] or 0) if previous is not None else 0
        reactivation_count = int(previous[2] or 0) if previous is not None else 0

        status, failure_count, demoted = policy._ORIGINAL_PROMOTION_STATE(
            state_conn, **kwargs
        )
        lifecycle = "demoted" if demoted else "probation"
        failure_windows = max(previous_windows + 1, int(failure_count))
        state_conn.execute(
            """
            INSERT INTO promotion_hysteresis_v042(
                candidate_type, candidate_signature, lifecycle_status,
                probation_since_step, evidence_volume, context_confidence,
                failure_windows, demotion_suppressed_reason, reactivation_count,
                updated_step
            ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
            ON CONFLICT(candidate_type, candidate_signature) DO UPDATE SET
                lifecycle_status=excluded.lifecycle_status,
                probation_since_step=excluded.probation_since_step,
                evidence_volume=excluded.evidence_volume,
                context_confidence=excluded.context_confidence,
                failure_windows=excluded.failure_windows,
                demotion_suppressed_reason=NULL,
                reactivation_count=excluded.reactivation_count,
                updated_step=excluded.updated_step
            """,
            (
                candidate_type,
                signature,
                lifecycle,
                None if demoted else updated_step,
                int(evidence_volume),
                max(0.5, float(context_confidence)),
                int(failure_windows),
                int(reactivation_count),
                updated_step,
            ),
        )
        return status, failure_count, demoted

    higher_order._update_promotion_validation_state = comparable_promotion_state
    _APPLIED = True
