from __future__ import annotations

"""v8.74 research-run integrity repairs.

This late layer repairs three production-observability/control gaps without changing
learning thresholds or cognitive mechanisms:

* publish M2 formation telemetry from the estimator that actually runs discover();
* close the transitive developmental-order gap for role emergence (H05);
* mirror verified worker WINs into the existing v8.34/v8.35 runtime-WIN channel.
"""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace


_INSTALLED = False
_BASE_RUNTIME_METRICS = None
_BASE_V82_EVALUATE = None
_BASE_RECORD_VERIFIED_SUCCESS = None


def _formation_metrics_v874(self):
    payload = dict(_BASE_RUNTIME_METRICS(self))
    existing = payload.get("formation_telemetry")
    merged = dict(existing) if isinstance(existing, dict) else {}

    peers = getattr(self, "peers", None)
    if peers is not None:
        # M2 discover() is executed by the promotion engine, not by the standalone
        # compression peer's evaluate() path.  Read the estimator that actually ran.
        promotion = getattr(peers, "promotion", None)
        production_m2 = getattr(promotion, "generative_compression", None)
        fallback_m2 = getattr(peers, "compression", None)
        for estimator in (production_m2, fallback_m2):
            telemetry = getattr(estimator, "_v870_formation_telemetry", None)
            if isinstance(telemetry, dict) and telemetry:
                merged.update(telemetry)
                break

        roles = getattr(peers, "roles", None)
        telemetry = getattr(roles, "_v870_formation_telemetry", None)
        if isinstance(telemetry, dict) and telemetry:
            merged.update(telemetry)

    payload["formation_telemetry"] = merged
    return payload


def enforce_role_developmental_order_v874(decisions, ordering):
    """H05 cannot validate when its required family->carrier precursor failed.

    v8.2 already gates H04 on family_before_carrier and H05 on carrier_before_role.
    H05 is one developmental step later, so a failed family->carrier precursor also
    falsifies the claimed family->carrier->role ordering.  This is deliberately not
    a generic cross-hypothesis status gate: independently causal hypotheses such as
    held-out transfer remain evaluable from their own evidence.
    """

    family_gate = str(ordering.get("family_before_carrier", "NOT_REACHED"))
    if family_gate != "FAIL":
        return tuple(decisions)

    result = []
    for row in decisions:
        if row.hypothesis_id != "H05" or row.final_decision != "VALID":
            result.append(row)
            continue
        blocker = str(row.blocker)
        message = "developmental ordering failed: family_before_carrier"
        if message not in blocker:
            blocker = (blocker + "; " if blocker else "") + message
        result.append(
            replace(
                row,
                final_decision="INVALID",
                ordering_gate="FAIL",
                blocker=blocker,
            )
        )
    return tuple(result)


def _evaluate_v874(self, evidence):
    from v8.scientific_traceability import ordering_gates

    rows = tuple(evidence)
    decisions = _BASE_V82_EVALUATE(self, rows)
    return enforce_role_developmental_order_v874(decisions, ordering_gates(rows))


def _record_verified_success_v874(
    *,
    game_id: str,
    seed: int,
    terminal_state: str,
    levels_completed: int,
    actions,
    capture_step: int | None,
    trajectory_id: str | None = None,
    root: str | Path | None = None,
) -> bool:
    persisted = bool(
        _BASE_RECORD_VERIFIED_SUCCESS(
            game_id=str(game_id),
            seed=int(seed),
            terminal_state=str(terminal_state),
            levels_completed=int(levels_completed),
            actions=actions,
            capture_step=capture_step,
            trajectory_id=trajectory_id,
            root=root,
        )
    )
    if persisted and str(terminal_state).upper() == "WIN":
        # v8.34 is the public adaptive game-state authority and v8.35 supplies the
        # current-run session guard.  Feed verified worker success into that existing
        # channel rather than installing a competing game_state implementation.
        from v8 import runtime_win_optimization_v834 as v834

        v834._write_runtime_win_marker(
            str(game_id),
            SimpleNamespace(
                wins=1,
                levels_completed=max(1, int(levels_completed)),
                steps=max(1, int(capture_step or len(tuple(actions)) or 1)),
            ),
        )
    return persisted


def install_run_integrity_v874() -> None:
    global _INSTALLED
    global _BASE_RUNTIME_METRICS, _BASE_V82_EVALUATE
    global _BASE_RECORD_VERIFIED_SUCCESS
    if _INSTALLED:
        return

    from v8 import evaluation_v82
    from v8 import runtime_v82
    from v8 import verified_success_metrics_v866 as verified

    _BASE_RUNTIME_METRICS = runtime_v82.V82ContinuousMemoryRuntime.metrics
    _BASE_V82_EVALUATE = evaluation_v82.V82ScientificHypothesisEvaluator.evaluate
    _BASE_RECORD_VERIFIED_SUCCESS = verified.record_verified_success_v866

    runtime_v82.V82ContinuousMemoryRuntime.metrics = _formation_metrics_v874
    evaluation_v82.V82ScientificHypothesisEvaluator.evaluate = _evaluate_v874
    verified.record_verified_success_v866 = _record_verified_success_v874
    _INSTALLED = True
