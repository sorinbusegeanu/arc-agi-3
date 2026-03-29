from __future__ import annotations

from v3_1.eval.planning_mode_metrics import compute_planning_mode_metrics


def _validate_previous_mode_rows(rows: list[dict]) -> dict:
    violations = []
    for row in list(rows or []):
        round_id = int(dict(row).get("round_id", 0) or 0)
        if round_id <= 1:
            continue
        expected = row.get("chronology_previous_mode", row.get("previous_planning_mode"))
        trace_previous = row.get("trace_previous_mode")
        ledger_previous = row.get("ledger_previous_mode")
        trace_match = str(trace_previous or "none") == str(expected or "none")
        ledger_match = str(ledger_previous or "none") == str(expected or "none")
        if not trace_match or not ledger_match or trace_previous in {"", None} or ledger_previous in {"", None}:
            violations.append(
                {
                    "round_id": round_id,
                    "expected_previous_mode": expected,
                    "trace_previous_mode": trace_previous,
                    "ledger_previous_mode": ledger_previous,
                    "trace_matches": trace_match,
                    "ledger_matches": ledger_match,
                }
            )
    return {
        "passed": not violations,
        "violations": violations,
    }


def run_planning_mode_eval(
    *,
    current_mode_rows: list[dict],
    hysteresis_mode_rows: list[dict],
    games: list[str],
    seeds: list[int],
    round_budget: int,
    graph_settings: dict | None = None,
    hypothesis_settings: dict | None = None,
    probe_escalation_enabled: bool = True,
    exit_readiness_enabled: bool = True,
) -> dict:
    current_validation = _validate_previous_mode_rows(current_mode_rows)
    hysteresis_validation = _validate_previous_mode_rows(hysteresis_mode_rows)
    return {
        "games": list(games),
        "seeds": list(seeds),
        "round_budget": int(round_budget),
        "graph_settings": dict(graph_settings or {}),
        "hypothesis_settings": dict(hypothesis_settings or {}),
        "probe_escalation_enabled": bool(probe_escalation_enabled),
        "exit_readiness_enabled": bool(exit_readiness_enabled),
        "comparison": {
            "current_mode_logic": compute_planning_mode_metrics(current_mode_rows),
            "hysteresis_persistence_mode_logic": compute_planning_mode_metrics(hysteresis_mode_rows),
        },
        "telemetry_validation": {
            "current_mode_logic": current_validation,
            "hysteresis_persistence_mode_logic": hysteresis_validation,
            "passed": bool(current_validation.get("passed", False) and hysteresis_validation.get("passed", False)),
        },
    }
