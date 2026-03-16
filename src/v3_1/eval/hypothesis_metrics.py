from __future__ import annotations

from collections import Counter, defaultdict


def compute_hypothesis_metrics(
    *,
    deterministic_hypotheses: list[dict],
    llm_hypotheses: list[dict],
    validation_state: dict[str, str],
    first_support_round: dict[str, int] | None = None,
    first_contradiction_round: dict[str, int] | None = None,
    first_validation_round: dict[str, int] | None = None,
    session_ledger_records: list[dict] | None = None,
) -> dict:
    first_support_round = dict(first_support_round or {})
    first_contradiction_round = dict(first_contradiction_round or {})
    first_validation_round = dict(first_validation_round or {})
    session_ledger_records = list(session_ledger_records or [])

    def _rows_by_source() -> dict[str, list[dict]]:
        return {
            "deterministic_hypothesis": list(deterministic_hypotheses or []),
            "llm_hypothesis": list(llm_hypotheses or []),
        }

    by_source = _rows_by_source()

    def _validated(rows: list[dict]) -> list[dict]:
        return [row for row in rows if str(validation_state.get(str(row.get("proposal_id")), "")) == "validated"]

    def _contradicted(rows: list[dict]) -> list[dict]:
        return [row for row in rows if str(validation_state.get(str(row.get("proposal_id")), "")) == "contradicted"]

    def _latencies(source_rows: list[dict], target_rounds: dict[str, int]) -> list[int]:
        values = []
        for row in source_rows:
            proposal_id = str(row.get("proposal_id") or "")
            if not proposal_id or proposal_id not in target_rounds:
                continue
            values.append(max(0, int(target_rounds[proposal_id]) - int(row.get("round_id", 0) or 0)))
        return values

    llm_event_counts = Counter(str(row.get("event_type") or "") for row in session_ledger_records)
    llm_skip_reasons = Counter(
        str(dict(row.get("payload", {}) or {}).get("gating_reason") or "unknown")
        for row in session_ledger_records
        if str(row.get("event_type") or "") == "llm call skipped"
    )
    support_counts = defaultdict(int)
    waste_counts = defaultdict(int)
    for rows in by_source.values():
        for row in rows:
            source = str(row.get("provenance") or "unknown")
            metadata = dict(row.get("metadata", {}) or {})
            if list(metadata.get("experiment_supports_hypothesis_ids", []) or []):
                support_counts[source] += 1
            if str(row.get("proposal_kind") or "") == "test" and not list(metadata.get("experiment_supports_hypothesis_ids", []) or []) and not list(metadata.get("experiment_contradicts_hypothesis_ids", []) or []):
                waste_counts[source] += 1

    metrics = {
        "proposal_precision_by_source": {
            source: float(len(_validated(rows))) / float(max(1, len(rows)))
            for source, rows in by_source.items()
        },
        "proposal_contradiction_rate_by_source": {
            source: float(len(_contradicted(rows))) / float(max(1, len(rows)))
            for source, rows in by_source.items()
        },
        "validated_edge_recall_by_source": {
            source: len(_validated(rows))
            for source, rows in by_source.items()
        },
        "first_correct_prerequisite_round_by_source": {
            source: min([int(first_validation_round.get(str(row.get("proposal_id")), 10**9)) for row in _validated(rows)] or [None])
            for source, rows in by_source.items()
        },
        "first_win_after_proposal_by_source": {
            source: min([int(first_support_round.get(str(row.get("proposal_id")), 10**9)) for row in rows if str(row.get("proposal_kind") or "") in {"path", "edge"}] or [None])
            for source, rows in by_source.items()
        },
        "unnecessary_test_count_by_source": {
            source: sum(1 for row in rows if str(row.get("proposal_kind") or "") == "test" and str(validation_state.get(str(row.get("proposal_id")), "")) != "validated")
            for source, rows in by_source.items()
        },
        "source_agreement_rate": 0.0,
        "source_disagreement_resolved_by_later_evidence": sum(
            1 for proposal_id, state in validation_state.items() if str(state) == "validated" and proposal_id not in {str(row.get("proposal_id")) for row in deterministic_hypotheses if any(str(other.get("proposal_id")) == str(row.get("proposal_id")) for other in llm_hypotheses)}
        ),
        "proposal_to_validation_latency_by_source": {
            source: _latencies(rows, first_validation_round)
            for source, rows in by_source.items()
        },
        "proposal_to_contradiction_latency_by_source": {
            source: _latencies(rows, first_contradiction_round)
            for source, rows in by_source.items()
        },
        "experiment_support_rate_by_source": {
            source: float(support_counts.get(source, 0)) / float(max(1, sum(1 for row in rows if str(row.get("proposal_kind") or "") == "test")))
            for source, rows in by_source.items()
        },
        "experiment_waste_rate_by_source": {
            source: float(waste_counts.get(source, 0)) / float(max(1, sum(1 for row in rows if str(row.get("proposal_kind") or "") == "test")))
            for source, rows in by_source.items()
        },
        "winning_path_validation_rate_by_source": {
            source: float(sum(1 for row in rows if str(row.get("proposal_kind") or "") == "path" and str(validation_state.get(str(row.get("proposal_id")), "")) == "validated")) / float(max(1, sum(1 for row in rows if str(row.get("proposal_kind") or "") == "path")))
            for source, rows in by_source.items()
        },
        "llm_call_success_rate": float(llm_event_counts.get("llm call succeeded", 0)) / float(max(1, llm_event_counts.get("llm call attempted", 0))),
        "llm_call_skip_rate_by_gating_reason": {
            reason: float(count) / float(max(1, llm_event_counts.get("llm call skipped", 0)))
            for reason, count in llm_skip_reasons.items()
        },
    }
    deterministic_ids = {str(row.get("proposal_id")) for row in deterministic_hypotheses}
    llm_ids = {str(row.get("proposal_id")) for row in llm_hypotheses}
    metrics["source_agreement_rate"] = float(len(deterministic_ids & llm_ids)) / float(max(1, len(deterministic_ids | llm_ids)))
    return metrics
