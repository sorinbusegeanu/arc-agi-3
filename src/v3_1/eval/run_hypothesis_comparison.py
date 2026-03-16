from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from v3_1.eval.hypothesis_metrics import compute_hypothesis_metrics


def _mode_payload(label: str, payload: dict[str, Any]) -> dict[str, Any]:
    ledger_records = list(payload.get("session_ledger_records", []) or [])
    llm_events = [
        dict(row.get("payload", {}) or {})
        for row in ledger_records
        if str(row.get("event_type") or "") in {"llm call attempted", "llm call succeeded", "llm call failed", "llm call skipped"}
    ]
    prompt_sizes = [int(row.get("prompt_char_count", 0) or 0) for row in llm_events if int(row.get("prompt_char_count", 0) or 0) > 0]
    schema_valid_count = sum(1 for row in llm_events if row.get("error_code") in {None, ""})
    accepted_count = len(list(payload.get("llm_hypotheses", []) or []))
    useful_count = sum(
        1
        for row in list(payload.get("llm_hypotheses", []) or [])
        if bool(dict(row.get("metadata", {}) or {}).get("accepted", False))
        or str(dict(row.get("metadata", {}) or {}).get("validation_state") or "") == "validated"
    )
    return {
        "mode": label,
        "metrics": compute_hypothesis_metrics(
            deterministic_hypotheses=list(payload.get("deterministic_hypotheses", []) or []),
            llm_hypotheses=list(payload.get("llm_hypotheses", []) or []),
            validation_state=dict(payload.get("validation_state", {}) or {}),
            first_support_round=dict(payload.get("first_support_round", {}) or {}),
            first_contradiction_round=dict(payload.get("first_contradiction_round", {}) or {}),
            first_validation_round=dict(payload.get("first_validation_round", {}) or {}),
            session_ledger_records=ledger_records,
        ),
        "prompt_compression_metrics": {
            "prompt_size_avg": (sum(prompt_sizes) / float(len(prompt_sizes))) if prompt_sizes else 0.0,
            "prompt_size_max": max(prompt_sizes) if prompt_sizes else 0,
            "call_success_rate": float(sum(1 for row in llm_events if row.get("error_code") in {None, ""})) / float(max(1, sum(1 for row in llm_events if row))),
            "schema_valid_response_rate": float(schema_valid_count) / float(max(1, len(llm_events))),
            "accepted_proposal_count": accepted_count,
            "useful_proposal_count": useful_count,
        },
    }


def run_hypothesis_comparison(
    *,
    deterministic_only: dict,
    qwen_hypothesis_generator: dict,
    qwen_hypothesis_generator_broad_payload: dict | None = None,
    qwen_hypothesis_generator_focused_payload: dict | None = None,
    qwen_ambiguity_resolver: dict | None = None,
    qwen_experiment_suggester: dict | None = None,
) -> dict:
    per_run = {
        "deterministic_only": _mode_payload("deterministic_only", deterministic_only),
        "qwen_hypothesis_generator": _mode_payload("qwen_hypothesis_generator", qwen_hypothesis_generator),
        "qwen_hypothesis_generator_broad_payload": _mode_payload("qwen_hypothesis_generator_broad_payload", qwen_hypothesis_generator_broad_payload or {}),
        "qwen_hypothesis_generator_focused_payload": _mode_payload("qwen_hypothesis_generator_focused_payload", qwen_hypothesis_generator_focused_payload or {}),
        "qwen_ambiguity_resolver": _mode_payload("qwen_ambiguity_resolver", qwen_ambiguity_resolver or {}),
        "qwen_experiment_suggester": _mode_payload("qwen_experiment_suggester", qwen_experiment_suggester or {}),
    }
    aggregate_summary = {
        "modes": list(per_run.keys()),
        "llm_success_rate_delta": float(per_run["qwen_hypothesis_generator"]["metrics"].get("llm_call_success_rate", 0.0)) - float(per_run["qwen_ambiguity_resolver"]["metrics"].get("llm_call_success_rate", 0.0)),
        "deterministic_precision": float(per_run["deterministic_only"]["metrics"].get("proposal_precision_by_source", {}).get("deterministic_hypothesis", 0.0)),
        "prompt_compression_comparison": {
            "broad_payload_mode": dict(per_run["qwen_hypothesis_generator_broad_payload"].get("prompt_compression_metrics", {}) or {}),
            "focused_payload_mode": dict(per_run["qwen_hypothesis_generator_focused_payload"].get("prompt_compression_metrics", {}) or {}),
        },
    }
    return {"per_run_reports": per_run, "aggregate_summary": aggregate_summary}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--deterministic-only", required=True)
    parser.add_argument("--qwen-hypothesis-generator", required=True)
    parser.add_argument("--qwen-hypothesis-generator-broad-payload", required=False, default=None)
    parser.add_argument("--qwen-hypothesis-generator-focused-payload", required=False, default=None)
    parser.add_argument("--qwen-ambiguity-resolver", required=False, default=None)
    parser.add_argument("--qwen-experiment-suggester", required=False, default=None)
    args = parser.parse_args()
    deterministic_only = json.loads(Path(args.deterministic_only).read_text(encoding="utf-8"))
    qwen_hypothesis_generator = json.loads(Path(args.qwen_hypothesis_generator).read_text(encoding="utf-8"))
    qwen_hypothesis_generator_broad_payload = json.loads(Path(args.qwen_hypothesis_generator_broad_payload).read_text(encoding="utf-8")) if args.qwen_hypothesis_generator_broad_payload else {}
    qwen_hypothesis_generator_focused_payload = json.loads(Path(args.qwen_hypothesis_generator_focused_payload).read_text(encoding="utf-8")) if args.qwen_hypothesis_generator_focused_payload else {}
    qwen_ambiguity_resolver = json.loads(Path(args.qwen_ambiguity_resolver).read_text(encoding="utf-8")) if args.qwen_ambiguity_resolver else {}
    qwen_experiment_suggester = json.loads(Path(args.qwen_experiment_suggester).read_text(encoding="utf-8")) if args.qwen_experiment_suggester else {}
    print(json.dumps(run_hypothesis_comparison(deterministic_only=deterministic_only, qwen_hypothesis_generator=qwen_hypothesis_generator, qwen_hypothesis_generator_broad_payload=qwen_hypothesis_generator_broad_payload, qwen_hypothesis_generator_focused_payload=qwen_hypothesis_generator_focused_payload, qwen_ambiguity_resolver=qwen_ambiguity_resolver, qwen_experiment_suggester=qwen_experiment_suggester), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
