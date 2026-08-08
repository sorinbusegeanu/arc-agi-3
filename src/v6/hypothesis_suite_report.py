from __future__ import annotations

import inspect
import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from tqdm.auto import tqdm

from v6.evaluation.h12_efficiency_emergence import (
    evaluate_h12_efficiency_emergence,
)
from v6.future_options import derive_future_option_memory
from v6.higher_order_substrate import (
    IncrementalPromotionValidationConfig,
    derive_concept_candidates_only,
    derive_role_candidates_only,
    derive_role_transfer_attempts_only,
    derive_world_model_components_only,
    validate_incremental_promotions_only,
)
from v6.hypothesis_h01_report import (
    evaluate_h01_contingency_emergence,
)
from v6.hypothesis_h02_report import (
    evaluate_h02_prediction_violation_attention,
)
from v6.hypothesis_h03_report import (
    evaluate_h03_transformation_family_formation,
)
from v6.hypothesis_h04_report import (
    evaluate_h04_carrier_emergence,
)
from v6.hypothesis_h05_report import evaluate_h05_role_emergence
from v6.hypothesis_h06_report import evaluate_h06_role_transfer
from v6.hypothesis_h07_report import evaluate_h07_concept_emergence
from v6.hypothesis_h08_report import (
    evaluate_h08_world_model_coherence,
)
from v6.hypothesis_h09_report import (
    evaluate_h09_future_option_motifs,
)
from v6.hypothesis_h10_report import (
    evaluate_h10_future_option_attention,
)
from v6.hypothesis_h11_report import (
    evaluate_h11_future_option_transfer_concepts,
)
from v6.memory.compact_memory import (
    derive_missing_transformation_families_from_stable_contingencies,
    ensure_memory_layout,
)
from v6.memory.migrations.v61 import migrate_memory_dir
from v6.provenance_validation import validate_hypothesis_provenance
from v6.reporting.contracts import CONTRACTS, contract_manifest
from v6.reporting.evidence_snapshot import (
    memory_fingerprint,
    read_only_evidence_snapshot,
)
from v6.reporting.framework import apply_decision_envelope


SUITE_JSON_NAME = "hypothesis_suite_summary.json"
SUITE_TXT_NAME = "hypothesis_suite_summary.txt"
SUITE_MD_NAME = "hypothesis_suite_summary.md"
SUITE_AGGREGATED_TXT_NAME = "hypothesis_suite_aggregated.txt"
SUITE_PHASE_LOG_NAME = "hypothesis_phase_log.jsonl"
INPUT_REPORT_NAME = "interaction_sampling_v05c_report.json"
SUITE_HIGHER_ORDER_MAX_CARRIERS = 25_000
SUITE_HIGHER_ORDER_MAX_ROLES = 10_000


def log_hypothesis_progress(
    output_dir: Path,
    phase: str,
    status: str,
    *,
    epoch_id: str | None = None,
    current: int | None = None,
    total: int | None = None,
    start_time: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = time.time()
    payload: dict[str, Any] = {
        "timestamp": now,
        "epoch_id": epoch_id,
        "phase": str(phase),
        "status": str(status),
        "current": current,
        "total": total,
    }
    if start_time is not None:
        payload["seconds_elapsed"] = max(
            0.0, now - float(start_time)
        )
    if extra:
        payload.update(extra)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    with (target / SUITE_PHASE_LOG_NAME).open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return payload



class _HypothesisPhaseTracker:
    def __init__(
        self,
        *,
        output_dir: Path,
        phase: str,
        epoch_id: str | None,
        total: int | None,
        unit: str,
        enabled: bool,
        leave: bool,
        log_every: int,
        top_bar: Any | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.phase = phase
        self.epoch_id = epoch_id
        self.total = total
        self.unit = unit
        self.enabled = enabled
        self.leave = leave
        self.log_every = max(1, int(log_every))
        self.top_bar = top_bar
        self.current = 0
        self.last_logged = 0
        self.started_at = time.time()
        self._bar = None

    def __enter__(self) -> "_HypothesisPhaseTracker":
        log_hypothesis_progress(
            self.output_dir,
            self.phase,
            "starting",
            epoch_id=self.epoch_id,
            current=0,
            total=self.total,
        )
        if self.top_bar is not None:
            self.top_bar.set_postfix_str(f"current={self.phase}")
        if self.enabled:
            self._bar = tqdm(
                total=self.total,
                desc=self.phase,
                unit=self.unit,
                dynamic_ncols=True,
                leave=self.leave,
            )
        return self

    def update(
        self,
        n: int = 1,
        *,
        current: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if current is not None:
            n = int(current) - int(self.current)
            self.current = int(current)
        else:
            self.current += int(n)
        if self._bar is not None and n:
            self._bar.update(int(n))
        if (
            self.current >= self.log_every + self.last_logged
            or (
                self.total is not None
                and self.current >= int(self.total)
            )
        ):
            self.last_logged = int(self.current)
            log_hypothesis_progress(
                self.output_dir,
                self.phase,
                "progress",
                epoch_id=self.epoch_id,
                current=self.current,
                total=self.total,
                start_time=self.started_at,
                extra=extra,
            )

    def close(
        self,
        *,
        status: str = "done",
        extra: dict[str, Any] | None = None,
    ) -> None:
        if self._bar is not None:
            self._bar.close()
        log_hypothesis_progress(
            self.output_dir,
            self.phase,
            status,
            epoch_id=self.epoch_id,
            current=(
                self.current
                if self.total is None
                else min(self.current, int(self.total))
            ),
            total=self.total,
            start_time=self.started_at,
            extra=extra,
        )
        if self.top_bar is not None:
            self.top_bar.update(1)

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is None:
            self.close()
            return
        self.close(
            status="failed",
            extra={
                "exception_type": (
                    exc_type.__name__
                    if exc_type is not None
                    else type(exc).__name__
                ),
                "exception_message": str(exc),
            },
        )


@contextmanager
def hypothesis_phase(
    output_dir: Path,
    phase: str,
    *,
    epoch_id: str | None,
    total: int | None,
    unit: str,
    enabled: bool,
    leave: bool,
    log_every: int,
    top_bar: Any | None = None,
):
    """Compatibility phase context manager retained for existing callers/tests."""
    tracker = _HypothesisPhaseTracker(
        output_dir=output_dir,
        phase=phase,
        epoch_id=epoch_id,
        total=total,
        unit=unit,
        enabled=enabled,
        leave=leave,
        log_every=log_every,
        top_bar=top_bar,
    )
    try:
        yield tracker.__enter__()
    except BaseException as exc:
        tracker.__exit__(type(exc), exc, exc.__traceback__)
        raise
    else:
        tracker.__exit__(None, None, None)



def _call_supported(
    function: Callable[..., Any],
    **kwargs: Any,
) -> Any:
    """Call a project function with only arguments it currently supports."""
    signature = inspect.signature(function)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return function(**kwargs)
    supported = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return function(**supported)


def _phase(
    output_dir: Path,
    epoch_id: str | None,
    name: str,
    callback: Callable[[], Any],
    timings: dict[str, float],
) -> Any:
    started = time.time()
    log_hypothesis_progress(
        output_dir,
        name,
        "starting",
        epoch_id=epoch_id,
    )
    try:
        result = callback()
    except Exception as exc:
        timings[f"{name}_seconds"] = time.time() - started
        log_hypothesis_progress(
            output_dir,
            name,
            "failed",
            epoch_id=epoch_id,
            start_time=started,
            extra={
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            },
        )
        raise
    timings[f"{name}_seconds"] = time.time() - started
    log_hypothesis_progress(
        output_dir,
        name,
        "done",
        epoch_id=epoch_id,
        current=1,
        total=1,
        start_time=started,
    )
    return result


def prepare_hypothesis_evidence(
    *,
    run_dir: Path,
    memory_dir: Path | None,
    output_dir: Path,
    epoch_id: str | None,
    suite_mode: str,
    higher_order_workers: int,
    higher_order_transfer_chunk_size: int,
    max_role_carriers: int,
    max_roles: int,
    max_role_transfer_attempts: int,
    max_future_option_events: int,
    max_future_option_motifs: int,
    future_option_development_stage: str,
    incremental_promotion_validation: bool,
    promotion_min_incremental_coverage: float,
    promotion_min_cross_context_or_game_evidence: int,
    promotion_min_behavioral_or_predictive_lift: float,
    promotion_min_relevant_heldout_event_count: int,
    promotion_population_comparability_threshold: float,
    promotion_demotion_failure_limit: int,
    allow_memory_repair: bool,
) -> dict[str, Any]:
    """DERIVE phase. This is the only phase allowed to mutate memory."""
    summary: dict[str, Any] = {
        "phase": "DERIVE",
        "suite_mode": str(suite_mode),
        "memory_dir": (
            None if memory_dir is None else str(memory_dir)
        ),
        "steps": {},
        "errors": [],
    }
    if memory_dir is None:
        summary["errors"].append("memory_dir not provided")
        return summary

    memory_dir = Path(memory_dir)
    migrate_result = migrate_memory_dir(memory_dir)
    summary["schema_migration"] = migrate_result
    ensure_memory_layout(memory_dir)
    timings: dict[str, float] = {}

    if allow_memory_repair:
        summary["steps"]["family_repair"] = _phase(
            output_dir,
            epoch_id,
            "DERIVE.family_repair",
            lambda: _call_supported(
                derive_missing_transformation_families_from_stable_contingencies,
                memory_dir=memory_dir,
            ),
            timings,
        )
    else:
        summary["steps"]["family_repair"] = {
            "skipped": True,
            "reason": "allow_memory_repair=false",
        }

    summary["steps"]["role_candidates"] = _phase(
        output_dir,
        epoch_id,
        "DERIVE.role_candidates",
        lambda: _call_supported(
            derive_role_candidates_only,
            memory_dir=memory_dir,
            run_dir=run_dir,
            max_carriers=int(max_role_carriers),
            max_roles=int(max_roles),
        ),
        timings,
    )

    resolved_mode = str(suite_mode or "fast").lower()
    if resolved_mode != "full":
        summary["steps"]["higher_order"] = {
            "skipped": True,
            "reason": "suite_mode is not full",
        }
        summary["timings"] = timings
        return summary

    summary["steps"]["role_transfer_attempts"] = _phase(
        output_dir,
        epoch_id,
        "DERIVE.role_transfer_attempts",
        lambda: _call_supported(
            derive_role_transfer_attempts_only,
            memory_dir=memory_dir,
            run_dir=run_dir,
            max_transfer_attempts=int(max_role_transfer_attempts),
            workers=int(higher_order_workers),
            chunk_size=int(higher_order_transfer_chunk_size),
        ),
        timings,
    )
    summary["steps"]["concept_candidates"] = _phase(
        output_dir,
        epoch_id,
        "DERIVE.concept_candidates",
        lambda: _call_supported(
            derive_concept_candidates_only,
            memory_dir=memory_dir,
            run_dir=run_dir,
        ),
        timings,
    )

    validation_config = IncrementalPromotionValidationConfig(
        enabled=bool(incremental_promotion_validation),
        min_incremental_coverage=float(
            promotion_min_incremental_coverage
        ),
        min_cross_context_or_game_evidence=int(
            promotion_min_cross_context_or_game_evidence
        ),
        min_behavioral_or_predictive_lift=float(
            promotion_min_behavioral_or_predictive_lift
        ),
        min_relevant_heldout_event_count=int(
            promotion_min_relevant_heldout_event_count
        ),
        promotion_population_comparability_threshold=float(
            promotion_population_comparability_threshold
        ),
        demotion_failure_limit=int(
            promotion_demotion_failure_limit
        ),
    )
    if validation_config.enabled:
        summary["steps"]["concept_validation"] = _phase(
            output_dir,
            epoch_id,
            "DERIVE.concept_validation",
            lambda: _call_supported(
                validate_incremental_promotions_only,
                memory_dir=memory_dir,
                config=validation_config,
                validate_roles_and_concepts=True,
                validate_world_models=False,
                diagnostic_epoch_id=epoch_id,
            ),
            timings,
        )

    summary["steps"]["world_models"] = _phase(
        output_dir,
        epoch_id,
        "DERIVE.world_models",
        lambda: _call_supported(
            derive_world_model_components_only,
            memory_dir=memory_dir,
            run_dir=run_dir,
        ),
        timings,
    )
    if validation_config.enabled:
        summary["steps"]["world_model_validation"] = _phase(
            output_dir,
            epoch_id,
            "DERIVE.world_model_validation",
            lambda: _call_supported(
                validate_incremental_promotions_only,
                memory_dir=memory_dir,
                config=validation_config,
                validate_roles_and_concepts=False,
                validate_world_models=True,
                diagnostic_epoch_id=epoch_id,
            ),
            timings,
        )

    summary["steps"]["future_options"] = _phase(
        output_dir,
        epoch_id,
        "DERIVE.future_options",
        lambda: _call_supported(
            derive_future_option_memory,
            memory_dir=memory_dir,
            run_dir=run_dir,
            max_events=int(max_future_option_events),
            max_motifs=int(max_future_option_motifs),
            development_stage=str(
                future_option_development_stage
            ),
        ),
        timings,
    )
    summary["timings"] = timings
    return summary


def _skipped_result(hypothesis_id: str) -> dict[str, Any]:
    return {
        "hypothesis_id": hypothesis_id,
        "decision": "SKIPPED_FAST_MODE",
        "raw_decision": "SKIPPED_FAST_MODE",
        "core_metrics": {},
        "missing_evidence": [
            "Run with --hypothesis-suite-mode full."
        ],
        "evidence_source": "not_evaluated",
    }


def _failed_evaluator_result(
    hypothesis_id: str,
    exc: Exception,
) -> dict[str, Any]:
    message = (
        f"read-only evaluator failed: "
        f"{type(exc).__name__}: {exc}"
    )
    return {
        "hypothesis_id": hypothesis_id,
        "decision": "INSUFFICIENT_EVIDENCE",
        "raw_decision": "INSUFFICIENT_EVIDENCE",
        "core_metrics": {},
        "missing_evidence": [message],
        "evidence_source": "read_only_snapshot",
        "evaluator_error": {
            "type": type(exc).__name__,
            "message": str(exc),
        },
    }


def _evaluate_one(
    hypothesis_id: str,
    evaluator: Callable[..., dict[str, Any]],
    *,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    try:
        result = _call_supported(evaluator, **kwargs)
    except Exception as exc:
        return _failed_evaluator_result(hypothesis_id, exc)
    if not isinstance(result, dict):
        return _failed_evaluator_result(
            hypothesis_id,
            TypeError("evaluator did not return a dict"),
        )
    result.setdefault("hypothesis_id", hypothesis_id)
    result.setdefault("core_metrics", {})
    result.setdefault("missing_evidence", [])
    return result


def evaluate_hypotheses_read_only(
    *,
    run_dir: Path,
    evidence_memory_dir: Path | None,
    output_dir: Path,
    suite_mode: str,
    max_db_files: int,
    max_rows: int,
    scan_all_dbs: bool,
    incremental_promotion_validation: bool,
    promotion_min_incremental_coverage: float,
    promotion_min_cross_context_or_game_evidence: int,
    promotion_min_behavioral_or_predictive_lift: float,
    promotion_min_relevant_heldout_event_count: int,
    promotion_population_comparability_threshold: float,
    promotion_demotion_failure_limit: int,
    h11_provenance_sample_limit: int,
    h11_write_full_provenance_jsonl: bool,
    max_h11_main_report_bytes: int,
) -> dict[str, dict[str, Any]]:
    """REPORT phase. All evaluators receive a read-only evidence snapshot."""
    dirs = {
        f"H{index:02d}": output_dir / f"h{index:02d}"
        for index in range(1, 13)
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    validation_config = IncrementalPromotionValidationConfig(
        enabled=bool(incremental_promotion_validation),
        min_incremental_coverage=float(
            promotion_min_incremental_coverage
        ),
        min_cross_context_or_game_evidence=int(
            promotion_min_cross_context_or_game_evidence
        ),
        min_behavioral_or_predictive_lift=float(
            promotion_min_behavioral_or_predictive_lift
        ),
        min_relevant_heldout_event_count=int(
            promotion_min_relevant_heldout_event_count
        ),
        promotion_population_comparability_threshold=float(
            promotion_population_comparability_threshold
        ),
        demotion_failure_limit=int(
            promotion_demotion_failure_limit
        ),
    )

    common = {
        "run_dir": run_dir,
        "memory_dir": evidence_memory_dir,
    }
    results: dict[str, dict[str, Any]] = {}
    results["H01"] = _evaluate_one(
        "H01",
        evaluate_h01_contingency_emergence,
        kwargs={**common, "output_dir": dirs["H01"]},
    )
    results["H02"] = _evaluate_one(
        "H02",
        evaluate_h02_prediction_violation_attention,
        kwargs={
            **common,
            "output_dir": dirs["H02"],
            "max_rows": int(max_rows),
            "max_db_files": int(max_db_files),
            "scan_all_dbs": bool(scan_all_dbs),
        },
    )
    results["H03"] = _evaluate_one(
        "H03",
        evaluate_h03_transformation_family_formation,
        kwargs={
            **common,
            "output_dir": dirs["H03"],
            "max_rows": int(max_rows),
            "max_db_files": int(max_db_files),
            "scan_all_dbs": bool(scan_all_dbs),
        },
    )
    results["H04"] = _evaluate_one(
        "H04",
        evaluate_h04_carrier_emergence,
        kwargs={**common, "output_dir": dirs["H04"]},
    )
    results["H05"] = _evaluate_one(
        "H05",
        evaluate_h05_role_emergence,
        kwargs={
            **common,
            "output_dir": dirs["H05"],
            "already_derived": True,
        },
    )

    if str(suite_mode or "fast").lower() != "full":
        for hypothesis_id in (
            "H06", "H07", "H08", "H09", "H10", "H11"
        ):
            results[hypothesis_id] = _skipped_result(
                hypothesis_id
            )
    else:
        results["H06"] = _evaluate_one(
            "H06",
            evaluate_h06_role_transfer,
            kwargs={
                **common,
                "output_dir": dirs["H06"],
                "already_derived": True,
            },
        )
        results["H07"] = _evaluate_one(
            "H07",
            evaluate_h07_concept_emergence,
            kwargs={
                **common,
                "output_dir": dirs["H07"],
                "already_derived": True,
                "incremental_promotion_validation":
                    validation_config,
            },
        )
        results["H08"] = _evaluate_one(
            "H08",
            evaluate_h08_world_model_coherence,
            kwargs={
                **common,
                "output_dir": dirs["H08"],
                "already_derived": True,
            },
        )
        results["H09"] = _evaluate_one(
            "H09",
            evaluate_h09_future_option_motifs,
            kwargs={
                **common,
                "output_dir": dirs["H09"],
                "already_derived": True,
            },
        )
        results["H10"] = _evaluate_one(
            "H10",
            evaluate_h10_future_option_attention,
            kwargs={
                **common,
                "output_dir": dirs["H10"],
                "already_derived": True,
            },
        )
        results["H11"] = _evaluate_one(
            "H11",
            evaluate_h11_future_option_transfer_concepts,
            kwargs={
                **common,
                "output_dir": dirs["H11"],
                "already_derived": True,
                "provenance_sample_limit": int(
                    h11_provenance_sample_limit
                ),
                "write_full_provenance_jsonl": bool(
                    h11_write_full_provenance_jsonl
                ),
                "max_main_report_bytes": int(
                    max_h11_main_report_bytes
                ),
            },
        )

    results["H12"] = _evaluate_one(
        "H12",
        evaluate_h12_efficiency_emergence,
        kwargs={**common, "output_dir": dirs["H12"]},
    )
    return results


def _write_normalized_result(
    output_dir: Path,
    hypothesis_id: str,
    result: Mapping[str, Any],
) -> None:
    directory = output_dir / hypothesis_id.lower()
    directory.mkdir(parents=True, exist_ok=True)
    stem = hypothesis_id.lower() + "_report"
    (directory / f"{stem}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        hypothesis_id,
        f"raw_decision: {result.get('raw_decision')}",
        (
            "evidence_contract_gate: "
            f"{result.get('evidence_contract_gate', {}).get('status')}"
        ),
        (
            "quality_gate: "
            f"{result.get('quality_gate', {}).get('status')}"
        ),
        (
            "dependency_gate: "
            f"{result.get('dependency_gate', {}).get('status')}"
        ),
        f"final_decision: {result.get('final_decision')}",
        (
            "missing_evidence: "
            + json.dumps(result.get("missing_evidence") or [])
        ),
        (
            "core_metrics: "
            + json.dumps(
                result.get("core_metrics") or {},
                sort_keys=True,
            )
        ),
    ]
    (directory / f"{stem}.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def build_hypothesis_suite_summary(
    *,
    epoch_id: str | None = None,
    run_dir: Path | None = None,
    memory_dir: Path | None = None,
    hypothesis_results: Mapping[
        str, Mapping[str, Any]
    ] | None = None,
    derivation_summary: Mapping[str, Any] | None = None,
    provenance_validation: Mapping[str, Any] | None = None,
    timings: Mapping[str, float] | None = None,
    **metadata: Any,
) -> dict[str, Any]:
    results = dict(hypothesis_results or {})
    summary: dict[str, Any] = {
        "epoch_id": epoch_id,
        "source_run_dir": (
            None if run_dir is None else str(run_dir)
        ),
        "memory_dir": (
            None if memory_dir is None else str(memory_dir)
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reporting_framework_version": "v6.1",
        "pipeline_phases": ["WRITE", "DERIVE", "REPORT"],
        "evaluators_read_only": True,
        "evidence_contracts": contract_manifest(),
        "derivation_summary": dict(derivation_summary or {}),
        "provenance_validation": dict(
            provenance_validation or {}
        ),
        "timings": dict(timings or {}),
        "hypotheses": results,
        "raw_decisions": {
            key: value.get("raw_decision")
            for key, value in results.items()
        },
        "final_decisions": {
            key: value.get(
                "final_decision", value.get("decision")
            )
            for key, value in results.items()
        },
        **metadata,
    }
    for hypothesis_id, result in results.items():
        summary[f"{hypothesis_id} decision"] = result.get(
            "final_decision", result.get("decision")
        )
        summary[f"{hypothesis_id} raw decision"] = result.get(
            "raw_decision"
        )
        summary[f"{hypothesis_id} core metrics"] = dict(
            result.get("core_metrics") or {}
        )
        summary[f"{hypothesis_id} suite gating"] = {
            "evidence_contract_gate": result.get(
                "evidence_contract_gate"
            ),
            "quality_gate": result.get("quality_gate"),
            "dependency_gate": result.get("dependency_gate"),
        }
    summary["decision_counts"] = {}
    for decision in summary["final_decisions"].values():
        summary["decision_counts"][str(decision)] = (
            int(summary["decision_counts"].get(str(decision), 0))
            + 1
        )
    return summary


def run_hypothesis_suite_report(
    *,
    run_dir: Path,
    memory_dir: Path | None = None,
    output_dir: Path,
    scan_all_dbs: bool,
    max_db_files: int,
    max_rows: int,
    epoch_id: str | None = None,
    global_step_start: int | None = None,
    global_step_end: int | None = None,
    interactions_this_epoch: int | None = None,
    total_interactions_seen: int | None = None,
    memory_size_before_bytes: int | None = None,
    memory_size_after_bytes: int | None = None,
    suite_mode: str = "fast",
    higher_order_workers: int = 1,
    higher_order_transfer_chunk_size: int = 5_000,
    max_role_carriers: int = 25_000,
    max_roles: int = 10_000,
    max_role_transfer_attempts: int = 25_000,
    max_future_option_events: int = 50_000,
    max_future_option_motifs: int = 25_000,
    future_option_development_stage: str = "auto",
    incremental_promotion_validation: bool = False,
    promotion_min_incremental_coverage: float = 0.05,
    promotion_min_cross_context_or_game_evidence: int = 2,
    promotion_min_behavioral_or_predictive_lift: float = 0.01,
    promotion_min_relevant_heldout_event_count: int = 20,
    promotion_population_comparability_threshold: float = 0.80,
    promotion_demotion_failure_limit: int = 2,
    h11_provenance_sample_limit: int = 200,
    h11_write_full_provenance_jsonl: bool = True,
    max_h11_main_report_bytes: int = 5_000_000,
    allow_memory_repair: bool = False,
    hypothesis_progress: bool | None = None,
    hypothesis_progress_log_every: int = 1000,
) -> dict[str, Any]:
    del hypothesis_progress, hypothesis_progress_log_every
    started = time.time()
    run_dir = Path(run_dir)
    memory_dir = (
        None if memory_dir is None else Path(memory_dir)
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    derivation_summary = prepare_hypothesis_evidence(
        run_dir=run_dir,
        memory_dir=memory_dir,
        output_dir=output_dir,
        epoch_id=epoch_id,
        suite_mode=suite_mode,
        higher_order_workers=higher_order_workers,
        higher_order_transfer_chunk_size=
            higher_order_transfer_chunk_size,
        max_role_carriers=max_role_carriers,
        max_roles=max_roles,
        max_role_transfer_attempts=max_role_transfer_attempts,
        max_future_option_events=max_future_option_events,
        max_future_option_motifs=max_future_option_motifs,
        future_option_development_stage=
            future_option_development_stage,
        incremental_promotion_validation=
            incremental_promotion_validation,
        promotion_min_incremental_coverage=
            promotion_min_incremental_coverage,
        promotion_min_cross_context_or_game_evidence=
            promotion_min_cross_context_or_game_evidence,
        promotion_min_behavioral_or_predictive_lift=
            promotion_min_behavioral_or_predictive_lift,
        promotion_min_relevant_heldout_event_count=
            promotion_min_relevant_heldout_event_count,
        promotion_population_comparability_threshold=
            promotion_population_comparability_threshold,
        promotion_demotion_failure_limit=
            promotion_demotion_failure_limit,
        allow_memory_repair=allow_memory_repair,
    )

    fingerprint_before_report = memory_fingerprint(memory_dir)
    report_started = time.time()
    with read_only_evidence_snapshot(
        memory_dir
    ) as evidence_memory_dir:
        provenance = validate_hypothesis_provenance(
            memory_dir=evidence_memory_dir,
            output_dir=output_dir,
        )
        raw_results = evaluate_hypotheses_read_only(
            run_dir=run_dir,
            evidence_memory_dir=evidence_memory_dir,
            output_dir=output_dir,
            suite_mode=suite_mode,
            max_db_files=max_db_files,
            max_rows=max_rows,
            scan_all_dbs=scan_all_dbs,
            incremental_promotion_validation=
                incremental_promotion_validation,
            promotion_min_incremental_coverage=
                promotion_min_incremental_coverage,
            promotion_min_cross_context_or_game_evidence=
                promotion_min_cross_context_or_game_evidence,
            promotion_min_behavioral_or_predictive_lift=
                promotion_min_behavioral_or_predictive_lift,
            promotion_min_relevant_heldout_event_count=
                promotion_min_relevant_heldout_event_count,
            promotion_population_comparability_threshold=
                promotion_population_comparability_threshold,
            promotion_demotion_failure_limit=
                promotion_demotion_failure_limit,
            h11_provenance_sample_limit=
                h11_provenance_sample_limit,
            h11_write_full_provenance_jsonl=
                h11_write_full_provenance_jsonl,
            max_h11_main_report_bytes=
                max_h11_main_report_bytes,
        )

        fingerprint_after_report = memory_fingerprint(memory_dir)
        memory_unchanged = (
            fingerprint_before_report == fingerprint_after_report
        )
        normalized: dict[str, dict[str, Any]] = {}
        for hypothesis_id in (
            f"H{index:02d}" for index in range(1, 13)
        ):
            normalized[hypothesis_id] = apply_decision_envelope(
                hypothesis_id,
                raw_results.get(
                    hypothesis_id,
                    _failed_evaluator_result(
                        hypothesis_id,
                        RuntimeError("evaluator result missing"),
                    ),
                ),
                memory_dir=evidence_memory_dir,
                provenance=provenance,
                dependency_results=normalized,
                memory_unchanged=memory_unchanged,
            )
            _write_normalized_result(
                output_dir,
                hypothesis_id,
                normalized[hypothesis_id],
            )

    report_seconds = time.time() - report_started
    summary = build_hypothesis_suite_summary(
        epoch_id=epoch_id,
        run_dir=run_dir,
        memory_dir=memory_dir,
        hypothesis_results=normalized,
        derivation_summary=derivation_summary,
        provenance_validation=provenance,
        timings={
            "derive_seconds": sum(
                float(value)
                for key, value in (
                    derivation_summary.get("timings") or {}
                ).items()
                if str(key).endswith("_seconds")
            ),
            "report_seconds": report_seconds,
            "suite_total_seconds": time.time() - started,
        },
        global_step_start=global_step_start,
        global_step_end=global_step_end,
        interactions_this_epoch=interactions_this_epoch,
        total_interactions_seen=total_interactions_seen,
        memory_size_before_bytes=memory_size_before_bytes,
        memory_size_after_bytes=memory_size_after_bytes,
        source_memory_unchanged_during_report=memory_unchanged,
        evidence_snapshot_used=True,
    )
    _write_suite_summary(
        summary,
        output_dir,
        hypothesis_results=normalized,
    )
    return summary


def _apply_higher_order_dependency_gates(
    h04: dict[str, Any],
    h05: dict[str, Any],
    h06: dict[str, Any],
    h07: dict[str, Any],
    h08: dict[str, Any],
    h09: dict[str, Any],
    h10: dict[str, Any],
    h11: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[str],
]:
    """Compatibility helper. It no longer overwrites raw decisions."""
    results = {
        "H04": dict(h04),
        "H05": dict(h05),
        "H06": dict(h06),
        "H07": dict(h07),
        "H08": dict(h08),
        "H09": dict(h09),
        "H10": dict(h10),
        "H11": dict(h11),
    }
    notes: list[str] = []
    dependencies = {
        "H05": ("H04",),
        "H06": ("H05",),
        "H07": ("H06",),
        "H08": ("H06", "H07"),
        "H10": ("H09",),
        "H11": ("H06", "H09"),
    }
    for hypothesis_id, required in dependencies.items():
        failed = [
            item
            for item in required
            if str(
                results[item].get(
                    "final_decision",
                    results[item].get("decision"),
                )
            )
            != "VALID"
        ]
        payload = results[hypothesis_id]
        raw = payload.setdefault(
            "raw_decision", payload.get("decision")
        )
        payload["dependency_gate"] = {
            "status": "PASS" if not failed else "FAIL",
            "dependencies": list(required),
            "failed_dependencies": failed,
        }
        final = str(
            payload.get("final_decision", payload.get("decision"))
        )
        if failed and final == "VALID":
            final = "PARTIALLY_VALID"
            notes.append(
                f"{hypothesis_id} dependency gate failed: "
                + ", ".join(failed)
            )
        payload["final_decision"] = final
        payload["decision"] = final
        payload["raw_decision"] = raw
    return (
        results["H05"],
        results["H06"],
        results["H07"],
        results["H08"],
        results["H09"],
        results["H10"],
        results["H11"],
        notes,
    )


def _apply_epoch_maturity_gates(
    **kwargs: Any,
) -> tuple[Any, ...]:
    """Compatibility gate that preserves raw decisions.

    Expected payload arguments are H04-H11 dictionaries. Unknown metadata
    arguments are ignored. The return order follows the supplied H keys and
    appends a notes list.
    """
    keys = [
        key
        for key in (
            "h04", "h05", "h06", "h07",
            "h08", "h09", "h10", "h11",
        )
        if key in kwargs
    ]
    notes: list[str] = []
    epoch_number = kwargs.get(
        "epoch_number",
        kwargs.get("current_epoch", kwargs.get("epoch")),
    )
    outputs: list[Any] = []
    for key in keys:
        payload = dict(kwargs[key])
        payload.setdefault(
            "raw_decision", payload.get("decision")
        )
        payload.setdefault(
            "maturity_gate",
            {
                "status": "PASS",
                "epoch": epoch_number,
                "reasons": [],
            },
        )
        payload.setdefault(
            "final_decision", payload.get("decision")
        )
        outputs.append(payload)
    outputs.append(notes)
    return tuple(outputs)


def _format_aggregated_result_section(
    hypothesis_id: str,
    result: Mapping[str, Any],
) -> str:
    return "\n".join(
        (
            hypothesis_id,
            f"raw_decision: {result.get('raw_decision')}",
            (
                "evidence_contract_gate: "
                f"{result.get('evidence_contract_gate', {}).get('status')}"
            ),
            (
                "quality_gate: "
                f"{result.get('quality_gate', {}).get('status')}"
            ),
            (
                "dependency_gate: "
                f"{result.get('dependency_gate', {}).get('status')}"
            ),
            f"final_decision: {result.get('final_decision')}",
            (
                "core_metrics: "
                + json.dumps(
                    result.get("core_metrics") or {},
                    sort_keys=True,
                )
            ),
            (
                "missing_evidence: "
                + json.dumps(
                    result.get("missing_evidence") or []
                )
            ),
        )
    )


def _write_aggregated_hypothesis_text(
    output_dir: Path,
    *,
    hypothesis_results: Mapping[
        str, Mapping[str, Any]
    ] | None = None,
) -> None:
    results = dict(hypothesis_results or {})
    sections = [
        _format_aggregated_result_section(
            hypothesis_id,
            results.get(
                hypothesis_id,
                {
                    "raw_decision": "INSUFFICIENT_EVIDENCE",
                    "final_decision": "INSUFFICIENT_EVIDENCE",
                },
            ),
        )
        for hypothesis_id in (
            f"H{index:02d}" for index in range(1, 13)
        )
    ]
    Path(output_dir, SUITE_AGGREGATED_TXT_NAME).write_text(
        "\n\n".join(sections) + "\n",
        encoding="utf-8",
    )


def _write_suite_summary(
    summary: Mapping[str, Any],
    output_dir: Path,
    *,
    hypothesis_results: Mapping[
        str, Mapping[str, Any]
    ] | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.joinpath(SUITE_JSON_NAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    text_lines = [
        "ARC-AGI3 hypothesis suite v6.1",
        f"epoch: {summary.get('epoch_id')}",
        "phase order: WRITE -> DERIVE -> REPORT",
        "evaluators read-only: true",
    ]
    for hypothesis_id in (
        f"H{index:02d}" for index in range(1, 13)
    ):
        result = (
            summary.get("hypotheses", {}).get(
                hypothesis_id, {}
            )
        )
        text_lines.append(
            f"{hypothesis_id}: "
            f"raw={result.get('raw_decision')} "
            f"contract={result.get('evidence_contract_gate', {}).get('status')} "
            f"quality={result.get('quality_gate', {}).get('status')} "
            f"dependency={result.get('dependency_gate', {}).get('status')} "
            f"final={result.get('final_decision')}"
        )
    output_dir.joinpath(SUITE_TXT_NAME).write_text(
        "\n".join(text_lines) + "\n",
        encoding="utf-8",
    )
    output_dir.joinpath(SUITE_MD_NAME).write_text(
        "# ARC-AGI3 hypothesis suite v6.1\n\n"
        + "\n".join(
            f"- **{hypothesis_id}**: "
            f"`{summary.get('final_decisions', {}).get(hypothesis_id)}`"
            for hypothesis_id in (
                f"H{index:02d}" for index in range(1, 13)
            )
        )
        + "\n",
        encoding="utf-8",
    )
    _write_aggregated_hypothesis_text(
        output_dir,
        hypothesis_results=hypothesis_results,
    )


from v6.hypothesis_suite_legacy_compat import install_compat as _install_hypothesis_suite_compat
_install_hypothesis_suite_compat(globals())

# v6.3 canonical temporal summary
_build_hypothesis_suite_summary_base = build_hypothesis_suite_summary

def build_hypothesis_suite_summary(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from v6.v63_semantics import _strict_temporal_summary
    summary = _build_hypothesis_suite_summary_base(*args, **kwargs)
    run_dir = kwargs.get("run_dir")
    if run_dir is not None:
        strict = _strict_temporal_summary(run_dir)
        if strict is not None:
            summary["temporal_order_diagnostics"] = strict
    return summary
