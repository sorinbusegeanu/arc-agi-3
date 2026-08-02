from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from v6.memory.direct_streaming_fold import direct_streaming_manifest_exists

from v6.hypothesis_h01_report import evaluate_h01_contingency_emergence
from v6.hypothesis_h02_report import evaluate_h02_prediction_violation_attention
from v6.hypothesis_h03_report import evaluate_h03_transformation_family_formation
from v6.hypothesis_h04_report import evaluate_h04_carrier_emergence
from v6.higher_order_substrate import (
    IncrementalPromotionValidationConfig,
    derive_concept_candidates_only,
    derive_role_candidates_only,
    derive_role_transfer_attempts_only,
    derive_world_model_components_only,
    validate_incremental_promotions_only,
)
from v6.future_options import derive_future_option_memory
from v6.memory.compact_memory import (
    derive_missing_transformation_families_from_stable_contingencies,
    ensure_memory_layout,
)
from v6.hypothesis_h05_report import evaluate_h05_role_emergence
from v6.hypothesis_h06_report import evaluate_h06_role_transfer
from v6.hypothesis_h07_report import evaluate_h07_concept_emergence
from v6.hypothesis_h08_report import evaluate_h08_world_model_coherence
from v6.hypothesis_h09_report import evaluate_h09_future_option_motifs
from v6.hypothesis_h10_report import evaluate_h10_future_option_attention
from v6.hypothesis_h11_report import evaluate_h11_future_option_transfer_concepts
from v6.evaluation.h12_efficiency_emergence import evaluate_h12_efficiency_emergence
from v6.provenance_validation import validate_hypothesis_provenance


SUITE_JSON_NAME = "hypothesis_suite_summary.json"
SUITE_TXT_NAME = "hypothesis_suite_summary.txt"
SUITE_MD_NAME = "hypothesis_suite_summary.md"
SUITE_AGGREGATED_TXT_NAME = "hypothesis_suite_aggregated.txt"
INPUT_REPORT_NAME = "interaction_sampling_v05c_report.json"
SUITE_PHASE_LOG_NAME = "hypothesis_phase_log.jsonl"
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
        "current": None if current is None else int(current),
        "total": None if total is None else int(total),
    }
    elapsed = None if start_time is None else max(0.0, now - float(start_time))
    if elapsed is not None:
        payload["seconds_elapsed"] = float(elapsed)
        if current is not None and elapsed > 0.0:
            rate = float(current) / elapsed
            payload["rate_per_second"] = rate
            if total is not None and rate > 0.0:
                payload["eta_seconds"] = max(0.0, float(total - current) / rate)
    if extra:
        payload.update(dict(extra))
    with (output_dir / SUITE_PHASE_LOG_NAME).open("a", encoding="utf-8") as handle:
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

    def update(self, n: int = 1, *, current: int | None = None, extra: dict[str, Any] | None = None) -> None:
        if current is not None:
            n = int(current) - int(self.current)
            self.current = int(current)
        else:
            self.current += int(n)
        if self._bar is not None and n:
            self._bar.update(int(n))
        if self.current >= self.log_every + self.last_logged or (self.total is not None and self.current >= int(self.total)):
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

    def close(self, *, status: str = "done", extra: dict[str, Any] | None = None) -> None:
        if self._bar is not None:
            self._bar.close()
        log_hypothesis_progress(
            self.output_dir,
            self.phase,
            status,
            epoch_id=self.epoch_id,
            current=self.current if self.total is None else min(self.current, int(self.total)),
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
                "exception_type": exc_type.__name__ if exc_type is not None else type(exc).__name__,
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
    allow_memory_repair: bool = False,
    hypothesis_progress: bool | None = None,
    hypothesis_progress_log_every: int = 1000,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    memory_dir = None if memory_dir is None else Path(memory_dir)
    validation_state_reset_applied_this_run = (
        ensure_memory_layout(memory_dir).validation_state_reset_applied_this_run
        if memory_dir is not None else False
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    h01_dir = output_dir / "h01"
    h02_dir = output_dir / "h02"
    h03_dir = output_dir / "h03"
    h04_dir = output_dir / "h04"
    h05_dir = output_dir / "h05"
    h06_dir = output_dir / "h06"
    h07_dir = output_dir / "h07"
    h08_dir = output_dir / "h08"
    h09_dir = output_dir / "h09"
    h10_dir = output_dir / "h10"
    h11_dir = output_dir / "h11"
    h12_dir = output_dir / "h12"
    suite_started_at = time.time()
    resolved_suite_mode = str(suite_mode or "fast").strip().lower()
    if resolved_suite_mode not in {"fast", "full"}:
        resolved_suite_mode = "fast"
    progress_enabled = True if hypothesis_progress is None else bool(hypothesis_progress)
    phase_names = [
        "family_repair", "H01", "H02", "H03", "H04",
        "derive_role_candidates", "H05", "derive_role_transfer_attempts", "H06",
        "derive_concept_candidates", "H07", "derive_world_model_components", "H08",
        "derive_future_option_events", "derive_future_option_motifs", "derive_future_option_attention_links",
        "derive_future_option_transfer_links", "H09", "H10", "H11", "H12", "summary_write",
    ]
    if incremental_promotion_validation:
        phase_names[phase_names.index("H07"):phase_names.index("H07")] = ["validate_concept_promotions"]
        phase_names[phase_names.index("H08"):phase_names.index("H08")] = ["validate_world_model_promotions"]
    top_bar = tqdm(total=len(phase_names), desc="hypothesis suite", unit="phase", dynamic_ncols=True, leave=True, disable=not progress_enabled)
    timings: dict[str, float] = {
        "family_repair_seconds": 0.0,
        "h01_seconds": 0.0,
        "h02_seconds": 0.0,
        "h03_seconds": 0.0,
        "h04_seconds": 0.0,
        "derive_role_candidates_seconds": 0.0,
        "derive_role_transfer_attempts_seconds": 0.0,
        "derive_concept_candidates_seconds": 0.0,
        "derive_world_model_components_seconds": 0.0,
        "validate_concept_promotions_seconds": 0.0,
        "validate_world_model_promotions_seconds": 0.0,
        "h05_seconds": 0.0,
        "h06_seconds": 0.0,
        "h07_seconds": 0.0,
        "h08_seconds": 0.0,
        "derive_future_option_events_seconds": 0.0,
        "derive_future_option_motifs_seconds": 0.0,
        "derive_future_option_attention_links_seconds": 0.0,
        "derive_future_option_transfer_links_seconds": 0.0,
        "h09_seconds": 0.0,
        "h10_seconds": 0.0,
        "h11_seconds": 0.0,
        "h12_seconds": 0.0,
        "suite_total_seconds": 0.0,
    }
    promotion_validation_summary: dict[str, Any] = {
        "incremental_promotion_validation_enabled": bool(incremental_promotion_validation),
    }

    @contextmanager
    def _phase(name: str):
        with hypothesis_phase(
            output_dir,
            name,
            epoch_id=epoch_id,
            total=1,
            unit="phase",
            enabled=progress_enabled,
            leave=False,
            log_every=hypothesis_progress_log_every,
            top_bar=top_bar,
        ) as tracker:
            yield tracker

    def _progress_factory(phase: str, total: int | None, unit: str, leave: bool = False) -> _HypothesisPhaseTracker:
        return _HypothesisPhaseTracker(
            output_dir=output_dir,
            phase=phase,
            epoch_id=epoch_id,
            total=total,
            unit=unit,
            enabled=progress_enabled,
            leave=leave,
            log_every=hypothesis_progress_log_every,
        ).__enter__()

    family_repair_summary: dict[str, Any] = {}
    with _phase("family_repair"):
        if memory_dir is not None:
            t0 = time.time()
            if allow_memory_repair:
                family_repair_summary = derive_missing_transformation_families_from_stable_contingencies(memory_dir)
            timings["family_repair_seconds"] = float(time.time() - t0)
    with _phase("H01"):
        t0 = time.time()
        h01 = evaluate_h01_contingency_emergence(run_dir=run_dir, output_dir=h01_dir, memory_dir=memory_dir)
        timings["h01_seconds"] = float(time.time() - t0)
    with _phase("H02"):
        t0 = time.time()
        h02 = evaluate_h02_prediction_violation_attention(
            run_dir=run_dir,
            output_dir=h02_dir,
            memory_dir=memory_dir,
            max_rows=int(max_rows),
            max_db_files=int(max_db_files),
            scan_all_dbs=bool(scan_all_dbs),
        )
        timings["h02_seconds"] = float(time.time() - t0)
    with _phase("H03"):
        t0 = time.time()
        h03 = evaluate_h03_transformation_family_formation(
            run_dir=run_dir,
            output_dir=h03_dir,
            memory_dir=memory_dir,
            max_db_files=int(max_db_files),
            max_rows=int(max_rows),
            scan_all_dbs=bool(scan_all_dbs),
        )
        timings["h03_seconds"] = float(time.time() - t0)
    if isinstance(h03, dict):
        h03.update(family_repair_summary)
    with _phase("H04"):
        t0 = time.time()
        h04 = (
            evaluate_h04_carrier_emergence(memory_dir=memory_dir, run_dir=run_dir, output_dir=h04_dir)
            if memory_dir is not None
            else {"hypothesis_id": "H04", "decision": "INCONCLUSIVE", "core_metrics": {}, "missing_evidence": ["memory_dir not provided"]}
        )
        timings["h04_seconds"] = float(time.time() - t0)

    if memory_dir is not None:
        promotion_validation_config = IncrementalPromotionValidationConfig(
            enabled=bool(incremental_promotion_validation),
            min_incremental_coverage=float(promotion_min_incremental_coverage),
            min_cross_context_or_game_evidence=int(promotion_min_cross_context_or_game_evidence),
            min_behavioral_or_predictive_lift=float(promotion_min_behavioral_or_predictive_lift),
            min_relevant_heldout_event_count=int(promotion_min_relevant_heldout_event_count),
            promotion_population_comparability_threshold=float(promotion_population_comparability_threshold),
            demotion_failure_limit=int(promotion_demotion_failure_limit),
        )
        with _phase("derive_role_candidates"):
            t0 = time.time()
            derive_role_candidates_only(memory_dir=memory_dir, run_dir=run_dir, max_carriers=int(max_role_carriers), max_roles=int(max_roles), progress_factory=_progress_factory)
            timings["derive_role_candidates_seconds"] = float(time.time() - t0)
        with _phase("H05"):
            t0 = time.time()
            h05 = evaluate_h05_role_emergence(memory_dir=memory_dir, run_dir=run_dir, output_dir=h05_dir, already_derived=True)
            timings["h05_seconds"] = float(time.time() - t0)
        if resolved_suite_mode == "full":
            with _phase("derive_role_transfer_attempts"):
                t0 = time.time()
                derive_role_transfer_attempts_only(memory_dir=memory_dir, run_dir=run_dir, max_transfer_attempts=int(max_role_transfer_attempts), workers=int(higher_order_workers), chunk_size=int(higher_order_transfer_chunk_size), progress_factory=_progress_factory)
                timings["derive_role_transfer_attempts_seconds"] = float(time.time() - t0)
            with _phase("H06"):
                t0 = time.time()
                h06 = evaluate_h06_role_transfer(memory_dir=memory_dir, run_dir=run_dir, output_dir=h06_dir, already_derived=True)
                timings["h06_seconds"] = float(time.time() - t0)
            with _phase("derive_concept_candidates"):
                t0 = time.time()
                derive_concept_candidates_only(memory_dir=memory_dir, run_dir=run_dir, progress_factory=_progress_factory)
                timings["derive_concept_candidates_seconds"] = float(time.time() - t0)
            if promotion_validation_config.enabled:
                with _phase("validate_concept_promotions"):
                    t0 = time.time()
                    promotion_validation_summary = validate_incremental_promotions_only(
                        memory_dir=memory_dir,
                        config=promotion_validation_config,
                        validate_roles_and_concepts=True,
                        validate_world_models=False,
                        diagnostic_epoch_id=epoch_id,
                        explanation_events_path=h07_dir / "h07_concept_explanation_events.jsonl",
                        validation_state_reset_applied_this_run=validation_state_reset_applied_this_run,
                    )
                    timings["validate_concept_promotions_seconds"] = float(time.time() - t0)
            with _phase("H07"):
                t0 = time.time()
                h07 = evaluate_h07_concept_emergence(
                    memory_dir=memory_dir,
                    run_dir=run_dir,
                    output_dir=h07_dir,
                    already_derived=True,
                    incremental_promotion_validation=promotion_validation_config,
                )
                timings["h07_seconds"] = float(time.time() - t0)
            with _phase("derive_world_model_components"):
                t0 = time.time()
                derive_world_model_components_only(memory_dir=memory_dir, run_dir=run_dir, progress_factory=_progress_factory)
                timings["derive_world_model_components_seconds"] = float(time.time() - t0)
            if promotion_validation_config.enabled:
                with _phase("validate_world_model_promotions"):
                    t0 = time.time()
                    world_validation_summary = validate_incremental_promotions_only(
                        memory_dir=memory_dir,
                        config=promotion_validation_config,
                        validate_roles_and_concepts=False,
                        validate_world_models=True,
                        diagnostic_epoch_id=epoch_id,
                        validation_state_reset_applied_this_run=False,
                    )
                    promotion_validation_summary["world_model_components_demoted"] = int(
                        world_validation_summary.get("world_model_components_demoted", 0) or 0
                    )
                    timings["validate_world_model_promotions_seconds"] = float(time.time() - t0)
            with _phase("H08"):
                t0 = time.time()
                h08 = evaluate_h08_world_model_coherence(memory_dir=memory_dir, run_dir=run_dir, output_dir=h08_dir, already_derived=True)
                timings["h08_seconds"] = float(time.time() - t0)
            with _phase("derive_future_option_events"):
                with _phase("derive_future_option_motifs"):
                    with _phase("derive_future_option_attention_links"):
                        with _phase("derive_future_option_transfer_links"):
                            future_t0 = time.time()
                            future_summary = derive_future_option_memory(
                                memory_dir=memory_dir,
                                run_dir=run_dir,
                                max_events=int(max_future_option_events),
                                max_motifs=int(max_future_option_motifs),
                                development_stage=str(future_option_development_stage),
                                progress_factory=_progress_factory,
                            )
                            timings["derive_future_option_events_seconds"] = float(future_summary.get("derive_future_option_events_seconds", time.time() - future_t0))
                            timings["derive_future_option_motifs_seconds"] = float(future_summary.get("derive_future_option_motifs_seconds", 0.0))
        else:
            h06 = _skipped_fast_mode_result("H06")
            h07 = _skipped_fast_mode_result("H07")
            h08 = _skipped_fast_mode_result("H08")
            h09 = _skipped_fast_mode_result("H09")
            h10 = _skipped_fast_mode_result("H10")
            h11 = _skipped_fast_mode_result("H11")
            skipped_phase_names = [
                "derive_role_transfer_attempts",
                "H06",
                "derive_concept_candidates",
            ]
            if promotion_validation_config.enabled:
                skipped_phase_names.append("validate_concept_promotions")
            skipped_phase_names.extend([
                "H07",
                "derive_world_model_components",
            ])
            if promotion_validation_config.enabled:
                skipped_phase_names.append("validate_world_model_promotions")
            skipped_phase_names.extend([
                "H08",
                "derive_future_option_events",
                "derive_future_option_motifs",
                "derive_future_option_attention_links",
                "derive_future_option_transfer_links",
                "H09",
                "H10",
                "H11",
            ])
            for phase_name in skipped_phase_names:
                with _phase(phase_name):
                    pass
        if resolved_suite_mode == "full":
            with _phase("H09"):
                t0 = time.time()
                h09 = evaluate_h09_future_option_motifs(memory_dir=memory_dir, run_dir=run_dir, output_dir=h09_dir, already_derived=True)
                timings["h09_seconds"] = float(time.time() - t0)
            with _phase("H10"):
                t0 = time.time()
                h10 = evaluate_h10_future_option_attention(memory_dir=memory_dir, run_dir=run_dir, output_dir=h10_dir, already_derived=True)
                timings["h10_seconds"] = float(time.time() - t0)
            with _phase("H11"):
                t0 = time.time()
                h11 = evaluate_h11_future_option_transfer_concepts(memory_dir=memory_dir, run_dir=run_dir, output_dir=h11_dir, already_derived=True)
                timings["h11_seconds"] = float(time.time() - t0)
        with _phase("H12"):
            t0 = time.time()
            h12 = evaluate_h12_efficiency_emergence(memory_dir=memory_dir, run_dir=run_dir, output_dir=h12_dir)
            timings["h12_seconds"] = float(time.time() - t0)
    else:
        missing = ["memory_dir not provided"]
        h05 = {"hypothesis_id": "H05", "decision": "INSUFFICIENT_EVIDENCE", "core_metrics": {}, "missing_evidence": missing, "evidence_source": "none"}
        h06 = {"hypothesis_id": "H06", "decision": "INSUFFICIENT_EVIDENCE", "core_metrics": {}, "missing_evidence": missing, "evidence_source": "none"}
        h07 = {"hypothesis_id": "H07", "decision": "INSUFFICIENT_EVIDENCE", "core_metrics": {}, "missing_evidence": missing, "evidence_source": "none"}
        h08 = {"hypothesis_id": "H08", "decision": "INSUFFICIENT_EVIDENCE", "core_metrics": {}, "missing_evidence": missing, "evidence_source": "none"}
        h09 = {"hypothesis_id": "H09", "decision": "INSUFFICIENT_EVIDENCE", "core_metrics": {}, "missing_evidence": missing, "evidence_source": "none"}
        h10 = {"hypothesis_id": "H10", "decision": "INSUFFICIENT_EVIDENCE", "core_metrics": {}, "missing_evidence": missing, "evidence_source": "none"}
        h11 = {"hypothesis_id": "H11", "decision": "INSUFFICIENT_EVIDENCE", "core_metrics": {}, "missing_evidence": missing, "evidence_source": "none"}
        h12 = {"hypothesis_id": "H12", "decision": "INSUFFICIENT_EVIDENCE", "core_metrics": {}, "missing_evidence": missing, "evidence_source": "none"}
        for phase_name in ("derive_role_candidates", "H05", "derive_role_transfer_attempts", "H06", "derive_concept_candidates", "H07", "derive_world_model_components", "H08", "derive_future_option_events", "derive_future_option_motifs", "derive_future_option_attention_links", "derive_future_option_transfer_links", "H09", "H10", "H11", "H12"):
            with _phase(phase_name):
                pass
    provenance_validation = validate_hypothesis_provenance(memory_dir=memory_dir, output_dir=output_dir)
    _apply_global_provenance_gates(
        provenance_validation,
        {"H06": h06, "H07": h07, "H08": h08, "H09": h09, "H11": h11},
    )
    input_report = _load_json(Path(run_dir) / INPUT_REPORT_NAME) or {}
    runs = [dict(item) for item in input_report.get("runs", []) if isinstance(item, dict)]
    games = sorted({str(row.get("game")) for row in runs if row.get("game")})
    if not games:
        games = [str(item) for item in input_report.get("games", []) if item]
    samplers = sorted({str(row.get("sampler_name")) for row in runs if row.get("sampler_name")})
    if not samplers:
        samplers = [str(item) for item in input_report.get("samplers", []) if item]
    total_interactions_resolved = _resolve_total_interactions(
        runs=runs,
        h01=h01,
        interactions_this_epoch=interactions_this_epoch,
        total_interactions_seen=total_interactions_seen,
    )
    h04, h05, h06, h07, h08, h09, h10, h11, maturity_notes = _apply_epoch_maturity_gates(
        h04=h04,
        h05=h05,
        h06=h06,
        h07=h07,
        h08=h08,
        h09=h09,
        h10=h10,
        h11=h11,
        total_interactions=int(total_interactions_resolved["total_interactions"]),
        interactions_this_epoch=interactions_this_epoch,
        game_count=len(games),
        sampler_count=len(samplers),
    )
    h05, h06, h07, h08, h09, h10, h11, dependency_notes = _apply_higher_order_dependency_gates(h04, h05, h06, h07, h08, h09, h10, h11)
    summary = build_hypothesis_suite_summary(
        run_dir=run_dir,
        memory_dir=memory_dir,
        h01=h01,
        h02=h02,
        h03=h03,
        h04=h04,
        h05=h05,
        h06=h06,
        h07=h07,
        h08=h08,
        h09=h09,
        h10=h10,
        h11=h11,
        h12=h12,
        epoch_id=epoch_id,
        global_step_start=global_step_start,
        global_step_end=global_step_end,
        interactions_this_epoch=interactions_this_epoch,
        total_interactions_seen=total_interactions_seen,
        memory_size_before_bytes=memory_size_before_bytes,
        memory_size_after_bytes=memory_size_after_bytes,
        resolved_total_interactions=total_interactions_resolved,
        higher_order_dependency_gate_notes=dependency_notes,
        epoch_maturity_gate_notes=maturity_notes,
    )
    timings["suite_total_seconds"] = float(time.time() - suite_started_at)
    for key, payload in {
        "H01": h01,
        "H02": h02,
        "H03": h03,
        "H04": h04,
        "H05": h05,
        "H06": h06,
        "H07": h07,
        "H08": h08,
        "H09": h09,
        "H10": h10,
        "H11": h11,
        "H12": h12,
    }.items():
        if isinstance(payload, dict):
            payload["phase_seconds"] = {f"{key.lower()}_seconds": timings.get(f"{key.lower()}_seconds", 0.0)}
    summary.update(timings)
    summary["suite_mode"] = resolved_suite_mode
    summary["memory_repair_ran_during_report"] = bool(allow_memory_repair)
    summary["max_role_transfer_attempts"] = int(max_role_transfer_attempts)
    summary["max_future_option_events"] = int(max_future_option_events)
    summary["max_future_option_motifs"] = int(max_future_option_motifs)
    summary["incremental_promotion_validation"] = promotion_validation_summary
    summary["provenance_validation"] = provenance_validation
    consistency_warnings = _collect_hypothesis_consistency_warnings(
        h03=h03, h04=h04, h05=h05, h07=h07, h08=h08, h09=h09, h10=h10, h12=h12
    )
    summary["hypothesis_consistency_warnings"] = list(consistency_warnings)
    with hypothesis_phase(
        output_dir,
        "summary_write",
        epoch_id=epoch_id,
        total=1,
        unit="phase",
        enabled=progress_enabled,
        leave=False,
        log_every=hypothesis_progress_log_every,
        top_bar=top_bar,
    ):
        _write_suite_summary(
            summary,
            output_dir,
            hypothesis_results={
                "H01": h01,
                "H02": h02,
                "H03": h03,
                "H04": h04,
                "H05": h05,
                "H06": h06,
                "H07": h07,
                "H08": h08,
                "H09": h09,
                "H10": h10,
                "H11": h11,
                "H12": h12,
            },
        )
    top_bar.close()
    return summary


def _apply_global_provenance_gates(
    validation: dict[str, Any],
    hypotheses: dict[str, dict[str, Any]],
) -> None:
    """No VALID decision may survive a contradictory provenance claim."""
    invalid_by_hypothesis = dict(validation.get("invalid_by_hypothesis") or {})
    for hypothesis_id, result in hypotheses.items():
        invalid_count = int(invalid_by_hypothesis.get(hypothesis_id, 0) or 0)
        result["provenance_validation_invalid_claim_count"] = invalid_count
        result["provenance_validation_report"] = "provenance_validation_report.json"
        if invalid_count > 0 and result.get("decision") == "VALID":
            result["decision"] = "INSUFFICIENT_EVIDENCE"
            missing = result.setdefault("missing_evidence", [])
            message = f"Global provenance validation found {invalid_count} invalid claim(s); VALID is not admissible."
            if message not in missing:
                missing.append(message)


def build_hypothesis_suite_summary(
    *,
    run_dir: Path,
    memory_dir: Path | None = None,
    h01: dict[str, Any],
    h02: dict[str, Any],
    h03: dict[str, Any],
    h04: dict[str, Any] | None = None,
    h05: dict[str, Any] | None = None,
    h06: dict[str, Any] | None = None,
    h07: dict[str, Any] | None = None,
    h08: dict[str, Any] | None = None,
    h09: dict[str, Any] | None = None,
    h10: dict[str, Any] | None = None,
    h11: dict[str, Any] | None = None,
    h12: dict[str, Any] | None = None,
    epoch_id: str | None = None,
    global_step_start: int | None = None,
    global_step_end: int | None = None,
    interactions_this_epoch: int | None = None,
    total_interactions_seen: int | None = None,
    memory_size_before_bytes: int | None = None,
    memory_size_after_bytes: int | None = None,
    resolved_total_interactions: dict[str, Any] | None = None,
    higher_order_dependency_gate_notes: list[str] | None = None,
    epoch_maturity_gate_notes: list[str] | None = None,
) -> dict[str, Any]:
    input_report = _load_json(Path(run_dir) / INPUT_REPORT_NAME) or {}
    runs = [dict(item) for item in input_report.get("runs", []) if isinstance(item, dict)]
    temporal_rows = list(dict(item) for item in ((input_report.get("temporal_milestones") or {}).get("by_game_sampler_seed", []) or []) if isinstance(item, dict))
    games = sorted({str(row.get("game")) for row in runs if row.get("game")})
    if not games:
        games = [str(item) for item in input_report.get("games", []) if item]
    samplers = sorted({str(row.get("sampler_name")) for row in runs if row.get("sampler_name")})
    if not samplers:
        samplers = [str(item) for item in input_report.get("samplers", []) if item]
    seeds = sorted({int(item.get("seed")) for item in temporal_rows if item.get("seed") is not None})
    if not seeds:
        seeds = [int(item) for item in input_report.get("seeds", []) if item is not None]

    per_game = _per_game_diagnostics(runs)
    per_sampler = _per_sampler_diagnostics(runs)
    temporal = _temporal_order_diagnostics(temporal_rows)
    resolved_total_interactions = resolved_total_interactions or _resolve_total_interactions(
        runs=runs,
        h01=h01,
        interactions_this_epoch=interactions_this_epoch,
        total_interactions_seen=total_interactions_seen,
    )
    total_interactions = int(resolved_total_interactions["total_interactions"])
    raw_total_interactions = int(resolved_total_interactions["raw_report_total_interactions"])
    total_interactions_source = str(resolved_total_interactions["total_interactions_source"])

    missing_evidence = _merge_unique(
        list(h01.get("missing_evidence", [])),
        list(h02.get("missing_evidence", [])),
        list(h03.get("missing_evidence", [])),
        list((h04 or {}).get("missing_evidence", [])),
        list((h05 or {}).get("missing_evidence", [])),
        list((h06 or {}).get("missing_evidence", [])),
        list((h07 or {}).get("missing_evidence", [])),
        list((h08 or {}).get("missing_evidence", [])),
        list((h09 or {}).get("missing_evidence", [])),
        list((h10 or {}).get("missing_evidence", [])),
        list((h11 or {}).get("missing_evidence", [])),
        list((h12 or {}).get("missing_evidence", [])),
    )
    h04 = h04 or {"decision": "INCONCLUSIVE", "core_metrics": {}}
    h05 = h05 or {"decision": "INCONCLUSIVE", "core_metrics": {}}
    h06 = h06 or {"decision": "INCONCLUSIVE", "core_metrics": {}}
    h07 = h07 or {"decision": "INCONCLUSIVE", "core_metrics": {}}
    h08 = h08 or {"decision": "INCONCLUSIVE", "core_metrics": {}}
    h09 = h09 or {"decision": "INCONCLUSIVE", "core_metrics": {}}
    h10 = h10 or {"decision": "INCONCLUSIVE", "core_metrics": {}}
    h11 = h11 or {"decision": "INCONCLUSIVE", "core_metrics": {}}
    h12 = h12 or {"decision": "INCONCLUSIVE", "core_metrics": {}}
    summary = {
        "epoch_id": epoch_id,
        "global_step_start": global_step_start,
        "global_step_end": global_step_end,
        "source_run_dir": str(run_dir),
        "memory_dir": None if memory_dir is None else str(memory_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "game_count": len(games),
        "sampler_count": len(samplers),
        "seed_count": len(seeds),
        "total_interactions": total_interactions,
        "raw_report_total_interactions": raw_total_interactions,
        "total_interactions_source": total_interactions_source,
        "interactions_this_epoch": interactions_this_epoch,
        "total_interactions_seen": total_interactions_seen,
        "memory_size_before_bytes": memory_size_before_bytes,
        "memory_size_after_bytes": memory_size_after_bytes,
        "direct_streaming_fold_used": bool(memory_dir is not None and direct_streaming_manifest_exists(memory_dir)),
        "raw_epoch_db_cleanup_used": bool(memory_dir is not None and direct_streaming_manifest_exists(memory_dir) and not list(Path(run_dir).rglob("*.sqlite"))),
        "raw_db_fallback_disabled": bool(memory_dir is not None and direct_streaming_manifest_exists(memory_dir)),
        "direct_streaming_fold_manifest_path": None if memory_dir is None else str(Path(memory_dir) / "direct_streaming_fold_manifest.sqlite"),
        "compact_family_repair_used": h03.get("compact_family_repair_used"),
        "compact_family_repair_reason": h03.get("compact_family_repair_reason"),
        "compact_family_repair_family_count": h03.get("compact_family_repair_family_count"),
        "compact_family_repair_member_count": h03.get("compact_family_repair_member_count"),
        "Levels": int(input_report.get("Levels", 0) or 0),
        "Games": int(input_report.get("Games", 0) or 0),
        "Total_Levels": int(input_report.get("Total_Levels", 0) or 0),
        "Total_Games": int(input_report.get("Total_Games", 0) or 0),
        "H01 decision": h01.get("decision"),
        "H02 decision": h02.get("decision"),
        "H03 decision": h03.get("decision"),
        "H04 decision": h04.get("decision", "INCONCLUSIVE"),
        "H05 decision": h05.get("decision", "INCONCLUSIVE"),
        "H06 decision": h06.get("decision", "INCONCLUSIVE"),
        "H07 decision": h07.get("decision", "INCONCLUSIVE"),
        "H08 decision": h08.get("decision", "INCONCLUSIVE"),
        "H08 hypothesis name": h08.get("hypothesis_name", "World-model coherence from promoted concepts"),
        "H09 decision": h09.get("decision", "INCONCLUSIVE"),
        "H10 decision": h10.get("decision", "INCONCLUSIVE"),
        "H11 decision": h11.get("decision", "INCONCLUSIVE"),
        "H12 decision": h12.get("decision", "INCONCLUSIVE"),
        "H04 suite gating": _suite_gate_status(h04),
        "H05 suite gating": _suite_gate_status(h05),
        "H06 suite gating": _suite_gate_status(h06),
        "H07 suite gating": _suite_gate_status(h07),
        "H08 suite gating": _suite_gate_status(h08),
        "H09 suite gating": _suite_gate_status(h09),
        "H10 suite gating": _suite_gate_status(h10),
        "H11 suite gating": _suite_gate_status(h11),
        "H01 core metrics": {
            "stable_contingency_count": h01.get("stable_contingency_count"),
            "interaction_count": h01.get("total_interaction_count"),
            "mean_prediction_accuracy": h01.get("mean_prediction_accuracy"),
            "games_with_stable_contingencies": _count_positive(h01.get("per_game_contingency_counts")),
            "samplers_with_stable_contingencies": _count_positive(h01.get("per_sampler_contingency_counts")),
        },
        "H02 core metrics": {
            "h02a_replay_attention_decision": h02.get("h02a_replay_attention_decision"),
            "h02b_pre_carrier_timing_decision": h02.get("h02b_pre_carrier_timing_decision"),
            "h02_final_decision_basis": h02.get("h02_final_decision_basis"),
            "carrier_timing_note": h02.get("carrier_timing_note"),
            "prediction_violation_replay_lift": h02.get("prediction_violation_replay_lift"),
            "prediction_violation_base_ratio": h02.get("prediction_violation_base_ratio"),
            "high_priority_replay_prediction_violation_ratio": h02.get("high_priority_replay_prediction_violation_ratio"),
            "mean_replay_priority_for_prediction_violating_interactions": h02.get("mean_replay_priority_for_prediction_violating_interactions"),
            "mean_replay_priority_for_non_prediction_violating_interactions": h02.get("mean_replay_priority_for_non_prediction_violating_interactions"),
            "direct_replay_lift_available": h02.get("direct_replay_lift_available"),
        },
        "H03 core metrics": {
            "transformation_family_count": h03.get("transformation_family_count"),
            "compression_ratio": h03.get("compression_ratio"),
            "compression_gain": h03.get("compression_gain"),
            "singleton_family_ratio": h03.get("singleton_family_ratio"),
            "family_cross_game_count": h03.get("family_cross_game_count"),
            "family_cross_sampler_count": h03.get("family_cross_sampler_count"),
            "family_cross_context_count": h03.get("family_cross_context_count"),
            "relaxed_singleton_family_ratio": h03.get("relaxed_singleton_family_ratio"),
            "merge_safety_passed": h03.get("merge_safety_passed"),
            "max_rows_applied": h03.get("max_rows_applied"),
            "row_count_used": h03.get("row_count_used"),
            "row_count_available": h03.get("row_count_available"),
            "family_prediction_lift_mean": h03.get("family_prediction_lift_mean"),
            "over_specific_singleton_count": h03.get("over_specific_singleton_count"),
            "over_specific_singleton_ratio": h03.get("over_specific_singleton_ratio"),
            "singleton_family_diagnostics": h03.get("singleton_family_diagnostics"),
            "compact_family_repair_used": h03.get("compact_family_repair_used"),
            "compact_family_repair_reason": h03.get("compact_family_repair_reason"),
            "compact_family_repair_family_count": h03.get("compact_family_repair_family_count"),
            "compact_family_repair_member_count": h03.get("compact_family_repair_member_count"),
        },
        "H04 core metrics": dict(h04.get("core_metrics", {})),
        "H05 core metrics": {
            "role_candidate_count": h05.get("role_candidate_count"),
            "emergent_role_count": h05.get("emergent_role_count"),
            "stable_role_count": h05.get("stable_role_count"),
            "singleton_role_ratio": h05.get("singleton_role_ratio"),
            "multi_carrier_role_count": h05.get("multi_carrier_role_count"),
            "cross_context_role_count": h05.get("cross_context_role_count"),
            "cross_game_role_count": h05.get("cross_game_role_count"),
            "mean_carriers_per_role": h05.get("mean_carriers_per_role"),
            "max_carriers_per_role": h05.get("max_carriers_per_role"),
        },
        "H06 core metrics": {
            "transfer_attempt_count": h06.get("transfer_attempt_count"),
            "successful_transfer_count": h06.get("successful_transfer_count"),
            "transfer_success_rate": h06.get("transfer_success_rate"),
            "cross_game_success_count": h06.get("cross_game_success_count"),
            "cross_context_success_count": h06.get("cross_context_success_count"),
            "successful_role_count": h06.get("successful_role_count"),
            "role_mismatch_count": h06.get("role_mismatch_count"),
            "low_similarity_count": h06.get("low_similarity_count"),
            "insufficient_source_support_count": h06.get("insufficient_source_support_count"),
            "no_source_profile_count": h06.get("no_source_profile_count"),
            "mean_transfer_score": h06.get("mean_transfer_score"),
            "max_transfer_score": h06.get("max_transfer_score"),
            "mean_best_margin": h06.get("mean_best_margin"),
            "mean_source_carrier_count": h06.get("mean_source_carrier_count"),
            "candidate_role_count_mean": h06.get("candidate_role_count_mean"),
        },
        "H07 core metrics": {
            "concept_candidate_count": h07.get("concept_candidate_count"),
            "promoted_concept_count": h07.get("promoted_concept_count"),
            "mean_compression_gain": h07.get("mean_compression_gain"),
            "max_compression_gain": h07.get("max_compression_gain"),
            "concept_transfer_success_count": h07.get("concept_transfer_success_count"),
            "concept_strong_transfer_success_count": h07.get("concept_strong_transfer_success_count"),
            "transfer_success_rate": h07.get("transfer_success_rate"),
            "max_promotion_score": h07.get("max_promotion_score"),
            "cross_context_concept_count": h07.get("cross_context_concept_count"),
            "cross_game_concept_count": h07.get("cross_game_concept_count"),
            "concept_cross_game_count_max": h07.get("concept_cross_game_count_max"),
            "concept_cross_context_count_max": h07.get("concept_cross_context_count_max"),
            "source_role_count_mean": h07.get("source_role_count_mean"),
            "source_carrier_count_mean": h07.get("source_carrier_count_mean"),
            "max_source_role_count": h07.get("max_source_role_count"),
            "max_source_family_count": h07.get("max_source_family_count"),
            "concept_transfer_success_concentration": h07.get("concept_transfer_success_concentration"),
            "overconcentrated_concept_count": h07.get("overconcentrated_concept_count"),
            "promoted_overconcentrated_concept_count": h07.get("promoted_overconcentrated_concept_count"),
        },
        "H08 core metrics": {
            "world_model_component_count": h08.get("world_model_component_count"),
            "coherent_world_model_component_count": h08.get("coherent_world_model_component_count"),
            "candidate_only_world_model_component_count": h08.get("candidate_only_world_model_component_count"),
            "promoted_concept_count": h08.get("promoted_concept_count"),
            "mean_coherence_score": h08.get("mean_coherence_score"),
            "max_coherence_score": h08.get("max_coherence_score"),
            "mean_explanatory_coverage": h08.get("mean_explanatory_coverage"),
            "max_explanatory_coverage": h08.get("max_explanatory_coverage"),
            "coherent_cross_context_component_count": h08.get("coherent_cross_context_component_count"),
            "coherent_cross_game_component_count": h08.get("coherent_cross_game_component_count"),
            "component_cross_context_count": h08.get("component_cross_context_count"),
            "component_cross_game_count": h08.get("component_cross_game_count"),
            "predicted_outcome_count": h08.get("predicted_outcome_count"),
            "predicted_outcome_count_is_proxy_count": h08.get("predicted_outcome_count_is_proxy_count"),
            "supported_context_count": h08.get("supported_context_count"),
            "concept_link_count": h08.get("concept_link_count"),
            "role_link_count": h08.get("role_link_count"),
            "family_link_count": h08.get("family_link_count"),
            "contradiction_coverage_count": h08.get("contradiction_coverage_count"),
        },
        "H09 core metrics": {
            "future_option_event_count": h09.get("future_option_event_count"),
            "future_option_motif_count": h09.get("future_option_motif_count"),
            "emergent_future_option_motif_count": h09.get("emergent_future_option_motif_count"),
            "motif_type_counts": h09.get("motif_type_counts"),
            "motif_type_source_counts": h09.get("motif_type_source_counts"),
            "cross_context_motif_count": h09.get("cross_context_motif_count"),
            "cross_game_motif_count": h09.get("cross_game_motif_count"),
            "mean_abs_option_delta": h09.get("mean_abs_option_delta"),
            "max_abs_option_delta": h09.get("max_abs_option_delta"),
            "mean_motif_stability_score": h09.get("mean_motif_stability_score"),
            "unknown_motif_count": h09.get("unknown_motif_count"),
            "unknown_motif_ratio": h09.get("unknown_motif_ratio"),
            "unknown_motif_event_count": h09.get("unknown_motif_event_count"),
            "unknown_motif_event_ratio": h09.get("unknown_motif_event_ratio"),
            "unknown_motif_source_count": h09.get("unknown_motif_source_count"),
            "unknown_motif_source_ratio": h09.get("unknown_motif_source_ratio"),
            "live_delta_event_count": h09.get("live_delta_event_count"),
            "structured_effect_event_count": h09.get("structured_effect_event_count"),
            "text_keyword_event_count": h09.get("text_keyword_event_count"),
            "future_option_edge_event_count": h09.get("future_option_edge_event_count"),
        },
        "H10 core metrics": {
            "future_option_attention_link_count": h10.get("future_option_attention_link_count"),
            "h10_attention_target_definition": h10.get("h10_attention_target_definition"),
            "live_future_option_delta_count": h10.get("live_future_option_delta_count"),
            "heuristic_future_option_delta_count": h10.get("heuristic_future_option_delta_count"),
            "null_future_option_delta_count": h10.get("null_future_option_delta_count"),
            "high_option_change_count": h10.get("high_option_change_count"),
            "high_option_change_source": h10.get("high_option_change_source"),
            "high_attention_count": h10.get("high_attention_count"),
            "high_option_change_attention_count": h10.get("high_option_change_attention_count"),
            "low_option_change_attention_count": h10.get("low_option_change_attention_count"),
            "high_option_change_attention_rate": h10.get("high_option_change_attention_rate"),
            "low_option_change_attention_rate": h10.get("low_option_change_attention_rate"),
            "option_attention_lift": h10.get("option_attention_lift"),
            "option_attention_lift_unbounded": h10.get("option_attention_lift_unbounded"),
            "replay_attention_count": h10.get("replay_attention_count"),
            "contradiction_attention_count": h10.get("contradiction_attention_count"),
            "replay_or_contradiction_attention_count": h10.get("replay_or_contradiction_attention_count"),
            "attention_all_high_saturation": h10.get("attention_all_high_saturation"),
            "attention_all_low_saturation": h10.get("attention_all_low_saturation"),
            "attention_saturation": h10.get("attention_saturation"),
            "replay_attention_saturation": h10.get("replay_attention_saturation"),
            "contradiction_attention_saturation": h10.get("contradiction_attention_saturation"),
        },
        "H11 core metrics": {
            "future_option_transfer_link_count": h11.get("future_option_transfer_link_count"),
            "motifs_with_transfer_count": h11.get("motifs_with_transfer_count"),
            "motifs_with_strong_transfer_count": h11.get("motifs_with_strong_transfer_count"),
            "motifs_with_promoted_concept_count": h11.get("motifs_with_promoted_concept_count"),
            "motif_transfer_success_rate": h11.get("motif_transfer_success_rate"),
            "motif_strong_transfer_success_rate": h11.get("motif_strong_transfer_success_rate"),
            "promoted_concept_motif_count": h11.get("promoted_concept_motif_count"),
            "emergent_future_option_motif_count": h11.get("emergent_future_option_motif_count"),
            "emergent_motif_transfer_link_count": h11.get("emergent_motif_transfer_link_count"),
            "emergent_motifs_with_transfer_count": h11.get("emergent_motifs_with_transfer_count"),
            "emergent_motifs_with_strong_transfer_count": h11.get("emergent_motifs_with_strong_transfer_count"),
            "emergent_motifs_with_promoted_concept_count": h11.get("emergent_motifs_with_promoted_concept_count"),
            "emergent_motif_transfer_success_rate": h11.get("emergent_motif_transfer_success_rate"),
            "emergent_motif_strong_transfer_success_rate": h11.get("emergent_motif_strong_transfer_success_rate"),
            "promoted_concept_emergent_motif_count": h11.get("promoted_concept_emergent_motif_count"),
            "non_emergent_motif_transfer_link_count": h11.get("non_emergent_motif_transfer_link_count"),
            "non_emergent_motifs_with_strong_transfer_count": h11.get("non_emergent_motifs_with_strong_transfer_count"),
            "non_emergent_motifs_with_promoted_concept_count": h11.get("non_emergent_motifs_with_promoted_concept_count"),
            "successful_role_transfer_count": h11.get("successful_role_transfer_count"),
            "promoted_concept_count": h11.get("promoted_concept_count"),
        },
        "H12 core metrics": {
            "successful_trajectories": h12.get("successful_trajectories"),
            "comparable_trajectory_groups": h12.get("comparable_trajectory_groups"),
            "efficiency_active_trajectory_count": h12.get("efficiency_active_trajectory_count"),
            "trajectories_with_memory_bonus": h12.get("trajectories_with_memory_bonus"),
            "trajectories_with_replay_bonus": h12.get("trajectories_with_replay_bonus"),
            "mean_efficiency_memory_bonus": h12.get("mean_efficiency_memory_bonus"),
            "mean_efficiency_replay_bonus": h12.get("mean_efficiency_replay_bonus"),
            "best_known_solution_improvements": h12.get("best_known_solution_improvements"),
            "median_steps_to_success": h12.get("median_steps_to_success"),
            "mean_normalized_solve_efficiency": h12.get("mean_normalized_solve_efficiency"),
            "mean_loop_ratio": h12.get("mean_loop_ratio"),
            "mean_repeated_state_ratio": h12.get("mean_repeated_state_ratio"),
            "mean_blocked_action_ratio": h12.get("mean_blocked_action_ratio"),
        },
        "per_game_status_table": per_game,
        "per-game status table": per_game,
        "per_sampler_status_table": per_sampler,
        "per-sampler status table": per_sampler,
        "temporal_order_diagnostics": temporal,
        "missing_evidence": missing_evidence,
        "missing evidence": missing_evidence,
        "higher_order_dependency_gates_applied": True,
        "higher_order_dependency_gate_notes": list(higher_order_dependency_gate_notes or []),
        "epoch_maturity_gate_notes": list(epoch_maturity_gate_notes or []),
        "next_recommended_action": _next_recommended_action(h01, h02, h03, h04, h05, h06, h07, h08, h09, h10, h11, missing_evidence),
        "next recommended action": _next_recommended_action(h01, h02, h03, h04, h05, h06, h07, h08, h09, h10, h11, missing_evidence),
    }
    return summary


def _per_game_diagnostics(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_game: dict[str, list[dict[str, Any]]] = {}
    for row in runs:
        game = str(row.get("game") or "")
        if not game:
            continue
        by_game.setdefault(game, []).append(row)
    output: list[dict[str, Any]] = []
    for game, items in sorted(by_game.items()):
        interaction_count = int(sum(int(row.get("total_interactions", 0) or 0) for row in items))
        stable_contingency_count = int(sum(int(row.get("stable_contingency_count", 0) or 0) for row in items))
        mean_prediction_accuracy = _mean([row.get("prediction_accuracy") for row in items])
        transformation_family_count = int(sum(int(row.get("unique_transformation_families", 0) or 0) for row in items))
        prediction_violation_replay_lift = None
        compression_ratio = None
        singleton_family_ratio = None
        h01_signal = stable_contingency_count > 0
        h02_signal = any(
            (row.get("mean_isf_prediction_error") or 0.0) > 0.0 and int(row.get("high_priority_replay_count", 0) or 0) > 0
            for row in items
        )
        h03_signal = transformation_family_count > 0
        if interaction_count <= 0:
            status = "missing"
        elif h01_signal and h02_signal and h03_signal:
            status = "supported"
        elif h01_signal:
            status = "partial"
        elif interaction_count > 0 and stable_contingency_count <= 1:
            status = "weak"
        else:
            status = "failed"
        output.append(
            {
                "game": game,
                "interaction_count": interaction_count,
                "stable_contingency_count": stable_contingency_count,
                "mean_prediction_accuracy": mean_prediction_accuracy,
                "prediction_violation_replay_lift": prediction_violation_replay_lift,
                "transformation_family_count": transformation_family_count,
                "compression_ratio": compression_ratio,
                "singleton_family_ratio": singleton_family_ratio,
                "status": status,
            }
        )
    return output


def _per_sampler_diagnostics(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_sampler: dict[str, list[dict[str, Any]]] = {}
    for row in runs:
        sampler = str(row.get("sampler_name") or "")
        if not sampler:
            continue
        by_sampler.setdefault(sampler, []).append(row)
    output: list[dict[str, Any]] = []
    for sampler, items in sorted(by_sampler.items()):
        stable_contingency_count = int(sum(int(row.get("stable_contingency_count", 0) or 0) for row in items))
        transformation_family_count = int(sum(int(row.get("unique_transformation_families", 0) or 0) for row in items))
        replay_pressure = int(sum(int(row.get("high_priority_replay_count", 0) or 0) for row in items))
        if stable_contingency_count > 0 and transformation_family_count > 0 and replay_pressure > 0:
            status = "supported"
        elif stable_contingency_count > 0:
            status = "partial"
        elif items:
            status = "weak"
        else:
            status = "missing"
        output.append(
            {
                "sampler": sampler,
                "interaction_count": int(sum(int(row.get("total_interactions", 0) or 0) for row in items)),
                "stable_contingency_count": stable_contingency_count,
                "transformation_family_count": transformation_family_count,
                "high_priority_replay_count": replay_pressure,
                "status": status,
            }
        )
    return output


def _temporal_order_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    h01_before_h03_values: list[bool] = []
    h02_before_h03_values: list[bool] = []
    h03_before_h04_values: list[bool] = []
    missing_count = 0
    cases_available = 0
    diagnostics_rows: list[dict[str, Any]] = []
    for row in rows:
        h01_before_h03 = _ordered_bool(row.get("first_stable_contingency_step"), row.get("first_transformation_family_step"))
        h02_before_h03 = _ordered_bool(row.get("first_prediction_violation_step"), row.get("first_transformation_family_step"))
        h03_before_h04 = _ordered_bool(row.get("first_stable_transformation_family_step"), row.get("first_emergent_carrier_step"))
        diagnostics_rows.append(
            {
                "game": row.get("game"),
                "sampler": row.get("sampler"),
                "seed": row.get("seed"),
                "h01_before_h03": h01_before_h03,
                "h02_before_h03": h02_before_h03,
                "h03_before_h04": h03_before_h04,
            }
        )
        local_values = [h01_before_h03, h02_before_h03, h03_before_h04]
        if any(value is not None for value in local_values):
            cases_available += 1
        missing_count += sum(1 for value in local_values if value is None)
        if h01_before_h03 is not None:
            h01_before_h03_values.append(h01_before_h03)
        if h02_before_h03 is not None:
            h02_before_h03_values.append(h02_before_h03)
        if h03_before_h04 is not None:
            h03_before_h04_values.append(h03_before_h04)
    return {
        "per_case": diagnostics_rows,
        "temporal_order_cases_available": cases_available,
        "h01_before_h03_ratio": _true_ratio(h01_before_h03_values),
        "h02_before_h03_ratio": _true_ratio(h02_before_h03_values),
        "h03_before_h04_ratio": _true_ratio(h03_before_h04_values),
        "temporal_order_missing_count": missing_count,
    }


def _ordered_bool(left: Any, right: Any) -> bool | None:
    if left is None or right is None:
        return None
    return int(left) <= int(right)


def _true_ratio(values: list[bool]) -> float | None:
    if not values:
        return None
    return float(sum(1 for value in values if value) / len(values))


def _resolve_total_interactions(
    *,
    runs: list[dict[str, Any]],
    h01: dict[str, Any],
    interactions_this_epoch: int | None,
    total_interactions_seen: int | None,
) -> dict[str, Any]:
    total_interactions = int(sum(int(row.get("total_interactions", 0) or 0) for row in runs))
    raw_total_interactions = total_interactions
    source = "raw_report_runs" if raw_total_interactions > 0 else "unavailable"
    if total_interactions <= 0 and interactions_this_epoch is not None:
        total_interactions = int(interactions_this_epoch or 0)
        if total_interactions > 0:
            source = "continuous_epoch_argument"
    if total_interactions <= 0:
        h01_interactions = h01.get("total_interaction_count")
        if h01_interactions is not None:
            total_interactions = int(h01_interactions or 0)
            if total_interactions > 0:
                source = "h01_total_interaction_count"
    if total_interactions <= 0 and total_interactions_seen is not None:
        total_interactions = int(total_interactions_seen or 0)
        if total_interactions > 0:
            source = "compact_memory_total_interactions_seen"
    return {
        "total_interactions": int(total_interactions),
        "raw_report_total_interactions": int(raw_total_interactions),
        "total_interactions_source": source if total_interactions > 0 or raw_total_interactions > 0 else "unavailable",
    }


def _apply_epoch_maturity_gates(
    *,
    h04: dict[str, Any],
    h05: dict[str, Any],
    h06: dict[str, Any],
    h07: dict[str, Any],
    h08: dict[str, Any],
    h09: dict[str, Any],
    h10: dict[str, Any],
    h11: dict[str, Any],
    total_interactions: int,
    interactions_this_epoch: int | None,
    game_count: int,
    sampler_count: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], list[str]]:
    maturity_threshold = max(1000, int(game_count) * int(sampler_count))
    notes: list[str] = []
    demote = total_interactions <= 0
    if demote:
        notes.append("Full H04-H11 validation blocked because total interaction count is unavailable.")
    elif interactions_this_epoch is not None and int(interactions_this_epoch) < maturity_threshold:
        demote = True
        notes.append("Full H04-H11 validation blocked because epoch interaction budget is below maturity threshold.")

    def _demote(payload: dict[str, Any]) -> dict[str, Any]:
        updated = dict(payload)
        if str(updated.get("decision")) == "VALID" and demote:
            original = str(updated.get("decision"))
            updated["original_decision_before_epoch_maturity_gate"] = "VALID"
            updated.setdefault("individual_decision_before_suite_gates", original)
            updated["epoch_maturity_demoted"] = True
            updated["epoch_maturity_threshold"] = maturity_threshold
            updated["epoch_interactions_used_for_gate"] = interactions_this_epoch if interactions_this_epoch is not None else total_interactions
            updated["decision"] = "PARTIALLY_VALID"
            updated["suite_gated_decision"] = updated["decision"]
            missing = list(updated.get("missing_evidence", []))
            gate_reasons = list(updated.get("suite_gate_reasons", []))
            for note in notes:
                if note not in missing:
                    missing.append(note)
                if note not in gate_reasons:
                    gate_reasons.append(note)
            updated["missing_evidence"] = missing
            updated["suite_gate_reasons"] = gate_reasons
        return updated

    return _demote(h04), _demote(h05), _demote(h06), _demote(h07), _demote(h08), _demote(h09), _demote(h10), _demote(h11), notes


def _apply_higher_order_dependency_gates(
    h04: dict[str, Any],
    h05: dict[str, Any],
    h06: dict[str, Any],
    h07: dict[str, Any],
    h08: dict[str, Any],
    h09: dict[str, Any],
    h10: dict[str, Any],
    h11: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], list[str]]:
    notes: list[str] = []

    def _demote(payload: dict[str, Any], message: str) -> dict[str, Any]:
        updated = dict(payload)
        if str(updated.get("decision")) == "VALID":
            original = str(updated.get("decision"))
            updated["original_decision_before_dependency_gate"] = "VALID"
            updated.setdefault("individual_decision_before_suite_gates", original)
            updated["dependency_demoted"] = True
            updated["decision"] = "PARTIALLY_VALID"
            updated["suite_gated_decision"] = updated["decision"]
            missing = list(updated.get("missing_evidence", []))
            gate_reasons = list(updated.get("suite_gate_reasons", []))
            if message not in missing:
                missing.append(message)
            if message not in gate_reasons:
                gate_reasons.append(message)
            updated["missing_evidence"] = missing
            updated["suite_gate_reasons"] = gate_reasons
            notes.append(message)
        return updated

    if str(h04.get("decision")) == "INVALID" and str(h05.get("decision")) == "VALID":
        h05 = _demote(h05, "H05 depends on invalid H04 carrier emergence.")
        h05["h05_depends_on_invalid_h04"] = True
    if str(h05.get("decision")) == "VALID" and (
        str(h04.get("decision")) != "VALID"
        or h04.get("h04_graph_quality_pass") is not True
        or int(h04.get("usable_emergent_carrier_count") or 0) <= 0
    ):
        h05 = _demote(
            h05,
            "H05 cannot be fully VALID until H04 has VALID usable carrier emergence with graph-quality pass.",
        )
    if str(h05.get("decision")) != "VALID" and str(h06.get("decision")) == "VALID":
        h06 = _demote(h06, "H06 cannot be fully VALID until H05 role emergence is VALID.")
    if str(h06.get("decision")) != "VALID" and str(h07.get("decision")) == "VALID":
        h07 = _demote(h07, "H07 cannot be fully VALID until H06 role transfer is VALID.")
    if str(h07.get("decision")) != "VALID" and str(h08.get("decision")) == "VALID":
        h08 = _demote(h08, "H08 cannot be fully VALID until H07 concept emergence is VALID.")
    if str(h06.get("decision")) != "VALID" and str(h08.get("decision")) == "VALID":
        h08 = _demote(h08, "H08 cannot be fully VALID until H06 role transfer is VALID.")
    if str(h09.get("decision")) != "VALID" and str(h10.get("decision")) == "VALID":
        h10 = _demote(h10, "H10 cannot be fully VALID until H09 future-option motifs are VALID.")
    if str(h09.get("decision")) != "VALID" and str(h11.get("decision")) == "VALID":
        h11 = _demote(h11, "H11 cannot be fully VALID until H09 future-option motifs are VALID.")
    if str(h06.get("decision")) in {"INCONCLUSIVE", "INVALID"} and str(h11.get("decision")) == "VALID":
        h11 = _demote(h11, "H11 cannot be fully VALID until H06 role transfer is at least PARTIALLY_VALID.")
    return h05, h06, h07, h08, h09, h10, h11, notes


def _next_recommended_action(
    h01: dict[str, Any],
    h02: dict[str, Any],
    h03: dict[str, Any],
    h04: dict[str, Any],
    h05: dict[str, Any],
    h06: dict[str, Any],
    h07: dict[str, Any],
    h08: dict[str, Any],
    h09: dict[str, Any],
    h10: dict[str, Any],
    h11: dict[str, Any],
    missing_evidence: list[str],
) -> str:
    if str(h02.get("decision")) == "INVALID":
        return "Inspect H02 replay/contradiction evidence and raw-to-compact fallback before interpreting higher-order results."
    if any("maturity" in str(item).lower() or "cannot be fully valid until" in str(item).lower() for item in missing_evidence):
        return "Continue more epochs before treating H04-H11 as full validations; current higher-order results are maturity-gated."
    decisions = {
        str(h01.get("decision")),
        str(h02.get("decision")),
        str(h03.get("decision")),
        str(h04.get("decision")),
        str(h05.get("decision")),
        str(h06.get("decision")),
        str(h07.get("decision")),
        str(h08.get("decision")),
        str(h09.get("decision")),
        str(h10.get("decision")),
        str(h11.get("decision")),
    }
    if "INVALID" in decisions:
        return "Inspect invalidated hypothesis outputs and repair the shared compact-memory substrate before continuing higher-order evaluation."
    if "INCONCLUSIVE" in decisions or missing_evidence:
        return "Keep the shared run, fill the missing evidence paths, and rerun the suite summary before relying on higher-order conclusions."
    if "PARTIALLY_VALID" in decisions:
        return "Use this shared dataset for targeted follow-up diagnostics, then rerun H05-H11 on the same compact memory."
    return "Proceed with the existing compact memory and compare H01-H11 trends across subsequent epochs."


def _skipped_fast_mode_result(hypothesis_id: str) -> dict[str, Any]:
    return {
        "hypothesis_id": hypothesis_id,
        "decision": "SKIPPED_FAST_MODE",
        "evidence_stage": "not_evaluated_this_epoch",
        "missing_evidence": ["Run with --hypothesis-suite-mode full to evaluate this hypothesis."],
        "core_metrics": {},
        "evidence_source": "cached_derived_memory",
    }


def _collect_hypothesis_consistency_warnings(
    *,
    h03: dict[str, Any],
    h04: dict[str, Any],
    h05: dict[str, Any],
    h07: dict[str, Any],
    h08: dict[str, Any],
    h09: dict[str, Any],
    h10: dict[str, Any],
    h12: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    if str(h03.get("decision")) == "VALID" and h03.get("family_prediction_lift_mean") is None:
        warnings.append("H03 inconsistency: VALID reported without family prediction-lift evidence.")
    if str(h04.get("decision")) == "VALID" and h04.get("h03_before_h04") is None:
        warnings.append("H04 inconsistency: VALID reported without H03-before-H04 temporal evidence.")
    if str(h05.get("decision")) == "VALID" and h05.get("h04_before_h05") is None:
        warnings.append("H05 inconsistency: VALID reported without H04-before-H05 temporal evidence.")
    if str(h07.get("decision")) == "PARTIALLY_VALID" and int(h07.get("concept_candidate_count") or 0) <= 1 and int(h07.get("promoted_concept_count") or 0) == 0:
        warnings.append("H07 inconsistency: PARTIALLY_VALID reported from precursor-only concept evidence.")
    if str(h08.get("decision")) == "PARTIALLY_VALID" and int(h08.get("promoted_concept_count") or 0) == 0 and int(h08.get("coherent_world_model_component_count") or 0) == 0:
        warnings.append("H08 inconsistency: PARTIALLY_VALID reported from candidate-only proxy world-model evidence.")
    if int(h09.get("future_option_event_count") or 0) == 0 and (
        int(h09.get("stable_contingencies_count") or 0) > 0 or int(h09.get("transformation_families_count") or 0) > 0
    ):
        warnings.append("H09 derivation warning: zero future-option events despite available stable substrate.")
    if int(h10.get("future_option_event_count") or 0) == 0 and not bool(h10.get("h10_blocked_by_h09")):
        warnings.append("H10 inconsistency: missing blocked_by_h09 despite zero future-option events.")
    if not h12.get("trajectory_reconstruction_diagnostics") and bool(h12.get("blocked_by_missing_trajectory_evidence")):
        warnings.append("H12 inconsistency: blocked by missing trajectory evidence without diagnostics.")
    return warnings


def _blocker_flags_for_result(hypothesis_id: str, result: dict[str, Any]) -> dict[str, bool]:
    decision = str(result.get("decision") or "")
    missing_evidence = list(result.get("missing_evidence") or [])
    core_metrics = dict(result.get("core_metrics") or {})
    h03_before_h04 = result.get("h03_before_h04", core_metrics.get("h03_before_h04"))
    h04_before_h05 = result.get("h04_before_h05", core_metrics.get("h04_before_h05"))
    if decision == "SKIPPED_FAST_MODE":
        return {
            "h02_missing_direct_linkage": False,
            "h03_missing_prediction_lift": False,
            "h04_missing_temporal_order": False,
            "h04_temporal_order_failed": False,
            "h05_missing_temporal_order": False,
            "h06_transfer_sampling_capped": False,
            "h07_no_promoted_concepts": False,
            "h08_proxy_only_world_model": False,
            "h09_no_future_option_events": False,
            "h10_blocked_by_h09": False,
            "h11_blocked_by_no_motifs": False,
            "h12_missing_trajectory_evidence": False,
            "skipped_fast_mode": True,
        }
    return {
        "h02_missing_direct_linkage": hypothesis_id == "H02" and (
            result.get("direct_replay_lift_available") is not True
            or result.get("raw_cleanup_prevents_direct_linkage") is True
            or any("linkage unavailable" in str(msg).lower() for msg in missing_evidence)
        ),
        "h03_missing_prediction_lift": hypothesis_id == "H03" and result.get("family_prediction_lift_mean") is None,
        "h04_missing_temporal_order": hypothesis_id == "H04" and h03_before_h04 is None,
        "h04_temporal_order_failed": hypothesis_id == "H04" and h03_before_h04 is False,
        "h05_missing_temporal_order": hypothesis_id == "H05" and h04_before_h05 is None,
        "h06_transfer_sampling_capped": hypothesis_id == "H06" and int(result.get("skipped_by_cap_count") or 0) > 0,
        "h07_no_promoted_concepts": hypothesis_id == "H07" and int(result.get("promoted_concept_count") or 0) == 0,
        "h08_proxy_only_world_model": hypothesis_id == "H08" and bool(result.get("candidate_proxy_only")),
        "h09_no_future_option_events": hypothesis_id == "H09" and int(result.get("future_option_event_count") or 0) == 0,
        "h10_blocked_by_h09": hypothesis_id == "H10" and bool(result.get("h10_blocked_by_h09")),
        "h11_blocked_by_no_motifs": hypothesis_id == "H11" and bool(result.get("h11_blocked_by_no_motifs")),
        "h12_missing_trajectory_evidence": hypothesis_id == "H12" and bool(result.get("blocked_by_missing_trajectory_evidence")),
        "skipped_fast_mode": False,
    }


def _write_suite_summary(
    summary: dict[str, Any],
    output_dir: Path,
    *,
    hypothesis_results: dict[str, dict[str, Any]] | None = None,
) -> None:
    (output_dir / SUITE_JSON_NAME).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / SUITE_TXT_NAME).write_text(_format_text(summary), encoding="utf-8")
    (output_dir / SUITE_MD_NAME).write_text(_format_md(summary), encoding="utf-8")
    (output_dir / "hypothesis_consistency_warnings.json").write_text(
        json.dumps(summary.get("hypothesis_consistency_warnings", []), indent=2),
        encoding="utf-8",
    )
    _write_aggregated_hypothesis_text(output_dir, hypothesis_results=hypothesis_results)


def _format_aggregated_result_section(hypothesis_id: str, result: dict[str, Any]) -> str:
    lines = [
        f"{hypothesis_id}",
        f"decision: {result.get('decision')}",
        f"evidence_stage: {result.get('evidence_stage')}",
        f"blocker_flags: {json.dumps(_blocker_flags_for_result(hypothesis_id, result), sort_keys=True)}",
        f"phase_seconds: {json.dumps(result.get('phase_seconds') or {}, sort_keys=True)}",
        f"core_metrics: {json.dumps(result.get('core_metrics') or {}, sort_keys=True)}",
        f"missing_evidence: {json.dumps(result.get('missing_evidence') or [], ensure_ascii=True)}",
        f"evidence_diagnostics: {json.dumps(result.get('evidence_diagnostics') or {}, sort_keys=True)}",
    ]
    return "\n".join(lines).strip()


def _write_aggregated_hypothesis_text(
    output_dir: Path,
    *,
    hypothesis_results: dict[str, dict[str, Any]] | None = None,
) -> None:
    sections: list[str] = []
    placeholders: dict[int, str] = {}
    for hypothesis_id in range(1, 13):
        hypothesis_key = f"H{hypothesis_id:02d}"
        result_payload = None if hypothesis_results is None else hypothesis_results.get(hypothesis_key)
        if isinstance(result_payload, dict):
            section = _format_aggregated_result_section(hypothesis_key, result_payload)
            sections.append(section)
            subdir = output_dir / f"h{hypothesis_id:02d}"
            subdir.mkdir(parents=True, exist_ok=True)
            target = subdir / f"h{hypothesis_id:02d}_report.txt"
            if not target.exists():
                target.write_text(section + "\n", encoding="utf-8")
            continue
        label = f"h{hypothesis_id:02d}"
        subdir = output_dir / label
        if not subdir.exists():
            placeholders[hypothesis_id] = f"H{hypothesis_id:02d} report unavailable"
            continue
        txt_files = sorted(subdir.glob("*.txt"))
        if not txt_files:
            placeholders[hypothesis_id] = f"H{hypothesis_id:02d} report unavailable"
            continue
        text = txt_files[0].read_text(encoding="utf-8").strip()
        if not text:
            placeholders[hypothesis_id] = f"H{hypothesis_id:02d} report unavailable"
            continue
        sections.append(text)
    if not sections:
        summary_text = output_dir / SUITE_TXT_NAME
        if summary_text.exists():
            fallback = summary_text.read_text(encoding="utf-8").strip()
            if fallback:
                sections.append(fallback)
    if not sections:
        sections = [f"H{hypothesis_id:02d} report unavailable" for hypothesis_id in range(1, 12)]
    for hypothesis_id, text in placeholders.items():
        if hypothesis_id > 11:
            continue
        subdir = output_dir / f"h{hypothesis_id:02d}"
        subdir.mkdir(parents=True, exist_ok=True)
        target = subdir / f"h{hypothesis_id:02d}_report.txt"
        if not target.exists():
            target.write_text(text + "\n", encoding="utf-8")
    aggregated = "\n\n".join(sections).strip()
    if aggregated:
        aggregated += "\n"
    (output_dir / SUITE_AGGREGATED_TXT_NAME).write_text(aggregated, encoding="utf-8")


def _format_text(summary: dict[str, Any]) -> str:
    gate_lines: list[str] = []
    for hypothesis_id in range(4, 12):
        key = f"H{hypothesis_id:02d} suite gating"
        payload = summary.get(key) or {}
        if not payload or payload.get("individual_decision_before_suite_gates") in (None, payload.get("decision")):
            continue
        gate_lines.extend(
            [
                f"H{hypothesis_id:02d} individual: {payload.get('individual_decision_before_suite_gates')}",
                f"H{hypothesis_id:02d} suite-gated: {payload.get('suite_gated_decision') or payload.get('decision')}",
                f"H{hypothesis_id:02d} suite-gate reasons: {', '.join(payload.get('suite_gate_reasons') or [])}",
            ]
        )
    lines = [
        "Hypothesis Suite Summary",
        f"source_run_dir: {summary['source_run_dir']}",
        f"H01: {summary['H01 decision']}",
        f"H02: {summary['H02 decision']}",
        f"H03: {summary['H03 decision']}",
        f"H04: {summary['H04 decision']}",
        f"H05: {summary['H05 decision']}",
        f"H06: {summary['H06 decision']}",
        f"H07: {summary['H07 decision']}",
        f"H08 (world-model coherence): {summary['H08 decision']}",
        f"H09: {summary['H09 decision']}",
        f"H10: {summary['H10 decision']}",
        f"H11: {summary['H11 decision']}",
        f"H12: {summary.get('H12 decision', 'INCONCLUSIVE')}",
        f"games: {summary['game_count']} samplers: {summary['sampler_count']} seeds: {summary['seed_count']}",
        f"total_interactions: {summary['total_interactions']}",
        "Epoch completion:",
        f"- Levels={summary.get('Levels', 0)} Games={summary.get('Games', 0)} Total_Levels={summary.get('Total_Levels', 0)} Total_Games={summary.get('Total_Games', 0)}",
        f"- solved games: {','.join(summary.get('solved_games', []) or [])}",
        *gate_lines,
        f"next_recommended_action: {summary['next_recommended_action']}",
    ]
    return "\n".join(lines) + "\n"


def _format_md(summary: dict[str, Any]) -> str:
    gate_lines: list[str] = []
    for hypothesis_id in range(4, 12):
        key = f"H{hypothesis_id:02d} suite gating"
        payload = summary.get(key) or {}
        if not payload or payload.get("individual_decision_before_suite_gates") in (None, payload.get("decision")):
            continue
        gate_lines.extend(
            [
                f"- H{hypothesis_id:02d} individual: `{payload.get('individual_decision_before_suite_gates')}`",
                f"- H{hypothesis_id:02d} suite-gated: `{payload.get('suite_gated_decision') or payload.get('decision')}`",
                f"- H{hypothesis_id:02d} reasons: `{', '.join(payload.get('suite_gate_reasons') or [])}`",
            ]
        )
    lines = [
        "# Hypothesis Suite Summary",
        "",
        f"- source run: `{summary['source_run_dir']}`",
        f"- H01: `{summary['H01 decision']}`",
        f"- H02: `{summary['H02 decision']}`",
        f"- H03: `{summary['H03 decision']}`",
        f"- H04: `{summary['H04 decision']}`",
        f"- H05: `{summary['H05 decision']}`",
        f"- H06: `{summary['H06 decision']}`",
        f"- H07: `{summary['H07 decision']}`",
        f"- H08 (world-model coherence): `{summary['H08 decision']}`",
        f"- H09: `{summary['H09 decision']}`",
        f"- H10: `{summary['H10 decision']}`",
        f"- H11: `{summary['H11 decision']}`",
        f"- H12: `{summary.get('H12 decision', 'INCONCLUSIVE')}`",
        f"- total interactions: `{summary['total_interactions']}`",
        f"- Levels=`{summary.get('Levels', 0)}` Games=`{summary.get('Games', 0)}` Total_Levels=`{summary.get('Total_Levels', 0)}` Total_Games=`{summary.get('Total_Games', 0)}`",
        f"- solved games: `{', '.join(summary.get('solved_games', []) or [])}`",
        *gate_lines,
        "",
        "## Next Action",
        "",
        summary["next_recommended_action"],
    ]
    return "\n".join(lines) + "\n"


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: list[Any]) -> float | None:
    items = [float(value) for value in values if value is not None]
    if not items:
        return None
    return float(sum(items) / len(items))


def _count_positive(mapping: Any) -> int | None:
    if not isinstance(mapping, dict):
        return None
    return sum(1 for value in mapping.values() if int(value or 0) > 0)


def _merge_unique(*groups: list[Any]) -> list[Any]:
    output: list[Any] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            key = str(item)
            if key in seen:
                continue
            seen.add(key)
            output.append(item)
    return output


def _suite_gate_status(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision": payload.get("decision"),
        "individual_decision_before_suite_gates": payload.get("individual_decision_before_suite_gates"),
        "suite_gated_decision": payload.get("suite_gated_decision", payload.get("decision")),
        "suite_gate_reasons": list(payload.get("suite_gate_reasons", []) or []),
        "epoch_maturity_demoted": bool(payload.get("epoch_maturity_demoted", False)),
        "dependency_demoted": bool(payload.get("dependency_demoted", False)),
    }
