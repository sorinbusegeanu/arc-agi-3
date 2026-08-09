from __future__ import annotations

import copy
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


_INSTALLED = False
_ORIGINAL_VALIDATE_ONLY: Any = None
_ORIGINAL_BUILD_FUNCTIONAL: Any = None
_ACTIVE_FUNCTIONAL_CACHE: dict[tuple[str, int | None, tuple[str, ...]], tuple[Any, Any, Any]] | None = None


def install_v63_validation_parallel_completion() -> None:
    """Use validation_workers for read-only per-concept held-out diagnostics."""
    global _INSTALLED
    global _ORIGINAL_VALIDATE_ONLY
    global _ORIGINAL_BUILD_FUNCTIONAL

    from v6 import higher_order_substrate as substrate
    from v6 import hypothesis_suite_report as suite

    if not _INSTALLED:
        _ORIGINAL_VALIDATE_ONLY = substrate.validate_incremental_promotions_only
        _ORIGINAL_BUILD_FUNCTIONAL = substrate._build_functional_explanation_diagnostics
        _INSTALLED = True

    # Reapply on every migration entry point.  The suite imports the validator
    # by value, so both bindings must be refreshed.
    substrate.validate_incremental_promotions_only = _validate_incremental_promotions_only_parallel
    substrate._build_functional_explanation_diagnostics = _build_functional_from_cache
    suite.validate_incremental_promotions_only = _validate_incremental_promotions_only_parallel


def _resolve_validation_workers(memory_dir: str | Path, requested: int | None = None) -> int:
    if requested is not None:
        return min(16, max(1, int(requested)))
    root = Path(memory_dir).parent
    epochs_dir = root / "epochs"
    if not epochs_dir.exists():
        return 1
    reports = sorted(
        epochs_dir.glob("epoch_*/raw/interaction_sampling_v05c_report.json"),
        reverse=True,
    )
    for report_path in reports:
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            value = int(payload.get("validation_workers_requested") or 0)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if value > 0:
            return min(16, max(1, value))
    return 1


def _validate_incremental_promotions_only_parallel(
    *,
    memory_dir: Path,
    config: Any,
    validate_roles_and_concepts: bool,
    validate_world_models: bool,
    diagnostic_epoch_id: str | int | None = None,
    explanation_events_path: Path | None = None,
    validation_state_reset_applied_this_run: bool | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    requested_workers = _resolve_validation_workers(memory_dir, workers)
    effective_workers = requested_workers
    cache: dict[tuple[str, int | None, tuple[str, ...]], tuple[Any, Any, Any]] = {}

    if (
        bool(getattr(config, "enabled", False))
        and bool(validate_roles_and_concepts)
        and requested_workers > 1
    ):
        cache = _precompute_functional_diagnostics(
            memory_dir=Path(memory_dir),
            config=config,
            diagnostic_epoch_id=diagnostic_epoch_id,
            workers=requested_workers,
        )
        effective_workers = min(requested_workers, max(1, len(cache)))
    else:
        effective_workers = 1

    global _ACTIVE_FUNCTIONAL_CACHE
    previous_cache = _ACTIVE_FUNCTIONAL_CACHE
    _ACTIVE_FUNCTIONAL_CACHE = cache if cache else None
    try:
        result = _ORIGINAL_VALIDATE_ONLY(
            memory_dir=memory_dir,
            config=config,
            validate_roles_and_concepts=validate_roles_and_concepts,
            validate_world_models=validate_world_models,
            diagnostic_epoch_id=diagnostic_epoch_id,
            explanation_events_path=explanation_events_path,
            validation_state_reset_applied_this_run=validation_state_reset_applied_this_run,
        )
    finally:
        _ACTIVE_FUNCTIONAL_CACHE = previous_cache

    summary = dict(result)
    summary["validation_workers_requested"] = int(requested_workers)
    summary["validation_workers_effective"] = int(effective_workers)
    summary["concept_functional_diagnostics_precomputed"] = int(len(cache))
    return summary


def _precompute_functional_diagnostics(
    *,
    memory_dir: Path,
    config: Any,
    diagnostic_epoch_id: str | int | None,
    workers: int,
) -> dict[tuple[str, int | None, tuple[str, ...]], tuple[Any, Any, Any]]:
    from v6 import higher_order_substrate as substrate

    current_state = Path(memory_dir) / "current_state.sqlite"
    if not current_state.exists():
        return {}

    with sqlite3.connect(current_state) as state_conn:
        state_conn.row_factory = sqlite3.Row
        role_links = substrate._links_by_signature(
            state_conn, "role_links", "role_signature"
        )
        concept_links = substrate._links_by_signature(
            state_conn, "concept_links", "concept_signature"
        )
        transfer_rows = [
            dict(row)
            for row in state_conn.execute(
                """
                SELECT attempt_id, source_role_signature AS role_signature, reuse_success, last_seen_global_step,
                       observed_target_role_signature AS observed_role_signature,
                       predicted_target_role_signature AS predicted_role_signature,
                       source_carrier_signature, target_carrier_signature,
                       source_game_key, target_game_key, source_context_key, target_context_key,
                       provenance_mode
                FROM role_transfer_attempts
                WHERE provenance_mode = 'single_source'
                ORDER BY role_signature ASC, attempt_id ASC
                """
            ).fetchall()
        ]
        future_rows = [
            dict(row)
            for row in state_conn.execute(
                """
                SELECT event_id, source_role_id, owner_type, owner_key, option_delta,
                       first_seen_global_step, last_seen_global_step
                FROM future_option_events
                """
            ).fetchall()
        ]
        previous_states = substrate._load_incremental_coverage_states(state_conn)
        concept_rows = [
            dict(row)
            for row in state_conn.execute(
                """
                SELECT concept_signature, first_seen_global_step
                FROM concept_candidates
                ORDER BY concept_signature ASC
                """
            ).fetchall()
        ]

    transfer_history = substrate._build_transfer_history_index(transfer_rows)
    task_specs: list[dict[str, Any]] = []
    for concept_row in concept_rows:
        concept_signature = str(concept_row["concept_signature"])
        first_seen = (
            None
            if concept_row["first_seen_global_step"] is None
            else int(concept_row["first_seen_global_step"])
        )
        links = concept_links.get(concept_signature, {})
        roles = sorted(str(value) for value in links.get("role", set()))
        relevance_links: dict[str, set[str]] = {
            link_type: set(str(value) for value in identifiers)
            for link_type, identifiers in links.items()
        }
        relevance_links.setdefault("_candidate_family", set()).update(
            relevance_links.get("family", set())
        )
        relevance_links.setdefault("_candidate_role_family", set())
        for role in roles:
            for link_type, identifiers in role_links.get(role, {}).items():
                relevance_links.setdefault(link_type, set()).update(
                    str(value) for value in identifiers
                )
                if link_type == "family":
                    relevance_links["_candidate_role_family"].update(
                        str(value) for value in identifiers
                    )
        task_specs.append(
            {
                "key": (concept_signature, first_seen, tuple(roles)),
                "candidate_signature": concept_signature,
                "source_roles": roles,
                "first_seen_global_step": first_seen,
                "candidate_links": relevance_links,
                "previous_state": previous_states.get(concept_signature),
            }
        )

    if not task_specs:
        return {}
    worker_count = min(16, max(1, int(workers)), len(task_specs))
    if worker_count <= 1:
        return {
            spec["key"]: _compute_functional_task(
                current_state=current_state,
                spec=spec,
                transfer_rows=transfer_rows,
                transfer_history=transfer_history,
                future_rows=future_rows,
                diagnostic_epoch_id=diagnostic_epoch_id,
                config=config,
            )
            for spec in task_specs
        }

    cache: dict[tuple[str, int | None, tuple[str, ...]], tuple[Any, Any, Any]] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            (
                spec["key"],
                executor.submit(
                    _compute_functional_task,
                    current_state=current_state,
                    spec=spec,
                    transfer_rows=transfer_rows,
                    transfer_history=transfer_history,
                    future_rows=future_rows,
                    diagnostic_epoch_id=diagnostic_epoch_id,
                    config=config,
                ),
            )
            for spec in task_specs
        ]
        # Consume in deterministic concept order even though work executes in parallel.
        for key, future in futures:
            cache[key] = future.result()
    return cache


def _compute_functional_task(
    *,
    current_state: Path,
    spec: dict[str, Any],
    transfer_rows: list[dict[str, Any]],
    transfer_history: Any,
    future_rows: list[dict[str, Any]],
    diagnostic_epoch_id: str | int | None,
    config: Any,
) -> tuple[Any, Any, Any]:
    uri = f"file:{current_state.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as state_conn:
        state_conn.row_factory = sqlite3.Row
        return _ORIGINAL_BUILD_FUNCTIONAL(
            state_conn=state_conn,
            candidate_signature=spec["candidate_signature"],
            source_roles=list(spec["source_roles"]),
            first_seen_global_step=spec["first_seen_global_step"],
            transfer_rows=transfer_rows,
            transfer_history=transfer_history,
            future_rows=future_rows,
            previous_state=spec["previous_state"],
            diagnostic_epoch_id=diagnostic_epoch_id,
            config=config,
            candidate_links=spec["candidate_links"],
        )


def _build_functional_from_cache(
    *,
    state_conn: sqlite3.Connection,
    candidate_signature: str,
    source_roles: list[str],
    first_seen_global_step: int | None,
    transfer_rows: list[Any],
    transfer_history: Any,
    future_rows: list[Any],
    previous_state: dict[str, Any] | None,
    diagnostic_epoch_id: str | int | None,
    config: Any,
    candidate_links: dict[str, set[str]] | None = None,
) -> tuple[Any, Any, Any]:
    key = (
        str(candidate_signature),
        None if first_seen_global_step is None else int(first_seen_global_step),
        tuple(str(value) for value in source_roles),
    )
    if _ACTIVE_FUNCTIONAL_CACHE is not None and key in _ACTIVE_FUNCTIONAL_CACHE:
        return copy.deepcopy(_ACTIVE_FUNCTIONAL_CACHE[key])
    return _ORIGINAL_BUILD_FUNCTIONAL(
        state_conn=state_conn,
        candidate_signature=candidate_signature,
        source_roles=source_roles,
        first_seen_global_step=first_seen_global_step,
        transfer_rows=transfer_rows,
        transfer_history=transfer_history,
        future_rows=future_rows,
        previous_state=previous_state,
        diagnostic_epoch_id=diagnostic_epoch_id,
        config=config,
        candidate_links=candidate_links,
    )
