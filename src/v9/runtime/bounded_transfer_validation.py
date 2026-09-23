from __future__ import annotations

from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
    as_completed,
)
import multiprocessing as mp
from pathlib import Path
import time
from typing import Any, Callable

from .progress import progress_iter
from . import transfer_validation as _base


def run_transfer_validation_interval(
    runtime: Any,
    specs: tuple[Any, ...],
    args: Any,
    *,
    epoch: int,
    adapter_factory: Callable[..., Any] | None,
) -> _base.TransferValidationStats:
    mode = str(runtime.config.scientific.transfer_validation_mode)
    if mode == "learning_only":
        return _base.TransferValidationStats(
            blocker="automatic transfer validation disabled"
        )
    if adapter_factory is None:
        return _base.TransferValidationStats(blocker="no validation adapter factory")

    config = runtime.config.scientific
    budget = max(900, int(config.transfer_validation_trials_per_interval))
    workers = max(30, int(getattr(config, "transfer_validation_workers", 1)))
    minimum_trials = max(1, int(config.transfer_minimum_trials))
    horizon = max(1, min(32, int(getattr(args, "steps_per_game", 32))))
    time_budget = float(config.transfer_validation_time_budget_seconds)
    deadline = time.monotonic() + time_budget
    candidates = tuple(runtime.transfer_validation_candidates(limit=max(1, budget)))
    candidate_environment_types = {
        (
            tuple(str(value) for value in candidate.get("source_environment_types", ()))
            or ("__unknown__",)
        )[0]
        for candidate in candidates
    }
    candidates = _base._balance_transfer_candidates(candidates)
    log_root = Path(getattr(args, "root", "."))
    threshold = float(config.transfer_effect_threshold)

    runtime.set_telemetry_gauge("transfer_validation_workers", workers)
    runtime.set_telemetry_gauge("transfer_validation_trial_budget", budget)
    runtime.set_telemetry_gauge(
        "transfer_validation_candidate_environment_types",
        len(candidate_environment_types),
    )
    runtime.set_telemetry_gauge("transfer_validation_deadline_exceeded", 0)

    if not candidates:
        blocker = "no M4 concept with grounded action evidence"
        _base._append_transfer_log(
            log_root,
            {
                "epoch": int(epoch),
                "event": "interval_summary",
                "attempted": 0,
                "completed": 0,
                "passed": 0,
                "validated": 0,
                "blocker": blocker,
            },
        )
        return _base.TransferValidationStats(blocker=blocker)

    snapshot = runtime.actor_policy_snapshot()
    tasks: list[tuple[dict[str, Any], Any, int]] = []
    blocked_concepts: dict[Any, list[str]] = {}
    scheduled_concepts: set[Any] = set()
    base_seed = (
        int(getattr(args, "seed", 0))
        + int(epoch) * 10_000_019
        + 7_000_001
    )

    for candidate in candidates:
        if len(tasks) >= budget:
            break
        concept_uid = candidate["concept_uid"]
        concept_confidence = float(
            runtime.graph.payloads.get(concept_uid, {}).get(
                "evidence_confidence", 1.0
            )
        )
        if concept_confidence < 0.25:
            blocked_concepts.setdefault(concept_uid, []).append(
                "source evidence viability confidence below transfer threshold"
            )
            continue
        source_types = set(
            str(value) for value in candidate["source_environment_types"]
        )
        target_specs = _base._eligible_target_specs(tuple(specs), source_types)
        if not target_specs:
            blocked_concepts.setdefault(concept_uid, []).append(
                "no compatible held-out target specification"
            )
            continue
        trials_per_candidate = max(
            minimum_trials, min(8, minimum_trials * 4)
        )
        for trial_index in range(trials_per_candidate):
            if len(tasks) >= budget:
                break
            spec = target_specs[trial_index % len(target_specs)]
            seed = base_seed + len(tasks) * 1009 + trial_index
            tasks.append((candidate, spec, seed))
            scheduled_concepts.add(concept_uid)

    if not tasks:
        blocker = "no eligible transfer-validation trials"
        for concept_uid, blockers in blocked_concepts.items():
            _base._append_transfer_log(
                log_root,
                {
                    "epoch": int(epoch),
                    "event": "concept_blocker",
                    "concept_uid": str(concept_uid),
                    "blocker": blockers[-1],
                },
            )
        _base._append_transfer_log(
            log_root,
            {
                "epoch": int(epoch),
                "event": "interval_summary",
                "attempted": 0,
                "completed": 0,
                "passed": 0,
                "validated": 0,
                "blocker": blocker,
            },
        )
        return _base.TransferValidationStats(blocker=blocker)

    attempted = 0
    completed = 0
    passed = 0
    deadline_exceeded = False
    results_by_concept: dict[Any, list[_base._TransferTrialExecution]] = {}
    with runtime._lock:
        validated_before_uids = {
            uid for uid, concept in runtime._m4.items() if bool(concept.validated)
        }

    factory_qualname = str(getattr(adapter_factory, "__qualname__", ""))
    process_safe = bool(getattr(adapter_factory, "__module__", "")) and (
        "<locals>" not in factory_qualname
    )
    executor_type = ProcessPoolExecutor if process_safe else ThreadPoolExecutor
    max_workers = min(workers, len(tasks))
    executor_kwargs: dict[str, Any] = {"max_workers": max_workers}
    if process_safe:
        executor_kwargs["mp_context"] = mp.get_context("spawn")
        executor_kwargs["initializer"] = _base._transfer_worker_init
    else:
        executor_kwargs["thread_name_prefix"] = "v9-transfer-fallback"
    runtime.set_telemetry_gauge(
        "transfer_validation_executor",
        "process" if process_safe else "thread_fallback",
    )

    pool = executor_type(**executor_kwargs)
    task_index = 0

    def completed_futures():
        nonlocal attempted, task_index, deadline_exceeded
        while task_index < len(tasks):
            if mode == "validation_budgeted" and time.monotonic() >= deadline:
                deadline_exceeded = True
                break
            wave = tasks[task_index : task_index + max_workers]
            task_index += len(wave)
            futures = [
                pool.submit(
                    _base._execute_transfer_trial,
                    candidate,
                    spec,
                    seed=seed,
                    horizon=horizon,
                    snapshot=snapshot,
                    adapter_factory=adapter_factory,
                    args=args,
                )
                for candidate, spec, seed in wave
            ]
            attempted += len(futures)
            runtime.set_telemetry_gauge(
                "transfer_validation_trials_submitted", attempted
            )
            timeout = None
            if mode == "validation_budgeted":
                timeout = max(0.0, deadline - time.monotonic())
            try:
                for future in as_completed(futures, timeout=timeout):
                    yield future
            except FuturesTimeoutError:
                deadline_exceeded = True
                for future in futures:
                    if not future.done():
                        future.cancel()
                break

    try:
        iterator = progress_iter(
            completed_futures(),
            label=f"{time.strftime('[%H:%M]')} epoch {epoch} transfer validation",
            total=len(tasks),
            interval_seconds=max(
                1.0, float(getattr(args, "progress_interval_seconds", 60.0))
            ),
            detail=lambda count, elapsed: (
                f"submitted={attempted} completed={completed} "
                f"passed={passed} elapsed={elapsed:.0f}s"
            ),
        )
        for future in iterator:
            result = future.result()
            results_by_concept.setdefault(result.concept_uid, []).append(result)
            if result.blocker is not None:
                blocked_concepts.setdefault(result.concept_uid, []).append(
                    result.blocker
                )
                continue
            if (
                result.target_environment_id is None
                or result.target_action is None
            ):
                continue

            runtime.record_transfer_validation(
                result.concept_uid,
                target_environment_id=int(result.target_environment_id),
                target_native_action=int(result.target_action),
                enabled_metric=float(result.enabled_metric),
                ablated_metric=float(result.ablated_metric),
                matched=bool(result.matched),
                held_out=True,
                context_scope_id=int(result.context_scope_id),
            )
            completed += 1
            trial_passed = bool(result.matched and result.effect > threshold)
            passed += int(trial_passed)
            _base._append_transfer_log(
                log_root,
                {
                    "epoch": int(epoch),
                    "event": "trial",
                    "concept_uid": str(result.concept_uid),
                    "source_types": list(result.source_types),
                    "formation_scope": list(result.formation_scope),
                    "target_environment_type": result.target_environment_type,
                    "target_environment_id": int(result.target_environment_id),
                    "target_action": int(result.target_action),
                    "context_scope_id": int(result.context_scope_id),
                    "enabled_metric": float(result.enabled_metric),
                    "ablated_metric": float(result.ablated_metric),
                    "effect": float(result.effect),
                    "threshold": threshold,
                    "passed": trial_passed,
                    "matched": bool(result.matched),
                    "held_out": True,
                    "minimum_trials": int(minimum_trials),
                    "horizon": int(horizon),
                    "positive_evidence": int(result.positive_evidence),
                    "negative_evidence": int(result.negative_evidence),
                    "support": int(result.support),
                },
            )
    finally:
        pool.shutdown(wait=True, cancel_futures=True)

    if deadline_exceeded:
        runtime.set_telemetry_gauge("transfer_validation_deadline_exceeded", 1)
    runtime.set_telemetry_gauge(
        "transfer_validation_trials_not_submitted",
        max(0, len(tasks) - attempted),
    )

    concept_uids = {candidate["concept_uid"] for candidate in candidates}
    for concept_uid in concept_uids:
        rows = results_by_concept.get(concept_uid, [])
        blockers = list(blocked_concepts.get(concept_uid, []))
        if concept_uid not in scheduled_concepts and not blockers:
            blockers.append("not scheduled: interval trial budget exhausted")
        now_validated = bool(runtime.is_concept_validated(concept_uid))
        concept_passed = sum(
            int(
                row.blocker is None
                and row.matched
                and row.effect > threshold
            )
            for row in rows
        )
        _base._append_transfer_log(
            log_root,
            {
                "epoch": int(epoch),
                "event": "concept_summary",
                "concept_uid": str(concept_uid),
                "trials": sum(int(row.blocker is None) for row in rows),
                "passed": int(concept_passed),
                "validated_before": concept_uid in validated_before_uids,
                "validated_after": now_validated,
                "blockers": blockers,
            },
        )

    with runtime._lock:
        validated_after_uids = {
            uid for uid, concept in runtime._m4.items() if bool(concept.validated)
        }
    newly_validated_uids = validated_after_uids - validated_before_uids
    validated = len(newly_validated_uids)
    runtime.set_telemetry_gauge(
        "transfer_validated_before", len(validated_before_uids)
    )
    runtime.set_telemetry_gauge(
        "transfer_validated_after", len(validated_after_uids)
    )
    runtime.set_telemetry_gauge("transfer_newly_validated", validated)

    if deadline_exceeded:
        last_blocker = "transfer validation time budget exhausted"
    else:
        last_blocker = None if completed else "all scheduled transfer trials were blocked"
    _base._append_transfer_log(
        log_root,
        {
            "epoch": int(epoch),
            "event": "interval_summary",
            "attempted": int(attempted),
            "planned": int(len(tasks)),
            "completed": int(completed),
            "passed": int(passed),
            "validated": int(validated),
            "workers": int(workers),
            "deadline_exceeded": bool(deadline_exceeded),
            "time_budget_seconds": float(time_budget),
            "blocker": last_blocker,
        },
    )
    return _base.TransferValidationStats(
        attempted, completed, passed, validated, last_blocker
    )
