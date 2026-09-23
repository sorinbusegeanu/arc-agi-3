from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path
from threading import RLock
import time
from typing import Any, Callable

from .progress import InlineProgress
from . import transfer_validation as _base


def released_validation_capacity(
    *, target_actor_slots: int, active_actors: int, pending_sampling_jobs: int, max_workers: int
) -> int:
    """Return validation capacity that cannot displace remaining sampling work."""
    if int(pending_sampling_jobs) > 0:
        return 0
    return min(
        max(0, int(max_workers)),
        max(0, int(target_actor_slots) - int(active_actors)),
    )


def deadline_unprocessed(*, pending_trials: int, active_trials: int) -> int:
    """Concrete validation work left unfinished when the deadline expires."""
    return max(0, int(pending_trials)) + max(0, int(active_trials))


@dataclass(frozen=True, slots=True)
class _TrialTicket:
    candidate: dict[str, Any]
    spec: Any
    seed: int
    snapshot: Any
    concept_uid: Any
    evidence_signature: tuple[Any, ...]
    formation_scope: tuple[int, ...]
    capture_watermark: int


class ConcurrentTransferValidationSession:
    """Run held-out transfer trials using actor capacity released during sampling."""

    def __init__(
        self,
        runtime: Any,
        specs: tuple[Any, ...],
        args: Any,
        *,
        epoch: int,
        adapter_factory: Callable[..., Any],
    ) -> None:
        self.runtime = runtime
        self.specs = tuple(specs)
        self.args = args
        self.epoch = int(epoch)
        self.adapter_factory = adapter_factory
        config = runtime.config.scientific
        self.mode = str(config.transfer_validation_mode)
        self.budget = max(900, int(config.transfer_validation_trials_per_interval))
        self.max_workers = max(1, int(getattr(config, "transfer_validation_workers", 30)))
        self.minimum_trials = max(1, int(config.transfer_minimum_trials))
        self.horizon = max(1, min(32, int(getattr(args, "steps_per_game", 32))))
        self.time_budget = max(0.0, float(config.transfer_validation_time_budget_seconds))
        self.threshold = float(config.transfer_effect_threshold)
        self.log_root = Path(getattr(args, "root", "."))
        self.base_seed = int(getattr(args, "seed", 0)) + self.epoch * 10_000_019 + 7_000_001
        self.progress_interval = max(
            0.1, float(getattr(args, "progress_interval_seconds", 60.0))
        )

        self._lock = RLock()
        self._pending: deque[_TrialTicket] = deque()
        self._futures: dict[Any, _TrialTicket] = {}
        self._seen_concepts: set[Any] = set()
        self._blocked_concepts: dict[Any, list[str]] = {}
        self._results_by_concept: dict[Any, list[Any]] = {}
        self._executor: Any | None = None
        self._worker_capacity = 0
        self._sampling_complete = False
        self._deadline: float | None = None
        self._next_discovery = 0.0
        self._ticket_sequence = 0
        self._attempted = 0
        self._completed = 0
        self._passed = 0
        self._stale = 0
        self._completed_during_sampling = 0
        self._deadline_exceeded = False
        self._closed = False
        self._final_stats: _base.TransferValidationStats | None = None

        with runtime._lock:
            self._validated_before = {
                uid for uid, concept in runtime._m4.items() if bool(concept.validated)
            }

        runtime.set_telemetry_gauge("transfer_validation_concurrent", 1)
        runtime.set_telemetry_gauge("transfer_validation_workers", self.max_workers)
        runtime.set_telemetry_gauge("transfer_validation_trial_budget", self.budget)
        runtime.set_telemetry_gauge("transfer_validation_available_workers", 0)
        runtime.set_telemetry_gauge("transfer_validation_trials_submitted", 0)
        runtime.set_telemetry_gauge("transfer_validation_active_trials", 0)
        runtime.set_telemetry_gauge("transfer_validation_pending_trials", 0)
        runtime.set_telemetry_gauge("transfer_validation_completed_during_sampling", 0)
        runtime.set_telemetry_gauge("transfer_validation_deadline_exceeded", 0)
        runtime.set_telemetry_gauge("transfer_validation_unprocessed_wasted", 0)

    @property
    def post_sampling_deadline(self) -> float | None:
        with self._lock:
            return self._deadline

    def diagnostics(self) -> dict[str, int | float | bool]:
        with self._lock:
            return {
                "capacity": int(self._worker_capacity),
                "submitted": int(self._attempted),
                "completed": int(self._completed),
                "passed": int(self._passed),
                "active": int(len(self._futures)),
                "pending": int(len(self._pending)),
                "completed_during_sampling": int(self._completed_during_sampling),
                "sampling_complete": bool(self._sampling_complete),
            }

    def update_sampling_capacity(self, *, active: int, target: int, pending: int) -> None:
        with self._lock:
            if self._sampling_complete or self._closed:
                return
            self._worker_capacity = released_validation_capacity(
                target_actor_slots=int(target),
                active_actors=int(active),
                pending_sampling_jobs=int(pending),
                max_workers=self.max_workers,
            )
            self.runtime.set_telemetry_gauge(
                "transfer_validation_available_workers", self._worker_capacity
            )

    def mark_sampling_complete(self) -> None:
        with self._lock:
            if self._sampling_complete:
                return
            self._sampling_complete = True
            self._deadline = time.monotonic() + self.time_budget
            self._worker_capacity = self.max_workers
            self.runtime.set_telemetry_gauge(
                "transfer_validation_available_workers", self._worker_capacity
            )
            self.runtime.set_telemetry_gauge(
                "transfer_validation_post_sampling_budget_seconds", self.time_budget
            )

    def _ensure_executor(self) -> Any:
        if self._executor is not None:
            return self._executor
        factory_qualname = str(getattr(self.adapter_factory, "__qualname__", ""))
        process_safe = bool(getattr(self.adapter_factory, "__module__", "")) and (
            "<locals>" not in factory_qualname
        )
        if process_safe:
            self._executor = ProcessPoolExecutor(
                max_workers=self.max_workers,
                mp_context=mp.get_context("spawn"),
                initializer=_base._transfer_worker_init,
            )
            executor_name = "process"
        else:
            self._executor = ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="v9-transfer-concurrent",
            )
            executor_name = "thread_fallback"
        self.runtime.set_telemetry_gauge("transfer_validation_executor", executor_name)
        return self._executor

    def _concept_signature(
        self, concept_uid: Any
    ) -> tuple[tuple[Any, ...], tuple[int, ...]] | None:
        with self.runtime._lock:
            concept = self.runtime._m4.get(concept_uid)
            if concept is None:
                return None
            return (
                tuple(concept.provenance.evidence),
                tuple(int(value) for value in concept.provenance.formation_scope),
            )

    def _discover(self, *, force: bool = False) -> None:
        if self._attempted + len(self._pending) >= self.budget:
            return
        now = time.monotonic()
        if not force and now < self._next_discovery:
            return
        self._next_discovery = now + 1.0
        candidates = _base._balance_transfer_candidates(
            tuple(
                self.runtime.transfer_validation_candidates(
                    limit=max(1, self.budget)
                )
            )
        )
        self.runtime.set_telemetry_gauge(
            "transfer_validation_candidate_count", len(candidates)
        )
        if not candidates:
            return
        snapshot = self.runtime.actor_policy_snapshot()
        for candidate in candidates:
            if self._attempted + len(self._pending) >= self.budget:
                break
            concept_uid = candidate["concept_uid"]
            if concept_uid in self._seen_concepts:
                continue
            concept_confidence = float(
                self.runtime.graph.payloads.get(concept_uid, {}).get(
                    "evidence_confidence", 1.0
                )
            )
            if concept_confidence < 0.25:
                self._blocked_concepts.setdefault(concept_uid, []).append(
                    "source evidence viability confidence below transfer threshold"
                )
                continue
            source_types = set(
                str(value) for value in candidate["source_environment_types"]
            )
            target_specs = _base._eligible_target_specs(self.specs, source_types)
            if not target_specs:
                self._blocked_concepts.setdefault(concept_uid, []).append(
                    "no compatible held-out target specification"
                )
                continue
            signature = self._concept_signature(concept_uid)
            if signature is None:
                continue
            evidence_signature, formation_scope = signature
            self._seen_concepts.add(concept_uid)
            trials_per_candidate = max(
                self.minimum_trials, min(8, self.minimum_trials * 4)
            )
            for trial_index in range(trials_per_candidate):
                if self._attempted + len(self._pending) >= self.budget:
                    break
                spec = target_specs[trial_index % len(target_specs)]
                seed = self.base_seed + self._ticket_sequence * 1009 + trial_index
                self._ticket_sequence += 1
                self._pending.append(
                    _TrialTicket(
                        candidate=dict(candidate),
                        spec=spec,
                        seed=int(seed),
                        snapshot=snapshot,
                        concept_uid=concept_uid,
                        evidence_signature=evidence_signature,
                        formation_scope=formation_scope,
                        capture_watermark=int(self.runtime.watermark),
                    )
                )

    def _ticket_is_current(self, ticket: _TrialTicket) -> bool:
        signature = self._concept_signature(ticket.concept_uid)
        if signature is None:
            return False
        evidence_signature, formation_scope = signature
        return (
            evidence_signature == ticket.evidence_signature
            and formation_scope == ticket.formation_scope
        )

    def _record_result(self, ticket: _TrialTicket, result: Any) -> None:
        self._results_by_concept.setdefault(result.concept_uid, []).append(result)
        if result.blocker is not None:
            self._blocked_concepts.setdefault(result.concept_uid, []).append(
                result.blocker
            )
            return
        if result.target_environment_id is None or result.target_action is None:
            return
        if not self._ticket_is_current(ticket):
            self._stale += 1
            self.runtime.set_telemetry_gauge(
                "transfer_validation_stale_tickets", self._stale
            )
            return

        self.runtime.record_transfer_validation(
            result.concept_uid,
            target_environment_id=int(result.target_environment_id),
            target_native_action=int(result.target_action),
            enabled_metric=float(result.enabled_metric),
            ablated_metric=float(result.ablated_metric),
            matched=bool(result.matched),
            held_out=True,
            context_scope_id=int(result.context_scope_id),
        )
        self._completed += 1
        if not self._sampling_complete:
            self._completed_during_sampling += 1
        trial_passed = bool(result.matched and result.effect > self.threshold)
        self._passed += int(trial_passed)
        _base._append_transfer_log(
            self.log_root,
            {
                "epoch": self.epoch,
                "event": "trial",
                "concurrent": True,
                "concept_uid": str(result.concept_uid),
                "capture_watermark": int(ticket.capture_watermark),
                "source_types": list(result.source_types),
                "formation_scope": list(result.formation_scope),
                "target_environment_type": result.target_environment_type,
                "target_environment_id": int(result.target_environment_id),
                "target_action": int(result.target_action),
                "context_scope_id": int(result.context_scope_id),
                "enabled_metric": float(result.enabled_metric),
                "ablated_metric": float(result.ablated_metric),
                "effect": float(result.effect),
                "threshold": self.threshold,
                "passed": trial_passed,
                "matched": bool(result.matched),
                "held_out": True,
            },
        )

    def _drain_completed(self) -> None:
        done = [future for future in tuple(self._futures) if future.done()]
        for future in done:
            ticket = self._futures.pop(future)
            try:
                result = future.result()
            except BaseException as exc:
                self._blocked_concepts.setdefault(ticket.concept_uid, []).append(
                    f"concurrent trial failed: {type(exc).__name__}: {exc}"
                )
                continue
            self._record_result(ticket, result)

    def _deadline_reached(self) -> bool:
        return bool(
            self._sampling_complete
            and self._deadline is not None
            and time.monotonic() >= self._deadline
        )

    def _publish_runtime_counters(self) -> None:
        self.runtime.set_telemetry_gauge(
            "transfer_validation_trials_submitted", self._attempted
        )
        self.runtime.set_telemetry_gauge(
            "transfer_validation_active_trials", len(self._futures)
        )
        self.runtime.set_telemetry_gauge(
            "transfer_validation_pending_trials", len(self._pending)
        )
        self.runtime.set_telemetry_gauge(
            "transfer_validation_completed_during_sampling",
            self._completed_during_sampling,
        )

    def service(self, *, force_discovery: bool = False) -> None:
        with self._lock:
            if self._closed or self.mode == "learning_only":
                return
            self._drain_completed()
            if self._deadline_reached():
                self._deadline_exceeded = True
                self.runtime.set_telemetry_gauge(
                    "transfer_validation_deadline_exceeded", 1
                )
                self._publish_runtime_counters()
                return
            capacity = min(self.max_workers, max(0, int(self._worker_capacity)))
            if capacity <= 0:
                self._publish_runtime_counters()
                return
            if len(self._pending) < capacity:
                self._discover(force=force_discovery)
            if not self._pending:
                self._publish_runtime_counters()
                return
            executor = self._ensure_executor()
            while (
                self._pending
                and len(self._futures) < capacity
                and self._attempted < self.budget
            ):
                ticket = self._pending.popleft()
                future = executor.submit(
                    _base._execute_transfer_trial,
                    ticket.candidate,
                    ticket.spec,
                    seed=ticket.seed,
                    horizon=self.horizon,
                    snapshot=ticket.snapshot,
                    adapter_factory=self.adapter_factory,
                    args=self.args,
                )
                self._futures[future] = ticket
                self._attempted += 1
            self._publish_runtime_counters()

    def _stats(self) -> _base.TransferValidationStats:
        with self.runtime._lock:
            validated_after = {
                uid for uid, concept in self.runtime._m4.items() if bool(concept.validated)
            }
        newly_validated = validated_after - self._validated_before
        validated = len(newly_validated)
        self.runtime.set_telemetry_gauge(
            "transfer_validated_before", len(self._validated_before)
        )
        self.runtime.set_telemetry_gauge(
            "transfer_validated_after", len(validated_after)
        )
        self.runtime.set_telemetry_gauge("transfer_newly_validated", validated)
        if self._deadline_exceeded:
            blocker = "transfer validation post-sampling time budget exhausted"
        elif self._attempted == 0:
            blocker = "no eligible transfer-validation trials"
        elif self._completed == 0:
            blocker = "all scheduled transfer trials were blocked"
        else:
            blocker = None
        return _base.TransferValidationStats(
            self._attempted,
            self._completed,
            self._passed,
            validated,
            blocker,
        )

    def finish(self) -> _base.TransferValidationStats:
        with self._lock:
            if self._final_stats is not None:
                return self._final_stats
            if self.mode == "learning_only":
                self._closed = True
                self._final_stats = _base.TransferValidationStats(
                    blocker="automatic transfer validation disabled"
                )
                return self._final_stats
            if not self._sampling_complete:
                self.mark_sampling_complete()

        progress = InlineProgress(
            f"{time.strftime('[%H:%M]')} epoch {self.epoch} transfer validation"
        )
        next_refresh = 0.0
        deadline_announced = False
        try:
            while True:
                with self._lock:
                    self.service(force_discovery=True)
                    now = time.monotonic()
                    if now >= next_refresh:
                        diag = self.diagnostics()
                        progress.update(
                            "submitted={submitted} completed={completed} passed={passed} "
                            "active={active} pending={pending}".format(**diag)
                        )
                        next_refresh = now + self.progress_interval

                    if self._deadline_reached():
                        self._deadline_exceeded = True
                        pending = len(self._pending)
                        active = len(self._futures)
                        wasted = deadline_unprocessed(
                            pending_trials=pending, active_trials=active
                        )
                        self.runtime.set_telemetry_gauge(
                            "transfer_validation_deadline_exceeded", 1
                        )
                        self.runtime.set_telemetry_gauge(
                            "transfer_validation_unprocessed_wasted", wasted
                        )
                        if not deadline_announced:
                            progress.finish(
                                "deadline reached "
                                f"submitted={self._attempted} completed={self._completed} "
                                f"active={active} pending={pending} "
                                f"unprocessed/wasted={wasted}"
                            )
                            deadline_announced = True
                        break

                    no_work = not self._pending and not self._futures
                    budget_done = self._attempted >= self.budget and not self._futures
                    if no_work or budget_done:
                        break
                time.sleep(0.01)

            with self._lock:
                self._drain_completed()
                if not deadline_announced:
                    progress.finish(
                        f"completed={self._completed} passed={self._passed} "
                        f"validated_pending_summary={len(self._pending)}"
                    )
                self._final_stats = self._stats()
                _base._append_transfer_log(
                    self.log_root,
                    {
                        "epoch": self.epoch,
                        "event": "interval_summary",
                        "concurrent": True,
                        "attempted": int(self._attempted),
                        "completed": int(self._completed),
                        "passed": int(self._passed),
                        "validated": int(self._final_stats.validated_concepts),
                        "completed_during_sampling": int(
                            self._completed_during_sampling
                        ),
                        "stale_tickets": int(self._stale),
                        "deadline_exceeded": bool(self._deadline_exceeded),
                        "unprocessed_wasted": int(
                            deadline_unprocessed(
                                pending_trials=len(self._pending),
                                active_trials=len(self._futures),
                            )
                            if self._deadline_exceeded
                            else 0
                        ),
                        "blocker": self._final_stats.blocker,
                    },
                )
                return self._final_stats
        finally:
            with self._lock:
                self._closed = True
                for future in tuple(self._futures):
                    if not future.running():
                        future.cancel()
                executor = self._executor
                self._executor = None
            if executor is not None:
                executor.shutdown(wait=not self._deadline_exceeded, cancel_futures=True)
