from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections import deque
from dataclasses import dataclass
import json
import logging
import multiprocessing as mp
from pathlib import Path
from random import Random
import time
import warnings
from typing import Any, Callable

from v9.cognition.action_selection import choose_action
from v9.research.experiments import run_matched_transfer_trial


def _transfer_worker_init() -> None:
    logging.disable(logging.INFO)
    logging.getLogger("arc_agi").setLevel(logging.WARNING)
    logging.getLogger("arc_agi.scorecard").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore")


def _append_transfer_log(root: str | Path, payload: dict[str, Any]) -> None:
    path = Path(root) / "transfer_validation.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


@dataclass(frozen=True, slots=True)
class TransferValidationStats:
    attempted: int = 0
    completed: int = 0
    passed: int = 0
    validated_concepts: int = 0
    blocker: str | None = None


class _BoundaryScoredEnvironment:
    """Score one matched branch only from environment-emitted primary valence."""

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter
        self.score = 0.0
        self.terminal = False
        self.family = str(adapter.identity().family)
        progress = getattr(adapter, "task_progress", None)
        self._last_progress = progress() if callable(progress) else None

    def capture_state(self) -> Any:
        return self.adapter.capture_state()

    def restore_state(self, state: Any) -> None:
        self.adapter.restore_state(state)
        self.score = 0.0
        self.terminal = False

    def available_actions(self) -> tuple[int, ...]:
        if self.terminal:
            return ()
        return tuple(int(value) for value in self.adapter.available_actions())

    def step(self, action: int) -> Any:
        observation = self.adapter.step(int(action))
        boundary = self.adapter.boundary_event()
        progress_fn = getattr(self.adapter, "task_progress", None)
        progress = progress_fn() if callable(progress_fn) else None
        self.score += 10.0 * float(boundary.primary_valence)
        if progress is not None:
            if self.family == "gymnasium":
                self.score += float(progress.score)
            elif self._last_progress is not None:
                self.score += float(progress.score) - float(self._last_progress.score)
            if progress.success:
                self.score += 10.0
            elif progress.failure:
                self.score -= 10.0
            self._last_progress = progress
        if not boundary.continuation:
            self.terminal = True
        return observation


def _baseline_policy(
    adapter: Any,
    snapshot: Any,
    *,
    environment_id: int,
    environment_type: str,
    seed: int,
) -> Callable[[Any, tuple[int, ...], int], int]:
    def policy(observation: Any, actions: tuple[int, ...], step: int) -> int:
        current = adapter.observe() if observation is None else observation
        context_signature = int(adapter.encode_observation(current))
        learned_scores = snapshot.learned_scores(
            environment_id,
            actions,
            environment_type=environment_type,
            context_signature=context_signature,
        )
        # Per-step RNG keeps the ablated continuation identical after a one-step
        # transfer intervention; it does not inherit a shifted RNG stream.
        rng = Random(int(seed) + int(step) * 1_000_003)
        return choose_action(
            snapshot,
            actions,
            rng=rng,
            epsilon=0.0,
            learned_scores=learned_scores,
            target_environment_id=environment_id,
            action_schema_id=int(adapter.action_schema().schema_id),
            environment_type=environment_type,
        )

    return policy


def _eligible_target_specs(specs: tuple[Any, ...], source_types: set[str]) -> tuple[Any, ...]:
    """Select target specs from metadata before constructing any environment."""
    if not source_types:
        return tuple(specs)
    exact = tuple(
        spec for spec in specs
        if str(getattr(spec, "game_id", "")) in source_types
    )
    return exact


@dataclass(frozen=True, slots=True)
class _TransferTrialExecution:
    concept_uid: Any
    before_validated: bool
    source_types: tuple[str, ...]
    formation_scope: tuple[int, ...]
    target_environment_type: str | None = None
    target_environment_id: int | None = None
    target_action: int | None = None
    context_scope_id: int = 0
    enabled_metric: float = 0.0
    ablated_metric: float = 0.0
    matched: bool = False
    blocker: str | None = None
    positive_evidence: int = 0
    negative_evidence: int = 0
    support: int = 0

    @property
    def effect(self) -> float:
        return float(self.enabled_metric) - float(self.ablated_metric)


def _execute_transfer_trial(
    candidate: dict[str, Any],
    spec: Any,
    *,
    seed: int,
    horizon: int,
    snapshot: Any,
    adapter_factory: Callable[..., Any],
    args: Any,
) -> _TransferTrialExecution:
    concept_uid = candidate["concept_uid"]
    formation_scope = tuple(sorted(int(value) for value in candidate["formation_scope"]))
    formation_set = set(formation_scope)
    source_types = tuple(sorted(str(value) for value in candidate["source_environment_types"]))
    source_type_set = set(source_types)
    action_candidates = tuple(int(value) for value in candidate["actions"])
    context_candidates = {int(value) for value in candidate.get("contexts", ())}
    common = {
        "concept_uid": concept_uid,
        "before_validated": bool(candidate["validated"]),
        "source_types": source_types,
        "formation_scope": formation_scope,
        "positive_evidence": int(candidate.get("positive_evidence", 0)),
        "negative_evidence": int(candidate.get("negative_evidence", 0)),
        "support": int(candidate.get("support", 0)),
    }
    adapter = None
    try:
        adapter = adapter_factory(
            spec,
            seed=int(seed),
            env_root=getattr(args, "env_root", None),
            alfred_backend_factory=getattr(args, "alfred_backend_factory", None),
        )
        if not callable(getattr(adapter, "capture_state", None)) or not callable(getattr(adapter, "restore_state", None)):
            return _TransferTrialExecution(**common, blocker=f"{adapter.identity().family} adapter lacks exact snapshot/restore")
        identity = adapter.identity()
        environment_type = str(identity.environment_type)
        target_environment_id = int(identity.instance_id.value)
        if source_type_set and environment_type not in source_type_set:
            return _TransferTrialExecution(**common, blocker="target metadata/adapter environment type mismatch")
        if target_environment_id in formation_set:
            return _TransferTrialExecution(**common, blocker="target environment is part of concept formation provenance")

        baseline = _baseline_policy(
            adapter,
            snapshot,
            environment_id=target_environment_id,
            environment_type=environment_type,
            seed=int(seed),
        )
        current_actions = tuple(int(value) for value in adapter.available_actions())
        fallback_action = next((action for action in action_candidates if action in current_actions), None)
        fallback_state = adapter.capture_state() if fallback_action is not None else None

        if context_candidates:
            matched_context = int(adapter.encode_observation(adapter.observe())) in context_candidates
            seek_limit = max(4, min(128, int(horizon) * 4))
            seek_step = 0
            while not matched_context and seek_step < seek_limit:
                actions = tuple(int(value) for value in adapter.available_actions())
                if not actions:
                    break
                seek_action = baseline(adapter.observe(), actions, seek_step)
                adapter.step(seek_action)
                seek_step += 1
                matched_context = int(adapter.encode_observation(adapter.observe())) in context_candidates
                if not adapter.boundary_event().continuation:
                    adapter.reset()
            if not matched_context and fallback_state is not None:
                adapter.restore_state(fallback_state)

        context_scope_id = int(adapter.encode_observation(adapter.observe()))
        initial_actions = tuple(sorted(set(int(value) for value in adapter.available_actions())))
        target_action = next((action for action in action_candidates if action in initial_actions), None)
        if target_action is None:
            return _TransferTrialExecution(**common, blocker="concept has no target-local grounded action")

        environment = _BoundaryScoredEnvironment(adapter)

        def enabled(observation: Any, actions: tuple[int, ...], step: int) -> int:
            if step == 0 and target_action in actions:
                return int(target_action)
            return baseline(observation, actions, step)

        def ablated(observation: Any, actions: tuple[int, ...], step: int) -> int:
            if step == 0 and target_action in actions and len(actions) > 1:
                controls = tuple(action for action in actions if action != target_action)
                return baseline(observation, controls, step)
            return baseline(observation, actions, step)

        result = run_matched_transfer_trial(
            environment,
            enabled_policy=enabled,
            ablated_policy=ablated,
            metric=lambda row: float(row.score),
            horizon=int(horizon),
        )
        return _TransferTrialExecution(
            **common,
            target_environment_type=environment_type,
            target_environment_id=target_environment_id,
            target_action=int(target_action),
            context_scope_id=int(context_scope_id),
            enabled_metric=float(result.enabled_metric),
            ablated_metric=float(result.ablated_metric),
            matched=bool(result.matched),
        )
    except BaseException as exc:
        return _TransferTrialExecution(**common, blocker=f"trial execution failed: {type(exc).__name__}: {exc}")
    finally:
        if adapter is not None:
            close = getattr(adapter, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException:
                    pass


def run_transfer_validation_interval(
    runtime: Any,
    specs: tuple[Any, ...],
    args: Any,
    *,
    epoch: int,
    adapter_factory: Callable[..., Any] | None,
) -> TransferValidationStats:
    mode = str(runtime.config.scientific.transfer_validation_mode)
    if mode == "learning_only":
        return TransferValidationStats(blocker="automatic transfer validation disabled")
    if adapter_factory is None:
        return TransferValidationStats(blocker="no validation adapter factory")

    config = runtime.config.scientific
    budget = max(900, int(config.transfer_validation_trials_per_interval))
    workers = max(30, int(getattr(config, "transfer_validation_workers", 1)))
    minimum_trials = max(1, int(config.transfer_minimum_trials))
    horizon = max(1, min(32, int(getattr(args, "steps_per_game", 32))))
    deadline = time.monotonic() + float(config.transfer_validation_time_budget_seconds)
    candidates = tuple(runtime.transfer_validation_candidates(limit=max(1, budget)))
    candidate_groups: dict[str, deque[dict[str, Any]]] = {}
    for candidate in candidates:
        types = tuple(str(value) for value in candidate.get("source_environment_types", ()))
        key = types[0] if types else "__unknown__"
        candidate_groups.setdefault(key, deque()).append(candidate)
    balanced_candidates: list[dict[str, Any]] = []
    group_order = sorted(candidate_groups)
    while group_order:
        next_order: list[str] = []
        for key in group_order:
            group = candidate_groups[key]
            if group:
                balanced_candidates.append(group.popleft())
            if group:
                next_order.append(key)
        group_order = next_order
    candidates = tuple(balanced_candidates)
    log_root = Path(getattr(args, "root", "."))
    threshold = float(config.transfer_effect_threshold)

    runtime.set_telemetry_gauge("transfer_validation_workers", workers)
    runtime.set_telemetry_gauge("transfer_validation_trial_budget", budget)
    runtime.set_telemetry_gauge("transfer_validation_candidate_environment_types", len(candidate_groups))

    if not candidates:
        blocker = "no M4 concept with grounded action evidence"
        _append_transfer_log(log_root, {"epoch": int(epoch), "event": "interval_summary", "attempted": 0, "completed": 0, "passed": 0, "validated": 0, "blocker": blocker})
        return TransferValidationStats(blocker=blocker)

    snapshot = runtime.actor_policy_snapshot()
    tasks: list[tuple[dict[str, Any], Any, int]] = []
    blocked_concepts: dict[Any, list[str]] = {}
    base_seed = int(getattr(args, "seed", 0)) + int(epoch) * 10_000_019 + 7_000_001

    for candidate in candidates:
        if len(tasks) >= budget:
            break
        concept_uid = candidate["concept_uid"]
        source_types = set(str(value) for value in candidate["source_environment_types"])
        target_specs = _eligible_target_specs(tuple(specs), source_types)
        if not target_specs:
            blocked_concepts.setdefault(concept_uid, []).append("no compatible held-out target specification")
            continue
        trials_per_candidate = max(minimum_trials, min(8, minimum_trials * 4))
        for trial_index in range(trials_per_candidate):
            if len(tasks) >= budget:
                break
            spec = target_specs[trial_index % len(target_specs)]
            seed = base_seed + len(tasks) * 1009 + trial_index
            tasks.append((candidate, spec, seed))

    if not tasks:
        blocker = "no eligible transfer-validation trials"
        for concept_uid, blockers in blocked_concepts.items():
            _append_transfer_log(log_root, {"epoch": int(epoch), "event": "concept_blocker", "concept_uid": str(concept_uid), "blocker": blockers[-1]})
        _append_transfer_log(log_root, {"epoch": int(epoch), "event": "interval_summary", "attempted": 0, "completed": 0, "passed": 0, "validated": 0, "blocker": blocker})
        return TransferValidationStats(blocker=blocker)

    attempted = len(tasks)
    completed = 0
    passed = 0
    results_by_concept: dict[Any, list[_TransferTrialExecution]] = {}
    with runtime._lock:
        validated_before_uids = {
            uid for uid, concept in runtime._m4.items() if bool(concept.validated)
        }

    factory_qualname = str(getattr(adapter_factory, "__qualname__", ""))
    process_safe = bool(getattr(adapter_factory, "__module__", "")) and "<locals>" not in factory_qualname
    executor_type = ProcessPoolExecutor if process_safe else ThreadPoolExecutor
    executor_kwargs: dict[str, Any] = {"max_workers": min(workers, len(tasks))}
    if process_safe:
        executor_kwargs["mp_context"] = mp.get_context("spawn")
        executor_kwargs["initializer"] = _transfer_worker_init
    else:
        executor_kwargs["thread_name_prefix"] = "v9-transfer-fallback"
    runtime.set_telemetry_gauge("transfer_validation_executor", "process" if process_safe else "thread_fallback")

    with executor_type(**executor_kwargs) as pool:
        futures = [
            pool.submit(
                _execute_transfer_trial,
                candidate,
                spec,
                seed=seed,
                horizon=horizon,
                snapshot=snapshot,
                adapter_factory=adapter_factory,
                args=args,
            )
            for candidate, spec, seed in tasks
        ]
        for future in as_completed(futures):
            result = future.result()
            results_by_concept.setdefault(result.concept_uid, []).append(result)
            if result.blocker is not None:
                blocked_concepts.setdefault(result.concept_uid, []).append(result.blocker)
                continue
            if result.target_environment_id is None or result.target_action is None:
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
            _append_transfer_log(
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
            if mode == "validation_budgeted" and time.monotonic() >= deadline:
                runtime.set_telemetry_gauge("transfer_validation_deadline_exceeded", 1)

    concept_uids = {candidate["concept_uid"] for candidate in candidates}
    for concept_uid in concept_uids:
        rows = results_by_concept.get(concept_uid, [])
        blockers = blocked_concepts.get(concept_uid, [])
        now_validated = bool(runtime.is_concept_validated(concept_uid))
        concept_passed = sum(int(row.blocker is None and row.matched and row.effect > threshold) for row in rows)
        _append_transfer_log(
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
    runtime.set_telemetry_gauge("transfer_validated_before", len(validated_before_uids))
    runtime.set_telemetry_gauge("transfer_validated_after", len(validated_after_uids))
    runtime.set_telemetry_gauge("transfer_newly_validated", validated)

    last_blocker = None if completed else "all scheduled transfer trials were blocked"
    _append_transfer_log(
        log_root,
        {
            "epoch": int(epoch),
            "event": "interval_summary",
            "attempted": int(attempted),
            "completed": int(completed),
            "passed": int(passed),
            "validated": int(validated),
            "workers": int(workers),
            "blocker": last_blocker,
        },
    )
    return TransferValidationStats(attempted, completed, passed, validated, last_blocker)

