from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from random import Random
import time
from typing import Any, Callable

from v9.cognition.action_selection import choose_action
from v9.research.experiments import run_matched_transfer_trial


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

    budget = max(1, int(runtime.config.scientific.transfer_validation_trials_per_interval))
    minimum_trials = max(1, int(runtime.config.scientific.transfer_minimum_trials))
    horizon = max(1, min(32, int(getattr(args, "steps_per_game", 32))))
    deadline = time.monotonic() + float(runtime.config.scientific.transfer_validation_time_budget_seconds)
    candidates = tuple(runtime.transfer_validation_candidates(limit=max(1, budget)))
    log_root = Path(getattr(args, "root", "."))
    threshold = float(runtime.config.scientific.transfer_effect_threshold)
    if not candidates:
        blocker = "no M4 concept with grounded action evidence"
        _append_transfer_log(log_root, {"epoch": int(epoch), "event": "interval_summary", "attempted": 0, "completed": 0, "passed": 0, "validated": 0, "blocker": blocker})
        return TransferValidationStats(blocker=blocker)

    attempted = completed = passed = validated = 0
    last_blocker: str | None = None
    target_cursor_by_type: dict[tuple[str, ...], int] = {}
    per_candidate_seconds = max(
        0.25,
        float(runtime.config.scientific.transfer_validation_time_budget_seconds) / max(1, len(candidates)),
    )

    for candidate in candidates:
        if attempted >= budget:
            break
        concept_uid = candidate["concept_uid"]
        formation_scope = set(int(value) for value in candidate["formation_scope"])
        source_types = set(str(value) for value in candidate["source_environment_types"])
        action_candidates = tuple(int(value) for value in candidate["actions"])
        context_candidates = {int(value) for value in candidate.get("contexts", ())}
        before_validated = bool(candidate["validated"])
        trials_for_concept = 0
        concept_passed = 0
        concept_blockers: list[str] = []

        target_specs = _eligible_target_specs(tuple(specs), source_types)
        if not target_specs:
            last_blocker = "no compatible held-out target specification"
            concept_blockers.append(last_blocker)
            _append_transfer_log(log_root, {"epoch": int(epoch), "event": "concept_blocker", "concept_uid": str(concept_uid), "blocker": last_blocker, "source_types": sorted(source_types), "formation_scope": sorted(formation_scope)})
            continue

        type_key = tuple(sorted(source_types))
        target_cursor = target_cursor_by_type.get(type_key, 0)
        candidate_deadline = min(deadline, time.monotonic() + per_candidate_seconds)
        scans = 0
        maximum_scans = max(len(target_specs) * 2, minimum_trials * 2)
        while attempted < budget and trials_for_concept < minimum_trials:
            now = time.monotonic()
            if mode == "validation_budgeted" and now >= candidate_deadline:
                last_blocker = "candidate transfer validation time budget exhausted"
                concept_blockers.append(last_blocker)
                break
            if scans >= maximum_scans:
                last_blocker = last_blocker or "no eligible held-out target instance"
                concept_blockers.append(last_blocker)
                break
            spec = target_specs[target_cursor % len(target_specs)]
            target_cursor += 1
            target_cursor_by_type[type_key] = target_cursor
            scans += 1
            seed = int(getattr(args, "seed", 0)) + int(epoch) * 10_000_019 + (attempted + scans) * 1009 + 7_000_001
            adapter = None
            try:
                adapter = adapter_factory(
                    spec,
                    seed=seed,
                    env_root=getattr(args, "env_root", None),
                    alfred_backend_factory=getattr(args, "alfred_backend_factory", None),
                )
                if not callable(getattr(adapter, "capture_state", None)) or not callable(getattr(adapter, "restore_state", None)):
                    last_blocker = f"{adapter.identity().family} adapter lacks exact snapshot/restore"
                    concept_blockers.append(last_blocker)
                    continue
                identity = adapter.identity()
                if source_types and str(identity.environment_type) not in source_types:
                    last_blocker = "target metadata/adapter environment type mismatch"
                    concept_blockers.append(last_blocker)
                    continue
                target_environment_id = int(identity.instance_id.value)
                if target_environment_id in formation_scope:
                    last_blocker = "target environment is part of concept formation provenance"
                    concept_blockers.append(last_blocker)
                    continue
                snapshot = runtime.actor_policy_snapshot()
                baseline = _baseline_policy(
                    adapter,
                    snapshot,
                    environment_id=target_environment_id,
                    environment_type=str(identity.environment_type),
                    seed=seed,
                )
                current_actions = tuple(int(value) for value in adapter.available_actions())
                fallback_action = next((action for action in action_candidates if action in current_actions), None)
                fallback_state = adapter.capture_state() if fallback_action is not None else None
                if context_candidates:
                    matched_context = int(adapter.encode_observation(adapter.observe())) in context_candidates
                    seek_limit = max(4, min(128, horizon * 4))
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

                initial_actions = tuple(sorted(set(int(value) for value in adapter.available_actions())))
                target_action = next((action for action in action_candidates if action in initial_actions), None)
                if target_action is None:
                    last_blocker = "concept has no target-local grounded action"
                    concept_blockers.append(last_blocker)
                    continue
                attempted += 1

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
                    horizon=horizon,
                )
                runtime.record_transfer_validation(
                    concept_uid,
                    target_environment_id=target_environment_id,
                    target_native_action=int(target_action),
                    enabled_metric=result.enabled_metric,
                    ablated_metric=result.ablated_metric,
                    matched=result.matched,
                    held_out=True,
                )
                completed += 1
                trials_for_concept += 1
                trial_passed = bool(result.effect > threshold)
                concept_passed += int(trial_passed)
                passed += int(trial_passed)
                _append_transfer_log(
                    log_root,
                    {
                        "epoch": int(epoch),
                        "event": "trial",
                        "concept_uid": str(concept_uid),
                        "source_types": sorted(source_types),
                        "formation_scope": sorted(formation_scope),
                        "target_environment_type": str(identity.environment_type),
                        "target_environment_id": int(target_environment_id),
                        "target_action": int(target_action),
                        "enabled_metric": float(result.enabled_metric),
                        "ablated_metric": float(result.ablated_metric),
                        "effect": float(result.effect),
                        "threshold": threshold,
                        "passed": trial_passed,
                        "matched": bool(result.matched),
                        "held_out": True,
                        "trial_index_for_concept": int(trials_for_concept),
                        "minimum_trials": int(minimum_trials),
                        "horizon": int(horizon),
                        "positive_evidence": int(candidate.get("positive_evidence", 0)),
                        "negative_evidence": int(candidate.get("negative_evidence", 0)),
                        "support": int(candidate.get("support", 0)),
                    },
                )
            finally:
                if adapter is not None:
                    close = getattr(adapter, "close", None)
                    if callable(close):
                        close()

        if not before_validated and runtime.is_concept_validated(concept_uid):
            validated += 1
        if mode == "validation_budgeted" and time.monotonic() >= deadline:
            break

    if completed:
        last_blocker = None
    return TransferValidationStats(attempted, completed, passed, validated, last_blocker)
