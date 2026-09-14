from __future__ import annotations

from dataclasses import dataclass
from random import Random
import time
from typing import Any, Callable

from v9.cognition.action_selection import choose_action
from v9.research.experiments import run_matched_transfer_trial


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
        self.score += float(boundary.primary_valence)
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
        )

    return policy


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
    if not candidates:
        return TransferValidationStats(blocker="no M4 concept with grounded action evidence")

    attempted = completed = passed = validated = 0
    last_blocker: str | None = None
    target_cursor = 0

    for candidate in candidates:
        if attempted >= budget:
            break
        concept_uid = candidate["concept_uid"]
        formation_scope = set(int(value) for value in candidate["formation_scope"])
        source_types = set(str(value) for value in candidate["source_environment_types"])
        action_candidates = tuple(int(value) for value in candidate["actions"])
        before_validated = bool(candidate["validated"])
        trials_for_concept = 0

        preferred = tuple(spec for spec in specs if str(getattr(spec, "game_id", "")) not in source_types)
        target_specs = preferred or tuple(specs)
        if not target_specs:
            last_blocker = "no held-out target specification"
            continue

        while attempted < budget and trials_for_concept < minimum_trials:
            if mode == "validation_budgeted" and time.monotonic() >= deadline:
                last_blocker = "transfer validation time budget exhausted"
                break
            spec = target_specs[target_cursor % len(target_specs)]
            target_cursor += 1
            seed = int(getattr(args, "seed", 0)) + int(epoch) * 10_000_019 + attempted * 1009 + 7_000_001
            adapter = None
            attempted += 1
            try:
                adapter = adapter_factory(
                    spec,
                    seed=seed,
                    env_root=getattr(args, "env_root", None),
                    alfred_backend_factory=getattr(args, "alfred_backend_factory", None),
                )
                if not callable(getattr(adapter, "capture_state", None)) or not callable(getattr(adapter, "restore_state", None)):
                    last_blocker = f"{adapter.identity().family} adapter lacks exact snapshot/restore"
                    continue
                identity = adapter.identity()
                target_environment_id = int(identity.instance_id.value)
                if target_environment_id in formation_scope:
                    last_blocker = "target environment is part of concept formation provenance"
                    continue
                initial_actions = tuple(sorted(set(int(value) for value in adapter.available_actions())))
                target_action = next((action for action in action_candidates if action in initial_actions), None)
                if target_action is None:
                    last_blocker = "concept has no target-local grounded action"
                    continue

                environment = _BoundaryScoredEnvironment(adapter)
                snapshot = runtime.actor_policy_snapshot()
                baseline = _baseline_policy(
                    adapter,
                    snapshot,
                    environment_id=target_environment_id,
                    environment_type=str(identity.environment_type),
                    seed=seed,
                )

                def enabled(observation: Any, actions: tuple[int, ...], step: int) -> int:
                    if step == 0 and target_action in actions:
                        return int(target_action)
                    return baseline(observation, actions, step)

                result = run_matched_transfer_trial(
                    environment,
                    enabled_policy=enabled,
                    ablated_policy=baseline,
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
                passed += int(result.effect > float(runtime.config.scientific.transfer_effect_threshold))
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
