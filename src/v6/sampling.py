from __future__ import annotations

import math
import random
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class ActionSampler(Protocol):
    name: str

    def choose_action(self, system, actions: list[int]) -> int: ...


def sampler_registry() -> dict[str, type["BaseSampler"]]:
    return {
        "random_baseline": RandomBaselineSampler,
        "action_balance": ActionBalanceSampler,
        "action_balance_sampler": ActionBalanceSampler,
        "no_change_avoidance": NoChangeAvoidanceSampler,
        "no_change_avoidance_sampler": NoChangeAvoidanceSampler,
        "low_confidence": LowConfidenceSampler,
        "low_confidence_sampler": LowConfidenceSampler,
        "novelty_delta": NoveltyDeltaSampler,
        "novelty_delta_sampler": NoveltyDeltaSampler,
        "mixed": MixedExplorerSampler,
        "mixed_explorer": MixedExplorerSampler,
        "reset_aware_mixed": ResetAwareMixedExplorerSampler,
        "reset_aware_mixed_explorer": ResetAwareMixedExplorerSampler,
    }


def make_sampler(name: str, *, seed: int | None = None, temperature: float = 0.50, epsilon: float = 0.15) -> "BaseSampler":
    registry = sampler_registry()
    key = str(name).strip()
    if key not in registry:
        raise ValueError(f"unknown sampler: {name}")
    return registry[key](seed=seed, temperature=temperature, epsilon=epsilon)


@dataclass
class BaseSampler:
    seed: int | None = None
    temperature: float = 0.50
    epsilon: float = 0.15
    name: str = "base"

    def __post_init__(self) -> None:
        self.name = str(getattr(type(self), "name", self.name))
        self.rng = random.Random(self.seed)
        self.action_counts: Counter[int] = Counter()
        self.action_no_change: dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=50))
        self.recent_no_change: deque[int] = deque(maxlen=100)
        self.action_non_empty: Counter[int] = Counter()
        self.action_total: Counter[int] = Counter()
        self.context_action_family_counts: dict[tuple, Counter[int]] = defaultdict(Counter)
        self.family_counts: Counter[int] = Counter()
        self.recent_families: deque[int | None] = deque(maxlen=500)
        self.reset_count = 0
        self.reset_unavailable = False

    def choose_action(self, system, actions: list[int]) -> int:
        return int(self.rng.choice(actions))

    def before_step(self, system) -> None:
        return None

    def record_result(
        self,
        *,
        action: int,
        delta,
        actual_family: int | None,
        predicted_family: int | None,
        prediction_error: int | None,
        reset_boundary: bool,
    ) -> None:
        action = int(action)
        changed = int(getattr(delta, "changed_cells", 0))
        no_change = 1 if changed <= 0 else 0
        self.action_counts[action] += 1
        self.action_total[action] += 1
        self.action_non_empty[action] += 0 if no_change else 1
        self.action_no_change[action].append(no_change)
        self.recent_no_change.append(no_change)
        if actual_family is not None:
            family = int(actual_family)
            self.family_counts[family] += 1
            self.recent_families.append(family)
        else:
            self.recent_families.append(None)

    def action_balance_score(self, action: int) -> float:
        return 1.0 / math.sqrt(1.0 + float(self.action_counts[int(action)]))

    def no_change_avoidance_score(self, action: int) -> float:
        window = self.action_no_change[int(action)]
        ratio = 0.0 if not window else sum(window) / len(window)
        return max(0.05, 1.0 - ratio)

    def low_confidence_score(self, system, action: int) -> float:
        contexts = system.context_builder.multi_scale_signatures(int(action), max_level=system.config.context_length)
        contingency = system.contingency_learner.best_stable_for_action(contexts, int(action))
        context = contexts.get(system.config.context_length, ())
        total = sum(
            count
            for (level, known_context, known_action), count in system.contingency_learner.context_action_totals.items()
            if int(known_level := level) == system.config.context_length
            and known_context == context
            and int(known_action) == int(action)
        )
        context_support = sum(
            count
            for (level, known_context, _known_action), count in system.contingency_learner.context_action_totals.items()
            if int(level) == system.config.context_length and known_context == context
        )
        context_rarity = 1.0 / math.sqrt(1.0 + float(context_support))
        if contingency is None:
            confidence = 0.0
            support = total
        else:
            confidence = float(contingency.confidence)
            support = int(contingency.support_count)
        return 0.50 * (1.0 - confidence) + 0.30 * (1.0 / math.sqrt(1.0 + support)) + 0.20 * context_rarity

    def novelty_delta_score(self, system, action: int) -> float:
        contexts = system.context_builder.multi_scale_signatures(int(action), max_level=system.config.context_length)
        best_distribution: dict[int, float] = {}
        for level in sorted(contexts, reverse=True):
            distribution = system.contingency_learner.distribution_at_level(level, contexts[level], int(action))
            if distribution:
                best_distribution = distribution
                break
        entropy = _entropy_from_probs(best_distribution.values())
        rare = 1.0
        if best_distribution:
            rare = sum(probability * (1.0 / math.sqrt(1.0 + self.family_counts[int(family)])) for family, probability in best_distribution.items())
        total = self.action_total[int(action)]
        non_empty = 0.5 if total <= 0 else self.action_non_empty[int(action)] / total
        return 0.40 * entropy + 0.30 * rare + 0.30 * non_empty

    def softmax_sample(self, actions: list[int], scores: list[float]) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(actions))
        temperature = max(1e-6, float(self.temperature))
        values = np.asarray(scores, dtype=float) / temperature
        values = values - np.max(values)
        weights = np.exp(values)
        total = float(np.sum(weights))
        if total <= 0.0 or not np.isfinite(total):
            return int(self.rng.choice(actions))
        probabilities = weights / total
        return int(self.rng.choices(list(actions), weights=probabilities.tolist(), k=1)[0])


class RandomBaselineSampler(BaseSampler):
    name = "random_baseline"

    def choose_action(self, system, actions: list[int]) -> int:
        return int(self.rng.choice(actions))


class ActionBalanceSampler(BaseSampler):
    name = "action_balance"

    def choose_action(self, system, actions: list[int]) -> int:
        return self.softmax_sample(actions, [self.action_balance_score(action) for action in actions])


class NoChangeAvoidanceSampler(BaseSampler):
    name = "no_change_avoidance"

    def choose_action(self, system, actions: list[int]) -> int:
        return self.softmax_sample(actions, [self.no_change_avoidance_score(action) for action in actions])


class LowConfidenceSampler(BaseSampler):
    name = "low_confidence"

    def choose_action(self, system, actions: list[int]) -> int:
        return self.softmax_sample(actions, [self.low_confidence_score(system, action) for action in actions])


class NoveltyDeltaSampler(BaseSampler):
    name = "novelty_delta"

    def choose_action(self, system, actions: list[int]) -> int:
        return self.softmax_sample(actions, [self.novelty_delta_score(system, action) for action in actions])


class MixedExplorerSampler(BaseSampler):
    name = "mixed"

    def choose_action(self, system, actions: list[int]) -> int:
        scores = [
            0.20 * self.action_balance_score(action)
            + 0.25 * self.no_change_avoidance_score(action)
            + 0.25 * self.low_confidence_score(system, action)
            + 0.30 * self.novelty_delta_score(system, action)
            for action in actions
        ]
        return self.softmax_sample(actions, scores)


class ResetAwareMixedExplorerSampler(MixedExplorerSampler):
    name = "reset_aware_mixed"

    def before_step(self, system) -> None:
        should_reset = False
        if len(self.recent_no_change) >= 100 and sum(self.recent_no_change) / len(self.recent_no_change) > 0.80:
            should_reset = True
        if len(self.recent_families) >= 50:
            last = list(self.recent_families)[-50:]
            if last[0] is not None and all(item == last[0] for item in last):
                should_reset = True
        if len(self.recent_families) >= 500 and len({item for item in self.recent_families if item is not None}) == 0:
            should_reset = True
        if not should_reset:
            return
        reset = getattr(system.env, "reset", None)
        if callable(reset):
            reset()
            self.reset_count += 1
        else:
            raw_env = getattr(system.env, "env", None)
            raw_reset = getattr(raw_env, "reset", None)
            if callable(raw_reset):
                raw = raw_reset()
                if hasattr(system.env, "_last_raw"):
                    system.env._last_raw = raw
                if hasattr(system.env, "_last_grid"):
                    from v6.environment.arc_adapter import _grid_from_raw

                    system.env._last_grid = _grid_from_raw(raw)
                if hasattr(system.env, "reset_count"):
                    system.env.reset_count += 1
                self.reset_count += 1
            else:
                self.reset_unavailable = True


def _entropy_from_probs(values) -> float:
    probs = [float(value) for value in values if float(value) > 0.0]
    total = sum(probs)
    if total <= 0.0:
        return 0.0
    normalized = [value / total for value in probs]
    max_entropy = math.log(max(2, len(normalized)))
    entropy = -sum(value * math.log(value) for value in normalized)
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0
