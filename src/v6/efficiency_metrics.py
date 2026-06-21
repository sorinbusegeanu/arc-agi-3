from __future__ import annotations

import json
from collections import Counter, deque
from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class EfficiencyEvent:
    interaction_id: str
    step_index: int
    context_signature: str | None
    action_signature: str | None
    outcome_signature: str | None
    state_signature: str | None
    delta_signature: str | None
    action_cost: float
    cumulative_cost: float
    repeated_state: bool
    repeated_context_action: bool
    no_effect_action: bool
    terminal_outcome: bool
    reward_value: float
    best_known_cost_for_outcome: float | None
    normalized_solve_efficiency: float | None
    equivalent_outcome_cost_gap: float | None
    future_option_gain_per_cost: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EfficiencySummary:
    efficiency_event_count: int
    total_action_cost: float
    mean_action_cost: float
    no_effect_action_count: int
    repeated_state_count: int
    repeated_context_action_count: int
    terminal_outcome_count: int
    distinct_outcome_count: int
    mean_normalized_solve_efficiency: float | None
    max_normalized_solve_efficiency: float | None
    mean_equivalent_outcome_cost_gap: float | None
    max_equivalent_outcome_cost_gap: float | None
    mean_future_option_gain_per_cost: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def scalarize_reward(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Mapping):
        return _largest_signed_number(value.values())
    if isinstance(value, (list, tuple)):
        return _largest_signed_number(value)
    return 0.0


def bool_scalar(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, Mapping):
        return any(bool_scalar(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(bool_scalar(item) for item in value)
    return False


def stable_signature(value: Any, *, max_len: int = 512) -> str | None:
    if value is None:
        return None
    try:
        text = json.dumps(value, sort_keys=True, default=str)
    except TypeError:
        text = str(value)
    return text[: max(0, int(max_len))]


def extract_delta_signature(delta: Any) -> str | None:
    if delta is None:
        return None
    for key in ("signature", "delta_signature", "delta_id", "id", "changed_cells", "position", "source_position", "target_position"):
        value = _lookup(delta, key)
        if value is not None:
            return stable_signature({key: value}, max_len=256)
    return stable_signature(delta, max_len=256)


def is_no_effect_delta(delta: Any) -> bool:
    if delta is None:
        return True
    for key in ("changed_cells", "changes", "effects"):
        value = _lookup(delta, key)
        if value == [] or value == () or value == {}:
            return True
        if isinstance(value, int) and int(value) == 0:
            return True
    text = str(delta).lower()
    return any(marker in text for marker in ("no_change", "noop", "no_effect"))


class EfficiencyTracker:
    def __init__(
        self,
        *,
        action_cost_default: float = 1.0,
        recent_window_size: int = 100,
    ) -> None:
        self.action_cost_default = float(action_cost_default)
        self.recent_window_size = int(recent_window_size)
        self.events: list[EfficiencyEvent] = []
        self.step_index = 0
        self.cumulative_cost = 0.0
        self.seen_state_signatures: Counter[str] = Counter()
        self.seen_context_actions: Counter[str] = Counter()
        self.best_cost_by_outcome: dict[str, float] = {}
        self.recent_outcomes: deque[str] = deque(maxlen=self.recent_window_size)

    def record_interaction(
        self,
        *,
        interaction_id: str,
        before_observation: Any,
        after_observation: Any,
        delta: Any,
        context_signature: str | None,
        action_signature: str | None,
        reward: Any,
        terminated: Any,
        truncated: Any,
        future_option_delta: float | None = None,
        action_cost: float | None = None,
    ) -> EfficiencyEvent:
        cost = float(action_cost if action_cost is not None else self.action_cost_default)
        self.step_index += 1
        self.cumulative_cost += cost
        state_signature = stable_signature(before_observation)
        delta_signature = extract_delta_signature(delta)
        reward_value = scalarize_reward(reward)
        terminal_outcome = bool_scalar(terminated) or bool_scalar(truncated)
        outcome_signature = stable_signature(
            {
                "after": stable_signature(after_observation, max_len=256),
                "delta": delta_signature,
                "reward": reward_value,
                "terminal": terminal_outcome,
            }
        )
        repeated_state = state_signature is not None and self.seen_state_signatures[state_signature] > 0
        context_action_key = f"{context_signature}|{action_signature}"
        repeated_context_action = self.seen_context_actions[context_action_key] > 0
        no_effect_action = is_no_effect_delta(delta)
        best_known = None if outcome_signature is None else self.best_cost_by_outcome.get(outcome_signature)
        if terminal_outcome and best_known is not None and self.cumulative_cost > 0:
            normalized = min(1.0, best_known / self.cumulative_cost)
        elif terminal_outcome:
            normalized = 1.0
        else:
            normalized = None
        gap = None if best_known is None else max(0.0, self.cumulative_cost - best_known)
        if future_option_delta is not None and cost > 0:
            gain_per_cost = float(future_option_delta) / cost
        else:
            gain_per_cost = None
        event = EfficiencyEvent(
            interaction_id=str(interaction_id),
            step_index=self.step_index,
            context_signature=None if context_signature is None else str(context_signature),
            action_signature=None if action_signature is None else str(action_signature),
            outcome_signature=outcome_signature,
            state_signature=state_signature,
            delta_signature=delta_signature,
            action_cost=cost,
            cumulative_cost=self.cumulative_cost,
            repeated_state=bool(repeated_state),
            repeated_context_action=bool(repeated_context_action),
            no_effect_action=bool(no_effect_action),
            terminal_outcome=bool(terminal_outcome),
            reward_value=reward_value,
            best_known_cost_for_outcome=best_known,
            normalized_solve_efficiency=normalized,
            equivalent_outcome_cost_gap=gap,
            future_option_gain_per_cost=gain_per_cost,
        )
        if outcome_signature is not None:
            if outcome_signature not in self.best_cost_by_outcome:
                self.best_cost_by_outcome[outcome_signature] = self.cumulative_cost
            else:
                self.best_cost_by_outcome[outcome_signature] = min(self.best_cost_by_outcome[outcome_signature], self.cumulative_cost)
            self.recent_outcomes.append(outcome_signature)
        if state_signature is not None:
            self.seen_state_signatures[state_signature] += 1
        self.seen_context_actions[context_action_key] += 1
        self.events.append(event)
        return event

    def summary(self) -> dict[str, Any]:
        normalized = [float(item.normalized_solve_efficiency) for item in self.events if item.normalized_solve_efficiency is not None]
        gaps = [float(item.equivalent_outcome_cost_gap) for item in self.events if item.equivalent_outcome_cost_gap is not None]
        gains = [float(item.future_option_gain_per_cost) for item in self.events if item.future_option_gain_per_cost is not None]
        summary = EfficiencySummary(
            efficiency_event_count=len(self.events),
            total_action_cost=self.cumulative_cost,
            mean_action_cost=(self.cumulative_cost / len(self.events)) if self.events else 0.0,
            no_effect_action_count=sum(1 for item in self.events if item.no_effect_action),
            repeated_state_count=sum(1 for item in self.events if item.repeated_state),
            repeated_context_action_count=sum(1 for item in self.events if item.repeated_context_action),
            terminal_outcome_count=sum(1 for item in self.events if item.terminal_outcome),
            distinct_outcome_count=len({item.outcome_signature for item in self.events if item.outcome_signature is not None}),
            mean_normalized_solve_efficiency=(sum(normalized) / len(normalized)) if normalized else None,
            max_normalized_solve_efficiency=max(normalized) if normalized else None,
            mean_equivalent_outcome_cost_gap=(sum(gaps) / len(gaps)) if gaps else None,
            max_equivalent_outcome_cost_gap=max(gaps) if gaps else None,
            mean_future_option_gain_per_cost=(sum(gains) / len(gains)) if gains else None,
        )
        return summary.to_dict()


def _largest_signed_number(values: Any) -> float:
    chosen = 0.0
    found = False
    for item in values:
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float)):
            if not found or abs(float(item)) > abs(chosen):
                chosen = float(item)
                found = True
    return chosen if found else 0.0


def _lookup(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)
