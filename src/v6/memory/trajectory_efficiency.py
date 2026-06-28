from __future__ import annotations

import json
import math
import re
import sqlite3
from dataclasses import asdict, dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any

import numpy as np


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def compact_state_hash(value: Any) -> str:
    array = np.asarray(value, dtype=int)
    payload = array.tobytes() + str(tuple(int(item) for item in array.shape)).encode("utf-8")
    return sha1(payload).hexdigest()[:20]


def infer_epoch_from_path(path: str | Path | None) -> int | None:
    if path is None:
        return None
    match = re.search(r"epoch_(\d+)", str(path))
    if match is None:
        return None
    return int(match.group(1))


def load_best_known_solution_lengths(path: str | Path) -> dict[str, int]:
    candidate = Path(path)
    if not candidate.exists():
        return {}
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:
        return {}
    result: dict[str, int] = {}
    for key, value in dict(payload or {}).items():
        try:
            result[str(key)] = int(value)
        except Exception:
            continue
    return result


def save_best_known_solution_lengths(path: str | Path, values: dict[str, int]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cooked = {str(key): int(value) for key, value in sorted(values.items())}
    target.write_text(json.dumps(cooked, indent=2, sort_keys=True), encoding="utf-8")


@dataclass(frozen=True)
class TrajectoryStep:
    interaction_id: int
    global_step: int
    state_hash_before: str | None
    state_hash_after: str | None
    future_option_delta: float | None
    repeated_state: bool
    repeated_context_action: bool
    no_effect_action: bool
    outcome_state: str | None
    level_completed_event: bool
    action_cost: float
    memory_fitness_base: float
    memory_replay_priority_base: float


@dataclass(frozen=True)
class TrajectoryEfficiencyRecord:
    trajectory_id: str
    game_id: str
    level_id: str | None
    sampler: str | None
    seed: int | None
    epoch: int | None
    outcome_class: str
    comparable_outcome_group_id: str
    efficiency_active: bool
    success: bool
    terminal: bool
    trajectory_length: int
    steps_to_success: int | None
    best_known_solution_length: int | None
    normalized_solve_efficiency: float | None
    future_option_gain: float | None
    future_option_gain_per_action: float | None
    equivalent_outcome_cost_gap: float | None
    loop_count: int
    loop_ratio: float
    repeated_state_count: int
    repeated_state_ratio: float
    blocked_action_count: int
    blocked_action_ratio: float
    wasted_action_count: int
    wasted_action_ratio: float
    unique_state_count: int
    efficiency_score: float | None
    efficiency_memory_bonus: float
    efficiency_replay_bonus: float
    efficiency_retention_bonus: float
    efficiency_promotion_bonus: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TrajectoryEfficiencyStore:
    def __init__(self, connection: sqlite3.Connection, *, auto_commit: bool = True) -> None:
        self.connection = connection
        self.auto_commit = bool(auto_commit)
        self.ensure_schema()

    def ensure_schema(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS trajectory_efficiency (
                trajectory_id TEXT PRIMARY KEY,
                game_id TEXT NOT NULL,
                level_id TEXT,
                sampler TEXT,
                seed INTEGER,
                epoch INTEGER,
                outcome_class TEXT NOT NULL,
                comparable_outcome_group_id TEXT NOT NULL,
                efficiency_active INTEGER NOT NULL DEFAULT 0,
                success INTEGER NOT NULL DEFAULT 0,
                terminal INTEGER NOT NULL DEFAULT 0,
                trajectory_length INTEGER NOT NULL,
                steps_to_success INTEGER,
                best_known_solution_length INTEGER,
                normalized_solve_efficiency REAL,
                future_option_gain REAL,
                future_option_gain_per_action REAL,
                equivalent_outcome_cost_gap REAL,
                loop_count INTEGER,
                loop_ratio REAL,
                repeated_state_count INTEGER,
                repeated_state_ratio REAL,
                blocked_action_count INTEGER,
                blocked_action_ratio REAL,
                wasted_action_count INTEGER,
                wasted_action_ratio REAL,
                unique_state_count INTEGER,
                efficiency_score REAL,
                efficiency_memory_bonus REAL,
                efficiency_replay_bonus REAL,
                efficiency_retention_bonus REAL,
                efficiency_promotion_bonus REAL
            )
            """
        )
        if self.auto_commit:
            self.connection.commit()

    def upsert(self, record: TrajectoryEfficiencyRecord) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO trajectory_efficiency (
                trajectory_id, game_id, level_id, sampler, seed, epoch, outcome_class,
                comparable_outcome_group_id, efficiency_active, success, terminal,
                trajectory_length, steps_to_success, best_known_solution_length,
                normalized_solve_efficiency, future_option_gain, future_option_gain_per_action,
                equivalent_outcome_cost_gap, loop_count, loop_ratio, repeated_state_count,
                repeated_state_ratio, blocked_action_count, blocked_action_ratio,
                wasted_action_count, wasted_action_ratio, unique_state_count, efficiency_score,
                efficiency_memory_bonus, efficiency_replay_bonus, efficiency_retention_bonus,
                efficiency_promotion_bonus
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.trajectory_id,
                record.game_id,
                record.level_id,
                record.sampler,
                None if record.seed is None else int(record.seed),
                None if record.epoch is None else int(record.epoch),
                record.outcome_class,
                record.comparable_outcome_group_id,
                int(bool(record.efficiency_active)),
                int(bool(record.success)),
                int(bool(record.terminal)),
                int(record.trajectory_length),
                None if record.steps_to_success is None else int(record.steps_to_success),
                None if record.best_known_solution_length is None else int(record.best_known_solution_length),
                None if record.normalized_solve_efficiency is None else float(record.normalized_solve_efficiency),
                None if record.future_option_gain is None else float(record.future_option_gain),
                None if record.future_option_gain_per_action is None else float(record.future_option_gain_per_action),
                None if record.equivalent_outcome_cost_gap is None else float(record.equivalent_outcome_cost_gap),
                int(record.loop_count),
                float(record.loop_ratio),
                int(record.repeated_state_count),
                float(record.repeated_state_ratio),
                int(record.blocked_action_count),
                float(record.blocked_action_ratio),
                int(record.wasted_action_count),
                float(record.wasted_action_ratio),
                int(record.unique_state_count),
                None if record.efficiency_score is None else float(record.efficiency_score),
                float(record.efficiency_memory_bonus),
                float(record.efficiency_replay_bonus),
                float(record.efficiency_retention_bonus),
                float(record.efficiency_promotion_bonus),
            ),
        )
        if self.auto_commit:
            self.connection.commit()


class TrajectoryEfficiencyTracker:
    def __init__(
        self,
        *,
        best_known_solution_lengths: dict[str, int] | None = None,
        efficiency_weight: float = 0.05,
        efficiency_replay_weight: float = 0.05,
        efficiency_retention_weight: float = 0.05,
        efficiency_promotion_weight: float = 0.05,
    ) -> None:
        self.best_known_solution_lengths = {str(key): int(value) for key, value in dict(best_known_solution_lengths or {}).items()}
        self.group_counts: dict[str, int] = {}
        self.efficiency_weight = float(efficiency_weight)
        self.efficiency_replay_weight = float(efficiency_replay_weight)
        self.efficiency_retention_weight = float(efficiency_retention_weight)
        self.efficiency_promotion_weight = float(efficiency_promotion_weight)

    def finalize_trajectory(
        self,
        *,
        trajectory_id: str,
        game_id: str,
        level_id: str | None,
        sampler: str | None,
        seed: int | None,
        epoch: int | None,
        outcome_class: str,
        terminal: bool,
        steps: list[TrajectoryStep],
    ) -> TrajectoryEfficiencyRecord:
        length = max(0, len(steps))
        success = str(outcome_class) in {"WIN", "LEVEL_COMPLETE"}
        total_future_option_gain = sum(float(step.future_option_delta or 0.0) for step in steps)
        future_option_gain = float(total_future_option_gain) if steps else None
        future_option_gain_per_action = (float(total_future_option_gain) / float(length)) if length > 0 else None
        final_state_hash = steps[-1].state_hash_after if steps else None
        group_id = self._comparable_group_id(
            game_id=game_id,
            level_id=level_id,
            outcome_class=outcome_class,
            success=success,
            final_state_hash=final_state_hash,
            future_option_gain_per_action=future_option_gain_per_action,
        )
        comparable_count = int(self.group_counts.get(group_id, 0))
        best_key = self._best_key(game_id=game_id, level_id=level_id)
        best_known_before = self.best_known_solution_lengths.get(best_key)
        steps_to_success = length if success else None
        if success and steps_to_success is not None:
            best_known = int(min(best_known_before, steps_to_success)) if best_known_before is not None else int(steps_to_success)
        else:
            best_known = best_known_before
        normalized = None
        if success and steps_to_success and best_known is not None and steps_to_success > 0:
            normalized = clamp01(float(best_known) / float(steps_to_success))
        cost_gap = None
        if success and steps_to_success is not None and best_known is not None:
            cost_gap = float(steps_to_success - best_known)
        state_sequence = [step.state_hash_before for step in steps if step.state_hash_before] + [steps[-1].state_hash_after] if steps and steps[-1].state_hash_after else [step.state_hash_before for step in steps if step.state_hash_before]
        state_sequence = [item for item in state_sequence if item]
        seen_states: set[str] = set()
        loop_count = 0
        for state_hash in state_sequence:
            if state_hash in seen_states:
                loop_count += 1
            seen_states.add(state_hash)
        repeated_state_count = sum(1 for step in steps if step.repeated_state)
        blocked_action_count = sum(1 for step in steps if step.no_effect_action)
        wasted_action_count = sum(1 for step in steps if step.no_effect_action or step.repeated_state or step.repeated_context_action)
        unique_state_count = len(seen_states)
        efficiency_active = False
        efficiency_score = None
        if success:
            efficiency_active = comparable_count > 0 and normalized is not None
            efficiency_score = normalized if efficiency_active else None
        else:
            positive_future_gain = future_option_gain_per_action is not None and float(future_option_gain_per_action) > 0.0
            efficiency_active = comparable_count > 0 and positive_future_gain
            efficiency_score = float(future_option_gain_per_action) if efficiency_active and future_option_gain_per_action is not None else None
        bounded_score = 0.0 if efficiency_score is None else clamp01(float(efficiency_score))
        record = TrajectoryEfficiencyRecord(
            trajectory_id=str(trajectory_id),
            game_id=str(game_id),
            level_id=None if level_id in (None, "") else str(level_id),
            sampler=None if sampler in (None, "") else str(sampler),
            seed=None if seed is None else int(seed),
            epoch=None if epoch is None else int(epoch),
            outcome_class=str(outcome_class),
            comparable_outcome_group_id=group_id,
            efficiency_active=bool(efficiency_active),
            success=bool(success),
            terminal=bool(terminal),
            trajectory_length=int(length),
            steps_to_success=None if steps_to_success is None else int(steps_to_success),
            best_known_solution_length=None if best_known is None else int(best_known),
            normalized_solve_efficiency=None if normalized is None else float(normalized),
            future_option_gain=future_option_gain,
            future_option_gain_per_action=future_option_gain_per_action,
            equivalent_outcome_cost_gap=cost_gap,
            loop_count=int(loop_count),
            loop_ratio=(float(loop_count) / float(length)) if length > 0 else 0.0,
            repeated_state_count=int(repeated_state_count),
            repeated_state_ratio=(float(repeated_state_count) / float(length)) if length > 0 else 0.0,
            blocked_action_count=int(blocked_action_count),
            blocked_action_ratio=(float(blocked_action_count) / float(length)) if length > 0 else 0.0,
            wasted_action_count=int(wasted_action_count),
            wasted_action_ratio=(float(wasted_action_count) / float(length)) if length > 0 else 0.0,
            unique_state_count=int(unique_state_count),
            efficiency_score=None if efficiency_score is None else float(efficiency_score),
            efficiency_memory_bonus=float(self.efficiency_weight) * bounded_score,
            efficiency_replay_bonus=float(self.efficiency_replay_weight) * bounded_score,
            efficiency_retention_bonus=float(self.efficiency_retention_weight) * bounded_score,
            efficiency_promotion_bonus=float(self.efficiency_promotion_weight) * bounded_score,
        )
        self.group_counts[group_id] = comparable_count + 1
        if success and steps_to_success is not None:
            current_best = self.best_known_solution_lengths.get(best_key)
            self.best_known_solution_lengths[best_key] = int(steps_to_success if current_best is None else min(current_best, steps_to_success))
        return record

    def _best_key(self, *, game_id: str, level_id: str | None) -> str:
        return f"{game_id}|{level_id or '__none__'}"

    def _future_option_bucket(self, value: float | None) -> str:
        if value is None:
            return "unknown"
        if float(value) > 0.01:
            return "positive"
        if float(value) < -0.01:
            return "negative"
        return "neutral"

    def _comparable_group_id(
        self,
        *,
        game_id: str,
        level_id: str | None,
        outcome_class: str,
        success: bool,
        final_state_hash: str | None,
        future_option_gain_per_action: float | None,
    ) -> str:
        base = f"{game_id}|{level_id or '__none__'}|{outcome_class}"
        if success:
            return f"{base}|success"
        if final_state_hash:
            return f"{base}|state:{final_state_hash}"
        return f"{base}|future:{self._future_option_bucket(future_option_gain_per_action)}"
