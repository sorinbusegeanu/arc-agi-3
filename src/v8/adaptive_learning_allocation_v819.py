from __future__ import annotations

import json
import math
import os
import queue
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Iterable

from v8.model import MemoryLevel, MemoryType, MemoryUid, ValidationState, stable_u64


_INSTALLED = False
_SAMPLING_MODE_ENV = "ARC_AGI3_V8_SAMPLING_MODE"
_ALTERNATIVE_EXCLUDE_ENV = "ARC_AGI3_V8_ALTERNATIVE_EXCLUDE"
_DEFAULT_LEASE_STEPS = 4096
_DEFAULT_STABILIZATION_GENERATIONS = 256
_DEFAULT_MAX_VALIDATIONS_WITHOUT_IMPROVEMENT = 256
_DEFAULT_OPTIMIZATION_VALIDATION_BUDGET = 2048
_DEFAULT_MIN_MEANINGFUL_IMPROVEMENT = 1
_ALLOCATION_STDOUT_SECONDS = 300.0
_ALLOCATION_LOG_SECONDS = 60.0

_BASE_RUN_ACTOR_JOBS = None
_BASE_PLAN_CANDIDATES = None
_BASE_RUNTIME_INIT = None
_BASE_RUNTIME_AUX = None
_BASE_RUNTIME_METRICS = None
_BASE_SERVICE_SUBMIT = None
_BASE_SUCCESS_TO_DICT = None
_BASE_SUCCESS_FROM_DICT = None
_BASE_ROUTE_CANDIDATE = None

_SOURCE_KIND_BY_TRAJECTORY: dict[str, str] = {}


class GameLearningState(str, Enum):
    UNSOLVED = "UNSOLVED"
    SOLVED_OPTIMIZING = "SOLVED_OPTIMIZING"
    SOLVED_STABLE = "SOLVED_STABLE"


class SamplingMode(str, Enum):
    DISCOVERY = "DISCOVERY"
    VERIFY = "VERIFY"
    ALTERNATIVE = "ALTERNATIVE"
    TRANSFER = "TRANSFER"


class FrontierSource(str, Enum):
    SAMPLER = "SAMPLER"
    TRAJECTORY_OPTIMIZER = "TRAJECTORY_OPTIMIZER"
    TRANSFER = "TRANSFER"


@dataclass(frozen=True, slots=True, order=True)
class FrontierScope:
    game_id: str
    level: int
    context_bucket: int
    outcome_hi: int
    outcome_lo: int

    @property
    def outcome_uid(self) -> MemoryUid:
        return MemoryUid(int(self.outcome_hi), int(self.outcome_lo))

    def key(self) -> str:
        return (
            f"{self.game_id}|{int(self.level)}|{int(self.context_bucket)}|"
            f"{int(self.outcome_hi):016x}{int(self.outcome_lo):016x}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "game_id": self.game_id,
            "level": int(self.level),
            "context_bucket": int(self.context_bucket),
            "outcome_uid": [int(self.outcome_hi), int(self.outcome_lo)],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "FrontierScope":
        uid = raw.get("outcome_uid", (0, 0))
        if not isinstance(uid, (list, tuple)) or len(uid) < 2:
            uid = (0, 0)
        return cls(
            str(raw.get("game_id", "")),
            int(raw.get("level", 0)),
            int(raw.get("context_bucket", 0)),
            int(uid[0]),
            int(uid[1]),
        )


@dataclass(frozen=True, slots=True)
class FrontierCandidate:
    strategy_uid: MemoryUid
    trajectory_id: str
    action_hash: int
    cost: int
    attempts: int
    successes: int
    validation_state: int
    source: FrontierSource
    generation: int
    parent_strategy_uid: MemoryUid = MemoryUid(0, 0)

    @property
    def reliability(self) -> float:
        return 0.0 if int(self.attempts) <= 0 else max(
            0.0,
            min(1.0, float(self.successes) / float(self.attempts)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy_uid": [int(self.strategy_uid.hi), int(self.strategy_uid.lo)],
            "trajectory_id": str(self.trajectory_id),
            "action_hash": int(self.action_hash),
            "cost": int(self.cost),
            "attempts": int(self.attempts),
            "successes": int(self.successes),
            "validation_state": int(self.validation_state),
            "source": self.source.value,
            "generation": int(self.generation),
            "parent_strategy_uid": [
                int(self.parent_strategy_uid.hi),
                int(self.parent_strategy_uid.lo),
            ],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "FrontierCandidate":
        strategy = raw.get("strategy_uid", (0, 0))
        parent = raw.get("parent_strategy_uid", (0, 0))
        if not isinstance(strategy, (list, tuple)) or len(strategy) < 2:
            strategy = (0, 0)
        if not isinstance(parent, (list, tuple)) or len(parent) < 2:
            parent = (0, 0)
        source_raw = str(raw.get("source", FrontierSource.SAMPLER.value))
        try:
            source = FrontierSource(source_raw)
        except ValueError:
            source = FrontierSource.SAMPLER
        return cls(
            MemoryUid(int(strategy[0]), int(strategy[1])),
            str(raw.get("trajectory_id", "")),
            int(raw.get("action_hash", 0)),
            int(raw.get("cost", 0)),
            int(raw.get("attempts", 0)),
            int(raw.get("successes", 0)),
            int(raw.get("validation_state", int(ValidationState.TESTED))),
            source,
            int(raw.get("generation", 0)),
            MemoryUid(int(parent[0]), int(parent[1])),
        )


def _dominates(left: FrontierCandidate, right: FrontierCandidate) -> bool:
    no_worse = (
        int(left.cost) <= int(right.cost)
        and float(left.reliability) >= float(right.reliability)
        and int(left.validation_state) >= int(right.validation_state)
    )
    strictly = (
        int(left.cost) < int(right.cost)
        or float(left.reliability) > float(right.reliability) + 1e-12
        or int(left.validation_state) > int(right.validation_state)
    )
    return bool(no_worse and strictly)


class M7StrategyFrontier:
    def __init__(self, *, max_candidates_per_scope: int = 16) -> None:
        self.max_candidates_per_scope = max(2, int(max_candidates_per_scope))
        self._rows: dict[FrontierScope, tuple[FrontierCandidate, ...]] = {}
        self._versions: dict[FrontierScope, int] = {}

    def candidates(self, scope: FrontierScope) -> tuple[FrontierCandidate, ...]:
        return self._rows.get(scope, ())

    def version(self, scope: FrontierScope) -> int:
        return int(self._versions.get(scope, 0))

    def scopes(self) -> tuple[FrontierScope, ...]:
        return tuple(sorted(self._rows))

    def add(
        self,
        scope: FrontierScope,
        candidate: FrontierCandidate,
    ) -> tuple[bool, int]:
        prior = list(self._rows.get(scope, ()))
        for row in prior:
            if (
                row.trajectory_id == candidate.trajectory_id
                and int(row.action_hash) == int(candidate.action_hash)
            ):
                if (
                    int(candidate.attempts) <= int(row.attempts)
                    and int(candidate.successes) <= int(row.successes)
                    and int(candidate.validation_state) <= int(row.validation_state)
                ):
                    return False, self.version(scope)
                prior.remove(row)
                break
        if any(_dominates(row, candidate) for row in prior):
            return False, self.version(scope)
        survivors = [row for row in prior if not _dominates(candidate, row)]
        survivors.append(candidate)
        survivors.sort(
            key=lambda row: (
                int(row.cost),
                -float(row.reliability),
                -int(row.validation_state),
                row.trajectory_id,
            )
        )
        survivors = survivors[: self.max_candidates_per_scope]
        changed = tuple(survivors) != tuple(self._rows.get(scope, ()))
        if changed:
            self._rows[scope] = tuple(survivors)
            self._versions[scope] = self.version(scope) + 1
        return changed, self.version(scope)

    def winner(self, scope: FrontierScope) -> FrontierCandidate | None:
        rows = self._rows.get(scope, ())
        if not rows:
            return None
        return min(
            rows,
            key=lambda row: (
                int(row.cost),
                -float(row.reliability),
                -int(row.validation_state),
                row.trajectory_id,
            ),
        )

    def best_for_game(self, game_id: str) -> tuple[FrontierScope, FrontierCandidate] | None:
        choices = []
        for scope, rows in self._rows.items():
            if scope.game_id != str(game_id) or not rows:
                continue
            winner = self.winner(scope)
            if winner is not None:
                choices.append((scope, winner))
        if not choices:
            return None
        return min(
            choices,
            key=lambda item: (
                -int(item[0].level),
                int(item[1].cost),
                -float(item[1].reliability),
                item[1].trajectory_id,
            ),
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "max_candidates_per_scope": self.max_candidates_per_scope,
            "scopes": [
                {
                    "scope": scope.to_dict(),
                    "version": self.version(scope),
                    "candidates": [row.to_dict() for row in self._rows.get(scope, ())],
                }
                for scope in self.scopes()
            ],
        }

    def load_state(self, raw: dict[str, object] | None) -> None:
        if not isinstance(raw, dict):
            return
        self._rows.clear()
        self._versions.clear()
        for item in raw.get("scopes", ()):
            if not isinstance(item, dict):
                continue
            scope_raw = item.get("scope", {})
            if not isinstance(scope_raw, dict):
                continue
            scope = FrontierScope.from_dict(scope_raw)
            rows = []
            for candidate_raw in item.get("candidates", ()):
                if isinstance(candidate_raw, dict):
                    rows.append(FrontierCandidate.from_dict(candidate_raw))
            if rows:
                self._rows[scope] = tuple(rows[: self.max_candidates_per_scope])
                self._versions[scope] = max(1, int(item.get("version", 1)))


@dataclass(slots=True)
class GameLevelLearningRecord:
    state: GameLearningState = GameLearningState.UNSOLVED
    first_success_generation: int = 0
    last_success_generation: int = 0
    last_frontier_improvement_generation: int = 0
    frontier_version: int = 0
    optimizer_exhausted_version: int = -1
    validations_since_improvement: int = 0
    optimization_rounds: int = 0
    consumed_optimization_budget: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "first_success_generation": int(self.first_success_generation),
            "last_success_generation": int(self.last_success_generation),
            "last_frontier_improvement_generation": int(
                self.last_frontier_improvement_generation
            ),
            "frontier_version": int(self.frontier_version),
            "optimizer_exhausted_version": int(self.optimizer_exhausted_version),
            "validations_since_improvement": int(self.validations_since_improvement),
            "optimization_rounds": int(self.optimization_rounds),
            "consumed_optimization_budget": int(self.consumed_optimization_budget),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "GameLevelLearningRecord":
        try:
            state = GameLearningState(str(raw.get("state", GameLearningState.UNSOLVED.value)))
        except ValueError:
            state = GameLearningState.UNSOLVED
        return cls(
            state,
            int(raw.get("first_success_generation", 0)),
            int(raw.get("last_success_generation", 0)),
            int(raw.get("last_frontier_improvement_generation", 0)),
            int(raw.get("frontier_version", 0)),
            int(raw.get("optimizer_exhausted_version", -1)),
            int(raw.get("validations_since_improvement", 0)),
            int(raw.get("optimization_rounds", 0)),
            int(raw.get("consumed_optimization_budget", 0)),
        )


@dataclass(slots=True)
class GamePrioritySignals:
    competence_improvement: float = 1.0
    prediction_error: float = 1.0
    contradiction: float = 1.0
    novelty: float = 1.0
    transfer_opportunity: float = 1.0

    @property
    def multiplier(self) -> float:
        value = (
            float(self.competence_improvement)
            * float(self.prediction_error)
            * float(self.contradiction)
            * float(self.novelty)
            * float(self.transfer_opportunity)
        )
        return max(0.25, min(4.0, value))


@dataclass(frozen=True, slots=True)
class AdaptiveLearningConfig:
    unsolved_weight: float = 1.0
    optimizing_weight: float = 0.20
    stable_weight: float = 0.075
    lease_steps: int = _DEFAULT_LEASE_STEPS
    stabilization_generations: int = _DEFAULT_STABILIZATION_GENERATIONS
    max_validations_without_improvement: int = (
        _DEFAULT_MAX_VALIDATIONS_WITHOUT_IMPROVEMENT
    )
    optimization_validation_budget: int = _DEFAULT_OPTIMIZATION_VALIDATION_BUDGET
    min_meaningful_improvement: int = _DEFAULT_MIN_MEANINGFUL_IMPROVEMENT

    @classmethod
    def from_environment(cls) -> "AdaptiveLearningConfig":
        def env_float(name: str, default: float) -> float:
            try:
                return float(os.environ.get(name, default))
            except (TypeError, ValueError):
                return float(default)

        def env_int(name: str, default: int) -> int:
            try:
                return int(os.environ.get(name, default))
            except (TypeError, ValueError):
                return int(default)

        return cls(
            unsolved_weight=max(
                1e-6, env_float("ARC_AGI3_V8_UNSOLVED_WEIGHT", 1.0)
            ),
            optimizing_weight=max(
                1e-6, env_float("ARC_AGI3_V8_OPTIMIZING_WEIGHT", 0.20)
            ),
            stable_weight=max(
                1e-6, env_float("ARC_AGI3_V8_STABLE_WEIGHT", 0.075)
            ),
            lease_steps=max(
                64, env_int("ARC_AGI3_V8_ALLOCATION_LEASE_STEPS", _DEFAULT_LEASE_STEPS)
            ),
            stabilization_generations=max(
                1,
                env_int(
                    "ARC_AGI3_V8_STABILIZATION_GENERATIONS",
                    _DEFAULT_STABILIZATION_GENERATIONS,
                ),
            ),
            max_validations_without_improvement=max(
                1,
                env_int(
                    "ARC_AGI3_V8_MAX_VALIDATIONS_WITHOUT_IMPROVEMENT",
                    _DEFAULT_MAX_VALIDATIONS_WITHOUT_IMPROVEMENT,
                ),
            ),
            optimization_validation_budget=max(
                1,
                env_int(
                    "ARC_AGI3_V8_OPTIMIZATION_VALIDATION_BUDGET",
                    _DEFAULT_OPTIMIZATION_VALIDATION_BUDGET,
                ),
            ),
            min_meaningful_improvement=max(
                1,
                env_int(
                    "ARC_AGI3_V8_MIN_MEANINGFUL_IMPROVEMENT",
                    _DEFAULT_MIN_MEANINGFUL_IMPROVEMENT,
                ),
            ),
        )


@dataclass(slots=True)
class GameRunTelemetry:
    sample_steps: int = 0
    leases: int = 0
    last_mode: SamplingMode = SamplingMode.DISCOVERY
    verify_attempts: int = 0
    alternative_attempts: int = 0
    transfer_attempts: int = 0
    optimizer_candidates: int = 0
    optimizer_validations: int = 0
    optimizer_successes: int = 0
    optimizer_saved_actions: int = 0


@dataclass(frozen=True, slots=True)
class GameLearningTelemetrySnapshot:
    game_id: str
    state: str
    sample_steps: int
    sample_share: float
    sampling_weight: float
    sampling_mode: str
    frontier_cost: int
    frontier_reliability: float
    frontier_source: str
    frontier_version: int
    optimizer_candidates: int
    optimizer_validations: int
    optimizer_successes: int
    optimizer_saved_actions: int
    optimizer_active: bool
    alternative_attempts: int
    transfer_attempts: int


class AdaptiveLearningCoordinator:
    def __init__(
        self,
        *,
        config: AdaptiveLearningConfig | None = None,
        event_sink=None,
    ) -> None:
        self.config = config or AdaptiveLearningConfig.from_environment()
        self.frontier = M7StrategyFrontier()
        self._lock = threading.RLock()
        self._records: dict[tuple[str, int], GameLevelLearningRecord] = {}
        self._game_won: dict[str, bool] = {}
        self._games: set[str] = set()
        self._signals: dict[str, GamePrioritySignals] = {}
        self._run: dict[str, GameRunTelemetry] = {}
        self._mode_cursor: dict[str, int] = {}
        self._event_sink = event_sink

    def register_games(self, games: Iterable[str]) -> None:
        with self._lock:
            for game in games:
                value = str(game)
                self._games.add(value)
                self._game_won.setdefault(value, False)
                self._signals.setdefault(value, GamePrioritySignals())
                self._run.setdefault(value, GameRunTelemetry())

    def _emit(self, message: str) -> None:
        sink = self._event_sink
        if sink is not None:
            sink(str(message))

    def _record(self, game_id: str, level: int) -> GameLevelLearningRecord:
        key = (str(game_id), max(1, int(level)))
        row = self._records.get(key)
        if row is None:
            row = GameLevelLearningRecord()
            self._records[key] = row
        return row

    def record_priority_signals(
        self,
        game_id: str,
        *,
        competence_improvement: float | None = None,
        prediction_error: float | None = None,
        contradiction: float | None = None,
        novelty: float | None = None,
        transfer_opportunity: float | None = None,
    ) -> None:
        with self._lock:
            row = self._signals.setdefault(str(game_id), GamePrioritySignals())
            for name, value in (
                ("competence_improvement", competence_improvement),
                ("prediction_error", prediction_error),
                ("contradiction", contradiction),
                ("novelty", novelty),
                ("transfer_opportunity", transfer_opportunity),
            ):
                if value is not None:
                    setattr(row, name, max(0.5, min(2.0, float(value))))

    def observe_frontier_candidate(
        self,
        scope: FrontierScope,
        candidate: FrontierCandidate,
        *,
        terminal_state: str,
        generation: int,
    ) -> bool:
        with self._lock:
            self.register_games((scope.game_id,))
            changed, version = self.frontier.add(scope, candidate)
            record = self._record(scope.game_id, scope.level)
            previous_state = record.state
            if record.first_success_generation <= 0:
                record.first_success_generation = max(1, int(generation))
            record.last_success_generation = max(1, int(generation))
            if record.state == GameLearningState.UNSOLVED:
                record.state = GameLearningState.SOLVED_OPTIMIZING
            if str(terminal_state) == "WIN":
                self._game_won[scope.game_id] = True
            if changed:
                prior_best = None
                prior_rows = self.frontier.candidates(scope)
                if prior_rows:
                    prior_best = min(int(row.cost) for row in prior_rows)
                record.frontier_version = int(version)
                record.optimizer_exhausted_version = -1
                record.validations_since_improvement = 0
                record.last_frontier_improvement_generation = max(1, int(generation))
                if previous_state == GameLearningState.SOLVED_STABLE:
                    record.state = GameLearningState.SOLVED_OPTIMIZING
                signals = self._signals.setdefault(scope.game_id, GamePrioritySignals())
                signals.competence_improvement = min(1.5, signals.competence_improvement + 0.10)
                signals.novelty = min(1.35, signals.novelty + 0.05)
                self._emit(
                    f"frontier game={scope.game_id} level={scope.level} "
                    f"source={candidate.source.value} cost={candidate.cost} "
                    f"reliability={candidate.reliability:.2f} version={version}"
                )
            if previous_state != record.state:
                self._emit(
                    f"learning state game={scope.game_id} level={scope.level} "
                    f"{previous_state.value}->{record.state.value}"
                )
            return bool(changed)

    def reserve_optimization(
        self,
        *,
        game_id: str,
        level: int,
        attempts: int,
    ) -> bool:
        with self._lock:
            row = self._record(game_id, level)
            requested = max(1, int(attempts))
            if (
                row.consumed_optimization_budget + requested
                > int(self.config.optimization_validation_budget)
            ):
                row.optimizer_exhausted_version = int(row.frontier_version)
                return False
            if (
                row.validations_since_improvement
                >= int(self.config.max_validations_without_improvement)
            ):
                row.optimizer_exhausted_version = int(row.frontier_version)
                return False
            row.consumed_optimization_budget += requested
            run = self._run.setdefault(str(game_id), GameRunTelemetry())
            run.optimizer_candidates += 1
            return True

    def record_optimizer_validation(
        self,
        *,
        game_id: str,
        level: int,
        attempts: int,
        successes: int,
        saved_actions: int,
        improved: bool,
        generation: int,
    ) -> None:
        with self._lock:
            row = self._record(game_id, level)
            run = self._run.setdefault(str(game_id), GameRunTelemetry())
            attempts = max(1, int(attempts))
            successes = max(0, int(successes))
            run.optimizer_validations += attempts
            run.optimizer_successes += successes
            run.optimizer_saved_actions += max(0, int(saved_actions)) if improved else 0
            row.optimization_rounds = max(row.optimization_rounds, 1)
            if improved and int(saved_actions) >= int(self.config.min_meaningful_improvement):
                row.validations_since_improvement = 0
                row.last_frontier_improvement_generation = max(1, int(generation))
            else:
                row.validations_since_improvement += attempts
                if row.validations_since_improvement >= int(
                    self.config.max_validations_without_improvement
                ):
                    row.optimizer_exhausted_version = int(row.frontier_version)
            signals = self._signals.setdefault(str(game_id), GamePrioritySignals())
            if successes <= 0:
                signals.contradiction = min(1.35, signals.contradiction + 0.02)
            else:
                signals.contradiction = max(1.0, signals.contradiction * 0.98)

    def mark_optimizer_idle(self, *, generation: int) -> None:
        with self._lock:
            for row in self._records.values():
                if row.state == GameLearningState.SOLVED_OPTIMIZING:
                    row.optimizer_exhausted_version = int(row.frontier_version)
            self.stabilize(generation=generation)

    def stabilize(self, *, generation: int) -> None:
        with self._lock:
            for (game_id, level), row in self._records.items():
                if row.state != GameLearningState.SOLVED_OPTIMIZING:
                    continue
                if int(row.optimizer_exhausted_version) != int(row.frontier_version):
                    continue
                if (
                    int(generation) - int(row.last_frontier_improvement_generation)
                    < int(self.config.stabilization_generations)
                ):
                    continue
                row.state = GameLearningState.SOLVED_STABLE
                self._emit(
                    f"learning state game={game_id} level={level} "
                    f"SOLVED_OPTIMIZING->SOLVED_STABLE"
                )

    def game_state(self, game_id: str) -> GameLearningState:
        with self._lock:
            game = str(game_id)
            if not bool(self._game_won.get(game, False)):
                return GameLearningState.UNSOLVED
            rows = [row for (owner, _level), row in self._records.items() if owner == game]
            if not rows:
                return GameLearningState.UNSOLVED
            if any(row.state == GameLearningState.SOLVED_OPTIMIZING for row in rows):
                return GameLearningState.SOLVED_OPTIMIZING
            return GameLearningState.SOLVED_STABLE

    def sampling_weight(self, game_id: str) -> float:
        state = self.game_state(game_id)
        base = {
            GameLearningState.UNSOLVED: float(self.config.unsolved_weight),
            GameLearningState.SOLVED_OPTIMIZING: float(self.config.optimizing_weight),
            GameLearningState.SOLVED_STABLE: float(self.config.stable_weight),
        }[state]
        with self._lock:
            signals = self._signals.setdefault(str(game_id), GamePrioritySignals())
            return max(1e-9, base * signals.multiplier)

    def choose_game(self, games: Iterable[str]) -> str:
        candidates = tuple(dict.fromkeys(str(game) for game in games))
        if not candidates:
            raise ValueError("adaptive allocator requires at least one game")
        self.register_games(candidates)
        with self._lock:
            # Weighted-fair allocation: choose the game with the smallest used/weight
            # ratio.  This is deterministic, avoids RNG identity, and continuously
            # redirects remaining credits as learning state changes.
            return min(
                candidates,
                key=lambda game: (
                    float(self._run[game].sample_steps)
                    / max(1e-9, float(self.sampling_weight(game))),
                    self._run[game].leases,
                    game,
                ),
            )

    def choose_mode(self, game_id: str) -> SamplingMode:
        game = str(game_id)
        state = self.game_state(game)
        if state == GameLearningState.UNSOLVED:
            return SamplingMode.DISCOVERY
        with self._lock:
            cursor = int(self._mode_cursor.get(game, 0))
            self._mode_cursor[game] = cursor + 1
        if state == GameLearningState.SOLVED_OPTIMIZING:
            cycle = (
                SamplingMode.VERIFY,
                SamplingMode.ALTERNATIVE,
                SamplingMode.VERIFY,
                SamplingMode.TRANSFER,
                SamplingMode.ALTERNATIVE,
            )
        else:
            cycle = (
                SamplingMode.ALTERNATIVE,
                SamplingMode.TRANSFER,
                SamplingMode.VERIFY,
                SamplingMode.ALTERNATIVE,
            )
        return cycle[cursor % len(cycle)]

    def recommended_lease_steps(self, game_id: str, remaining: int) -> int:
        base = max(64, int(self.config.lease_steps))
        best = self.frontier.best_for_game(str(game_id))
        if best is not None:
            _scope, winner = best
            base = max(base, int(math.ceil(max(1, winner.cost) * 1.25)))
        return max(1, min(int(remaining), int(base)))

    def alternative_exclusion(self, game_id: str) -> MemoryUid:
        best = self.frontier.best_for_game(str(game_id))
        return MemoryUid.zero() if best is None else best[1].strategy_uid

    def record_lease(self, game_id: str, mode: SamplingMode, steps: int) -> None:
        with self._lock:
            row = self._run.setdefault(str(game_id), GameRunTelemetry())
            row.sample_steps += max(0, int(steps))
            row.leases += 1
            row.last_mode = mode
            if mode == SamplingMode.VERIFY:
                row.verify_attempts += 1
            elif mode == SamplingMode.ALTERNATIVE:
                row.alternative_attempts += 1
            elif mode == SamplingMode.TRANSFER:
                row.transfer_attempts += 1

    def total_sample_steps(self) -> int:
        with self._lock:
            return sum(int(row.sample_steps) for row in self._run.values())

    def telemetry(
        self,
        *,
        optimizer_service=None,
    ) -> tuple[GameLearningTelemetrySnapshot, ...]:
        with self._lock:
            games = tuple(sorted(self._games))
            total = max(1, self.total_sample_steps())
            result = []
            for game in games:
                run = self._run.setdefault(game, GameRunTelemetry())
                best = self.frontier.best_for_game(game)
                if best is None:
                    cost = version = 0
                    reliability = 0.0
                    source = ""
                else:
                    scope, winner = best
                    cost = int(winner.cost)
                    reliability = float(winner.reliability)
                    source = winner.source.value
                    version = self.frontier.version(scope)
                optimizer_active = False
                if optimizer_service is not None:
                    try:
                        with optimizer_service._v818_validator_lock:
                            q = optimizer_service._v818_game_queues.get(game)
                            thread = optimizer_service._v818_validator_threads.get(game)
                            optimizer_active = bool(
                                (q is not None and q.unfinished_tasks > 0)
                                or (thread is not None and thread.is_alive())
                            )
                    except BaseException:
                        optimizer_active = False
                result.append(
                    GameLearningTelemetrySnapshot(
                        game,
                        self.game_state(game).value,
                        int(run.sample_steps),
                        float(run.sample_steps) / float(total),
                        float(self.sampling_weight(game)),
                        run.last_mode.value,
                        cost,
                        reliability,
                        source,
                        version,
                        int(run.optimizer_candidates),
                        int(run.optimizer_validations),
                        int(run.optimizer_successes),
                        int(run.optimizer_saved_actions),
                        optimizer_active,
                        int(run.alternative_attempts),
                        int(run.transfer_attempts),
                    )
                )
            return tuple(result)

    def state_dict(self) -> dict[str, object]:
        with self._lock:
            return {
                "version": 1,
                "games": sorted(self._games),
                "game_won": {game: bool(value) for game, value in sorted(self._game_won.items())},
                "game_level_states": [
                    {
                        "game_id": game,
                        "level": int(level),
                        **record.to_dict(),
                    }
                    for (game, level), record in sorted(self._records.items())
                ],
                "frontier": self.frontier.state_dict(),
                "signals": {
                    game: asdict(signals)
                    for game, signals in sorted(self._signals.items())
                },
                "sampling_weight": {
                    game: float(self.sampling_weight(game)) for game in sorted(self._games)
                },
            }

    def load_state(self, raw: dict[str, object] | None) -> None:
        if not isinstance(raw, dict):
            return
        with self._lock:
            self.register_games(raw.get("games", ()))
            game_won = raw.get("game_won", {})
            if isinstance(game_won, dict):
                for game, value in game_won.items():
                    self._game_won[str(game)] = bool(value)
            self._records.clear()
            for item in raw.get("game_level_states", ()):
                if not isinstance(item, dict):
                    continue
                game = str(item.get("game_id", ""))
                level = max(1, int(item.get("level", 1)))
                if game:
                    self._records[(game, level)] = GameLevelLearningRecord.from_dict(item)
            frontier = raw.get("frontier")
            if isinstance(frontier, dict):
                self.frontier.load_state(frontier)
            signals_raw = raw.get("signals", {})
            if isinstance(signals_raw, dict):
                for game, values in signals_raw.items():
                    if not isinstance(values, dict):
                        continue
                    self._signals[str(game)] = GamePrioritySignals(
                        competence_improvement=float(values.get("competence_improvement", 1.0)),
                        prediction_error=float(values.get("prediction_error", 1.0)),
                        contradiction=float(values.get("contradiction", 1.0)),
                        novelty=float(values.get("novelty", 1.0)),
                        transfer_opportunity=float(values.get("transfer_opportunity", 1.0)),
                    )
            # Run allocation counters intentionally restart at zero.  Learning state,
            # frontier ownership and priority modifiers persist; active leases do not.
            self._run = {game: GameRunTelemetry() for game in self._games}
            self._mode_cursor.clear()


@dataclass(frozen=True, slots=True)
class ActorLease:
    lease_id: int
    worker_id: int
    game_id: str
    steps: int
    seed: int
    env_root: str | None
    epsilon: float
    graph_check_steps: int
    mode: SamplingMode
    excluded_strategy_uid: MemoryUid = MemoryUid(0, 0)


@dataclass(frozen=True, slots=True)
class _LeaseProgress:
    worker_id: int
    lease_id: int
    row: object


@dataclass(frozen=True, slots=True)
class _LeaseResult:
    worker_id: int
    lease: ActorLease
    result: object


class _ProgressAdapter:
    def __init__(self, target, worker_id: int, lease_id: int) -> None:
        self.target = target
        self.worker_id = int(worker_id)
        self.lease_id = int(lease_id)

    def put_nowait(self, row) -> None:
        self.target.put_nowait(_LeaseProgress(self.worker_id, self.lease_id, row))


class _ResultAdapter:
    def __init__(self, target, worker_id: int, lease: ActorLease) -> None:
        self.target = target
        self.worker_id = int(worker_id)
        self.lease = lease

    def put(self, row) -> None:
        self.target.put(_LeaseResult(self.worker_id, self.lease, row))


def _uid_env(uid: MemoryUid) -> str:
    return "" if uid.is_zero else f"{int(uid.hi)}:{int(uid.lo)}"


def _uid_from_env(raw: str | None) -> MemoryUid:
    if not raw:
        return MemoryUid.zero()
    try:
        hi, lo = str(raw).split(":", 1)
        return MemoryUid(int(hi), int(lo))
    except (TypeError, ValueError):
        return MemoryUid.zero()


def _persistent_lease_worker(
    *,
    worker_id: int,
    assignment_queue,
    event_queue,
    ready_event,
    experience_ring_args,
    read_descriptors,
    watermark,
    stop_event,
    actor_throttle,
    snapshot_freeze,
) -> None:
    from v8 import actor as actor_module

    ready_event.set()
    while not stop_event.is_set():
        try:
            lease = assignment_queue.get(timeout=0.10)
        except queue.Empty:
            continue
        if lease is None:
            return
        if not isinstance(lease, ActorLease):
            continue
        prior_mode = os.environ.get(_SAMPLING_MODE_ENV)
        prior_excluded = os.environ.get(_ALTERNATIVE_EXCLUDE_ENV)
        os.environ[_SAMPLING_MODE_ENV] = lease.mode.value
        os.environ[_ALTERNATIVE_EXCLUDE_ENV] = _uid_env(lease.excluded_strategy_uid)
        epsilon = float(lease.epsilon)
        if lease.mode == SamplingMode.VERIFY:
            epsilon = 0.0
        elif lease.mode == SamplingMode.ALTERNATIVE:
            epsilon = max(0.15, epsilon)
        elif lease.mode == SamplingMode.TRANSFER:
            epsilon = max(0.20, epsilon)
        job = actor_module.ActorJob(
            actor_id=int(worker_id),
            game_id=str(lease.game_id),
            steps=int(lease.steps),
            seed=int(lease.seed),
            env_root=lease.env_root,
            epsilon=epsilon,
            graph_check_steps=int(lease.graph_check_steps),
        )
        try:
            actor_module.actor_worker(
                job=job,
                experience_ring_args=experience_ring_args,
                read_descriptors=read_descriptors,
                watermark=watermark,
                stop_event=stop_event,
                result_queue=_ResultAdapter(event_queue, worker_id, lease),
                progress_queue=_ProgressAdapter(event_queue, worker_id, lease.lease_id),
                reporting_queue=None,
                actor_throttle=actor_throttle,
                snapshot_freeze=snapshot_freeze,
                startup_ready=None,
                startup_gate=None,
            )
        finally:
            if prior_mode is None:
                os.environ.pop(_SAMPLING_MODE_ENV, None)
            else:
                os.environ[_SAMPLING_MODE_ENV] = prior_mode
            if prior_excluded is None:
                os.environ.pop(_ALTERNATIVE_EXCLUDE_ENV, None)
            else:
                os.environ[_ALTERNATIVE_EXCLUDE_ENV] = prior_excluded


def _optimizer_idle(service) -> bool:
    if service is None:
        return True
    try:
        with service._lock:
            active = int(service._active_validations)
        with service._v818_validator_lock:
            queues = tuple(service._v818_game_queues.values())
        return bool(
            service._sources.unfinished_tasks == 0
            and all(q.unfinished_tasks == 0 for q in queues)
            and active == 0
        )
    except BaseException:
        return False


def _write_allocation_log(runtime, coordinator: AdaptiveLearningCoordinator) -> None:
    service = getattr(runtime, "_v814_trajectory_optimizer", None)
    payload = {
        "time": time.time(),
        "generation": int(getattr(runtime, "generation", 0)),
        "watermark": int(getattr(runtime, "watermark", 0)),
        "games": [asdict(row) for row in coordinator.telemetry(optimizer_service=service)],
    }
    target = Path(runtime.root) / "sampling_allocation.log"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _allocation_stdout(coordinator: AdaptiveLearningCoordinator) -> None:
    rows = coordinator.telemetry()
    counts = {state.value: 0 for state in GameLearningState}
    for row in rows:
        counts[row.state] = counts.get(row.state, 0) + 1
    leaders = sorted(rows, key=lambda row: (-row.sample_share, row.game_id))[:5]
    shares = ",".join(f"{row.game_id}:{100.0 * row.sample_share:.0f}%" for row in leaders)
    print(
        f'[{time.strftime("%H:%M")}] sampling allocation '
        f"unsolved={counts.get(GameLearningState.UNSOLVED.value, 0)} "
        f"optimizing={counts.get(GameLearningState.SOLVED_OPTIMIZING.value, 0)} "
        f"stable={counts.get(GameLearningState.SOLVED_STABLE.value, 0)} "
        f"steps={coordinator.total_sample_steps()} top={shares}",
        flush=True,
    )


def _adaptive_progress_rows(
    actor_module,
    jobs,
    completed_by_game,
    active_progress,
    active_leases,
):
    totals: dict[str, dict[str, int]] = {
        game: dict(values) for game, values in completed_by_game.items()
    }
    for worker_id, progress in active_progress.items():
        lease = active_leases.get(worker_id)
        if lease is None or progress is None:
            continue
        bucket = totals.setdefault(
            lease.game_id,
            {
                "steps": 0,
                "wins": 0,
                "failures": 0,
                "levels_completed": 0,
                "replans": 0,
                "planned_steps": 0,
                "first_win_step": 0,
                "resets": 0,
            },
        )
        base_steps = int(bucket["steps"])
        bucket["steps"] += int(getattr(progress, "steps", 0))
        bucket["wins"] += int(getattr(progress, "wins", 0))
        bucket["failures"] += int(getattr(progress, "failures", 0))
        bucket["levels_completed"] += int(getattr(progress, "levels_completed", 0))
        bucket["replans"] += int(getattr(progress, "replans", 0))
        bucket["planned_steps"] += int(getattr(progress, "planned_steps", 0))
        local_first = int(getattr(progress, "first_win_step", 0) or 0)
        if bucket["first_win_step"] <= 0 and local_first > 0:
            bucket["first_win_step"] = base_steps + local_first

    first_job: dict[str, object] = {}
    for job in jobs:
        first_job.setdefault(str(job.game_id), job)
    rows = []
    for job in jobs:
        game = str(job.game_id)
        values = totals.get(
            game,
            {
                "steps": 0,
                "wins": 0,
                "failures": 0,
                "levels_completed": 0,
                "replans": 0,
                "planned_steps": 0,
                "first_win_step": 0,
            },
        )
        if first_job[game] is not job:
            values = {key: 0 for key in values}
        kwargs = dict(
            actor_id=int(job.actor_id),
            game_id=game,
            steps=int(values.get("steps", 0)),
            wins=int(values.get("wins", 0)),
            failures=int(values.get("failures", 0)),
            levels_completed=int(values.get("levels_completed", 0)),
            replans=int(values.get("replans", 0)),
            planned_steps=int(values.get("planned_steps", 0)),
        )
        try:
            row = actor_module.ActorProgress(
                **kwargs,
                first_win_step=int(values.get("first_win_step", 0)),
            )
        except TypeError:
            row = actor_module.ActorProgress(**kwargs)
        rows.append(row)
    return tuple(rows)


def _adaptive_run_actor_jobs(
    runtime,
    jobs: Iterable[object],
    *,
    timeout: float | None = None,
    progress_interval_seconds: float = 60.0,
    progress_callback=None,
    reporting_queue=None,
):
    from v8 import actor as actor_module

    jobs = tuple(jobs)
    coordinator = getattr(runtime, "_v819_adaptive_learning", None)
    if coordinator is None or not jobs:
        return _BASE_RUN_ACTOR_JOBS(
            runtime,
            jobs,
            timeout=timeout,
            progress_interval_seconds=progress_interval_seconds,
            progress_callback=progress_callback,
            reporting_queue=reporting_queue,
        )
    if progress_interval_seconds <= 0:
        raise ValueError("progress_interval_seconds must be positive")

    games = tuple(dict.fromkeys(str(job.game_id) for job in jobs))
    coordinator.register_games(games)
    total_budget = sum(max(0, int(job.steps)) for job in jobs)
    if total_budget <= 0:
        return ()
    worker_count = max(1, len(jobs))
    template = jobs[0]
    peers = getattr(runtime, "peers", None)
    if peers is not None:
        peers.pause()
    runtime.start()
    ctx = runtime._mp_ctx
    if peers is not None:
        startup_timeout = 300.0 if timeout is None else max(0.01, min(float(timeout), 300.0))
        if not peers.wait_idle(startup_timeout):
            raise TimeoutError("v8 peers did not pause before adaptive actor startup")
        runtime.wait_quiescent(
            timeout=startup_timeout,
            resume_peers=False,
            settle_peers=False,
        )

    assignments = [ctx.Queue(maxsize=2) for _ in range(worker_count)]
    events = ctx.Queue(maxsize=max(64, worker_count * 16))
    ready = [ctx.Event() for _ in range(worker_count)]
    processes = [
        ctx.Process(
            target=_persistent_lease_worker,
            kwargs={
                "worker_id": index + 1,
                "assignment_queue": assignments[index],
                "event_queue": events,
                "ready_event": ready[index],
                "experience_ring_args": runtime._stage_rings[0].attachment_args(),
                "read_descriptors": runtime.shard_descriptors,
                "watermark": runtime._watermark,
                "stop_event": runtime._stop,
                "actor_throttle": runtime._actor_throttle,
                "snapshot_freeze": runtime._snapshot_freeze,
            },
            name=f"v8-adaptive-actor-{index + 1:03d}",
            daemon=True,
        )
        for index in range(worker_count)
    ]

    started = time.monotonic()
    deadline = None if timeout is None else started + float(timeout)
    next_progress = started + float(progress_interval_seconds)
    next_log = started + _ALLOCATION_LOG_SECONDS
    next_stdout = started + _ALLOCATION_STDOUT_SECONDS
    reserved = 0
    consumed = 0
    lease_id = 0
    active_leases: dict[int, ActorLease] = {}
    active_progress: dict[int, object] = {}
    completed_by_game: dict[str, dict[str, int]] = {}
    no_progress_retries = 0

    def bucket(game: str) -> dict[str, int]:
        return completed_by_game.setdefault(
            game,
            {
                "steps": 0,
                "wins": 0,
                "failures": 0,
                "levels_completed": 0,
                "replans": 0,
                "planned_steps": 0,
                "first_win_step": 0,
                "resets": 0,
            },
        )

    def assign(worker_id: int) -> bool:
        nonlocal reserved, lease_id
        available = total_budget - reserved
        if available <= 0:
            return False
        game = coordinator.choose_game(games)
        mode = coordinator.choose_mode(game)
        steps = coordinator.recommended_lease_steps(game, available)
        lease_id += 1
        excluded = (
            coordinator.alternative_exclusion(game)
            if mode == SamplingMode.ALTERNATIVE
            else MemoryUid.zero()
        )
        lease = ActorLease(
            lease_id,
            worker_id,
            game,
            steps,
            int(getattr(template, "seed", 0)) + lease_id * 7919 + worker_id * 1009,
            getattr(template, "env_root", None),
            float(getattr(template, "epsilon", 0.10)),
            int(getattr(template, "graph_check_steps", 1000)),
            mode,
            excluded,
        )
        reserved += steps
        active_leases[worker_id] = lease
        active_progress[worker_id] = None
        assignments[worker_id - 1].put(lease)
        return True

    try:
        for process in processes:
            process.start()
        while not all(event.is_set() for event in ready):
            failed = [process for process in processes if process.exitcode not in (None, 0)]
            if failed:
                detail = ", ".join(f"{p.name}={p.exitcode}" for p in failed)
                raise RuntimeError(f"adaptive actor failed during startup: {detail}")
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("adaptive actor startup timed out")
            time.sleep(0.01)
        if peers is not None:
            peers.resume()
        for worker_id in range(1, worker_count + 1):
            if not assign(worker_id):
                break

        while consumed < total_budget or active_leases:
            try:
                event = events.get(timeout=0.05)
            except queue.Empty:
                event = None

            if isinstance(event, _LeaseProgress):
                row = event.row
                if isinstance(row, actor_module.ActorLearningBatch):
                    runtime.record_actor_results((row,))
                elif isinstance(row, actor_module.ActorProgress):
                    current = active_leases.get(int(event.worker_id))
                    if current is not None and int(current.lease_id) == int(event.lease_id):
                        active_progress[int(event.worker_id)] = row
            elif isinstance(event, _LeaseResult):
                worker_id = int(event.worker_id)
                lease = event.lease
                result = event.result
                active_leases.pop(worker_id, None)
                active_progress.pop(worker_id, None)
                actual = max(0, int(getattr(result, "steps", 0)))
                shortfall = max(0, int(lease.steps) - actual)
                if shortfall:
                    reserved -= shortfall
                consumed += actual
                coordinator.record_lease(lease.game_id, lease.mode, actual)
                values = bucket(lease.game_id)
                prior_steps = int(values["steps"])
                values["steps"] += actual
                values["wins"] += int(getattr(result, "wins", 0))
                values["failures"] += int(getattr(result, "failures", 0))
                values["levels_completed"] += int(getattr(result, "levels_completed", 0))
                values["replans"] += int(getattr(result, "replans", 0))
                values["planned_steps"] += int(getattr(result, "planned_steps", 0))
                values["resets"] += int(getattr(result, "resets", 0))
                pending = getattr(result, "pending_learning", None)
                if pending is not None:
                    runtime.record_actor_results((pending,))
                if int(getattr(result, "wins", 0)) > 0 and values["first_win_step"] <= 0:
                    # ActorResult does not carry first_win_step; the final progress
                    # row normally did, but if it was dropped use the lease end.
                    values["first_win_step"] = prior_steps + max(1, actual)
                if actual <= 0:
                    no_progress_retries += 1
                else:
                    no_progress_retries = 0
                if no_progress_retries > max(32, worker_count * 4):
                    raise RuntimeError("adaptive allocator could not consume interaction credits")
                assign(worker_id)

            retry = getattr(runtime, "_v819_retry_deferred", None)
            if retry is not None:
                retry()
            service = getattr(runtime, "_v814_trajectory_optimizer", None)
            if _optimizer_idle(service):
                coordinator.mark_optimizer_idle(generation=int(runtime.generation))
            else:
                coordinator.stabilize(generation=int(runtime.generation))

            failed = [process for process in processes if process.exitcode not in (None, 0)]
            if failed:
                detail = ", ".join(f"{p.name}={p.exitcode}" for p in failed)
                raise RuntimeError(f"adaptive actor failed: {detail}")
            now = time.monotonic()
            rows = None
            if now >= next_progress:
                rows = _adaptive_progress_rows(
                    actor_module,
                    jobs,
                    completed_by_game,
                    active_progress,
                    active_leases,
                )
                if reporting_queue is not None:
                    for row in rows:
                        try:
                            reporting_queue.put_nowait(row)
                        except queue.Full:
                            break
                if progress_callback is not None:
                    progress_callback(rows)
                while next_progress <= now:
                    next_progress += float(progress_interval_seconds)
            if now >= next_log:
                _write_allocation_log(runtime, coordinator)
                next_log = now + _ALLOCATION_LOG_SECONDS
            if now >= next_stdout:
                _allocation_stdout(coordinator)
                next_stdout = now + _ALLOCATION_STDOUT_SECONDS
            if deadline is not None and now >= deadline:
                raise TimeoutError("adaptive actor jobs timed out")

        final_rows = _adaptive_progress_rows(
            actor_module,
            jobs,
            completed_by_game,
            {},
            {},
        )
        if reporting_queue is not None:
            for row in final_rows:
                try:
                    reporting_queue.put_nowait(row)
                except queue.Full:
                    break
        if progress_callback is not None:
            progress_callback(final_rows)
        _write_allocation_log(runtime, coordinator)

        first_job: dict[str, object] = {}
        for job in jobs:
            first_job.setdefault(str(job.game_id), job)
        results = []
        for job in jobs:
            game = str(job.game_id)
            values = completed_by_game.get(game, {})
            if first_job[game] is not job:
                values = {}
            results.append(
                actor_module.ActorResult(
                    int(job.actor_id),
                    game,
                    int(values.get("steps", 0)),
                    int(values.get("wins", 0)),
                    int(values.get("failures", 0)),
                    int(values.get("levels_completed", 0)),
                    int(values.get("resets", 0)),
                    int(values.get("replans", 0)),
                    int(values.get("planned_steps", 0)),
                    (),
                    (),
                    (),
                    None,
                )
            )
        return tuple(results)
    except BaseException:
        for q in assignments:
            try:
                q.put_nowait(None)
            except BaseException:
                pass
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=2.0)
        raise
    finally:
        for q in assignments:
            try:
                q.put_nowait(None)
            except BaseException:
                pass
        for process in processes:
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        for q in assignments:
            q.cancel_join_thread()
            q.close()
        events.cancel_join_thread()
        events.close()


def _scope_from_validation(candidate, result, target_uid: MemoryUid) -> FrontierScope:
    context = int(getattr(result, "terminal_context", 0))
    context_bucket = stable_u64(context, person=b"v8-context") if context else 0
    return FrontierScope(
        str(candidate.source.anchor.source_id),
        max(1, int(candidate.source.target.levels_completed)),
        int(context_bucket),
        int(target_uid.hi),
        int(target_uid.lo),
    )


def _frontier_candidate_from_validation(
    optimizer,
    candidate,
    result,
    *,
    target_uid: MemoryUid,
    source: FrontierSource,
    generation: int,
    strategy_uid: MemoryUid | None = None,
) -> FrontierCandidate:
    resolved_source = replace(candidate.source, target_outcome_uid=target_uid)
    resolved_candidate = replace(candidate, source=resolved_source)
    uid = strategy_uid or optimizer.variant_strategy_uid(resolved_candidate)
    return FrontierCandidate(
        uid,
        str(candidate.source.trajectory_id),
        int(optimizer.action_sequence_hash(candidate.actions)),
        int(candidate.cost),
        max(1, int(getattr(result, "attempts", 1))),
        max(0, int(getattr(result, "successes", int(bool(getattr(result, "success", False)))))),
        int(ValidationState.TESTED),
        source,
        int(generation),
        candidate.source.parent_strategy_uid,
    )


def _source_validation_candidate(optimizer, source):
    return optimizer.TrajectoryCandidate(
        f"source-{source.trajectory_id}",
        source,
        "VALIDATE_SOURCE",
        tuple(source.actions),
        0,
        0,
        0,
        0,
    )


def _service_submit_v819(service, trajectory) -> bool:
    from v8 import trajectory_optimizer_v814 as optimizer
    from v8 import trajectory_optimizer_v818 as v818

    if int(getattr(trajectory, "round_index", 0)) != 0:
        runtime = getattr(service, "_v819_runtime", None)
        if runtime is not None and not trajectory.target_outcome_uid.is_zero:
            coord = runtime._v819_adaptive_learning
            level = max(1, int(trajectory.target.levels_completed))
            record = coord._record(trajectory.anchor.source_id, level)
            if (
                record.consumed_optimization_budget >= coord.config.optimization_validation_budget
                or record.validations_since_improvement >= coord.config.max_validations_without_improvement
            ):
                record.optimizer_exhausted_version = record.frontier_version
                return False
        return _BASE_SERVICE_SUBMIT(service, trajectory)

    with service._v819_lock:
        if trajectory.trajectory_id in service._v819_source_seen:
            return False
        service._v819_source_seen.add(trajectory.trajectory_id)
        service._v819_source_pending[trajectory.trajectory_id] = trajectory
        source_kind = _SOURCE_KIND_BY_TRAJECTORY.pop(
            trajectory.trajectory_id,
            FrontierSource.SAMPLER.value,
        )
        service._v819_source_kind[trajectory.trajectory_id] = str(source_kind)
    candidate = _source_validation_candidate(optimizer, trajectory)
    routed = bool(v818._route_candidate(service, candidate))
    if not routed:
        with service._v819_lock:
            service._v819_source_seen.discard(trajectory.trajectory_id)
            service._v819_source_pending.pop(trajectory.trajectory_id, None)
            service._v819_source_kind.pop(trajectory.trajectory_id, None)
    return routed


def _route_candidate_v819(service, candidate) -> bool:
    if str(getattr(candidate, "edit_kind", "")) == "VALIDATE_SOURCE":
        return _BASE_ROUTE_CANDIDATE(service, candidate)
    runtime = getattr(service, "_v819_runtime", None)
    if runtime is None:
        return _BASE_ROUTE_CANDIDATE(service, candidate)
    coord = runtime._v819_adaptive_learning
    attempts = 2
    try:
        from v8 import trajectory_optimizer_v818 as v818
        attempts = max(1, len(tuple(getattr(v818, "_VALIDATION_SEEDS", (0, 1)))))
    except BaseException:
        pass
    if not coord.reserve_optimization(
        game_id=str(candidate.source.anchor.source_id),
        level=max(1, int(candidate.source.target.levels_completed)),
        attempts=attempts,
    ):
        return False
    return _BASE_ROUTE_CANDIDATE(service, candidate)


def _validated_source_row(optimizer, candidate, result, target_uid: MemoryUid):
    resolved_source = replace(candidate.source, target_outcome_uid=target_uid)
    resolved_candidate = replace(candidate, source=resolved_source)
    strategy_uid = optimizer.variant_strategy_uid(resolved_candidate)
    return optimizer.ValidatedTrajectory(
        str(candidate.candidate_id),
        resolved_source.anchor,
        resolved_source.target,
        tuple(resolved_candidate.actions),
        strategy_uid,
        target_uid,
        resolved_source.parent_strategy_uid,
        int(resolved_source.cost),
        "VALIDATE_SOURCE",
        max(1, int(getattr(result, "attempts", 1))),
        max(0, int(getattr(result, "successes", 1))),
    )


def _publish_validated_source(runtime, candidate, result, target_uid: MemoryUid) -> None:
    from v8 import trajectory_optimizer_v814 as optimizer
    from v8 import trajectory_optimizer_v818 as v818

    service = runtime._v814_trajectory_optimizer
    source_kind_raw = service._v819_source_kind.get(
        candidate.source.trajectory_id,
        FrontierSource.SAMPLER.value,
    )
    source_kind = (
        FrontierSource.TRANSFER
        if source_kind_raw == FrontierSource.TRANSFER.value
        else FrontierSource.SAMPLER
    )
    row = _validated_source_row(optimizer, candidate, result, target_uid)
    resolved_source = replace(
        candidate.source,
        parent_strategy_uid=row.strategy_uid,
        target_outcome_uid=target_uid,
    )
    resolved_candidate = replace(candidate, source=resolved_source)
    v818._publish_resolved_validation(runtime, resolved_candidate, result, row, target_uid)
    scope = _scope_from_validation(resolved_candidate, result, target_uid)
    frontier_row = _frontier_candidate_from_validation(
        optimizer,
        resolved_candidate,
        result,
        target_uid=target_uid,
        source=source_kind,
        generation=int(runtime.generation),
        strategy_uid=row.strategy_uid,
    )
    runtime._v819_adaptive_learning.observe_frontier_candidate(
        scope,
        frontier_row,
        terminal_state=str(candidate.source.target.terminal_state),
        generation=int(runtime.generation),
    )
    with service._v819_lock:
        service._v819_source_pending.pop(candidate.source.trajectory_id, None)
    _BASE_SERVICE_SUBMIT(service, resolved_source)


def _retry_deferred_sources(runtime) -> None:
    from v8 import trajectory_optimizer_v818 as v818

    pending = list(getattr(runtime, "_v819_deferred_sources", ()))
    if not pending:
        return
    remaining = []
    for candidate, result in pending:
        target_uid = v818._resolve_target_outcome(runtime, candidate, result)
        if target_uid.is_zero:
            remaining.append((candidate, result))
            continue
        _publish_validated_source(runtime, candidate, result, target_uid)
    runtime._v819_deferred_sources = remaining


def _runtime_validation_callback_v819(runtime, candidate, result, validated) -> None:
    from v8 import trajectory_optimizer_v814 as optimizer
    from v8 import trajectory_optimizer_v818 as v818

    _retry_deferred_sources(runtime)
    if str(getattr(candidate, "edit_kind", "")) == "VALIDATE_SOURCE":
        if not bool(getattr(result, "success", False)):
            with runtime._v814_trajectory_optimizer._v819_lock:
                runtime._v814_trajectory_optimizer._v819_source_pending.pop(
                    candidate.source.trajectory_id,
                    None,
                )
            base = getattr(runtime, "_v819_base_validation_callback", None)
            if base is not None:
                base(candidate, result, None)
            return
        target_uid = v818._resolve_target_outcome(runtime, candidate, result)
        if target_uid.is_zero:
            runtime._v819_deferred_sources.append((candidate, result))
            return
        _publish_validated_source(runtime, candidate, result, target_uid)
        return

    base = getattr(runtime, "_v819_base_validation_callback", None)
    if base is not None:
        base(candidate, result, validated)

    coord = runtime._v819_adaptive_learning
    attempts = max(1, int(getattr(result, "attempts", 1)))
    successes = max(0, int(getattr(result, "successes", int(bool(getattr(result, "success", False))))))
    improved = False
    saved = 0
    if validated is not None and bool(getattr(result, "success", False)):
        target_uid = validated.target_outcome_uid
        if target_uid.is_zero:
            target_uid = v818._resolve_target_outcome(runtime, candidate, result)
        if not target_uid.is_zero:
            resolved_source = replace(candidate.source, target_outcome_uid=target_uid)
            resolved_candidate = replace(candidate, source=resolved_source)
            scope = _scope_from_validation(resolved_candidate, result, target_uid)
            row = _frontier_candidate_from_validation(
                optimizer,
                resolved_candidate,
                result,
                target_uid=target_uid,
                source=FrontierSource.TRAJECTORY_OPTIMIZER,
                generation=int(runtime.generation),
                strategy_uid=validated.strategy_uid,
            )
            improved = coord.observe_frontier_candidate(
                scope,
                row,
                terminal_state=str(candidate.source.target.terminal_state),
                generation=int(runtime.generation),
            )
            saved = max(0, int(candidate.source.cost) - int(candidate.cost))
    coord.record_optimizer_validation(
        game_id=str(candidate.source.anchor.source_id),
        level=max(1, int(candidate.source.target.levels_completed)),
        attempts=attempts,
        successes=successes,
        saved_actions=saved,
        improved=improved,
        generation=int(runtime.generation),
    )


def _success_to_dict_v819(self):
    raw = _BASE_SUCCESS_TO_DICT(self)
    mode = str(os.environ.get(_SAMPLING_MODE_ENV, SamplingMode.DISCOVERY.value))
    raw["frontier_source"] = (
        FrontierSource.TRANSFER.value
        if mode == SamplingMode.TRANSFER.value
        else FrontierSource.SAMPLER.value
    )
    raw["sampling_mode"] = mode
    return raw


def _success_from_dict_v819(cls, raw):
    row = _BASE_SUCCESS_FROM_DICT(raw)
    source = str(raw.get("frontier_source", FrontierSource.SAMPLER.value))
    _SOURCE_KIND_BY_TRAJECTORY[row.trajectory_id] = source
    return row


def _plan_candidates_v819(self, context_signature, action_ids, **kwargs):
    mode_raw = str(os.environ.get(_SAMPLING_MODE_ENV, SamplingMode.DISCOVERY.value))
    try:
        mode = SamplingMode(mode_raw)
    except ValueError:
        mode = SamplingMode.DISCOVERY
    if mode not in {SamplingMode.ALTERNATIVE, SamplingMode.TRANSFER}:
        return _BASE_PLAN_CANDIDATES(self, context_signature, action_ids, **kwargs)

    # Alternative/transfer probes deliberately bypass optimized sidecar playback.
    # They retain the normal context-sensitive canonical M7/M1 policy.  This is
    # the escape hatch that prevents an optimized trajectory from monopolizing
    # solved-game sampling.
    from v8 import trajectory_optimizer_v814 as optimizer

    call_kwargs = dict(kwargs)
    if mode == SamplingMode.ALTERNATIVE:
        excluded = set(call_kwargs.get("excluded_strategies", frozenset()))
        uid = _uid_from_env(os.environ.get(_ALTERNATIVE_EXCLUDE_ENV))
        if not uid.is_zero:
            excluded.add(uid)
        call_kwargs["excluded_strategies"] = frozenset(excluded)
    if mode == SamplingMode.TRANSFER:
        call_kwargs["ignore_preference"] = True
    return optimizer._BASE_PLAN_CANDIDATES(
        self,
        context_signature,
        action_ids,
        **call_kwargs,
    )


def _runtime_init_v819(self, *args, **kwargs):
    _BASE_RUNTIME_INIT(self, *args, **kwargs)
    from v8.snapshot import load_latest_auxiliary_state

    def event_sink(message: str) -> None:
        print(f'[{time.strftime("%H:%M")}] {message}', flush=True)

    coordinator = AdaptiveLearningCoordinator(event_sink=event_sink)
    if bool(getattr(self.config, "restore", False)):
        state = load_latest_auxiliary_state(self.root)
        if isinstance(state, dict):
            adaptive = state.get("adaptive_learning")
            if isinstance(adaptive, dict):
                coordinator.load_state(adaptive)
    self._v819_adaptive_learning = coordinator
    self._v819_deferred_sources = []
    self._v819_retry_deferred = lambda: _retry_deferred_sources(self)
    service = getattr(self, "_v814_trajectory_optimizer", None)
    if service is not None:
        service._v819_runtime = self
        service._v819_lock = threading.RLock()
        service._v819_source_seen = set()
        service._v819_source_pending = {}
        service._v819_source_kind = {}
        self._v819_base_validation_callback = service.on_validation
        service.on_validation = lambda candidate, result, validated: _runtime_validation_callback_v819(
            self,
            candidate,
            result,
            validated,
        )


def _runtime_aux_v819(self) -> str:
    payload = json.loads(_BASE_RUNTIME_AUX(self))
    coordinator = getattr(self, "_v819_adaptive_learning", None)
    if coordinator is not None:
        payload["adaptive_learning"] = coordinator.state_dict()
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _runtime_metrics_v819(self) -> dict[str, object]:
    payload = _BASE_RUNTIME_METRICS(self)
    coordinator = getattr(self, "_v819_adaptive_learning", None)
    if coordinator is not None:
        rows = coordinator.telemetry(
            optimizer_service=getattr(self, "_v814_trajectory_optimizer", None)
        )
        payload["adaptive_learning"] = {
            "version": 1,
            "states": {
                state.value: sum(1 for row in rows if row.state == state.value)
                for state in GameLearningState
            },
            "sample_steps": coordinator.total_sample_steps(),
            "games": [asdict(row) for row in rows],
        }
    return payload


def install_adaptive_learning_allocation_v819() -> None:
    global _INSTALLED
    global _BASE_RUN_ACTOR_JOBS, _BASE_PLAN_CANDIDATES
    global _BASE_RUNTIME_INIT, _BASE_RUNTIME_AUX, _BASE_RUNTIME_METRICS
    global _BASE_SERVICE_SUBMIT, _BASE_SUCCESS_TO_DICT, _BASE_SUCCESS_FROM_DICT
    global _BASE_ROUTE_CANDIDATE
    if _INSTALLED:
        return

    from v8 import actor as actor_module
    from v8 import trajectory_optimizer_v814 as optimizer
    from v8 import trajectory_optimizer_v818 as v818
    from v8.publication import LiveReadView
    from v8.runtime_v82 import V82ContinuousMemoryRuntime

    _BASE_RUN_ACTOR_JOBS = actor_module.run_actor_jobs
    _BASE_PLAN_CANDIDATES = LiveReadView.plan_candidates
    _BASE_RUNTIME_INIT = V82ContinuousMemoryRuntime.__init__
    _BASE_RUNTIME_AUX = V82ContinuousMemoryRuntime._auxiliary_state_json
    _BASE_RUNTIME_METRICS = V82ContinuousMemoryRuntime.metrics
    _BASE_SERVICE_SUBMIT = optimizer.TrajectoryOptimizationService.submit_trajectory
    _BASE_SUCCESS_TO_DICT = optimizer.SuccessfulTrajectory.to_dict
    _BASE_SUCCESS_FROM_DICT = optimizer.SuccessfulTrajectory.from_dict
    _BASE_ROUTE_CANDIDATE = v818._route_candidate

    actor_module.run_actor_jobs = _adaptive_run_actor_jobs
    LiveReadView.plan_candidates = _plan_candidates_v819
    optimizer.TrajectoryOptimizationService.submit_trajectory = _service_submit_v819
    optimizer.SuccessfulTrajectory.to_dict = _success_to_dict_v819
    optimizer.SuccessfulTrajectory.from_dict = classmethod(_success_from_dict_v819)
    v818._route_candidate = _route_candidate_v819
    V82ContinuousMemoryRuntime.__init__ = _runtime_init_v819
    V82ContinuousMemoryRuntime._auxiliary_state_json = _runtime_aux_v819
    V82ContinuousMemoryRuntime.metrics = _runtime_metrics_v819
    V82ContinuousMemoryRuntime.scientific_semantics_version = "v8.19-adaptive-learning-allocation"
    _INSTALLED = True
