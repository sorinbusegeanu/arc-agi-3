from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from v8.model import (
    CognitiveState,
    EventId,
    MemoryLevel,
    MemoryProposal,
    MemoryType,
    MemoryUid,
    RelationType,
    ValidationState,
    proposal_fingerprint,
    stable_u64,
)


_INSTALLED = False
_TRAJECTORY_ROOT_ENV = "ARC_AGI3_V8_TRAJECTORY_ROOT"
_SEQUENCE_MARKER = 1 << 63
_SEQUENCE_MASK = _SEQUENCE_MARKER - 1
_BASE_ACTOR_WORKER = None
_BASE_ENV_STEP = None
_BASE_ENV_RESET = None
_BASE_VIEW_INIT = None
_BASE_PLAN_CANDIDATES = None
_BASE_RUNTIME_INIT = None
_BASE_RUNTIME_START = None
_BASE_RUNTIME_AUX = None
_BASE_RUNTIME_METRICS = None
_BASE_RUNTIME_ERRORS = None
_BASE_RUNTIME_CLOSE = None

_CAPTURE_ACTIVE = False
_CAPTURE_SOURCE_ID = ""
_CAPTURE_SEED = 0
_CAPTURE_ENV_ROOT: str | None = None
_CAPTURE_PREFIX: list[int] = []
_CAPTURE_SEGMENT: list[int] = []
_ACTOR_ACTION_HISTORY: list[int] = []
_ACTOR_RESET_EPOCH = 0
_CAPTURED_FOR_TESTS: list["SuccessfulTrajectory"] = []


@dataclass(frozen=True, slots=True)
class ReplayAnchor:
    source_id: str
    seed: int
    prefix_actions: tuple[int, ...] = ()
    env_root: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "seed": int(self.seed),
            "prefix_actions": list(self.prefix_actions),
            "env_root": self.env_root,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "ReplayAnchor":
        return cls(
            str(raw.get("source_id", "")),
            int(raw.get("seed", 0)),
            tuple(int(value) for value in raw.get("prefix_actions", ())),
            None if raw.get("env_root") is None else str(raw.get("env_root")),
        )


@dataclass(frozen=True, slots=True)
class TrajectoryTarget:
    levels_completed: int
    terminal_state: str

    def to_dict(self) -> dict[str, object]:
        return {
            "levels_completed": int(self.levels_completed),
            "terminal_state": str(self.terminal_state),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "TrajectoryTarget":
        return cls(
            int(raw.get("levels_completed", 0)),
            str(raw.get("terminal_state", "LEVEL")),
        )


def _uid_to_list(uid: MemoryUid) -> list[int]:
    return [int(uid.hi), int(uid.lo)]


def _uid_from_raw(raw) -> MemoryUid:
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return MemoryUid.zero()
    return MemoryUid(int(raw[0]), int(raw[1]))


def action_sequence_hash(actions: Iterable[int]) -> int:
    values = tuple(int(value) for value in actions)
    value = stable_u64(len(values), person=b"v8.14-actions")
    for index, action in enumerate(values):
        value = stable_u64(value, index, action, person=b"v8.14-actions")
    return int(value)


def _anchor_hash(anchor: ReplayAnchor, target: TrajectoryTarget) -> int:
    return stable_u64(
        anchor.source_id,
        int(anchor.seed),
        action_sequence_hash(anchor.prefix_actions),
        int(target.levels_completed),
        str(target.terminal_state),
        person=b"v8.14-anchor",
    )


def _trajectory_id(
    anchor: ReplayAnchor,
    target: TrajectoryTarget,
    actions: Iterable[int],
) -> str:
    value = stable_u64(
        _anchor_hash(anchor, target),
        action_sequence_hash(actions),
        person=b"v8.14-trajectory",
    )
    return f"{value:016x}"


@dataclass(frozen=True, slots=True)
class SuccessfulTrajectory:
    trajectory_id: str
    anchor: ReplayAnchor
    target: TrajectoryTarget
    actions: tuple[int, ...]
    parent_strategy_uid: MemoryUid = MemoryUid(0, 0)
    target_outcome_uid: MemoryUid = MemoryUid(0, 0)
    round_index: int = 0

    @property
    def cost(self) -> int:
        return len(self.actions)

    @property
    def full_actions(self) -> tuple[int, ...]:
        return (*self.anchor.prefix_actions, *self.actions)

    def to_dict(self) -> dict[str, object]:
        return {
            "trajectory_id": self.trajectory_id,
            "anchor": self.anchor.to_dict(),
            "target": self.target.to_dict(),
            "actions": list(self.actions),
            "parent_strategy_uid": _uid_to_list(self.parent_strategy_uid),
            "target_outcome_uid": _uid_to_list(self.target_outcome_uid),
            "round_index": int(self.round_index),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "SuccessfulTrajectory":
        anchor = ReplayAnchor.from_dict(dict(raw.get("anchor", {})))
        target = TrajectoryTarget.from_dict(dict(raw.get("target", {})))
        actions = tuple(int(value) for value in raw.get("actions", ()))
        trajectory_id = str(raw.get("trajectory_id", "")) or _trajectory_id(
            anchor, target, actions
        )
        return cls(
            trajectory_id,
            anchor,
            target,
            actions,
            _uid_from_raw(raw.get("parent_strategy_uid")),
            _uid_from_raw(raw.get("target_outcome_uid")),
            int(raw.get("round_index", 0)),
        )


@dataclass(frozen=True, slots=True)
class TrajectoryCandidate:
    candidate_id: str
    source: SuccessfulTrajectory
    edit_kind: str
    actions: tuple[int, ...]
    removed_start: int
    removed_length: int
    repeat_block_length: int = 0
    repeat_count: int = 0

    @property
    def cost(self) -> int:
        return len(self.actions)


@dataclass(frozen=True, slots=True)
class ValidatedTrajectory:
    variant_id: str
    anchor: ReplayAnchor
    target: TrajectoryTarget
    actions: tuple[int, ...]
    strategy_uid: MemoryUid
    target_outcome_uid: MemoryUid
    parent_strategy_uid: MemoryUid
    parent_cost: int
    edit_kind: str
    attempts: int = 1
    successes: int = 1

    @property
    def cost(self) -> int:
        return len(self.actions)

    @property
    def saved_actions(self) -> int:
        return max(0, int(self.parent_cost) - self.cost)

    def to_dict(self) -> dict[str, object]:
        return {
            "variant_id": self.variant_id,
            "anchor": self.anchor.to_dict(),
            "target": self.target.to_dict(),
            "actions": list(self.actions),
            "strategy_uid": _uid_to_list(self.strategy_uid),
            "target_outcome_uid": _uid_to_list(self.target_outcome_uid),
            "parent_strategy_uid": _uid_to_list(self.parent_strategy_uid),
            "parent_cost": int(self.parent_cost),
            "edit_kind": self.edit_kind,
            "attempts": int(self.attempts),
            "successes": int(self.successes),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "ValidatedTrajectory":
        return cls(
            str(raw.get("variant_id", "")),
            ReplayAnchor.from_dict(dict(raw.get("anchor", {}))),
            TrajectoryTarget.from_dict(dict(raw.get("target", {}))),
            tuple(int(value) for value in raw.get("actions", ())),
            _uid_from_raw(raw.get("strategy_uid")),
            _uid_from_raw(raw.get("target_outcome_uid")),
            _uid_from_raw(raw.get("parent_strategy_uid")),
            int(raw.get("parent_cost", 0)),
            str(raw.get("edit_kind", "")),
            int(raw.get("attempts", 1)),
            int(raw.get("successes", 1)),
        )


@dataclass(frozen=True, slots=True)
class TrajectoryOptimizerConfig:
    max_candidates_per_round: int = 48
    max_repeat_block_length: int = 8
    max_segment_delete: int = 4
    max_optimization_rounds: int = 8
    poll_interval_seconds: float = 0.10


@dataclass(frozen=True, slots=True)
class TrajectoryOptimizerMetrics:
    trajectories_seen: int
    candidates_generated: int
    validations: int
    validation_successes: int
    validated_variants: int


def _candidate_id(source: SuccessfulTrajectory, kind: str, actions: tuple[int, ...]) -> str:
    value = stable_u64(
        source.trajectory_id,
        kind,
        action_sequence_hash(actions),
        person=b"v8.14-candidate",
    )
    return f"{value:016x}"


def _bounded_positions(length: int, limit: int) -> tuple[int, ...]:
    n = max(0, int(length))
    cap = max(0, int(limit))
    if n <= cap:
        return tuple(range(n))
    if cap <= 0:
        return ()
    result = []
    for index in range(cap):
        position = min(n - 1, int(index * n / cap))
        if not result or position != result[-1]:
            result.append(position)
    return tuple(result)


def generate_optimization_candidates(
    source: SuccessfulTrajectory,
    config: TrajectoryOptimizerConfig | None = None,
) -> tuple[TrajectoryCandidate, ...]:
    cfg = config or TrajectoryOptimizerConfig()
    actions = tuple(int(value) for value in source.actions)
    n = len(actions)
    if n <= 1:
        return ()

    candidates: dict[tuple[int, ...], TrajectoryCandidate] = {}

    def add(
        kind: str,
        candidate_actions: tuple[int, ...],
        start: int,
        removed: int,
        *,
        block: int = 0,
        repeats: int = 0,
    ) -> None:
        if not candidate_actions or len(candidate_actions) >= n:
            return
        prior = candidates.get(candidate_actions)
        row = TrajectoryCandidate(
            _candidate_id(source, kind, candidate_actions),
            source,
            kind,
            candidate_actions,
            int(start),
            int(removed),
            int(block),
            int(repeats),
        )
        priority = {"REDUCE_REPEAT": 0, "DELETE_ACTION": 1, "DELETE_SEGMENT": 2}
        if prior is None or priority.get(kind, 9) < priority.get(prior.edit_kind, 9):
            candidates[candidate_actions] = row

    repeat_budget = max(8, cfg.max_candidates_per_round)
    repeats_found = 0
    for start in range(n - 1):
        if repeats_found >= repeat_budget * 2:
            break
        max_block = min(int(cfg.max_repeat_block_length), (n - start) // 2)
        for block_length in range(1, max_block + 1):
            block = actions[start : start + block_length]
            repeat_count = 1
            position = start + block_length
            while (
                position + block_length <= n
                and actions[position : position + block_length] == block
            ):
                repeat_count += 1
                position += block_length
            if repeat_count < 2:
                continue
            for keep_count in range(1, repeat_count):
                candidate_actions = (
                    actions[:start]
                    + block * keep_count
                    + actions[start + repeat_count * block_length :]
                )
                add(
                    "REDUCE_REPEAT",
                    candidate_actions,
                    start + keep_count * block_length,
                    (repeat_count - keep_count) * block_length,
                    block=block_length,
                    repeats=repeat_count,
                )
                repeats_found += 1
                if repeats_found >= repeat_budget * 2:
                    break
            if repeats_found >= repeat_budget * 2:
                break

    for index in _bounded_positions(n, max(8, cfg.max_candidates_per_round // 2)):
        add("DELETE_ACTION", actions[:index] + actions[index + 1 :], index, 1)

    segment_budget = max(8, cfg.max_candidates_per_round // 2)
    for length in range(2, min(int(cfg.max_segment_delete), n - 1) + 1):
        for start in _bounded_positions(n - length + 1, segment_budget):
            add(
                "DELETE_SEGMENT",
                actions[:start] + actions[start + length :],
                start,
                length,
            )

    priority = {"REDUCE_REPEAT": 0, "DELETE_ACTION": 1, "DELETE_SEGMENT": 2}
    ordered = sorted(
        candidates.values(),
        key=lambda row: (
            priority.get(row.edit_kind, 9),
            len(row.actions),
            row.removed_start,
            row.candidate_id,
        ),
    )
    return tuple(ordered[: max(0, int(cfg.max_candidates_per_round))])


def variant_strategy_key(candidate: TrajectoryCandidate) -> tuple[int, int, int, int]:
    sequence = _SEQUENCE_MARKER | (action_sequence_hash(candidate.actions) & _SEQUENCE_MASK)
    target = candidate.source.target_outcome_uid
    return (
        int(sequence),
        int(target.hi),
        int(target.lo),
        int(_anchor_hash(candidate.source.anchor, candidate.source.target)),
    )


def variant_strategy_uid(candidate: TrajectoryCandidate) -> MemoryUid:
    return MemoryUid.from_key(
        MemoryLevel.M7,
        MemoryType.STRATEGY,
        variant_strategy_key(candidate),
    )


def _frontier_key(anchor: ReplayAnchor, target: TrajectoryTarget) -> str:
    return f"{_anchor_hash(anchor, target):016x}"


def select_validated_variant(
    rows: Iterable[ValidatedTrajectory],
    *,
    source_id: str,
    seed: int,
    action_history: Iterable[int],
    attempted: set[str] | None = None,
) -> ValidatedTrajectory | None:
    history = tuple(int(value) for value in action_history)
    blocked = attempted or set()
    candidates = [
        row
        for row in rows
        if row.variant_id not in blocked
        and row.anchor.source_id == str(source_id)
        and int(row.anchor.seed) == int(seed)
        and tuple(row.anchor.prefix_actions) == history
        and row.actions
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (-row.saved_actions, row.cost, -row.successes, row.variant_id),
    )


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


class TrajectoryOptimizationService:
    def __init__(
        self,
        root: str | Path,
        *,
        validator: Callable[[TrajectoryCandidate], object],
        on_validation: Callable[[TrajectoryCandidate, object, ValidatedTrajectory | None], None]
        | None = None,
        config: TrajectoryOptimizerConfig | None = None,
    ) -> None:
        self.root = Path(root)
        self.inbox = self.root / "inbox"
        self.validated_path = self.root / "validated.json"
        self.log_path = self.root / "trajectory_optimizer.log"
        self.validator = validator
        self.on_validation = on_validation
        self.config = config or TrajectoryOptimizerConfig()
        self._sources: queue.Queue[SuccessfulTrajectory] = queue.Queue(maxsize=512)
        self._candidates: queue.Queue[TrajectoryCandidate] = queue.Queue(maxsize=2048)
        self._stop = threading.Event()
        self._optimizer_thread: threading.Thread | None = None
        self._validator_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._seen_sources: set[str] = set()
        self._attempted: set[str] = set()
        self._validated: dict[str, ValidatedTrajectory] = {}
        self._last_error: str | None = None
        self._active_validations = 0
        self._trajectories_seen = 0
        self._candidates_generated = 0
        self._validations = 0
        self._validation_successes = 0

    def start(self) -> None:
        if self._optimizer_thread is not None and self._optimizer_thread.is_alive():
            return
        self.root.mkdir(parents=True, exist_ok=True)
        self.inbox.mkdir(parents=True, exist_ok=True)
        self._publish_validated()
        self._optimizer_thread = threading.Thread(
            target=self._optimizer_loop,
            name="v8-trajectory-optimizer",
            daemon=True,
        )
        self._validator_thread = threading.Thread(
            target=self._validator_loop,
            name="v8-trajectory-validator",
            daemon=True,
        )
        self._optimizer_thread.start()
        self._validator_thread.start()

    def submit_trajectory(self, trajectory: SuccessfulTrajectory) -> bool:
        if trajectory.cost <= 1:
            return False
        with self._lock:
            if trajectory.trajectory_id in self._seen_sources:
                return False
            self._seen_sources.add(trajectory.trajectory_id)
            self._trajectories_seen += 1
        try:
            self._sources.put_nowait(trajectory)
            return True
        except queue.Full:
            with self._lock:
                self._seen_sources.discard(trajectory.trajectory_id)
            return False

    def _ingest_inbox(self) -> None:
        self.inbox.mkdir(parents=True, exist_ok=True)
        for path in sorted(self.inbox.glob("*.json"))[:128]:
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                self.submit_trajectory(SuccessfulTrajectory.from_dict(raw))
            except BaseException as exc:
                self._log("inbox_error", path=str(path), error=f"{type(exc).__name__}: {exc}")
            finally:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _optimizer_loop(self) -> None:
        try:
            while not self._stop.is_set():
                self._ingest_inbox()
                try:
                    source = self._sources.get(timeout=float(self.config.poll_interval_seconds))
                except queue.Empty:
                    continue
                try:
                    if int(source.round_index) >= int(self.config.max_optimization_rounds):
                        continue
                    rows = generate_optimization_candidates(source, self.config)
                    with self._lock:
                        self._candidates_generated += len(rows)
                    self._log(
                        "candidates",
                        trajectory_id=source.trajectory_id,
                        parent_cost=source.cost,
                        count=len(rows),
                        round=source.round_index,
                    )
                    for candidate in rows:
                        if self._stop.is_set():
                            break
                        try:
                            self._candidates.put(candidate, timeout=0.05)
                        except queue.Full:
                            break
                finally:
                    self._sources.task_done()
        except BaseException as exc:
            self._fail(exc)

    def _validator_loop(self) -> None:
        try:
            while not self._stop.is_set() or not self._candidates.empty():
                try:
                    candidate = self._candidates.get(timeout=0.10)
                except queue.Empty:
                    continue
                try:
                    with self._lock:
                        if candidate.candidate_id in self._attempted:
                            continue
                        self._attempted.add(candidate.candidate_id)
                        self._active_validations += 1
                    try:
                        result = self.validator(candidate)
                    finally:
                        with self._lock:
                            self._active_validations -= 1
                    success = bool(getattr(result, "success", False)) and candidate.cost < candidate.source.cost
                    with self._lock:
                        self._validations += 1
                        self._validation_successes += int(success)
                    validated = self._accept(candidate) if success else None
                    self._log(
                        "validation",
                        candidate_id=candidate.candidate_id,
                        edit=candidate.edit_kind,
                        parent_cost=candidate.source.cost,
                        candidate_cost=candidate.cost,
                        success=success,
                        reason=str(getattr(result, "reason", "")),
                    )
                    if self.on_validation is not None:
                        self.on_validation(candidate, result, validated)
                    if validated is not None:
                        next_source = SuccessfulTrajectory(
                            _trajectory_id(validated.anchor, validated.target, validated.actions),
                            validated.anchor,
                            validated.target,
                            validated.actions,
                            validated.strategy_uid,
                            validated.target_outcome_uid,
                            int(candidate.source.round_index) + 1,
                        )
                        self.submit_trajectory(next_source)
                finally:
                    self._candidates.task_done()
        except BaseException as exc:
            self._fail(exc)

    def _accept(self, candidate: TrajectoryCandidate) -> ValidatedTrajectory:
        uid = variant_strategy_uid(candidate)
        row = ValidatedTrajectory(
            candidate.candidate_id,
            candidate.source.anchor,
            candidate.source.target,
            candidate.actions,
            uid,
            candidate.source.target_outcome_uid,
            candidate.source.parent_strategy_uid,
            candidate.source.cost,
            candidate.edit_kind,
            1,
            1,
        )
        key = _frontier_key(row.anchor, row.target)
        with self._lock:
            prior = self._validated.get(key)
            if prior is None or (row.cost, -row.saved_actions, row.variant_id) < (
                prior.cost,
                -prior.saved_actions,
                prior.variant_id,
            ):
                self._validated[key] = row
            else:
                row = prior
        self._publish_validated()
        return row

    def _publish_validated(self) -> None:
        with self._lock:
            rows = [row.to_dict() for row in sorted(self._validated.values(), key=lambda item: item.variant_id)]
        _atomic_json(self.validated_path, {"version": 1, "validated": rows})

    def _log(self, event: str, **values) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"event": str(event), "time": time.time(), **values}
        with self._lock:
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, sort_keys=True) + "\n")

    def _fail(self, exc: BaseException) -> None:
        with self._lock:
            self._last_error = f"{type(exc).__name__}: {exc}"
        self._stop.set()

    def raise_if_failed(self) -> None:
        with self._lock:
            error = self._last_error
        if error is not None:
            raise RuntimeError(f"v8 trajectory optimizer failure: {error}")

    def drain(self, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout))
        while time.monotonic() < deadline:
            self._ingest_inbox()
            with self._lock:
                active = self._active_validations
            if self._sources.unfinished_tasks == 0 and self._candidates.unfinished_tasks == 0 and active == 0:
                return True
            time.sleep(0.02)
        return False

    def stop(self, *, drain: bool = True, timeout: float = 10.0) -> None:
        if drain:
            self.drain(timeout=max(0.0, float(timeout) * 0.7))
        self._stop.set()
        for thread in (self._optimizer_thread, self._validator_thread):
            if thread is not None:
                thread.join(timeout=max(0.1, float(timeout) * 0.3))
        self._publish_validated()

    def metrics(self) -> TrajectoryOptimizerMetrics:
        with self._lock:
            result = TrajectoryOptimizerMetrics(
                self._trajectories_seen,
                self._candidates_generated,
                self._validations,
                self._validation_successes,
                len(self._validated),
            )
        from v8 import information_flow_diagnostics as flow

        observed = flow.counter_snapshot("trajectory_optimizer")
        counters = {
            "successful_trajectories_produced": int(observed.get("successful_trajectories_produced", 0)),
            "trajectories_submitted": int(observed.get("trajectories_submitted", 0)),
            "trajectories_received": int(observed.get("trajectories_received", 0)),
            "trajectories_seen": int(result.trajectories_seen),
            "candidates_generated": int(result.candidates_generated),
            "validations": int(result.validations),
            "validation_successes": int(result.validation_successes),
            "accepted_variants": int(result.validated_variants),
        }
        bypassed = max(0, counters["trajectories_received"] - counters["trajectories_seen"])
        flow.emit(
            "trajectory_optimizer", "pipeline_summary",
            input_count=counters["successful_trajectories_produced"],
            output_count=counters["accepted_variants"],
            rejection_counts=(
                {"received_without_legacy_trajectories_seen_increment": bypassed}
                if bypassed else {}
            ),
            fields={"counters": counters,
                    "trajectories_seen_counter_owner": "legacy_edit_source_queue",
                    "source_validation_bypasses_trajectories_seen": True,
                    "counter_scope": "current_process_observed"},
        )
        return result

    def state_dict(self) -> dict[str, object]:
        with self._lock:
            return {
                "version": 1,
                "seen_sources": sorted(self._seen_sources)[-4096:],
                "attempted": sorted(self._attempted)[-16384:],
                "validated": [
                    row.to_dict()
                    for row in sorted(self._validated.values(), key=lambda item: item.variant_id)
                ],
                "metrics": {
                    "trajectories_seen": self._trajectories_seen,
                    "candidates_generated": self._candidates_generated,
                    "validations": self._validations,
                    "validation_successes": self._validation_successes,
                },
            }

    def load_state(self, state: dict[str, object] | None) -> None:
        if not state:
            return
        with self._lock:
            self._seen_sources.update(str(value) for value in state.get("seen_sources", ()))
            self._attempted.update(str(value) for value in state.get("attempted", ()))
            for raw in state.get("validated", ()):
                if not isinstance(raw, dict):
                    continue
                row = ValidatedTrajectory.from_dict(raw)
                self._validated[_frontier_key(row.anchor, row.target)] = row
            metrics = state.get("metrics", {})
            if isinstance(metrics, dict):
                self._trajectories_seen = int(metrics.get("trajectories_seen", self._trajectories_seen))
                self._candidates_generated = int(metrics.get("candidates_generated", self._candidates_generated))
                self._validations = int(metrics.get("validations", self._validations))
                self._validation_successes = int(metrics.get("validation_successes", self._validation_successes))
        self._publish_validated()


def _reset_capture(job=None) -> None:
    global _CAPTURE_ACTIVE, _CAPTURE_SOURCE_ID, _CAPTURE_SEED, _CAPTURE_ENV_ROOT
    global _CAPTURE_PREFIX, _CAPTURE_SEGMENT, _ACTOR_ACTION_HISTORY, _ACTOR_RESET_EPOCH
    _CAPTURE_ACTIVE = job is not None
    _CAPTURE_SOURCE_ID = "" if job is None else str(job.game_id)
    _CAPTURE_SEED = 0 if job is None else int(job.seed)
    _CAPTURE_ENV_ROOT = None if job is None else job.env_root
    _CAPTURE_PREFIX = []
    _CAPTURE_SEGMENT = []
    _ACTOR_ACTION_HISTORY = []
    _ACTOR_RESET_EPOCH += 1


def _selected_plan(action: int):
    try:
        from v8 import behavior_recovery as behavior

        view = getattr(behavior, "_CURRENT_ACTOR_VIEW", None)
        if view is None:
            return None
        for plan in tuple(getattr(view, "_behavior_last_plans", ())):
            if int(plan.action_id) == int(action):
                return plan
    except BaseException:
        return None
    return None


def _write_successful_trajectory(row: SuccessfulTrajectory) -> None:
    root_raw = os.environ.get(_TRAJECTORY_ROOT_ENV)
    if not root_raw:
        _CAPTURED_FOR_TESTS.append(row)
        return
    inbox = Path(root_raw) / "inbox"
    target = inbox / f"{row.trajectory_id}-{os.getpid()}-{time.time_ns()}.json"
    _atomic_json(target, row.to_dict())


def _capture_env_step(self, action):
    global _CAPTURE_PREFIX, _CAPTURE_SEGMENT, _ACTOR_ACTION_HISTORY, _ACTOR_RESET_EPOCH
    plan = _selected_plan(int(action)) if _CAPTURE_ACTIVE else None
    result = _BASE_ENV_STEP(self, action)
    if not _CAPTURE_ACTIVE:
        return result

    value = int(action)
    _CAPTURE_SEGMENT.append(value)
    _ACTOR_ACTION_HISTORY.append(value)
    state = str(getattr(self, "last_outcome_state", ""))
    level_event = bool(getattr(self, "level_completed_event", False))
    success = level_event or state == "WIN"
    reset_boundary = bool(getattr(self, "last_step_was_reset_boundary", False))

    if success and _CAPTURE_SEGMENT:
        anchor = ReplayAnchor(
            _CAPTURE_SOURCE_ID,
            _CAPTURE_SEED,
            tuple(_CAPTURE_PREFIX),
            _CAPTURE_ENV_ROOT,
        )
        target = TrajectoryTarget(
            int(getattr(self, "last_levels_completed", 0)),
            "WIN" if state == "WIN" else "LEVEL",
        )
        strategy_uid = MemoryUid.zero() if plan is None else plan.strategy_uid
        outcome_uid = MemoryUid.zero() if plan is None else plan.outcome_uid
        actions = tuple(_CAPTURE_SEGMENT)
        row = SuccessfulTrajectory(
            _trajectory_id(anchor, target, actions),
            anchor,
            target,
            actions,
            strategy_uid,
            outcome_uid,
            0,
        )
        _write_successful_trajectory(row)
        if state == "WIN" or reset_boundary:
            _CAPTURE_PREFIX = []
            _CAPTURE_SEGMENT = []
            _ACTOR_ACTION_HISTORY = []
            _ACTOR_RESET_EPOCH += 1
        else:
            _CAPTURE_PREFIX.extend(_CAPTURE_SEGMENT)
            _CAPTURE_SEGMENT = []
    elif state == "GAME_OVER" or reset_boundary:
        _CAPTURE_PREFIX = []
        _CAPTURE_SEGMENT = []
        _ACTOR_ACTION_HISTORY = []
        _ACTOR_RESET_EPOCH += 1
    return result


def _capture_env_reset(self):
    global _CAPTURE_PREFIX, _CAPTURE_SEGMENT, _ACTOR_ACTION_HISTORY, _ACTOR_RESET_EPOCH
    result = _BASE_ENV_RESET(self)
    if _CAPTURE_ACTIVE:
        _CAPTURE_PREFIX = []
        _CAPTURE_SEGMENT = []
        _ACTOR_ACTION_HISTORY = []
        _ACTOR_RESET_EPOCH += 1
    return result


def _actor_worker_v814(*, job, **kwargs):
    if _BASE_ACTOR_WORKER is None:
        raise RuntimeError("v8.14 actor wrapper is not installed")
    _reset_capture(job)
    try:
        return _BASE_ACTOR_WORKER(job=job, **kwargs)
    finally:
        _reset_capture(None)


def _load_validated_rows(path: Path) -> tuple[ValidatedTrajectory, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return ()
    return tuple(
        ValidatedTrajectory.from_dict(raw)
        for raw in payload.get("validated", ())
        if isinstance(raw, dict)
    )


def _view_init_v814(self, *args, **kwargs):
    _BASE_VIEW_INIT(self, *args, **kwargs)
    self._v814_variants = ()
    self._v814_next_refresh = 0.0
    self._v814_active_variant = None
    self._v814_active_actions = ()
    self._v814_attempted_variants = set()
    self._v814_reset_epoch = int(_ACTOR_RESET_EPOCH)


def _refresh_view_variants(self) -> None:
    now = time.monotonic()
    if now < float(getattr(self, "_v814_next_refresh", 0.0)):
        return
    self._v814_next_refresh = now + 1.0
    root_raw = os.environ.get(_TRAJECTORY_ROOT_ENV)
    if not root_raw:
        self._v814_variants = ()
        return
    self._v814_variants = _load_validated_rows(Path(root_raw) / "validated.json")


def _plan_candidates_v814(self, context_signature, action_ids, **kwargs):
    del context_signature
    from v8.publication import PlannedAction

    if int(getattr(self, "_v814_reset_epoch", -1)) != int(_ACTOR_RESET_EPOCH):
        self._v814_reset_epoch = int(_ACTOR_RESET_EPOCH)
        self._v814_active_variant = None
        self._v814_active_actions = ()
        self._v814_attempted_variants = set()

    available = {int(value) for value in action_ids}
    active = getattr(self, "_v814_active_variant", None)
    remaining = tuple(getattr(self, "_v814_active_actions", ()))
    if active is not None and remaining:
        action = int(remaining[0])
        if action in available:
            self._v814_active_actions = remaining[1:]
            plan = PlannedAction(
                action,
                active.target_outcome_uid,
                active.strategy_uid,
                1_000_000.0,
                False,
            )
            self._behavior_last_plans = (plan,)
            return (plan,)
        self._v814_active_variant = None
        self._v814_active_actions = ()

    _refresh_view_variants(self)
    selected = select_validated_variant(
        tuple(getattr(self, "_v814_variants", ())),
        source_id=_CAPTURE_SOURCE_ID,
        seed=_CAPTURE_SEED,
        action_history=tuple(_ACTOR_ACTION_HISTORY),
        attempted=set(getattr(self, "_v814_attempted_variants", set())),
    )
    if selected is not None:
        self._v814_attempted_variants.add(selected.variant_id)
        action = int(selected.actions[0])
        if action in available:
            self._v814_active_variant = selected
            self._v814_active_actions = tuple(selected.actions[1:])
            plan = PlannedAction(
                action,
                selected.target_outcome_uid,
                selected.strategy_uid,
                1_000_000.0,
                False,
            )
            self._behavior_last_plans = (plan,)
            return (plan,)

    return _BASE_PLAN_CANDIDATES(self, context_signature, action_ids, **kwargs)


def _runtime_validation_callback(runtime, candidate, result, validated) -> None:
    peers = getattr(runtime, "peers", None)
    if peers is None:
        return
    from v8.evidence import EvidenceRecord

    watermark = int(runtime.watermark)
    source_hash = stable_u64(candidate.source.anchor.source_id, person=b"v8-game")
    uid = variant_strategy_uid(candidate)
    reduction = max(
        0.0,
        (float(candidate.source.cost) - float(candidate.cost))
        / max(1.0, float(candidate.source.cost)),
    )

    if validated is not None:
        target_uid = validated.target_outcome_uid
        if not target_uid.is_zero:
            target_exists = any(
                row.uid == target_uid
                for row in runtime.read_view.node_records(level=MemoryLevel.M6)
            )
            if target_exists:
                key = variant_strategy_key(candidate)
                runtime.submit_proposal(
                    MemoryProposal(
                        uid=uid,
                        fingerprint=proposal_fingerprint(
                            MemoryLevel.M7, MemoryType.STRATEGY, key
                        ),
                        event_id=EventId.from_producer(
                            0x7FFFFFFC,
                            stable_u64(candidate.candidate_id, watermark, person=b"v8.14-event"),
                        ),
                        watermark=watermark,
                        level=MemoryLevel.M7,
                        memory_type=MemoryType.STRATEGY,
                        key_parts=key,
                        support_delta=1,
                        significance_sum=reduction,
                        learning_value_sum=reduction,
                        score_weight=1.0,
                        success_sum=1.0,
                        cost_sum=float(candidate.cost),
                        attempt_weight=1.0,
                        parent_uid=target_uid,
                        relation_type=RelationType.LEADS_TO,
                        source_game_hash=source_hash,
                        cognitive_state=int(CognitiveState.PROBATION),
                        validation_state=int(ValidationState.TESTED),
                    )
                )
        peers.ledger.append(
            EvidenceRecord.for_uid(
                f"trajectory-optimization-success:{candidate.candidate_id}:{watermark}",
                uid,
                evidence_kind="trajectory_optimization_success",
                watermark=watermark,
                raw_value=reduction,
                normalized_value=min(1.0, reduction),
                developmental_stage=int(MemoryLevel.M7),
                validation_state=int(ValidationState.TESTED),
                source_game_hash=source_hash,
                causal_intervention="action_sequence_ablation",
                effect_direction=1,
                graph_generation=int(runtime.generation),
            )
        )
        peers.ledger.append(
            EvidenceRecord.for_uid(
                f"trajectory-cost-reduction:{candidate.candidate_id}:{watermark}",
                uid,
                evidence_kind="trajectory_cost_reduction",
                watermark=watermark,
                raw_value=float(candidate.source.cost - candidate.cost),
                normalized_value=min(1.0, reduction),
                developmental_stage=int(MemoryLevel.M7),
                validation_state=int(ValidationState.TESTED),
                source_game_hash=source_hash,
                causal_intervention="action_sequence_ablation",
                effect_direction=1,
                graph_generation=int(runtime.generation),
            )
        )
    else:
        failed_uid = (
            candidate.source.parent_strategy_uid
            if not candidate.source.parent_strategy_uid.is_zero
            else uid
        )
        peers.ledger.append(
            EvidenceRecord.for_uid(
                f"trajectory-optimization-failure:{candidate.candidate_id}:{watermark}",
                failed_uid,
                evidence_kind="trajectory_optimization_failure",
                watermark=watermark,
                raw_value=1.0,
                normalized_value=1.0,
                developmental_stage=int(MemoryLevel.M7),
                validation_state=int(ValidationState.FAILED),
                source_game_hash=source_hash,
                causal_intervention="action_sequence_ablation",
                effect_direction=-1,
                graph_generation=int(runtime.generation),
            )
        )


def _runtime_init_v814(self, *args, **kwargs):
    _BASE_RUNTIME_INIT(self, *args, **kwargs)
    from v8.snapshot import load_latest_auxiliary_state
    from v8.trajectory_validation_v814 import validate_arc_candidate

    service = TrajectoryOptimizationService(
        self.root / "trajectory_optimizer",
        validator=validate_arc_candidate,
        on_validation=lambda candidate, result, validated: _runtime_validation_callback(
            self, candidate, result, validated
        ),
    )
    if bool(getattr(self.config, "restore", False)):
        state = load_latest_auxiliary_state(self.root)
        if isinstance(state, dict):
            optimizer_state = state.get("trajectory_optimizer")
            if isinstance(optimizer_state, dict):
                service.load_state(optimizer_state)
    self._v814_trajectory_optimizer = service
    self._v814_prior_trajectory_root = os.environ.get(_TRAJECTORY_ROOT_ENV)


def _runtime_start_v814(self) -> None:
    _BASE_RUNTIME_START(self)
    service = getattr(self, "_v814_trajectory_optimizer", None)
    if service is not None:
        os.environ[_TRAJECTORY_ROOT_ENV] = str(service.root)
        service.start()


def _runtime_aux_v814(self) -> str:
    payload = json.loads(_BASE_RUNTIME_AUX(self))
    service = getattr(self, "_v814_trajectory_optimizer", None)
    if service is not None:
        payload["trajectory_optimizer"] = service.state_dict()
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _runtime_metrics_v814(self) -> dict[str, object]:
    payload = _BASE_RUNTIME_METRICS(self)
    service = getattr(self, "_v814_trajectory_optimizer", None)
    if service is not None:
        metrics = service.metrics()
        payload["trajectory_optimizer"] = {
            "trajectories_seen": metrics.trajectories_seen,
            "candidates_generated": metrics.candidates_generated,
            "validations": metrics.validations,
            "validation_successes": metrics.validation_successes,
            "validated_variants": metrics.validated_variants,
        }
    return payload


def _runtime_errors_v814(self) -> None:
    _BASE_RUNTIME_ERRORS(self)
    service = getattr(self, "_v814_trajectory_optimizer", None)
    if service is not None:
        service.raise_if_failed()


def _runtime_close_v814(self, *, normal: bool = True, timeout: float = 120.0):
    service = getattr(self, "_v814_trajectory_optimizer", None)
    if service is not None:
        service.stop(drain=bool(normal), timeout=min(15.0, max(1.0, float(timeout) * 0.2)))
    try:
        return _BASE_RUNTIME_CLOSE(self, normal=normal, timeout=timeout)
    finally:
        prior = getattr(self, "_v814_prior_trajectory_root", None)
        current = os.environ.get(_TRAJECTORY_ROOT_ENV)
        if service is not None and current == str(service.root):
            if prior is None:
                os.environ.pop(_TRAJECTORY_ROOT_ENV, None)
            else:
                os.environ[_TRAJECTORY_ROOT_ENV] = str(prior)


def install_trajectory_optimizer_v814() -> None:
    global _INSTALLED
    global _BASE_ACTOR_WORKER, _BASE_ENV_STEP, _BASE_ENV_RESET
    global _BASE_VIEW_INIT, _BASE_PLAN_CANDIDATES
    global _BASE_RUNTIME_INIT, _BASE_RUNTIME_START, _BASE_RUNTIME_AUX
    global _BASE_RUNTIME_METRICS, _BASE_RUNTIME_ERRORS, _BASE_RUNTIME_CLOSE
    if _INSTALLED:
        return

    from v7.environment.arc_adapter import ArcGridEnvironment
    from v8 import actor as actor_module
    from v8.publication import LiveReadView
    from v8.runtime_v82 import V82ContinuousMemoryRuntime

    _BASE_ACTOR_WORKER = actor_module.actor_worker
    _BASE_ENV_STEP = ArcGridEnvironment.step
    _BASE_ENV_RESET = ArcGridEnvironment.reset
    _BASE_VIEW_INIT = LiveReadView.__init__
    _BASE_PLAN_CANDIDATES = LiveReadView.plan_candidates
    _BASE_RUNTIME_INIT = V82ContinuousMemoryRuntime.__init__
    _BASE_RUNTIME_START = V82ContinuousMemoryRuntime.start
    _BASE_RUNTIME_AUX = V82ContinuousMemoryRuntime._auxiliary_state_json
    _BASE_RUNTIME_METRICS = V82ContinuousMemoryRuntime.metrics
    _BASE_RUNTIME_ERRORS = V82ContinuousMemoryRuntime.raise_worker_errors
    _BASE_RUNTIME_CLOSE = V82ContinuousMemoryRuntime.close

    actor_module.actor_worker = _actor_worker_v814
    ArcGridEnvironment.step = _capture_env_step
    ArcGridEnvironment.reset = _capture_env_reset
    LiveReadView.__init__ = _view_init_v814
    LiveReadView.plan_candidates = _plan_candidates_v814
    V82ContinuousMemoryRuntime.__init__ = _runtime_init_v814
    V82ContinuousMemoryRuntime.start = _runtime_start_v814
    V82ContinuousMemoryRuntime._auxiliary_state_json = _runtime_aux_v814
    V82ContinuousMemoryRuntime.metrics = _runtime_metrics_v814
    V82ContinuousMemoryRuntime.raise_worker_errors = _runtime_errors_v814
    V82ContinuousMemoryRuntime.close = _runtime_close_v814
    _INSTALLED = True
