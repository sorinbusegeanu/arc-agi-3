from __future__ import annotations

import math
import os
from collections import defaultdict, deque
from dataclasses import dataclass
from hashlib import blake2b
from typing import Iterable

from v8.arena import EdgeRecord, NodeRecord
from v8.model import (
    CognitiveState,
    MemoryLevel,
    MemoryType,
    MemoryUid,
    RelationType,
    ValidationState,
    stable_u64,
)
from v8.promotion import FormationCandidate


_INSTALLED = False
_CONTROL_SCOPE_ENV = "ARC_AGI3_V8_CONTROL_SCOPE"
_COMPLEX_ACTION_ID = 6
_COORDINATE_SIDE = 64
_COORDINATE_PAGE_SIZE = 64
_SEQUENCE_MARKER = 1 << 63
_SEQUENCE_MASK = _SEQUENCE_MARKER - 1
_MAX_SEQUENCE_ACTIONS = 6
_ESCAPE_STREAK = 32
_ESCAPE_BUDGET = 16
_PROGRESS_EPOCH = 0


def pack_action_choice(action_id: int, x: int | None = None, y: int | None = None) -> int:
    """Canonical reversible action token; simple actions keep their legacy ids."""
    action = int(action_id)
    if x is None and y is None:
        return action
    if action != _COMPLEX_ACTION_ID or x is None or y is None:
        raise ValueError("parameter payload is supported only for the coordinate action")
    x_i, y_i = int(x), int(y)
    if not (0 <= x_i < _COORDINATE_SIDE and 0 <= y_i < _COORDINATE_SIDE):
        raise ValueError("action coordinates must be in [0, 63]")
    return action | ((x_i + 1) << 8) | ((y_i + 1) << 15)


def unpack_action_choice(token: int) -> tuple[int, dict[str, int] | None]:
    value = int(token)
    if value < 0:
        raise ValueError("action token must be non-negative")
    action = value & 0xFF
    x_raw = (value >> 8) & 0x7F
    y_raw = (value >> 15) & 0x7F
    if x_raw == 0 and y_raw == 0:
        return action, None
    if action != _COMPLEX_ACTION_ID or x_raw == 0 or y_raw == 0:
        raise ValueError("invalid parameterized action token")
    x, y = x_raw - 1, y_raw - 1
    if not (0 <= x < _COORDINATE_SIDE and 0 <= y < _COORDINATE_SIDE):
        raise ValueError("invalid coordinate payload")
    return action, {"x": x, "y": y}


def control_context_signature(grid) -> int:
    """Lossless-with-respect-to-grid-content local control signature."""
    import numpy as np

    array = np.asarray(grid, dtype=np.int16, order="C")
    digest = blake2b(digest_size=8, person=b"v8.5-control")
    scope = os.environ.get(_CONTROL_SCOPE_ENV, "").encode("utf-8")
    digest.update(len(scope).to_bytes(2, "little"))
    digest.update(scope)
    digest.update(int(array.ndim).to_bytes(1, "little"))
    for dim in array.shape:
        digest.update(int(dim).to_bytes(2, "little"))
    digest.update(array.tobytes(order="C"))
    return int.from_bytes(digest.digest(), "little")


def _sequence_hash(path: Iterable[NodeRecord]) -> int:
    value = stable_u64(0, person=b"v8.5-sequence")
    for index, row in enumerate(path):
        value = stable_u64(
            value,
            index,
            int(row.uid.hi),
            int(row.uid.lo),
            int(row.key_parts[1]),
            person=b"v8.5-sequence",
        )
    return _SEQUENCE_MARKER | (int(value) & _SEQUENCE_MASK)


def is_composite_strategy(row: NodeRecord) -> bool:
    return bool(
        int(row.level) == int(MemoryLevel.M7)
        and int(row.memory_type) == int(MemoryType.STRATEGY)
        and len(row.key_parts) >= 4
        and (int(row.key_parts[0]) & _SEQUENCE_MARKER)
    )


def richer_outcome_key(consequence: NodeRecord) -> tuple[int, int, int]:
    """Keep the full learned consequence signature; no low-bit truncation."""
    if int(consequence.level) != int(MemoryLevel.M5) or len(consequence.key_parts) < 4:
        raise ValueError("canonical M6 outcome requires an M5 consequence")
    future = int(consequence.key_parts[3])
    consequence_signature = int(consequence.key_parts[2])
    context_variant = stable_u64(
        int(consequence.key_parts[0]),
        int(consequence.key_parts[1]),
        person=b"v8.5-outcome-context",
    )
    return future, consequence_signature, context_variant


class CompositePromotionEngine:
    """Mixin-style factory installed as a subclass of the active promotion engine."""


def _composite_candidates(engine, nodes, edges, *, limit: int) -> tuple[FormationCandidate, ...]:
    if limit <= 0:
        return ()
    from v8 import behavior_recovery as behavior

    m1 = [
        row
        for row in nodes
        if int(row.level) == int(MemoryLevel.M1)
        and int(row.memory_type) == int(MemoryType.CONTINGENCY)
        and len(row.key_parts) >= 4
        and row.support_count >= int(getattr(engine, "min_contingency_support", 3))
        and engine._admissible(row)
    ]
    incoming: dict[int, list[NodeRecord]] = defaultdict(list)
    for row in m1:
        incoming[int(row.key_parts[3])].append(row)
    for rows in incoming.values():
        rows.sort(key=lambda r: (-int(r.support_count), r.uid))

    outcomes = [
        row
        for row in nodes
        if int(row.level) == int(MemoryLevel.M6)
        and int(row.memory_type) == int(MemoryType.OUTCOME)
        and len(row.key_parts) >= 3
        and engine._admissible(row)
    ]
    result: list[FormationCandidate] = []
    seen: set[MemoryUid] = set()

    for outcome in sorted(outcomes, key=lambda r: r.uid):
        ancestors = behavior.causal_m1_ancestors(
            outcome.uid, nodes=tuple(nodes), edges=tuple(edges), max_depth=8
        )
        by_uid = {row.uid: row for row in m1}
        for target_uid in sorted(ancestors):
            target = by_uid.get(target_uid)
            if target is None:
                continue
            paths: list[list[NodeRecord]] = [[target]]
            for _depth in range(_MAX_SEQUENCE_ACTIONS - 1):
                extended: list[list[NodeRecord]] = []
                for path in paths:
                    before = int(path[0].key_parts[0])
                    for predecessor in incoming.get(before, ())[:2]:
                        if predecessor.uid in {row.uid for row in path}:
                            continue
                        candidate_path = [predecessor, *path]
                        extended.append(candidate_path)
                        if len(candidate_path) >= 2:
                            sequence_id = _sequence_hash(candidate_path)
                            key = (
                                int(sequence_id),
                                int(outcome.uid.hi),
                                int(outcome.uid.lo),
                                int(candidate_path[0].key_parts[0]),
                            )
                            uid = MemoryUid.from_key(
                                MemoryLevel.M7, MemoryType.STRATEGY, key
                            )
                            if uid in seen:
                                continue
                            seen.add(uid)
                            support = min(
                                [int(outcome.support_count)]
                                + [int(row.support_count) for row in candidate_path]
                            )
                            result.append(
                                FormationCandidate(
                                    uid=uid,
                                    level=MemoryLevel.M7,
                                    memory_type=MemoryType.STRATEGY,
                                    key_parts=key,
                                    parents=(outcome.uid,)
                                    + tuple(row.uid for row in candidate_path),
                                    support=max(1, support),
                                    significance=min(
                                        1.0,
                                        sum(float(row.significance) for row in candidate_path)
                                        / len(candidate_path),
                                    ),
                                    learning_value=min(
                                        1.0,
                                        sum(float(row.learning_value) for row in candidate_path)
                                        / len(candidate_path),
                                    ),
                                    transfer_prior=0.0,
                                    explanatory_reach=1.0,
                                    future_option_delta=float(
                                        candidate_path[-1].future_option_delta
                                    ),
                                    cognitive_state=int(CognitiveState.PROBATION),
                                    validation_state=int(ValidationState.STRUCTURAL),
                                    evidence_kind="multi_action_strategy",
                                    evidence_value=min(1.0, support / 4.0),
                                )
                            )
                            if len(result) >= limit:
                                return tuple(result)
                if not extended:
                    break
                paths = extended[:16]
    return tuple(result)


def _path_for_composite(view, row: NodeRecord, start_context: int) -> tuple[NodeRecord, ...]:
    dependencies = set(
        getattr(view, "_behavior_strategy_dependencies", {}).get(row.uid, set())
    )
    if not dependencies:
        return ()
    by_uid = getattr(view, "_node_by_uid", {})
    candidates = {
        uid: by_uid[uid]
        for uid in dependencies
        if uid in by_uid
        and int(by_uid[uid].level) == int(MemoryLevel.M1)
        and len(by_uid[uid].key_parts) >= 4
    }
    adjacency: dict[int, list[NodeRecord]] = defaultdict(list)
    for item in candidates.values():
        adjacency[int(item.key_parts[0])].append(item)
    for rows in adjacency.values():
        rows.sort(key=lambda r: r.uid)

    target_hash = int(row.key_parts[0])
    stack: list[tuple[int, tuple[NodeRecord, ...]]] = [(int(start_context), ())]
    while stack:
        context, path = stack.pop()
        if len(path) >= _MAX_SEQUENCE_ACTIONS:
            continue
        for edge in reversed(adjacency.get(context, ())):
            if edge in path:
                continue
            next_path = (*path, edge)
            if _sequence_hash(next_path) == target_hash:
                return next_path
            stack.append((int(edge.key_parts[3]), next_path))
    return ()


def _composite_plans(view, context_signature: int, action_ids) -> tuple[object, ...]:
    from v8.publication import PlannedAction

    view._refresh_strategy_cache()
    available = {int(value) for value in action_ids}
    by_uid = getattr(view, "_node_by_uid", {})
    plans: list[PlannedAction] = []
    for row in by_uid.values():
        if not is_composite_strategy(row):
            continue
        if int(row.key_parts[3]) != int(context_signature):
            continue
        path = _path_for_composite(view, row, int(context_signature))
        if not path:
            continue
        action = int(path[0].key_parts[1])
        if action not in available:
            continue
        outcome_uid = MemoryUid(int(row.key_parts[1]), int(row.key_parts[2]))
        outcome = by_uid.get(outcome_uid)
        strategy_value = float(getattr(row, "expected_primary_valence", 0.0)) * float(
            getattr(row, "primary_valence_confidence", 0.0)
        )
        outcome_value = (
            0.0
            if outcome is None
            else float(getattr(outcome, "expected_primary_valence", 0.0))
            * float(getattr(outcome, "primary_valence_confidence", 0.0))
        )
        reliability = float(getattr(row, "strategy_reliability", 0.0))
        support = max(0, int(row.support_count))
        sequence_efficiency = 1.0 / max(1, len(path))
        probe_bonus = 0.10 if float(row.attempt_weight) < 3.0 else 0.0
        score = (
            reliability
            + 1.5 * strategy_value
            + outcome_value
            + 0.10 * sequence_efficiency
            + 0.05 * math.log1p(support)
            + probe_bonus
        )
        plans.append(
            PlannedAction(action, outcome_uid, row.uid, float(score), False)
        )
    plans.sort(key=lambda item: (-item.score, item.action_id, item.strategy_uid))
    return tuple(plans)


def _install_environment_and_context() -> None:
    from v7.environment import arc_adapter as adapter
    from v7.environment import encoding
    from v8 import actor as actor_module

    base_available = adapter.ArcGridEnvironment.available_actions
    base_step = adapter.ArcGridEnvironment.step
    base_actor_worker = actor_module.actor_worker

    def available_actions(self):
        base = tuple(sorted(set(int(value) for value in base_available(self))))
        if _COMPLEX_ACTION_ID not in base:
            return list(base)
        page = int(getattr(self, "_v055_complex_page", 0)) % (
            (_COORDINATE_SIDE * _COORDINATE_SIDE) // _COORDINATE_PAGE_SIZE
        )
        start = page * _COORDINATE_PAGE_SIZE
        tokens = [
            pack_action_choice(
                _COMPLEX_ACTION_ID,
                index % _COORDINATE_SIDE,
                index // _COORDINATE_SIDE,
            )
            for index in range(start, start + _COORDINATE_PAGE_SIZE)
        ]
        return [value for value in base if value != _COMPLEX_ACTION_ID] + tokens

    def _advance_page(self):
        pages = (_COORDINATE_SIDE * _COORDINATE_SIDE) // _COORDINATE_PAGE_SIZE
        self._v055_complex_page = (int(getattr(self, "_v055_complex_page", 0)) + 1) % pages

    def step(self, action):
        global _PROGRESS_EPOCH
        action_id, data = unpack_action_choice(int(action))
        if data is None:
            result = base_step(self, action_id)
        else:
            previous_levels = self.last_levels_completed
            self.last_step_was_reset_boundary = False
            self.last_terminal_state = None
            try:
                from arcengine import GameAction

                raw = self.env.step(GameAction.from_id(action_id), data=data)
            except ImportError:
                raw = self.env.step(action_id, data=data)
            state = adapter._state_name(raw) or "NOT_FINISHED"
            levels = int(getattr(raw, "levels_completed", previous_levels) or 0)
            try:
                grid = adapter._grid_from_raw(raw)
            except ValueError:
                if not self.auto_reset_on_empty_frame:
                    raise
                self.last_outcome_state = (
                    state if state != "NOT_FINISHED" else "GAME_OVER"
                )
                self.last_levels_completed = levels
                self.level_completed_event = levels > previous_levels
                self.last_outcome_polarity = adapter._polarity(
                    self.last_outcome_state, self.level_completed_event
                )
                self.last_terminal_state = (
                    self.last_outcome_state
                    if self.last_outcome_state in {"WIN", "GAME_OVER"}
                    else None
                )
                self._wait_after_terminal_game()
                self._last_raw = self.env.reset()
                self._last_grid = adapter._grid_from_raw(self._last_raw)
                self._terminal_wait_armed = True
                self.reset_count += 1
                self.skipped_terminal_steps += 1
                self.last_step_was_reset_boundary = True
                result = self._last_grid.copy()
            else:
                self._last_raw = raw
                self._last_grid = grid
                self.last_outcome_state = state
                self.last_levels_completed = levels
                self.level_completed_event = levels > previous_levels
                self.last_outcome_polarity = adapter._polarity(
                    state, self.level_completed_event
                )
                self.last_terminal_state = (
                    state if state in {"WIN", "GAME_OVER"} else None
                )
                self._wait_after_terminal_game()
                result = grid.copy()
        _advance_page(self)
        if bool(getattr(self, "level_completed_event", False)) or str(
            getattr(self, "last_outcome_state", "")
        ) in {"WIN", "GAME_OVER"}:
            _PROGRESS_EPOCH += 1
        return result

    def actor_worker(*, job, **kwargs):
        prior = os.environ.get(_CONTROL_SCOPE_ENV)
        os.environ[_CONTROL_SCOPE_ENV] = str(job.game_id)
        try:
            return base_actor_worker(job=job, **kwargs)
        finally:
            if prior is None:
                os.environ.pop(_CONTROL_SCOPE_ENV, None)
            else:
                os.environ[_CONTROL_SCOPE_ENV] = prior

    adapter.ArcGridEnvironment.available_actions = available_actions
    adapter.ArcGridEnvironment.step = step
    encoding.structural_grid_signature = control_context_signature
    actor_module.actor_worker = actor_worker


def _install_credit_and_drive() -> None:
    from v8 import actor as actor_module
    from v8 import primary_valence as primary
    from v8.future_options import FutureOptionEstimator

    primary._VALENCE_GAMMA = 0.995
    primary._VALENCE_HORIZON = 4096
    primary._TRAJECTORY = deque(tuple(primary._TRAJECTORY), maxlen=primary._VALENCE_HORIZON)

    def local_significance(changed_cells: int, future_delta: float) -> float:
        structural = min(1.0, max(0, int(changed_cells)) / 32.0)
        option = math.tanh(abs(float(future_delta)))
        return 0.20 * structural + 0.80 * option

    actor_module._local_significance = local_significance

    base_init = FutureOptionEstimator.__init__

    def init(self, *, horizon: int = 8):
        return base_init(self, horizon=horizon)

    FutureOptionEstimator.__init__ = init


def _install_outcomes_and_world_model() -> None:
    from v8 import behavior_recovery as behavior
    from v8 import hypothesis_validation_v054 as validation
    from v8.world_model import WorldModelEstimator

    behavior.canonical_outcome_key = richer_outcome_key
    if validation._BASE_WORLD_MODEL_PROPOSE is not None:
        WorldModelEstimator.propose = validation._BASE_WORLD_MODEL_PROPOSE


def _install_composite_promotion() -> None:
    from v8 import behavior_recovery as behavior
    from v8 import peers_v82
    from v8 import promotion

    base_engine = behavior.CausalEvidenceGatedPromotionEngine

    class V055CompositePromotionEngine(base_engine):
        def propose(self, nodes, edges, *, budget: int = 256):
            limit = max(0, int(budget))
            if limit <= 0:
                return ()
            base_budget = max(1, (limit * 3) // 4)
            base = list(super().propose(nodes, edges, budget=base_budget))
            remaining = max(0, limit - len(base))
            base.extend(
                _composite_candidates(self, tuple(nodes), tuple(edges), limit=remaining)
            )
            return tuple(base[:limit])

    promotion.EvidenceGatedPromotionEngine = V055CompositePromotionEngine
    peers_v82.EvidenceGatedPromotionEngine = V055CompositePromotionEngine
    behavior.CausalEvidenceGatedPromotionEngine = V055CompositePromotionEngine


def _install_planning_and_stagnation() -> None:
    from v8 import behavior_recovery as behavior
    from v8.publication import ActionScore, LiveReadView

    base_plan = LiveReadView.plan_candidates
    base_score = LiveReadView.score_actions
    base_init = LiveReadView.__init__

    def init(self, *args, **kwargs):
        base_init(self, *args, **kwargs)
        self._v055_recent_contexts = deque(maxlen=64)
        self._v055_last_novel_tick = 0
        self._v055_tick = 0
        self._v055_progress_epoch = _PROGRESS_EPOCH
        self._v055_escape_budget = 0
        self._v055_active_sequence = None

    def plan_candidates(self, context_signature, action_ids, **kwargs):
        from v8.publication import PlannedAction

        self._v055_tick += 1
        context = int(context_signature)
        available = {int(value) for value in action_ids}

        active = self._v055_active_sequence
        if active is not None:
            strategy_uid, outcome_uid, remaining = active
            if remaining:
                next_row = remaining[0]
                action = int(next_row.key_parts[1])
                if int(next_row.key_parts[0]) == context and action in available:
                    row = getattr(self, "_node_by_uid", {}).get(strategy_uid)
                    outcome = getattr(self, "_node_by_uid", {}).get(outcome_uid)
                    strategy_value = 0.0 if row is None else float(getattr(row, "expected_primary_valence", 0.0)) * float(getattr(row, "primary_valence_confidence", 0.0))
                    outcome_value = 0.0 if outcome is None else float(getattr(outcome, "expected_primary_valence", 0.0)) * float(getattr(outcome, "primary_valence_confidence", 0.0))
                    self._v055_active_sequence = (strategy_uid, outcome_uid, remaining[1:])
                    plan = PlannedAction(action, outcome_uid, strategy_uid, 1.0 + 1.5 * strategy_value + outcome_value, False)
                    self._behavior_last_plans = (plan,)
                    return (plan,)
            self._v055_active_sequence = None
        if self._v055_progress_epoch != _PROGRESS_EPOCH:
            self._v055_progress_epoch = _PROGRESS_EPOCH
            self._v055_last_novel_tick = self._v055_tick
            self._v055_escape_budget = 0
            self._v055_recent_contexts.clear()
        if context not in self._v055_recent_contexts:
            self._v055_last_novel_tick = self._v055_tick
        self._v055_recent_contexts.append(context)
        if self._v055_tick - self._v055_last_novel_tick >= _ESCAPE_STREAK:
            self._v055_escape_budget = _ESCAPE_BUDGET
            self._v055_last_novel_tick = self._v055_tick
        if self._v055_escape_budget > 0:
            self._behavior_last_plans = ()
            return ()

        base = list(base_plan(self, context_signature, action_ids, **kwargs))
        by_uid = getattr(self, "_node_by_uid", {})
        exact_bucket = stable_u64(context, person=b"v8-context")
        filtered = []
        for plan in base:
            row = by_uid.get(plan.strategy_uid)
            if row is None or len(row.key_parts) < 4:
                continue
            if is_composite_strategy(row):
                continue
            strategy_bucket = int(row.key_parts[3])
            cross_context = strategy_bucket != int(exact_bucket)
            if cross_context and int(row.validation_state) != int(
                ValidationState.VALIDATED
            ):
                continue
            filtered.append(plan)

        composites = list(_composite_plans(self, context, action_ids))
        all_plans = composites + filtered
        all_plans.sort(
            key=lambda item: (-item.score, item.action_id, item.strategy_uid)
        )
        if all_plans:
            chosen = all_plans[0]
            chosen_row = by_uid.get(chosen.strategy_uid)
            if chosen_row is not None and is_composite_strategy(chosen_row):
                path = _path_for_composite(self, chosen_row, context)
                if path:
                    self._v055_active_sequence = (
                        chosen.strategy_uid,
                        chosen.outcome_uid,
                        tuple(path[1:]),
                    )
        self._behavior_last_plans = tuple(all_plans)
        return tuple(all_plans)

    def score_actions(self, context_signature, action_ids):
        scores = list(base_score(self, context_signature, action_ids))
        if self._v055_escape_budget <= 0 or not scores:
            return tuple(scores)
        self._v055_escape_budget -= 1
        ranked = sorted(
            scores, key=lambda row: (int(row.support_count), row.action_id)
        )
        explore = {row.action_id for row in ranked[: min(8, len(ranked))]}
        result = []
        for row in scores:
            if row.action_id in explore:
                result.append(ActionScore(row.action_id, 0, row.score, row.evidence_shards))
            else:
                result.append(
                    ActionScore(
                        row.action_id,
                        max(1, int(row.support_count)),
                        row.score,
                        row.evidence_shards,
                    )
                )
        return tuple(result)

    LiveReadView.__init__ = init
    LiveReadView.plan_candidates = plan_candidates
    LiveReadView.score_actions = score_actions


def _install_validation_semantics() -> None:
    from v8 import hypothesis_validation_v054 as validation
    from v8 import peers as peers_module

    validation._auto_transfer_trials = lambda self, nodes, edges: None
    validation._auto_outcome_holdout = lambda self, nodes, edges: None

    base_record_transfer = peers_module.DevelopmentalPeerSupervisor.record_transfer_trial

    def record_behavioral_transfer_trial(
        self,
        uid: MemoryUid,
        *,
        target_game_hash: int,
        metric_with_memory: float,
        metric_without_memory: float,
        formation_games: tuple[int, ...] = (),
    ):
        return base_record_transfer(
            self,
            uid,
            target_game_hash=target_game_hash,
            metric_on=float(metric_with_memory),
            metric_off=float(metric_without_memory),
            formation_games=formation_games,
            intervention="behavioral_memory_ablation",
        )

    peers_module.DevelopmentalPeerSupervisor.record_behavioral_transfer_trial = (
        record_behavioral_transfer_trial
    )


def _install_preference_dedupe() -> None:
    from v8 import preference as preference_module

    base_record = preference_module.PreferenceEstimator.record_probe
    base_load = preference_module.PreferenceEstimator.load_state

    def _seen(self):
        value = getattr(self, "_v055_seen_probes", None)
        if value is None:
            value = set()
            for probe in getattr(self, "_probes", ()):
                value.add(
                    (
                        probe.outcome_a,
                        probe.outcome_b,
                        int(probe.context_bucket),
                        probe.chosen_outcome,
                        bool(probe.both_reachable),
                        bool(probe.preference_influenced),
                    )
                )
            self._v055_seen_probes = value
        return value

    def record_probe(self, **kwargs):
        key = (
            kwargs["outcome_a"],
            kwargs["outcome_b"],
            int(kwargs["context_bucket"]),
            kwargs["chosen_outcome"],
            bool(kwargs["both_reachable"]),
            bool(kwargs["preference_influenced"]),
        )
        seen = _seen(self)
        if key in seen:
            return False
        accepted = base_record(self, **kwargs)
        if accepted:
            seen.add(key)
        return accepted

    def load_state(self, state):
        base_load(self, state)
        self._v055_seen_probes = None
        _seen(self)

    preference_module.PreferenceEstimator.record_probe = record_probe
    preference_module.PreferenceEstimator.load_state = load_state


def _install_corrected_result_recording() -> None:
    from v8 import hypothesis_validation_v054 as validation
    from v8 import runtime as runtime_module

    base = validation._BASE_RUNTIME_RECORD_RESULTS
    if base is None:
        return

    def record_actor_results(self, results):
        filtered = validation._filtered_replanning_results(tuple(results))
        return base(self, filtered)

    runtime_module.ContinuousMemoryRuntime.record_actor_results = record_actor_results


def _install_metadata() -> None:
    from v8.runtime_v82 import V82ContinuousMemoryRuntime

    V82ContinuousMemoryRuntime.scientific_semantics_version = "v8.5-learning-capability"


def install_learning_blockers_v055() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_environment_and_context()
    _install_credit_and_drive()
    _install_outcomes_and_world_model()
    _install_composite_promotion()
    _install_planning_and_stagnation()
    _install_validation_semantics()
    _install_preference_dedupe()
    _install_corrected_result_recording()
    _install_metadata()
    _INSTALLED = True
