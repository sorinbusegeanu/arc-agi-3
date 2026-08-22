from __future__ import annotations

"""v8.38 closure of the environment-neutral cognition contract.

This layer removes the remaining semantic leaks left after v8.37:
- environment transitions become adapter-authored runtime evidence;
- M6 outcome identity participates in trajectory/candidate/frontier identity;
- replay of an M6 target must empirically reproduce that M6 (and any boundary);
- best-solution persistence stores generic targets instead of manufacturing WIN;
- legacy FULL_WIN allocator state is migrated to the generic EPISODE:+1 scope;
- explicit TRANSFER scoring can use structurally grounded M1N evidence.
"""

import json
import os
import threading
from dataclasses import dataclass, replace
from types import SimpleNamespace

from v8.environment_contract import (
    BoundaryEvent,
    BoundaryScope,
    EnvironmentTransition,
    OptimizationScope,
    OptimizationScopeKind,
    TransitionSemantics,
    optimization_scope_for,
    target_boundary,
)
from v8.model import MemoryUid, stable_u64
from v8.environment_neutrality_v837 import V837TrajectoryTarget


_INSTALLED = False
_TRANSITION_LOCAL = threading.local()
_OLD_FULL_WIN_SCOPE_LEVEL = 1_000_000_000

_BASE_V829_ENV_STEP = None
_BASE_V829_ENV_RESET = None
_BASE_V829_SCORE = None
_BASE_V837_TRANSITION_SEMANTICS = None
_BASE_V819_WRITE = None
_BASE_SERVICE_SUBMIT = None
_BASE_V818_STATE_DICT = None
_BASE_V818_LOAD_STATE = None
_BASE_V835_RUNTIME_INIT = None
_BASE_COORDINATOR_STATE_DICT = None
_BASE_COORDINATOR_LOAD_STATE = None
_BASE_INSPECTION_PUBLISH = None


@dataclass(frozen=True, slots=True)
class V838TrajectoryTarget(V837TrajectoryTarget):
    """Compatible trajectory target with explicit generic outcome identity."""

    outcome_hi: int = 0
    outcome_lo: int = 0

    def __post_init__(self) -> None:
        V837TrajectoryTarget.__post_init__(self)
        object.__setattr__(self, "outcome_hi", int(self.outcome_hi))
        object.__setattr__(self, "outcome_lo", int(self.outcome_lo))

    @property
    def outcome_uid(self) -> MemoryUid:
        return MemoryUid(int(self.outcome_hi), int(self.outcome_lo))

    def to_dict(self) -> dict[str, object]:
        raw = V837TrajectoryTarget.to_dict(self)
        raw["outcome_uid"] = [int(self.outcome_hi), int(self.outcome_lo)]
        return raw

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "V838TrajectoryTarget":
        uid = raw.get("outcome_uid", (raw.get("outcome_hi", 0), raw.get("outcome_lo", 0)))
        if not isinstance(uid, (list, tuple)) or len(uid) < 2:
            uid = (0, 0)
        return cls(
            int(raw.get("levels_completed", 0)),
            str(raw.get("terminal_state", "LEVEL")),
            str(raw.get("boundary_scope", "")),
            int(raw.get("primary_valence", 0)),
            bool(raw.get("continuation", True)),
            int(uid[0]),
            int(uid[1]),
        )


def _uid_from_target(target) -> MemoryUid:
    return MemoryUid(
        int(getattr(target, "outcome_hi", 0)),
        int(getattr(target, "outcome_lo", 0)),
    )


def _target_with_outcome(target, uid: MemoryUid) -> V838TrajectoryTarget:
    resolved = uid if uid is not None and not uid.is_zero else _uid_from_target(target)
    return V838TrajectoryTarget(
        int(getattr(target, "levels_completed", 0)),
        str(getattr(target, "terminal_state", "LEVEL")),
        str(getattr(target, "boundary_scope", "")),
        int(getattr(target, "primary_valence", 0)),
        bool(getattr(target, "continuation", True)),
        int(resolved.hi),
        int(resolved.lo),
    )


def _target_identity_parts_v838(target) -> tuple[object, ...]:
    boundary = target_boundary(target)
    uid = _uid_from_target(target)
    parts: list[object] = []
    if not uid.is_zero:
        parts.extend(("OUTCOME", int(uid.hi), int(uid.lo)))
    if boundary.crossed:
        parts.extend(
            (
                "BOUNDARY",
                boundary.scope.value,
                int(boundary.primary_valence),
                bool(boundary.continuation),
            )
        )
        if boundary.scope is BoundaryScope.SUBEPISODE:
            parts.append(max(0, int(getattr(target, "levels_completed", 0))))
    elif not parts:
        parts.extend(
            (
                "LOCAL",
                max(0, int(getattr(target, "levels_completed", 0))),
                str(getattr(target, "terminal_state", "")),
            )
        )
    return tuple(parts)


def _normalize_source(row):
    from v8 import trajectory_optimizer_v814 as optimizer

    uid = getattr(row, "target_outcome_uid", MemoryUid.zero())
    if uid is None:
        uid = MemoryUid.zero()
    target_uid = _uid_from_target(getattr(row, "target", None))
    if uid.is_zero and not target_uid.is_zero:
        uid = target_uid
    target = _target_with_outcome(row.target, uid)
    trajectory_id = optimizer._trajectory_id(row.anchor, target, row.actions)
    return replace(row, trajectory_id=trajectory_id, target=target, target_outcome_uid=uid)


def _candidate_id_v838(source, kind: str, actions) -> str:
    from v8 import trajectory_optimizer_v814 as optimizer

    source = _normalize_source(source)
    value = stable_u64(
        str(source.trajectory_id),
        int(source.target_outcome_uid.hi),
        int(source.target_outcome_uid.lo),
        str(kind),
        int(optimizer.action_sequence_hash(actions)),
        person=b"v8.38-candidate",
    )
    return f"{value:016x}"


def _seedless_candidate_id_v838(_optimizer, source, kind: str, actions) -> str:
    return _candidate_id_v838(source, kind, actions)


def _seedless_validated_id_v838(optimizer, anchor, target, actions, edit_kind: str) -> str:
    value = stable_u64(
        int(optimizer._anchor_hash(anchor, target)),
        int(optimizer.action_sequence_hash(actions)),
        str(edit_kind),
        person=b"v8.38-validated",
    )
    return f"{value:016x}"


def _successful_from_dict_v838(cls, raw):
    from v8 import trajectory_optimizer_v814 as optimizer

    anchor = optimizer.ReplayAnchor.from_dict(dict(raw.get("anchor", {})))
    target = V838TrajectoryTarget.from_dict(dict(raw.get("target", {})))
    actions = tuple(int(value) for value in raw.get("actions", ()))
    uid = optimizer._uid_from_raw(raw.get("target_outcome_uid"))
    if uid.is_zero:
        uid = _uid_from_target(target)
    target = _target_with_outcome(target, uid)
    return cls(
        optimizer._trajectory_id(anchor, target, actions),
        anchor,
        target,
        actions,
        optimizer._uid_from_raw(raw.get("parent_strategy_uid")),
        uid,
        int(raw.get("round_index", 0)),
    )


def _validated_from_dict_v838(cls, raw):
    from v8 import trajectory_optimizer_v814 as optimizer

    anchor = optimizer.ReplayAnchor.from_dict(dict(raw.get("anchor", {})))
    target = V838TrajectoryTarget.from_dict(dict(raw.get("target", {})))
    actions = tuple(int(value) for value in raw.get("actions", ()))
    uid = optimizer._uid_from_raw(raw.get("target_outcome_uid"))
    if uid.is_zero:
        uid = _uid_from_target(target)
    target = _target_with_outcome(target, uid)
    provisional = cls(
        _seedless_validated_id_v838(
            optimizer, anchor, target, actions, str(raw.get("edit_kind", ""))
        ),
        anchor,
        target,
        actions,
        MemoryUid.zero(),
        uid,
        optimizer._uid_from_raw(raw.get("parent_strategy_uid")),
        int(raw.get("parent_cost", 0)),
        str(raw.get("edit_kind", "")),
        int(raw.get("attempts", 1)),
        int(raw.get("successes", 1)),
    )
    candidate = SimpleNamespace(
        actions=provisional.actions,
        source=SimpleNamespace(
            target_outcome_uid=uid,
            anchor=anchor,
            target=target,
        ),
    )
    return replace(provisional, strategy_uid=optimizer.variant_strategy_uid(candidate))


def _write_source_v838(row) -> None:
    return _BASE_V819_WRITE(_normalize_source(row))


def _submit_trajectory_v838(self, trajectory) -> bool:
    return _BASE_SERVICE_SUBMIT(self, _normalize_source(trajectory))


# ---------------------------------------------------------------------------
# Adapter-authored transition authority.
# ---------------------------------------------------------------------------


def _arc_cognitive_transition(
    self,
    *,
    before_observation,
    after_observation,
    action_token: int,
    available_actions_before,
    available_actions_after,
) -> EnvironmentTransition:
    from v7.environment.encoding import (
        changed_cell_count,
        structural_grid_signature,
        transition_signature,
    )
    from v8 import environment_neutrality_v837 as v837

    before_context = int(structural_grid_signature(before_observation))
    after_context = int(structural_grid_signature(after_observation))
    delta_signature = int(transition_signature(before_observation, after_observation))
    changed = int(changed_cell_count(before_observation, after_observation))
    return EnvironmentTransition(
        before_observation,
        after_observation,
        int(action_token),
        tuple(int(value) for value in available_actions_before),
        tuple(int(value) for value in available_actions_after),
        {"transition_signature": delta_signature, "changed_cells": changed},
        v837._arc_boundary_event(self),
        before_context,
        after_context,
        changed > 0,
    )


def _arc_step_capture_v838(self, action):
    before = self.observe()
    before_actions = tuple(sorted({int(value) for value in self.available_actions()}))
    result = _BASE_V829_ENV_STEP(self, action)
    after = self.observe()
    after_actions = tuple(sorted({int(value) for value in self.available_actions()}))
    transition = self.cognitive_transition(
        before_observation=before,
        after_observation=after,
        action_token=int(action),
        available_actions_before=before_actions,
        available_actions_after=after_actions,
    )
    self._v838_last_transition = transition
    _TRANSITION_LOCAL.transition = transition
    return result


def _arc_reset_capture_v838(self, *args, **kwargs):
    result = _BASE_V829_ENV_RESET(self, *args, **kwargs)
    self._v838_last_transition = None
    try:
        delattr(_TRANSITION_LOCAL, "transition")
    except AttributeError:
        pass
    return result


def _transition_from_kwargs(kwargs) -> EnvironmentTransition | None:
    supplied = kwargs.get("environment_transition")
    if isinstance(supplied, EnvironmentTransition):
        return supplied
    recent = getattr(_TRANSITION_LOCAL, "transition", None)
    if not isinstance(recent, EnvironmentTransition):
        return None
    action = kwargs.get("action")
    if action is not None and int(action) != int(recent.action_token):
        return None
    after_actions = kwargs.get("after_actions")
    if after_actions is not None:
        observed = tuple(sorted({int(value) for value in after_actions}))
        if observed != tuple(sorted(recent.available_actions_after)):
            return None
    return recent


def _transition_semantics_v838(kwargs) -> TransitionSemantics:
    transition = _transition_from_kwargs(kwargs)
    if transition is None:
        return _BASE_V837_TRANSITION_SEMANTICS(kwargs)
    return TransitionSemantics(
        transition.boundary,
        structural_changed=bool(transition.structural_changed),
        context_changed=int(transition.before_context) != int(transition.after_context),
    )


# ---------------------------------------------------------------------------
# M6-aware replay validation.
# ---------------------------------------------------------------------------


def _runtime_outcome_matcher(runtime, environment_scope, context, action, outcome_signature, uid) -> bool:
    try:
        from v8 import behavior_recovery as behavior
        from v8.normalized_memory_v086_fixups import _grounded_context

        source_hash = stable_u64(str(environment_scope), person=b"v8-game")
        contexts = (
            int(_grounded_context(source_hash, int(context))),
            int(context),
        )
        for candidate_context in contexts:
            matches = behavior.observed_outcome_uids(
                runtime.read_view,
                context_signature=int(candidate_context),
                action_id=int(action),
                outcome_signature=int(outcome_signature),
            )
            if uid in matches:
                return True
    except BaseException:
        return False
    return False


def _match_outcome(service, env, source, context: int, action: int, outcome_signature: int) -> bool:
    uid = getattr(source, "target_outcome_uid", MemoryUid.zero())
    if uid is None or uid.is_zero:
        uid = _uid_from_target(source.target)
    if uid.is_zero:
        return True

    local = getattr(env, "matches_outcome_uid", None)
    if local is not None:
        try:
            matched = local(
                uid,
                context_signature=int(context),
                action_id=int(action),
                outcome_signature=int(outcome_signature),
            )
        except TypeError:
            matched = local(uid)
        if matched is not None:
            return bool(matched)

    callback = getattr(service, "_v838_outcome_matcher", None)
    if callback is None:
        return False
    return bool(
        callback(
            str(source.anchor.source_id),
            int(context),
            int(action),
            int(outcome_signature),
            uid,
        )
    )


def _boundary_matches(env, target) -> bool:
    boundary = target_boundary(target)
    if not boundary.crossed:
        return True
    getter = getattr(env, "cognitive_boundary_event", None)
    if getter is None:
        return False
    current = getter()
    return bool(
        current.scope is boundary.scope
        and int(current.primary_valence) == int(boundary.primary_valence)
    )


def _trial_v838(self, candidate, execution_seed: int, prefix: tuple[int, ...]):
    from v8 import environment_neutrality_v837 as v837

    env = self._environment(execution_seed, candidate.source.anchor.env_root)
    prefix_executed = 0
    for action in prefix:
        if not self._action_available(env, int(action)):
            return False, 0, "prefix_action_unavailable", 0, 0, 0, prefix_executed
        env.step(int(action))
        prefix_executed += 1
        if v837._generic_failed_boundary(env):
            return False, 0, "anchor_failed", 0, 0, 0, prefix_executed

    target_uid = getattr(candidate.source, "target_outcome_uid", MemoryUid.zero())
    if target_uid is None or target_uid.is_zero:
        target_uid = _uid_from_target(candidate.source.target)
    if target_uid.is_zero and v837._generic_target_reached(env, candidate.source):
        return False, 0, "anchor_already_reaches_target", 0, 0, 0, prefix_executed

    candidate_steps = 0
    for action in candidate.actions:
        if not self._action_available(env, int(action)):
            return False, candidate_steps, "candidate_action_unavailable", 0, 0, 0, prefix_executed
        before = env.observe()
        context = int(getattr(env, "cognitive_context_signature", lambda: 0)())
        after = env.step(int(action))
        candidate_steps += 1
        outcome = int(getattr(env, "cognitive_transition_signature", lambda _b, _a: 0)(before, after))

        if target_uid.is_zero:
            reached = v837._generic_target_reached(env, candidate.source)
        else:
            reached = bool(
                _match_outcome(self.service, env, candidate.source, context, int(action), outcome)
                and _boundary_matches(env, candidate.source.target)
            )
        if reached:
            return True, candidate_steps, "target_preserved", context, int(action), outcome, prefix_executed
        if v837._generic_failed_boundary(env):
            reason = "outcome_not_preserved" if not target_uid.is_zero else "candidate_failed"
            return False, candidate_steps, reason, 0, 0, 0, prefix_executed

    return False, candidate_steps, (
        "outcome_not_preserved" if not target_uid.is_zero else "target_not_reached"
    ), 0, 0, 0, prefix_executed


def _publish_resolved_validation_v838(runtime, candidate, result, validated, target_uid: MemoryUid) -> None:
    from v8 import trajectory_optimizer_v814 as optimizer

    target = _target_with_outcome(candidate.source.target, target_uid)
    source = replace(candidate.source, target=target, target_outcome_uid=target_uid)
    source = _normalize_source(source)
    resolved_candidate = replace(
        candidate,
        candidate_id=_candidate_id_v838(source, candidate.edit_kind, candidate.actions),
        source=source,
    )
    resolved = replace(
        validated,
        variant_id=resolved_candidate.candidate_id,
        target=target,
        target_outcome_uid=target_uid,
    )
    resolved = replace(resolved, strategy_uid=optimizer.variant_strategy_uid(resolved_candidate))
    service = runtime._v814_trajectory_optimizer
    key = optimizer._frontier_key(resolved.anchor, resolved.target)
    with service._lock:
        service._validated[key] = resolved
    service._publish_validated()
    optimizer._runtime_validation_callback(runtime, resolved_candidate, result, resolved)


def _runtime_init_under_v835_v838(self, *args, **kwargs):
    result = _BASE_V835_RUNTIME_INIT(self, *args, **kwargs)
    service = getattr(self, "_v814_trajectory_optimizer", None)
    if service is not None:
        service._v838_runtime = self
        service._v838_outcome_matcher = lambda environment_scope, context, action, outcome, uid: _runtime_outcome_matcher(
            self, environment_scope, context, action, outcome, uid
        )
    return result


# ---------------------------------------------------------------------------
# Generic solution persistence.
# ---------------------------------------------------------------------------


def _segments_from_raw(raw) -> tuple[tuple[int, ...], ...] | None:
    rows = raw.get("segments")
    if rows is None:
        rows = raw.get("levels")
    if not isinstance(rows, list) or not rows:
        return None
    result = []
    for item in rows:
        if not isinstance(item, dict):
            return None
        actions = item.get("actions")
        if not isinstance(actions, list) or not actions:
            return None
        try:
            result.append(tuple(int(value) for value in actions))
        except (TypeError, ValueError):
            return None
    return tuple(result)


def _target_payload_from_source(source) -> dict[str, object]:
    scope = optimization_scope_for(source)
    boundary = target_boundary(source.target)
    uid = getattr(source, "target_outcome_uid", MemoryUid.zero())
    if uid is None or uid.is_zero:
        uid = _uid_from_target(source.target)
    return {
        "kind": scope.kind.value,
        "label": scope.label(),
        "boundary_scope": boundary.scope.value,
        "primary_valence": int(boundary.primary_valence),
        "continuation": bool(boundary.continuation),
        "outcome_uid": [int(uid.hi), int(uid.lo)],
        "local_scope": int(getattr(scope, "local_scope", 0)),
    }


def _legacy_target_payload(raw) -> dict[str, object] | None:
    terminal = str(raw.get("terminal_state", ""))
    if terminal == "WIN":
        return {
            "kind": OptimizationScopeKind.BOUNDARY.value,
            "label": "EPISODE:+1",
            "boundary_scope": BoundaryScope.EPISODE.value,
            "primary_valence": +1,
            "continuation": False,
            "outcome_uid": [0, 0],
            "local_scope": 0,
        }
    return None


def _validated_solution_record_v838(raw: object) -> dict[str, object] | None:
    if not isinstance(raw, dict):
        return None
    environment = str(raw.get("environment_scope", raw.get("game_id", ""))).strip()
    source = str(raw.get("source", ""))
    if not environment or source not in {"observed", "optimized"}:
        return None
    target = raw.get("target")
    if not isinstance(target, dict):
        target = _legacy_target_payload(raw)
    if not isinstance(target, dict):
        return None
    segments = _segments_from_raw(raw)
    if segments is None:
        return None
    total_cost = sum(len(row) for row in segments)
    try:
        if int(raw.get("total_cost", -1)) != total_cost:
            return None
        attempts = max(1, int(raw.get("attempts", 1)))
        successes = max(0, int(raw.get("successes", 1)))
        reliability = max(0.0, min(1.0, float(raw.get("reliability", successes / attempts))))
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    trajectory_id = str(raw.get("trajectory_id", ""))
    variant_id = str(raw.get("variant_id", ""))
    if source == "observed" and not trajectory_id:
        return None
    if source == "optimized" and not variant_id:
        return None
    segment_payload = [
        {"segment": int(index), "actions": [int(value) for value in actions]}
        for index, actions in enumerate(segments)
    ]
    record: dict[str, object] = {
        "environment_scope": environment,
        "game_id": environment,
        "source": source,
        "target": dict(target),
        "total_cost": total_cost,
        "segments": segment_payload,
        "levels": [
            {"level": int(index), "actions": [int(value) for value in actions]}
            for index, actions in enumerate(segments)
        ],
        "attempts": attempts,
        "successes": successes,
        "reliability": reliability,
    }
    if trajectory_id:
        record["trajectory_id"] = trajectory_id
    if variant_id:
        record["variant_id"] = variant_id
    return record


def _load_best_successful_v838(path) -> dict[str, dict[str, object]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    environments = raw.get("environments")
    if not isinstance(environments, dict):
        environments = raw.get("games", {})
    if not isinstance(environments, dict):
        return {}
    result = {}
    for environment, item in environments.items():
        record = _validated_solution_record_v838(item)
        if record is not None and str(record["environment_scope"]) == str(environment):
            result[str(environment)] = record
    return result


def _persist_best_successful_v838(service) -> None:
    from v8 import trajectory_optimizer_v814 as optimizer

    with service._v819_solution_lock:
        environments = {
            environment: dict(record)
            for environment, record in sorted(service._v819_best_successful.items())
        }
        payload = {
            "version": 1,
            "target_schema_version": 2,
            "environments": environments,
            "games": environments,
        }
    optimizer._atomic_json(service.best_successful_path, payload)


def _publish_optimized_solution_v838(service, candidate, result, validated) -> bool:
    from v8 import environment_neutrality_v837 as v837
    from v8 import trajectory_inspection_v819 as inspection
    from v8 import trajectory_optimizer_convergence_v836 as convergence

    if validated is None:
        return False
    if not v837._global_target_source(candidate.source):
        return bool(_BASE_INSPECTION_PUBLISH(service, candidate, result, validated))
    segments = convergence._replay_full_win_levels(service, candidate)
    if segments is None:
        return False
    attempts = max(1, int(getattr(validated, "attempts", getattr(result, "attempts", 1))))
    successes = max(0, int(getattr(validated, "successes", getattr(result, "successes", 1))))
    record = {
        "environment_scope": str(candidate.source.anchor.source_id),
        "variant_id": str(validated.variant_id),
        "source": "optimized",
        "target": _target_payload_from_source(candidate.source),
        "total_cost": sum(len(segment) for segment in segments),
        "segments": [
            {"segment": index, "actions": list(segment)}
            for index, segment in enumerate(segments)
        ],
        "levels": [
            {"level": index, "actions": list(segment)}
            for index, segment in enumerate(segments)
        ],
        "attempts": attempts,
        "successes": successes,
        "reliability": float(successes) / float(attempts),
    }
    improved = bool(inspection._consider_best_solution(service, record))
    if improved:
        service._log(
            "optimized_solution",
            environment=str(candidate.source.anchor.source_id),
            target=str(record["target"]["label"]),
            cost=int(record["total_cost"]),
            edit=str(candidate.edit_kind),
            variant_id=str(validated.variant_id),
        )
    return improved


# ---------------------------------------------------------------------------
# Snapshot and optimizer identity migrations.
# ---------------------------------------------------------------------------


def _episode_scope_key() -> int:
    return int(
        OptimizationScope(
            OptimizationScopeKind.BOUNDARY,
            "",
            BoundaryScope.EPISODE,
            +1,
        ).legacy_budget_key()
    )


def _migrate_adaptive_state(raw: dict[str, object] | None) -> dict[str, object] | None:
    if not isinstance(raw, dict):
        return raw
    migrated = dict(raw)
    rows = []
    new_key = _episode_scope_key()
    for item in raw.get("game_level_states", ()):
        if not isinstance(item, dict):
            continue
        row = dict(item)
        if int(row.get("level", 0)) == _OLD_FULL_WIN_SCOPE_LEVEL:
            row["level"] = new_key
            row["scope_label"] = "EPISODE:+1"
        rows.append(row)
    migrated["game_level_states"] = rows
    migrated["scope_schema_version"] = 2
    return migrated


def _coordinator_load_state_v838(self, raw) -> None:
    return _BASE_COORDINATOR_LOAD_STATE(self, _migrate_adaptive_state(raw))


def _coordinator_state_dict_v838(self) -> dict[str, object]:
    raw = _BASE_COORDINATOR_STATE_DICT(self)
    raw["scope_schema_version"] = 2
    return raw


def _service_state_dict_under_v818_v838(service) -> dict[str, object]:
    raw = _BASE_V818_STATE_DICT(service)
    raw["identity_schema_version"] = 2
    return raw


def _service_load_state_under_v818_v838(service, raw) -> None:
    if not isinstance(raw, dict):
        return _BASE_V818_LOAD_STATE(service, raw)
    migrated = dict(raw)
    if int(migrated.get("identity_schema_version", 0)) < 2:
        migrated["seen_sources"] = []
        migrated["attempted"] = []
    migrated["identity_schema_version"] = 2
    return _BASE_V818_LOAD_STATE(service, migrated)


# ---------------------------------------------------------------------------
# Explicit TRANSFER fallback for structurally grounded M1N.
# ---------------------------------------------------------------------------


def _score_grounded_transfer_v838(view, context_signature, action_ids):
    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import environment_neutrality_v837 as v837
    from v8.publication import ActionScore

    rows = tuple(_BASE_V829_SCORE(view, context_signature, action_ids))
    mode = str(os.environ.get(v819._SAMPLING_MODE_ENV, v819.SamplingMode.DISCOVERY.value))
    if mode != v819.SamplingMode.TRANSFER.value:
        return rows
    environment = v837._current_game_id()
    if not environment:
        return rows
    _m7, m1n = v837._grounded_transfer_index(view, environment)
    result = []
    for row in rows:
        grounded = m1n.get(int(row.action_id), ())
        if not grounded or (int(row.support_count) > 0 and float(row.score) > 0.0):
            result.append(row)
            continue
        score = max(float(value[0]) for value in grounded)
        support = max(1, len(grounded))
        result.append(ActionScore(int(row.action_id), support, float(score), int(row.evidence_shards)))
    return tuple(result)


def install_environment_neutrality_v838() -> None:
    global _INSTALLED
    global _BASE_V829_ENV_STEP, _BASE_V829_ENV_RESET, _BASE_V829_SCORE
    global _BASE_V837_TRANSITION_SEMANTICS, _BASE_V819_WRITE, _BASE_SERVICE_SUBMIT
    global _BASE_V818_STATE_DICT, _BASE_V818_LOAD_STATE, _BASE_V835_RUNTIME_INIT
    global _BASE_COORDINATOR_STATE_DICT, _BASE_COORDINATOR_LOAD_STATE
    global _BASE_INSPECTION_PUBLISH
    if _INSTALLED:
        return

    from v7.environment.arc_adapter import ArcGridEnvironment
    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import environment_neutrality_v837 as v837
    from v8 import environment_neutrality_v837_integrity as integrity
    from v8 import runtime_win_scope_v835 as v835
    from v8 import sampling_progress_control_v829 as v829
    from v8 import trajectory_inspection_v819 as inspection
    from v8 import trajectory_optimizer_convergence_v836 as convergence
    from v8 import trajectory_optimizer_v814 as optimizer
    from v8 import trajectory_optimizer_v818 as v818

    optimizer.TrajectoryTarget = V838TrajectoryTarget
    integrity._target_identity_parts = _target_identity_parts_v838
    optimizer.SuccessfulTrajectory.from_dict = classmethod(_successful_from_dict_v838)
    optimizer.ValidatedTrajectory.from_dict = classmethod(_validated_from_dict_v838)
    optimizer._candidate_id = _candidate_id_v838
    v818._seedless_candidate_id = _seedless_candidate_id_v838
    v818._seedless_validated_id = _seedless_validated_id_v838

    _BASE_V819_WRITE = inspection._BASE_WRITE_SUCCESSFUL_TRAJECTORY
    inspection._BASE_WRITE_SUCCESSFUL_TRAJECTORY = _write_source_v838
    _BASE_SERVICE_SUBMIT = optimizer.TrajectoryOptimizationService.submit_trajectory
    optimizer.TrajectoryOptimizationService.submit_trajectory = _submit_trajectory_v838

    ArcGridEnvironment.cognitive_transition = _arc_cognitive_transition
    _BASE_V829_ENV_STEP = v829._BASE_ENV_STEP
    _BASE_V829_ENV_RESET = v829._BASE_ENV_RESET
    v829._BASE_ENV_STEP = _arc_step_capture_v838
    v829._BASE_ENV_RESET = _arc_reset_capture_v838
    _BASE_V837_TRANSITION_SEMANTICS = v837._transition_semantics
    v837._transition_semantics = _transition_semantics_v838

    v837._EnvironmentReplayValidator._trial = _trial_v838
    v818._publish_resolved_validation = _publish_resolved_validation_v838
    _BASE_V835_RUNTIME_INIT = v835._BASE_RUNTIME_INIT
    v835._BASE_RUNTIME_INIT = _runtime_init_under_v835_v838

    inspection._validated_solution_record = _validated_solution_record_v838
    inspection._load_best_successful = _load_best_successful_v838
    inspection._persist_best_successful = _persist_best_successful_v838
    _BASE_INSPECTION_PUBLISH = inspection._publish_optimized_solution
    inspection._publish_optimized_solution = _publish_optimized_solution_v838
    convergence._publish_optimized_solution_v836 = _publish_optimized_solution_v838

    _BASE_COORDINATOR_STATE_DICT = v819.AdaptiveLearningCoordinator.state_dict
    _BASE_COORDINATOR_LOAD_STATE = v819.AdaptiveLearningCoordinator.load_state
    v819.AdaptiveLearningCoordinator.state_dict = _coordinator_state_dict_v838
    v819.AdaptiveLearningCoordinator.load_state = _coordinator_load_state_v838

    _BASE_V818_STATE_DICT = v818._BASE_STATE_DICT
    _BASE_V818_LOAD_STATE = v818._BASE_LOAD_STATE
    v818._BASE_STATE_DICT = _service_state_dict_under_v818_v838
    v818._BASE_LOAD_STATE = _service_load_state_under_v818_v838

    _BASE_V829_SCORE = v829._BASE_SCORE_ACTIONS
    v829._BASE_SCORE_ACTIONS = _score_grounded_transfer_v838

    _INSTALLED = True
