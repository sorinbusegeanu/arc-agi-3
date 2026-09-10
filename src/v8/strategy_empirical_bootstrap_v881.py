from __future__ import annotations

from collections import deque

from v8.actor import ActorResult, StrategyRunStat, _stats_tuple, _trajectory_step_cost
from v8.behavior_recovery import _score_strategy_rows, _strategy_can_probe
from v8.model import stable_u64
from v8.publication import LiveReadView


_INSTALLED = False
_BASE_PLAN_CANDIDATES = None
_BASE_RUN_MIXED_ACTOR_JOBS = None
_BOOTSTRAP_ATTEMPTS = 3


def _missing_bootstrap_attempts(row, local_attempts: int) -> int:
    committed = max(0, int(float(getattr(row, "attempt_weight", 0.0))))
    return max(0, _BOOTSTRAP_ATTEMPTS - committed - max(0, int(local_attempts)))


def _bootstrap_plan(view: LiveReadView, context_signature: int, action_ids):
    if not (
        bool(getattr(view, "_behavior_actor_mode", False))
        or bool(getattr(view, "_v881_generic_actor_mode", False))
    ):
        return None
    view._refresh_strategy_cache()
    context_bucket = stable_u64(int(context_signature), person=b"v8-context")
    available = {int(value) for value in action_ids}
    local = getattr(view, "_v881_bootstrap_attempts", None)
    if local is None:
        local = {}
        view._v881_bootstrap_attempts = local

    eligible = []
    effective_attempts = {}
    for item in tuple(getattr(view, "_strategy_by_context", {}).get(context_bucket, ())):
        source = getattr(view, "_node_by_uid", {}).get(item.strategy_uid)
        if source is None:
            continue
        local_count = int(local.get(item.strategy_uid, 0))
        if _missing_bootstrap_attempts(source, local_count) <= 0:
            continue
        if not _strategy_can_probe(view, item.strategy_uid, item.outcome_uid):
            continue
        if int(item.action_id) not in available:
            continue
        eligible.append(item)
        effective_attempts[item.strategy_uid] = float(source.attempt_weight) + local_count

    if not eligible:
        return None
    probes = _score_strategy_rows(
        view,
        eligible,
        available=available,
        outcome_uid=None,
        required_ancestor=None,
        excluded_strategies=frozenset(),
        ignore_preference=True,
        cross_context=False,
    )
    if not probes:
        return None
    chosen = min(
        probes,
        key=lambda item: (
            effective_attempts.get(item.strategy_uid, 0.0),
            int(item.action_id),
            item.strategy_uid,
        ),
    )
    local[chosen.strategy_uid] = int(local.get(chosen.strategy_uid, 0)) + 1
    view._behavior_last_plans = (chosen,)
    return chosen


def _plan_candidates_v881(self: LiveReadView, context_signature: int, action_ids, **kwargs):
    # Explicit replanning/target-selection calls keep their original semantics.
    if not kwargs:
        bootstrap = _bootstrap_plan(self, context_signature, action_ids)
        if bootstrap is not None:
            return (bootstrap,)
    return _BASE_PLAN_CANDIDATES(self, context_signature, action_ids, **kwargs)


def _selected_generic_plan(view, action: int, planned: bool):
    if not planned:
        return None
    for row in tuple(getattr(view, "_behavior_last_plans", ())):
        if int(row.action_id) == int(action):
            return row
    return None


def _accumulate_generic_strategy_stat(stats, plan, *, success: bool, cost: float) -> None:
    if plan is None:
        return
    row = stats.setdefault(plan.strategy_uid, [0.0, 0.0, 0.0])
    row[0] += 1.0
    row[1] += float(bool(success))
    row[2] += max(0.0, float(cost))


def _run_generic_actor_job_v881(runtime, job, *, reporting_queue=None) -> ActorResult:
    from random import Random
    import time
    from v8 import mixed_environment_v859 as mixed

    if not mixed.is_generic_game(job.game_id):
        raise ValueError(f"{job.game_id!r} is not a generic mixed-environment game")
    restored = mixed._restored_generic_candidate(job.game_id)
    restored_actions = (
        tuple(int(value) for value in restored.get("actions", ()))
        if isinstance(restored, dict)
        else ()
    )
    restored_index = 0
    restored_active = bool(restored_actions)
    adapter_seed = int(restored.get("seed", job.seed)) if isinstance(restored, dict) else int(job.seed)
    adapter = mixed.make_adapter(job.game_id, seed=adapter_seed)
    view, owns_view = mixed._generic_read_view(runtime)
    view._v881_generic_actor_mode = True
    rng = Random(int(job.seed))
    sequence = wins = failures = resets = planned_steps = 0
    strategy_stats = {}
    recent_contexts = deque(maxlen=8)
    sequence_base = max(0, int(getattr(runtime, "watermark", 0)))
    trajectory = mixed.trajectory_identity(
        adapter.identity.source_hash,
        producer_id=job.actor_id,
        episode_ordinal=0,
        sequence_base=sequence_base,
        namespace=b"v8.59-mix-trajectory",
    )
    next_progress = time.monotonic() + 5.0
    graph_check_steps = max(1, int(job.graph_check_steps))
    try:
        for requested_step in range(1, int(job.steps) + 1):
            if getattr(runtime, "_stop", None) is not None and runtime._stop.is_set():
                break
            if requested_step > 1 and (requested_step - 1) % graph_check_steps == 0:
                invalidate = getattr(view, "invalidate_strategy_cache", None)
                if callable(invalidate):
                    invalidate()
            before = adapter.observe()
            before_actions = tuple(sorted(set(map(int, adapter.available_actions()))))
            if not before_actions:
                adapter.reset()
                resets += 1
                recent_contexts.clear()
                trajectory = mixed.trajectory_identity(
                    adapter.identity.source_hash,
                    producer_id=job.actor_id,
                    episode_ordinal=resets,
                    sequence_base=sequence_base,
                    namespace=b"v8.59-mix-trajectory",
                )
                continue
            context = int(adapter.observation_signature(before))
            chosen_plan = None
            if restored_active:
                restored_action = int(restored_actions[restored_index])
                if restored_action in before_actions:
                    action, planned = restored_action, True
                    restored_index += 1
                else:
                    restored_active = False
                    action, planned = mixed._choose_action(view, context, before_actions, rng, float(job.epsilon))
                    chosen_plan = _selected_generic_plan(view, action, planned)
            else:
                action, planned = mixed._choose_action(view, context, before_actions, rng, float(job.epsilon))
                chosen_plan = _selected_generic_plan(view, action, planned)
            planned_steps += int(planned)
            distribution = view.outcome_distribution(context, action)
            after = adapter.step(action)
            after_actions = tuple(sorted(set(map(int, adapter.available_actions()))))
            after_context = int(adapter.observation_signature(after))
            outcome = int(adapter.cognitive_transition_signature(before, after))
            family = int(adapter.cognitive_family_signature(before, after))
            changed = max(0, int(adapter.cognitive_changed_extent(before, after)))
            boundary = adapter.cognitive_boundary_event()
            prediction_error = 0.0 if not distribution else max(0.0, 1.0 - float(distribution.get(outcome, 0.0)))
            trajectory = stable_u64(trajectory, context, action, outcome, person=b"v8.59-mix-trajectory")
            observation_schema = getattr(adapter, "observation_schema", None) or adapter.observation_codec.schema
            action_schema = getattr(adapter, "action_schema", None) or adapter.action_codec.schema
            carrier = stable_u64(
                int(observation_schema.schema_id),
                int(action_schema.schema_id),
                person=b"v8.59-schema-carrier",
            )
            producer_sequence = sequence_base + requested_step
            event = runtime.make_experience(
                producer_id=int(job.actor_id),
                producer_sequence=producer_sequence,
                source_game_hash=int(adapter.identity.source_hash),
                global_step=max(0, int(runtime.watermark)),
                context_signature=context,
                action_id=int(action),
                outcome_signature=outcome,
                family_signature=family,
                carrier_signature=carrier,
                future_option_delta=float(len(after_actions) - len(before_actions)),
                changed_cells=changed,
                terminal_polarity=int(boundary.primary_valence),
                trajectory_signature=trajectory,
                next_context_signature=after_context,
                prediction_error=prediction_error,
            )
            runtime.submit(event)
            sequence += 1

            if chosen_plan is not None:
                observed = set(
                    view.observed_outcome_uids(
                        context_signature=context,
                        action_id=action,
                        outcome_signature=outcome,
                    )
                )
                success = bool(boundary.primary_valence > 0 or chosen_plan.outcome_uid in observed)
                cost = _trajectory_step_cost(
                    context=context,
                    after_context=after_context,
                    changed_cells=changed,
                    negative_outcome=boundary.primary_valence < 0,
                    recent_contexts=tuple(recent_contexts),
                )
                _accumulate_generic_strategy_stat(strategy_stats, chosen_plan, success=success, cost=cost)
            recent_contexts.append(context)

            if not boundary.continuation:
                if boundary.primary_valence > 0:
                    wins += 1
                elif boundary.primary_valence < 0:
                    failures += 1
                restored_active = False
                adapter.reset()
                resets += 1
                recent_contexts.clear()
                trajectory = mixed.trajectory_identity(
                    adapter.identity.source_hash,
                    producer_id=job.actor_id,
                    episode_ordinal=resets,
                    sequence_base=producer_sequence,
                    namespace=b"v8.59-mix-trajectory",
                )
            elif restored_active and restored_index >= len(restored_actions):
                restored_active = False
                adapter.reset()
                resets += 1
                recent_contexts.clear()
                trajectory = mixed.trajectory_identity(
                    adapter.identity.source_hash,
                    producer_id=job.actor_id,
                    episode_ordinal=resets,
                    sequence_base=producer_sequence,
                    namespace=b"v8.59-mix-trajectory",
                )
            now = time.monotonic()
            if now >= next_progress:
                mixed._publish_generic_progress(
                    reporting_queue,
                    job,
                    steps=sequence,
                    wins=wins,
                    failures=failures,
                    planned_steps=planned_steps,
                )
                next_progress = now + 5.0
        mixed._publish_generic_progress(
            reporting_queue,
            job,
            steps=sequence,
            wins=wins,
            failures=failures,
            planned_steps=planned_steps,
        )
        return ActorResult(
            job.actor_id,
            job.game_id,
            sequence,
            wins,
            failures,
            0,
            resets,
            0,
            planned_steps,
            _stats_tuple(strategy_stats),
        )
    finally:
        if owns_view:
            view.close()
        adapter.close()


def _run_mixed_actor_jobs_v881(runtime, jobs, **kwargs):
    from v8 import mixed_environment_v859 as mixed

    results = tuple(_BASE_RUN_MIXED_ACTOR_JOBS(runtime, jobs, **kwargs))
    generic = tuple(
        row
        for row in results
        if mixed.is_generic_game(row.game_id) and tuple(getattr(row, "strategy_stats", ()))
    )
    if generic:
        runtime.record_actor_results(generic)
    return results


def install_strategy_empirical_bootstrap_v881() -> None:
    global _INSTALLED, _BASE_PLAN_CANDIDATES, _BASE_RUN_MIXED_ACTOR_JOBS
    if _INSTALLED:
        return
    from v8 import mixed_environment_v859 as mixed

    _BASE_PLAN_CANDIDATES = LiveReadView.plan_candidates
    _BASE_RUN_MIXED_ACTOR_JOBS = mixed.run_mixed_actor_jobs
    # Bootstrap is invoked by the final v8.26 planner authority. Keeping that
    # function installed avoids a late monkey-patch hiding the control contract.
    mixed.run_generic_actor_job = _run_generic_actor_job_v881
    mixed.run_mixed_actor_jobs = _run_mixed_actor_jobs_v881
    _INSTALLED = True
