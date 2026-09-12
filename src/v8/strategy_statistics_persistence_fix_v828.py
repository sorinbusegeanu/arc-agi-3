from __future__ import annotations

from collections import defaultdict

from v8.model import CognitiveState, MemoryLevel, MemoryType, MemoryUid


_INSTALLED = False
_BASE_LEARNING_BATCH = None
_BASE_RUN_ACTOR_JOBS = None
_BASE_RECORD_ACTOR_RESULTS = None
_BASE_PEER_RUN_ONCE = None


def _stats_map(rows) -> dict[MemoryUid, list[float]]:
    result: dict[MemoryUid, list[float]] = {}
    for row in rows:
        attempts = max(0, int(getattr(row, "attempts", 0)))
        if attempts <= 0:
            continue
        result[row.strategy_uid] = [
            float(attempts),
            float(max(0, int(getattr(row, "successes", 0)))),
            float(max(0.0, float(getattr(row, "cost", 0.0)))),
        ]
    return result


def _merge_strategy_stats(
    actor_module, strategy_stats, *, prefer_tracker: bool = False
) -> tuple[object, ...]:
    """Preserve normal actor stats; add tracker-only UIDs without double counting."""
    from v8.trajectory_efficiency_v054 import _TRACKER

    merged: dict[MemoryUid, list[float]] = {
        uid: [float(values[0]), float(values[1]), float(values[2])]
        for uid, values in strategy_stats.items()
        if float(values[0]) > 0.0
    }
    for uid, values in _TRACKER.stats.items():
        if float(values[0]) <= 0.0:
            continue
        if prefer_tracker or uid not in merged:
            merged[uid] = [float(values[0]), float(values[1]), float(values[2])]
    return tuple(
        actor_module.StrategyRunStat(uid, int(values[0]), int(values[1]), float(values[2]))
        for uid, values in sorted(merged.items())
        if values[0] > 0.0
    )


def _learning_batch_v828(*, job, strategy_stats, preference_probes, replanning_trials):
    from v8 import actor as actor_module
    from v8 import primary_valence as primary
    from v8.trajectory_efficiency_v054 import _TRACKER

    _TRACKER.flush_open_run()
    stats = _merge_strategy_stats(actor_module, strategy_stats, prefer_tracker=True)
    credits = primary._credit_tuple()
    preferences = tuple(primary._PENDING_VALENCE_PREFERENCES)
    if not stats and not preference_probes and not replanning_trials and not credits and not preferences:
        return None
    return primary.PrimaryValenceLearningBatch(
        int(job.actor_id),
        str(job.game_id),
        stats,
        tuple(preference_probes),
        tuple(replanning_trials),
        len(replanning_trials),
        credits,
        preferences,
    )


def _record_actor_results_v828(self, results) -> None:
    rows = tuple(results)
    if bool(getattr(self, "_v828_strategy_final_accounting_active", False)):
        published = getattr(self, "_v828_strategy_published", None)
        if published is None:
            published = defaultdict(lambda: [0.0, 0.0, 0.0])
            self._v828_strategy_published = published
        for row in rows:
            for stat in getattr(row, "strategy_stats", ()):
                key = (int(row.actor_id), str(row.game_id), stat.strategy_uid)
                values = published[key]
                values[0] += float(max(0, int(stat.attempts)))
                values[1] += float(max(0, int(stat.successes)))
                values[2] += float(max(0.0, float(stat.cost)))
    return _BASE_RECORD_ACTOR_RESULTS(self, rows)


def _final_residual_batches(actor_module, results, published) -> tuple[object, ...]:
    batches = []
    for result in results:
        residual = []
        for stat in getattr(result, "strategy_stats", ()):
            key = (int(result.actor_id), str(result.game_id), stat.strategy_uid)
            prior = published.get(key, (0.0, 0.0, 0.0))
            attempts = max(0, int(stat.attempts) - int(round(float(prior[0]))))
            if attempts <= 0:
                continue
            successes = max(0, int(stat.successes) - int(round(float(prior[1]))))
            cost = max(0.0, float(stat.cost) - float(prior[2]))
            residual.append(
                actor_module.StrategyRunStat(
                    stat.strategy_uid,
                    attempts,
                    successes,
                    cost,
                )
            )
        if residual:
            batches.append(
                actor_module.ActorLearningBatch(
                    int(result.actor_id),
                    str(result.game_id),
                    tuple(residual),
                    (),
                    (),
                    0,
                )
            )
    return tuple(batches)


def _run_actor_jobs_v828(runtime, jobs, **kwargs):
    """Keep result reporting separate from canonical incremental learning.

    The base runner records every queued learning batch and records an actor's
    ``pending_learning`` exactly once when the final queue write fails.  The
    cumulative counters on ``ActorResult`` use per-step units, while the
    trajectory tracker batches use completed-run units, so subtracting one from
    the other creates spurious residual attempts.
    """
    return _BASE_RUN_ACTOR_JOBS(runtime, jobs, **kwargs)


def _emit_committed_strategy_efficiency(supervisor) -> int:
    """Emit H12 evidence only from committed empirical same-outcome/context cohorts."""
    # During immutable-cut processing ``supervisor.read_view`` is deliberately
    # replaced by a frozen formation cut. Actor statistics commit independently
    # to the canonical live view and may be newer than that cut.
    runtime = getattr(getattr(supervisor, "submit_proposal", None), "__self__", None)
    view = (
        getattr(runtime, "read_view", None)
        or getattr(supervisor, "_v813_live_read_view", None)
        or supervisor.read_view
    )
    nodes = tuple(view.node_records())
    by_uid = {row.uid: row for row in nodes}
    empirical_states = {
        int(CognitiveState.CANDIDATE),
        int(CognitiveState.PROBATION),
        int(CognitiveState.ACTIVE),
        int(CognitiveState.VALIDATED),
        int(CognitiveState.REACTIVATED),
    }
    emitted = 0
    considered = 0
    rejection_counts: dict[str, int] = defaultdict(int)
    grouped: dict[tuple[int, MemoryUid], list[object]] = defaultdict(list)
    for source in nodes:
        if (
            int(source.level) == int(MemoryLevel.M7)
            and int(source.memory_type) == int(MemoryType.STRATEGY)
            and len(source.key_parts) >= 4
        ):
            outcome_uid = MemoryUid(
                int(source.key_parts[1]), int(source.key_parts[2])
            )
            grouped[(int(source.key_parts[3]), outcome_uid)].append(source)

    for (context_bucket, outcome_uid), cohort in grouped.items():
        empirical = []
        outcome = by_uid.get(outcome_uid)
        if outcome is None or int(outcome.level) != int(MemoryLevel.M6):
            rejection_counts["same_cohort_m6_outcome_unavailable"] += len(cohort)
            continue
        for source in cohort:
            considered += 1
            if int(source.cognitive_state) not in empirical_states:
                rejection_counts["strategy_not_empirically_admissible"] += 1
                continue
            if float(source.attempt_weight) <= 0.0:
                rejection_counts["strategy_without_empirical_attempt"] += 1
                continue
            empirical.append(source)
        if len(empirical) < 2:
            if empirical:
                rejection_counts["no_same_m6_outcome_comparator"] += len(empirical)
            continue
        mean_costs = {
            source.uid: max(
                1e-9, float(source.cost_sum) / float(source.attempt_weight)
            )
            for source in empirical
        }
        best = min(mean_costs.values())
        for source in empirical:
            if not supervisor._fresh(
                f"efficiency_committed:{int(context_bucket)}",
                source.uid,
                source.updated_watermark,
            ):
                continue
            value = best / mean_costs[source.uid]
            supervisor._append_evidence("strategy_efficiency", source, value)
            emitted += 1
    from v8 import information_flow_diagnostics as flow

    flow.emit_bounded(
        "strategy_efficiency",
        "matched_m6_outcome_cost_comparison",
        input_count=considered,
        output_count=emitted,
        rejection_counts=dict(rejection_counts),
    )
    return emitted


def _peer_run_once_v828(self) -> None:
    cancelled = getattr(self, "_v841_peer_cancel", None)
    if cancelled is not None and cancelled.is_set():
        return
    # Evaluate the canonical actor-statistics cut before the base peer pass can
    # publish lifecycle proposals for the same M7 rows.  Those proposals are
    # intentionally zero-delta, but later generation reconciliation may make a
    # stale pre-stat row visible until its next commit barrier.
    _emit_committed_strategy_efficiency(self)
    _BASE_PEER_RUN_ONCE(self)


def install_strategy_statistics_persistence_fix_v828() -> None:
    global _INSTALLED
    global _BASE_LEARNING_BATCH, _BASE_RUN_ACTOR_JOBS
    global _BASE_RECORD_ACTOR_RESULTS, _BASE_PEER_RUN_ONCE
    if _INSTALLED:
        return

    from v8 import actor as actor_module
    from v8 import peers_v82 as peers_v82_module
    from v8.runtime_v82 import V82ContinuousMemoryRuntime

    _BASE_LEARNING_BATCH = actor_module._learning_batch
    _BASE_RUN_ACTOR_JOBS = actor_module.run_actor_jobs
    _BASE_RECORD_ACTOR_RESULTS = V82ContinuousMemoryRuntime.record_actor_results
    _BASE_PEER_RUN_ONCE = peers_v82_module.V82DevelopmentalPeerSupervisor.run_once

    actor_module._learning_batch = _learning_batch_v828
    actor_module.run_actor_jobs = _run_actor_jobs_v828
    V82ContinuousMemoryRuntime.record_actor_results = _record_actor_results_v828
    peers_v82_module.V82DevelopmentalPeerSupervisor.run_once = _peer_run_once_v828
    _INSTALLED = True
