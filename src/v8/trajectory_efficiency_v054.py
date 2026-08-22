from __future__ import annotations

import math
from dataclasses import dataclass, replace

from v8 import primary_valence as _primary
from v8.model import CognitiveState, MemoryLevel, MemoryType, MemoryUid, stable_u64


_RELATIVE_EFFICIENCY_WEIGHT = 0.15
_INSTALLED = False
_BASE_OBSERVED = None
_BASE_LEARNING_BATCH = None
_BASE_PUBLISH_LEARNING = None
_BASE_RESET_CAPTURE = None
_BASE_SCORE_ROWS = None
_BASE_RUNTIME_RECORD_RESULTS = None
_BASE_APPEND_EVIDENCE = None


@dataclass(slots=True)
class _ActiveRun:
    strategy_uid: MemoryUid
    outcome_uid: MemoryUid
    start_step: int


class TrajectoryEfficiencyTracker:
    """Track realized action count from M7 strategy selection to M6 outcome/boundary."""

    def __init__(self) -> None:
        self.episode_step = 0
        self.active: _ActiveRun | None = None
        self.stats: dict[MemoryUid, list[float]] = {}

    def reset(self) -> None:
        self.episode_step = 0
        self.active = None
        self.stats.clear()

    def clear_published(self) -> None:
        self.stats.clear()

    def _finish(self, *, success: bool, end_step: int) -> None:
        active = self.active
        if active is None:
            return
        cost = max(1, int(end_step) - int(active.start_step) + 1)
        row = self.stats.setdefault(active.strategy_uid, [0.0, 0.0, 0.0])
        row[0] += 1.0
        row[1] += float(bool(success))
        row[2] += float(cost)
        self.active = None

    def observe(self, plan, observed: tuple[MemoryUid, ...], *, terminal: bool) -> None:
        current_step = self.episode_step + 1
        strategy_uid = None if plan is None else plan.strategy_uid

        if self.active is not None and strategy_uid != self.active.strategy_uid:
            previous_end = max(self.active.start_step, current_step - 1)
            self._finish(success=False, end_step=previous_end)

        if plan is not None and self.active is None:
            self.active = _ActiveRun(plan.strategy_uid, plan.outcome_uid, current_step)

        if self.active is not None and self.active.outcome_uid in set(observed):
            self._finish(success=True, end_step=current_step)

        if terminal:
            if self.active is not None:
                self._finish(success=False, end_step=current_step)
            self.episode_step = 0
        else:
            self.episode_step = current_step

    def flush_open_run(self) -> None:
        if self.active is not None:
            self._finish(success=False, end_step=max(self.active.start_step, self.episode_step))


_TRACKER = TrajectoryEfficiencyTracker()


def _relative_efficiency(rows) -> dict[MemoryUid, float]:
    """Normalize efficiency only inside outcome-comparable M7 cohorts."""
    grouped: dict[tuple[MemoryUid, int], list[tuple[MemoryUid, float]]] = {}
    for row in rows:
        outcome_uid = getattr(row, "outcome_uid", None)
        strategy_uid = getattr(row, "strategy_uid", None)
        if outcome_uid is None or strategy_uid is None:
            continue
        cost = max(1e-9, float(getattr(row, "mean_cost", 1.0)))
        context_bucket = int(getattr(row, "context_bucket", 0))
        grouped.setdefault((outcome_uid, context_bucket), []).append((strategy_uid, cost))

    result: dict[MemoryUid, float] = {}
    for members in grouped.values():
        if len(members) < 2:
            continue
        best = min(cost for _uid, cost in members)
        for uid, cost in members:
            result[uid] = max(0.0, min(1.0, best / max(1e-9, cost)))
    return result


def _reset_actor_capture_v054() -> None:
    if _BASE_RESET_CAPTURE is not None:
        _BASE_RESET_CAPTURE()
    _TRACKER.reset()


def _observed_outcome_uids_v054(**kwargs):
    entry = _primary._TRAJECTORY[-1] if _primary._TRAJECTORY else None
    plan = None if entry is None else entry.get("plan")
    result = _BASE_OBSERVED(**kwargs)
    observed = tuple(uid for uid in result if not uid.is_zero)
    _TRACKER.observe(plan, observed, terminal=int(kwargs.get("terminal_polarity", 0)) != 0)
    return result


def _strategy_stats_tuple(actor_module):
    return tuple(
        actor_module.StrategyRunStat(uid, int(values[0]), int(values[1]), float(values[2]))
        for uid, values in sorted(_TRACKER.stats.items())
        if values[0] > 0
    )


def _learning_batch_v054(*, job, strategy_stats, preference_probes, replanning_trials):
    del preference_probes
    _TRACKER.flush_open_run()
    from v8 import actor as actor_module

    stats = _strategy_stats_tuple(actor_module)
    if not stats and not bool(_primary._CAPTURE_ACTIVE):
        stats = tuple(
            actor_module.StrategyRunStat(uid, int(values[0]), int(values[1]), float(values[2]))
            for uid, values in sorted(strategy_stats.items())
            if values[0] > 0
        )
    credits = _primary._credit_tuple()
    preferences = tuple(_primary._PENDING_VALENCE_PREFERENCES)
    if not stats and not replanning_trials and not credits and not preferences:
        return None
    return _primary.PrimaryValenceLearningBatch(
        int(job.actor_id),
        str(job.game_id),
        stats,
        (),
        tuple(replanning_trials),
        len(replanning_trials),
        credits,
        preferences,
    )


def _publish_learning_v054(*args, **kwargs):
    published = _BASE_PUBLISH_LEARNING(*args, **kwargs)
    if published:
        _TRACKER.clear_published()
    return published


def _emit_terminal_credits_v054(primary_valence: int, behavior_module) -> None:
    """Back-propagate valence through causal outcome-achieving strategy trajectories."""
    if primary_valence == 0:
        return
    suffix_observed: set[MemoryUid] = set()
    view = behavior_module._CURRENT_ACTOR_VIEW
    for distance, entry in enumerate(reversed(tuple(_primary._TRAJECTORY))):
        value = float(primary_valence) * (_primary._VALENCE_GAMMA ** distance)
        event = entry["event"]
        observed = tuple(uid for uid in entry.get("observed", ()) if not uid.is_zero)
        suffix_observed.update(observed)

        if distance > 0:
            _primary._accumulate_credit(
                entry["m0_uid"],
                level=int(MemoryLevel.M0),
                memory_type=int(MemoryType.EPISODE),
                key_parts=entry["m0_key"],
                fingerprint=_primary._model.proposal_fingerprint(
                    MemoryLevel.M0, MemoryType.EPISODE, entry["m0_key"]
                ),
                value=value,
            )
            _primary._accumulate_credit(
                entry["m1_uid"],
                level=int(MemoryLevel.M1),
                memory_type=int(MemoryType.CONTINGENCY),
                key_parts=entry["m1_key"],
                fingerprint=_primary._model.proposal_fingerprint(
                    MemoryLevel.M1, MemoryType.CONTINGENCY, entry["m1_key"]
                ),
                value=value,
            )

        if observed and view is not None:
            outcome_uid = observed[0]
            outcome_row = getattr(view, "_node_by_uid", {}).get(outcome_uid)
            if outcome_row is not None:
                _primary._accumulate_credit(
                    outcome_uid,
                    level=int(outcome_row.level),
                    memory_type=int(outcome_row.memory_type),
                    key_parts=tuple(int(v) for v in outcome_row.key_parts),
                    fingerprint=int(outcome_row.fingerprint),
                    value=value,
                )

        plan = entry.get("plan")
        if plan is None or view is None or plan.outcome_uid not in suffix_observed:
            continue
        strategy_row = getattr(view, "_node_by_uid", {}).get(plan.strategy_uid)
        if strategy_row is not None:
            _primary._accumulate_credit(
                plan.strategy_uid,
                level=int(strategy_row.level),
                memory_type=int(strategy_row.memory_type),
                key_parts=tuple(int(v) for v in strategy_row.key_parts),
                fingerprint=int(strategy_row.fingerprint),
                value=value,
            )
        alternative = entry.get("alternative")
        if (
            value >= _primary._PREFERENCE_CREDIT_THRESHOLD
            and alternative is not None
            and not bool(plan.preference_influenced)
        ):
            _primary._PENDING_VALENCE_PREFERENCES.append(
                _primary.PrimaryValencePreference(
                    plan.outcome_uid,
                    alternative.outcome_uid,
                    stable_u64(int(event.context_signature), person=b"v8-context"),
                    float(value),
                )
            )


def _score_strategy_rows_v054(view, rows, **kwargs):
    plans = list(_BASE_SCORE_ROWS(view, rows, **kwargs))
    relative = _relative_efficiency(rows)
    row_by_uid = {row.strategy_uid: row for row in rows}
    adjusted = []
    for plan in plans:
        row = row_by_uid.get(plan.strategy_uid)
        if row is None:
            adjusted.append(plan)
            continue
        absolute_bonus = 0.10 / max(1e-9, float(row.mean_cost))
        relative_bonus = _RELATIVE_EFFICIENCY_WEIGHT * relative.get(plan.strategy_uid, 0.0)
        adjusted.append(replace(plan, score=float(plan.score) - absolute_bonus + relative_bonus))
    adjusted.sort(key=lambda item: (-item.score, item.action_id, item.strategy_uid))
    return tuple(adjusted)


def _append_evidence_v054(self, kind, row, value, **kwargs):
    if kind == "strategy_efficiency":
        outcome_uid = (
            MemoryUid(int(row.key_parts[1]), int(row.key_parts[2]))
            if len(row.key_parts) >= 3
            else MemoryUid.zero()
        )
        context_bucket = int(row.key_parts[3]) if len(row.key_parts) >= 4 else 0
        by_uid = getattr(self.read_view, "_node_by_uid", {})
        active_states = {
            int(CognitiveState.ACTIVE),
            int(CognitiveState.VALIDATED),
            int(CognitiveState.REACTIVATED),
        }
        cohort = tuple(
            item
            for item in getattr(self.read_view, "_strategy_by_context", {}).get(
                context_bucket, ()
            )
            if item.outcome_uid == outcome_uid
        )
        empirical = []
        for item in cohort:
            source = by_uid.get(item.strategy_uid)
            if source is None:
                continue
            attempts = max(0, int(round(float(source.attempt_weight))))
            if attempts <= 0 or int(source.cognitive_state) not in active_states:
                continue
            empirical.append(item)
        if len(empirical) < 2:
            return None
        best = min(item.mean_cost for item in empirical)
        current = next(
            (item for item in empirical if item.strategy_uid == row.uid),
            None,
        )
        if current is None:
            return None
        value = best / max(1e-9, current.mean_cost)
    return _BASE_APPEND_EVIDENCE(self, kind, row, value, **kwargs)


def _actions_from_discounted_valence(credit) -> float | None:
    weight = max(0.0, float(getattr(credit, "weight", 0.0)))
    if weight <= 0.0:
        return None
    mean = abs(float(getattr(credit, "valence_sum", 0.0))) / weight
    if mean <= 0.0:
        return None
    mean = min(1.0, mean)
    if mean >= 1.0 - 1e-12:
        return 1.0
    gamma = float(_primary._VALENCE_GAMMA)
    if gamma <= 0.0 or gamma >= 1.0:
        return None
    distance = math.log(mean) / math.log(gamma)
    return max(1.0, 1.0 + distance)


def _record_actor_results_v054(self, results):
    results = tuple(results)
    _BASE_RUNTIME_RECORD_RESULTS(self, results)
    if self.peers is None:
        return
    from v8.evidence import EvidenceRecord

    for result in results:
        game_hash = stable_u64(result.game_id, person=b"v8-game")
        for credit in getattr(result, "primary_valence_credits", ()):
            if int(credit.level) != int(MemoryLevel.M7):
                continue
            actions = _actions_from_discounted_valence(credit)
            if actions is None:
                continue
            self.peers.ledger.append(
                EvidenceRecord.for_uid(
                    f"primary-valence-efficiency:{result.actor_id}:{credit.uid.hex()}:{self.watermark}",
                    credit.uid,
                    evidence_kind="primary_valence_efficiency",
                    watermark=self.watermark,
                    raw_value=float(actions),
                    normalized_value=min(1.0, 1.0 / max(1.0, float(actions))),
                    developmental_stage=int(MemoryLevel.M7),
                    validation_state=3,
                    source_game_hash=int(game_hash),
                    effect_direction=1 if credit.valence_sum > 0 else -1,
                    graph_generation=self.generation,
                )
            )


def _install_traceability() -> None:
    from v8 import scientific_traceability as trace_module

    revised = []
    for record in trace_module.TRACEABILITY:
        if record.hypothesis_id == "H12":
            revised.append(
                replace(
                    record,
                    paper_claim=(
                        "Among strategies reaching the same or explicitly comparable M6 outcome, "
                        "lower realized actions/interaction cost is higher efficiency; achievement "
                        "reliability, primary valence, and efficiency remain separate evidence."
                    ),
                    candidate_evidence=(
                        "strategy_reuse",
                        "strategy_efficiency",
                        "primary_valence_efficiency",
                    ),
                    required_evidence=("strategy_efficiency",),
                )
            )
        else:
            revised.append(record)
    trace_module.TRACEABILITY = tuple(revised)


def _install_metadata() -> None:
    from v8 import runtime_v82 as runtime_v82_module

    runtime_v82_module.V82ContinuousMemoryRuntime.scientific_semantics_version = (
        "v8.4-outcome-conditioned-efficiency"
    )
    runtime_v82_module.V82ContinuousMemoryRuntime.research_paper_version = "0.5.4"


def install_trajectory_efficiency_v054() -> None:
    global _INSTALLED
    global _BASE_OBSERVED, _BASE_LEARNING_BATCH, _BASE_PUBLISH_LEARNING
    global _BASE_RESET_CAPTURE, _BASE_SCORE_ROWS, _BASE_RUNTIME_RECORD_RESULTS
    global _BASE_APPEND_EVIDENCE
    if _INSTALLED:
        return

    from v8 import actor as actor_module
    from v8 import behavior_recovery as behavior_module
    from v8 import peers as peers_module
    from v8 import peers_v82 as peers_v82_module
    from v8 import runtime as runtime_module

    _BASE_RESET_CAPTURE = _primary._reset_actor_capture
    _BASE_OBSERVED = actor_module._observed_outcome_uids
    _BASE_LEARNING_BATCH = actor_module._learning_batch
    _BASE_PUBLISH_LEARNING = actor_module._publish_learning
    _BASE_SCORE_ROWS = behavior_module._score_strategy_rows
    _BASE_RUNTIME_RECORD_RESULTS = runtime_module.ContinuousMemoryRuntime.record_actor_results
    _BASE_APPEND_EVIDENCE = peers_module.DevelopmentalPeerSupervisor._append_evidence

    _primary._reset_actor_capture = _reset_actor_capture_v054
    _primary._emit_terminal_credits = _emit_terminal_credits_v054
    actor_module._observed_outcome_uids = _observed_outcome_uids_v054
    actor_module._learning_batch = _learning_batch_v054
    actor_module._publish_learning = _publish_learning_v054
    behavior_module._score_strategy_rows = _score_strategy_rows_v054
    peers_module.DevelopmentalPeerSupervisor._append_evidence = _append_evidence_v054
    peers_v82_module.V82DevelopmentalPeerSupervisor._append_evidence = _append_evidence_v054
    runtime_module.ContinuousMemoryRuntime.record_actor_results = _record_actor_results_v054

    _install_traceability()
    _install_metadata()
    _INSTALLED = True
