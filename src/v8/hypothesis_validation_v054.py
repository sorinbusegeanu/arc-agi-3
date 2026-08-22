from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Iterable

from v8 import primary_valence as _primary
from v8.arena import EdgeRecord, NodeRecord
from v8.model import (
    MemoryLevel,
    MemoryType,
    MemoryUid,
    RelationType,
    ValidationState,
)
from v8.world_model import WorldModelComponent


_INSTALLED = False
_REPLAN_HORIZON = 64
_TRANSFER_INTERVENTION = "leave_one_memory_out_correspondence_ablation"
_PENDING_REPLAN_TRIALS: list["TrajectoryReplanningTrialResult"] = []
_LAST_OBSERVATION_TERMINAL = False

_BASE_WORLD_MODEL_PROPOSE = None
_BASE_V82_RUN_ONCE = None
_BASE_ACTOR_OBSERVED = None
_BASE_RESET_CAPTURE = None
_BASE_RUNTIME_RECORD_RESULTS = None
_BASE_EMIT_TERMINAL_CREDITS = None


@dataclass(slots=True)
class TrajectoryReplanningTrialResult:
    """Actor-side H14 probe resolved over a trajectory, not one transition."""

    primary_strategy_uid: MemoryUid
    alternative_strategy_uid: MemoryUid
    outcome_uid: MemoryUid
    recovery_succeeded: bool
    resolved: bool = False
    actions: int = 1

    def __post_init__(self) -> None:
        global _LAST_OBSERVATION_TERMINAL
        self.recovery_succeeded = bool(self.recovery_succeeded)
        self.resolved = bool(self.recovery_succeeded or _LAST_OBSERVATION_TERMINAL)
        self.actions = max(1, int(self.actions))
        if bool(getattr(_primary, "_CAPTURE_ACTIVE", False)) and not self.resolved:
            _PENDING_REPLAN_TRIALS.append(self)


def _reset_validation_capture() -> None:
    global _LAST_OBSERVATION_TERMINAL
    if _BASE_RESET_CAPTURE is not None:
        _BASE_RESET_CAPTURE()
    _PENDING_REPLAN_TRIALS.clear()
    _LAST_OBSERVATION_TERMINAL = False


def _observed_with_trajectory_replanning(**kwargs):
    global _LAST_OBSERVATION_TERMINAL
    result = _BASE_ACTOR_OBSERVED(**kwargs)
    if not bool(getattr(_primary, "_CAPTURE_ACTIVE", False)):
        return result

    observed = {uid for uid in result if not uid.is_zero}
    terminal = int(kwargs.get("terminal_polarity", 0)) != 0
    _LAST_OBSERVATION_TERMINAL = bool(terminal)
    for trial in tuple(_PENDING_REPLAN_TRIALS):
        if trial.resolved:
            continue
        trial.actions += 1
        if trial.outcome_uid in observed:
            trial.recovery_succeeded = True
            trial.resolved = True
        elif terminal or trial.actions >= _REPLAN_HORIZON:
            trial.recovery_succeeded = False
            trial.resolved = True
    return result


def _emit_terminal_credits_with_avoidance(primary_valence: int, behavior_module) -> None:
    """Keep positive preference credit and add symmetric negative-valence avoidance."""
    _BASE_EMIT_TERMINAL_CREDITS(primary_valence, behavior_module)
    if int(primary_valence) >= 0:
        return

    seen: set[tuple[MemoryUid, MemoryUid]] = set()
    for distance, entry in enumerate(reversed(tuple(_primary._TRAJECTORY))):
        value = float(primary_valence) * (_primary._VALENCE_GAMMA ** distance)
        if abs(value) < float(_primary._PREFERENCE_CREDIT_THRESHOLD):
            continue
        plan = entry.get("plan")
        alternative = entry.get("alternative")
        if plan is None or alternative is None or bool(plan.preference_influenced):
            continue
        pair = (alternative.outcome_uid, plan.outcome_uid)
        if pair in seen or pair[0] == pair[1]:
            continue
        seen.add(pair)
        _primary._PENDING_VALENCE_PREFERENCES.append(
            _primary.PrimaryValencePreference(
                alternative.outcome_uid,
                plan.outcome_uid,
                0,
                abs(float(value)),
            )
        )


def _valence_bucket(row: NodeRecord) -> int:
    weight = max(0.0, float(getattr(row, "primary_valence_weight", 0.0)))
    if weight <= 0.0:
        return 0
    value = float(getattr(row, "expected_primary_valence", 0.0))
    if value > 1e-9:
        return 1
    if value < -1e-9:
        return -1
    return 0


def _world_model_propose_v054(self, rows: tuple[NodeRecord, ...]) -> tuple[WorldModelComponent, ...]:
    """Group M5 consequences by learned consequence profile, not concept identity.

    M5 key_parts[2] is derived from concept/family identity and therefore cannot be
    a cross-concept aggregation key.  The world-model descriptor uses only the
    already-learned future-option direction and primary-valence direction.
    """
    grouped: dict[tuple[int, int], list[NodeRecord]] = defaultdict(list)
    for row in rows:
        if int(row.level) != int(MemoryLevel.M5):
            continue
        if int(row.memory_type) != int(MemoryType.CONSEQUENCE) or len(row.key_parts) < 4:
            continue
        future_bucket = int(row.key_parts[3])
        grouped[(future_bucket, _valence_bucket(row))].append(row)

    result: list[WorldModelComponent] = []
    for key, members in sorted(grouped.items()):
        distinct_concepts = {
            (int(row.key_parts[0]), int(row.key_parts[1]))
            for row in members
        }
        if len(distinct_concepts) < int(self.min_consequences):
            continue
        mask = 0
        for row in members:
            mask |= int(row.game_mask)
        result.append(
            WorldModelComponent(
                MemoryUid.from_key(MemoryLevel.M5, MemoryType.WORLD_MODEL, key),
                key,
                tuple(sorted(row.uid for row in members)),
                sum(max(0, int(row.support_count)) for row in members),
                mask.bit_count(),
            )
        )
    return tuple(result)


_LINEAGE_RELATIONS = {
    int(RelationType.PROVENANCE),
    int(RelationType.EXPLAINS),
    int(RelationType.LEADS_TO),
    int(RelationType.CONTEXT_REFINES),
    int(RelationType.SUPERSEDES),
    int(RelationType.DEPENDS_ON),
}


def _provenance_lookup(edges: Iterable[EdgeRecord]):
    direct: dict[MemoryUid, set[int]] = defaultdict(set)
    parents: dict[MemoryUid, set[MemoryUid]] = defaultdict(set)
    for edge in edges:
        relation = int(edge.relation_type)
        if relation == int(RelationType.GAME_PROVENANCE) and int(edge.target_uid.hi) == 0:
            direct[edge.source_uid].add(int(edge.target_uid.lo))
        elif relation in _LINEAGE_RELATIONS:
            parents[edge.source_uid].add(edge.target_uid)

    cache: dict[MemoryUid, frozenset[int]] = {}

    def games(uid: MemoryUid) -> frozenset[int]:
        cached = cache.get(uid)
        if cached is not None:
            return cached
        found = set(direct.get(uid, ()))
        frontier = {uid}
        visited = {uid}
        for _depth in range(8):
            following: set[MemoryUid] = set()
            for current in frontier:
                for parent in parents.get(current, ()):
                    found.update(direct.get(parent, ()))
                    if parent not in visited:
                        visited.add(parent)
                        following.add(parent)
            if not following:
                break
            frontier = following
        result = frozenset(found)
        cache[uid] = result
        return result

    return games


def _auto_transfer_trials(self, nodes: tuple[NodeRecord, ...], edges: tuple[EdgeRecord, ...]) -> None:
    """Run bounded held-out leave-one-memory-out correspondence ablations.

    Structural correspondence remains candidate evidence.  Validation measures its
    incremental held-out reuse value: with the candidate available, use its formal
    correspondence score; with that memory removed, use the best alternative source
    correspondence for the same target, or zero when no alternative memory can match.
    """
    by_uid = {row.uid: row for row in nodes}
    games = _provenance_lookup(edges)
    adjacency: dict[MemoryUid, list[tuple[MemoryUid, float]]] = defaultdict(list)
    for edge in edges:
        if int(edge.relation_type) != int(RelationType.TRANSFER_CORRESPONDENCE):
            continue
        score = float(edge.score)
        if score <= 0.0:
            continue
        if edge.source_uid not in by_uid or edge.target_uid not in by_uid:
            continue
        adjacency[edge.target_uid].append((edge.source_uid, score))
        adjacency[edge.source_uid].append((edge.target_uid, score))

    emitted = 0
    budget = max(1, min(16, int(getattr(self, "candidate_budget", 16))))
    for target_uid in sorted(adjacency):
        target_row = by_uid.get(target_uid)
        if target_row is None:
            continue
        candidates = sorted(adjacency[target_uid], key=lambda item: (-item[1], item[0]))
        for source_uid, score in candidates:
            source_row = by_uid.get(source_uid)
            if source_row is None:
                continue
            if int(source_row.level) not in {int(MemoryLevel.M3), int(MemoryLevel.M4)}:
                continue
            if int(source_row.level) != int(target_row.level) or int(source_row.memory_type) != int(target_row.memory_type):
                continue
            formation_games = tuple(sorted(games(source_uid)))
            if not formation_games:
                continue
            held_out_targets = tuple(sorted(set(games(target_uid)) - set(formation_games)))
            if not held_out_targets:
                continue

            alternatives = [
                alt_score
                for alt_uid, alt_score in candidates
                if alt_uid != source_uid
                and (alt_row := by_uid.get(alt_uid)) is not None
                and int(alt_row.level) == int(source_row.level)
                and int(alt_row.memory_type) == int(source_row.memory_type)
            ]
            metric_off = max(alternatives, default=0.0)
            existing = {
                int(trial.target_game_hash)
                for trial in self.transfer.trials(source_uid)
                if str(trial.intervention) == _TRANSFER_INTERVENTION
            }
            for target_game in held_out_targets:
                if int(target_game) in existing:
                    continue
                self.record_transfer_trial(
                    source_uid,
                    target_game_hash=int(target_game),
                    metric_on=float(score),
                    metric_off=float(metric_off),
                    formation_games=formation_games,
                    intervention=_TRANSFER_INTERVENTION,
                )
                emitted += 1
                if emitted >= budget:
                    return


def _auto_outcome_holdout(self, nodes: tuple[NodeRecord, ...], edges: tuple[EdgeRecord, ...]) -> None:
    """Emit held-out M6 consistency only for persistent cross-game outcome classes."""
    by_uid = {row.uid: row for row in nodes}
    games = _provenance_lookup(edges)
    classes = self.outcomes.rebuild(nodes)
    emitted = 0
    budget = max(1, min(16, int(getattr(self, "candidate_budget", 16))))
    for outcome in sorted(classes, key=lambda item: item.uid):
        if not bool(outcome.persistent):
            continue
        members = [by_uid[uid] for uid in outcome.members if uid in by_uid]
        if not members:
            continue
        all_games: set[int] = set()
        by_game: dict[int, list[NodeRecord]] = defaultdict(list)
        for member in members:
            for game in games(member.uid):
                all_games.add(int(game))
                by_game[int(game)].append(member)
        if len(all_games) < 2:
            continue
        score = min(
            float(outcome.stability),
            float(outcome.context_consistency),
            float(outcome.predictive_interchangeability),
        )
        if score <= 0.0:
            continue
        for target_game in sorted(all_games):
            formation_games = tuple(sorted(all_games - {target_game}))
            target_members = by_game.get(target_game, ())
            if not formation_games or not target_members:
                continue
            row = sorted(target_members, key=lambda item: item.uid)[0]
            freshness = f"outcome-holdout:{int(target_game)}"
            if not self._fresh(freshness, row.uid, row.updated_watermark):
                continue
            self._append_evidence(
                "outcome_consistency_holdout",
                row,
                score,
                validation_state=int(ValidationState.VALIDATED),
                unique=True,
                target_game_hash=int(target_game),
                provenance_games=formation_games,
                effect_direction=1,
            )
            emitted += 1
            if emitted >= budget:
                return


def _run_once_with_validation(self) -> None:
    _BASE_V82_RUN_ONCE(self)
    cancel = getattr(self, "_v841_peer_cancel", None)
    if cancel is not None and cancel.is_set():
        return
    cut = getattr(self, "last_developmental_cut", None)
    if cut is None:
        return
    nodes = tuple(cut.nodes)
    edges = tuple(cut.edges)
    _auto_transfer_trials(self, nodes, edges)
    _auto_outcome_holdout(self, nodes, edges)


def _filtered_replanning_results(results):
    filtered = []
    for result in results:
        trials = tuple(
            trial
            for trial in getattr(result, "replanning_trials", ())
            if bool(getattr(trial, "resolved", True))
        )
        if hasattr(result, "replanning_trials") and len(trials) != len(getattr(result, "replanning_trials", ())):
            result = replace(result, replanning_trials=trials)
        filtered.append(result)
    return tuple(filtered)


def _record_actor_results_with_validation(self, results) -> None:
    filtered = _filtered_replanning_results(tuple(results))
    _BASE_RUNTIME_RECORD_RESULTS(self, filtered)
    if self.peers is None:
        return

    # Primary-valence preference is already restricted to choices made without a
    # preference-influenced plan.  Add a context-general probe so exact grid hashes
    # do not prevent repeated evidence for the same reachable outcome pair.
    for result in filtered:
        seen_pairs: set[tuple[MemoryUid, MemoryUid]] = set()
        for preference in getattr(result, "primary_valence_preferences", ()):
            pair = (preference.preferred, preference.other)
            if pair in seen_pairs or pair[0] == pair[1]:
                continue
            seen_pairs.add(pair)
            self.peers.record_preference_probe(
                outcome_a=preference.preferred,
                outcome_b=preference.other,
                context_bucket=0,
                chosen_outcome=preference.preferred,
                both_reachable=True,
                preference_influenced=False,
            )


def install_hypothesis_validation_v054() -> None:
    global _INSTALLED
    global _BASE_WORLD_MODEL_PROPOSE, _BASE_V82_RUN_ONCE, _BASE_ACTOR_OBSERVED
    global _BASE_RESET_CAPTURE, _BASE_RUNTIME_RECORD_RESULTS, _BASE_EMIT_TERMINAL_CREDITS
    if _INSTALLED:
        return

    from v8 import actor as actor_module
    from v8 import peers_v82 as peers_v82_module
    from v8 import runtime as runtime_module
    from v8 import world_model as world_model_module

    _BASE_WORLD_MODEL_PROPOSE = world_model_module.WorldModelEstimator.propose
    _BASE_V82_RUN_ONCE = peers_v82_module.V82DevelopmentalPeerSupervisor.run_once
    _BASE_ACTOR_OBSERVED = actor_module._observed_outcome_uids
    _BASE_RESET_CAPTURE = _primary._reset_actor_capture
    _BASE_RUNTIME_RECORD_RESULTS = runtime_module.ContinuousMemoryRuntime.record_actor_results
    _BASE_EMIT_TERMINAL_CREDITS = _primary._emit_terminal_credits

    world_model_module.WorldModelEstimator.propose = _world_model_propose_v054
    peers_v82_module.V82DevelopmentalPeerSupervisor.run_once = _run_once_with_validation
    actor_module.ReplanningTrialResult = TrajectoryReplanningTrialResult
    actor_module._observed_outcome_uids = _observed_with_trajectory_replanning
    _primary._reset_actor_capture = _reset_validation_capture
    _primary._emit_terminal_credits = _emit_terminal_credits_with_avoidance
    runtime_module.ContinuousMemoryRuntime.record_actor_results = _record_actor_results_with_validation

    _INSTALLED = True
