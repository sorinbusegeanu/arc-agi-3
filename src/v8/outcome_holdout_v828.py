from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace

from v8.model import MemoryLevel, MemoryType, MemoryUid, RelationType, ValidationState
from v8.outcomes import OutcomeClass, OutcomeEquivalenceEstimator
from v8.peers import DevelopmentalPeerSupervisor


_LINEAGE_RELATIONS = {
    int(RelationType.PROVENANCE),
    int(RelationType.EXPLAINS),
    int(RelationType.LEADS_TO),
    int(RelationType.CONTEXT_REFINES),
    int(RelationType.DEPENDS_ON),
}


@dataclass(slots=True)
class OutcomeHoldoutValidation:
    class_uid: MemoryUid
    descriptor: tuple[int, int]
    training_members: tuple[MemoryUid, ...]
    holdout_members: tuple[MemoryUid, ...]
    training_games: tuple[int, ...]
    holdout_games: tuple[int, ...]
    target_game_hash: int
    training_persistent: bool
    holdout_consistent: bool
    training_class: OutcomeClass
    full_class: OutcomeClass
    training_occurrences: int
    holdout_occurrences: int


def _derive_class(
    descriptor: tuple[int, int],
    members: tuple[object, ...],
    *,
    uid: MemoryUid,
    version: int,
    estimator: OutcomeEquivalenceEstimator,
) -> OutcomeClass:
    support = sum(max(0, int(row.support_count)) for row in members)
    variants = [int(row.key_parts[2]) if len(row.key_parts) >= 3 else 0 for row in members]
    stability = min(1.0, support / max(1.0, 2.0 * len(members)))
    dominant_support = max((max(0, int(row.support_count)) for row in members), default=0)
    context_consistency = dominant_support / max(1, support)
    if len(variants) <= 1:
        diameter = 0.0
    else:
        span = max(variants) - min(variants)
        normalizer = max(1, max(abs(value) for value in variants))
        diameter = min(1.0, abs(span) / normalizer)
    interchangeability = max(0.0, 1.0 - diameter)
    persistent = bool(
        support >= estimator.min_support
        and stability >= estimator.stability_threshold
        and context_consistency >= estimator.context_consistency_threshold
        and diameter <= estimator.max_diameter
        and interchangeability >= estimator.interchangeability_threshold
    )
    return OutcomeClass(
        uid,
        tuple(sorted(row.uid for row in members)),
        descriptor,
        int(version),
        int(support),
        float(stability),
        float(context_consistency),
        float(diameter),
        float(interchangeability),
        persistent,
    )


def _m0_occurrences_by_world(read_view, roots: tuple[MemoryUid, ...], *, max_depth: int = 8):
    """Return exact M0 episode occurrences reachable from each fine M6 root.

    This intentionally avoids ``source_games(root)`` because that provenance is already
    unioned after canonical memories aggregate across worlds.  The validation unit is
    the lower-level episode carrying a direct GAME_PROVENANCE edge.
    """
    rows = {row.uid: row for row in read_view.node_records()}
    parents: dict[MemoryUid, set[MemoryUid]] = defaultdict(set)
    direct_games: dict[MemoryUid, set[int]] = defaultdict(set)
    for edge in read_view.edge_records():
        relation = int(edge.relation_type)
        if relation == int(RelationType.GAME_PROVENANCE) and int(edge.target_uid.hi) == 0:
            direct_games[edge.source_uid].add(int(edge.target_uid.lo))
        elif relation in _LINEAGE_RELATIONS:
            parents[edge.source_uid].add(edge.target_uid)

    result: dict[MemoryUid, dict[int, int]] = {}
    for root in roots:
        frontier = {root}
        visited = {root}
        episodes_by_game: dict[int, set[MemoryUid]] = defaultdict(set)
        for _depth in range(max(0, int(max_depth)) + 1):
            following: set[MemoryUid] = set()
            for uid in frontier:
                row = rows.get(uid)
                if row is not None and int(row.level) == int(MemoryLevel.M0):
                    for game in direct_games.get(uid, ()):
                        episodes_by_game[int(game)].add(uid)
                    continue
                for parent in parents.get(uid, ()):
                    if parent not in visited:
                        visited.add(parent)
                        following.add(parent)
            if not following:
                break
            frontier = following
        result[root] = {
            int(game): len(episodes)
            for game, episodes in episodes_by_game.items()
            if episodes
        }
    return result


def _pseudo_row(root, game: int, support: int):
    return SimpleNamespace(
        uid=MemoryUid.from_key(
            MemoryLevel.M6,
            MemoryType.OUTCOME,
            tuple(int(v) for v in root.key_parts[:3]) + (int(game),),
        ),
        support_count=max(1, int(support)),
        key_parts=tuple(int(v) for v in root.key_parts),
    )


def _select_occurrence_holdout_class(
    estimator: OutcomeEquivalenceEstimator,
    full_class: OutcomeClass,
    by_uid: dict[MemoryUid, object],
    occurrences: dict[MemoryUid, dict[int, int]],
) -> OutcomeHoldoutValidation | None:
    fine_uids = tuple(
        uid for uid in full_class.members if uid != full_class.uid and uid in by_uid
    )
    if len(fine_uids) < 2:
        return None

    candidate_games = sorted(
        {game for uid in fine_uids for game in occurrences.get(uid, {})}
    )
    for target_game in candidate_games:
        training_rows = []
        holdout_rows = []
        training_members: set[MemoryUid] = set()
        holdout_members: set[MemoryUid] = set()
        training_games: set[int] = set()
        for uid in fine_uids:
            root = by_uid[uid]
            for game, support in sorted(occurrences.get(uid, {}).items()):
                row = _pseudo_row(root, game, support)
                if int(game) == int(target_game):
                    holdout_rows.append(row)
                    holdout_members.add(uid)
                else:
                    training_rows.append(row)
                    training_members.add(uid)
                    training_games.add(int(game))
        if not holdout_rows or len(training_members) < 2 or not training_games:
            continue

        training_class = _derive_class(
            full_class.descriptor,
            tuple(training_rows),
            uid=full_class.uid,
            version=full_class.version,
            estimator=estimator,
        )
        if not training_class.persistent:
            continue
        full_shadow = _derive_class(
            full_class.descriptor,
            tuple(training_rows + holdout_rows),
            uid=full_class.uid,
            version=full_class.version,
            estimator=estimator,
        )
        return OutcomeHoldoutValidation(
            class_uid=full_class.uid,
            descriptor=full_class.descriptor,
            training_members=tuple(sorted(training_members)),
            holdout_members=tuple(sorted(holdout_members)),
            training_games=tuple(sorted(training_games)),
            holdout_games=(int(target_game),),
            target_game_hash=int(target_game),
            training_persistent=True,
            holdout_consistent=bool(full_shadow.persistent),
            training_class=training_class,
            full_class=full_shadow,
            training_occurrences=sum(int(row.support_count) for row in training_rows),
            holdout_occurrences=sum(int(row.support_count) for row in holdout_rows),
        )
    return None


def _consistency_score(outcome: OutcomeClass) -> float:
    return max(
        0.0,
        min(
            1.0,
            float(outcome.stability),
            float(outcome.context_consistency),
            float(outcome.predictive_interchangeability),
            max(0.0, 1.0 - float(outcome.within_class_diameter)),
        ),
    )


def _build_validations(supervisor: DevelopmentalPeerSupervisor):
    nodes = tuple(supervisor.read_view.node_records())
    by_uid = {row.uid: row for row in nodes}
    classes = tuple(supervisor.outcomes.rebuild(nodes))
    roots = tuple(
        uid
        for outcome in classes
        for uid in outcome.members
        if uid != outcome.uid and uid in by_uid
    )
    occurrences = _m0_occurrences_by_world(supervisor.read_view, roots)
    validations = []
    for outcome in classes:
        validation = _select_occurrence_holdout_class(
            supervisor.outcomes, outcome, by_uid, occurrences
        )
        if validation is not None:
            validations.append(validation)
    return tuple(validations)


def _emit_holdout_evidence(
    supervisor: DevelopmentalPeerSupervisor,
    validations: tuple[OutcomeHoldoutValidation, ...],
) -> None:
    if not validations:
        return
    rows = {row.uid: row for row in supervisor.read_view.node_records()}
    try:
        from v8 import information_flow_diagnostics as flow
    except Exception:
        flow = None

    for validation in validations:
        member_rows = [
            rows[uid]
            for uid in set(validation.training_members) | set(validation.holdout_members)
            if uid in rows
        ]
        watermark = max(
            (int(row.updated_watermark) for row in member_rows),
            default=int(supervisor.current_watermark()),
        )
        kind = (
            "outcome_consistency_holdout"
            if validation.holdout_consistent
            else "outcome_consistency_fail"
        )
        freshness = (
            f"v828-occurrence:{kind}:{validation.descriptor[0]}:"
            f"{validation.descriptor[1]}:{validation.target_game_hash}"
        )
        if not supervisor._fresh(freshness, validation.class_uid, watermark):
            continue
        proxy = SimpleNamespace(
            uid=validation.class_uid,
            level=MemoryLevel.M6,
            validation_state=int(ValidationState.STRUCTURAL),
        )
        score = _consistency_score(validation.full_class)
        value = score if validation.holdout_consistent else max(1e-9, 1.0 - score)
        supervisor._append_evidence(
            kind,
            proxy,
            value,
            validation_state=int(ValidationState.STRUCTURAL),
            unique=True,
            target_game_hash=validation.target_game_hash,
            provenance_games=validation.training_games,
            causal_intervention="m6_m0_world_occurrence_holdout",
            effect_direction=1 if validation.holdout_consistent else -1,
        )
        if flow is not None:
            flow.emit(
                "outcomes",
                kind,
                input_count=validation.holdout_occurrences,
                output_count=1,
                examples=[
                    {
                        "m6_uid": validation.class_uid.hex(),
                        "heldout_target_world": validation.target_game_hash,
                        "formation_provenance_games": list(validation.training_games),
                        "holdout_games": list(validation.holdout_games),
                        "training_m0_occurrences": validation.training_occurrences,
                        "holdout_m0_occurrences": validation.holdout_occurrences,
                        "training_member_count": len(validation.training_members),
                        "holdout_member_count": len(validation.holdout_members),
                        "target_world_excluded_from_validation_formation_support": True,
                        "training_persistent": validation.training_persistent,
                        "holdout_consistent": validation.holdout_consistent,
                        "training_support": validation.training_class.support,
                        "full_support": validation.full_class.support,
                        "training_diameter": validation.training_class.within_class_diameter,
                        "full_diameter": validation.full_class.within_class_diameter,
                        "training_interchangeability": validation.training_class.predictive_interchangeability,
                        "full_interchangeability": validation.full_class.predictive_interchangeability,
                        "criterion": "existing_m6_persistence_diameter_interchangeability",
                        "validation_unit": "direct_game_provenanced_m0_episode",
                    }
                ],
            )


def install_outcome_holdout_v828() -> None:
    if getattr(OutcomeEquivalenceEstimator, "_v828_holdout_installed", False):
        return

    original_base_run_once = DevelopmentalPeerSupervisor.run_once

    def base_run_once(self: DevelopmentalPeerSupervisor):
        validations = _build_validations(self)
        result = original_base_run_once(self)
        _emit_holdout_evidence(self, validations)
        return result

    DevelopmentalPeerSupervisor.run_once = base_run_once
    OutcomeEquivalenceEstimator._v828_holdout_installed = True
