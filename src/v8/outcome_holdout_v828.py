from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from types import SimpleNamespace

from v8.model import MemoryLevel, MemoryType, MemoryUid, RelationType, ValidationState
from v8.outcomes import OutcomeClass, OutcomeEquivalenceEstimator
from v8.peers import DevelopmentalPeerSupervisor


_HOLDOUT_RUN_ONCE_V828 = None


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
    holdout_class: OutcomeClass
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
    # Holdout rows are split by provenance world, so the same fine outcome
    # variant can appear in several pseudo rows.  Context consistency is a
    # property of the variant distribution, not of the largest world slice.
    support_by_variant: dict[int, int] = defaultdict(int)
    for row in members:
        variant = int(row.key_parts[2]) if len(row.key_parts) >= 3 else 0
        support_by_variant[variant] += max(0, int(row.support_count))
    dominant_support = max(support_by_variant.values(), default=0)
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


def _persistence_rejection(
    outcome: OutcomeClass, estimator: OutcomeEquivalenceEstimator, *, prefix: str
) -> str:
    if outcome.support < estimator.min_support:
        return f"{prefix}_support_below_minimum"
    if outcome.stability < estimator.stability_threshold:
        return f"{prefix}_stability_below_threshold"
    if outcome.context_consistency < estimator.context_consistency_threshold:
        return f"{prefix}_context_consistency_below_threshold"
    if outcome.within_class_diameter > estimator.max_diameter:
        return f"{prefix}_diameter_above_threshold"
    if outcome.predictive_interchangeability < estimator.interchangeability_threshold:
        return f"{prefix}_interchangeability_below_threshold"
    return f"{prefix}_not_persistent"


def _lineage_occurrences_by_world(
    read_view,
    roots: tuple[MemoryUid, ...],
    *,
    max_depth: int = 8,
):
    """Count direct game-provenance-bearing lineage occurrences for each fine M6 root."""
    parents: dict[MemoryUid, set[MemoryUid]] = defaultdict(set)
    try:
        node_levels = {
            row.uid: int(row.level) for row in read_view.node_records()
        }
    except (AttributeError, TypeError):
        node_levels = {}
    direct_games: dict[MemoryUid, dict[int, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for edge in read_view.edge_records():
        relation = int(edge.relation_type)
        if relation == int(RelationType.GAME_PROVENANCE) and int(edge.target_uid.hi) == 0:
            # A single M1 contingency is the canonical occurrence unit.  Higher
            # levels repeat the same lineage fact and must not multiply it.
            if (
                edge.source_uid in node_levels
                and node_levels[edge.source_uid] != int(MemoryLevel.M1)
            ):
                continue
            direct_games[edge.source_uid][int(edge.target_uid.lo)] += max(
                1, int(getattr(edge, "support_count", 1))
            )
        elif relation in _LINEAGE_RELATIONS:
            parents[edge.source_uid].add(edge.target_uid)

    result: dict[MemoryUid, dict[int, int]] = {}
    for root in roots:
        frontier = {root}
        visited = {root}
        occurrences_by_game: dict[int, int] = defaultdict(int)
        for _depth in range(max(0, int(max_depth)) + 1):
            following: set[MemoryUid] = set()
            for uid in frontier:
                for game, support in direct_games.get(uid, {}).items():
                    occurrences_by_game[int(game)] += int(support)
                for parent in parents.get(uid, ()):
                    if parent not in visited:
                        visited.add(parent)
                        following.add(parent)
            if not following:
                break
            frontier = following
        result[root] = {
            int(game): int(count)
            for game, count in occurrences_by_game.items()
            if count > 0
        }
    return result


_m0_occurrences_by_world = _lineage_occurrences_by_world


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
    rejection_counts: Counter | None = None,
) -> OutcomeHoldoutValidation | None:
    rejected = rejection_counts if rejection_counts is not None else Counter()
    fine_uids = tuple(
        uid for uid in full_class.members if uid != full_class.uid and uid in by_uid
    )
    if len(fine_uids) < 2:
        rejected["fewer_than_two_fine_members"] += 1
        return None

    candidate_games = sorted(
        {game for uid in fine_uids for game in occurrences.get(uid, {})}
    )
    if not candidate_games:
        rejected["no_lineage_world_occurrences"] += 1
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
        if not holdout_rows:
            rejected["target_has_no_holdout_occurrences"] += 1
            continue
        if len(training_members) < 2:
            rejected["fewer_than_two_training_members"] += 1
            continue
        if not training_games:
            rejected["no_training_worlds"] += 1
            continue

        training_class = _derive_class(
            full_class.descriptor,
            tuple(training_rows),
            uid=full_class.uid,
            version=full_class.version,
            estimator=estimator,
        )
        if not training_class.persistent:
            rejected[_persistence_rejection(
                training_class, estimator, prefix="training"
            )] += 1
            continue
        holdout_class = _derive_class(
            full_class.descriptor,
            tuple(holdout_rows),
            uid=full_class.uid,
            version=full_class.version,
            estimator=estimator,
        )
        training_variants = {
            int(row.key_parts[2]) for row in training_rows if len(row.key_parts) >= 3
        }
        holdout_variants = {
            int(row.key_parts[2]) for row in holdout_rows if len(row.key_parts) >= 3
        }
        holdout_consistent = bool(
            holdout_variants
            and holdout_variants.issubset(training_variants)
            and holdout_class.context_consistency >= estimator.context_consistency_threshold
            and holdout_class.within_class_diameter <= estimator.max_diameter
            and holdout_class.predictive_interchangeability
            >= estimator.interchangeability_threshold
        )
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
            holdout_consistent=holdout_consistent,
            training_class=training_class,
            holdout_class=holdout_class,
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


def _shadow_estimator(live: OutcomeEquivalenceEstimator) -> OutcomeEquivalenceEstimator:
    """Create an H13-only estimator with identical production criteria and no shared state."""
    return OutcomeEquivalenceEstimator(
        min_support=int(live.min_support),
        stability_threshold=float(live.stability_threshold),
        context_consistency_threshold=float(live.context_consistency_threshold),
        max_diameter=float(live.max_diameter),
        interchangeability_threshold=float(live.interchangeability_threshold),
    )


def _build_validations(supervisor: DevelopmentalPeerSupervisor):
    nodes = tuple(supervisor.read_view.node_records())
    by_uid = {row.uid: row for row in nodes}
    estimator = _shadow_estimator(supervisor.outcomes)
    classes = tuple(estimator.rebuild(nodes))
    roots = tuple(
        uid
        for outcome in classes
        for uid in outcome.members
        if uid != outcome.uid and uid in by_uid
    )
    occurrences = _lineage_occurrences_by_world(supervisor.read_view, roots)
    validations = []
    rejection_counts: Counter = Counter()
    for outcome in classes:
        validation = _select_occurrence_holdout_class(
            estimator, outcome, by_uid, occurrences, rejection_counts
        )
        if validation is not None:
            validations.append(validation)
    try:
        from v8 import information_flow_diagnostics as flow

        flow.emit(
            "outcomes",
            "outcome_holdout_selection",
            input_count=len(classes),
            output_count=len(validations),
            rejection_counts=rejection_counts,
            examples=[],
        )
    except Exception:
        pass
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
        score = _consistency_score(validation.holdout_class)
        value = score if validation.holdout_consistent else max(1e-9, 1.0 - score)
        supervisor._append_evidence(
            kind,
            proxy,
            value,
            validation_state=int(ValidationState.STRUCTURAL),
            unique=True,
            target_game_hash=validation.target_game_hash,
            provenance_games=validation.training_games,
            causal_intervention="m6_lineage_world_occurrence_holdout",
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
                        "training_lineage_occurrences": validation.training_occurrences,
                        "holdout_lineage_occurrences": validation.holdout_occurrences,
                        "training_member_count": len(validation.training_members),
                        "holdout_member_count": len(validation.holdout_members),
                        "target_world_excluded_from_validation_formation_support": True,
                        "training_persistent": validation.training_persistent,
                        "holdout_consistent": validation.holdout_consistent,
                        "training_support": validation.training_class.support,
                        "holdout_support": validation.holdout_class.support,
                        "full_support": validation.full_class.support,
                        "training_diameter": validation.training_class.within_class_diameter,
                        "full_diameter": validation.full_class.within_class_diameter,
                        "training_interchangeability": validation.training_class.predictive_interchangeability,
                        "holdout_interchangeability": validation.holdout_class.predictive_interchangeability,
                        "full_interchangeability": validation.full_class.predictive_interchangeability,
                        "criterion": "existing_m6_persistence_diameter_interchangeability",
                        "validation_unit": "m1_game_provenance_edge_support",
                    }
                ],
            )


def install_outcome_holdout_v828() -> None:
    global _HOLDOUT_RUN_ONCE_V828
    if getattr(OutcomeEquivalenceEstimator, "_v828_holdout_installed", False):
        return

    original_base_run_once = DevelopmentalPeerSupervisor.run_once

    def base_run_once(self: DevelopmentalPeerSupervisor):
        before_cycles = int(getattr(self, "_cycles", 0))
        result = original_base_run_once(self)
        if int(getattr(self, "_cycles", 0)) == before_cycles:
            return result
        validations = _build_validations(self)
        _emit_holdout_evidence(self, validations)
        return result

    DevelopmentalPeerSupervisor.run_once = base_run_once
    _HOLDOUT_RUN_ONCE_V828 = base_run_once
    OutcomeEquivalenceEstimator._v828_holdout_installed = True
