from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable

from v8.model import MemoryLevel, MemoryType, MemoryUid, RelationType, ValidationState
from v8.outcomes import OutcomeClass, OutcomeEquivalenceEstimator
from v8.peers import DevelopmentalPeerSupervisor
from v8.peers_v82 import V82DevelopmentalPeerSupervisor


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
    formed: bool = False

    @property
    def holdout_uid(self) -> MemoryUid:
        return self.holdout_members[0]


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


def _select_holdout_class(
    estimator: OutcomeEquivalenceEstimator,
    full_class: OutcomeClass,
    by_uid: dict[MemoryUid, object],
    provenance: Callable[[MemoryUid], frozenset[int] | set[int] | tuple[int, ...]],
) -> tuple[OutcomeClass, OutcomeHoldoutValidation | None]:
    """Withhold one complete world before the coarse M6 merge.

    Fine M6 nodes may already carry provenance from multiple worlds.  Holding out one
    fine node can therefore leak the target world through another member.  Select a
    target world first, exclude every fine member containing that world from training,
    and form the coarse M6 class only from the remaining members.
    """
    candidate_uids = tuple(
        uid for uid in full_class.members if uid != full_class.uid and uid in by_uid
    )
    if len(candidate_uids) < 3:
        return full_class, None

    games_by_uid = {
        uid: frozenset(int(v) for v in provenance(uid)) for uid in candidate_uids
    }
    candidate_games = sorted({game for games in games_by_uid.values() for game in games})
    partitions: list[tuple[int, int, tuple[MemoryUid, ...], tuple[MemoryUid, ...]]] = []
    for target_game in candidate_games:
        holdout_uids = tuple(uid for uid in candidate_uids if target_game in games_by_uid[uid])
        training_uids = tuple(uid for uid in candidate_uids if target_game not in games_by_uid[uid])
        if not holdout_uids or len(training_uids) < 2:
            continue
        partitions.append((len(holdout_uids), int(target_game), training_uids, holdout_uids))

    for _size, target_game, training_uids, holdout_uids in sorted(partitions):
        training_rows = tuple(by_uid[uid] for uid in training_uids)
        training_class = _derive_class(
            full_class.descriptor,
            training_rows,
            uid=full_class.uid,
            version=full_class.version,
            estimator=estimator,
        )
        if not training_class.persistent:
            continue
        training_games: set[int] = set()
        for uid in training_uids:
            training_games.update(games_by_uid[uid])
        if target_game in training_games:
            continue
        holdout_games: set[int] = set()
        for uid in holdout_uids:
            holdout_games.update(games_by_uid[uid])

        return training_class, OutcomeHoldoutValidation(
            class_uid=full_class.uid,
            descriptor=full_class.descriptor,
            training_members=tuple(sorted(training_uids)),
            holdout_members=tuple(sorted(holdout_uids)),
            training_games=tuple(sorted(training_games)),
            holdout_games=tuple(sorted(holdout_games)),
            target_game_hash=int(target_game),
            training_persistent=True,
            holdout_consistent=bool(full_class.persistent),
            training_class=training_class,
            full_class=full_class,
        )

    return full_class, None


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


def _heldout_m5_parents(read_view, holdout_uids: tuple[MemoryUid, ...]) -> tuple[MemoryUid, ...]:
    by_uid = {row.uid: row for row in read_view.node_records()}
    wanted = set(holdout_uids)
    result: set[MemoryUid] = set()
    for edge in read_view.edge_records():
        if edge.source_uid not in wanted or int(edge.relation_type) != int(RelationType.EXPLAINS):
            continue
        target = by_uid.get(edge.target_uid)
        if target is not None and int(target.level) == int(MemoryLevel.M5):
            result.add(target.uid)
    return tuple(sorted(result))


def _emit_holdout_evidence(supervisor: V82DevelopmentalPeerSupervisor) -> None:
    validations = tuple(getattr(supervisor.outcomes, "_v828_holdout_validations", ()))
    if not validations:
        return
    rows = {row.uid: row for row in supervisor.read_view.node_records()}
    try:
        from v8 import information_flow_diagnostics as flow
    except Exception:
        flow = None

    for validation in validations:
        if not validation.formed or not validation.training_persistent:
            continue
        holdout_rows = [rows[uid] for uid in validation.holdout_members if uid in rows]
        if not holdout_rows:
            continue
        watermark = max(
            [int(row.updated_watermark) for row in holdout_rows]
            + [int(rows[uid].updated_watermark) for uid in validation.training_members if uid in rows]
        )
        kind = "outcome_consistency_holdout" if validation.holdout_consistent else "outcome_consistency_fail"
        freshness = (
            f"v828:{kind}:{validation.descriptor[0]}:{validation.descriptor[1]}:"
            f"{validation.target_game_hash}"
        )
        proxy = SimpleNamespace(
            uid=validation.class_uid,
            level=MemoryLevel.M6,
            validation_state=int(ValidationState.STRUCTURAL),
        )
        if not supervisor._fresh(freshness, validation.class_uid, watermark):
            continue
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
            causal_intervention="m6_world_withheld_before_coarse_merge",
            effect_direction=1 if validation.holdout_consistent else -1,
        )
        if flow is not None:
            heldout_m5 = _heldout_m5_parents(supervisor.read_view, validation.holdout_members)
            flow.emit(
                "outcomes",
                kind,
                input_count=len(validation.holdout_members),
                output_count=1,
                examples=[
                    {
                        "m6_uid": validation.class_uid.hex(),
                        "heldout_m6_uids": [uid.hex() for uid in validation.holdout_members],
                        "heldout_m5_uids": [uid.hex() for uid in heldout_m5],
                        "heldout_target_world": validation.target_game_hash,
                        "formation_provenance_games": list(validation.training_games),
                        "holdout_games": list(validation.holdout_games),
                        "target_world_excluded_from_all_training_members": True,
                        "holdout_excluded_from_formation_support": True,
                        "training_persistent": validation.training_persistent,
                        "holdout_consistent": validation.holdout_consistent,
                        "training_member_count": len(validation.training_members),
                        "holdout_member_count": len(validation.holdout_members),
                        "training_support": validation.training_class.support,
                        "full_support": validation.full_class.support,
                        "training_diameter": validation.training_class.within_class_diameter,
                        "full_diameter": validation.full_class.within_class_diameter,
                        "training_interchangeability": validation.training_class.predictive_interchangeability,
                        "full_interchangeability": validation.full_class.predictive_interchangeability,
                        "criterion": "existing_m6_persistence_diameter_interchangeability",
                    }
                ],
            )


def install_outcome_holdout_v828() -> None:
    if getattr(OutcomeEquivalenceEstimator, "_v828_holdout_installed", False):
        return

    original_rebuild = OutcomeEquivalenceEstimator.rebuild
    original_merge_revision = OutcomeEquivalenceEstimator.merge_revision
    original_base_run_once = DevelopmentalPeerSupervisor.run_once

    def rebuild(self: OutcomeEquivalenceEstimator, rows):
        full_classes = tuple(original_rebuild(self, rows))
        provenance = getattr(self, "_v828_provenance", None)
        if provenance is None:
            self._v828_holdout_validations = ()
            return full_classes
        by_uid = {row.uid: row for row in rows}
        transformed: list[OutcomeClass] = []
        validations: list[OutcomeHoldoutValidation] = []
        for full_class in full_classes:
            training_class, validation = _select_holdout_class(self, full_class, by_uid, provenance)
            transformed.append(training_class)
            if validation is not None:
                validations.append(validation)
        self._classes = {item.descriptor: item for item in transformed}
        self._v828_holdout_validations = tuple(validations)
        return tuple(transformed)

    def merge_revision(self: OutcomeEquivalenceEstimator, outcome: OutcomeClass):
        revision = original_merge_revision(self, outcome)
        if revision is not None:
            for validation in getattr(self, "_v828_holdout_validations", ()):
                if validation.class_uid == outcome.uid and validation.descriptor == outcome.descriptor:
                    validation.formed = True
        return revision

    def base_run_once(self: DevelopmentalPeerSupervisor):
        self.outcomes._v828_provenance = lambda uid: self.read_view.source_games(uid)
        result = original_base_run_once(self)
        _emit_holdout_evidence(self)
        return result

    OutcomeEquivalenceEstimator.rebuild = rebuild
    OutcomeEquivalenceEstimator.merge_revision = merge_revision
    DevelopmentalPeerSupervisor.run_once = base_run_once
    OutcomeEquivalenceEstimator._v828_holdout_installed = True
