from __future__ import annotations

from collections import Counter

from v8 import outcome_holdout_v828 as holdout
from v8 import outcome_holdout_diagnostics_v828 as diagnostics
from v8 import information_flow_diagnostics as flow


def _aggregated_rows(full_class, by_uid, occurrences, target_game: int):
    training_rows = []
    holdout_rows = []
    full_rows = []
    training_members = set()
    holdout_members = set()
    training_games = set()
    fine_uids = tuple(
        uid for uid in full_class.members if uid != full_class.uid and uid in by_uid
    )
    for uid in fine_uids:
        root = by_uid[uid]
        per_game = occurrences.get(uid, {})
        target_support = int(per_game.get(int(target_game), 0))
        training_support = sum(
            int(support)
            for game, support in per_game.items()
            if int(game) != int(target_game)
        )
        total_support = training_support + target_support
        if training_support > 0:
            training_rows.append(holdout._pseudo_row(root, -1, training_support))
            training_members.add(uid)
            training_games.update(
                int(game) for game, support in per_game.items()
                if int(game) != int(target_game) and int(support) > 0
            )
        if target_support > 0:
            holdout_rows.append(holdout._pseudo_row(root, int(target_game), target_support))
            holdout_members.add(uid)
        if total_support > 0:
            full_rows.append(holdout._pseudo_row(root, -2, total_support))
    return (
        fine_uids,
        tuple(training_rows),
        tuple(holdout_rows),
        tuple(full_rows),
        training_members,
        holdout_members,
        training_games,
    )


def _select_occurrence_holdout_class(
    estimator,
    full_class,
    by_uid,
    occurrences,
    rejection_counts=None,
):
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
        (
            _fine_uids,
            training_rows,
            holdout_rows,
            full_rows,
            training_members,
            holdout_members,
            training_games,
        ) = _aggregated_rows(full_class, by_uid, occurrences, int(target_game))
        if not holdout_rows or len(training_members) < 2 or not training_games:
            rejected["insufficient_disjoint_training_members"] += 1
            continue
        training_class = holdout._derive_class(
            full_class.descriptor,
            training_rows,
            uid=full_class.uid,
            version=full_class.version,
            estimator=estimator,
        )
        if not training_class.persistent:
            rejected["training_class_not_persistent"] += 1
            continue
        full_shadow = holdout._derive_class(
            full_class.descriptor,
            full_rows,
            uid=full_class.uid,
            version=full_class.version,
            estimator=estimator,
        )
        return holdout.OutcomeHoldoutValidation(
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


def _diagnose_h13_holdout(supervisor):
    nodes = tuple(supervisor.read_view.node_records())
    by_uid = {row.uid: row for row in nodes}
    estimator = holdout._shadow_estimator(supervisor.outcomes)
    classes = tuple(estimator.rebuild(nodes))
    roots = tuple(
        uid
        for outcome in classes
        for uid in outcome.members
        if uid != outcome.uid and uid in by_uid
    )
    occurrences = holdout._lineage_occurrences_by_world(supervisor.read_view, roots)
    rejections: Counter[str] = Counter()
    examples = []
    accepted_candidates = 0
    classes_with_provenance = 0
    candidate_worlds_total = 0
    for outcome in classes:
        fine_uids = tuple(
            uid for uid in outcome.members if uid != outcome.uid and uid in by_uid
        )
        member_worlds = {
            uid.hex(): sorted(int(game) for game in occurrences.get(uid, {}))
            for uid in fine_uids
        }
        candidate_games = sorted(
            {game for uid in fine_uids for game in occurrences.get(uid, {})}
        )
        candidate_worlds_total += len(candidate_games)
        if len(fine_uids) < 2:
            rejections["class_fewer_than_2_fine_members"] += 1
            continue
        if not candidate_games:
            rejections["no_lineage_game_provenance"] += 1
            continue
        classes_with_provenance += 1
        class_accepted = False
        for target_game in candidate_games:
            (
                _fine_uids,
                training_rows,
                holdout_rows,
                _full_rows,
                training_members,
                _holdout_members,
                training_games,
            ) = _aggregated_rows(outcome, by_uid, occurrences, int(target_game))
            rejection = ""
            training_class = None
            if not holdout_rows:
                rejection = "target_has_no_holdout_occurrences"
            elif len(training_members) < 2:
                rejection = "fewer_than_2_training_members"
            elif not training_games:
                rejection = "no_training_worlds"
            else:
                training_class = holdout._derive_class(
                    outcome.descriptor,
                    training_rows,
                    uid=outcome.uid,
                    version=outcome.version,
                    estimator=estimator,
                )
                if not training_class.persistent:
                    if training_class.support < estimator.min_support:
                        rejection = "training_support_below_minimum"
                    elif training_class.stability < estimator.stability_threshold:
                        rejection = "training_stability_below_threshold"
                    elif training_class.context_consistency < estimator.context_consistency_threshold:
                        rejection = "training_context_consistency_below_threshold"
                    elif training_class.within_class_diameter > estimator.max_diameter:
                        rejection = "training_diameter_above_threshold"
                    elif training_class.predictive_interchangeability < estimator.interchangeability_threshold:
                        rejection = "training_interchangeability_below_threshold"
                    else:
                        rejection = "training_not_persistent"
            if rejection:
                rejections[rejection] += 1
            else:
                accepted_candidates += 1
                class_accepted = True
            if len(examples) < flow.MAX_EXAMPLES:
                examples.append({
                    "m6_uid": outcome.uid.hex(),
                    "descriptor": list(outcome.descriptor),
                    "fine_member_count": len(fine_uids),
                    "member_worlds": member_worlds,
                    "candidate_worlds": candidate_games,
                    "target_world": int(target_game),
                    "training_member_count": len(training_members),
                    "training_worlds": sorted(training_games),
                    "training_occurrences": sum(int(row.support_count) for row in training_rows),
                    "holdout_occurrences": sum(int(row.support_count) for row in holdout_rows),
                    "training_support": 0 if training_class is None else int(training_class.support),
                    "training_stability": 0.0 if training_class is None else float(training_class.stability),
                    "training_context_consistency": 0.0 if training_class is None else float(training_class.context_consistency),
                    "training_diameter": 0.0 if training_class is None else float(training_class.within_class_diameter),
                    "training_interchangeability": 0.0 if training_class is None else float(training_class.predictive_interchangeability),
                    "aggregation": "one_shadow_row_per_fine_m6_member",
                    "accepted": not bool(rejection),
                    "rejection": rejection,
                })
            if not rejection:
                break
        if not class_accepted:
            rejections["class_has_no_eligible_holdout"] += 1
    return {
        "class_count": len(classes),
        "fine_root_count": len(set(roots)),
        "roots_with_provenance": sum(1 for uid in set(roots) if occurrences.get(uid)),
        "classes_with_provenance": classes_with_provenance,
        "candidate_worlds_total": candidate_worlds_total,
        "accepted_candidates": accepted_candidates,
        "rejections": dict(rejections),
        "examples": examples,
    }


def install_outcome_holdout_member_aggregation_fix_v828() -> None:
    holdout._select_occurrence_holdout_class = _select_occurrence_holdout_class
    diagnostics._diagnose_h13_holdout = _diagnose_h13_holdout
