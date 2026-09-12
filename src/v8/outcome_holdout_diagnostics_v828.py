from __future__ import annotations

from collections import Counter

from v8 import information_flow_diagnostics as flow
from v8.outcome_holdout_v828 import (
    _derive_class,
    _lineage_occurrences_by_world,
    _pseudo_row,
    _shadow_estimator,
)
from v8.peers_v82 import V82DevelopmentalPeerSupervisor


_DIAGNOSTIC_RUN_ONCE_V828 = None


def _diagnose_h13_holdout(supervisor: V82DevelopmentalPeerSupervisor) -> dict[str, object]:
    """Describe why H13 holdout candidates are accepted or rejected.

    This is diagnostic-only: it uses an isolated shadow estimator and never appends
    evidence, changes memory, or mutates the production outcome estimator.
    """
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

    rejections: Counter[str] = Counter()
    examples: list[dict[str, object]] = []
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
            if len(examples) < flow.MAX_EXAMPLES:
                examples.append(
                    {
                        "m6_uid": outcome.uid.hex(),
                        "descriptor": list(outcome.descriptor),
                        "fine_member_count": len(fine_uids),
                        "member_worlds": member_worlds,
                        "candidate_worlds": candidate_games,
                        "rejection": "class_fewer_than_2_fine_members",
                    }
                )
            continue

        if not candidate_games:
            rejections["no_lineage_game_provenance"] += 1
            if len(examples) < flow.MAX_EXAMPLES:
                examples.append(
                    {
                        "m6_uid": outcome.uid.hex(),
                        "descriptor": list(outcome.descriptor),
                        "fine_member_count": len(fine_uids),
                        "member_worlds": member_worlds,
                        "candidate_worlds": [],
                        "rejection": "no_lineage_game_provenance",
                    }
                )
            continue

        classes_with_provenance += 1
        class_accepted = False
        for target_game in candidate_games:
            training_rows = []
            holdout_rows = []
            training_members = set()
            training_games = set()
            for uid in fine_uids:
                root = by_uid[uid]
                for game, support in sorted(occurrences.get(uid, {}).items()):
                    row = _pseudo_row(root, game, support)
                    if int(game) == int(target_game):
                        holdout_rows.append(row)
                    else:
                        training_rows.append(row)
                        training_members.add(uid)
                        training_games.add(int(game))

            rejection = ""
            training_class = None
            if not holdout_rows:
                rejection = "target_has_no_holdout_occurrences"
            elif len(training_members) < 2:
                rejection = "fewer_than_2_training_members"
            elif not training_games:
                rejection = "no_training_worlds"
            else:
                training_class = _derive_class(
                    outcome.descriptor,
                    tuple(training_rows),
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
                examples.append(
                    {
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
                        "accepted": not bool(rejection),
                        "rejection": rejection,
                    }
                )

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


def _emit_h13_diagnostics(supervisor: V82DevelopmentalPeerSupervisor) -> None:
    try:
        snapshot = _diagnose_h13_holdout(supervisor)
        if int(snapshot["class_count"]) <= 0:
            return
        flow.emit_bounded(
            "outcomes",
            "h13_holdout_diagnostics",
            input_count=int(snapshot["class_count"]),
            output_count=int(snapshot["accepted_candidates"]),
            rejection_counts=snapshot["rejections"],
            examples=snapshot["examples"],
            fields={
                "fine_root_count": int(snapshot["fine_root_count"]),
                "roots_with_provenance": int(snapshot["roots_with_provenance"]),
                "classes_with_provenance": int(snapshot["classes_with_provenance"]),
                "candidate_worlds_total": int(snapshot["candidate_worlds_total"]),
                "accepted_candidates": int(snapshot["accepted_candidates"]),
                "diagnostic_only": True,
            },
        )
    except BaseException:
        return


def install_outcome_holdout_diagnostics_v828() -> None:
    global _DIAGNOSTIC_RUN_ONCE_V828
    if getattr(V82DevelopmentalPeerSupervisor, "_v828_holdout_diagnostics_installed", False):
        return

    original_run_once = V82DevelopmentalPeerSupervisor.run_once

    def run_once(self: V82DevelopmentalPeerSupervisor):
        before_cycles = int(getattr(self, "_cycles", 0))
        result = original_run_once(self)
        if int(getattr(self, "_cycles", 0)) == before_cycles:
            return result
        _emit_h13_diagnostics(self)
        return result

    V82DevelopmentalPeerSupervisor.run_once = run_once
    _DIAGNOSTIC_RUN_ONCE_V828 = run_once
    V82DevelopmentalPeerSupervisor._v828_holdout_diagnostics_installed = True
