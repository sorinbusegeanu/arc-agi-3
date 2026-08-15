from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from v7.derivation.scientific import (
    TYPE_CONCEPT,
    TYPE_FAMILY,
    TYPE_ROLE,
    TYPE_STRATEGY,
    TYPE_WORLD_MODEL,
    world_transition_signature,
)
from v7.memory.concept_validation import ConceptValidationStatus
from v7.memory.evidence_types import EvidenceType
from v7.memory.ids import MemoryId, MemoryLevel
from v7.memory.planning import TYPE_EXECUTABLE_PROCEDURE, planning_context
from v7.memory.reporting import StrictHypothesisReporter, report_as_dict
from v7.memory.status import MemoryStatus, memory_is_active
from v7.runtime import V7Runtime

NAMES = {
    "H01": "Contingency emergence from interaction history",
    "H02": "Prediction violations drive attention and memory",
    "H03": "Transformation-family formation",
    "H04": "Carrier emergence",
    "H05": "Functional-role emergence",
    "H06": "Role transfer across contexts or games",
    "H07": "Concept emergence through transfer validation",
    "H08": "World-model coherence and later predictive recurrence",
    "H09": "Future-option motif emergence",
    "H10": "Future-option change attracts selective attention",
    "H11": "Future-option transfer supports validated concepts",
    "H12": "Trajectory-efficiency emergence",
}

DEPENDENCIES = {
    "H03": ("H01",),
    "H04": ("H03",),
    "H05": ("H04",),
    "H06": ("H05",),
    "H07": ("H06",),
    "H08": ("H06", "H07"),
    "H10": ("H09",),
    "H11": ("H06", "H09"),
}

ISSUES = {
    "H01": [
        "Whole-grid context identities can fragment recurrence when irrelevant pixels differ.",
        "Support is observational under the current sampler and is not an intervention test.",
    ],
    "H02": [
        "Prediction trackers are worker-local and restart for each sampling job, so violations do not yet use a shared learned predictor.",
        "Replay evidence is linked at memory identity level, not to the exact violating event.",
        "Prediction error and learning value are coupled, so a clean causal claim still requires ablation.",
    ],
    "H03": [
        "Translation-normalized signatures may merge causally different transformations with identical local change patterns.",
        "Action+transformation families may require richer relational alignment for stronger abstraction.",
    ],
    "H04": [
        "Carrier signatures are localized change-region hypotheses, not persistent tracked objects.",
        "Identical local geometry/value patterns can conflate distinct carriers.",
        "H03-before-H04 order is partly construction-imposed because carriers are recognized only after family links exist.",
    ],
    "H05": [
        "Role identity can still overmerge distinct functions when their empirical signatures coincide.",
        "Role evidence is recurring support, not a separately learned carrier-to-role causal model.",
    ],
    "H06": [
        "Transfer trials are observational and policy-selected rather than counterfactual.",
        "Trajectory credit remains distributed across all memories actually used before the terminal outcome.",
        "Multi-source provenance is excluded from the strongest single-source transfer test.",
    ],
    "H07": [
        "VALID requires transfer evidence into games outside the concept formation provenance.",
        "Later unseen-scope trials remain stronger evidence than formation-set recurrence.",
    ],
    "H08": [
        "World-model creation alone does not prove prediction; VALID requires the same model to recur later and across games.",
        "M5 influences action selection through indexed planning strength rather than a separately trained predictor head.",
    ],
    "H09": [
        "Direct evidence measures immediate available-action breadth; deeper reachable-state option sets remain outside this test.",
    ],
    "H10": [
        "The paired scorer ablation isolates the additive future-option mechanism but does not ablate all planning mechanisms.",
        "VALID additionally requires the intervention to change action ordering or top choice in high-option-change states.",
    ],
    "H11": [
        "Only successful post-validation transfer into games outside the validation scope counts as strong held-out evidence.",
    ],
    "H12": [
        "Only successful trajectories are comparable; fixed horizons censor unsolved trajectories.",
        "Local level ordinals may not perfectly align semantics across resets.",
        "M6 construction alone is not behavioral preference without later replay, promotion, or actual procedure use.",
    ],
}


def evaluate_hypothesis_suite(
    runtime: V7Runtime,
    *,
    epoch: int,
    output_root: str | Path,
    workers: int = 1,
) -> dict[str, dict[str, Any]]:
    snapshot = _Snapshot(runtime)
    rows = {
        "H01": _h01(snapshot),
        "H02": _h02(snapshot),
        "H03": _h03(snapshot),
        "H04": _h04(snapshot),
        "H05": _h05(snapshot),
        "H06": _h06(snapshot),
        "H07": _h07(snapshot),
        "H08": _h08(snapshot),
        "H09": _h09(snapshot),
        "H10": _h10(snapshot),
        "H11": _h11(snapshot),
        "H12": _h12(snapshot),
    }

    # Dependency contracts are strict and transitive. PARTIALLY_VALID evidence
    # is not sufficient to validate a downstream scientific claim.
    for hypothesis_id, dependencies in DEPENDENCIES.items():
        blocked = [
            dependency
            for dependency in dependencies
            if rows[dependency]["raw_decision"] != "VALID"
            or rows[dependency].get("dependency_gate", "PASS") != "PASS"
        ]
        if blocked:
            rows[hypothesis_id]["dependency_gate"] = "FAIL"
            rows[hypothesis_id]["evidence"]["blocked_by_dependencies"] = blocked

    reports = StrictHypothesisReporter().evaluate_suite(rows, workers=workers)
    output = Path(output_root) / "reports" / f"epoch_{epoch + 1:04d}"
    output.mkdir(parents=True, exist_ok=True)
    detailed: dict[str, dict[str, Any]] = {}
    blocker_rows: list[dict[str, Any]] = []
    for hypothesis_id, report in reports.items():
        payload = report_as_dict(report)
        payload["hypothesis_name"] = NAMES[hypothesis_id]
        payload["missing_evidence"] = list(
            rows[hypothesis_id].get("missing_evidence", ())
        )
        payload["potential_issues"] = list(ISSUES[hypothesis_id])
        dependency_blockers = [
            f"blocked by {dependency}={rows[dependency]['raw_decision']}"
            for dependency in rows[hypothesis_id]["evidence"].get(
                "blocked_by_dependencies", ()
            )
        ]
        local_blockers, next_required = _validity_blockers(
            hypothesis_id,
            rows[hypothesis_id],
        )
        blockers = list(dependency_blockers) + local_blockers
        if report.final_decision != "VALID":
            if report.quality_gate != "PASS":
                blockers.append(f"quality gate is {report.quality_gate}")
            blockers.extend(
                f"missing required report field: {field}"
                for field in report.missing_fields
            )
            if bool(report.evidence.get("proxy_only")):
                blockers.append("proxy-only evidence cannot produce VALID")
            if not blockers:
                blockers.append(
                    f"raw decision remains {report.raw_decision} despite satisfied VALID measurements"
                )
        else:
            dependency_blockers = []
            blockers = []
            next_required = []
        payload["dependency_blockers"] = dependency_blockers
        payload["blockers"] = blockers
        payload["next_required_evidence"] = next_required
        detailed[hypothesis_id] = payload
        blocker_rows.append(
            {
                "epoch": epoch + 1,
                "hypothesis_id": hypothesis_id,
                "hypothesis_name": NAMES[hypothesis_id],
                "raw_decision": report.raw_decision,
                "final_decision": report.final_decision,
                "valid": report.final_decision == "VALID",
                "dependency_blockers": dependency_blockers,
                "blockers": blockers,
                "next_required_evidence": next_required,
                "measurements": dict(report.evidence["measurement"]),
            }
        )
        (output / f"{hypothesis_id.lower()}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    (output / "hypotheses.json").write_text(
        json.dumps(detailed, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_blocker_log(output / "hypothesis_blockers.jsonl", blocker_rows)
    return detailed


def _write_blocker_log(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _validity_blockers(
    hypothesis_id: str,
    row: dict[str, Any],
) -> tuple[list[str], list[str]]:
    measurements = row["evidence"]["measurement"]
    blockers: list[str] = []
    required: list[str] = []

    def threshold(field: str, target: float, label: str) -> None:
        value = measurements.get(field)
        if value is None or float(value) < target:
            shown = (
                "unavailable"
                if value is None
                else f"{float(value):.3f}"
                if isinstance(value, float)
                else str(value)
            )
            goal = (
                f"{target:.3f}"
                if isinstance(target, float) and not target.is_integer()
                else str(int(target))
            )
            blockers.append(f"{label} {shown}/{goal}")
            required.append(f"increase {label} to at least {goal}")

    def truth(field: str, blocker: str, requirement: str) -> None:
        if measurements.get(field) is not True:
            blockers.append(blocker)
            required.append(requirement)

    def positive(field: str, label: str) -> None:
        value = measurements.get(field)
        if value is None or float(value) <= 0:
            shown = (
                "unavailable" if value is None else f"{float(value):.3f}"
            )
            blockers.append(f"{label} is {shown}; required > 0")
            required.append(f"increase {label} above 0")

    if hypothesis_id == "H01":
        threshold("stable_contingency_count", 1, "stable contingencies")
        threshold(
            "games_with_stable_contingencies",
            2,
            "games with stable contingencies",
        )
        threshold("prediction_violation_count", 1, "prediction violations")
    elif hypothesis_id == "H02":
        threshold("prediction_violation_count", 1, "prediction violations")
        positive(
            "replay_rate_prediction_violation",
            "prediction-violation replay rate",
        )
        threshold("replay_lift", 1.25, "replay lift")
        truth(
            "prediction_violation_precedes_carrier",
            "prediction violation did not precede carrier emergence",
            "observe a prediction violation no later than first carrier emergence",
        )
    elif hypothesis_id == "H03":
        threshold(
            "families_with_multiple_members",
            1,
            "families with at least 2 M1 parents",
        )
        if (
            int(measurements.get("cross_context_family_count") or 0) <= 0
            and int(measurements.get("cross_game_family_count") or 0) <= 0
        ):
            blockers.append("no family recurs across contexts or games")
            required.append("observe at least 1 family across contexts or games")
    elif hypothesis_id == "H04":
        threshold(
            "usable_emergent_carrier_count",
            1,
            "carriers linking at least 2 families across at least 2 contexts",
        )
        truth(
            "family_precedes_carrier",
            "family emergence did not precede carrier emergence",
            "observe family emergence no later than carrier emergence",
        )
    elif hypothesis_id == "H05":
        threshold(
            "usable_role_count",
            1,
            "roles linking at least 2 M1 parents across contexts or games",
        )
        truth(
            "carrier_precedes_role",
            "carrier emergence did not precede role emergence",
            "observe carrier emergence no later than role emergence",
        )
    elif hypothesis_id == "H06":
        threshold(
            "verified_single_source_cross_game_trials",
            4,
            "verified single-source cross-game trials",
        )
        threshold("successful_verified_trials", 2, "successful verified trials")
        threshold("successful_role_count", 2, "successful roles")
        threshold("distinct_game_pair_count", 2, "successful distinct game pairs")
        threshold("transfer_success_rate", 0.25, "transfer success rate")
    elif hypothesis_id == "H07":
        threshold("robust_validated_concept_count", 1, "robust validated concepts")
    elif hypothesis_id == "H08":
        threshold("world_model_count", 1, "M5 world models")
        threshold(
            "robust_recurrent_model_count",
            1,
            "same M5 models with post-creation and cross-game recurrence",
        )
    elif hypothesis_id == "H09":
        threshold(
            "recurring_future_option_motifs",
            1,
            "recurring future-option motifs",
        )
        threshold(
            "cross_game_future_option_motifs",
            1,
            "cross-game future-option motifs",
        )
    elif hypothesis_id == "H10":
        threshold("high_option_change_count", 5, "high option-change events")
        threshold("low_option_change_count", 5, "low option-change events")
        threshold("causal_attention_lift", 1.25, "causal attention lift")
        threshold(
            "high_option_choice_change_count",
            1,
            "high-option states with changed top action",
        )
        truth(
            "behavioral_effect_concentrated_in_high_option_states",
            "future-option ablation does not change action ordering more often in high-option states",
            "observe a stronger top-action/rank effect in high-option-change states than low-option states",
        )
        if measurements.get("causal_ablation_available") is not True:
            blockers.append("causal ablation is unavailable")
            required.append("produce paired future-option scorer ablation evidence")
    elif hypothesis_id == "H11":
        threshold(
            "successful_post_validation_chains",
            2,
            "successful post-validation chains",
        )
        threshold(
            "distinct_post_validation_target_games",
            2,
            "distinct held-out target games",
        )
        if int(
            measurements.get("validated_concepts_with_recorded_generation") or 0
        ) <= 0:
            blockers.append("validated concepts with recorded generation 0/1")
            required.append("record validation generation for at least 1 concept")
    elif hypothesis_id == "H12":
        threshold(
            "comparable_trajectory_groups",
            1,
            "comparable trajectory groups",
        )
        threshold(
            "best_known_improvement_count",
            1,
            "best-known trajectory improvements",
        )
        threshold("strategy_count", 1, "M6 strategies")
        truth(
            "strategy_behavioral_or_lifecycle_link",
            "no M6 strategy has post-creation behavioral use, replay, or promotion",
            "observe post-creation M6 use or link at least 1 M6 strategy to replay/promotion",
        )
    else:
        raise KeyError(hypothesis_id)
    return blockers, required


class _Snapshot:
    def __init__(self, runtime: V7Runtime) -> None:
        self.runtime = runtime
        self.view = runtime.writer.published_view
        self.nodes = self.view.nodes
        self.registry = getattr(runtime.writer, "_canonical_registry")
        self.parents: dict[int, set[int]] = defaultdict(set)
        self.direct_games: dict[int, set[str]] = defaultdict(set)
        self.direct_contexts: dict[int, set[str]] = defaultdict(set)
        rows = runtime.lifecycle_evidence.connection.execute(
            "SELECT memory_id,parent_memory_id,source_game,source_context FROM provenance_records"
        ).fetchall()
        for mid, parent, game, context in rows:
            memory_id = int(mid)
            if parent is not None:
                self.parents[memory_id].add(int(parent))
            if game:
                self.direct_games[memory_id].add(str(game))
            if context:
                self.direct_contexts[memory_id].add(str(context))
        self._games: dict[int, set[str]] = {}
        self._contexts: dict[int, set[str]] = {}
        self.episodes = self._load(EvidenceType.EPISODE)
        self.trajectories = self._load(EvidenceType.TRAJECTORY)
        self.concept_validations = self._load(EvidenceType.CONCEPT_VALIDATION)
        self.replay_ids = {
            int(row[0])
            for row in runtime.evidence.connection.execute(
                "SELECT DISTINCT memory_id FROM evidence_records WHERE evidence_type=? AND memory_id IS NOT NULL",
                (int(EvidenceType.REPLAY),),
            ).fetchall()
        }
        self.promotion_ids = {
            int(row[0])
            for row in runtime.evidence.connection.execute(
                "SELECT DISTINCT memory_id FROM evidence_records WHERE evidence_type=? AND memory_id IS NOT NULL",
                (int(EvidenceType.PROMOTION),),
            ).fetchall()
        }
        self.transfer_trials = self._transfers()
        self.children: dict[int, set[int]] = defaultdict(set)
        for child, parents in self.parents.items():
            for parent in parents:
                self.children[parent].add(child)

    def _load(self, evidence_type: EvidenceType) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        rows = self.runtime.evidence.connection.execute(
            "SELECT memory_id,source_game,source_context,source_global_step,payload_json,generation_id FROM evidence_records WHERE evidence_type=? ORDER BY evidence_id",
            (int(evidence_type),),
        ).fetchall()
        for mid, game, context, step, payload_json, generation in rows:
            try:
                payload = json.loads(str(payload_json or "{}"))
            except json.JSONDecodeError:
                payload = {}
            payload.update(
                {
                    "memory_id": None if mid is None else int(mid),
                    "source_game": game,
                    "source_context": context,
                    "source_global_step": step,
                    "generation_id": int(generation),
                }
            )
            result.append(payload)
        return result

    def _transfers(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        rows = self.runtime.lifecycle_evidence.connection.execute(
            "SELECT memory_id,source_game,target_game,success,score,payload_json,generation_id FROM transfer_trials ORDER BY transfer_trial_id"
        ).fetchall()
        for mid, source, target, success, score, payload_json, generation in rows:
            try:
                payload = json.loads(str(payload_json or "{}"))
            except json.JSONDecodeError:
                payload = {}
            payload.update(
                {
                    "memory_id": int(mid),
                    "source_game": str(source),
                    "target_game": str(target),
                    "success": bool(success),
                    "score": float(score),
                    "generation_id": int(generation),
                }
            )
            result.append(payload)
        return result

    def _scope(
        self,
        mid: int,
        direct: dict[int, set[str]],
        cache: dict[int, set[str]],
    ) -> set[str]:
        if mid in cache:
            return set(cache[mid])
        visited: set[int] = set()

        def walk(memory_id: int) -> set[str]:
            if memory_id in visited:
                return set()
            visited.add(memory_id)
            values = set(direct.get(memory_id, ()))
            for parent in self.parents.get(memory_id, ()):
                values.update(walk(parent))
            return values

        value = walk(mid)
        cache[mid] = set(value)
        return value

    def source_games(self, mid: int) -> set[str]:
        return self._scope(int(mid), self.direct_games, self._games)

    def source_contexts(self, mid: int) -> set[str]:
        return self._scope(int(mid), self.direct_contexts, self._contexts)

    def nodes_at(
        self,
        level: MemoryLevel,
        type_id: int | None = None,
    ):
        return [
            (int(mid), node)
            for mid, node in self.nodes.items()
            if node.level == level
            and (type_id is None or node.type_id == type_id)
        ]

    def ancestors_at_level(self, mid: int, level: MemoryLevel) -> set[int]:
        found: set[int] = set()
        visited: set[int] = set()
        stack = list(self.parents.get(int(mid), ()))
        while stack:
            current = int(stack.pop())
            if current in visited:
                continue
            visited.add(current)
            node = self.nodes.get(MemoryId(current))
            if node is not None and node.level == level:
                found.add(current)
                continue
            stack.extend(self.parents.get(current, ()))
        return found


def _base(
    decision: str,
    rows: int,
    measurement: Any,
    *,
    proxy: bool = False,
    missing: Iterable[str] = (),
) -> dict[str, Any]:
    return {
        "raw_decision": decision,
        "quality_gate": "PASS",
        "dependency_gate": "PASS",
        "evidence": {
            "evidence_rows": int(rows),
            "measurement": measurement,
            "proxy_only": bool(proxy),
        },
        "missing_evidence": list(missing),
    }


def _h01(snapshot: _Snapshot):
    m1 = snapshot.nodes_at(MemoryLevel.M1)
    stable = [(mid, node) for mid, node in m1 if int(node.support_count) >= 2]
    games = {
        game
        for mid, _node in stable
        for game in snapshot.source_games(mid)
    }
    prediction_violations = sum(
        float(row.get("prediction_error") or 0) > 0
        for row in snapshot.episodes
    )
    decision = (
        "VALID"
        if stable and len(games) >= 2 and prediction_violations > 0
        else "PARTIALLY_VALID"
        if m1
        else "INSUFFICIENT_EVIDENCE"
    )
    return _base(
        decision,
        len(snapshot.episodes),
        {
            "interaction_count": len(snapshot.episodes),
            "contingency_count": len(m1),
            "stable_contingency_count": len(stable),
            "games_with_stable_contingencies": len(games),
            "prediction_violation_count": prediction_violations,
        },
    )


def _carrier_metrics(snapshot: _Snapshot) -> dict[str, Any]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in snapshot.episodes:
        if row.get("carrier_signature") is not None:
            groups[int(row["carrier_signature"])].append(row)
    usable = 0
    cross_game = 0
    max_families = 0
    first: int | None = None
    for rows in groups.values():
        families: set[int] = set()
        contexts: set[str] = set()
        games: set[str] = set()
        raw_generation: int | None = None
        for row in sorted(
            rows,
            key=lambda item: (
                int(item.get("generation_id") or 0),
                int(item.get("source_global_step") or 0),
            ),
        ):
            families.update(
                child
                for child in snapshot.children.get(
                    int(row.get("memory_id") or -1), ()
                )
                if child in snapshot.nodes
                and snapshot.nodes[MemoryId(child)].level == MemoryLevel.M2
            )
            contexts.add(str(row.get("source_context") or ""))
            games.add(str(row.get("source_game") or ""))
            if len(contexts) >= 2 and raw_generation is None:
                raw_generation = int(row.get("generation_id") or 0)
        max_families = max(max_families, len(families))
        if len(families) >= 2 and len(contexts) >= 2:
            usable += 1
            cross_game += int(len(games) >= 2)
            family_ready = max(
                (
                    int(snapshot.nodes[MemoryId(family_id)].created_generation)
                    for family_id in families
                ),
                default=raw_generation or 0,
            )
            recognized = max(raw_generation or 0, family_ready)
            first = recognized if first is None else min(first, recognized)
    return {
        "carrier_candidate_count": len(groups),
        "usable_emergent_carrier_count": usable,
        "cross_game_carrier_count": cross_game,
        "max_linked_family_count": max_families,
        "first_emergent_carrier_generation": first,
    }


def _h02(snapshot: _Snapshot):
    violating = [
        row
        for row in snapshot.episodes
        if float(row.get("prediction_error") or 0) > 0
    ]
    non_violating = [
        row
        for row in snapshot.episodes
        if float(row.get("prediction_error") or 0) <= 0
    ]

    def rate(rows):
        if not rows:
            return None
        return sum(
            int(row.get("memory_id") or -1) in snapshot.replay_ids
            for row in rows
        ) / len(rows)

    high = rate(violating)
    low = rate(non_violating)
    if high is None or low is None:
        lift = None
    elif low == 0 and high > 0:
        lift = float("inf")
    elif low > 0:
        lift = high / low
    else:
        lift = 1.0
    carrier = _carrier_metrics(snapshot)["first_emergent_carrier_generation"]
    first = min(
        (int(row.get("generation_id") or 0) for row in violating),
        default=None,
    )
    precedes = first is not None and (carrier is None or first <= carrier)
    decision = (
        "VALID"
        if violating
        and high
        and (lift == float("inf") or (lift is not None and lift >= 1.25))
        and precedes
        else "PARTIALLY_VALID"
        if violating and high
        else "INVALID"
        if len(violating) >= 5
        else "INSUFFICIENT_EVIDENCE"
    )
    return _base(
        decision,
        len(violating),
        {
            "prediction_violation_count": len(violating),
            "replay_rate_prediction_violation": high,
            "replay_rate_non_violation": low,
            "replay_lift": lift,
            "prediction_violation_precedes_carrier": precedes,
        },
    )


def _h03(snapshot: _Snapshot):
    families = snapshot.nodes_at(MemoryLevel.M2, TYPE_FAMILY)
    multi = cross_context = cross_game = 0
    for mid, _node in families:
        m1 = [
            parent
            for parent in snapshot.parents.get(mid, ())
            if parent in snapshot.nodes
            and snapshot.nodes[MemoryId(parent)].level == MemoryLevel.M1
        ]
        multi += int(len(m1) >= 2)
        cross_context += int(len(snapshot.source_contexts(mid)) >= 2)
        cross_game += int(len(snapshot.source_games(mid)) >= 2)
    decision = (
        "VALID"
        if multi and (cross_context or cross_game)
        else "PARTIALLY_VALID"
        if families
        else "INVALID"
        if snapshot.nodes_at(MemoryLevel.M1)
        else "INSUFFICIENT_EVIDENCE"
    )
    return _base(
        decision,
        len(families),
        {
            "family_count": len(families),
            "families_with_multiple_members": multi,
            "cross_context_family_count": cross_context,
            "cross_game_family_count": cross_game,
        },
    )


def _h04(snapshot: _Snapshot):
    measurement = _carrier_metrics(snapshot)
    first_family = min(
        (
            int(node.created_generation)
            for _mid, node in snapshot.nodes_at(MemoryLevel.M2, TYPE_FAMILY)
        ),
        default=None,
    )
    first_carrier = measurement["first_emergent_carrier_generation"]
    temporal = (
        first_family is not None
        and first_carrier is not None
        and first_family <= first_carrier
    )
    measurement.update(
        {
            "first_family_generation": first_family,
            "family_precedes_carrier": temporal
            if first_carrier is not None
            else None,
        }
    )
    decision = (
        "VALID"
        if measurement["usable_emergent_carrier_count"] and temporal
        else "PARTIALLY_VALID"
        if measurement["carrier_candidate_count"]
        else "INSUFFICIENT_EVIDENCE"
    )
    return _base(
        decision,
        measurement["carrier_candidate_count"],
        measurement,
    )


def _h05(snapshot: _Snapshot):
    roles = snapshot.nodes_at(MemoryLevel.M3, TYPE_ROLE)
    usable = cross_context = cross_game = 0
    for mid, _node in roles:
        m1 = snapshot.ancestors_at_level(mid, MemoryLevel.M1)
        contexts = snapshot.source_contexts(mid)
        games = snapshot.source_games(mid)
        cross_context += int(len(contexts) >= 2)
        cross_game += int(len(games) >= 2)
        usable += int(len(m1) >= 2 and (len(contexts) >= 2 or len(games) >= 2))
    carrier = _carrier_metrics(snapshot)
    first_role = min(
        (int(node.created_generation) for _mid, node in roles),
        default=None,
    )
    first_carrier = carrier["first_emergent_carrier_generation"]
    temporal = (
        first_role is not None
        and first_carrier is not None
        and first_carrier <= first_role
    )
    decision = (
        "VALID"
        if usable and temporal
        else "PARTIALLY_VALID"
        if roles
        else "INSUFFICIENT_EVIDENCE"
    )
    return _base(
        decision,
        len(roles),
        {
            "role_count": len(roles),
            "usable_role_count": usable,
            "cross_context_role_count": cross_context,
            "cross_game_role_count": cross_game,
            "carrier_precedes_role": temporal
            if first_carrier is not None
            else None,
        },
    )


def _preferred_transfer_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trajectory = [
        row for row in rows if row.get("attribution") == "trajectory_usage"
    ]
    return trajectory or rows


def _h06(snapshot: _Snapshot):
    role_ids = {
        mid
        for mid, node in snapshot.nodes_at(MemoryLevel.M3, TYPE_ROLE)
        if memory_is_active(node)
    }
    recorded = [
        row
        for row in snapshot.transfer_trials
        if int(row["memory_id"]) in role_ids
    ]
    rows = _preferred_transfer_rows(recorded)
    verified = [
        row
        for row in rows
        if int(row.get("source_game_count") or 1) == 1
        and row.get("source_game") != row.get("target_game")
    ]
    successful = [row for row in verified if row.get("success")]
    roles = {int(row["memory_id"]) for row in successful}
    successful_pairs = {
        (row["source_game"], row["target_game"]) for row in successful
    }
    rate = len(successful) / len(verified) if verified else None
    decision = (
        "VALID"
        if len(verified) >= 4
        and len(successful) >= 2
        and len(roles) >= 2
        and len(successful_pairs) >= 2
        and (rate or 0) >= 0.25
        else "INVALID"
        if len(verified) >= 4 and not successful
        else "PARTIALLY_VALID"
        if verified
        else "INSUFFICIENT_EVIDENCE"
    )
    return _base(
        decision,
        len(verified),
        {
            "recorded_role_transfer_trials": len(recorded),
            "verified_single_source_cross_game_trials": len(verified),
            "successful_verified_trials": len(successful),
            "successful_role_count": len(roles),
            "distinct_game_pair_count": len(successful_pairs),
            "transfer_success_rate": rate,
        },
    )


def _h07(snapshot: _Snapshot):
    concepts = snapshot.nodes_at(MemoryLevel.M4, TYPE_CONCEPT)
    candidates = [
        (mid, node)
        for mid, node in concepts
        if int(node.status_flags) & int(ConceptValidationStatus.CANDIDATE)
    ]
    validated = [
        (mid, node)
        for mid, node in concepts
        if memory_is_active(node)
        and int(node.status_flags)
        & int(ConceptValidationStatus.TRANSFER_VALIDATED)
    ]
    rejected = [
        (mid, node)
        for mid, node in concepts
        if int(node.status_flags)
        & int(ConceptValidationStatus.TRANSFER_REJECTED)
    ]
    heldout_validations: dict[int, dict[str, Any]] = {}
    for row in snapshot.concept_validations:
        if (
            row.get("memory_id") is None
            or not row.get("validated")
            or row.get("heldout_validation") is not True
        ):
            continue
        mid = int(row["memory_id"])
        previous = heldout_validations.get(mid)
        if previous is None or int(row.get("generation_id") or 0) < int(
            previous.get("generation_id") or 0
        ):
            heldout_validations[mid] = row

    robust = 0
    for mid, _node in validated:
        validation = heldout_validations.get(mid)
        if validation is None:
            continue
        scope = {
            str(value)
            for value in validation.get("validation_source_games", ())
            if value
        }
        role_parents = [
            parent
            for parent in snapshot.parents.get(mid, ())
            if parent in snapshot.nodes
            and snapshot.nodes[MemoryId(parent)].level == MemoryLevel.M3
        ]
        candidate_trials = [
            row
            for row in snapshot.transfer_trials
            if int(row["memory_id"]) == mid
            and str(row.get("target_game") or "") not in scope
        ]
        trials = _preferred_transfer_rows(candidate_trials)
        successful = [row for row in trials if row.get("success")]
        robust += int(
            len(role_parents) >= 2
            and len(trials) >= 2
            and bool(successful)
            and len(snapshot.source_games(mid)) >= 2
        )
    decision = (
        "VALID"
        if robust
        else "INVALID"
        if concepts and len(rejected) == len(concepts)
        else "PARTIALLY_VALID"
        if concepts
        else "INSUFFICIENT_EVIDENCE"
    )
    return _base(
        decision,
        len(concepts),
        {
            "concept_count": len(concepts),
            "concept_candidate_count": len(candidates),
            "transfer_validated_concept_count": len(validated),
            "transfer_rejected_concept_count": len(rejected),
            "heldout_validation_record_count": len(heldout_validations),
            "robust_validated_concept_count": robust,
        },
    )


def _continuous_rows(prior: dict[str, Any], current: dict[str, Any]) -> bool:
    prior_segment = str(prior.get("trajectory_segment_id") or "")
    current_segment = str(current.get("trajectory_segment_id") or "")
    if prior_segment or current_segment:
        if not prior_segment or prior_segment != current_segment:
            return False
    if bool(current.get("reset_boundary_before_step")):
        return False
    if int(prior.get("terminal_polarity") or 0) != 0:
        return False
    prior_step = prior.get("source_global_step")
    current_step = current.get("source_global_step")
    if prior_step is not None and current_step is not None:
        if int(current_step) != int(prior_step) + 1:
            return False
    next_contexts = prior.get("next_context_signatures", ()) or ()
    if next_contexts:
        after_context = planning_context(next_contexts, fallback=-1)
        before_context = planning_context(
            current.get("context_signatures", ()) or (),
            fallback=int(current.get("context_signature") or -2),
        )
        if after_context < 0 or before_context < 0 or after_context != before_context:
            return False
    return True


def _transitions(snapshot: _Snapshot):
    validated = {
        mid
        for mid, node in snapshot.nodes_at(MemoryLevel.M4, TYPE_CONCEPT)
        if memory_is_active(node)
        and int(node.status_flags)
        & int(ConceptValidationStatus.TRANSFER_VALIDATED)
    }
    by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
    output: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in snapshot.episodes:
        if row.get("source_game"):
            by_game[str(row["source_game"])].append(row)
    for rows in by_game.values():
        prior_row: dict[str, Any] | None = None
        prior: tuple[int, ...] = ()
        prior_action: int | None = None
        for row in sorted(
            rows,
            key=lambda item: int(item.get("source_global_step") or -1),
        ):
            current = tuple(
                sorted(
                    {
                        int(value)
                        for value in row.get("decision_concept_ids", ()) or ()
                        if int(value) in validated
                    }
                )
            )
            if (
                prior_row is not None
                and _continuous_rows(prior_row, row)
                and prior
                and current
                and prior_action is not None
                and len(set(prior) | set(current)) >= 2
            ):
                signature = world_transition_signature(
                    prior,
                    prior_action,
                    current,
                )
                output[signature].append(row)
            prior_row = row
            prior = current
            prior_action = int(row.get("action_id") or 0)
    return output


def _h08(snapshot: _Snapshot):
    models = [
        (mid, node)
        for mid, node in snapshot.nodes_at(MemoryLevel.M5, TYPE_WORLD_MODEL)
        if memory_is_active(node)
    ]
    occurrences = _transitions(snapshot)
    heldout = 0
    cross_game = 0
    robust = 0
    for mid, node in models:
        key = snapshot.registry.key_for(MemoryId(mid))
        rows = (
            ()
            if key is None or not key.parts
            else occurrences.get(int(key.parts[0]), ())
        )
        post_creation = [
            row
            for row in rows
            if int(row.get("generation_id") or 0)
            > int(node.created_generation)
        ]
        is_heldout = bool(post_creation)
        is_cross_game = (
            len({str(row.get("source_game") or "") for row in rows}) >= 2
        )
        heldout += int(is_heldout)
        cross_game += int(is_cross_game)
        robust += int(is_heldout and is_cross_game)
    decision = (
        "VALID"
        if models and robust
        else "PARTIALLY_VALID"
        if models
        else "INSUFFICIENT_EVIDENCE"
    )
    return _base(
        decision,
        len(models),
        {
            "world_model_count": len(models),
            "models_with_post_creation_recurrence": heldout,
            "cross_game_recurrent_model_count": cross_game,
            "robust_recurrent_model_count": robust,
        },
    )


def _h09(snapshot: _Snapshot):
    nonzero = [
        row
        for row in snapshot.episodes
        if row.get("future_option_observable", True) is True
        and abs(float(row.get("raw_action_option_delta") or 0)) > 0
    ]
    motifs: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in nonzero:
        if row.get("carrier_signature") is None:
            continue
        direction = (
            1 if float(row.get("raw_action_option_delta") or 0) > 0 else -1
        )
        motifs[
            (
                int(row["carrier_signature"]),
                int(row.get("action_id") or 0),
                direction,
            )
        ].append(row)
    recurring = [
        rows
        for rows in motifs.values()
        if len(rows) >= 2
        and (
            len({str(row.get("source_context") or "") for row in rows}) >= 2
            or len({str(row.get("source_game") or "") for row in rows}) >= 2
        )
    ]
    cross = sum(
        len({str(row.get("source_game") or "") for row in rows}) >= 2
        for rows in recurring
    )
    decision = (
        "VALID"
        if recurring and cross
        else "PARTIALLY_VALID"
        if nonzero
        else "INSUFFICIENT_EVIDENCE"
    )
    return _base(
        decision,
        len(nonzero),
        {
            "direct_option_change_events": len(nonzero),
            "recurring_future_option_motifs": len(recurring),
            "cross_game_future_option_motifs": cross,
            "measurement_definition": "direct delta in environment-reported available actions after versus before the selected action; auto-reset terminal frames are excluded",
        },
    )


def _h10(snapshot: _Snapshot):
    rows = [
        row
        for row in snapshot.episodes
        if row.get("future_option_ablation_available") is True
        and row.get("future_option_observable", True) is True
    ]
    if not rows:
        return _base(
            "INSUFFICIENT_EVIDENCE",
            0,
            {
                "episode_count": len(snapshot.episodes),
                "causal_ablation_available": False,
            },
        )
    values = sorted(
        abs(float(row.get("raw_action_option_delta") or 0)) for row in rows
    )
    threshold = values[min(len(values) - 1, int(0.8 * len(values)))]
    nonzero = [value for value in values if value > 0]
    threshold = (
        threshold
        if threshold > 0
        else min(nonzero)
        if nonzero
        else 0
    )
    high = [
        row
        for row in rows
        if threshold > 0
        and abs(float(row.get("raw_action_option_delta") or 0)) >= threshold
    ]
    low = (
        [
            row
            for row in rows
            if abs(float(row.get("raw_action_option_delta") or 0)) < threshold
        ]
        if threshold > 0
        else []
    )

    def mean_effect(items):
        if not items:
            return None
        return sum(
            abs(float(row.get("future_option_ablation_score_delta") or 0))
            for row in items
        ) / len(items)

    high_effect = mean_effect(high)
    low_effect = mean_effect(low)
    if high_effect is None or low_effect is None:
        lift = None
    elif low_effect == 0 and high_effect > 0:
        lift = float("inf")
    elif low_effect > 0:
        lift = high_effect / low_effect
    else:
        lift = 1.0
    rank_lift = (
        None
        if not high
        else sum(
            float(row.get("future_option_ablation_rank_lift") or 0)
            for row in high
        )
        / len(high)
    )
    high_choice_changes = sum(
        bool(row.get("future_option_ablation_choice_changed")) for row in high
    )
    low_choice_changes = sum(
        bool(row.get("future_option_ablation_choice_changed")) for row in low
    )
    high_choice_rate = high_choice_changes / len(high) if high else None
    low_choice_rate = low_choice_changes / len(low) if low else None
    behavioral_concentration = (
        high_choice_rate is not None
        and low_choice_rate is not None
        and high_choice_rate > low_choice_rate
        and high_choice_changes > 0
    )
    enough = len(high) >= 5 and len(low) >= 5
    decision = (
        "VALID"
        if enough
        and lift is not None
        and lift >= 1.25
        and behavioral_concentration
        else "INVALID"
        if enough
        and (
            (lift is not None and lift < 1.0)
            or high_choice_changes == 0
        )
        else "PARTIALLY_VALID"
    )
    return _base(
        decision,
        len(rows),
        {
            "high_option_change_threshold": threshold,
            "high_option_change_count": len(high),
            "low_option_change_count": len(low),
            "causal_score_effect_high": high_effect,
            "causal_score_effect_low": low_effect,
            "causal_attention_lift": lift,
            "mean_high_option_rank_lift": rank_lift,
            "high_option_choice_change_count": high_choice_changes,
            "low_option_choice_change_count": low_choice_changes,
            "high_option_choice_change_rate": high_choice_rate,
            "low_option_choice_change_rate": low_choice_rate,
            "behavioral_effect_concentrated_in_high_option_states": behavioral_concentration,
            "causal_ablation_available": True,
            "measurement_definition": "paired same-state scorer intervention with future-option additive terms enabled versus removed; VALID requires an action-ordering effect",
        },
    )


def _h11(snapshot: _Snapshot):
    concept_ids = {
        mid
        for mid, node in snapshot.nodes_at(MemoryLevel.M4, TYPE_CONCEPT)
        if memory_is_active(node)
    }
    validation: dict[int, tuple[int, frozenset[str]]] = {}
    for row in snapshot.concept_validations:
        if (
            row.get("memory_id") is None
            or not row.get("validated")
            or row.get("heldout_validation") is not True
        ):
            continue
        mid = int(row["memory_id"])
        generation = int(row["generation_id"])
        scope = frozenset(
            str(value)
            for value in row.get("validation_source_games", ())
            if value
        )
        if mid not in validation or generation < validation[mid][0]:
            validation[mid] = (generation, scope)
    post: list[dict[str, Any]] = []
    held: list[dict[str, Any]] = []
    for row in snapshot.transfer_trials:
        mid = int(row["memory_id"])
        if (
            mid not in concept_ids
            or mid not in validation
            or row.get("attribution") != "trajectory_usage"
            or abs(float(row.get("raw_action_option_delta") or 0)) <= 0
        ):
            continue
        generation, scope = validation[mid]
        if (
            int(row.get("generation_id") or 0) <= generation
            or str(row.get("target_game") or "") in scope
        ):
            continue
        post.append(row)
        if row.get("success"):
            held.append(row)
    targets = {row["target_game"] for row in held}
    pairs = {
        (row.get("source_game"), row.get("target_game")) for row in held
    }
    decision = (
        "VALID"
        if len(held) >= 2 and len(targets) >= 2
        else "PARTIALLY_VALID"
        if post or validation
        else "INSUFFICIENT_EVIDENCE"
    )
    return _base(
        decision,
        len(post),
        {
            "verified_future_option_concept_transfer_chains": len(post),
            "successful_post_validation_chains": len(held),
            "distinct_post_validation_target_games": len(targets),
            "distinct_post_validation_game_pairs": len(pairs),
            "validated_concepts_with_recorded_generation": len(validation),
        },
    )


def _h12(snapshot: _Snapshot):
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    successful_trajectories = [
        row for row in snapshot.trajectories if row.get("success")
    ]
    for row in successful_trajectories:
        groups[
            (
                str(row.get("source_game") or ""),
                str(row.get("level_key") or "level"),
            )
        ].append(row)
    comparable = improvements = 0
    for rows in groups.values():
        if len(rows) < 2:
            continue
        comparable += 1
        best: int | None = None
        for row in sorted(
            rows,
            key=lambda item: int(item.get("source_global_step") or -1),
        ):
            steps = int(row.get("steps_to_success") or 0)
            if steps <= 0:
                continue
            if best is not None and steps < best:
                improvements += 1
            best = steps if best is None else min(best, steps)

    strategies = [
        (mid, node)
        for mid, node in snapshot.nodes_at(MemoryLevel.M6)
        if node.type_id in {TYPE_STRATEGY, TYPE_EXECUTABLE_PROCEDURE}
        and memory_is_active(node)
    ]
    ids = {mid for mid, _node in strategies}
    nodes = {mid: node for mid, node in strategies}
    lifecycle_link = bool(ids & (snapshot.replay_ids | snapshot.promotion_ids)) or any(
        int(node.status_flags)
        & int(MemoryStatus.PROMOTED | MemoryStatus.REPLAY_QUEUED)
        for _mid, node in strategies
    )
    post_creation_uses = 0
    executable_post_creation_uses = 0
    for row in snapshot.episodes:
        generation = int(row.get("generation_id") or 0)
        for raw_id in row.get("decision_strategy_ids", ()) or ():
            memory_id = int(raw_id)
            node = nodes.get(memory_id)
            if node is None or generation <= int(node.created_generation):
                continue
            post_creation_uses += 1
            executable_post_creation_uses += int(
                int(node.type_id) == int(TYPE_EXECUTABLE_PROCEDURE)
            )
    linked = lifecycle_link or post_creation_uses > 0
    decision = (
        "INSUFFICIENT_EVIDENCE"
        if comparable <= 0
        else "VALID"
        if improvements > 0 and strategies and linked
        else "PARTIALLY_VALID"
    )
    return _base(
        decision,
        len(successful_trajectories),
        {
            "successful_trajectory_count": len(successful_trajectories),
            "comparable_trajectory_groups": comparable,
            "best_known_improvement_count": improvements,
            "strategy_count": len(strategies),
            "strategy_replay_or_promotion_link": lifecycle_link,
            "strategy_post_creation_use_count": post_creation_uses,
            "executable_strategy_post_creation_use_count": executable_post_creation_uses,
            "strategy_behavioral_or_lifecycle_link": linked,
        },
    )
