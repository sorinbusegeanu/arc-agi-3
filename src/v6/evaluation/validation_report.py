from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class StableContingencyReport:
    context_level: int
    action: int
    transformation_family: int
    support_count: int
    confidence: float
    context_signature: tuple


@dataclass(frozen=True)
class AmbiguousActionReport:
    action: int
    total_count: int
    competitors: tuple[tuple[int, int, float], ...]
    resolved_by_level: int | None
    resolved_contingencies: tuple[StableContingencyReport, ...]


@dataclass(frozen=True)
class GameValidationReport:
    game_id: str
    classification: str
    minimum_context_level: int | None
    max_context_level: int
    transformation_family_count: int
    prediction_accuracy: float | None
    stable_contingency_count: int
    stable_by_level: dict[int, tuple[StableContingencyReport, ...]]
    ambiguous_k0_actions: tuple[AmbiguousActionReport, ...]


def build_validation_report(
    db_path: str,
    *,
    game_id: str,
    confidence_threshold: float = 0.8,
    required_actions: tuple[int, ...] = (1, 2, 3, 4),
    max_context_level: int = 3,
) -> GameValidationReport:
    connection = sqlite3.connect(db_path)
    try:
        stable = _stable_by_level(connection)
        family_count = _count_rows(connection, "transformation_families")
        accuracy = _prediction_accuracy(connection)
        maximum_level = max(0, int(max_context_level))
        minimum_level = _minimum_covering_level(stable, required_actions=required_actions, max_context_level=maximum_level)
        ambiguous = _ambiguous_k0_actions(
            connection,
            stable=stable,
            confidence_threshold=confidence_threshold,
            required_actions=required_actions,
            max_context_level=maximum_level,
        )
        return GameValidationReport(
            game_id=game_id,
            classification=_classification_for_level(minimum_level),
            minimum_context_level=minimum_level,
            max_context_level=maximum_level,
            transformation_family_count=family_count,
            prediction_accuracy=accuracy,
            stable_contingency_count=sum(len(items) for items in stable.values()),
            stable_by_level=stable,
            ambiguous_k0_actions=ambiguous,
        )
    finally:
        connection.close()


def format_validation_report(report: GameValidationReport) -> str:
    lines = [
        f"{report.game_id}: class={report.classification} min_context={_level_name(report.minimum_context_level)}",
        f"  transformation_family_count={report.transformation_family_count}",
        f"  prediction_accuracy={_format_optional_float(report.prediction_accuracy)}",
        f"  stable_contingency_count={report.stable_contingency_count}",
        "  stable contingencies:",
    ]
    for level in range(report.max_context_level + 1):
        contingencies = report.stable_by_level.get(level, ())
        lines.append(f"    K{level}:")
        if not contingencies:
            lines.append("      none")
        for contingency in contingencies:
            lines.append(
                f"      {_action_name(contingency.action)} -> T{contingency.transformation_family} "
                f"confidence={contingency.confidence:.3f} support={contingency.support_count} "
                f"context={list(contingency.context_signature)}"
            )
    lines.append("  ambiguous K0 actions:")
    if not report.ambiguous_k0_actions:
        lines.append("    none")
    for ambiguous in report.ambiguous_k0_actions:
        competitors = ", ".join(
            f"T{family}:{count}/{ambiguous.total_count} ({confidence:.3f})"
            for family, count, confidence in ambiguous.competitors
        )
        lines.append(
            f"    {_action_name(ambiguous.action)}: {competitors}; "
            f"resolved_by={_level_name(ambiguous.resolved_by_level)}"
        )
        for contingency in ambiguous.resolved_contingencies:
            lines.append(
                f"      K{contingency.context_level} {_action_name(contingency.action)} -> "
                f"T{contingency.transformation_family} confidence={contingency.confidence:.3f} "
                f"support={contingency.support_count} context={list(contingency.context_signature)}"
            )
    return "\n".join(lines)


def validation_report_to_dict(report: GameValidationReport) -> dict:
    return {
        "game_id": report.game_id,
        "classification": report.classification,
        "minimum_context_level": report.minimum_context_level,
        "max_context_level": report.max_context_level,
        "transformation_family_count": report.transformation_family_count,
        "prediction_accuracy": report.prediction_accuracy,
        "stable_contingency_count": report.stable_contingency_count,
        "stable_by_level": {
            f"K{level}": [_contingency_to_dict(item) for item in items]
            for level, items in sorted(report.stable_by_level.items())
        },
        "ambiguous_k0_actions": [
            {
                "action": _action_name(item.action),
                "action_id": item.action,
                "total_count": item.total_count,
                "competitors": [
                    {"transformation_family": family, "support_count": count, "confidence": confidence}
                    for family, count, confidence in item.competitors
                ],
                "resolved_by_level": item.resolved_by_level,
                "resolved_contingencies": [_contingency_to_dict(contingency) for contingency in item.resolved_contingencies],
            }
            for item in report.ambiguous_k0_actions
        ],
    }


def format_validation_reports(reports: tuple[GameValidationReport, ...]) -> str:
    return "\n\n".join(format_validation_report(report) for report in reports)


def validation_reports_to_json(reports: tuple[GameValidationReport, ...]) -> str:
    return json.dumps([validation_report_to_dict(report) for report in reports], indent=2)


def _stable_by_level(connection: sqlite3.Connection) -> dict[int, tuple[StableContingencyReport, ...]]:
    rows = connection.execute(
        """
        SELECT context_level, context_signature, action, transformation_family, support_count, confidence
        FROM contingencies
        ORDER BY context_level, action, confidence DESC, support_count DESC
        """
    ).fetchall()
    by_level: dict[int, list[StableContingencyReport]] = {}
    for level, context, action, family, support, confidence in rows:
        item = StableContingencyReport(
            context_level=int(level),
            action=int(action),
            transformation_family=int(family),
            support_count=int(support),
            confidence=float(confidence),
            context_signature=tuple(json.loads(context)),
        )
        by_level.setdefault(item.context_level, []).append(item)
    return {level: tuple(items) for level, items in by_level.items()}


def _minimum_covering_level(
    stable: dict[int, tuple[StableContingencyReport, ...]],
    *,
    required_actions: tuple[int, ...],
    max_context_level: int,
) -> int | None:
    covered: set[int] = set()
    required = set(int(action) for action in required_actions)
    for level in range(max(0, int(max_context_level)) + 1):
        covered.update(item.action for item in stable.get(level, ()))
        if required <= covered:
            return level
    return None


def _ambiguous_k0_actions(
    connection: sqlite3.Connection,
    *,
    stable: dict[int, tuple[StableContingencyReport, ...]],
    confidence_threshold: float,
    required_actions: tuple[int, ...],
    max_context_level: int,
) -> tuple[AmbiguousActionReport, ...]:
    rows = connection.execute(
        """
        SELECT action, actual_family, COUNT(*)
        FROM prediction_results
        WHERE actual_family IS NOT NULL
        GROUP BY action, actual_family
        ORDER BY action, COUNT(*) DESC
        """
    ).fetchall()
    counts_by_action: dict[int, Counter[int]] = {}
    for action, family, count in rows:
        counts_by_action.setdefault(int(action), Counter())[int(family)] = int(count)

    ambiguous: list[AmbiguousActionReport] = []
    for action in required_actions:
        counter = counts_by_action.get(int(action), Counter())
        total = sum(counter.values())
        if total <= 0:
            continue
        competitors = tuple(
            (family, count, count / total)
            for family, count in counter.most_common()
            if count / total >= 1.0 - confidence_threshold
        )
        best_confidence = counter.most_common(1)[0][1] / total if counter else 0.0
        if len(competitors) <= 1 and best_confidence >= confidence_threshold:
            continue
        resolved = _resolved_by_higher_context(stable, int(action), max_context_level=max_context_level)
        ambiguous.append(
            AmbiguousActionReport(
                action=int(action),
                total_count=total,
                competitors=competitors,
                resolved_by_level=resolved[0],
                resolved_contingencies=resolved[1],
            )
        )
    return tuple(ambiguous)


def _resolved_by_higher_context(
    stable: dict[int, tuple[StableContingencyReport, ...]],
    action: int,
    *,
    max_context_level: int,
) -> tuple[int | None, tuple[StableContingencyReport, ...]]:
    for level in range(1, max(0, int(max_context_level)) + 1):
        matches = tuple(item for item in stable.get(level, ()) if item.action == int(action))
        if matches:
            return level, matches
    return None, ()


def _count_rows(connection: sqlite3.Connection, table: str) -> int:
    row = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return int(row[0])


def _prediction_accuracy(connection: sqlite3.Connection) -> float | None:
    row = connection.execute(
        """
        SELECT AVG(CASE WHEN prediction_error = 0 THEN 1.0 ELSE 0.0 END), COUNT(*)
        FROM prediction_results
        WHERE prediction_error IS NOT NULL
        """
    ).fetchone()
    if row is None or int(row[1]) <= 0:
        return None
    return float(row[0])


def _classification_for_level(level: int | None) -> str:
    if level is None:
        return "D"
    if level == 0:
        return "A"
    if level == 1:
        return "B"
    return "C"


def _level_name(level: int | None) -> str:
    return "none" if level is None else f"K{int(level)}"


def _format_optional_float(value: float | None) -> str:
    return "NA" if value is None else f"{float(value):.3f}"


def _contingency_to_dict(contingency: StableContingencyReport) -> dict:
    return {
        "context_level": contingency.context_level,
        "action": _action_name(contingency.action),
        "action_id": contingency.action,
        "transformation_family": contingency.transformation_family,
        "support_count": contingency.support_count,
        "confidence": contingency.confidence,
        "context_signature": list(contingency.context_signature),
    }


def _action_name(action: int) -> str:
    return {
        0: "RESET",
        1: "UP",
        2: "DOWN",
        3: "LEFT",
        4: "RIGHT",
        5: "INTERACT",
        6: "CLICK",
        7: "UNDO",
    }.get(int(action), f"ACTION{int(action)}")
