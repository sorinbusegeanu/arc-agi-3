from __future__ import annotations

from typing import Any, Mapping

from .identity import stable_u64


def _field(observable: str, name: str, default: str = "") -> str:
    parts = str(observable).split(":")
    try:
        index = parts.index(name)
    except ValueError:
        return default
    return parts[index + 1] if index + 1 < len(parts) else default


def _sign(value: Any) -> int:
    numeric = float(value)
    return 1 if numeric > 0 else -1 if numeric < 0 else 0


def _semantic_pattern(payload: Mapping[str, Any] | None) -> tuple[tuple[int, int, int], ...]:
    if not payload:
        return ()
    rows = payload.get("semantic_effects", ()) or ()
    pattern = {
        (int(row[0]), int(row[2]), _sign(row[4]))
        for row in rows
        if isinstance(row, (list, tuple)) and len(row) == 5
    }
    return tuple(sorted(pattern))


def modality_neutral_descriptor(relation: Any, payload: Mapping[str, Any] | None = None) -> tuple[object, ...]:
    """Return a channel/environment/action-neutral transformation descriptor.

    The descriptor deliberately excludes native action identifiers, environment
    instance identity, entity identifiers, raw observation signatures and symbol
    identity. It keeps only observable structural consequence/boundary/progress
    shape so equivalent WORLD/SYMBOL/CROSS_MODAL evidence can share M2/M3.
    """
    observable = str(getattr(relation, "observable_relation", ""))
    semantic_pattern = _semantic_pattern(payload)
    changed = int(bool(semantic_pattern))
    boundary = _field(observable, "BOUNDARY", "NONE")
    success = int(_field(observable, "SUCCESS", "0") == "1")
    failure = int(_field(observable, "FAILURE", "0") == "1")
    truncated = int(_field(observable, "TRUNCATED", "0") == "1")
    levels_completed = int(_field(observable, "LEVELS_COMPLETED", "0") or 0)
    progress = int(success or levels_completed > 0)
    return (
        "V978_TRANSFORMATION",
        changed,
        len(semantic_pattern),
        semantic_pattern,
        str(boundary),
        success,
        failure,
        truncated,
        progress,
    )


def modality_neutral_family_signature(relation: Any, payload: Mapping[str, Any] | None = None) -> int:
    return stable_u64(*modality_neutral_descriptor(relation, payload), person=b"v978-family")
