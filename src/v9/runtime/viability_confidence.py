from __future__ import annotations

from typing import Any


def _row_confidence(runtime: Any, payload: dict[str, Any]) -> float:
    environment = payload.get("environment_instance_id")
    if environment is None:
        return float(payload.get("evidence_confidence", 1.0))
    positive = bool(payload.get("task_success", False))
    positive = positive or int(payload.get("levels_completed", 0) or 0) > 0
    positive = positive or float(payload.get("primary_valence", 0) or 0.0) > 0.0
    if positive:
        return 1.0
    confidence = getattr(runtime, "_environment_evidence_confidence", {})
    return max(0.0, min(1.0, float(confidence.get(int(environment), 1.0))))


def install_viability_confidence(runtime_cls: type) -> None:
    """Attach environment viability confidence before derivation candidates form."""
    if getattr(runtime_cls, "_viability_confidence_installed", False):
        return
    original_defer = runtime_cls._defer_base_group

    def defer_base_group(self: Any, rows: tuple[tuple[Any, dict[str, Any], tuple[Any, ...]], ...]) -> None:
        adjusted = []
        for node, raw_payload, evidence in rows:
            payload = dict(raw_payload)
            payload["evidence_confidence"] = _row_confidence(self, payload)
            adjusted.append((node, payload, evidence))
        original_defer(self, tuple(adjusted))

    runtime_cls._defer_base_group = defer_base_group
    runtime_cls._viability_confidence_installed = True
