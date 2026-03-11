from __future__ import annotations

from pathlib import Path
from typing import Any

from v3_1.config.defaults import DEFAULT_CONFIG
from v3_1.config.schema import (
    AnalysisSection,
    DebuggingSection,
    EnvironmentSection,
    ExecutionSection,
    FeatureFlagsSection,
    MemorySection,
    PlanningSection,
    RaySection,
    RuntimeSection,
    StorageSection,
    V31Config,
    VisualizationSection,
)
from v3_1.config.validation import validate_config


def _section(section_cls, payload: dict[str, Any] | None):
    return section_cls(**(payload or {}))


def _parse_scalar(raw_value: str) -> Any:
    value = raw_value.strip()
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _set_nested(mapping: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = [part.strip() for part in dotted_key.split(".") if part.strip()]
    if not parts:
        return
    cursor = mapping
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[parts[-1]] = value


def _load_conf_payload(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        _set_nested(payload, key.strip(), _parse_scalar(raw_value))
    return payload


def load_config(path: str | None = None) -> V31Config:
    if path is None:
        config = DEFAULT_CONFIG
        validate_config(config)
        return config
    config_path = Path(path)
    payload = _load_conf_payload(config_path)
    config = V31Config(
        runtime=_section(RuntimeSection, payload.get("runtime")),
        ray=_section(RaySection, payload.get("ray")),
        environment=_section(EnvironmentSection, payload.get("environment")),
        analysis=_section(AnalysisSection, payload.get("analysis")),
        planning=_section(PlanningSection, payload.get("planning")),
        execution=_section(ExecutionSection, payload.get("execution")),
        memory=_section(MemorySection, payload.get("memory")),
        storage=_section(StorageSection, payload.get("storage")),
        visualization=_section(VisualizationSection, payload.get("visualization")),
        debugging=_section(DebuggingSection, payload.get("debugging")),
        feature_flags=_section(FeatureFlagsSection, payload.get("feature_flags")),
    )
    validate_config(config)
    return config
