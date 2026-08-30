from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


_REQUIRED_SECTIONS = (
    "run",
    "outcomes",
    "development",
    "transfer",
    "control",
    "chain_audit",
    "hypotheses",
)


def build_evidence_package(**sections: Mapping[str, Any]) -> dict[str, Any]:
    missing = [name for name in _REQUIRED_SECTIONS if name not in sections]
    if missing:
        raise ValueError(f"missing evidence sections: {', '.join(missing)}")
    package = {name: dict(value) for name, value in sections.items()}
    run = package["run"]
    for key in ("revision", "games", "steps"):
        if key not in run:
            raise ValueError(f"run.{key} is required")
    return package


def write_evidence_package(path: str | Path, package: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(json.dumps(package, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp, target)
    return target
