from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_report(root: Path, name: str, payload: dict[str, Any]) -> Path:
    if "/" in name or ".." in name:
        raise ValueError("report name must be a simple filename")
    target = root / "reports" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target

