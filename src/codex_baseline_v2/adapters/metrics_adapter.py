from __future__ import annotations

import json
from typing import Any, Dict, List


def load_v1_episode_metrics(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def summarize_v1_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"episodes": 0, "mean_return": 0.0, "mean_steps": 0.0}
    total_return = 0.0
    total_steps = 0.0
    for row in rows:
        total_return += float(row.get("return", row.get("episode_return", 0.0)))
        total_steps += float(row.get("num_steps", row.get("steps", 0.0)))
    return {
        "episodes": len(rows),
        "mean_return": total_return / max(1, len(rows)),
        "mean_steps": total_steps / max(1, len(rows)),
    }
