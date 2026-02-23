from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def iter_log_events(log_dir: str):
    for path in sorted(Path(log_dir).rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            yield path, json.loads(line)


def episode_outcomes(log_dir: str) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for _path, ev in iter_log_events(log_dir):
        eid = ev.get("episode_id")
        if not isinstance(eid, str):
            continue
        if ev.get("type") == "episode_end":
            out[eid] = bool(ev.get("won", False))
    return out


def grouped_steps(log_dir: str) -> dict[str, list[dict[str, object]]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for _path, ev in iter_log_events(log_dir):
        if ev.get("type") != "step":
            continue
        eid = ev.get("episode_id")
        if isinstance(eid, str):
            groups[eid].append(ev)
    for eid in groups:
        groups[eid].sort(key=lambda x: int(x.get("step_idx", 0)))
    return groups
