from __future__ import annotations

import json
import os
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


EVIDENCE_NAME = "EXPERIMENT_EVIDENCE.md"
DECISION_NAME = "RESEARCH_DECISION.md"
_BOUNDARY_NAME = ".experiment_start.json"
_DECISION_BEGIN = "<!-- RESEARCH_DECISION_METADATA_BEGIN -->"
_DECISION_END = "<!-- RESEARCH_DECISION_METADATA_END -->"


def _git_revision() -> str:
    env = os.environ.get("GIT_COMMIT") or os.environ.get("GITHUB_SHA")
    if env:
        return str(env)
    try:
        run = subprocess.run(("git", "rev-parse", "HEAD"), capture_output=True, text=True, timeout=2.0, check=False)
        if run.returncode == 0 and run.stdout.strip():
            return run.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(temp, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(value, encoding="utf-8")
    os.replace(temp, path)


def _command(argv: Sequence[str]) -> str:
    return "PYTHONPATH=src python -m v8 " + " ".join(str(value) for value in argv)


def _extract_decision_metadata(text: str) -> dict[str, Any]:
    if not text:
        return {}
    start = text.find(_DECISION_BEGIN)
    stop = text.find(_DECISION_END)
    if start < 0 or stop <= start:
        return {}
    raw = text[start + len(_DECISION_BEGIN):stop].strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"parse_error": "invalid JSON between decision metadata markers"}
    return dict(value) if isinstance(value, Mapping) else {"parse_error": "decision metadata must be a JSON object"}


def _summary_state(summary: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(summary, Mapping):
        return {
            "available": False,
            "watermark": 0,
            "memories": 0,
            "edges": 0,
            "evidence_records": 0,
            "level_counts": {},
            "formation_telemetry": {},
            "verified_success": {},
            "trajectory_optimizer": {},
        }
    metrics = summary.get("metrics", {})
    if not isinstance(metrics, Mapping):
        metrics = {}
    return {
        "available": True,
        "watermark": int(metrics.get("watermark", 0) or 0),
        "memories": int(metrics.get("memories", 0) or 0),
        "edges": int(metrics.get("edges", 0) or 0),
        "evidence_records": int(metrics.get("evidence_records", 0) or 0),
        "level_counts": dict(metrics.get("level_counts", {}) or {}),
        "formation_telemetry": dict(metrics.get("formation_telemetry", {}) or {}),
        "verified_success": dict(metrics.get("verified_success", {}) or {}),
        "trajectory_optimizer": dict(metrics.get("trajectory_optimizer", {}) or {}),
    }


def _evidence_digest(path: Path, *, start_offset: int = 0) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False, "record_count": 0, "start_offset": int(start_offset)}
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    if start_offset < 0 or start_offset > size:
        return {
            "available": False,
            "record_count": 0,
            "start_offset": int(start_offset),
            "end_offset": int(size),
            "scope_error": "ledger was truncated or rewritten; experiment-local append slice is unavailable",
        }
    kinds: Counter[str] = Counter()
    interventions: Counter[str] = Counter()
    effects: Counter[str] = Counter()
    hypotheses: Counter[str] = Counter()
    sources: set[int] = set()
    targets: set[int] = set()
    records = 0
    invalid = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            handle.seek(int(start_offset))
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    invalid += 1
                    continue
                if not isinstance(row, Mapping):
                    invalid += 1
                    continue
                records += 1
                kinds[str(row.get("evidence_kind", "unknown"))] += 1
                intervention = str(row.get("causal_intervention", "") or "")
                if intervention:
                    interventions[intervention] += 1
                hypothesis = str(row.get("hypothesis_id", "") or "")
                if hypothesis:
                    hypotheses[hypothesis] += 1
                effect = int(row.get("effect_direction", 0) or 0)
                effects["positive" if effect > 0 else "negative" if effect < 0 else "neutral"] += 1
                source = int(row.get("source_game_hash", 0) or 0)
                target = int(row.get("target_game_hash", 0) or 0)
                if source:
                    sources.add(source)
                if target:
                    targets.add(target)
    except OSError as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "available": True,
        "record_count": records,
        "invalid_lines": invalid,
        "start_offset": int(start_offset),
        "end_offset": int(size),
        "evidence_kind_counts": dict(sorted(kinds.items())),
        "causal_intervention_counts": dict(sorted(interventions.items())),
        "effect_direction_counts": dict(sorted(effects.items())),
        "hypothesis_id_counts": dict(sorted(hypotheses.items())),
        "distinct_source_games": len(sources),
        "distinct_target_games": len(targets),
    }


def _game_outcomes(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: dict[str, dict[str, int]] = {}
    actors = summary.get("actors", [])
    if not isinstance(actors, list):
        return []
    for actor in actors:
        if not isinstance(actor, Mapping):
            continue
        game = str(actor.get("game_id", ""))
        if not game:
            continue
        row = result.setdefault(game, {"steps": 0, "wins": 0, "failures": 0, "levels_completed": 0, "resets": 0})
        for key in tuple(row):
            row[key] += int(actor.get(key, 0) or 0)
    return [{"game_id": game, **result[game]} for game in sorted(result)]


def _numeric_delta(start: Any, end: Any) -> Any:
    if isinstance(start, Mapping) and isinstance(end, Mapping):
        keys = sorted(set(start) | set(end), key=str)
        return {str(key): _numeric_delta(start.get(key, 0), end.get(key, 0)) for key in keys}
    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
        return end - start
    return None


def _state_delta(start: Mapping[str, Any], end: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "watermark": int(end.get("watermark", 0)) - int(start.get("watermark", 0)),
        "memories": int(end.get("memories", 0)) - int(start.get("memories", 0)),
        "edges": int(end.get("edges", 0)) - int(start.get("edges", 0)),
        "evidence_records": int(end.get("evidence_records", 0)) - int(start.get("evidence_records", 0)),
        "level_counts": _numeric_delta(start.get("level_counts", {}), end.get("level_counts", {})),
    }


def _h_summary(path: Path) -> list[dict[str, Any]]:
    rows = _read_json(path, [])
    if not isinstance(rows, list):
        return []
    result = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        result.append({
            "hypothesis_id": row.get("hypothesis_id"),
            "decision": row.get("final_decision", row.get("raw_decision")),
            "evidence_count": row.get("evidence_count", 0),
            "blocker": row.get("blocker", ""),
        })
    return result


def _previous_experiment_id(text: str) -> str | None:
    match = re.search(r"^- experiment_id: `([^`]+)`", text or "", re.MULTILINE)
    return match.group(1) if match else None


def capture_experiment_start(root: str | Path, *, argv: Sequence[str]) -> Path:
    root = Path(root)
    research_root = root / "research"
    research_root.mkdir(parents=True, exist_ok=True)
    evidence_path = research_root / EVIDENCE_NAME
    previous_evidence = ""
    try:
        previous_evidence = evidence_path.read_text(encoding="utf-8")
    except OSError:
        pass
    decision_text = ""
    try:
        decision_text = (research_root / DECISION_NAME).read_text(encoding="utf-8")
    except OSError:
        pass
    ledger = root / "evidence" / "v8_evidence.jsonl"
    try:
        ledger_offset = ledger.stat().st_size
    except OSError:
        ledger_offset = 0
    started_ns = time.time_ns()
    revision = _git_revision()
    boundary = {
        "experiment_id": f"exp-{started_ns}-{revision[:8]}",
        "parent_experiment_id": _previous_experiment_id(previous_evidence),
        "revision": revision,
        "command": _command(argv),
        "argv": [str(value) for value in argv],
        "started_ns": started_ns,
        "start_state": _summary_state(_read_json(root / "v8_run_summary.json", None)),
        "start_state_source": "previous durable v8_run_summary.json; memory is intentionally reused across runs",
        "evidence_ledger_start_offset": int(ledger_offset),
        "decision_metadata": _extract_decision_metadata(decision_text),
        "decision_text": decision_text,
    }
    path = research_root / _BOUNDARY_NAME
    _atomic_json(path, boundary)
    return path


def build_experiment_evidence(
    summary: Mapping[str, Any], *, boundary: Mapping[str, Any], root: Path, exit_code: int,
) -> str:
    end_state = _summary_state(summary)
    start_state = dict(boundary.get("start_state", {}) or {})
    ledger_path = root / "evidence" / "v8_evidence.jsonl"
    local_digest = _evidence_digest(ledger_path, start_offset=int(boundary.get("evidence_ledger_start_offset", 0) or 0))
    cumulative_digest = _evidence_digest(ledger_path, start_offset=0)
    decision_metadata = dict(boundary.get("decision_metadata", {}) or {})
    decision_status = "DECLARED" if decision_metadata and "parse_error" not in decision_metadata else "UNDECLARED_OR_INVALID"
    automatic_transfer = summary.get("automatic_transfer_experiments", {})
    metrics = summary.get("metrics", {}) if isinstance(summary.get("metrics", {}), Mapping) else {}
    formation = metrics.get("formation_telemetry", {}) if isinstance(metrics.get("formation_telemetry", {}), Mapping) else {}
    command = str(boundary.get("command", ""))
    finished_ns = time.time_ns()
    return f"""# EXPERIMENT_EVIDENCE

This file is factual experiment evidence. It must not prescribe the next architecture or mechanism change.
Persistent memory is intentional: causal interpretation must use experiment-local evidence and start/end deltas, not cumulative totals alone.

## Experiment identity

- experiment_id: `{boundary.get('experiment_id')}`
- parent_experiment_id: `{boundary.get('parent_experiment_id')}`
- revision: `{boundary.get('revision')}`
- command: `{command}`
- started_ns: `{boundary.get('started_ns')}`
- finished_ns: `{finished_ns}`
- exit_code: `{int(exit_code)}`
- memory_policy: `REUSE`

## Applied intervention declaration

- decision_status: `{decision_status}`
- The authoritative declaration is the `RESEARCH_DECISION.md` metadata captured before this run.

```json
{json.dumps(decision_metadata, indent=2, sort_keys=True, default=str)}
```

## Start state

Source: {boundary.get('start_state_source', '')}

```json
{json.dumps(start_state, indent=2, sort_keys=True, default=str)}
```

## Experiment-local activity

```json
{json.dumps({'games': summary.get('games', []), 'outcomes_by_game': _game_outcomes(summary), 'automatic_transfer_experiments': automatic_transfer}, indent=2, sort_keys=True, default=str)}
```

## End state

```json
{json.dumps(end_state, indent=2, sort_keys=True, default=str)}
```

## Experiment deltas

```json
{json.dumps(_state_delta(start_state, end_state), indent=2, sort_keys=True, default=str)}
```

## Formation causal funnel

These are the production M1N→M2 and M3-carrier→role gate counters from the end-of-run runtime metrics. Rejection examples are bounded by the telemetry layer.

```json
{json.dumps(dict(formation), indent=2, sort_keys=True, default=str)}
```

## Experiment-local evidence ledger

Only evidence appended after the pre-run ledger byte boundary is counted here. If the ledger was truncated/rewritten, this section explicitly reports that the local slice is unavailable.

```json
{json.dumps(local_digest, indent=2, sort_keys=True, default=str)}
```

## H01-H15 status at end of experiment

These decisions are cumulative status at the end of the run. They are not automatically attributed to this experiment. Use the experiment-local evidence ledger above to establish what this run actually added.

```json
{json.dumps(_h_summary(root / 'reports' / 'h01_h15.json'), indent=2, sort_keys=True, default=str)}
```

## Integrity and possible confounders

```json
{json.dumps({'reporting_cut': _read_json(root / 'reports' / 'reporting_cut.json', {}), 'trajectory_optimizer': metrics.get('trajectory_optimizer', {}), 'adaptive_learning': metrics.get('adaptive_learning', {}), 'experiment_local_ledger_available': bool(local_digest.get('available')), 'decision_status': decision_status}, indent=2, sort_keys=True, default=str)}
```

## Cumulative state — context only

Do not use this section as experiment-local causal evidence.

```json
{json.dumps({'memory': end_state, 'evidence_ledger': cumulative_digest}, indent=2, sort_keys=True, default=str)}
```
"""


def write_experiment_evidence(root: str | Path, *, exit_code: int) -> Path:
    root = Path(root)
    boundary = _read_json(root / "research" / _BOUNDARY_NAME, None)
    if not isinstance(boundary, Mapping):
        raise FileNotFoundError(f"missing experiment start boundary: {root / 'research' / _BOUNDARY_NAME}")
    summary = _read_json(root / "v8_run_summary.json", None)
    if not isinstance(summary, Mapping):
        raise FileNotFoundError(f"missing or invalid run summary: {root / 'v8_run_summary.json'}")
    target = root / "research" / EVIDENCE_NAME
    _atomic_text(target, build_experiment_evidence(summary, boundary=boundary, root=root, exit_code=exit_code))
    try:
        (root / "research" / _BOUNDARY_NAME).unlink()
    except FileNotFoundError:
        pass
    return target
