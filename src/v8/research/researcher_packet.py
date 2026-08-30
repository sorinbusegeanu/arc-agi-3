from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .chain_audit import audit_chain
from .contracts import ChainStatus
from .default_analysis import derive_chain_evidence


PACKET_NAME = "LLM_RESEARCH_PACKET.md"
_MAX_LOG_LINES = 80
_STALE_HANDOFF_OUTPUTS = (
    "research_summary.md",
    "causal_chain_report.json",
    "runtime_hypotheses.json",
    "next_experiment.json",
    "latest_evidence_package.json",
    "evidence_packages",
)


def _git_revision() -> str:
    env = os.environ.get("GIT_COMMIT") or os.environ.get("GITHUB_SHA")
    if env:
        return str(env)
    try:
        run = subprocess.run(
            ("git", "rev-parse", "HEAD"), capture_output=True, text=True,
            timeout=2.0, check=False,
        )
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


def _mix_requested(argv: Sequence[str]) -> bool:
    values = tuple(str(value) for value in argv)
    for index, value in enumerate(values[:-1]):
        if value == "--games" and values[index + 1].strip().lower() == "mix":
            return True
    return False


def _chain_payload(summary: Mapping[str, Any], *, mix_requested: bool = False) -> dict[str, Any]:
    result = audit_chain(derive_chain_evidence(summary))
    first_unresolved = next(
        (edge.edge for edge in result.edges if edge.status != ChainStatus.PASS), None
    )
    payload = {
        "complete": result.complete,
        "first_broken_link": result.first_broken_link,
        "first_unresolved_link": first_unresolved,
        "edges": [
            {
                "edge": edge.edge,
                "status": edge.status.value,
                "evidence_count": edge.evidence_count,
                "blocker": edge.blocker,
            }
            for edge in result.edges
        ],
    }
    if mix_requested:
        payload["automatic_transfer_scope"] = "ARC-only subset of mix"
        payload["scope_warning"] = (
            "Automatic transfer interventions do not test Gym, Chess, or Sudoku. "
            "Do not interpret them as cross-family causal transfer evidence."
        )
    return payload


def _evidence_digest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False, "record_count": 0}
    kinds: Counter[str] = Counter()
    interventions: Counter[str] = Counter()
    effects: Counter[str] = Counter()
    sources: set[int] = set()
    targets: set[int] = set()
    records = 0
    invalid = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    invalid += 1
                    continue
                if not isinstance(row, dict):
                    invalid += 1
                    continue
                records += 1
                kinds[str(row.get("evidence_kind", "unknown"))] += 1
                intervention = str(row.get("causal_intervention", "") or "")
                if intervention:
                    interventions[intervention] += 1
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
        "evidence_kind_counts": dict(sorted(kinds.items())),
        "causal_intervention_counts": dict(sorted(interventions.items())),
        "effect_direction_counts": dict(sorted(effects.items())),
        "distinct_source_games": len(sources),
        "distinct_target_games": len(targets),
        "scope_note": "Counts cover the complete persisted ledger and may include restored historical evidence.",
    }


def _hypothesis_summary(h_report: Any) -> list[dict[str, Any]]:
    if not isinstance(h_report, list):
        return []
    result: list[dict[str, Any]] = []
    for row in h_report:
        if not isinstance(row, Mapping):
            continue
        result.append({
            "hypothesis_id": row.get("hypothesis_id"),
            "decision": row.get("final_decision", row.get("raw_decision")),
            "evidence_count": row.get("evidence_count", 0),
            "blocker": row.get("blocker", ""),
        })
    return result


def _game_outcomes(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    aggregates: dict[str, dict[str, int]] = {}
    actors = summary.get("actors", [])
    if not isinstance(actors, list):
        return []
    for actor in actors:
        if not isinstance(actor, Mapping):
            continue
        game = str(actor.get("game_id", ""))
        if not game:
            continue
        row = aggregates.setdefault(game, {
            "steps": 0,
            "wins": 0,
            "failures": 0,
            "levels_completed": 0,
            "resets": 0,
        })
        for key in tuple(row):
            row[key] += int(actor.get(key, 0) or 0)
    return [{"game_id": game, **aggregates[game]} for game in sorted(aggregates)]


def _key_run_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    metrics = summary.get("metrics", {})
    if not isinstance(metrics, Mapping):
        metrics = {}
    adaptive = metrics.get("adaptive_learning", {})
    adaptive_summary: dict[str, Any] = {}
    if isinstance(adaptive, Mapping):
        adaptive_summary = {
            "states": adaptive.get("states", {}),
            "sample_steps": adaptive.get("sample_steps", 0),
        }
    return {
        "games": list(summary.get("games", [])) if isinstance(summary.get("games", []), list) else [],
        "current_run_outcomes_by_game": _game_outcomes(summary),
        "automatic_transfer_experiments": summary.get("automatic_transfer_experiments", {}),
        "memory": {
            "watermark": metrics.get("watermark"),
            "memories": metrics.get("memories"),
            "edges": metrics.get("edges"),
            "evidence_records": metrics.get("evidence_records"),
            "level_counts": metrics.get("level_counts", {}),
        },
        "verified_success": metrics.get("verified_success", {}),
        "trajectory_optimizer": metrics.get("trajectory_optimizer", {}),
        "adaptive_learning": adaptive_summary,
    }


def _log_tail(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-_MAX_LOG_LINES:])


def _dump(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def build_packet(
    summary: Mapping[str, Any], *, revision: str, argv: Sequence[str],
    h_report: Any, reporting_cut: Any, evidence_digest: Mapping[str, Any], log_tail: str,
) -> str:
    del reporting_cut
    command = "PYTHONPATH=src python -m v8 " + " ".join(str(x) for x in argv)
    mixed = _mix_requested(argv)
    scope_note = (
        "- Automatic transfer intervention scope: `ARC-only subset of mix`\n"
        "- Important: Gym, Chess, and Sudoku share the memory graph during sampling, but the current automatic transfer intervention harness does not causally test those families.\n"
        if mixed
        else "- Automatic transfer intervention scope: selected ARC games supported by the experiment harness.\n"
    )
    return f"""# ARC-AGI-3 LLM Research Packet

Upload this **single file** to a frontier LLM researcher after the run.
The local system prepared evidence and conservative diagnostics only. It has **not selected a research hypothesis or next experiment**.

## Researcher task

1. Separate observations from explanations.
2. Classify the limitation: `BUG`, `PERFORMANCE_BOTTLENECK`, `EXPERIMENTAL_ARTIFACT`, `MECHANISM_FAILURE`, `ARCHITECTURAL_LIMITATION`, or `UNKNOWN`.
3. Identify the earliest causal link actually supported as broken or unresolved. Never turn missing evidence into failure.
4. Produce at least two competing runtime hypotheses.
5. Cite evidence for and against each hypothesis from this packet.
6. Select the smallest discriminating experiment that can distinguish the leading explanations. Prefer telemetry, existing ablations, or localized interventions over redesign.
7. Preregister predicted metric changes, expected unchanged metrics, and a falsifying result before the experiment is run.
8. Recommend architecture change only after bugs, artifacts, and existing-mechanism explanations are reasonably excluded.
9. If evidence cannot discriminate explanations, state `INSUFFICIENT_EVIDENCE` and specify what evidence is missing.
10. Treat cumulative restored evidence and current-run actor outcomes as different evidence scopes; do not infer that historical evidence was produced by this run.

Return exactly these sections:

- Observations
- Failure classification
- Causal bottleneck
- Competing hypotheses
- Evidence for/against each
- Most informative next experiment
- Preregistered predictions
- Falsifier
- Architecture change required: YES/NO
- What to inspect after the next run

## Run metadata

- Revision: `{revision}`
- Command: `{command}`
{scope_note}
## Deterministic causal-chain diagnostic

`INSUFFICIENT_EVIDENCE` means the local system refused to infer a causal failure.

```json
{_dump(_chain_payload(summary, mix_requested=mixed))}
```

## H01-H15 compact status

```json
{_dump(_hypothesis_summary(h_report))}
```

## Key current-run evidence

```json
{_dump(_key_run_summary(summary))}
```

## Evidence-ledger aggregate digest

Counts cover the complete persisted ledger. Individual evidence rows are intentionally omitted because they add volume without materially improving discrimination at handoff time.

```json
{_dump(dict(evidence_digest))}
```

## Recent runtime log

Last {_MAX_LOG_LINES} lines when available.

```text
{log_tail}
```

## Research discipline

A bug fix restores intended implementation and does not by itself validate a cognitive hypothesis. Correlation may generate a hypothesis but does not establish causality. Prefer matched interventions and ablations. The objective is to determine where useful information stops propagating through `experience -> memory -> abstraction -> transfer/retrieval -> action -> outcome`, then choose the experiment that most reduces uncertainty.
"""


def _remove_stale_handoff_outputs(research_root: Path) -> None:
    for name in _STALE_HANDOFF_OUTPUTS:
        path = research_root / name
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def write_researcher_packet(root: str | Path, *, argv: Sequence[str]) -> Path:
    root = Path(root)
    summary = _read_json(root / "v8_run_summary.json", None)
    if not isinstance(summary, Mapping):
        raise FileNotFoundError(f"missing or invalid run summary: {root / 'v8_run_summary.json'}")
    packet = build_packet(
        summary,
        revision=_git_revision(),
        argv=argv,
        h_report=_read_json(root / "reports" / "h01_h15.json", []),
        reporting_cut=_read_json(root / "reports" / "reporting_cut.json", {}),
        evidence_digest=_evidence_digest(root / "evidence" / "v8_evidence.jsonl"),
        log_tail=_log_tail(root / "log.txt"),
    )
    research_root = root / "research"
    research_root.mkdir(parents=True, exist_ok=True)
    _remove_stale_handoff_outputs(research_root)
    target = research_root / PACKET_NAME
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(packet, encoding="utf-8")
    os.replace(temp, target)
    return target
