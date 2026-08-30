from __future__ import annotations

import json
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .chain_audit import audit_chain
from .contracts import ChainStatus
from .default_analysis import derive_chain_evidence


PACKET_NAME = "LLM_RESEARCH_PACKET.md"
_MAX_EVIDENCE_SAMPLES = 96
_MAX_SAMPLES_PER_KIND = 3
_MAX_LOG_LINES = 200


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


def _chain_payload(summary: Mapping[str, Any]) -> dict[str, Any]:
    result = audit_chain(derive_chain_evidence(summary))
    first_unresolved = next(
        (edge.edge for edge in result.edges if edge.status != ChainStatus.PASS), None
    )
    return {
        "complete": result.complete,
        "first_broken_link": result.first_broken_link,
        "first_unresolved_link": first_unresolved,
        "edges": [
            {
                "edge": edge.edge,
                "status": edge.status.value,
                "evidence_count": edge.evidence_count,
                "evidence_ids": list(edge.evidence_ids),
                "blocker": edge.blocker,
            }
            for edge in result.edges
        ],
    }


def _evidence_digest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False, "record_count": 0, "samples": []}
    kinds: Counter[str] = Counter()
    interventions: Counter[str] = Counter()
    effects: Counter[str] = Counter()
    sources: set[int] = set()
    targets: set[int] = set()
    samples_per_kind: defaultdict[str, int] = defaultdict(int)
    samples: list[dict[str, Any]] = []
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
                kind = str(row.get("evidence_kind", "unknown"))
                kinds[kind] += 1
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
                if len(samples) < _MAX_EVIDENCE_SAMPLES and samples_per_kind[kind] < _MAX_SAMPLES_PER_KIND:
                    samples.append(row)
                    samples_per_kind[kind] += 1
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
        "samples": samples,
        "sampling_note": f"Counts cover the full ledger; samples are capped at {_MAX_SAMPLES_PER_KIND} per kind and {_MAX_EVIDENCE_SAMPLES} total.",
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
    command = "PYTHONPATH=src python -m v8 " + " ".join(str(x) for x in argv)
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

## Deterministic causal-chain diagnostic

`INSUFFICIENT_EVIDENCE` means the local system refused to infer a causal failure.

```json
{_dump(_chain_payload(summary))}
```

## H01-H15 scientific report

```json
{_dump(h_report)}
```

## Reporting cut

```json
{_dump(reporting_cut)}
```

## Full run summary

```json
{_dump(summary)}
```

## Evidence-ledger digest

Counts cover the complete persisted ledger; representative raw rows are bounded to keep this file practical to upload.

```json
{_dump(evidence_digest)}
```

## Recent runtime log

Last {_MAX_LOG_LINES} lines when available.

```text
{log_tail}
```

## Research discipline

A bug fix restores intended implementation and does not by itself validate a cognitive hypothesis. Correlation may generate a hypothesis but does not establish causality. Prefer matched interventions and ablations. The objective is to determine where useful information stops propagating through `experience -> memory -> abstraction -> transfer/retrieval -> action -> outcome`, then choose the experiment that most reduces uncertainty.
"""


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
    target = research_root / PACKET_NAME
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(packet, encoding="utf-8")
    os.replace(temp, target)
    return target
