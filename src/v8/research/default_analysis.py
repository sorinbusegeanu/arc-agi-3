from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from .chain_audit import audit_chain
from .contracts import ChainStatus, Confidence, FailureCategory, FailureLevel
from .evidence_package import build_evidence_package, write_evidence_package
from .models import ChainEdgeEvidence, RuntimeHypothesis
from .store import ResearchStore


_HYPOTHESES = {
    "M1_FORMATION": (
        "Low-level memory formation is not preserving accepted interaction evidence.",
        "Accepted interactions are not reaching the M0/M1 memory path.",
    ),
    "M2_ABSTRACTION": (
        "Lower-level memories exist but are not producing reusable M2 abstractions.",
        "M2 admission gates lack sufficient support rather than the abstraction mechanism being broken.",
    ),
    "M3_ROLE_FORMATION": (
        "M2 structure exists but relational role induction is not producing usable M3 roles.",
        "The current games do not yet contain enough repeated relational structure to justify M3 formation.",
    ),
    "M4_RELEVANT_CANDIDATE": (
        "Higher-order structures exist but no M4 candidate is sufficiently relevant for held-out transfer testing.",
        "M4 candidates exist but transfer-test scheduling is the limiting step.",
    ),
    "CROSS_WORLD_RETRIEVAL": (
        "Transferable memory exists but cross-world retrieval/reuse is not demonstrated.",
        "The limitation is structural correspondence or transfer validation rather than retrieval itself.",
    ),
    "ACTION_INTEGRATION": (
        "Higher-order strategy memory exists but does not materially influence action selection.",
        "Planning opportunities are too sparse for current action-integration telemetry to establish use.",
    ),
    "BEHAVIORAL_IMPROVEMENT": (
        "Memory mechanisms operate but do not yet produce a positive matched behavioral effect.",
        "A useful effect exists but current controlled experiments are underpowered or target the wrong games.",
    ),
}


_EXPERIMENTS = {
    "M1_FORMATION": {
        "purpose": "DIAGNOSTIC",
        "control": "accepted actor events",
        "treatment": "M0/M1 admission trace",
        "prediction": "Every accepted interaction should produce auditable low-level memory evidence.",
        "falsifier": "Accepted events are present and M1 evidence is formed normally.",
    },
    "M2_ABSTRACTION": {
        "purpose": "DIAGNOSTIC",
        "control": "current M1 population",
        "treatment": "trace M1N -> M2 candidate/gate decisions",
        "prediction": "Supported recurrent M1N structures should reach M2 candidate generation.",
        "falsifier": "M2 candidates are generated and rejected for preregistered evidence reasons.",
    },
    "M3_ROLE_FORMATION": {
        "purpose": "DIAGNOSTIC",
        "control": "current M2 population",
        "treatment": "trace M2 -> M3 relational-role candidate/gate decisions",
        "prediction": "Repeated compatible M2 structures should produce auditable M3 role candidates.",
        "falsifier": "M3 candidates are generated and rejected because relational evidence is insufficient.",
    },
    "M4_RELEVANT_CANDIDATE": {
        "purpose": "DISCRIMINATION",
        "control": "current concept/transfer scheduling",
        "treatment": "targeted audit of M4 candidate -> transfer-test eligibility",
        "prediction": "At least one supported higher-order structure should become an admissible held-out transfer candidate.",
        "falsifier": "All candidates fail explicit structural or provenance gates.",
    },
    "CROSS_WORLD_RETRIEVAL": {
        "purpose": "ABLATION",
        "control": "normal",
        "treatment": "transfer_off",
        "prediction": "Normal memory should outperform transfer_off on matched held-out targets if cross-world reuse is useful.",
        "falsifier": "Matched target behavior is unchanged when transfer is disabled.",
    },
    "ACTION_INTEGRATION": {
        "purpose": "DISCRIMINATION",
        "control": "normal",
        "treatment": "concepts_off / transfer_off matched control",
        "prediction": "Removing higher-order memory should change selected actions where the normal run reports planning use.",
        "falsifier": "Action choices remain unchanged despite available higher-order strategies.",
    },
    "BEHAVIORAL_IMPROVEMENT": {
        "purpose": "ABLATION",
        "control": "memory_off",
        "treatment": "normal",
        "prediction": "Normal learned memory should improve matched held-out behavior over memory_off.",
        "falsifier": "Normal does not outperform memory_off under matched seeds and interaction budgets.",
    },
}


def _git_revision() -> str:
    env = os.environ.get("GIT_COMMIT") or os.environ.get("GITHUB_SHA")
    if env:
        return str(env)
    try:
        completed = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        value = completed.stdout.strip()
        if completed.returncode == 0 and value:
            return value
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _level_count(metrics: Mapping[str, Any], level: int) -> int:
    counts = metrics.get("level_counts", {})
    if isinstance(counts, Mapping):
        for key in (level, str(level), f"M{level}"):
            if key in counts:
                try:
                    return max(0, int(counts[key]))
                except (TypeError, ValueError):
                    return 0
        return 0
    if isinstance(counts, (list, tuple)) and 0 <= level < len(counts):
        try:
            return max(0, int(counts[level]))
        except (TypeError, ValueError):
            return 0
    return 0


def _actor_totals(summary: Mapping[str, Any]) -> dict[str, int]:
    actors = summary.get("actors", ())
    if not isinstance(actors, (list, tuple)):
        actors = ()
    fields = ("steps", "wins", "failures", "levels_completed", "replans", "planned_steps")
    totals = {name: 0 for name in fields}
    for actor in actors:
        if not isinstance(actor, Mapping):
            continue
        for name in fields:
            try:
                totals[name] += max(0, int(actor.get(name, 0)))
            except (TypeError, ValueError):
                pass
    totals["actor_count"] = len(actors)
    totals["games_with_wins"] = len(
        {
            str(actor.get("game_id"))
            for actor in actors
            if isinstance(actor, Mapping) and int(actor.get("wins", 0) or 0) > 0
        }
    )
    return totals


def derive_chain_evidence(summary: Mapping[str, Any]) -> dict[str, ChainEdgeEvidence]:
    metrics = summary.get("metrics", {})
    if not isinstance(metrics, Mapping):
        metrics = {}
    totals = _actor_totals(summary)
    transfer = summary.get("automatic_transfer_experiments", {})
    if not isinstance(transfer, Mapping):
        transfer = {}

    attempted = max(0, int(transfer.get("attempted", 0) or 0))
    completed = max(0, int(transfer.get("completed", 0) or 0))
    passed = max(0, int(transfer.get("passed", 0) or 0))

    m1 = _level_count(metrics, 1)
    m2 = _level_count(metrics, 2)
    m3 = _level_count(metrics, 3)
    m4 = _level_count(metrics, 4)
    m7 = _level_count(metrics, 7)

    evidence: dict[str, ChainEdgeEvidence] = {}

    if m1 > 0:
        evidence["M1_FORMATION"] = ChainEdgeEvidence(
            "M1_FORMATION", ChainStatus.PASS, m1, ("metrics.level_counts.M1",)
        )
    elif totals["steps"] > 0:
        evidence["M1_FORMATION"] = ChainEdgeEvidence(
            "M1_FORMATION",
            ChainStatus.FAIL,
            0,
            ("v8_run_summary.actors.steps",),
            "actor interactions were executed but no M1 memory is present",
        )

    if m2 > 0:
        evidence["M2_ABSTRACTION"] = ChainEdgeEvidence(
            "M2_ABSTRACTION", ChainStatus.PASS, m2, ("metrics.level_counts.M2",)
        )
    elif m1 > 0:
        evidence["M2_ABSTRACTION"] = ChainEdgeEvidence(
            "M2_ABSTRACTION",
            ChainStatus.INSUFFICIENT_EVIDENCE,
            0,
            ("metrics.level_counts.M1",),
            "lower-level memory exists but this run does not establish why no M2 abstraction is present",
        )

    if m3 > 0:
        evidence["M3_ROLE_FORMATION"] = ChainEdgeEvidence(
            "M3_ROLE_FORMATION", ChainStatus.PASS, m3, ("metrics.level_counts.M3",)
        )
    elif m2 > 0:
        evidence["M3_ROLE_FORMATION"] = ChainEdgeEvidence(
            "M3_ROLE_FORMATION",
            ChainStatus.INSUFFICIENT_EVIDENCE,
            0,
            ("metrics.level_counts.M2",),
            "M2 structure exists but this run does not establish whether M3 evidence was sufficient",
        )

    if attempted > 0:
        evidence["M4_RELEVANT_CANDIDATE"] = ChainEdgeEvidence(
            "M4_RELEVANT_CANDIDATE",
            ChainStatus.PASS,
            attempted,
            ("automatic_transfer_experiments.attempted",),
        )
    elif m4 > 0:
        evidence["M4_RELEVANT_CANDIDATE"] = ChainEdgeEvidence(
            "M4_RELEVANT_CANDIDATE",
            ChainStatus.INSUFFICIENT_EVIDENCE,
            m4,
            ("metrics.level_counts.M4",),
            "M4 memories exist but no admissible held-out candidate is demonstrated by this run summary",
        )

    if completed > 0:
        evidence["CROSS_WORLD_RETRIEVAL"] = ChainEdgeEvidence(
            "CROSS_WORLD_RETRIEVAL",
            ChainStatus.PASS,
            completed,
            ("automatic_transfer_experiments.completed",),
        )

    if totals["planned_steps"] > 0 or totals["replans"] > 0:
        evidence["ACTION_INTEGRATION"] = ChainEdgeEvidence(
            "ACTION_INTEGRATION",
            ChainStatus.PASS,
            totals["planned_steps"] + totals["replans"],
            ("actors.planned_steps", "actors.replans"),
        )
    elif m7 > 0 and totals["steps"] > 0:
        evidence["ACTION_INTEGRATION"] = ChainEdgeEvidence(
            "ACTION_INTEGRATION",
            ChainStatus.FAIL,
            m7,
            ("metrics.level_counts.M7", "actors.planned_steps"),
            "M7 strategy memory exists but actors report no planned steps or replanning",
        )

    if passed > 0:
        evidence["BEHAVIORAL_IMPROVEMENT"] = ChainEdgeEvidence(
            "BEHAVIORAL_IMPROVEMENT",
            ChainStatus.PASS,
            passed,
            ("automatic_transfer_experiments.passed",),
        )
    elif completed > 0:
        evidence["BEHAVIORAL_IMPROVEMENT"] = ChainEdgeEvidence(
            "BEHAVIORAL_IMPROVEMENT",
            ChainStatus.FAIL,
            completed,
            ("automatic_transfer_experiments.completed", "automatic_transfer_experiments.passed"),
            "controlled transfer experiments completed without a passing behavioral effect",
        )

    return evidence


def _first_unresolved(audit) -> str | None:
    for edge in audit.edges:
        if edge.status != ChainStatus.PASS:
            return edge.edge
    return None


def _hypotheses_for(edge: str | None, *, run_id: str) -> list[dict[str, Any]]:
    if edge is None:
        return []
    claims = _HYPOTHESES.get(
        edge,
        ("The current causal link is unresolved.", "A neighboring causal link is the true limitation."),
    )
    return [
        asdict(
            RuntimeHypothesis(
                hypothesis_id=f"RH-{run_id}-{edge}-A",
                claim=claims[0],
                target_chain_edge=edge,
                confidence=Confidence.MEDIUM,
                failure_level=FailureLevel.LOCAL,
                category=FailureCategory.MECHANISM_FAILURE,
            )
        ),
        asdict(
            RuntimeHypothesis(
                hypothesis_id=f"RH-{run_id}-{edge}-B",
                claim=claims[1],
                target_chain_edge=edge,
                confidence=Confidence.LOW,
                failure_level=FailureLevel.LOCAL,
                category=FailureCategory.UNKNOWN,
            )
        ),
    ]


def _jsonable(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def build_default_analysis(
    summary: Mapping[str, Any],
    *,
    revision: str | None = None,
) -> dict[str, Any]:
    metrics = summary.get("metrics", {})
    if not isinstance(metrics, Mapping):
        metrics = {}
    totals = _actor_totals(summary)
    games = [str(game) for game in summary.get("games", ())]

    watermark = max(0, int(metrics.get("watermark", 0) or 0))
    run_id = f"wm{watermark:012d}"
    revision = str(revision or _git_revision())

    chain = audit_chain(derive_chain_evidence(summary))
    focus = _first_unresolved(chain)
    hypotheses = _hypotheses_for(focus, run_id=run_id)
    experiment = dict(_EXPERIMENTS.get(focus, {})) if focus else {}
    if focus:
        experiment.update(
            {
                "experiment_id": f"EXP-{run_id}-{focus}",
                "target_chain_edge": focus,
                "games": games,
                "interaction_budget": totals["steps"],
                "preregistered": True,
            }
        )

    transfer = summary.get("automatic_transfer_experiments", {})
    if not isinstance(transfer, Mapping):
        transfer = {}

    outcomes = {
        "wins": totals["wins"],
        "failures": totals["failures"],
        "levels_completed": totals["levels_completed"],
        "games_with_wins": totals["games_with_wins"],
        "actor_count": totals["actor_count"],
        "win_rate_per_actor_step": totals["wins"] / max(1, totals["steps"]),
    }
    development = {
        "memories": int(metrics.get("memories", 0) or 0),
        "edges": int(metrics.get("edges", 0) or 0),
        "level_counts": _jsonable(metrics.get("level_counts", {})),
        "generation": int(metrics.get("generation", 0) or 0),
        "peers": _jsonable(metrics.get("peers")),
    }
    control = {
        "planned_steps": totals["planned_steps"],
        "replans": totals["replans"],
        "actor_steps": totals["steps"],
    }
    chain_payload = {
        "complete": chain.complete,
        "first_broken_link": chain.first_broken_link,
        "first_unresolved_link": focus,
        "edges": [
            {
                "edge": item.edge,
                "status": item.status.value,
                "evidence_count": item.evidence_count,
                "evidence_ids": list(item.evidence_ids),
                "blocker": item.blocker,
            }
            for item in chain.edges
        ],
    }
    package = build_evidence_package(
        run={
            "run_id": run_id,
            "revision": revision,
            "games": games,
            "steps": totals["steps"],
            "watermark": watermark,
        },
        outcomes=outcomes,
        development=development,
        transfer=_jsonable(dict(transfer)),
        control=control,
        chain_audit=chain_payload,
        hypotheses=_jsonable(dict(summary.get("hypotheses", {}) or {})),
    )
    package["runtime_hypotheses"] = _jsonable(hypotheses)
    package["next_experiment"] = _jsonable(experiment)
    return {
        "run_id": run_id,
        "package": package,
        "chain": chain_payload,
        "runtime_hypotheses": _jsonable(hypotheses),
        "next_experiment": _jsonable(experiment),
    }


def _render_report(analysis: Mapping[str, Any]) -> str:
    package = analysis["package"]
    chain = analysis["chain"]
    hypotheses = analysis["runtime_hypotheses"]
    experiment = analysis["next_experiment"]
    run = package["run"]
    outcomes = package["outcomes"]
    development = package["development"]

    lines = [
        "# Recursive Research Analysis",
        "",
        f"**Run:** `{run['run_id']}`  ",
        f"**Revision:** `{run['revision']}`  ",
        f"**Games:** {', '.join(run['games']) or 'none'}  ",
        f"**Actor steps:** {run['steps']}  ",
        "",
        "## Current diagnosis",
        "",
        f"- First explicit failed link: `{chain['first_broken_link'] or 'none'}`",
        f"- First unresolved causal link: `{chain['first_unresolved_link'] or 'none'}`",
        f"- Chain complete: `{chain['complete']}`",
        "",
        "## Causal chain",
        "",
        "| Link | Status | Evidence | Blocker |",
        "|---|---|---:|---|",
    ]
    for item in chain["edges"]:
        lines.append(
            f"| {item['edge']} | {item['status']} | {item['evidence_count']} | {item['blocker'] or ''} |"
        )

    lines.extend(
        [
            "",
            "## Observed outcomes",
            "",
            f"- Wins: {outcomes['wins']}",
            f"- Levels completed: {outcomes['levels_completed']}",
            f"- Games with wins: {outcomes['games_with_wins']}",
            f"- Persistent memories: {development['memories']}",
            f"- Graph edges: {development['edges']}",
            "",
            "## Competing runtime hypotheses",
            "",
        ]
    )
    if hypotheses:
        for item in hypotheses:
            lines.append(
                f"- **{item['hypothesis_id']}** ({item['confidence']}): {item['claim']}"
            )
    else:
        lines.append("- No unresolved causal link in the current default chain.")

    lines.extend(["", "## Next discriminating experiment", ""])
    if experiment:
        lines.extend(
            [
                f"- Purpose: `{experiment.get('purpose', 'DIAGNOSTIC')}`",
                f"- Target: `{experiment.get('target_chain_edge', '')}`",
                f"- Control: {experiment.get('control', '')}",
                f"- Treatment: {experiment.get('treatment', '')}",
                f"- Prediction: {experiment.get('prediction', '')}",
                f"- Falsifier: {experiment.get('falsifier', '')}",
            ]
        )
    else:
        lines.append("- No experiment proposed because every default causal link passed.")

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "A PASS is emitted only from persisted evidence available in the completed run. "
            "Missing counterfactual evidence remains INSUFFICIENT_EVIDENCE. "
            "Game wins alone are not treated as proof that memory caused the improvement.",
            "",
        ]
    )
    return "\n".join(lines)


def run_default_research_analysis(root: str | Path) -> dict[str, Any]:
    root_path = Path(root)
    summary_path = root_path / "v8_run_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing completed run summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    analysis = build_default_analysis(summary)

    research_root = root_path / "research"
    run_root = research_root / "evidence_packages"
    run_path = run_root / f"{analysis['run_id']}.json"
    latest_path = research_root / "latest_evidence_package.json"
    write_evidence_package(run_path, analysis["package"])
    write_evidence_package(latest_path, analysis["package"])

    report_path = research_root / "research_summary.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(analysis), encoding="utf-8")
    (research_root / "causal_chain_report.json").write_text(
        json.dumps(analysis["chain"], indent=2, sort_keys=True), encoding="utf-8"
    )
    (research_root / "runtime_hypotheses.json").write_text(
        json.dumps(analysis["runtime_hypotheses"], indent=2, sort_keys=True), encoding="utf-8"
    )
    (research_root / "next_experiment.json").write_text(
        json.dumps(analysis["next_experiment"], indent=2, sort_keys=True), encoding="utf-8"
    )

    with ResearchStore(research_root / "research.db") as store:
        for item in analysis["runtime_hypotheses"]:
            store.upsert_hypothesis(
                RuntimeHypothesis(
                    hypothesis_id=str(item["hypothesis_id"]),
                    claim=str(item["claim"]),
                    target_chain_edge=str(item["target_chain_edge"]),
                    status=str(item["status"]),
                    confidence=Confidence(str(item["confidence"])),
                    failure_level=FailureLevel(str(item["failure_level"])),
                    category=FailureCategory(str(item["category"])),
                    parent_hypothesis_id=item.get("parent_hypothesis_id"),
                    scope=dict(item.get("scope") or {}),
                )
            )

    return {
        **analysis,
        "report_path": str(report_path),
        "evidence_path": str(run_path),
    }
