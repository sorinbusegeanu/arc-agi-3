from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from v7.memory.concept_validation import ConceptValidationStatus
from v7.memory.ids import MemoryId, MemoryLevel
from v7.memory.lifecycle import MemoryStatus

EVIDENCE_INTERACTION = 2001
EVIDENCE_TRAJECTORY = 2012

HYPOTHESIS_RISKS: dict[str, tuple[str, ...]] = {
    "H01": ("Support count alone can overstate stability when repeated evidence comes from one game/context.",),
    "H02": ("The prediction tracker is worker-local and resets for each sampling job; cross-epoch prediction learning is not yet represented.", "Replay can be triggered by signals other than prediction error, so direct per-memory linkage is required."),
    "H03": ("Transformation families are keyed by action/outcome signature; weak outcome signatures can merge unrelated transformations.",),
    "H04": ("v7 has no separate canonical carrier memory type; carrier evidence is inferred from M3 structural precursors, so H04 cannot be classified VALID yet."),
    "H05": ("M3 role identity contains context, which can fragment one functional role across contexts and suppress cross-context reuse evidence.",),
    "H06": ("Role transfer requires genuinely different source/target games; repeated trials in one game do not count.", "Action selection can bias which transfer opportunities are observed."),
    "H07": ("Concept validation requires retrospective transfer trials; newly created concepts cannot validate in the same epoch that creates them.",),
    "H08": ("World models require at least two transfer-validated concepts, creating a developmental lag of at least one additional derivation cycle.",),
    "H09": ("Future-option delta is currently measured as change in available action IDs, which may miss latent option changes not exposed by the environment API.",),
    "H10": ("Attention saturation can make high- and low-option-change populations indistinguishable.", "Replay status is memory-level while future-option evidence is interaction-level, so aggregation can blur causality."),
    "H11": ("A future-option-bearing ancestor does not by itself prove the future-option signal caused transfer; provenance establishes linkage, not causality.",),
    "H12": ("Trajectory comparisons are meaningful only across comparable successful episodes; sparse wins can leave this hypothesis underpowered.", "Different ARC levels can have very different intrinsic solution lengths and should not be pooled blindly."),
}


def build_hypothesis_rows(runtime, *, epoch: int) -> dict[str, dict[str, object]]:
    view = runtime.writer.published_view
    level_ids: dict[MemoryLevel, tuple[MemoryId, ...]] = {
        level: tuple(sorted((memory_id for memory_id, node in view.nodes.items() if node.level == level), key=int))
        for level in MemoryLevel
    }
    raw = _interaction_rows(runtime)
    transfer_rows = _transfer_rows(runtime)

    rows: dict[str, dict[str, object]] = {}

    def put(hid: str, decision: str, measurement: object, evidence_rows: int, metrics: dict[str, object], *, quality_gate: str = "PASS") -> None:
        rows[hid] = {
            "raw_decision": decision,
            "quality_gate": quality_gate,
            "evidence": {
                "evidence_rows": int(evidence_rows),
                "measurement": measurement,
                "metrics": metrics,
                "issues": list(HYPOTHESIS_RISKS[hid]),
            },
        }

    m1 = level_ids[MemoryLevel.M1]
    supported_m1 = tuple(mid for mid in m1 if int(view.nodes[mid].support_count) >= 2)
    source_games = {row.get("source_game") for row in raw if row.get("source_game")}
    h01_decision = "VALID" if supported_m1 and len(source_games) >= 2 else "PARTIALLY_VALID" if m1 else "INSUFFICIENT_EVIDENCE"
    put("H01", h01_decision, len(supported_m1), len(raw), {"m1": len(m1), "supported_m1": len(supported_m1), "source_games": len(source_games)})

    pe_rows = [row for row in raw if float(row.get("prediction_error") or 0.0) > 0.0]
    no_pe_rows = [row for row in raw if float(row.get("prediction_error") or 0.0) <= 0.0]
    pe_rate = _replay_rate(pe_rows, view)
    no_pe_rate = _replay_rate(no_pe_rows, view)
    if len(pe_rows) >= 5 and len(no_pe_rows) >= 5 and pe_rate is not None and no_pe_rate is not None:
        h02_decision = "VALID" if pe_rate > no_pe_rate else "INVALID"
    elif pe_rows:
        h02_decision = "PARTIALLY_VALID"
    else:
        h02_decision = "INSUFFICIENT_EVIDENCE"
    put("H02", h02_decision, None if pe_rate is None or no_pe_rate is None else pe_rate - no_pe_rate, len(pe_rows), {"prediction_violation_rows": len(pe_rows), "baseline_rows": len(no_pe_rows), "prediction_violation_replay_rate": pe_rate, "baseline_replay_rate": no_pe_rate})

    m2 = level_ids[MemoryLevel.M2]
    valid_families = [mid for mid in m2 if _parent_count(runtime, mid, level=MemoryLevel.M1, view=view) >= 2]
    h03_decision = "VALID" if valid_families else "PARTIALLY_VALID" if m2 else "INSUFFICIENT_EVIDENCE"
    put("H03", h03_decision, len(valid_families), len(m2), {"m2": len(m2), "families_with_2plus_m1_parents": len(valid_families)})

    m3 = level_ids[MemoryLevel.M3]
    carrier_precursors = [mid for mid in m3 if _parent_count(runtime, mid, level=MemoryLevel.M2, view=view) >= 1 and _parent_count(runtime, mid, level=MemoryLevel.M1, view=view) >= 1]
    h04_decision = "PARTIALLY_VALID" if carrier_precursors else "INSUFFICIENT_EVIDENCE"
    put("H04", h04_decision, len(carrier_precursors), len(m3), {"carrier_precursor_count": len(carrier_precursors), "explicit_carrier_type_available": False})

    supported_roles = [mid for mid in m3 if int(view.nodes[mid].support_count) >= 1 and _parent_count(runtime, mid, level=MemoryLevel.M2, view=view) >= 1]
    h05_decision = "VALID" if supported_roles else "INSUFFICIENT_EVIDENCE"
    put("H05", h05_decision, len(supported_roles), len(m3), {"m3": len(m3), "supported_roles": len(supported_roles)})

    role_trials = [row for row in transfer_rows if _node_level(view, row["memory_id"]) == MemoryLevel.M3]
    role_success = [row for row in role_trials if row["success"]]
    role_pairs = {(row["source_game"], row["target_game"]) for row in role_trials if row["source_game"] != row["target_game"]}
    if len(role_trials) >= 2 and role_pairs:
        h06_decision = "VALID" if role_success else "INVALID"
    elif role_trials:
        h06_decision = "PARTIALLY_VALID"
    else:
        h06_decision = "INSUFFICIENT_EVIDENCE"
    put("H06", h06_decision, _rate(len(role_success), len(role_trials)), len(role_trials), {"role_transfer_trials": len(role_trials), "successful_role_transfers": len(role_success), "distinct_game_pairs": len(role_pairs)})

    m4 = level_ids[MemoryLevel.M4]
    candidates = [mid for mid in m4 if int(view.nodes[mid].status_flags) & int(ConceptValidationStatus.CANDIDATE)]
    validated = [mid for mid in m4 if int(view.nodes[mid].status_flags) & int(ConceptValidationStatus.TRANSFER_VALIDATED)]
    h07_decision = "VALID" if validated else "PARTIALLY_VALID" if candidates or m4 else "INSUFFICIENT_EVIDENCE"
    put("H07", h07_decision, len(validated), len(m4), {"m4": len(m4), "concept_candidates": len(candidates), "transfer_validated_concepts": len(validated)})

    m5 = level_ids[MemoryLevel.M5]
    coherent_models = [mid for mid in m5 if _validated_concept_parent_count(runtime, mid, view) >= 2]
    h08_decision = "VALID" if coherent_models else "PARTIALLY_VALID" if m5 else "INSUFFICIENT_EVIDENCE"
    put("H08", h08_decision, len(coherent_models), len(m5), {"m5": len(m5), "models_with_2plus_validated_concepts": len(coherent_models)})

    fo_rows = [row for row in raw if float(row.get("future_option_delta") or 0.0) != 0.0]
    motifs: dict[tuple[int, int], set[str]] = {}
    motif_counts: dict[tuple[int, int], int] = {}
    for item in fo_rows:
        delta = float(item.get("future_option_delta") or 0.0)
        key = (int(item.get("action_id") or 0), 1 if delta > 0 else -1)
        motif_counts[key] = motif_counts.get(key, 0) + 1
        game = str(item.get("source_game") or "")
        if game:
            motifs.setdefault(key, set()).add(game)
    repeated_motifs = [key for key, count in motif_counts.items() if count >= 2 and len(motifs.get(key, set())) >= 2]
    h09_decision = "VALID" if repeated_motifs else "PARTIALLY_VALID" if fo_rows else "INSUFFICIENT_EVIDENCE"
    put("H09", h09_decision, len(repeated_motifs), len(fo_rows), {"future_option_events": len(fo_rows), "cross_game_repeated_motifs": len(repeated_motifs)})

    zero_fo_rows = [row for row in raw if float(row.get("future_option_delta") or 0.0) == 0.0]
    high_fo_replay = _replay_rate(fo_rows, view)
    low_fo_replay = _replay_rate(zero_fo_rows, view)
    lift = None if high_fo_replay is None or low_fo_replay is None or low_fo_replay <= 0 else high_fo_replay / low_fo_replay
    if len(fo_rows) >= 5 and len(zero_fo_rows) >= 5 and high_fo_replay is not None and low_fo_replay is not None:
        if high_fo_replay == low_fo_replay:
            h10_decision = "INSUFFICIENT_EVIDENCE"
        else:
            h10_decision = "VALID" if (lift is not None and lift > 1.25) or (low_fo_replay == 0 and high_fo_replay > 0) else "INVALID"
    elif fo_rows:
        h10_decision = "PARTIALLY_VALID"
    else:
        h10_decision = "INSUFFICIENT_EVIDENCE"
    put("H10", h10_decision, lift, len(fo_rows), {"high_option_change_rows": len(fo_rows), "zero_option_change_rows": len(zero_fo_rows), "high_option_replay_rate": high_fo_replay, "baseline_replay_rate": low_fo_replay, "replay_lift": lift})

    concept_trials = [row for row in transfer_rows if _node_level(view, row["memory_id"]) == MemoryLevel.M4]
    fo_memory_ids = {MemoryId(int(row["memory_id"])) for row in fo_rows if row.get("memory_id") is not None}
    linked_trials = []
    for trial in concept_trials:
        ancestors = _ancestor_ids(runtime, MemoryId(int(trial["memory_id"])))
        if ancestors & fo_memory_ids:
            linked_trials.append(trial)
    successful_linked = [row for row in linked_trials if row["success"]]
    linked_pairs = {(row["source_game"], row["target_game"]) for row in successful_linked}
    if len(successful_linked) >= 2 and len(linked_pairs) >= 2:
        h11_decision = "VALID"
    elif linked_trials:
        h11_decision = "PARTIALLY_VALID"
    else:
        h11_decision = "INSUFFICIENT_EVIDENCE"
    put("H11", h11_decision, len(successful_linked), len(linked_trials), {"future_option_linked_concept_transfer_trials": len(linked_trials), "successful_linked_transfers": len(successful_linked), "distinct_success_pairs": len(linked_pairs)})

    trajectories = _trajectory_rows(runtime)
    success_trajectories = [row for row in trajectories if bool(row.get("success"))]
    current = [row for row in success_trajectories if int(row.get("epoch", -1)) == int(epoch)]
    previous = [row for row in success_trajectories if int(row.get("epoch", -1)) < int(epoch)]
    current_mean = _mean([float(row["steps"]) for row in current if row.get("steps") is not None])
    previous_mean = _mean([float(row["steps"]) for row in previous if row.get("steps") is not None])
    improvement = None if current_mean is None or previous_mean is None else previous_mean - current_mean
    if len(current) >= 2 and previous_mean is not None:
        h12_decision = "VALID" if improvement is not None and improvement > 0 else "PARTIALLY_VALID"
    elif success_trajectories:
        h12_decision = "PARTIALLY_VALID"
    else:
        h12_decision = "INSUFFICIENT_EVIDENCE"
    put("H12", h12_decision, improvement, len(trajectories), {"successful_trajectories": len(success_trajectories), "current_epoch_successes": len(current), "current_mean_steps": current_mean, "previous_mean_steps": previous_mean, "step_improvement": improvement})

    _apply_dependencies(rows)
    return rows


def write_hypothesis_diagnostics(root: str | Path, *, epoch: int, rows: dict[str, dict[str, object]]) -> None:
    target = Path(root) / "reports" / f"hypotheses_epoch_{epoch + 1:04d}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"epoch": epoch + 1, "hypotheses": rows}, indent=2, sort_keys=True), encoding="utf-8")


def _interaction_rows(runtime) -> list[dict[str, Any]]:
    rows = runtime.evidence.connection.execute(
        "SELECT memory_id, source_game, source_context, source_global_step, payload_json FROM evidence_records WHERE evidence_type=? ORDER BY evidence_id",
        (EVIDENCE_INTERACTION,),
    ).fetchall()
    result: list[dict[str, Any]] = []
    for memory_id, game, context, step, payload_json in rows:
        try:
            payload = json.loads(str(payload_json or "{}"))
        except (TypeError, json.JSONDecodeError):
            payload = {}
        payload.update({"memory_id": memory_id, "source_game": game, "source_context": context, "source_global_step": step})
        result.append(payload)
    return result


def _trajectory_rows(runtime) -> list[dict[str, Any]]:
    rows = runtime.evidence.connection.execute(
        "SELECT source_game, source_global_step, payload_json FROM evidence_records WHERE evidence_type=? ORDER BY evidence_id",
        (EVIDENCE_TRAJECTORY,),
    ).fetchall()
    result: list[dict[str, Any]] = []
    for game, step, payload_json in rows:
        try:
            payload = json.loads(str(payload_json or "{}"))
        except (TypeError, json.JSONDecodeError):
            payload = {}
        payload.update({"source_game": game, "source_global_step": step})
        result.append(payload)
    return result


def _transfer_rows(runtime) -> list[dict[str, Any]]:
    return [
        {"memory_id": int(memory_id), "source_game": str(source_game), "target_game": str(target_game), "success": bool(success), "score": float(score)}
        for memory_id, source_game, target_game, success, score in runtime.lifecycle_evidence.connection.execute(
            "SELECT memory_id, source_game, target_game, success, score FROM transfer_trials ORDER BY transfer_trial_id"
        ).fetchall()
    ]


def _replay_rate(rows: list[dict[str, Any]], view) -> float | None:
    ids = [MemoryId(int(row["memory_id"])) for row in rows if row.get("memory_id") is not None and MemoryId(int(row["memory_id"])) in view.nodes]
    if not ids:
        return None
    replayed = sum(bool(int(view.nodes[mid].status_flags) & int(MemoryStatus.REPLAY_QUEUED)) for mid in ids)
    return replayed / len(ids)


def _parent_count(runtime, memory_id: MemoryId, *, level: MemoryLevel, view) -> int:
    return sum(1 for parent in runtime.lifecycle_evidence.provenance_parents(memory_id) if parent in view.nodes and view.nodes[parent].level == level)


def _validated_concept_parent_count(runtime, memory_id: MemoryId, view) -> int:
    return sum(
        1
        for parent in runtime.lifecycle_evidence.provenance_parents(memory_id)
        if parent in view.nodes
        and view.nodes[parent].level == MemoryLevel.M4
        and int(view.nodes[parent].status_flags) & int(ConceptValidationStatus.TRANSFER_VALIDATED)
    )


def _ancestor_ids(runtime, memory_id: MemoryId) -> set[MemoryId]:
    rows = runtime.lifecycle_evidence.connection.execute(
        """
        WITH RECURSIVE ancestry(memory_id) AS (
            SELECT ?
            UNION
            SELECT p.parent_memory_id FROM provenance_records p JOIN ancestry a ON p.memory_id=a.memory_id WHERE p.parent_memory_id IS NOT NULL
        )
        SELECT memory_id FROM ancestry
        """,
        (int(memory_id),),
    ).fetchall()
    return {MemoryId(int(row[0])) for row in rows}


def _node_level(view, memory_id: int) -> MemoryLevel | None:
    node = view.nodes.get(MemoryId(int(memory_id)))
    return None if node is None else node.level


def _apply_dependencies(rows: dict[str, dict[str, object]]) -> None:
    deps = {
        "H03": ("H01",), "H04": ("H03",), "H05": ("H04",), "H06": ("H05",),
        "H07": ("H06",), "H08": ("H07",), "H10": ("H09",), "H11": ("H06", "H09"),
    }
    for hid, required in deps.items():
        blocked = any(str(rows[parent]["raw_decision"]) in {"INVALID", "INSUFFICIENT_EVIDENCE"} for parent in required)
        rows[hid]["dependency_gate"] = "FAIL" if blocked else "PASS"


def _rate(a: int, b: int) -> float | None:
    return None if b <= 0 else a / b


def _mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)
