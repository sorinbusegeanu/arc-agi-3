from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def install_compat(ns: dict[str, Any]) -> None:
    strict_build = ns["build_hypothesis_suite_summary"]

    def _load_report(run_dir: Path) -> dict[str, Any]:
        path = Path(run_dir) / ns.get("INPUT_REPORT_NAME", "interaction_sampling_v05c_report.json")
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _mean(values: list[Any]) -> float | None:
        cooked = [float(value) for value in values if value is not None]
        return sum(cooked) / len(cooked) if cooked else None

    def _ordered(left: Any, right: Any) -> bool | None:
        if left is None or right is None:
            return None
        return int(left) <= int(right)

    def _ratio(values: list[bool]) -> float | None:
        return (sum(1 for value in values if value) / len(values)) if values else None

    def _temporal(rows: list[dict[str, Any]]) -> dict[str, Any]:
        h01_values: list[bool] = []
        h02_values: list[bool] = []
        h03_values: list[bool] = []
        per_case: list[dict[str, Any]] = []
        missing = 0
        cases = 0
        for row in rows:
            h01 = _ordered(row.get("first_stable_contingency_step"), row.get("first_transformation_family_step"))
            h02 = _ordered(row.get("first_prediction_violation_step"), row.get("first_transformation_family_step"))
            h03 = _ordered(row.get("first_stable_transformation_family_step"), row.get("first_emergent_carrier_step"))
            per_case.append({
                "game": row.get("game"), "sampler": row.get("sampler"), "seed": row.get("seed"),
                "h01_before_h03": h01, "h02_before_h03": h02, "h03_before_h04": h03,
            })
            local = [h01, h02, h03]
            if any(value is not None for value in local):
                cases += 1
            missing += sum(value is None for value in local)
            if h01 is not None: h01_values.append(h01)
            if h02 is not None: h02_values.append(h02)
            if h03 is not None: h03_values.append(h03)
        return {
            "per_case": per_case,
            "temporal_order_cases_available": cases,
            "h01_before_h03_ratio": _ratio(h01_values),
            "h02_before_h03_ratio": _ratio(h02_values),
            "h03_before_h04_ratio": _ratio(h03_values),
            "temporal_order_missing_count": missing,
        }

    def _per_game(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in runs:
            game = str(row.get("game") or "")
            if game:
                grouped.setdefault(game, []).append(row)
        output: list[dict[str, Any]] = []
        for game, items in sorted(grouped.items()):
            interactions = sum(int(row.get("total_interactions", 0) or 0) for row in items)
            stable = sum(int(row.get("stable_contingency_count", 0) or 0) for row in items)
            families = sum(int(row.get("unique_transformation_families", 0) or 0) for row in items)
            h02_signal = any((row.get("mean_isf_prediction_error") or 0.0) > 0.0 and int(row.get("high_priority_replay_count", 0) or 0) > 0 for row in items)
            status = "missing" if interactions <= 0 else "supported" if stable > 0 and families > 0 and h02_signal else "partial" if stable > 0 else "weak"
            output.append({
                "game": game, "interaction_count": interactions, "stable_contingency_count": stable,
                "mean_prediction_accuracy": _mean([row.get("prediction_accuracy") for row in items]),
                "transformation_family_count": families, "status": status,
            })
        return output

    def _per_sampler(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in runs:
            sampler = str(row.get("sampler_name") or row.get("sampler") or "")
            if sampler:
                grouped.setdefault(sampler, []).append(row)
        output: list[dict[str, Any]] = []
        for sampler, items in sorted(grouped.items()):
            stable = sum(int(row.get("stable_contingency_count", 0) or 0) for row in items)
            families = sum(int(row.get("unique_transformation_families", 0) or 0) for row in items)
            replay = sum(int(row.get("high_priority_replay_count", 0) or 0) for row in items)
            output.append({
                "sampler": sampler,
                "interaction_count": sum(int(row.get("total_interactions", 0) or 0) for row in items),
                "stable_contingency_count": stable, "transformation_family_count": families,
                "high_priority_replay_count": replay,
                "status": "supported" if stable > 0 and families > 0 and replay > 0 else "partial" if stable > 0 else "weak",
            })
        return output

    def build_hypothesis_suite_summary(*, run_dir: Path, memory_dir: Path | None = None, output_dir: Path | None = None,
                                       epoch_id: str | None = None, global_step_start: int | None = None,
                                       global_step_end: int | None = None, interactions_this_epoch: int | None = None,
                                       total_interactions_seen: int | None = None, memory_size_before_bytes: int | None = None,
                                       memory_size_after_bytes: int | None = None, hypothesis_results: dict[str, dict[str, Any]] | None = None,
                                       evidence_snapshot: Any | None = None, derivation_summary: dict[str, Any] | None = None,
                                       provenance_validation: dict[str, Any] | None = None, timings: dict[str, float] | None = None,
                                       **metadata: Any) -> dict[str, Any]:
        legacy_results: dict[str, dict[str, Any]] = {}
        for number in range(1, 13):
            value = metadata.pop(f"h{number:02d}", None)
            if isinstance(value, dict):
                legacy_results[f"H{number:02d}"] = dict(value)
        if hypothesis_results is None:
            hypothesis_results = legacy_results
        elif legacy_results:
            hypothesis_results = {**legacy_results, **hypothesis_results}
        summary = strict_build(
            run_dir=run_dir, memory_dir=memory_dir, output_dir=output_dir, epoch_id=epoch_id,
            global_step_start=global_step_start, global_step_end=global_step_end,
            interactions_this_epoch=interactions_this_epoch, total_interactions_seen=total_interactions_seen,
            memory_size_before_bytes=memory_size_before_bytes, memory_size_after_bytes=memory_size_after_bytes,
            hypothesis_results=hypothesis_results, evidence_snapshot=evidence_snapshot,
            derivation_summary=derivation_summary, provenance_validation=provenance_validation,
            timings=timings, **metadata,
        )
        report = _load_report(Path(run_dir))
        runs = [dict(item) for item in report.get("runs", []) if isinstance(item, dict)]
        temporal_rows = [dict(item) for item in ((report.get("temporal_milestones") or {}).get("by_game_sampler_seed", []) or []) if isinstance(item, dict)]
        games = sorted({str(row.get("game")) for row in runs if row.get("game")}) or [str(x) for x in report.get("games", []) if x]
        samplers = sorted({str(row.get("sampler_name") or row.get("sampler")) for row in runs if row.get("sampler_name") or row.get("sampler")}) or [str(x) for x in report.get("samplers", []) if x]
        seeds = sorted({int(row.get("seed")) for row in temporal_rows if row.get("seed") is not None}) or [int(x) for x in report.get("seeds", []) if x is not None]
        raw_total = sum(int(row.get("total_interactions", 0) or 0) for row in runs)
        h01 = (hypothesis_results or {}).get("H01", {})
        total = raw_total
        source = "raw_report_runs" if total > 0 else "unavailable"
        if total <= 0 and interactions_this_epoch is not None:
            total = int(interactions_this_epoch or 0); source = "continuous_epoch_argument" if total > 0 else source
        if total <= 0 and h01.get("total_interaction_count") is not None:
            total = int(h01.get("total_interaction_count") or 0); source = "h01_total_interaction_count" if total > 0 else source
        if total <= 0 and total_interactions_seen is not None:
            total = int(total_interactions_seen or 0); source = "compact_memory_total_interactions_seen" if total > 0 else source
        summary.update({
            "game_count": len(games), "sampler_count": len(samplers), "seed_count": len(seeds),
            "total_interactions": int(total), "raw_report_total_interactions": int(raw_total),
            "total_interactions_source": source, "per_game_status_table": _per_game(runs),
            "per-game status table": _per_game(runs), "per_sampler_status_table": _per_sampler(runs),
            "per-sampler status table": _per_sampler(runs), "temporal_order_diagnostics": _temporal(temporal_rows),
            "suite_mode": (derivation_summary or {}).get("suite_mode", metadata.get("suite_mode", "fast")),
        })
        results = hypothesis_results or {}
        for number in range(1, 13):
            key = f"H{number:02d}"
            payload = results.get(key, {})
            decision = payload.get("decision") or payload.get("final_decision") or "INCONCLUSIVE"
            if memory_dir is None and number >= 4 and key not in legacy_results:
                decision = "INCONCLUSIVE"
            summary[f"{key} decision"] = decision
            core = dict(payload.get("core_metrics") or {})
            if key == "H02":
                for name in ("h02a_replay_attention_decision", "h02b_pre_carrier_timing_decision", "h02_final_decision_basis", "carrier_timing_note", "prediction_violation_replay_lift"):
                    core.setdefault(name, payload.get(name))
            if key in {"H09", "H10", "H11"}:
                for name, value in payload.items():
                    if name not in {"core_metrics", "missing_evidence"} and not isinstance(value, (dict, list)):
                        core.setdefault(name, value)
            summary[f"{key} core metrics"] = core
        combined_timings = dict(timings or {})
        combined_timings.update(dict((derivation_summary or {}).get("timings") or {}))
        for name, value in combined_timings.items():
            summary[name] = value
            if name.startswith("DERIVE."):
                summary["derive_" + name[len("DERIVE."):]] = value
        summary.setdefault("suite_total_seconds", float((timings or {}).get("suite_total_seconds", 0.0)))
        summary.setdefault("missing_evidence", [])
        return summary

    def _demote(payload: dict[str, Any], message: str, *, maturity: bool = False) -> dict[str, Any]:
        updated = dict(payload)
        if str(updated.get("decision")) != "VALID":
            return updated
        updated.setdefault("raw_decision", "VALID")
        updated.setdefault("individual_decision_before_suite_gates", "VALID")
        updated["decision"] = "PARTIALLY_VALID"
        updated["final_decision"] = "PARTIALLY_VALID"
        updated["suite_gated_decision"] = "PARTIALLY_VALID"
        updated["epoch_maturity_demoted" if maturity else "dependency_demoted"] = True
        missing = list(updated.get("missing_evidence", []) or [])
        reasons = list(updated.get("suite_gate_reasons", []) or [])
        if message not in missing: missing.append(message)
        if message not in reasons: reasons.append(message)
        updated["missing_evidence"] = missing
        updated["suite_gate_reasons"] = reasons
        return updated

    def _apply_higher_order_dependency_gates(h04: dict[str, Any], h05: dict[str, Any], h06: dict[str, Any], h07: dict[str, Any], h08: dict[str, Any], h09: dict[str, Any], h10: dict[str, Any], h11: dict[str, Any]):
        notes: list[str] = []
        def gate(payload: dict[str, Any], message: str) -> dict[str, Any]:
            before = str(payload.get("decision"))
            updated = _demote(payload, message)
            if before == "VALID" and str(updated.get("decision")) != before: notes.append(message)
            return updated
        if str(h04.get("decision")) == "INVALID" and str(h05.get("decision")) == "VALID":
            h05 = gate(h05, "H05 depends on invalid H04 carrier emergence."); h05["h05_depends_on_invalid_h04"] = True
        if str(h05.get("decision")) == "VALID" and (str(h04.get("decision")) != "VALID" or h04.get("h04_graph_quality_pass") is not True or int(h04.get("usable_emergent_carrier_count") or 0) <= 0):
            h05 = gate(h05, "H05 cannot be fully VALID until H04 has VALID usable carrier emergence with graph-quality pass.")
        if str(h05.get("decision")) != "VALID" and str(h06.get("decision")) == "VALID": h06 = gate(h06, "H06 cannot be fully VALID until H05 role emergence is VALID.")
        if str(h06.get("decision")) != "VALID" and str(h07.get("decision")) == "VALID": h07 = gate(h07, "H07 cannot be fully VALID until H06 role transfer is VALID.")
        if str(h07.get("decision")) != "VALID" and str(h08.get("decision")) == "VALID": h08 = gate(h08, "H08 cannot be fully VALID until H07 concept emergence is VALID.")
        if str(h06.get("decision")) != "VALID" and str(h08.get("decision")) == "VALID": h08 = gate(h08, "H08 cannot be fully VALID until H06 role transfer is VALID.")
        if str(h09.get("decision")) != "VALID" and str(h10.get("decision")) == "VALID": h10 = gate(h10, "H10 cannot be fully VALID until H09 future-option motifs are VALID.")
        if str(h09.get("decision")) != "VALID" and str(h11.get("decision")) == "VALID": h11 = gate(h11, "H11 cannot be fully VALID until H09 future-option motifs are VALID.")
        if str(h06.get("decision")) in {"INCONCLUSIVE", "INVALID", "INSUFFICIENT_EVIDENCE"} and str(h11.get("decision")) == "VALID": h11 = gate(h11, "H11 cannot be fully VALID until H06 role transfer is at least PARTIALLY_VALID.")
        return h05, h06, h07, h08, h09, h10, h11, notes

    def _apply_epoch_maturity_gates(*, h04: dict[str, Any], h05: dict[str, Any], h06: dict[str, Any], h07: dict[str, Any], h08: dict[str, Any], h09: dict[str, Any], h10: dict[str, Any], h11: dict[str, Any], total_interactions: int, interactions_this_epoch: int | None, game_count: int, sampler_count: int):
        threshold = max(1000, int(game_count) * int(sampler_count))
        notes: list[str] = []
        blocked = total_interactions <= 0
        if blocked: notes.append("Full H04-H11 validation blocked because total interaction count is unavailable.")
        elif interactions_this_epoch is not None and int(interactions_this_epoch) < threshold:
            blocked = True; notes.append("Full H04-H11 validation blocked because epoch interaction budget is below maturity threshold.")
        def apply(payload: dict[str, Any]) -> dict[str, Any]:
            updated = dict(payload)
            if blocked and str(updated.get("decision")) == "VALID":
                updated = _demote(updated, notes[0], maturity=True)
                updated["epoch_maturity_threshold"] = threshold
                updated["epoch_interactions_used_for_gate"] = interactions_this_epoch if interactions_this_epoch is not None else total_interactions
            return updated
        return apply(h04), apply(h05), apply(h06), apply(h07), apply(h08), apply(h09), apply(h10), apply(h11), notes

    def _blocker_flags_for_result(hypothesis_id: str, result: dict[str, Any]) -> dict[str, bool]:
        decision = str(result.get("decision") or "")
        missing = list(result.get("missing_evidence") or [])
        core = dict(result.get("core_metrics") or {})
        h03_before_h04 = result.get("h03_before_h04", core.get("h03_before_h04"))
        h04_before_h05 = result.get("h04_before_h05", core.get("h04_before_h05"))
        base = {
            "h02_missing_direct_linkage": False, "h03_missing_prediction_lift": False,
            "h04_missing_temporal_order": False, "h04_temporal_order_failed": False,
            "h05_missing_temporal_order": False, "h06_transfer_sampling_capped": False,
            "h07_no_promoted_concepts": False, "h08_proxy_only_world_model": False,
            "h09_no_future_option_events": False, "h10_blocked_by_h09": False,
            "h11_blocked_by_no_motifs": False, "h12_missing_trajectory_evidence": False,
            "skipped_fast_mode": decision == "SKIPPED_FAST_MODE",
        }
        if base["skipped_fast_mode"]: return base
        base.update({
            "h02_missing_direct_linkage": hypothesis_id == "H02" and (result.get("direct_replay_lift_available") is not True or result.get("raw_cleanup_prevents_direct_linkage") is True or any("linkage unavailable" in str(msg).lower() for msg in missing)),
            "h03_missing_prediction_lift": hypothesis_id == "H03" and result.get("family_prediction_lift_mean") is None,
            "h04_missing_temporal_order": hypothesis_id == "H04" and h03_before_h04 is None,
            "h04_temporal_order_failed": hypothesis_id == "H04" and h03_before_h04 is False,
            "h05_missing_temporal_order": hypothesis_id == "H05" and h04_before_h05 is None,
            "h06_transfer_sampling_capped": hypothesis_id == "H06" and int(result.get("skipped_by_cap_count") or 0) > 0,
            "h07_no_promoted_concepts": hypothesis_id == "H07" and int(result.get("promoted_concept_count") or 0) == 0,
            "h08_proxy_only_world_model": hypothesis_id == "H08" and bool(result.get("candidate_proxy_only")),
            "h09_no_future_option_events": hypothesis_id == "H09" and int(result.get("future_option_event_count") or 0) == 0,
            "h10_blocked_by_h09": hypothesis_id == "H10" and bool(result.get("h10_blocked_by_h09")),
            "h11_blocked_by_no_motifs": hypothesis_id == "H11" and bool(result.get("h11_blocked_by_no_motifs")),
            "h12_missing_trajectory_evidence": hypothesis_id == "H12" and bool(result.get("blocked_by_missing_trajectory_evidence")),
        })
        return base

    def _format_section(hypothesis_id: str, result: dict[str, Any]) -> str:
        return "\n".join([
            hypothesis_id, f"decision: {result.get('decision')}", f"evidence_stage: {result.get('evidence_stage')}",
            f"blocker_flags: {json.dumps(_blocker_flags_for_result(hypothesis_id, result), sort_keys=True)}",
            f"phase_seconds: {json.dumps(result.get('phase_seconds') or {}, sort_keys=True)}",
            f"core_metrics: {json.dumps(result.get('core_metrics') or {}, sort_keys=True)}",
            f"missing_evidence: {json.dumps(result.get('missing_evidence') or [], ensure_ascii=True)}",
            f"evidence_diagnostics: {json.dumps(result.get('evidence_diagnostics') or {}, sort_keys=True)}",
        ]).strip()

    def _write_aggregated_hypothesis_text(output_dir: Path, *, hypothesis_results: dict[str, dict[str, Any]] | None = None) -> None:
        output_dir = Path(output_dir)
        sections: list[str] = []
        placeholders: dict[int, str] = {}
        for number in range(1, 13):
            key = f"H{number:02d}"; subdir = output_dir / key.lower()
            result = None if hypothesis_results is None else hypothesis_results.get(key)
            if isinstance(result, dict):
                section = _format_section(key, result); sections.append(section); subdir.mkdir(parents=True, exist_ok=True)
                target = subdir / f"h{number:02d}_report.txt"
                if not target.exists(): target.write_text(section + "\n", encoding="utf-8")
                continue
            if not subdir.exists(): placeholders[number] = f"{key} report unavailable"; continue
            files = sorted(subdir.glob("*.txt"))
            if not files: placeholders[number] = f"{key} report unavailable"; continue
            text = files[0].read_text(encoding="utf-8").strip()
            if text: sections.append(text)
            else: placeholders[number] = f"{key} report unavailable"
        if not sections:
            summary_path = output_dir / ns.get("SUITE_TXT_NAME", "hypothesis_suite_summary.txt")
            if summary_path.exists() and summary_path.read_text(encoding="utf-8").strip(): sections.append(summary_path.read_text(encoding="utf-8").strip())
        if not sections: sections = [f"H{number:02d} report unavailable" for number in range(1, 12)]
        for number, text in placeholders.items():
            if number > 11: continue
            subdir = output_dir / f"h{number:02d}"; subdir.mkdir(parents=True, exist_ok=True)
            target = subdir / f"h{number:02d}_report.txt"
            if not target.exists(): target.write_text(text + "\n", encoding="utf-8")
        (output_dir / ns.get("SUITE_AGGREGATED_TXT_NAME", "hypothesis_suite_aggregated.txt")).write_text("\n\n".join(sections).strip() + "\n", encoding="utf-8")

    def _format_text(summary: dict[str, Any]) -> str:
        lines = ["Hypothesis Suite Summary", f"source_run_dir: {summary.get('source_run_dir')}"]
        for number in range(1, 13): lines.append(f"H{number:02d}: {summary.get(f'H{number:02d} decision', 'INCONCLUSIVE')}")
        lines.append(f"games: {summary.get('game_count', 0)} samplers: {summary.get('sampler_count', 0)} seeds: {summary.get('seed_count', 0)}")
        lines.append(f"total_interactions: {summary.get('total_interactions', 0)}")
        return "\n".join(lines) + "\n"

    ns["build_hypothesis_suite_summary"] = build_hypothesis_suite_summary
    ns["_apply_higher_order_dependency_gates"] = _apply_higher_order_dependency_gates
    ns["_apply_epoch_maturity_gates"] = _apply_epoch_maturity_gates
    ns["_blocker_flags_for_result"] = _blocker_flags_for_result
    ns["_write_aggregated_hypothesis_text"] = _write_aggregated_hypothesis_text
    ns["_format_aggregated_result_section"] = _format_section
    ns["_format_text"] = _format_text
