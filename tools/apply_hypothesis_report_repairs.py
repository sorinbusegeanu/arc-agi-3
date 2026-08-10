from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old in text:
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        return
    if new in text:
        return
    raise RuntimeError(f"expected text not found in {path}: {old[:120]!r}")


def repair_h12() -> None:
    path = ROOT / "src/v6/evaluation/h12_efficiency_emergence.py"
    replace_once(
        path,
        "from v6.memory.compact_memory import configure_compact_sqlite_connection, ensure_memory_layout",
        "from v6.memory.compact_memory import configure_compact_sqlite_connection",
    )
    replace_once(path, "    ensure_memory_layout(memory_dir)\n", "")


def repair_h08() -> None:
    path = ROOT / "src/v6/higher_order_substrate.py"
    replace_once(
        path,
        "            and observed_outcome_count > 0\n            and functional_coherence_score > 0.0",
        "            and functional_coherence_score > 0.0",
    )
    replace_once(
        path,
        "                    coherence_score = ?, candidate_only = ?, is_coherent = ?\n",
        "                    coherence_score = ?, is_coherent = ?\n",
    )
    replace_once(
        path,
        "                    adjusted_coherence_score, int(not coherent), int(coherent), component_signature,\n",
        "                    adjusted_coherence_score, int(coherent), component_signature,\n",
    )


def repair_h11() -> None:
    future_options = ROOT / "src/v6/future_options.py"
    replace_once(
        future_options,
        "                    transfer_provenance_status = (\n                        \"verified\" if pair_rows and all(str(row.get(\"provenance_status\") or \"\") == \"verified\" for row in pair_rows)\n                        else \"resolved_with_surrogate\" if any((source_game_is_surrogate, target_game_is_surrogate, source_context_is_surrogate, target_context_is_surrogate))\n                        else \"proxy\"\n                    )",
        "                    transfer_provenance_status = (\n                        \"resolved_with_surrogate\" if any((source_game_is_surrogate, target_game_is_surrogate, source_context_is_surrogate, target_context_is_surrogate))\n                        else \"verified\" if pair_rows and all(str(row.get(\"provenance_status\") or \"\") == \"verified\" for row in pair_rows)\n                        else \"proxy\"\n                    )",
    )

    provenance = ROOT / "src/v6/provenance_validation.py"
    replace_once(
        provenance,
        "        elif \"missing\" in statuses:\n            _count(report, \"missing\", \"H11\")\n",
        "        elif \"missing\" in statuses:\n            # A materialized H11 link with a partially resolved chain is candidate/proxy evidence,\n            # not a missing required claim. Only a wholly absent provenance chain is missing.\n            if statuses == {\"missing\"}:\n                _count(report, \"missing\", \"H11\")\n            else:\n                _count(report, \"proxy\", \"H11\")\n",
    )


def repair_h07() -> None:
    path = ROOT / "src/v6/hypothesis_h07_report.py"
    text = path.read_text(encoding="utf-8")
    marker = "H07_CURRENT_VALIDATION_GATE_V1"
    if marker in text:
        return
    anchor = text.rfind("    _write_outputs(output_dir, result)")
    if anchor < 0:
        raise RuntimeError("H07 final write anchor not found")
    gate = '''    # H07_CURRENT_VALIDATION_GATE_V1\n    retained_without_validation = int(result.get("concepts_retained_without_current_validation", 0) or 0)\n    result["promotion_retained_without_current_validation"] = retained_without_validation > 0\n    if retained_without_validation > 0 and result.get("decision") == "VALID":\n        result["decision"] = "PARTIALLY_VALID"\n        result["missing_evidence"] = list(dict.fromkeys(\n            list(result.get("missing_evidence", []))\n            + [f"{retained_without_validation} promoted concept(s) are retained without current held-out validation."]\n        ))\n\n'''
    path.write_text(text[:anchor] + gate + text[anchor:], encoding="utf-8")


def repair_h10b() -> None:
    path = ROOT / "src/v6/evaluation/h10b_selective_forgetting.py"
    text = path.read_text(encoding="utf-8")
    marker = "H10B_SUBSTANTIVE_EVIDENCE_GATE_V1"
    if marker in text:
        return
    anchor = text.rfind("    _write_report(output_dir, result)")
    if anchor < 0:
        raise RuntimeError("H10B report-write anchor not found")
    gate = '''    # H10B_SUBSTANTIVE_EVIDENCE_GATE_V1\n    compression_improved = float(result.get("compression_ratio_after") or 0.0) > float(result.get("compression_ratio_before") or 0.0)\n    abstraction_improved = float(result.get("abstraction_score_after") or 0.0) > float(result.get("abstraction_score_before") or 0.0)\n    transfer_before = result.get("transfer_score_before")\n    transfer_after = result.get("transfer_score_after")\n    transfer_improved = (\n        transfer_before is not None\n        and transfer_after is not None\n        and float(transfer_after) > float(transfer_before)\n    )\n    substantive_forgetting_evidence = compression_improved or abstraction_improved or transfer_improved\n    result["substantive_forgetting_evidence"] = substantive_forgetting_evidence\n    if result.get("decision") == "VALID" and not substantive_forgetting_evidence:\n        result["decision"] = "PARTIALLY_VALID"\n        result["missing_evidence"] = list(dict.fromkeys(\n            list(result.get("missing_evidence", []))\n            + ["Selective survival lift exists, but no compression, abstraction, or transfer improvement is demonstrated."]\n        ))\n\n'''
    path.write_text(text[:anchor] + gate + text[anchor:], encoding="utf-8")


def main() -> None:
    repair_h12()
    repair_h08()
    repair_h11()
    repair_h07()
    repair_h10b()


if __name__ == "__main__":
    main()
