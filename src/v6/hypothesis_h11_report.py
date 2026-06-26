from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from v6.higher_order_substrate import derive_higher_order_memory
from v6.future_options import derive_future_option_memory
from v6.memory.compact_memory import ensure_memory_layout


def evaluate_h11_future_option_transfer_concepts(
    *,
    memory_dir: Path,
    run_dir: Path | None,
    output_dir: Path,
    already_derived: bool = False,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_memory_layout(memory_dir)
    if not already_derived:
        derive_higher_order_memory(memory_dir=memory_dir, run_dir=run_dir)
        derive_future_option_memory(memory_dir=memory_dir, run_dir=run_dir)
    with sqlite3.connect(Path(memory_dir) / "current_state.sqlite") as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(row) for row in conn.execute("SELECT * FROM future_option_transfer_links ORDER BY motif_signature ASC, role_signature ASC, concept_signature ASC").fetchall()]
        emergent_motifs = int(conn.execute("SELECT COUNT(*) FROM future_option_motifs WHERE COALESCE(is_emergent, 0) = 1").fetchone()[0])
        successful_role_transfer_count = int(conn.execute("SELECT COUNT(*) FROM role_transfer_attempts WHERE COALESCE(reuse_success, 0) = 1").fetchone()[0])
        promoted_concept_count = int(conn.execute("SELECT COUNT(*) FROM concept_candidates WHERE COALESCE(is_promoted, 0) = 1").fetchone()[0])
    motifs_with_transfer = len({str(row["motif_signature"]) for row in rows if int(row["transfer_attempt_count"] or 0) > 0})
    motifs_with_strong = len({str(row["motif_signature"]) for row in rows if int(row["strong_transfer_success_count"] or 0) > 0})
    motifs_with_promoted = len({str(row["motif_signature"]) for row in rows if int(row["promoted_concept_count"] or 0) > 0})
    total_attempts = sum(int(row["transfer_attempt_count"] or 0) for row in rows)
    total_successes = sum(int(row["successful_transfer_count"] or 0) for row in rows)
    total_strong = sum(int(row["strong_transfer_success_count"] or 0) for row in rows)
    result = {
        "hypothesis_id": "H11",
        "evidence_source": "compact_memory",
        "future_option_transfer_link_count": len(rows),
        "motifs_with_transfer_count": motifs_with_transfer,
        "motifs_with_strong_transfer_count": motifs_with_strong,
        "motifs_with_promoted_concept_count": motifs_with_promoted,
        "motif_transfer_success_rate": (total_successes / total_attempts) if total_attempts else None,
        "motif_strong_transfer_success_rate": (total_strong / total_attempts) if total_attempts else None,
        "promoted_concept_motif_count": motifs_with_promoted,
        "emergent_future_option_motif_count": emergent_motifs,
        "successful_role_transfer_count": successful_role_transfer_count,
        "promoted_concept_count": promoted_concept_count,
        "missing_evidence": [],
    }
    if emergent_motifs == 0:
        result["decision"] = "INCONCLUSIVE"
    elif successful_role_transfer_count == 0 and promoted_concept_count == 0:
        result["decision"] = "INCONCLUSIVE"
    elif len(rows) > 0 and motifs_with_strong == 0:
        result["decision"] = "PARTIALLY_VALID"
    elif (
        len(rows) >= 5
        and motifs_with_strong >= 1
        and motifs_with_promoted >= 1
        and (result["motif_strong_transfer_success_rate"] or 0.0) > 0.0
    ):
        result["decision"] = "VALID"
    elif successful_role_transfer_count > 0 and promoted_concept_count > 0 and len(rows) == 0:
        result["decision"] = "INVALID"
    else:
        result["decision"] = "PARTIALLY_VALID"
    result["core_metrics"] = {
        key: result.get(key)
        for key in (
            "future_option_transfer_link_count",
            "motifs_with_transfer_count",
            "motifs_with_strong_transfer_count",
            "motifs_with_promoted_concept_count",
            "motif_transfer_success_rate",
            "motif_strong_transfer_success_rate",
            "promoted_concept_motif_count",
            "emergent_future_option_motif_count",
            "successful_role_transfer_count",
            "promoted_concept_count",
        )
    }
    _write(output_dir, result)
    return result


def _write(output_dir: Path, result: dict[str, object]) -> None:
    (output_dir / "h11_future_option_transfer_concepts_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    text = (
        f"H11 decision: {result.get('decision')}\n"
        f"future-option transfer links: {result.get('future_option_transfer_link_count')}\n"
        f"motifs with strong transfer: {result.get('motifs_with_strong_transfer_count')}\n"
        f"motifs with promoted concepts: {result.get('motifs_with_promoted_concept_count')}\n"
    )
    (output_dir / "h11_future_option_transfer_concepts_report.txt").write_text(text, encoding="utf-8")
    (output_dir / "h11_future_option_transfer_concepts.md").write_text("```\n" + text + "```\n", encoding="utf-8")
