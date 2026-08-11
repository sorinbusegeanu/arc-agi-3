from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"expected text not found in {path}: {old[:160]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def patch_h08_threshold() -> None:
    path = ROOT / "src/v6/higher_order_substrate.py"
    replace_once(
        path,
        "            and coherence_score >= 0.55\n",
        "            and coherence_score >= 0.45\n",
    )


def patch_future_option_transfer_derivation() -> None:
    path = ROOT / "src/v6/future_options.py"
    replace_once(
        path,
        "    fully_verified_emergent_chain_count = 0\n    partially_verified_emergent_chain_count = 0\n    unverified_emergent_chain_count = 0\n    for motif_signature in sorted(motif_links):\n",
        "    fully_verified_emergent_chain_count = 0\n    partially_verified_emergent_chain_count = 0\n    unverified_emergent_chain_count = 0\n\n    # Pairing role-transfer evidence is role-local and does not depend on the motif.\n    # Build it once instead of rebuilding identical dictionaries for every motif-role link.\n    transfer_pairs_by_role: dict[\n        str,\n        dict[tuple[str | None, str | None, str | None, str | None], list[dict[str, Any]]],\n    ] = {}\n    for cached_role_signature, cached_rows in transfers_by_role.items():\n        cached_pairs: dict[\n            tuple[str | None, str | None, str | None, str | None],\n            list[dict[str, Any]],\n        ] = defaultdict(list)\n        for transfer_row in cached_rows:\n            cached_pairs[(\n                transfer_row.get(\"source_game_key\"),\n                transfer_row.get(\"target_game_key\"),\n                transfer_row.get(\"source_context_key\"),\n                transfer_row.get(\"target_context_key\"),\n            )].append(transfer_row)\n        if not cached_pairs:\n            cached_pairs[(None, None, None, None)] = []\n        transfer_pairs_by_role[cached_role_signature] = cached_pairs\n\n    # first/last evidence bounds depend only on motif-role-concept, not on the\n    # source/target scope pair. Large runs can emit many scope pairs per triple.\n    evidence_bounds_cache: dict[tuple[str, str, str], tuple[Any, Any]] = {}\n\n    for motif_signature in sorted(motif_links):\n",
    )
    replace_once(
        path,
        "            rows_by_pair: dict[tuple[str | None, str | None, str | None, str | None], list[dict[str, Any]]] = defaultdict(list)\n            for transfer_row in role_transfer_rows:\n                pair = (\n                    transfer_row.get(\"source_game_key\"), transfer_row.get(\"target_game_key\"),\n                    transfer_row.get(\"source_context_key\"), transfer_row.get(\"target_context_key\"),\n                )\n                rows_by_pair[pair].append(transfer_row)\n            if not rows_by_pair:\n                rows_by_pair[(None, None, None, None)] = []\n",
        "            rows_by_pair = transfer_pairs_by_role.get(\n                role_signature,\n                {(None, None, None, None): []},\n            )\n",
    )
    replace_once(
        path,
        "                    first_seen = _safe_min(state_conn, motif_signature, role_signature, concept_signature, \"first\")\n                    last_seen = _safe_min(state_conn, motif_signature, role_signature, concept_signature, \"last\")\n",
        "                    bounds_key = (motif_signature, role_signature, concept_signature)\n                    bounds = evidence_bounds_cache.get(bounds_key)\n                    if bounds is None:\n                        bounds = (\n                            _safe_min(state_conn, motif_signature, role_signature, concept_signature, \"first\"),\n                            _safe_min(state_conn, motif_signature, role_signature, concept_signature, \"last\"),\n                        )\n                        evidence_bounds_cache[bounds_key] = bounds\n                    first_seen, last_seen = bounds\n",
    )


def patch_h11_report_scan() -> None:
    path = ROOT / "src/v6/hypothesis_h11_report.py"
    old = '''    for motif_signature in {\n        str(row.get("motif_signature"))\n        for row in links\n        if row.get("motif_signature") not in (None, "")\n    }:\n        motif_rows = [\n            row\n            for row in links\n            if str(row.get("motif_signature")) == motif_signature\n        ]\n        if (\n            "motifs_skipped_no_role_links" not in derivation_summary\n            and not any(\n                row.get("role_signature") not in (None, "")\n                or row.get("source_role_signature") not in (None, "")\n                for row in motif_rows\n            )\n        ):\n            motifs_skipped_no_role_links += 1\n        if (\n            "motifs_skipped_no_transfer_attempts" not in derivation_summary\n            and not any(\n                _int(row.get("transfer_attempt_count")) > 0\n                for row in motif_rows\n            )\n        ):\n            motifs_skipped_no_transfer_attempts += 1\n        if (\n            "motifs_skipped_no_concepts" not in derivation_summary\n            and not any(\n                row.get("concept_signature")\n                not in (None, "", "__none__")\n                for row in motif_rows\n            )\n        ):\n            motifs_skipped_no_concepts += 1\n'''
    new = '''    fallback_keys = (\n        "motifs_skipped_no_role_links",\n        "motifs_skipped_no_transfer_attempts",\n        "motifs_skipped_no_concepts",\n    )\n    if any(key not in derivation_summary for key in fallback_keys):\n        motif_rows_by_signature: dict[str, list[dict[str, object]]] = defaultdict(list)\n        for row in links:\n            motif_signature = row.get("motif_signature")\n            if motif_signature not in (None, ""):\n                motif_rows_by_signature[str(motif_signature)].append(row)\n        for motif_rows in motif_rows_by_signature.values():\n            if (\n                "motifs_skipped_no_role_links" not in derivation_summary\n                and not any(\n                    row.get("role_signature") not in (None, "")\n                    or row.get("source_role_signature") not in (None, "")\n                    for row in motif_rows\n                )\n            ):\n                motifs_skipped_no_role_links += 1\n            if (\n                "motifs_skipped_no_transfer_attempts" not in derivation_summary\n                and not any(\n                    _int(row.get("transfer_attempt_count")) > 0\n                    for row in motif_rows\n                )\n            ):\n                motifs_skipped_no_transfer_attempts += 1\n            if (\n                "motifs_skipped_no_concepts" not in derivation_summary\n                and not any(\n                    row.get("concept_signature") not in (None, "", "__none__")\n                    for row in motif_rows\n                )\n            ):\n                motifs_skipped_no_concepts += 1\n'''
    replace_once(path, old, new)


def patch_suite_profiler() -> None:
    path = ROOT / "src/v6/hypothesis_suite_report.py"
    replace_once(
        path,
        "def evaluate_hypotheses_read_only(\n",
        "def evaluate_hypotheses_read_only(\n",
    )
    replace_once(
        path,
        "    for hypothesis_id, evaluator, kwargs in tasks:\n        results[hypothesis_id] = _evaluate_one(\n            hypothesis_id, evaluator, kwargs=kwargs\n        )\n    return results\n",
        "    evaluator_timings: dict[str, float] = {}\n    for hypothesis_id, evaluator, kwargs in tasks:\n        evaluator_started = time.perf_counter()\n        results[hypothesis_id] = _evaluate_one(\n            hypothesis_id, evaluator, kwargs=kwargs\n        )\n        evaluator_timings[hypothesis_id] = time.perf_counter() - evaluator_started\n        results[hypothesis_id][\"evaluator_seconds\"] = evaluator_timings[hypothesis_id]\n    results[\"__timings__\"] = {\n        \"evaluator_seconds\": evaluator_timings,\n        \"total_evaluator_seconds\": sum(evaluator_timings.values()),\n    }\n    return results\n",
    )
    replace_once(
        path,
        "        fingerprint_after_report = memory_fingerprint(memory_dir)\n",
        "        evaluator_profile = dict(raw_results.pop(\"__timings__\", {}))\n\n        fingerprint_after_report = memory_fingerprint(memory_dir)\n",
    )
    replace_once(
        path,
        "            \"report_seconds\": report_seconds,\n            \"suite_total_seconds\": time.time() - started,\n",
        "            \"report_seconds\": report_seconds,\n            \"suite_total_seconds\": time.time() - started,\n            **{\n                f\"evaluator_{key.lower()}_seconds\": float(value)\n                for key, value in dict(evaluator_profile.get(\"evaluator_seconds\", {})).items()\n            },\n            \"evaluator_total_seconds\": float(evaluator_profile.get(\"total_evaluator_seconds\", 0.0) or 0.0),\n",
    )


def main() -> None:
    patch_h08_threshold()
    patch_future_option_transfer_derivation()
    patch_h11_report_scan()
    patch_suite_profiler()


if __name__ == "__main__":
    main()
