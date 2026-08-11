from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HIGHER = ROOT / "src/v6/higher_order_substrate.py"
H11 = ROOT / "src/v6/hypothesis_h11_report.py"
SUITE = ROOT / "src/v6/hypothesis_suite_report.py"
CONTINUOUS = ROOT / "src/v6/continuous_research.py"
CLI = ROOT / "src/v6/cli.py"


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"expected text not found in {path}: {old[:180]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def patch_h08() -> None:
    replace_once(
        HIGHER,
        "            and linked_carrier_count >= 2\n            and (cross_context_count >= 3 or cross_game_count >= 2)\n",
        "            and (cross_context_count >= 3 or cross_game_count >= 2)\n",
    )

    replace_once(
        HIGHER,
        '''        component_links = _links_by_signature(state_conn, "world_model_links", "component_signature")\n        for row in component_rows:\n            component_signature = str(row["component_signature"])\n            concepts = component_links.get(component_signature, {}).get("concept", set())\n            validations = [concept_validation[item] for item in concepts if item in concept_validation]\n            prediction_lift = _mean_row_metric(validations, "validation_prediction_lift")\n            action_lift = _mean_row_metric(validations, "validation_action_selection_lift")\n            transfer_lift = _mean_row_metric(validations, "validation_transfer_lift")\n            passed = bool(validations) and all(int(item.get("is_promoted", 0) or 0) == 1 for item in validations)\n''',
        '''        component_links = _links_by_signature(state_conn, "world_model_links", "component_signature")\n        # Real components have a deterministic owning concept. Other concept\n        # graph links are contextual evidence and must not demote the owner.\n        # Synthetic/legacy component ids fall back to their explicit links.\n        owner_validation = {\n            f"wm:{sha1(signature.encode('utf-8')).hexdigest()[:20]}": record\n            for signature, record in concept_validation.items()\n        }\n        for row in component_rows:\n            component_signature = str(row["component_signature"])\n            owner = owner_validation.get(component_signature)\n            if owner is not None:\n                validations = [owner]\n            else:\n                concepts = component_links.get(component_signature, {}).get("concept", set())\n                validations = [concept_validation[item] for item in concepts if item in concept_validation]\n            prediction_lift = _mean_row_metric(validations, "validation_prediction_lift")\n            action_lift = _mean_row_metric(validations, "validation_action_selection_lift")\n            transfer_lift = _mean_row_metric(validations, "validation_transfer_lift")\n            passed = bool(validations) and all(int(item.get("is_promoted", 0) or 0) == 1 for item in validations)\n''',
    )

    replace_once(
        HIGHER,
        '''            state_conn.execute(\n                """\n                UPDATE world_model_components\n                SET validation_prediction_lift = ?, validation_action_selection_lift = ?,\n                    validation_transfer_lift = ?, promotion_status = ?, promotion_failure_count = ?,\n                    coherence_score = ?, is_coherent = ?\n                WHERE component_signature = ?\n                """,\n                (\n                    prediction_lift, action_lift, transfer_lift, status, failure_count,\n                    adjusted_coherence_score, int(coherent), component_signature,\n                ),\n            )\n            if demoted:\n''',
        '''            state_conn.execute(\n                """\n                UPDATE world_model_components\n                SET validation_prediction_lift = ?, validation_action_selection_lift = ?,\n                    validation_transfer_lift = ?, promotion_status = ?, promotion_failure_count = ?,\n                    coherence_score = ?, is_coherent = ?\n                WHERE component_signature = ?\n                """,\n                (\n                    prediction_lift, action_lift, transfer_lift, status, failure_count,\n                    adjusted_coherence_score, int(coherent), component_signature,\n                ),\n            )\n            state_conn.execute(\n                """\n                UPDATE world_model_component_state\n                SET historically_coherent = MAX(historically_coherent, ?),\n                    currently_coherent = ?,\n                    first_coherent_global_step = COALESCE(\n                        first_coherent_global_step,\n                        CASE WHEN ? = 1 THEN ? ELSE NULL END\n                    ),\n                    last_validated_global_step = ?,\n                    consecutive_validation_failures = ?,\n                    validation_status = ?,\n                    updated_at = datetime('now')\n                WHERE component_signature = ?\n                """,\n                (\n                    int(coherent), int(coherent), int(coherent),\n                    None if row["first_seen_global_step"] is None else int(row["first_seen_global_step"]),\n                    None if row["first_seen_global_step"] is None else int(row["first_seen_global_step"]),\n                    failure_count, "passed" if coherent else status, component_signature,\n                ),\n            )\n            if demoted:\n''',
    )


def patch_h11() -> None:
    replace_once(
        H11,
        '''    full_handle = (\n        full_path.open("w", encoding="utf-8")\n        if write_full_provenance_jsonl\n        else None\n    )\n    try:\n        for row in links:\n            output, game_pair, context_pair = (\n                _chain_output_row(row)\n            )\n            fully_verified = _is_fully_verified(row)\n            _add_pair_row(\n                game_pairs[game_pair],\n                output,\n                fully_verified,\n            )\n            _add_pair_row(\n                context_pairs[context_pair],\n                output,\n                fully_verified,\n            )\n\n            source_context_id = output.get(\n                "source_context_id"\n            )\n            target_context_id = output.get(\n                "target_context_id"\n            )\n            if source_context_id:\n                context_lookup[str(source_context_id)] = str(\n                    row.get("source_context_key") or ""\n                )\n            if target_context_id:\n                context_lookup[str(target_context_id)] = str(\n                    row.get("target_context_key") or ""\n                )\n\n            if (\n                len(provenance_sample)\n                < provenance_sample_limit\n            ):\n                provenance_sample.append(output)\n            if full_handle is not None:\n                full_handle.write(\n                    json.dumps(output, sort_keys=True)\n                    + "\\n"\n                )\n    finally:\n        if full_handle is not None:\n            full_handle.close()\n''',
        '''    full_handle = (\n        full_path.open("w", encoding="utf-8")\n        if write_full_provenance_jsonl\n        else None\n    )\n    for row in links[:provenance_sample_limit]:\n        provenance_sample.append(_chain_output_row(row)[0])\n    # Preserve historical pair-artifact semantics on normal/small fixtures.\n    # On very large reports, the DB remains the complete evidence store and\n    # pair artifacts need only the fully verified chains used for validation.\n    artifact_rows = (\n        links\n        if full_handle is not None or len(links) <= 100_000\n        else fully_verified_links\n    )\n    try:\n        for row in artifact_rows:\n            output, game_pair, context_pair = _chain_output_row(row)\n            fully_verified = _is_fully_verified(row)\n            _add_pair_row(game_pairs[game_pair], output, fully_verified)\n            _add_pair_row(context_pairs[context_pair], output, fully_verified)\n            source_context_id = output.get("source_context_id")\n            target_context_id = output.get("target_context_id")\n            if source_context_id:\n                context_lookup[str(source_context_id)] = str(\n                    row.get("source_context_key") or ""\n                )\n            if target_context_id:\n                context_lookup[str(target_context_id)] = str(\n                    row.get("target_context_key") or ""\n                )\n            if full_handle is not None:\n                full_handle.write(json.dumps(output, sort_keys=True) + "\\n")\n    finally:\n        if full_handle is not None:\n            full_handle.close()\n''',
    )

    replace_once(
        SUITE,
        "    h11_write_full_provenance_jsonl: bool = True,\n",
        "    h11_write_full_provenance_jsonl: bool = False,\n",
    )
    replace_once(
        CONTINUOUS,
        "    h11_write_full_provenance_jsonl: bool = True\n",
        "    h11_write_full_provenance_jsonl: bool = False\n",
    )
    text = CLI.read_text(encoding="utf-8")
    old_flag = 'add_argument("--h11-write-full-provenance-jsonl", dest="h11_write_full_provenance_jsonl", action="store_true", default=True)'
    new_flag = 'add_argument("--h11-write-full-provenance-jsonl", dest="h11_write_full_provenance_jsonl", action="store_true", default=False)'
    count = text.count(old_flag)
    if count != 2:
        raise RuntimeError(f"expected two H11 CLI defaults, found {count}")
    CLI.write_text(text.replace(old_flag, new_flag), encoding="utf-8")


def main() -> None:
    patch_h08()
    patch_h11()


if __name__ == "__main__":
    main()
