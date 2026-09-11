from pathlib import Path

p = Path('src/v6/higher_order_substrate.py')
text = p.read_text()

old = '''def _prior_role_success_rate(
    transfer_rows: list[sqlite3.Row],
    *,
    role: str,
    before_step: int,
    source_game_key: str | None = None,
    source_context_key: str | None = None,
    target_game_key: str | None = None,
    target_context_key: str | None = None,
    transfer_history: _TransferHistoryIndex | None = None,
) -> tuple[float, int]:
    if transfer_history is not None:
        return transfer_history.rate_before(
            role=role,
            step=before_step,
            source_game_key=source_game_key,
            source_context_key=source_context_key,
            target_game_key=target_game_key,
            target_context_key=target_context_key,
        )
'''
new = '''def _prior_role_success_rate(
    transfer_rows: list[sqlite3.Row],
    *,
    role: str,
    before_step: int,
    source_game_key: str | None = None,
    source_context_key: str | None = None,
    target_game_key: str | None = None,
    target_context_key: str | None = None,
    transfer_history: _TransferHistoryIndex | None = None,
    rate_cache: dict[tuple[str, int, str, str, str, str], tuple[float, int]] | None = None,
) -> tuple[float, int]:
    if transfer_history is not None:
        cache_key = (role, int(before_step), source_game_key or "", source_context_key or "", target_game_key or "", target_context_key or "")
        if rate_cache is not None and cache_key in rate_cache:
            return rate_cache[cache_key]
        result = transfer_history.rate_before(
            role=role, step=before_step, source_game_key=source_game_key,
            source_context_key=source_context_key, target_game_key=target_game_key,
            target_context_key=target_context_key,
        )
        if rate_cache is not None:
            rate_cache[cache_key] = result
        return result
'''
assert old in text
text = text.replace(old, new, 1)

old = '''def _transfer_explanation_events(
    *,
    candidate_signature: str,
    source_roles: list[str],
    first_seen_global_step: int | None,
    transfer_rows: list[sqlite3.Row],
    transfer_history: _TransferHistoryIndex | None = None,
) -> list[dict[str, Any]]:
'''
new = '''def _transfer_explanation_events(
    *,
    candidate_signature: str,
    source_roles: list[str],
    first_seen_global_step: int | None,
    transfer_rows: list[sqlite3.Row],
    transfer_history: _TransferHistoryIndex | None = None,
    rate_cache: dict[tuple[str, int, str, str, str, str], tuple[float, int]] | None = None,
) -> list[dict[str, Any]]:
'''
assert old in text
text = text.replace(old, new, 1)
text = text.replace('transfer_history=transfer_history,\n            )[0]\n            for role in source_roles', 'transfer_history=transfer_history,\n                rate_cache=rate_cache,\n            )[0]\n            for role in source_roles', 1)
text = text.replace('target_context_key=target_context_key,\n                    transfer_history=transfer_history,\n                )', 'target_context_key=target_context_key,\n                    transfer_history=transfer_history,\n                    rate_cache=rate_cache,\n                )', 1)

old = '''def _future_option_motif_explanation_events(
    state_conn: sqlite3.Connection,
    *,
    candidate_signature: str,
    source_roles: list[str],
    first_seen_global_step: int | None,
    future_rows: list[sqlite3.Row],
) -> list[dict[str, Any]]:
'''
new = '''def _future_option_motif_explanation_events(
    state_conn: sqlite3.Connection,
    *,
    candidate_signature: str,
    source_roles: list[str],
    first_seen_global_step: int | None,
    future_rows: list[sqlite3.Row],
    role_rate_cache: dict[tuple[str, int], float] | None = None,
) -> list[dict[str, Any]]:
'''
assert old in text
text = text.replace(old, new, 1)
old = '''    role_rates: dict[str, float] = {}
    for role in source_roles:
        values = [
            1.0 if float(row["option_delta"] or 0.0) > 0.0 else 0.0
            for row in future_rows
            if (str(row["source_role_id"] or row["owner_key"] or "") == role)
            and row["last_seen_global_step"] is not None
            and int(row["last_seen_global_step"]) < first_seen_global_step
        ]
        role_rates[role] = sum(values) / len(values) if values else 0.0
'''
new = '''    role_rates: dict[str, float] = {}
    for role in source_roles:
        cache_key = (role, int(first_seen_global_step))
        if role_rate_cache is not None and cache_key in role_rate_cache:
            role_rates[role] = role_rate_cache[cache_key]
            continue
        values = [
            1.0 if float(row["option_delta"] or 0.0) > 0.0 else 0.0
            for row in future_rows
            if (str(row["source_role_id"] or row["owner_key"] or "") == role)
            and row["last_seen_global_step"] is not None
            and int(row["last_seen_global_step"]) < first_seen_global_step
        ]
        role_rates[role] = sum(values) / len(values) if values else 0.0
        if role_rate_cache is not None:
            role_rate_cache[cache_key] = role_rates[role]
'''
assert old in text
text = text.replace(old, new, 1)

for doc in ('Evaluate later lower-level predictions without candidate-concept features.', 'Evaluate held-out contradiction detection/resolution opportunities.'):
    marker = '    transfer_history: _TransferHistoryIndex | None = None,\n) -> list[dict[str, Any]]:\n    """' + doc
    repl = '    transfer_history: _TransferHistoryIndex | None = None,\n    rate_cache: dict[tuple[str, int, str, str, str, str], tuple[float, int]] | None = None,\n) -> list[dict[str, Any]]:\n    """' + doc
    assert marker in text
    text = text.replace(marker, repl, 1)
text = text.replace('transfer_history=transfer_history,\n            )[0]\n            for role in source_roles\n        ]\n        baseline = max(rates', 'transfer_history=transfer_history,\n                rate_cache=rate_cache,\n            )[0]\n            for role in source_roles\n        ]\n        baseline = max(rates', 1)
text = text.replace('transfer_history=transfer_history,\n            )[0]\n            for role in source_roles\n        ]\n        baseline = max(failure_rates', 'transfer_history=transfer_history,\n                rate_cache=rate_cache,\n            )[0]\n            for role in source_roles\n        ]\n        baseline = max(failure_rates', 1)

old = '''    config: IncrementalPromotionValidationConfig,
    candidate_links: dict[str, set[str]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
'''
new = '''    config: IncrementalPromotionValidationConfig,
    candidate_links: dict[str, set[str]] | None = None,
    role_links: dict[str, dict[str, set[str]]] | None = None,
    transfer_rate_cache: dict[tuple[str, int, str, str, str, str], tuple[float, int]] | None = None,
    future_role_rate_cache: dict[tuple[str, int], float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
'''
assert old in text
text = text.replace(old, new, 1)
text = text.replace('transfer_history=transfer_history,\n    )\n    events.extend(_future_option', 'transfer_history=transfer_history,\n        rate_cache=transfer_rate_cache,\n    )\n    events.extend(_future_option', 1)
text = text.replace('future_rows=future_rows,\n    ))\n    role_links = _links_by_signature(state_conn, "role_links", "role_signature")', 'future_rows=future_rows,\n        role_rate_cache=future_role_rate_cache,\n    ))\n    role_links = role_links or _links_by_signature(state_conn, "role_links", "role_signature")', 1)
text = text.replace('role_links=role_links,\n        transfer_history=transfer_history,\n    ))', 'role_links=role_links,\n        transfer_history=transfer_history,\n        rate_cache=transfer_rate_cache,\n    ))', 2)

old = '''    transfer_history = _build_transfer_history_index(transfer_rows)
    transfers_by_role: dict[str, list[sqlite3.Row]] = defaultdict(list)
'''
new = '''    transfer_history = _build_transfer_history_index(transfer_rows)
    transfer_rate_cache: dict[tuple[str, int, str, str, str, str], tuple[float, int]] = {}
    future_role_rate_cache: dict[tuple[str, int], float] = {}
    transfers_by_role: dict[str, list[sqlite3.Row]] = defaultdict(list)
'''
assert old in text
text = text.replace(old, new, 1)
old = '''                config=config,
                candidate_links=relevance_links_by_candidate.get(concept_signature, {}),
            )
'''
new = '''                config=config,
                candidate_links=relevance_links_by_candidate.get(concept_signature, {}),
                role_links=role_links,
                transfer_rate_cache=transfer_rate_cache,
                future_role_rate_cache=future_role_rate_cache,
            )
'''
assert old in text
text = text.replace(old, new, 1)

old = '''    contradiction_keys = {
        str(row["canonical_key"])
        for row in state_conn.execute("SELECT canonical_key FROM contradiction_clusters ORDER BY canonical_key ASC").fetchall()
    }
    promoted_rows = [row for row in concept_rows if int(row.get("is_promoted", 0) or 0) == 1]
'''
new = '''    contradiction_keys = {
        str(row["canonical_key"])
        for row in state_conn.execute("SELECT canonical_key FROM contradiction_clusters ORDER BY canonical_key ASC").fetchall()
    }
    family_support = {str(row["family_signature"]): int(row["support"] or 0) for row in state_conn.execute(
        "SELECT family_signature, COALESCE(SUM(support_count), 0) AS support FROM family_members GROUP BY family_signature").fetchall()}
    future_event_count_by_family = {str(row["source_family_id"]): int(row["event_count"] or 0) for row in state_conn.execute(
        "SELECT source_family_id, COUNT(*) AS event_count FROM future_option_events WHERE source_family_id IS NOT NULL GROUP BY source_family_id").fetchall()}
    family_prediction_gain = {str(row["canonical_signature"]): float(row["prediction_lift"] or 0.0) for row in state_conn.execute(
        "SELECT canonical_signature, prediction_lift FROM transformation_families").fetchall()}
    promoted_rows = [row for row in concept_rows if int(row.get("is_promoted", 0) or 0) == 1]
'''
assert old in text
text = text.replace(old, new, 1)
old = '''        for family in candidate_families:
            support_row = state_conn.execute(
                "SELECT COALESCE(SUM(support_count), 0) FROM family_members WHERE family_signature = ?",
                (family,),
            ).fetchone()
            role_count = sum(
                1 for role in roles if family in role_links.get(role, {}).get("family", set())
            )
            event_count = int(state_conn.execute(
                "SELECT COUNT(*) FROM future_option_events WHERE source_family_id = ?", (family,)
            ).fetchone()[0])
            prediction_gain_row = state_conn.execute(
                "SELECT AVG(prediction_lift) FROM transformation_families WHERE canonical_signature = ?", (family,)
            ).fetchone()
            prediction_gain = float(prediction_gain_row[0] or 0.0)
            family_candidates.append({
                "family": family,
                "support": int(support_row[0] or 0), "role_count": role_count,
'''
new = '''        for family in candidate_families:
            role_count = sum(
                1 for role in roles if family in role_links.get(role, {}).get("family", set())
            )
            event_count = future_event_count_by_family.get(family, 0)
            prediction_gain = family_prediction_gain.get(family, 0.0)
            family_candidates.append({
                "family": family,
                "support": family_support.get(family, 0), "role_count": role_count,
'''
assert old in text
text = text.replace(old, new, 1)

p.write_text(text)
