from pathlib import Path

SOURCE = Path("src/v6/higher_order_substrate.py")
TEST = Path("src/v6/tests/test_v6_higher_order.py")

text = SOURCE.read_text(encoding="utf-8")

old = '''        "source_context_game_keys_json": json.dumps(source_context_game_keys, sort_keys=True),
        "provenance_mode": provenance_mode,
'''
new = '''        "source_context_game_keys_json": json.dumps(source_context_game_keys, sort_keys=True),
        # Ephemeral expansion evidence: when a context occurs in multiple games,
        # concrete cross-context provenance must bind both sides to one shared game.
        # This key is consumed before persistence and is intentionally not part of
        # the role_transfer_attempts schema.
        "target_game_keys_json": json.dumps(target_games),
        "provenance_mode": provenance_mode,
'''
assert old in text, "target game evidence insertion point not found"
text = text.replace(old, new, 1)

old = '''    target_games = (str(attempt["target_game_key"]),) if attempt.get("target_game_key") else ()
    result: list[dict[str, Any]] = []
'''
new = '''    target_games = tuple(
        str(value)
        for value in json.loads(str(attempt.get("target_game_keys_json") or "[]"))
        if value not in (None, "")
    )
    if not target_games and attempt.get("target_game_key"):
        target_games = (str(attempt["target_game_key"]),)
    result: list[dict[str, Any]] = []
'''
assert old in text, "target game expansion block not found"
text = text.replace(old, new, 1)

old = '''            else:
                concrete["source_scope_type"] = "context"
                concrete["source_scope_key"] = source_key
                concrete["source_context_key"] = source_key
                candidate_games = tuple(sorted(str(value) for value in source_context_game_keys.get(source_key, ())))
                target_game = str(concrete.get("target_game_key") or "")
                concrete["source_game_key"] = (
                    target_game if target_game and target_game in candidate_games
                    else candidate_games[0] if candidate_games
                    else source_games[0] if len(source_games) == 1 else None
                )
'''
new = '''            else:
                concrete["source_scope_type"] = "context"
                concrete["source_scope_key"] = source_key
                concrete["source_context_key"] = source_key
                candidate_games = tuple(sorted(str(value) for value in source_context_game_keys.get(source_key, ())))
                # A cross-context claim is only valid within one real game.
                # Select a deterministic game observed for both the source and
                # target contexts.  Cross-game evidence is generated separately
                # by the cross_game population and must never leak into this one.
                shared_games = tuple(sorted(set(candidate_games) & set(target_games)))
                if not shared_games:
                    continue
                shared_game = shared_games[0]
                concrete["source_game_key"] = shared_game
                concrete["target_game_key"] = shared_game
'''
assert old in text, "cross-context concrete game block not found"
text = text.replace(old, new, 1)

old = '''    real_cross_context = not source["context_is_surrogate"] and not target["context_is_surrogate"] and source["context_key"] != target["context_key"]
    if str(attempt.get("provenance_mode")) == "single_source" and (real_cross_game or real_cross_context):
        attempt["provenance_status"] = "verified"
'''
new = '''    real_cross_context = not source["context_is_surrogate"] and not target["context_is_surrogate"] and source["context_key"] != target["context_key"]
    transfer_kind = str(attempt.get("transfer_kind") or "")
    kind_matches_resolved_scope = (
        real_cross_game
        if transfer_kind == "cross_game"
        else (real_cross_context and not real_cross_game)
        if transfer_kind == "cross_context"
        else False
    )
    if str(attempt.get("provenance_mode")) == "single_source" and kind_matches_resolved_scope:
        attempt["provenance_status"] = "verified"
'''
assert old in text, "resolved provenance verification block not found"
text = text.replace(old, new, 1)

old = '''    if str(attempt.get("provenance_status")) == "verified" and not (
        mode == "single_source" and (real_cross_game or real_cross_context)
    ):
        raise ValueError("verified transfer attempt lacks a real cross-scope dimension")
'''
new = '''    if kind == "cross_game" and not bool(attempt.get("source_game_is_surrogate")) and not bool(attempt.get("target_game_is_surrogate")) and not real_cross_game:
        raise ValueError("cross-game transfer attempt does not span distinct real games")
    if kind == "cross_context" and real_cross_game:
        raise ValueError("cross-context transfer attempt spans distinct real games")
    kind_matches_resolved_scope = real_cross_game if kind == "cross_game" else (real_cross_context and not real_cross_game)
    if str(attempt.get("provenance_status")) == "verified" and not (
        mode == "single_source" and kind_matches_resolved_scope
    ):
        raise ValueError("verified transfer attempt lacks the requested real cross-scope dimension")
'''
assert old in text, "persistence provenance validation block not found"
text = text.replace(old, new, 1)

SOURCE.write_text(text, encoding="utf-8")

test_text = TEST.read_text(encoding="utf-8")
marker = "test_cross_context_expansion_binds_source_and_target_to_same_real_game"
if marker not in test_text:
    test_text += '''\n\ndef test_cross_context_expansion_binds_source_and_target_to_same_real_game() -> None:\n    attempt = {\n        "attempt_id": "aggregate",\n        "role_signature": "role-target",\n        "transfer_kind": "cross_context",\n        "source_scope_type": "context",\n        "source_scope_key": None,\n        "target_scope_type": "context",\n        "target_scope_key": "ctx-target",\n        "source_game_key": None,\n        "target_game_key": None,\n        "source_context_key": None,\n        "target_context_key": "ctx-target",\n        "source_carrier_signature": None,\n        "source_role_signature": "role-source",\n        "predicted_target_role_signature": "role-source",\n        "observed_target_role_signature": "role-target",\n        "source_carrier_signatures_json": json.dumps(["carrier-source"]),\n        "source_game_keys_json": json.dumps(["game-a", "game-b"]),\n        "source_context_keys_json": json.dumps(["ctx-source"]),\n        "source_context_game_keys_json": json.dumps({"ctx-source": ["game-a", "game-b"]}),\n        "target_game_keys_json": json.dumps(["game-b", "game-c"]),\n        "provenance_mode": "single_source",\n        "provenance_status": "verified",\n        "target_carrier_signature": "carrier-target",\n        "predicted_role_signature": "role-source",\n        "observed_role_signature": "role-target",\n        "similarity_score": 1.0,\n        "transfer_score": 1.0,\n        "reuse_success": 1,\n        "failure_reason": "success",\n        "best_margin": 0.2,\n        "source_carrier_count": 1,\n        "source_evidence_support_count": 2,\n        "support_gate_passed": 1,\n        "similarity_gate_passed": 1,\n        "role_match_gate_passed": 1,\n        "candidate_role_count": 2,\n        "first_seen_global_step": 1,\n        "last_seen_global_step": 2,\n    }\n    rows = higher_order_substrate._expand_transfer_attempt_provenance(attempt)\n    assert len(rows) == 1\n    assert rows[0]["source_game_key"] == "game-b"\n    assert rows[0]["target_game_key"] == "game-b"\n\n\ndef test_cross_context_persistence_rejects_distinct_real_games() -> None:\n    attempt = {\n        "provenance_mode": "single_source",\n        "transfer_kind": "cross_context",\n        "source_interaction_id": "i1",\n        "target_interaction_id": "i2",\n        "source_game_key": "game-a",\n        "target_game_key": "game-b",\n        "source_context_key": "ctx-a",\n        "target_context_key": "ctx-b",\n        "source_game_is_surrogate": 0,\n        "target_game_is_surrogate": 0,\n        "source_context_is_surrogate": 0,\n        "target_context_is_surrogate": 0,\n        "source_game_resolution_source": "direct_attempt",\n        "target_game_resolution_source": "direct_attempt",\n        "source_context_resolution_source": "direct_attempt",\n        "target_context_resolution_source": "direct_attempt",\n        "provenance_status": "verified",\n        "reuse_success": 1,\n        "source_role_signature": "role-source",\n        "source_carrier_signature": "carrier-source",\n    }\n    with pytest.raises(ValueError, match="spans distinct real games"):\n        higher_order_substrate._validate_transfer_attempt_provenance(attempt)\n'''
    TEST.write_text(test_text, encoding="utf-8")
