from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

from v6 import concept_validation_fastpath as fast
from v6 import concept_validation_profiler_context_fix as fix
from v6 import concept_validation_sparse_cache as sparse


def test_active_profiler_context_is_reused(monkeypatch, tmp_path) -> None:
    ctx = {
        "cache": {},
        "role_score_cache": {},
        "generic_role_score_cache": {},
        "scoped_role_score_cache": {},
        "timings": {},
        "call_counts": {},
        "event_counts": {},
        "index_stats": {},
        "cache_stats": {},
        "per_concept": {},
    }

    monkeypatch.setattr(fast, "_evidence_frontier", lambda memory_dir: {"concept_candidates": 1})
    monkeypatch.setattr(fast, "_load_frontier", lambda memory_dir: None)
    monkeypatch.setattr(fast, "_store_frontier", lambda memory_dir, payload: None)

    def base_validate(*args, **kwargs):
        assert fast._ACTIVE.get() is ctx
        ctx["per_concept"]["concept-a"] = {"calls": 1, "seconds": 0.25}
        ctx["timings"]["functional_diagnostics.total"] = 0.25
        ctx["call_counts"]["functional_diagnostics"] = 1
        ctx["event_counts"]["transfer"] += 3
        return {"concept_candidate_count": 1}

    monkeypatch.setitem(fast._ORIGINALS, "validate_incremental_promotions_only", base_validate)

    token = fast._ACTIVE.set(ctx)
    try:
        result = fix._validate_with_active_profile(
            memory_dir=tmp_path,
            config=SimpleNamespace(enabled=True),
            validate_roles_and_concepts=True,
            validate_world_models=False,
        )
        assert fast._ACTIVE.get() is ctx
    finally:
        fast._ACTIVE.reset(token)

    profile = result["concept_validation_fastpath_profile"]
    assert ctx["per_concept"]["concept-a"]["calls"] == 1
    assert profile["call_counts"]["functional_diagnostics"] == 1
    assert profile["event_counts"]["transfer"] == 3
    assert profile["timings"]["functional_diagnostics.total"] == 0.25


def test_sparse_profiler_reports_concepts_through_context_fix(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(fast, "_evidence_frontier", lambda memory_dir: {"concept_candidates": 1})
    monkeypatch.setattr(fast, "_load_frontier", lambda memory_dir: None)
    monkeypatch.setattr(fast, "_store_frontier", lambda memory_dir, payload: None)

    def functional_diagnostics(*args, **kwargs):
        return ([{"event_type": "transfer", "is_relevant": True, "invalid": False}], {})

    monkeypatch.setattr(sparse, "_ORIGINAL_FUNCTIONAL_DIAGNOSTICS", functional_diagnostics)

    def base_validate(*args, **kwargs):
        sparse._profiled_functional_diagnostics(candidate_signature="concept-a")
        return {"concept_candidate_count": 1}

    monkeypatch.setitem(fast._ORIGINALS, "validate_incremental_promotions_only", base_validate)
    monkeypatch.setattr(sparse, "_ORIGINAL", fix._validate_with_active_profile)

    result = sparse._profiled_validate(
        memory_dir=tmp_path,
        config=SimpleNamespace(enabled=True),
        validate_roles_and_concepts=True,
        validate_world_models=False,
    )

    profile = result["concept_validation_fastpath_profile"]
    assert profile["worker"]["concept_count"] == 1
    assert profile["per_concept"]["concept-a"]["calls"] == 1
    assert profile["per_concept"]["concept-a"]["transfer_rows_examined"] == 1
    assert profile["per_concept"]["concept-a"]["transfer_rows_accepted"] == 1
