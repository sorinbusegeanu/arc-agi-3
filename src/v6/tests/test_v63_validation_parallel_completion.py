from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from v6.higher_order_substrate import IncrementalPromotionValidationConfig
from v6.memory.compact_memory import ensure_memory_layout
from v6.memory import v63_validation_parallel_completion as completion


def test_validation_workers_are_resolved_from_current_epoch_report(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    ensure_memory_layout(memory_dir)
    raw_dir = tmp_path / "epochs" / "epoch_0002" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "interaction_sampling_v05c_report.json").write_text(
        json.dumps({"validation_workers_requested": 12}),
        encoding="utf-8",
    )

    assert completion._resolve_validation_workers(memory_dir) == 12
    assert completion._resolve_validation_workers(memory_dir, 99) == 16


def test_parallel_precompute_builds_each_concept_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = ensure_memory_layout(tmp_path / "memory")
    with sqlite3.connect(paths.current_state) as connection:
        connection.executemany(
            """
            INSERT INTO concept_candidates (concept_signature, first_seen_global_step)
            VALUES (?, ?)
            """,
            [("concept:a", 10), ("concept:b", 20), ("concept:c", 30)],
        )
        connection.commit()

    calls: list[str] = []

    def fake_build(**kwargs):
        signature = str(kwargs["candidate_signature"])
        calls.append(signature)
        return (
            [{"concept_id": signature, "event_id": f"event:{signature}"}],
            {"relevant_incremental_coverage": 0.5, "signature": signature},
            {"structure_fingerprint": signature, "relevant_event_ids": []},
        )

    monkeypatch.setattr(completion, "_ORIGINAL_BUILD_FUNCTIONAL", fake_build)
    cache = completion._precompute_functional_diagnostics(
        memory_dir=paths.root,
        config=IncrementalPromotionValidationConfig(enabled=True),
        diagnostic_epoch_id="epoch_0001",
        workers=3,
    )

    assert sorted(calls) == ["concept:a", "concept:b", "concept:c"]
    assert len(cache) == 3
    assert sorted(key[0] for key in cache) == ["concept:a", "concept:b", "concept:c"]


def test_cached_functional_diagnostics_are_reused_without_shared_mutation(
    monkeypatch,
) -> None:
    key = ("concept:a", 10, ("role:a",))
    cached = (
        [{"concept_id": "concept:a", "event_id": "event:1"}],
        {"relevant_incremental_coverage": 0.5},
        {"structure_fingerprint": "abc"},
    )
    monkeypatch.setattr(completion, "_ACTIVE_FUNCTIONAL_CACHE", {key: cached})
    monkeypatch.setattr(
        completion,
        "_ORIGINAL_BUILD_FUNCTIONAL",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("cache was not used")),
    )

    first = completion._build_functional_from_cache(
        state_conn=None,
        candidate_signature="concept:a",
        source_roles=["role:a"],
        first_seen_global_step=10,
        transfer_rows=[],
        transfer_history=None,
        future_rows=[],
        previous_state=None,
        diagnostic_epoch_id="epoch_0001",
        config=IncrementalPromotionValidationConfig(enabled=True),
        candidate_links={},
    )
    first[0][0]["event_id"] = "changed"
    second = completion._build_functional_from_cache(
        state_conn=None,
        candidate_signature="concept:a",
        source_roles=["role:a"],
        first_seen_global_step=10,
        transfer_rows=[],
        transfer_history=None,
        future_rows=[],
        previous_state=None,
        diagnostic_epoch_id="epoch_0001",
        config=IncrementalPromotionValidationConfig(enabled=True),
        candidate_links={},
    )

    assert second[0][0]["event_id"] == "event:1"


def test_installer_patches_suite_validator_binding() -> None:
    completion.install_v63_validation_parallel_completion()
    from v6 import higher_order_substrate as substrate
    from v6 import hypothesis_suite_report as suite

    assert substrate.validate_incremental_promotions_only is completion._validate_incremental_promotions_only_parallel
    assert suite.validate_incremental_promotions_only is completion._validate_incremental_promotions_only_parallel
