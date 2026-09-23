from __future__ import annotations

import shutil

import pytest

from v9.runtime.canonical_store import CanonicalStore


def _advance(store: CanonicalStore, lsn: int, value: int) -> None:
    overlay = store.begin_overlay(lsn)
    overlay.put("graph", "node", value)
    store.finalize_overlay(overlay)


def test_snapshot_publication_crash_preserves_previous_complete_generation(tmp_path) -> None:
    path = tmp_path / "snapshot"
    store = CanonicalStore()
    _advance(store, 1, 1)
    store.write_snapshot(path)
    _advance(store, 2, 2)

    def crash(stage: str) -> None:
        if stage == "previous_preserved":
            raise RuntimeError("injected snapshot publication crash")

    with pytest.raises(RuntimeError, match="injected snapshot publication crash"):
        store.write_snapshot(path, crash_hook=crash)

    assert not path.exists()
    restored = CanonicalStore.from_snapshot(path)
    assert restored.current_handle.canonical_applied_lsn == 1
    assert restored.read_from_handle(restored.current_handle, "graph", "node") == 1


def test_snapshot_restore_prefers_new_complete_generation_after_publish_crash(tmp_path) -> None:
    path = tmp_path / "snapshot"
    store = CanonicalStore()
    _advance(store, 1, 1)
    store.write_snapshot(path)
    _advance(store, 2, 2)

    def crash(stage: str) -> None:
        if stage == "snapshot_published":
            raise RuntimeError("injected post-publication crash")

    with pytest.raises(RuntimeError, match="injected post-publication crash"):
        store.write_snapshot(path, crash_hook=crash)

    restored = CanonicalStore.from_snapshot(path)
    assert restored.current_handle.canonical_applied_lsn == 2
    assert restored.read_from_handle(restored.current_handle, "graph", "node") == 2

    previous = path.with_name(f".{path.name}.previous")
    shutil.rmtree(previous)
