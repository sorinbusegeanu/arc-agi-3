from __future__ import annotations

import pytest

from v9.memory.identity import MemoryUid
from v9.memory.m1_normalized import M1NormalizedRelation, NormalizedChannel
from v9.memory.model import MemoryLevel, MemoryType
from v9.memory.provenance import DerivationProvenance
from v9.runtime.signature_index import SignatureIndexStore


def test_dirty_state_and_scan_cursor_survive_restart(tmp_path) -> None:
    path = tmp_path / "signatures.sqlite"
    with SignatureIndexStore(path) as store:
        for signature in (10, 20, 30):
            store.observe(signature)
        assert tuple(row.signature for row in store.scan_page(limit=2)) == (10, 20)
    with SignatureIndexStore(path) as restored:
        assert restored.scan_cursor == 20
        assert {row.signature for row in restored.dirty_window()} == {10, 20, 30}
        restored.mark_derived(20, 1)
    with SignatureIndexStore(path) as restored_again:
        assert not restored_again.get(20).derivation_dirty


def test_runtime_dirty_signature_authority_survives_restart(tmp_path) -> None:
    from v9.runtime import ContinuousMemoryRuntime, RuntimeConfig

    root = tmp_path / "runtime"
    first = ContinuousMemoryRuntime(
        RuntimeConfig.from_path(root, restore=False, enable_snapshots=False)
    )
    first._m1n_supports[77] = 3
    first._m1n_dirty.add(77)
    assert 77 in first._m1n_dirty
    first.close(normal=False)

    restored = ContinuousMemoryRuntime(
        RuntimeConfig.from_path(root, restore=False, enable_snapshots=False)
    )
    try:
        assert 77 in restored._m1n_dirty
        assert len(restored._m1n_dirty) == 1
        assert restored.signature_support(77) == 3
    finally:
        restored.close(normal=False)


def test_failed_dirty_publication_remains_retryable(tmp_path, monkeypatch) -> None:
    from v9.runtime import ContinuousMemoryRuntime, RuntimeConfig

    runtime = ContinuousMemoryRuntime(
        RuntimeConfig.from_path(tmp_path / "runtime", restore=False, enable_snapshots=False)
    )
    signature = 91
    uid = MemoryUid.from_key(MemoryLevel.M1, MemoryType.NORMALIZED_RELATION, (signature,))
    evidence = MemoryUid.from_key(MemoryLevel.M0, MemoryType.EPISODE, (1,))
    runtime._m1n_occurrences[signature] = [
        M1NormalizedRelation(
            uid,
            "retryable",
            NormalizedChannel.WORLD,
            signature,
            DerivationProvenance((evidence,), (evidence,)),
        )
    ]
    runtime._m1n_supports[signature] = 2
    runtime._m1n_dirty.add(signature)
    monkeypatch.setattr(
        runtime,
        "_publish",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected publication failure")),
    )
    try:
        with pytest.raises(RuntimeError, match="injected publication failure"):
            runtime.flush_deferred_memory_updates()
        assert signature in runtime._m1n_dirty
        assert runtime.signature_support(signature) == 2
    finally:
        runtime.close(normal=False)
