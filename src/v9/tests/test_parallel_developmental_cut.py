from __future__ import annotations

from v9 import ContinuousMemoryRuntime, RuntimeConfig
from v9.memory.model import MemoryLevel


def _runtime(tmp_path):
    return ContinuousMemoryRuntime(
        RuntimeConfig.from_path(
            tmp_path,
            restore=False,
            enable_snapshots=False,
            enable_canonical_durability=False,
        )
    )


def _seed_recurrent_signature(runtime):
    for index in range(2):
        runtime.submit(
            runtime.make_experience(
                producer_id=1,
                producer_sequence=index + 1,
                environment_instance_id=7,
                global_step=index,
                context_signature=3,
                action_id=2,
                outcome_signature=4,
                family_signature=5,
            )
        )
    signature = next(iter(runtime._m1n_occurrences))
    runtime._m1n_dirty.add(signature)
    return signature


def test_consolidation_retries_if_support_changes_after_cut(tmp_path, monkeypatch) -> None:
    runtime = _runtime(tmp_path)
    signature = _seed_recurrent_signature(runtime)
    original = runtime._prepare_dirty_rows_parallel
    injected = False

    def prepare_with_support_advance(snapshots):
        nonlocal injected
        rows = original(snapshots)
        if not injected:
            injected = True
            with runtime._lock:
                runtime._m1n_occurrences[signature].append(
                    runtime._m1n_occurrences[signature][-1]
                )
                support = len(runtime._m1n_occurrences[signature])
                runtime._m1n_supports[signature] = support
                runtime._m1n_dirty.add(signature)
                runtime.signature_index.mark_dirty(signature, support)
        return rows

    monkeypatch.setattr(runtime, "_prepare_dirty_rows_parallel", prepare_with_support_advance)
    runtime.flush_deferred_memory_updates()

    relation = runtime._m1n_occurrences[signature][0]
    assert runtime.graph.payloads[relation.uid]["support"] == runtime.signature_support(signature)
    assert runtime.signature_index.support(signature) == runtime.signature_support(signature)


def test_replay_retries_if_authoritative_cut_changes(tmp_path, monkeypatch) -> None:
    runtime = _runtime(tmp_path)
    _seed_recurrent_signature(runtime)
    runtime.flush_deferred_memory_updates()
    before_m0 = runtime.graph.memory_count(MemoryLevel.M0)
    original = runtime._derive_tasks_parallel
    calls = 0

    def derive_with_cut_change(tasks):
        nonlocal calls
        calls += 1
        rows = original(tasks)
        if calls == 1:
            with runtime._lock:
                runtime._watermark += 1
        return rows

    monkeypatch.setattr(runtime, "_derive_tasks_parallel", derive_with_cut_change)
    result = runtime.replay_once()

    assert calls >= 2
    assert result.selected >= 1
    assert runtime.graph.memory_count(MemoryLevel.M0) == before_m0
