from __future__ import annotations

import gc
from unittest.mock import patch
from weakref import ref

import v7.memory.developmental_policy as developmental_policy
from v7.memory.generation import GenerationId
from v7.memory.read_view import MemoryReadView


def _empty_view(generation: int = 0) -> MemoryReadView:
    return MemoryReadView.freeze(
        generation_id=GenerationId(generation),
        nodes={},
        scores={},
        adjacency={},
    )


def test_profile_for_view_scans_each_immutable_view_once() -> None:
    view = _empty_view()
    calls = 0
    original = developmental_policy.infer_development_stage

    def counted(candidate: MemoryReadView):
        nonlocal calls
        calls += 1
        return original(candidate)

    developmental_policy._PROFILE_CACHE_VIEW = None
    developmental_policy._PROFILE_CACHE_VALUE = None
    with patch.object(developmental_policy, "infer_development_stage", counted):
        first = developmental_policy.profile_for_view(view)
        for _ in range(20):
            assert developmental_policy.profile_for_view(view) is first

    assert calls == 1


def test_profile_cache_does_not_retain_old_generation_view() -> None:
    developmental_policy._PROFILE_CACHE_VIEW = None
    developmental_policy._PROFILE_CACHE_VALUE = None
    view = _empty_view()
    weak_view = ref(view)
    developmental_policy.profile_for_view(view)
    del view
    gc.collect()
    assert weak_view() is None
