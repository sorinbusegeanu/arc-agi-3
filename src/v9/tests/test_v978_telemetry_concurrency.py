from __future__ import annotations

from threading import RLock
from types import SimpleNamespace

from v9.memory.identity import MemoryUid
from v9.runtime.v978_conformant import V978ContinuousMemoryRuntime


class _NoLiveItemsDict(dict):
    def items(self):
        raise AssertionError("live evidence dictionary iteration is unsafe for telemetry")


def test_patch_m1n_evidence_uses_stable_snapshot() -> None:
    runtime = object.__new__(V978ContinuousMemoryRuntime)
    uid = MemoryUid(1, 2)
    runtime._v978_evidence_lock = RLock()
    runtime._v978_m1n_evidence = _NoLiveItemsDict(
        {
            7: {
                "uid": uid,
                "support": 3.0,
                "contradiction": 1.0,
                "offsets": {-1, 0, 2},
                "causal_watermark": 11,
                "family_signature": 17,
                "context_signature": 19,
            }
        }
    )
    runtime.graph = SimpleNamespace(payloads={uid: {}})
    runtime._deferred_base_nodes = {}

    runtime._patch_m1n_evidence()

    payload = runtime.graph.payloads[uid]
    assert payload["support"] == 3.0
    assert payload["contradiction"] == 1.0
    assert payload["temporal_offsets"] == [-1, 0, 2]
    assert payload["causal_watermark"] == 11
    assert payload["family_signature"] == 17
    assert payload["context_signature"] == 19
