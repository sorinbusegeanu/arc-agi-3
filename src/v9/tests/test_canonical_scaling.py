from types import SimpleNamespace

from v9.memory.identity import MemoryUid
from v9.runtime.canonical_commit_derivation import derivation_candidates
from v9.runtime.actor_production_telemetry import _CountingStageQueue


class _NoFullScan(dict):
    def values(self):
        raise AssertionError("derivation candidate lookup scanned all M1N occurrences")


def _row(uid_low: int, evidence_low: int, *, signature: int, family: int):
    return SimpleNamespace(
        uid=MemoryUid(1, uid_low),
        channel=SimpleNamespace(value="WORLD"),
        provenance=SimpleNamespace(evidence=(MemoryUid(2, evidence_low),)),
        family_signature=family,
        structural_signature=signature,
        support=1.0,
        contradiction=0.0,
    )


def test_derivation_candidates_use_touched_family_index_only() -> None:
    first = _row(10, 100, signature=11, family=77)
    second = _row(20, 200, signature=12, family=77)
    unrelated = _row(30, 300, signature=99, family=88)
    runtime = SimpleNamespace(
        _m1n_occurrences=_NoFullScan({11: [first], 99: [unrelated]}),
        _m1n_family_occurrences={77: [first, second], 88: [unrelated]},
        graph=SimpleNamespace(
            payloads={
                MemoryUid(2, 100): {"environment_instance_id": 1, "evidence_confidence": 1.0},
                MemoryUid(2, 200): {"environment_instance_id": 2, "evidence_confidence": 0.5},
            }
        ),
        _deferred_base_nodes={},
        _formation_environments=set(),
        _watermark=123,
    )

    candidates = derivation_candidates(runtime, {11})

    assert len(candidates) == 1
    assert candidates[0].structural_signature == 77
    assert candidates[0].support == 2
    assert candidates[0].environment_scope == (1, 2)
    assert candidates[0].evidence_confidence == 0.75


def test_actor_production_counter_increments_after_queue_accepts() -> None:
    class Queue:
        def __init__(self) -> None:
            self.rows = []

        def put(self, item, *args, **kwargs):
            self.rows.append(item)

    counters = [5]
    queue = Queue()
    wrapped = _CountingStageQueue(queue, counters, 0)

    wrapped.put("transition")

    assert queue.rows == ["transition"]
    assert counters[0] == 6
