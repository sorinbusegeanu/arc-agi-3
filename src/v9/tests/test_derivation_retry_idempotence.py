from __future__ import annotations

from v9.runtime.derivation_merge import DerivationLeaseManager, DerivationTaskIdentity


def test_duplicate_task_identity_is_not_enqueued_twice() -> None:
    manager = DerivationLeaseManager()
    identity = DerivationTaskIdentity(1, 2, 3)
    assert manager.enqueue(identity)
    assert not manager.enqueue(identity)
    lease = manager.lease()[0]
    assert not manager.enqueue(identity)
    assert manager.complete(lease, {"result": 1})
    assert not manager.enqueue(identity)
