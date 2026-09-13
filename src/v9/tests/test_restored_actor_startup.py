from __future__ import annotations

import multiprocessing as mp
import time

import pytest

from v9.curriculum import EnvironmentSpec
from v9.runtime.actor_policy import ActorPolicySnapshot
from v9.runtime.multiprocess import ActorDone, ProcessTopology, ensure_process_server_ready


@pytest.mark.skipif("forkserver" not in mp.get_all_start_methods(), reason="forkserver is Linux-specific")
def test_preinitialized_forkserver_serves_full_actor_fanout_after_parent_growth() -> None:
    # The production invariant is architectural: forkserver must exist before the
    # parent becomes large. Later actors must also use that same forkserver; they
    # must never use direct spawn/fork from the restored parent.
    assert ensure_process_server_ready("forkserver") == "forkserver"

    resident_state = bytearray(128 * 1024 * 1024)
    resident_state[0] = 1
    resident_state[-1] = 1

    topology = ProcessTopology(
        actors=30,
        stage_workers=2,
        shards=4,
        queue_capacity=256,
        start_method="forkserver",
    )
    assert topology.worker_start_method == "forkserver"
    assert topology.actor_start_method == "forkserver"

    topology.start_workers()
    try:
        policy = ActorPolicySnapshot.build(
            generation=1,
            normalized_action_supports={},
            hgt_action_scores={},
            model_version="test",
        )
        started = time.monotonic()
        for index in range(30):
            topology.start_actor(
                index=index,
                spec=EnvironmentSpec("synthetic_symbolic", "synthetic-symbolic"),
                actor_id=index + 1,
                steps=1,
                seed=index + 1,
                env_root=None,
                adapter_factory_path="v9.cli:make_adapter",
                alfred_backend_factory=None,
                run_nonce=1,
                initial_policy=policy,
                epsilon=0.0,
                policy_refresh_steps=64,
                policy_refresh_ms=250.0,
            )
        assert time.monotonic() - started < 20.0

        completed = set()
        deadline = time.monotonic() + 30.0
        while len(completed) < 30 and time.monotonic() < deadline:
            result = topology.result_queue.get(timeout=5.0)
            assert isinstance(result, ActorDone)
            completed.add(result.actor_id)
        assert completed == set(range(1, 31))
    finally:
        topology.terminate()
        del resident_state
