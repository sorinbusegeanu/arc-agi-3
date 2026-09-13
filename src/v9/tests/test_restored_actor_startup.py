from __future__ import annotations

import multiprocessing as mp
import time

import pytest

from v9.curriculum import EnvironmentSpec
from v9.runtime.actor_policy import ActorPolicySnapshot
from v9.runtime.multiprocess import ActorDone, ProcessTopology


@pytest.mark.skipif("forkserver" not in mp.get_all_start_methods(), reason="forkserver is Linux-specific")
def test_large_resident_parent_does_not_forkserver_launch_actor() -> None:
    # Reproduce the important restored-run shape without building a million-node
    # fixture: the parent already owns substantial resident state and fixed
    # forkserver workers are alive before an actor is launched.
    resident_state = bytearray(64 * 1024 * 1024)
    resident_state[0] = 1
    resident_state[-1] = 1

    topology = ProcessTopology(
        actors=1,
        stage_workers=1,
        shards=1,
        queue_capacity=64,
        start_method="forkserver",
    )
    assert topology.worker_start_method == "forkserver"
    assert topology.actor_start_method == "spawn"

    topology.start_workers()
    try:
        policy = ActorPolicySnapshot.build(
            generation=1,
            normalized_action_supports={},
            hgt_action_scores={},
            model_version="test",
        )
        started = time.monotonic()
        topology.start_actor(
            index=0,
            spec=EnvironmentSpec("synthetic_symbolic", "synthetic-symbolic"),
            actor_id=1,
            steps=1,
            seed=1,
            env_root=None,
            adapter_factory_path="v9.cli:make_adapter",
            alfred_backend_factory=None,
            run_nonce=1,
            initial_policy=policy,
            epsilon=0.0,
            policy_refresh_steps=64,
            policy_refresh_ms=250.0,
        )
        assert time.monotonic() - started < 10.0
        result = topology.result_queue.get(timeout=20.0)
        assert isinstance(result, ActorDone)
        assert result.actor_id == 1
        assert result.steps == 1
    finally:
        topology.terminate()
        del resident_state
