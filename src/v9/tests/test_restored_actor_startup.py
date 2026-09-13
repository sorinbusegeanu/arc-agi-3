from __future__ import annotations

import gc
import multiprocessing as mp
import os
import threading
import time

import pytest

from v9.curriculum import EnvironmentSpec
from v9.runtime import ContinuousMemoryRuntime, RuntimeConfig
from v9.runtime.actor_policy import ActorPolicySnapshot
from v9.runtime.multiprocess import ActorDone, ProcessTopology, ensure_process_server_ready
from v9.runtime.parallel_memory_coordinator import run_parallel_memory_jobs


@pytest.mark.skipif("forkserver" not in mp.get_all_start_methods(), reason="forkserver is Linux-specific")
def test_preinitialized_forkserver_serves_full_actor_fanout_after_parent_growth() -> None:
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
        topology.close(drain=False)
        del resident_state


def _fd_count() -> int:
    return len(os.listdir("/proc/self/fd"))


@pytest.mark.skipif("forkserver" not in mp.get_all_start_methods() or not os.path.isdir("/proc/self/fd"), reason="Linux forkserver regression")
def test_three_full_topology_lifecycles_do_not_leak_descriptors_or_feeder_threads(tmp_path) -> None:
    assert ensure_process_server_ready("forkserver") == "forkserver"
    runtime = ContinuousMemoryRuntime(RuntimeConfig.from_path(tmp_path, restore=False, enable_snapshots=False))
    spec = EnvironmentSpec("synthetic_symbolic", "synthetic-symbolic")

    fd_counts: list[int] = []
    thread_counts: list[int] = []
    try:
        for cycle in range(3):
            actor_base = cycle * 30
            jobs = [
                (actor_base + index + 1, spec, 1, actor_base + index + 1)
                for index in range(30)
            ]
            results = run_parallel_memory_jobs(
                runtime,
                jobs,
                actor_limit=30,
                stage_workers=2,
                shards=4,
                queue_capacity=256,
                epsilon=0.0,
                env_root=None,
                alfred_backend_factory=None,
                start_method="forkserver",
                progress_interval_seconds=60.0,
                ingest_workers=4,
                derivation_workers=4,
                ingest_queue_capacity=256,
                derivation_queue_capacity=256,
                publication_queue_capacity=1024,
            )
            assert len(results) == 30
            gc.collect()
            time.sleep(0.05)
            fd_counts.append(_fd_count())
            thread_counts.append(threading.active_count())

        assert max(fd_counts) - min(fd_counts) <= 8, fd_counts
        assert max(thread_counts) - min(thread_counts) <= 2, thread_counts
    finally:
        runtime.close(normal=False)


def test_public_runtime_has_no_canonical_record_ceiling(tmp_path) -> None:
    runtime = ContinuousMemoryRuntime(RuntimeConfig.from_path(tmp_path, restore=False, enable_snapshots=False))
    try:
        assert runtime.graph.node_capacity_per_partition is None
        assert runtime.graph.edge_capacity_per_partition is None
    finally:
        runtime.close(normal=False)
