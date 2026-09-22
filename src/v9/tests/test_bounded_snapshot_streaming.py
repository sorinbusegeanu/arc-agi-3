from __future__ import annotations

from pathlib import Path
import pickle
from types import SimpleNamespace

from v9.memory.identity import MemoryUid
from v9.memory.model import CanonicalNode, MemoryLevel, MemoryType
from v9.mutation.proposals import MutationKind, MutationProposal, MutationWrite
from v9.mutation.read_sets import ReadSet
from v9.runtime import ContinuousMemoryRuntime, RuntimeConfig
from v9.runtime.canonical_store import CanonicalStore
from v9.runtime.snapshot_backend import load_snapshot_direct


def test_runtime_snapshot_avoids_full_graph_partition_materialization(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = ContinuousMemoryRuntime(
        RuntimeConfig.from_path(
            tmp_path / "run",
            restore=False,
            enable_snapshots=True,
            enable_canonical_durability=False,
        )
    )
    try:
        for index in range(8):
            runtime.submit(
                runtime.make_experience(
                    producer_id=1,
                    producer_sequence=index + 1,
                    environment_instance_id=7,
                    global_step=index,
                    context_signature=index % 2,
                    action_id=index % 3,
                    outcome_signature=index + 10,
                    family_signature=5,
                )
            )
        runtime.wait_quiescent()
        runtime.flush_deferred_memory_updates()

        def fail_partition_state(*_args, **_kwargs):
            raise AssertionError("snapshot materialized a complete graph partition")

        def fail_pickle_dumps(*_args, **_kwargs):
            raise AssertionError("snapshot materialized the complete runtime pickle")

        monkeypatch.setattr(runtime.graph, "_partition_state", fail_partition_state)
        monkeypatch.setattr(
            "v9.runtime.runtime_integrity.pickle",
            SimpleNamespace(
                dump=pickle.dump,
                loads=pickle.loads,
                dumps=fail_pickle_dumps,
            ),
        )

        result = runtime.snapshot()
        assert (result.path / "COMPLETE").is_file()
        state, header, shards = load_snapshot_direct(
            result.path,
            expected_config_id=runtime.config.scientific.config_id.value,
        )
        assert int(state["watermark"]) == int(runtime.watermark)
        assert int(header["generation"]) == int(runtime.graph.generation)
        assert len(shards) == int(runtime.graph.partition_count)
        assert runtime.unified_telemetry.gauges["snapshot_in_progress"] == 0
    finally:
        runtime.close(normal=False)


def test_canonical_snapshot_streams_pickle_without_full_state_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    store = CanonicalStore()
    overlay = store.begin_overlay(1)
    overlay.put("graph", "node", {"value": 7})
    store.finalize_overlay(overlay)

    def fail_pickle_dumps(*_args, **_kwargs):
        raise AssertionError("canonical snapshot created a complete pickle byte copy")

    monkeypatch.setattr("v9.runtime.canonical_store.pickle.dumps", fail_pickle_dumps)
    path = store.write_snapshot(tmp_path / "canonical")
    restored = CanonicalStore.from_snapshot(path)
    assert restored.read_from_handle(
        restored.current_handle, "graph", "node"
    ) == {"value": 7}
