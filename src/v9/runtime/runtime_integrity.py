from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import json
import os
import pickle
from pathlib import Path
import shutil
import time
import weakref
from multiprocessing import resource_tracker, shared_memory
from typing import Any, Iterable

from v9.cognition.grounding import GroundingRegistry
from v9.memory.model import MemoryLevel
from v9.runtime.publication import edge_ref, node_ref


_GROUNDING_STATE_LIMIT = 65_536
_GROUNDING_TRIAL_ID_LIMIT = 64
_VIABILITY_PROFILE_LIMIT = 4_096
_ENVIRONMENTS_PER_GAME_LIMIT = 256


@dataclass(frozen=True, slots=True)
class V9716MigrationStatus:
    """Feature-gated progress marker; it never changes scientific behavior."""

    target_design_version: str
    active_design_version: str
    capabilities: tuple[tuple[str, bool], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "target_design_version": self.target_design_version,
            "active_design_version": self.active_design_version,
            "capabilities": dict(self.capabilities),
        }


def v9716_migration_status() -> V9716MigrationStatus:
    capabilities = (
        ("scientific_identity_and_modes", True),
        ("immutable_canonical_store", True),
        ("canonical_wal", True),
        ("persistent_signature_index", True),
        ("transport_slab_pool", True),
        ("canonical_work_bounds", True),
        ("developmental_cut", True),
        ("epoch_inference_view", True),
        ("wal_backed_training_evidence", True),
        ("deterministic_training_cut", True),
        ("h17_isolation", True),
        ("fixed_h19_trials", True),
        ("h18_structural_prior_transform", False),
        ("h16_grounding_controls", True),
        ("research_prediction_registry", True),
        ("durable_storage_governance", True),
        ("whole_system_governor", True),
        ("dual_mode_restart", True),
        ("crash_reproducibility_soak", False),
        ("legacy_cleanup_and_cutover", False),
    )
    return V9716MigrationStatus("9.7.16", "9.7.9", capabilities)


class _ExperimentStateHandle:
    __slots__ = ("path", "_finalizer", "__weakref__")

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._finalizer = weakref.finalize(self, shutil.rmtree, self.path, True)


def _graph_header(graph: Any) -> dict[str, Any]:
    return {
        "schema_version": int(graph.SCHEMA_VERSION),
        "partition_count": int(graph.partition_count),
        "node_capacity_per_partition": graph.node_capacity_per_partition,
        "edge_capacity_per_partition": graph.edge_capacity_per_partition,
        "applied_proposal_capacity": int(graph.applied_proposal_capacity),
        "generation": int(graph.generation),
        "retired_tombstones": [],
        "versions": graph.versions.state_dict(),
        "applied_proposals": list(graph._applied_order),
        "low_level_nodes_inserted_total": int(getattr(graph, "low_level_nodes_inserted_total", 0)),
        "low_level_nodes_deleted_total": int(getattr(graph, "low_level_nodes_deleted_total", 0)),
        "low_level_edges_deleted_total": int(getattr(graph, "low_level_edges_deleted_total", 0)),
        "low_level_delete_batches_total": int(getattr(graph, "low_level_delete_batches_total", 0)),
    }


def _runtime_state_without_materialized_graph(runtime: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    graph = runtime.graph
    header = _graph_header(graph)
    had_override = "state_dict" in graph.__dict__
    previous = graph.__dict__.get("state_dict")
    graph.__dict__["state_dict"] = lambda: dict(header)
    try:
        state = runtime.state_dict()
    finally:
        if had_override:
            graph.__dict__["state_dict"] = previous
        else:
            graph.__dict__.pop("state_dict", None)
    state["graph"] = dict(header)
    return state, header


def _dump_pickle_file(path: Path, value: Any) -> None:
    with Path(path).open("wb") as stream:
        pickle.dump(value, stream, protocol=5)
        stream.flush()


def _write_experiment_cut(runtime: Any, target: Path, chunked_snapshot: Any) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    capture_dir = temporary / ".capture"
    capture_dir.mkdir()
    try:
        with runtime._lock, runtime.graph._publication_lock:
            state, header = _runtime_state_without_materialized_graph(runtime)
            _dump_pickle_file(temporary / "runtime.pkl", state)
            _dump_pickle_file(temporary / "graph_header.pkl", header)
            del state, header
            captures = tuple(
                chunked_snapshot.capture_graph_shard(
                    runtime.graph, partition, capture_dir
                )
                for partition in range(int(runtime.graph.partition_count))
            )
        for capture in captures:
            chunked_snapshot.write_graph_shard_capture_file(
                capture,
                temporary / f"graph-{int(capture.partition):04d}.bin",
            )
        shutil.rmtree(capture_dir, ignore_errors=True)
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)

def _restore_experiment_cut(runtime: Any, handle: _ExperimentStateHandle, graph_cls: type, chunked_snapshot: Any) -> None:
    path = handle.path
    state = pickle.loads((path / "runtime.pkl").read_bytes())
    header = pickle.loads((path / "graph_header.pkl").read_bytes())
    graph = graph_cls.from_sharded_state(header, ())
    for partition in range(int(graph.partition_count)):
        payload = (path / f"graph-{partition:04d}.bin").read_bytes()
        decoded = chunked_snapshot._decode_graph_shard(payload, expected_partition=partition)
        graph.install_sharded_rows((decoded,))
        del payload, decoded
    runtime._restore({"state": state}, graph_override=graph)


def _write_streaming_snapshot(runtime: Any, chunked_snapshot: Any) -> Any:
    from v9.runtime.snapshot_backend import SYMBOL_GRAPH_SCHEMA_VERSION

    runtime.wait_quiescent()
    runtime.flush_deferred_memory_updates()
    runtime.evidence.flush()

    root = Path(runtime.root)
    snapshots = root / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    runtime.set_telemetry_gauge("snapshot_in_progress", 1)
    temporary: Path | None = None
    snapshot_id = 0
    try:
        with runtime._lock, runtime.graph._publication_lock:
            runtime._snapshot_id += 1
            runtime.telemetry["snapshot_writes"] += 1
            snapshot_id = int(runtime._snapshot_id)
            watermark = int(runtime._watermark)
            generation = int(runtime.graph.generation)
            target = snapshots / f"snapshot-{snapshot_id:020d}"
            temporary = snapshots / f".{target.name}.{os.getpid()}.tmp"
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)
            temporary.mkdir(parents=True)
            capture_dir = temporary / ".capture"
            capture_dir.mkdir()

            state, graph_header = _runtime_state_without_materialized_graph(runtime)
            state.pop("graph", None)
            runtime_state_path = temporary / "runtime.pkl"
            graph_header_path = temporary / "graph_header.pkl"
            _dump_pickle_file(runtime_state_path, state)
            _dump_pickle_file(graph_header_path, graph_header)
            del state, graph_header
            captures = tuple(
                chunked_snapshot.capture_graph_shard(
                    runtime.graph, partition, capture_dir
                )
                for partition in range(int(runtime.graph.partition_count))
            )

        capture_done = time.perf_counter()
        runtime.set_telemetry_gauge(
            "snapshot_capture_seconds", capture_done - started
        )

        runtime_chunks, runtime_bytes, runtime_sha = chunked_snapshot.write_file_chunks(
            root, runtime_state_path
        )
        header_chunks, header_bytes, header_sha = chunked_snapshot.write_file_chunks(
            root, graph_header_path
        )
        graph_shards = [
            chunked_snapshot.publish_graph_shard_capture(root, capture)
            for capture in captures
        ]
        runtime_state_path.unlink(missing_ok=True)
        graph_header_path.unlink(missing_ok=True)
        shutil.rmtree(capture_dir, ignore_errors=True)

        manifest = {
            "schema": chunked_snapshot.NATIVE_SCHEMA,
            "snapshot_version": chunked_snapshot.SNAPSHOT_VERSION,
            "state_format": chunked_snapshot.STATE_FORMAT,
            "scientific_config_id": runtime.config.scientific.config_id.value,
            "snapshot_id": snapshot_id,
            "watermark": watermark,
            "graph_generation": generation,
            "chunk_bytes": chunked_snapshot.CHUNK_BYTES,
            "runtime_state_bytes": int(runtime_bytes),
            "runtime_state_sha256": str(runtime_sha),
            "runtime_state_chunks": runtime_chunks,
            "graph_header_bytes": int(header_bytes),
            "graph_header_sha256": str(header_sha),
            "graph_header_chunks": header_chunks,
            "graph_shards": graph_shards,
            "symbol_graph_schema_version": int(SYMBOL_GRAPH_SCHEMA_VERSION),
            "symbol_grounding_design_version": "9.7.9",
        }
        payload_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
        (temporary / "manifest.json").write_bytes(payload_bytes)
        (temporary / "COMPLETE").write_text(
            chunked_snapshot.sha256(payload_bytes) + "\n", encoding="ascii"
        )
        if target.exists():
            shutil.rmtree(target)
        os.replace(temporary, target)
        temporary = None
        runtime.set_telemetry_gauge(
            "snapshot_publish_seconds", time.perf_counter() - capture_done
        )
        runtime.set_telemetry_gauge(
            "snapshot_total_seconds", time.perf_counter() - started
        )
        return chunked_snapshot.SnapshotResult(
            target, snapshot_id, watermark, generation
        )
    except BaseException:
        with runtime._lock:
            runtime.telemetry["snapshot_writes"] = max(
                0, int(runtime.telemetry["snapshot_writes"]) - 1
            )
        raise
    finally:
        runtime.set_telemetry_gauge("snapshot_in_progress", 0)
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)

def _prune_grounding_registry(registry: GroundingRegistry) -> None:
    if len(registry.states) <= _GROUNDING_STATE_LIMIT:
        return
    for key in tuple(registry.states):
        if len(registry.states) <= _GROUNDING_STATE_LIMIT:
            break
        row = registry.states.get(key)
        if row is not None and not row.behavior_eligible:
            registry.states.pop(key, None)
    while len(registry.states) > _GROUNDING_STATE_LIMIT:
        registry.states.pop(next(iter(registry.states)), None)


def _install_grounding_bounds() -> None:
    if getattr(GroundingRegistry, "_runtime_integrity_bounds_installed", False):
        return
    original_observe = GroundingRegistry.observe
    original_from_state_dict = GroundingRegistry.from_state_dict

    def observe(self: GroundingRegistry, evidence: Any):
        row = original_observe(self, evidence)
        key = (
            int(evidence.symbol_structure_uid),
            int(evidence.interaction_structure_uid),
            int(evidence.environment_instance_id),
            int(evidence.context_scope_id),
            int(evidence.lineage_uid),
        )
        if len(row.validation_trial_ids) > _GROUNDING_TRIAL_ID_LIMIT:
            row = replace(row, validation_trial_ids=tuple(row.validation_trial_ids[-_GROUNDING_TRIAL_ID_LIMIT:]))
        self.states.pop(key, None)
        self.states[key] = row
        _prune_grounding_registry(self)
        return row

    @classmethod
    def from_state_dict(cls, state: dict[str, object]):
        result = original_from_state_dict(state)
        for key, row in tuple(result.states.items()):
            if len(row.validation_trial_ids) > _GROUNDING_TRIAL_ID_LIMIT:
                result.states[key] = replace(row, validation_trial_ids=tuple(row.validation_trial_ids[-_GROUNDING_TRIAL_ID_LIMIT:]))
        _prune_grounding_registry(result)
        return result

    GroundingRegistry.observe = observe
    GroundingRegistry.from_state_dict = from_state_dict
    GroundingRegistry._runtime_integrity_bounds_installed = True


def _install_viability_bounds_and_schema(environment_viability: Any) -> None:
    controller_cls = environment_viability.EnvironmentViabilityController
    if getattr(controller_cls, "_runtime_integrity_installed", False):
        return
    original_profile = controller_cls._profile
    original_load_state = controller_cls.load_state

    def trim(self: Any, keep_environment: int | None = None) -> None:
        while len(self.profiles) > _VIABILITY_PROFILE_LIMIT:
            candidates = [
                (int(profile.last_watermark), int(environment_id))
                for environment_id, profile in self.profiles.items()
                if keep_environment is None or int(environment_id) != int(keep_environment)
            ]
            if not candidates:
                break
            _, victim = min(candidates)
            self.profiles.pop(victim, None)

    def profile(self: Any, transition: Any):
        row = original_profile(self, transition)
        trim(self, int(row.environment_id))
        return row

    def load_state(self: Any, state: Any) -> None:
        original_load_state(self, state)
        trim(self)

    def append_log(self: Any, profile: Any, *, epoch: int, previous_state: str) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "record_type": "environment_profile",
            "epoch": int(epoch),
            "game": str(profile.game_scenario),
            "game_environment": str(profile.game_scenario),
            "previous_state": str(previous_state),
            "state": profile.state.value,
            "evidence_confidence": float(profile.viability_confidence),
            **profile.metrics(),
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")

    controller_cls._profile = profile
    controller_cls.load_state = load_state
    controller_cls._append_log = append_log
    controller_cls._runtime_integrity_installed = True


def _install_epoch_viability_unification(epoch_runner: Any) -> None:
    if getattr(epoch_runner, "_runtime_integrity_installed", False):
        return
    original_run_epochs = epoch_runner.run_epochs

    def run_epochs(runtime: Any, specs: tuple[Any, ...], args: Any, *, adapter_factory: Any | None = None):
        previous_runtime = getattr(epoch_runner, "_runtime_integrity_runtime", None)
        epoch_runner._runtime_integrity_runtime = runtime
        try:
            return original_run_epochs(runtime, specs, args, adapter_factory=adapter_factory)
        finally:
            epoch_runner._runtime_integrity_runtime = previous_runtime

    def environment_viability(rows: list[Any], previous: dict[str, dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
        runtime = getattr(epoch_runner, "_runtime_integrity_runtime", None)
        controller = None if runtime is None else getattr(runtime, "_environment_viability", None)
        grouped_rows: dict[str, list[Any]] = {}
        for row in rows:
            grouped_rows.setdefault(str(row.game_id), []).append(row)
        previous = previous or {}
        if controller is None:
            return {game: dict(previous.get(game, {})) for game in grouped_rows if game in previous}

        profiles_by_game: dict[str, list[Any]] = {}
        for profile in controller.profiles.values():
            profiles_by_game.setdefault(str(profile.game_scenario), []).append(profile)
        result: dict[str, dict[str, Any]] = {}
        state_priority = {
            "VIABILITY_ANOMALY": 0,
            "LOW_EVIDENCE": 1,
            "PROBING": 2,
            "RECOVERING": 3,
            "VIABLE": 4,
        }
        for game, game_rows in grouped_rows.items():
            candidates = sorted(
                profiles_by_game.get(game, ()),
                key=lambda profile: (int(profile.last_watermark), int(profile.environment_id)),
                reverse=True,
            )[: max(1, len(game_rows))]
            if not candidates:
                if game in previous:
                    result[game] = dict(previous[game])
                continue
            weights = [max(1, int(profile.observations)) for profile in candidates]
            total_weight = sum(weights)
            evidence_confidence = sum(float(profile.viability_confidence) * weight for profile, weight in zip(candidates, weights)) / max(1, total_weight)
            stagnation = sum(float(profile.stagnation) * weight for profile, weight in zip(candidates, weights)) / max(1, total_weight)
            coverage = sum(float(profile.action_coverage) * weight for profile, weight in zip(candidates, weights)) / max(1, total_weight)
            mean_branching = sum(float(profile.mean_branching_factor) * weight for profile, weight in zip(candidates, weights)) / max(1, total_weight)
            state = max((profile.state.value for profile in candidates), key=lambda value: state_priority.get(value, -1))
            if any(profile.state.value == "VIABILITY_ANOMALY" for profile in candidates) and not any(profile.state.value == "VIABLE" for profile in candidates):
                state = "VIABILITY_ANOMALY"
            steps = sum(int(row.steps) for row in game_rows)
            successes = sum(int(row.task_successes) for row in game_rows)
            failures = sum(int(row.task_failures) for row in game_rows)
            truncations = sum(int(row.task_truncations) for row in game_rows)
            levels = max((int(row.levels_completed) for row in game_rows), default=0)
            complete = successes + failures + truncations
            result[game] = {
                "state": state,
                "confidence": float(evidence_confidence),
                "evidence_confidence": float(evidence_confidence),
                "steps": steps,
                "complete_episodes": complete,
                "successes": successes,
                "failures": failures,
                "truncations": truncations,
                "levels_completed": levels,
                "mean_branching_factor": float(mean_branching),
                "max_branching_factor": max((int(profile.max_branching_factor) for profile in candidates), default=0),
                "action_coverage": float(coverage),
                "action_influence": sum(float(profile.action_influence_score) * weight for profile, weight in zip(candidates, weights)) / max(1, total_weight),
                "behavioral_rate": successes / max(1, complete),
                "behavioral_improvement": float(sum(profile.learning_progress * weight for profile, weight in zip(candidates, weights)) / max(1, total_weight)),
                "stagnation": float(stagnation),
                "reasons": sorted({reason for profile in candidates for reason in profile.anomaly_reasons}),
            }
        return result

    def append_environment_viability(root: str | Path, *, epoch: int, profiles: dict[str, dict[str, Any]]) -> None:
        target = Path(root) / "environment_viability.log"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            for game in sorted(profiles):
                handle.write(json.dumps({"record_type": "epoch_summary", "epoch": int(epoch), "game": game, **profiles[game]}, sort_keys=True) + "\n")

    def load_viability_history(root: str | Path) -> dict[str, dict[str, Any]]:
        target = Path(root) / "environment_viability.log"
        if not target.exists():
            return {}
        latest: dict[str, dict[str, Any]] = {}
        try:
            with target.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    game = str(row.get("game", row.get("game_environment", row.get("game_scenario", ""))))
                    if not game:
                        continue
                    normalized = dict(row)
                    normalized.setdefault("state", normalized.get("viability_state", "PROBING"))
                    normalized.setdefault("evidence_confidence", normalized.get("viability_confidence", 1.0))
                    normalized.setdefault("stagnation", 0.0)
                    latest[game] = normalized
        except (OSError, ValueError, TypeError):
            return {}
        return latest

    epoch_runner.run_epochs = run_epochs
    epoch_runner._environment_viability = environment_viability
    epoch_runner._append_environment_viability = append_environment_viability
    epoch_runner._load_viability_history = load_viability_history
    epoch_runner._runtime_integrity_installed = True


def _cleanup_owned_shared_memory(names: set[str]) -> None:
    for name in tuple(names):
        tracker_name = name if str(name).startswith("/") else f"/{name}"
        try:
            segment = shared_memory.SharedMemory(name=name, create=False)
        except FileNotFoundError:
            try:
                resource_tracker.unregister(tracker_name, "shared_memory")
            except (KeyError, ValueError):
                pass
            continue
        try:
            segment.close()
            segment.unlink()
        except FileNotFoundError:
            try:
                resource_tracker.unregister(tracker_name, "shared_memory")
            except (KeyError, ValueError):
                pass


def _install_shared_memory_cleanup(memory_pipeline: Any, topology_module: Any, shared_batch_transport: Any, worker_stop_cls: type) -> None:
    if getattr(memory_pipeline, "_runtime_integrity_shm_installed", False):
        return

    def publish_shared_batch(value: Any, *, start_sequence: int, end_sequence: int, rows: int):
        started = time.perf_counter()
        payload = pickle.dumps(value, protocol=5)
        segment = shared_memory.SharedMemory(create=True, size=max(1, len(payload)))
        try:
            segment.buf[: len(payload)] = payload
            name = segment.name
        finally:
            segment.close()
        return shared_batch_transport.SharedBatchDescriptor(
            name=name,
            size=len(payload),
            start_sequence=int(start_sequence),
            end_sequence=int(end_sequence),
            rows=int(rows),
            encode_ms=1000.0 * (time.perf_counter() - started),
        )

    # Cleanup/integrity owns transport allocation only. Worker behavior remains
    # authoritative in memory_pipeline.py so performance wrappers cannot regress it.
    shared_batch_transport.publish_shared_batch = publish_shared_batch
    memory_pipeline.publish_shared_batch = publish_shared_batch
    memory_pipeline._runtime_integrity_shm_installed = True


def _install_hgt_promotion_guard(hgt_training: Any) -> None:
    if getattr(hgt_training, "_runtime_integrity_promotion_installed", False):
        return
    original_train = hgt_training.train_hgt_epoch

    def train_hgt_epoch(runtime: Any, *, epoch: int, training_epochs: int, learning_rate: float, root: str | Path, allow_promotion: bool = True, _budget_scale: float = 1.0, _oom_retry: int = 0):
        if allow_promotion:
            return original_train(
                runtime,
                epoch=epoch,
                training_epochs=training_epochs,
                learning_rate=learning_rate,
                root=root,
                allow_promotion=True,
                _budget_scale=_budget_scale,
                _oom_retry=_oom_retry,
            )
        root_path = Path(root)
        manifest_path = root_path / "models" / "hgt_manifest.json"
        manifest_before = manifest_path.read_bytes() if manifest_path.exists() else None
        manifest_before_row = json.loads(manifest_before.decode("utf-8")) if manifest_before else {}
        policy_before = runtime.capture_hgt_policy_state()
        result = original_train(
            runtime,
            epoch=epoch,
            training_epochs=training_epochs,
            learning_rate=learning_rate,
            root=root,
            allow_promotion=False,
            _budget_scale=_budget_scale,
            _oom_retry=_oom_retry,
        )
        created_checkpoint = result.checkpoint
        previous_checkpoint = manifest_before_row.get("current_checkpoint") or manifest_before_row.get("accepted_checkpoint")
        if created_checkpoint and created_checkpoint != previous_checkpoint:
            checkpoint = root_path / str(created_checkpoint)
            checkpoint.unlink(missing_ok=True)
            checkpoint.with_suffix(".metadata.json").unlink(missing_ok=True)
        if manifest_before is None:
            manifest_path.unlink(missing_ok=True)
        else:
            temporary = manifest_path.with_suffix(".json.tmp")
            temporary.write_bytes(manifest_before)
            os.replace(temporary, manifest_path)
        runtime.set_hgt_action_scores(policy_before["scores"], context_action_scores=policy_before.get("context_scores", {}))
        runtime.unified_telemetry.model_version = policy_before["model_version"]
        runtime.set_telemetry_gauge("hgt_promotion_suppressed", 1)
        if int(result.training_steps) <= 0:
            return replace(result, model_version=str(policy_before["model_version"]), checkpoint=previous_checkpoint)
        return replace(
            result,
            status="TRAINED_NOT_PROMOTED",
            model_version=str(policy_before["model_version"]),
            checkpoint=previous_checkpoint,
        )

    hgt_training.train_hgt_epoch = train_hgt_epoch
    hgt_training._runtime_integrity_promotion_installed = True


def install_runtime_integrity(
    runtime_cls: type,
    graph_cls: type,
    pipeline_cls: type,
    hgt_training: Any,
) -> None:
    """Install final v9 correctness/bounded-memory repairs after all other wrappers."""
    if getattr(runtime_cls, "_runtime_integrity_installed", False):
        return

    from v9.runtime import chunked_snapshot
    from v9.runtime import environment_viability
    from v9.runtime import epoch_runner
    from v9.runtime import memory_pipeline
    from v9.runtime import memory_worker_topology
    from v9.runtime import shared_batch_transport
    from v9.runtime.multiprocess import WorkerStop

    _install_grounding_bounds()
    _install_viability_bounds_and_schema(environment_viability)
    _install_epoch_viability_unification(epoch_runner)
    _install_shared_memory_cleanup(memory_pipeline, memory_worker_topology, shared_batch_transport, WorkerStop)
    _install_hgt_promotion_guard(hgt_training)

    original_restore = runtime_cls._restore
    original_on_low_level_deleted = runtime_cls.on_low_level_deleted
    original_defer_base_group = runtime_cls._defer_base_group
    original_delete = graph_cls.delete_low_level_nodes_batch
    original_dispatch = pipeline_cls.dispatch_transition

    def restore(self: Any, snapshot: dict[str, Any], *, graph_override: Any | None = None) -> None:
        self._m1n_occurrences = {}
        self._m1n_supports = {}
        # SignatureIndexStore is the persistent dirty authority. Replacing its
        # bounded set-compatible view here would silently lose restart work.
        if not hasattr(self, "signature_index"):
            self._m1n_dirty = set()
        self._cross_modal_signatures = {}
        self._m2 = {}
        self._m3 = {}
        self._m4 = {}
        self._m5 = {}
        self._m6 = {}
        self._m7 = {}
        self._transfer_trials = {}
        self._deferred_base_nodes = {}
        self._memory_uids_by_environment = None
        self.__dict__.pop("_actor_policy_snapshot_cache", None)
        original_restore(self, snapshot, graph_override=graph_override)
        _prune_grounding_registry(self.grounding)
        self._memory_uids_by_environment = None
        self.__dict__.pop("_actor_policy_snapshot_cache", None)
        rebuild = getattr(self.graph, "rebuild_bounded_indexes", None)
        if callable(rebuild):
            manager = getattr(self, "_resident_memory", None)
            if manager is not None:
                scientific = self.config.scientific
                recent_capacity = max(
                    8192,
                    min(
                        int(scientific.resident_max_scan_batch),
                        max(int(scientific.hgt_max_subgraph_nodes) * 4, int(scientific.hgt_max_total_nodes)),
                    ),
                )
                edge_capacity = max(
                    64,
                    min(1024, int(scientific.hgt_max_subgraph_edges) // max(1, int(scientific.hgt_max_subgraph_nodes) // 4)),
                )
                rebuild(recent_capacity=recent_capacity, edge_capacity=edge_capacity)
                manager._recount_graph()
                manager.last_insert_check = int(getattr(self.graph, "low_level_nodes_inserted_total", 0))

    def apply_environment_evidence_confidence(self: Any, confidence_by_environment: dict[int, float]) -> int:
        confidence = {int(key): max(0.0, min(1.0, float(value))) for key, value in confidence_by_environment.items()}
        graph = self.graph
        with self._lock, graph._publication_lock:
            self._environment_evidence_confidence = confidence
            if self._memory_uids_by_environment is None:
                index: dict[int, set[Any]] = {}
                for uid, payload in graph.payloads.items():
                    environment_id = payload.get("environment_instance_id")
                    if environment_id is not None:
                        index.setdefault(int(environment_id), set()).add(uid)
                self._memory_uids_by_environment = index
            changed = 0
            for environment_id, value in confidence.items():
                for uid in tuple(self._memory_uids_by_environment.get(environment_id, ())):
                    payload = graph.payloads.get(uid)
                    if payload is None:
                        self._memory_uids_by_environment[environment_id].discard(uid)
                        continue
                    if abs(float(payload.get("evidence_confidence", 1.0)) - value) <= 1e-12:
                        continue
                    payload["evidence_confidence"] = value
                    graph.versions.bump(node_ref(uid))
                    changed += 1
            if changed:
                graph.generation += 1
                graph._cached_read_view = None
                self.__dict__.pop("_actor_policy_snapshot_cache", None)
            game_index = getattr(self, "_environment_ids_by_game", None)
            if isinstance(game_index, dict):
                for game in tuple(game_index):
                    ids = set(int(value) for value in game_index[game])
                    if len(ids) > _ENVIRONMENTS_PER_GAME_LIMIT:
                        live = [
                            environment_id
                            for environment_id in ids
                            if environment_id in self._memory_uids_by_environment
                            and self._memory_uids_by_environment[environment_id]
                        ]
                        retained = set(sorted(live)[-_ENVIRONMENTS_PER_GAME_LIMIT:])
                        game_index[game] = retained
            return changed

    def defer_base_group(self: Any, rows: tuple[tuple[Any, dict[str, Any], tuple[Any, ...]], ...]) -> None:
        original_defer_base_group(self, rows)
        index = getattr(self, "_memory_uids_by_environment", None)
        if index is None:
            return
        for node, payload, _evidence in rows:
            environment_id = payload.get("environment_instance_id")
            if environment_id is not None:
                index.setdefault(int(environment_id), set()).add(node.uid)

    def on_low_level_deleted(
        self: Any,
        deleted_uids: Iterable[Any],
        *,
        affected_signatures: Iterable[int] = (),
    ) -> None:
        deleted = tuple(deleted_uids)
        original_on_low_level_deleted(
            self,
            deleted,
            affected_signatures=tuple(int(value) for value in affected_signatures),
        )
        grounding_index = getattr(self, "_grounding_action_payload_by_low", None)
        if isinstance(grounding_index, dict):
            for uid in deleted:
                grounding_index.pop(int(uid.lo), None)
        self.__dict__.pop("_actor_policy_snapshot_cache", None)

    def delete_low_level_nodes_batch(self: Any, plans: tuple[Any, ...]):
        requested = {row[0] for row in plans}
        replacements = {row[1] for row in plans}
        incident: dict[Any, tuple[Any, Any]] = {}
        affected = set(replacements)
        frontier = list(replacements)
        for uid in requested:
            affected.update(self._provenance_sources_by_target.get(uid, ()))
            for key in tuple(self._outgoing_edge_keys.get(uid, ())) + tuple(self._incoming_edge_keys.get(uid, ())):
                edge = self.edges.get(key)
                if edge is not None:
                    incident[key] = (edge.source, edge.target)
        while frontier and len(affected) < 65_536:
            target = frontier.pop()
            for source in self._provenance_sources_by_target.get(target, ()):
                if source in affected:
                    continue
                affected.add(source)
                frontier.append(source)
        deleted = tuple(original_delete(self, plans))
        if not deleted:
            return deleted
        bounded = getattr(self, "_bounded_edges_by_uid", None)
        if isinstance(bounded, dict):
            for key, endpoints in incident.items():
                for uid in endpoints:
                    adjacency = bounded.get(uid)
                    if adjacency is not None:
                        adjacency.pop(key, None)
                        if not adjacency:
                            bounded.pop(uid, None)
        for uid in affected:
            if uid in self.nodes:
                self.versions.bump(node_ref(uid))
                for key in tuple(self._outgoing_edge_keys.get(uid, ())):
                    edge = self.edges.get(key)
                    if edge is not None:
                        self.versions.bump(edge_ref(edge))
        self._cached_read_view = None
        return deleted

    def dispatch_transition(self: Any, transition: Any) -> None:
        runtime = self.runtime
        marker = object()
        previous = runtime.__dict__.get("observe_environment_transition", marker)
        runtime.__dict__["observe_environment_transition"] = lambda *args, **kwargs: None
        try:
            return original_dispatch(self, transition)
        finally:
            if previous is marker:
                runtime.__dict__.pop("observe_environment_transition", None)
            else:
                runtime.__dict__["observe_environment_transition"] = previous

    def capture_experiment_state(self: Any) -> _ExperimentStateHandle:
        self.wait_quiescent()
        self.flush_deferred_memory_updates()
        counter = int(getattr(self, "_experiment_state_counter", 0)) + 1
        self._experiment_state_counter = counter
        target = Path(self.root) / ".experiment_states" / f"state-{os.getpid()}-{counter:08d}"
        _write_experiment_cut(self, target, chunked_snapshot)
        self.set_telemetry_gauge("experiment_state_disk_backed", 1)
        self.set_telemetry_gauge("experiment_state_captures", counter)
        return _ExperimentStateHandle(target)

    def restore_experiment_state(self: Any, captured: Any) -> None:
        self.wait_quiescent()
        if isinstance(captured, _ExperimentStateHandle):
            _restore_experiment_cut(self, captured, graph_cls, chunked_snapshot)
            return
        with self._lock:
            self._restore(captured)

    def snapshot(self: Any):
        result = _write_streaming_snapshot(self, chunked_snapshot)
        self.write_canonical_snapshot(self._snapshot_id)
        return result

    runtime_cls._restore = restore
    runtime_cls.apply_environment_evidence_confidence = apply_environment_evidence_confidence
    runtime_cls._defer_base_group = defer_base_group
    runtime_cls.on_low_level_deleted = on_low_level_deleted
    runtime_cls.capture_experiment_state = capture_experiment_state
    runtime_cls.restore_experiment_state = restore_experiment_state
    runtime_cls.snapshot = snapshot
    graph_cls.delete_low_level_nodes_batch = delete_low_level_nodes_batch
    pipeline_cls.dispatch_transition = dispatch_transition
    runtime_cls._runtime_integrity_installed = True
