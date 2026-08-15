from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from v7.derivation.executor import ParallelDerivationConfig, ParallelDerivationExecutor
from v7.derivation.online_runtime import OnlineDerivationStats, OnlineHierarchyBuilder
from v7.derivation.pipeline import MemoryLearningPipeline
from v7.derivation.planning_runtime import Phase1PlanningBuilder, PlanningDerivationStats
from v7.derivation.scientific import EpisodeEvidence, TYPE_CONTINGENCY
from v7.memory.canonical import CanonicalMemoryKey
from v7.memory.coordinator import GenerationCommitCoordinator, GenerationCommitResult
from v7.memory.development import DevelopmentalLifecycleRuntime
from v7.memory.durable_store import DurableGenerationStore
from v7.memory.evidence_lifecycle import (
    EvidenceLifecycleStore,
    GateTrialRecord,
    ProvenanceRecord,
)
from v7.memory.evidence_store import EvidenceStore
from v7.memory.ids import MemoryId, MemoryLevel
from v7.memory.publisher import GenerationPublisher
from v7.memory.restart import RuntimeSnapshotStore
from v7.memory.state import GateId, gate_for_identity
from v7.memory.transport.mmap_segments import SegmentedMmapReadViewTransport
from v7.memory.writer import CanonicalMemoryWriter


_DISK_MEMORY_FILES = ("state.sqlite", "evidence.sqlite", "lifecycle.sqlite")
_SQLITE_SIDECARS = ("", "-wal", "-shm", "-journal")


def reset_disk_memory(root: str | Path) -> tuple[Path, ...]:
    """Delete persisted v7 memory while preserving reports and run summaries."""
    root_path = Path(root).expanduser().resolve()
    if root_path == Path(root_path.anchor) or root_path == Path.home().resolve():
        raise ValueError("refusing to reset memory at a filesystem or home root")
    removed = []
    for filename in _DISK_MEMORY_FILES:
        for suffix in _SQLITE_SIDECARS:
            path = root_path / f"{filename}{suffix}"
            if path.is_file() or path.is_symlink():
                path.unlink()
                removed.append(path)
    segments = root_path / "segments"
    if segments.is_symlink() or segments.is_file():
        segments.unlink()
        removed.append(segments)
    elif segments.is_dir():
        shutil.rmtree(segments)
        removed.append(segments)
    return tuple(removed)


@dataclass(frozen=True, slots=True)
class V7RuntimeConfig:
    root: Path
    restore: bool = True
    derivation_workers: int = 4
    derivation_chunk_size: int = 256
    max_tasks_per_child: int | None = None
    derive_hierarchy: bool = True

    @classmethod
    def from_path(
        cls,
        root: str | Path,
        *,
        restore: bool = True,
        derivation_workers: int = 4,
        derivation_chunk_size: int = 256,
        max_tasks_per_child: int | None = None,
        derive_hierarchy: bool = True,
    ) -> "V7RuntimeConfig":
        return cls(
            Path(root),
            restore,
            derivation_workers,
            derivation_chunk_size,
            max_tasks_per_child,
            derive_hierarchy,
        )


@dataclass(frozen=True, slots=True)
class _G01Prediction:
    memory_id: MemoryId
    predicted_outcome: int
    target_game: str | None
    target_context: str
    source_global_step: int | None


class V7Runtime:
    def __init__(self, config: V7RuntimeConfig) -> None:
        self.config = config
        config.root.mkdir(parents=True, exist_ok=True)
        self.durable = DurableGenerationStore(config.root / "state.sqlite")
        self.snapshots = RuntimeSnapshotStore(self.durable)
        self.writer = self.snapshots.restore() if config.restore else CanonicalMemoryWriter()
        self.evidence = EvidenceStore(config.root / "evidence.sqlite")
        self.lifecycle_evidence = EvidenceLifecycleStore(config.root / "lifecycle.sqlite")
        self.pipeline = MemoryLearningPipeline(
            self.writer, self.lifecycle_evidence, self.evidence
        )
        self.hierarchy = OnlineHierarchyBuilder(
            self.writer,
            self.pipeline,
            self.evidence,
            self.lifecycle_evidence,
        )
        self.planning_builder = Phase1PlanningBuilder(self.writer, self.evidence)
        self.last_derivation_stats = OnlineDerivationStats()
        self.last_planning_stats = PlanningDerivationStats()
        self.transport = SegmentedMmapReadViewTransport(config.root / "segments")
        self.publisher = GenerationPublisher(self.transport)
        self.publisher.ensure_published(self.writer.published_view)
        self.coordinator = GenerationCommitCoordinator(
            writer=self.writer,
            durable_store=self.durable,
            publisher=self.publisher,
        )
        self.lifecycle = DevelopmentalLifecycleRuntime(
            evidence_lifecycle=self.lifecycle_evidence,
            evidence_store=self.evidence,
        )
        self.derivation_executor = ParallelDerivationExecutor(
            directory=config.root / "segments",
            config=ParallelDerivationConfig(
                workers=max(1, int(config.derivation_workers)),
                max_tasks_per_child=config.max_tasks_per_child,
                chunk_size=max(1, int(config.derivation_chunk_size)),
            ),
        )

    def close(self) -> None:
        self.derivation_executor.close()
        self.evidence.close()
        self.lifecycle_evidence.close()
        self.durable.close()

    @staticmethod
    def _contexts(evidence: EpisodeEvidence) -> tuple[int, ...]:
        values = tuple(
            int(value)
            for value in getattr(evidence, "context_signatures", ()) or ()
        )
        return values or (int(evidence.context_signature),)

    def _capture_g01_predictions(
        self,
        batch: tuple[EpisodeEvidence, ...],
    ) -> tuple[_G01Prediction, ...]:
        builder = getattr(self.writer, "_cognition_indexes")
        registry = getattr(self.writer, "_canonical_registry")
        nodes = getattr(self.writer, "_nodes")
        rows: list[_G01Prediction] = []
        for evidence in batch:
            for context in self._contexts(evidence):
                memory_ids = tuple(
                    getattr(builder, "_contingencies", {}).get(
                        (int(context), int(evidence.action_id)),
                        (),
                    )
                )
                candidates = []
                for memory_id in memory_ids:
                    node = nodes.get(memory_id)
                    key = registry.key_for(memory_id)
                    if (
                        node is None
                        or key is None
                        or node.level != MemoryLevel.M1
                        or int(node.type_id) != TYPE_CONTINGENCY
                        or len(key.parts) < 3
                    ):
                        continue
                    candidates.append(
                        (
                            int(node.support_count),
                            -int(memory_id),
                            memory_id,
                            int(key.parts[2]),
                        )
                    )
                if not candidates:
                    continue
                _support, _neg_id, memory_id, predicted_outcome = max(candidates)
                node = nodes[memory_id]
                self.lifecycle_evidence.freeze_candidate_scope(
                    memory_id,
                    int(node.created_generation),
                )
                episode_context = (
                    f"{int(context)}#step:{evidence.source_global_step}"
                )
                rows.append(
                    _G01Prediction(
                        memory_id,
                        predicted_outcome,
                        evidence.source_game,
                        episode_context,
                        evidence.source_global_step,
                    )
                )
        return tuple(rows)

    def _complete_m1_provenance(
        self,
        batch: tuple[EpisodeEvidence, ...],
        primary_ids: tuple[MemoryId, ...],
    ) -> None:
        records: list[ProvenanceRecord] = []
        generation = int(self.writer.mutable_generation_id)
        nodes = getattr(self.writer, "_nodes")
        for evidence, primary_id in zip(batch, primary_ids, strict=True):
            for context in self._contexts(evidence):
                key = CanonicalMemoryKey(
                    MemoryLevel.M1,
                    TYPE_CONTINGENCY,
                    (
                        int(context),
                        int(evidence.action_id),
                        int(evidence.outcome_signature),
                    ),
                )
                memory_id = self.writer.canonical_memory_id(key)
                if memory_id is None:
                    continue
                if memory_id != primary_id:
                    records.append(
                        ProvenanceRecord(
                            memory_id=memory_id,
                            generation_id=generation,
                            source_game=evidence.source_game,
                            source_context=str(context),
                            source_global_step=evidence.source_global_step,
                        )
                    )
                node = nodes.get(memory_id)
                if node is not None and self.lifecycle_evidence.candidate_scope(memory_id) is None:
                    # Freeze immediately after the formation observation has
                    # been recorded. Later evidence cannot expand this scope.
                    if memory_id == primary_id or records:
                        pass
        if records:
            self.lifecycle_evidence.append_provenance(records)
        for evidence in batch:
            for context in self._contexts(evidence):
                memory_id = self.writer.canonical_memory_id(
                    CanonicalMemoryKey(
                        MemoryLevel.M1,
                        TYPE_CONTINGENCY,
                        (
                            int(context),
                            int(evidence.action_id),
                            int(evidence.outcome_signature),
                        ),
                    )
                )
                if memory_id is None:
                    continue
                node = nodes.get(memory_id)
                if node is not None:
                    self.lifecycle_evidence.freeze_candidate_scope(
                        memory_id,
                        int(node.created_generation),
                    )

    def _write_g01_trials(
        self,
        predictions: tuple[_G01Prediction, ...],
        batch: tuple[EpisodeEvidence, ...],
    ) -> None:
        actual_by_step = {
            (evidence.source_game, evidence.source_global_step): evidence
            for evidence in batch
        }
        generation = int(self.writer.mutable_generation_id)
        records: list[GateTrialRecord] = []
        for prediction in predictions:
            evidence = actual_by_step.get(
                (prediction.target_game, prediction.source_global_step)
            )
            if evidence is None:
                continue
            correct = int(prediction.predicted_outcome) == int(
                evidence.outcome_signature
            )
            gain = 0.50 if correct else -0.50
            scope = self.lifecycle_evidence.candidate_scope(prediction.memory_id)
            if scope is None:
                continue
            records.append(
                GateTrialRecord(
                    memory_id=prediction.memory_id,
                    generation_id=generation,
                    gate_id=GateId.G01,
                    candidate_generation=scope.candidate_generation,
                    target_game=prediction.target_game,
                    target_context=prediction.target_context,
                    participated=True,
                    contribution=1.0,
                    causal_gain=gain,
                    prediction_gain=gain,
                    intervention_type="heldout_prediction_ablation",
                    paired_trial_id=(
                        f"g01:{int(prediction.memory_id)}:"
                        f"{prediction.target_game}:{prediction.source_global_step}:"
                        f"{prediction.target_context}"
                    ),
                    payload={
                        "predicted_outcome": int(prediction.predicted_outcome),
                        "actual_outcome": int(evidence.outcome_signature),
                        "prediction_error": float(evidence.prediction_error),
                    },
                )
            )
        self.lifecycle_evidence.append_gate_trials(records)

    def _write_decision_gate_trials(
        self,
        batch: tuple[EpisodeEvidence, ...],
    ) -> None:
        nodes = getattr(self.writer, "_nodes")
        generation = int(self.writer.mutable_generation_id)
        records: list[GateTrialRecord] = []
        for evidence in batch:
            contributions = tuple(
                getattr(evidence, "decision_memory_contributions", ()) or ()
            )
            for raw_memory_id, raw_contribution in contributions:
                memory_id = MemoryId(int(raw_memory_id))
                contribution = float(raw_contribution)
                if contribution == 0.0:
                    continue
                node = nodes.get(memory_id)
                if node is None:
                    continue
                gate = gate_for_identity(node.level, node.type_id)
                if gate not in {GateId.G23R, GateId.G34, GateId.G45, GateId.G56}:
                    continue
                scope = self.lifecycle_evidence.freeze_candidate_scope(
                    memory_id,
                    int(node.created_generation),
                )
                records.append(
                    GateTrialRecord(
                        memory_id=memory_id,
                        generation_id=generation,
                        gate_id=gate,
                        candidate_generation=scope.candidate_generation,
                        target_game=evidence.source_game,
                        target_context=evidence.source_context,
                        participated=True,
                        contribution=contribution,
                        causal_gain=contribution,
                        terminal_gain=float(
                            1
                            if int(evidence.terminal_polarity) > 0
                            else -1
                            if int(evidence.terminal_polarity) < 0
                            else 0
                        ),
                        intervention_type="decision_score_ablation",
                        paired_trial_id=(
                            f"decision:{int(memory_id)}:{evidence.source_game}:"
                            f"{evidence.source_global_step}"
                        ),
                        payload={
                            "source_global_step": evidence.source_global_step,
                            "decision_score": float(evidence.decision_score),
                            "max_action_score": float(evidence.max_action_score),
                            "selection_mode": str(
                                getattr(evidence, "selection_mode", "") or ""
                            ),
                        },
                    )
                )
        self.lifecycle_evidence.append_gate_trials(records)

    def observe(self, evidence: EpisodeEvidence):
        return self.observe_batch((evidence,))[0]

    def observe_batch(self, rows) -> tuple:
        batch = tuple(rows)
        for evidence in batch:
            if evidence.source_global_step is not None:
                self.writer.observe_global_step(evidence.source_global_step)
        predictions = self._capture_g01_predictions(batch)
        primary_ids = self.pipeline.observe_batch(batch)
        self._complete_m1_provenance(batch, primary_ids)
        self._write_g01_trials(predictions, batch)
        self._write_decision_gate_trials(batch)
        return primary_ids

    def commit(
        self,
        *,
        batch_id: int = 0,
        run_lifecycle: bool = True,
        derive_hierarchy: bool | None = None,
    ) -> GenerationCommitResult:
        result = self.coordinator.commit(batch_id=batch_id)
        next_batch = int(batch_id) + 1
        should_derive = (
            self.config.derive_hierarchy
            if derive_hierarchy is None
            else bool(derive_hierarchy)
        )
        if should_derive:
            self.last_derivation_stats = self.hierarchy.derive()
            self.last_planning_stats = self.planning_builder.derive()
            dirty = self.writer.dirty_counts
            if dirty["nodes"] or dirty["scores"] or dirty["edges"] or dirty["cognition"]:
                result = self.coordinator.commit(batch_id=next_batch)
                next_batch += 1
        else:
            self.last_derivation_stats = OnlineDerivationStats()
            self.last_planning_stats = PlanningDerivationStats()
        if run_lifecycle:
            self.lifecycle.run(result.view, writer=self.writer)
            dirty = self.writer.dirty_counts
            if dirty["nodes"] or dirty["scores"] or dirty["edges"] or dirty["cognition"]:
                result = self.coordinator.commit(batch_id=next_batch)
        self.snapshots.persist(self.writer)
        return result

    def pending_derivation_plan(self):
        return self.writer.dirty_derivation_plan()

    def consume_derivation_plan(self):
        return self.writer.consume_dirty_derivation_plan()

    def run_dirty_derivation(self, kernel):
        plan = self.consume_derivation_plan()
        record = self.publisher.current_record
        if record is None:
            return ()
        return self.derivation_executor.run_dirty_plan(
            handle=record.handle,
            plan=plan,
            kernel=kernel,
        )
