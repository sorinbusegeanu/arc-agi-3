from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from v7.derivation.executor import ParallelDerivationConfig, ParallelDerivationExecutor
from v7.derivation.online_runtime import OnlineDerivationStats, OnlineHierarchyBuilder
from v7.derivation.pipeline import MemoryLearningPipeline
from v7.derivation.planning_runtime import Phase1PlanningBuilder, PlanningDerivationStats
from v7.derivation.scientific import EpisodeEvidence
from v7.memory.coordinator import GenerationCommitCoordinator, GenerationCommitResult
from v7.memory.development import DevelopmentalLifecycleRuntime
from v7.memory.durable_store import DurableGenerationStore
from v7.memory.evidence_lifecycle import EvidenceLifecycleStore
from v7.memory.evidence_store import EvidenceStore
from v7.memory.publisher import GenerationPublisher
from v7.memory.restart import RuntimeSnapshotStore
from v7.memory.transport.mmap_segments import SegmentedMmapReadViewTransport
from v7.memory.writer import CanonicalMemoryWriter


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

    def observe(self, evidence: EpisodeEvidence):
        if evidence.source_global_step is not None:
            self.writer.observe_global_step(evidence.source_global_step)
        return self.pipeline.observe_episode(evidence)

    def observe_batch(self, rows) -> tuple:
        batch = tuple(rows)
        for evidence in batch:
            if evidence.source_global_step is not None:
                self.writer.observe_global_step(evidence.source_global_step)
        return self.pipeline.observe_batch(batch)

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
            # Phase 1 adds persistent planning edges and executable M6
            # procedures after the ordinary hierarchy exists. Both are
            # published atomically in the next immutable generation.
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
