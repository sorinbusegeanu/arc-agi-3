from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from v7.derivation.pipeline import MemoryLearningPipeline
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

    @classmethod
    def from_path(cls, root: str | Path, *, restore: bool = True) -> "V7RuntimeConfig":
        return cls(Path(root), restore)


class V7Runtime:
    """End-to-end v7 research runtime around one canonical writer."""

    def __init__(self, config: V7RuntimeConfig) -> None:
        self.config = config
        config.root.mkdir(parents=True, exist_ok=True)
        self.durable = DurableGenerationStore(config.root / "state.sqlite")
        self.snapshots = RuntimeSnapshotStore(self.durable)
        self.writer = self.snapshots.restore() if config.restore else CanonicalMemoryWriter()
        self.evidence = EvidenceStore(config.root / "evidence.sqlite")
        self.lifecycle_evidence = EvidenceLifecycleStore(config.root / "lifecycle.sqlite")
        self.pipeline = MemoryLearningPipeline(self.writer, self.lifecycle_evidence)
        self.transport = SegmentedMmapReadViewTransport(config.root / "segments")
        self.publisher = GenerationPublisher(self.transport)
        if int(self.writer.published_view.generation_id) > 0:
            self.publisher.ensure_published(self.writer.published_view)
        self.coordinator = GenerationCommitCoordinator(writer=self.writer, durable_store=self.durable, publisher=self.publisher)
        self.lifecycle = DevelopmentalLifecycleRuntime(evidence_lifecycle=self.lifecycle_evidence, evidence_store=self.evidence)

    def close(self) -> None:
        self.evidence.close()
        self.lifecycle_evidence.close()
        self.durable.close()

    def observe(self, evidence: EpisodeEvidence):
        if evidence.source_global_step is not None:
            self.writer.observe_global_step(evidence.source_global_step)
        return self.pipeline.observe_episode(evidence)

    def commit(self, *, batch_id: int = 0, run_lifecycle: bool = True) -> GenerationCommitResult:
        result = self.coordinator.commit(batch_id=batch_id)
        self.snapshots.persist(self.writer)
        if run_lifecycle:
            self.lifecycle.run(result.view, writer=self.writer)
        return result

    def pending_derivation_plan(self):
        return self.writer.dirty_derivation_plan()

    def consume_derivation_plan(self):
        return self.writer.consume_dirty_derivation_plan()
