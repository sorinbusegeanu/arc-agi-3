from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from v7.derivation.executor import ParallelDerivationConfig, ParallelDerivationExecutor
from v7.derivation.online import OnlineDerivationStats, OnlineHierarchyBuilder
from v7.derivation.pipeline import MemoryLearningPipeline
from v7.derivation.scientific import EpisodeEvidence
from v7.memory.coordinator import GenerationCommitCoordinator, GenerationCommitResult
from v7.memory.development import DevelopmentalLifecycleRuntime
from v7.memory.durable_store import DurableGenerationStore
from v7.memory.evidence_lifecycle import ContradictionRecord, EvidenceLifecycleStore, TransferTrialRecord
from v7.memory.evidence_store import EvidenceRecord, EvidenceStore
from v7.memory.publisher import GenerationPublisher
from v7.memory.restart import RuntimeSnapshotStore
from v7.memory.transport.mmap_segments import SegmentedMmapReadViewTransport
from v7.memory.writer import CanonicalMemoryWriter

EVIDENCE_INTERACTION = 2001


@dataclass(frozen=True, slots=True)
class V7RuntimeConfig:
    root: Path
    restore: bool = True
    derive_hierarchy: bool = True
    derivation_workers: int = 4
    derivation_chunk_size: int = 256
    max_tasks_per_child: int | None = None

    @classmethod
    def from_path(
        cls,
        root: str | Path,
        *,
        restore: bool = True,
        derive_hierarchy: bool = True,
        derivation_workers: int = 4,
        derivation_chunk_size: int = 256,
        max_tasks_per_child: int | None = None,
    ) -> 'V7RuntimeConfig':
        return cls(Path(root), restore, derive_hierarchy, derivation_workers, derivation_chunk_size, max_tasks_per_child)


class V7Runtime:
    """End-to-end v7 runtime with durable generations, evidence, derivation and lifecycle."""

    def __init__(self, config: V7RuntimeConfig) -> None:
        self.config = config
        config.root.mkdir(parents=True, exist_ok=True)
        self.durable = DurableGenerationStore(config.root / 'state.sqlite')
        self.snapshots = RuntimeSnapshotStore(self.durable)
        self.writer = self.snapshots.restore() if config.restore else CanonicalMemoryWriter()
        self.evidence = EvidenceStore(config.root / 'evidence.sqlite')
        self.lifecycle_evidence = EvidenceLifecycleStore(config.root / 'lifecycle.sqlite')
        self.pipeline = MemoryLearningPipeline(self.writer, self.lifecycle_evidence)
        self.hierarchy = OnlineHierarchyBuilder(self.writer, self.pipeline)
        self.last_derivation_stats = OnlineDerivationStats()
        self.transport = SegmentedMmapReadViewTransport(config.root / 'segments')
        self.publisher = GenerationPublisher(self.transport)
        self.publisher.ensure_published(self.writer.published_view)
        self.coordinator = GenerationCommitCoordinator(writer=self.writer, durable_store=self.durable, publisher=self.publisher)
        self.lifecycle = DevelopmentalLifecycleRuntime(evidence_lifecycle=self.lifecycle_evidence, evidence_store=self.evidence)
        self.derivation_executor = ParallelDerivationExecutor(
            directory=config.root / 'segments',
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
        prior_view = self.writer.published_view
        score_input = prior_view.score_inputs(
            context_signature=evidence.context_signature,
            action_ids=(evidence.action_id,),
        )[0]
        memory_id = self.pipeline.observe_episode(evidence)
        self.evidence.append_evidence_batch((EvidenceRecord(
            memory_id=memory_id,
            evidence_type=EVIDENCE_INTERACTION,
            generation_id=int(self.writer.mutable_generation_id),
            source_game=evidence.source_game,
            source_context=evidence.source_context,
            source_global_step=evidence.source_global_step,
            payload={
                'context_signature': int(evidence.context_signature),
                'action_id': int(evidence.action_id),
                'outcome_signature': int(evidence.outcome_signature),
                'success': bool(evidence.success),
                'outcome_polarity': evidence.outcome_polarity,
                'prediction_error': float(evidence.prediction_error),
                'future_option_delta': float(evidence.future_option_delta),
            },
        ),))
        self._record_retrospective_signals(evidence, score_input)
        return memory_id

    def _record_retrospective_signals(self, evidence: EpisodeEvidence, score_input) -> None:
        generation_id = int(self.writer.mutable_generation_id)
        if float(evidence.prediction_error) > 0.0 and score_input.contingency_ids:
            self.lifecycle_evidence.append_contradictions(
                ContradictionRecord(
                    memory_id=memory_id,
                    generation_id=generation_id,
                    severity=float(evidence.prediction_error),
                    source_game=evidence.source_game,
                    source_context=evidence.source_context,
                    source_global_step=evidence.source_global_step,
                    payload={'action_id': int(evidence.action_id), 'outcome_signature': int(evidence.outcome_signature)},
                )
                for memory_id in score_input.contingency_ids
            )
        if evidence.outcome_polarity not in {'positive', 'negative'} or not evidence.source_game:
            return
        trials: list[TransferTrialRecord] = []
        for candidate_id in tuple(sorted(set((*score_input.role_ids, *score_input.concept_ids)), key=int)):
            source_games = tuple(game for game in self._provenance_source_games(candidate_id) if game != evidence.source_game)
            if not source_games or self._transfer_trial_exists(candidate_id, evidence.source_game, evidence.source_global_step):
                continue
            success = evidence.outcome_polarity == 'positive'
            trials.append(TransferTrialRecord(
                memory_id=candidate_id,
                generation_id=generation_id,
                source_game=source_games[0],
                target_game=evidence.source_game,
                success=success,
                score=1.0 if success else 0.0,
                payload={
                    'context_signature': int(evidence.context_signature),
                    'action_id': int(evidence.action_id),
                    'outcome_signature': int(evidence.outcome_signature),
                    'source_global_step': evidence.source_global_step,
                    'source_games': list(source_games),
                },
            ))
        if trials:
            self.lifecycle_evidence.append_transfer_trials(trials)

    def _provenance_source_games(self, memory_id) -> tuple[str, ...]:
        rows = self.lifecycle_evidence.connection.execute(
            """
            WITH RECURSIVE ancestry(memory_id) AS (
                SELECT ?
                UNION
                SELECT p.parent_memory_id FROM provenance_records p
                JOIN ancestry a ON p.memory_id=a.memory_id
                WHERE p.parent_memory_id IS NOT NULL
            )
            SELECT DISTINCT p.source_game FROM provenance_records p
            JOIN ancestry a ON p.memory_id=a.memory_id
            WHERE p.source_game IS NOT NULL AND p.source_game <> ''
            ORDER BY p.source_game
            """,
            (int(memory_id),),
        ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def _transfer_trial_exists(self, memory_id, target_game: str, source_global_step: int | None) -> bool:
        rows = self.lifecycle_evidence.connection.execute(
            'SELECT payload_json FROM transfer_trials WHERE memory_id=? AND target_game=?',
            (int(memory_id), str(target_game)),
        ).fetchall()
        import json
        for (payload_json,) in rows:
            try:
                payload = json.loads(str(payload_json or '{}'))
            except (TypeError, json.JSONDecodeError):
                continue
            if payload.get('source_global_step') == source_global_step:
                return True
        return False

    def max_source_global_step(self) -> int:
        row = self.evidence.connection.execute('SELECT MAX(source_global_step) FROM evidence_records').fetchone()
        return -1 if row is None or row[0] is None else int(row[0])

    def observe_batch(self, rows) -> tuple:
        return tuple(self.observe(row) for row in rows)

    def commit(
        self,
        *,
        batch_id: int = 0,
        run_lifecycle: bool = True,
        derive_hierarchy: bool | None = None,
    ) -> GenerationCommitResult:
        should_derive = self.config.derive_hierarchy if derive_hierarchy is None else bool(derive_hierarchy)
        self.last_derivation_stats = self.hierarchy.derive() if should_derive else OnlineDerivationStats()
        result = self.coordinator.commit(batch_id=batch_id)
        if run_lifecycle:
            self.lifecycle.run(result.view, writer=self.writer)
            dirty = self.writer.dirty_counts
            if dirty['nodes'] or dirty['scores'] or dirty['edges'] or dirty['cognition']:
                result = self.coordinator.commit(batch_id=batch_id + 1)
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
        return self.derivation_executor.run_dirty_plan(handle=record.handle, plan=plan, kernel=kernel)
