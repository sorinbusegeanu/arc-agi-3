from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

from v8.compression import CompressionEstimator
from v8.context_refinement import ContextRefiner
from v8.evidence import EvidenceLedger, EvidenceRecord
from v8.future_options import FutureOptionEstimator
from v8.lifecycle import LifecycleController
from v8.model import (
    CognitiveState,
    EventId,
    MemoryLevel,
    MemoryProposal,
    MemoryType,
    MemoryUid,
    RelationType,
    ValidationState,
    proposal_fingerprint,
)
from v8.outcomes import OutcomeEquivalenceEstimator
from v8.prediction import PredictionEstimator
from v8.preference import PreferenceEstimator
from v8.pruning import PruningPlanner
from v8.replanning import ReplanningController, ReplanningTrial
from v8.replay import ReplayScheduler
from v8.roles import FunctionalRoleEstimator
from v8.similarity import BoundedNeighborhoodSimilarity
from v8.strategies import StrategyEstimator
from v8.transfer import TransferTrial, TransferValidator
from v8.world_model import WorldModelEstimator


@dataclass(frozen=True, slots=True)
class PeerMetrics:
    cycles: int
    proposals: int
    evidence_records: int
    interval_seconds: float
    candidate_budget: int
    failures: int
    similarity_comparisons: int = 0
    similarity_processed_descriptors: int = 0


class DevelopmentalPeerSupervisor:
    """Concurrent read-snapshot peers which only emit canonical mutation proposals."""

    def __init__(
        self,
        *,
        read_view,
        submit_proposal: Callable[[MemoryProposal], None],
        watermark: Callable[[], int],
        generation: Callable[[], int] | None = None,
        interval_seconds: float = 0.5,
    ) -> None:
        self.read_view = read_view
        self.submit_proposal = submit_proposal
        self.current_watermark = watermark
        self.current_generation = generation or (lambda: 0)
        self.interval_seconds = float(interval_seconds)
        self.candidate_budget = 256
        self.prediction = PredictionEstimator()
        self.context = ContextRefiner()
        self.roles = FunctionalRoleEstimator()
        self.future_options = FutureOptionEstimator()
        self.compression = CompressionEstimator()
        self.similarity = BoundedNeighborhoodSimilarity()
        self.transfer = TransferValidator()
        self.outcomes = OutcomeEquivalenceEstimator()
        self.strategies = StrategyEstimator()
        self.preference = PreferenceEstimator()
        self.replanning = ReplanningController()
        self.lifecycle = LifecycleController()
        self.pruning = PruningPlanner()
        self.replay = ReplayScheduler()
        self.world_model = WorldModelEstimator()
        self.ledger = EvidenceLedger()
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._thread: threading.Thread | None = None
        self._sequence = 0
        self._evidence_sequence = 0
        self._seen: dict[tuple[str, int, int], int] = {}
        self._cycles = 0
        self._proposals = 0
        self._failures = 0
        self._last_error: str | None = None
        self._run_lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run,
            name="v8-developmental-peers",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def pause(self) -> None:
        self._pause.set()

    def wait_idle(self, timeout: float) -> bool:
        """Wait until an already-running peer cycle has released its write lock."""
        acquired = self._run_lock.acquire(timeout=max(0.0, float(timeout)))
        if acquired:
            self._run_lock.release()
        return acquired

    def resume(self) -> None:
        self._pause.clear()

    def set_interval(self, seconds: float) -> None:
        self.interval_seconds = max(0.05, float(seconds))

    def set_candidate_budget(self, budget: int) -> None:
        self.candidate_budget = max(8, int(budget))
        self.similarity.set_budget(min(32, self.candidate_budget))

    def raise_if_failed(self) -> None:
        if self._last_error is not None:
            message = self._last_error
            self._last_error = None
            raise RuntimeError(f"v8 developmental peer failure: {message}")

    def metrics(self) -> PeerMetrics:
        return PeerMetrics(
            self._cycles,
            self._proposals,
            len(self.ledger.cut(self.current_watermark())),
            self.interval_seconds,
            self.candidate_budget,
            self._failures,
            self.similarity.candidate_comparisons,
            self.similarity.processed_descriptors,
        )

    def _event_id(self) -> EventId:
        self._sequence += 1
        return EventId.from_producer(0x7FFFFFFE, self._sequence)

    def _fresh(self, kind: str, uid: MemoryUid, watermark: int) -> bool:
        key = (kind, int(uid.hi), int(uid.lo))
        prior = self._seen.get(key, -1)
        if int(watermark) <= prior:
            return False
        self._seen[key] = int(watermark)
        return True

    def _append_evidence(
        self,
        kind: str,
        row,
        value: float,
        *,
        validation_state: int | None = None,
        unique: bool = False,
        target_game_hash: int = 0,
        provenance_games: tuple[int, ...] | None = None,
        causal_intervention: str = "",
        effect_direction: int = 0,
        quality: float = 1.0,
    ) -> None:
        watermark = int(self.current_watermark())
        suffix = ""
        if unique:
            self._evidence_sequence += 1
            suffix = f":{self._evidence_sequence}"
        evidence_id = f"{kind}:{row.uid.hex()}:{watermark}{suffix}"
        games = (
            tuple(sorted(self.read_view.source_games(row.uid)))
            if provenance_games is None
            else tuple(sorted(set(int(v) for v in provenance_games)))
        )
        self.ledger.append(
            EvidenceRecord.for_uid(
                evidence_id,
                row.uid,
                evidence_kind=kind,
                watermark=watermark,
                raw_value=float(value),
                normalized_value=max(0.0, min(1.0, abs(float(value)))),
                developmental_stage=int(row.level),
                validation_state=int(
                    row.validation_state
                    if validation_state is None
                    else validation_state
                ),
                source_game_hash=(games[0] if len(games) == 1 else 0),
                target_game_hash=int(target_game_hash),
                provenance_games=games,
                causal_intervention=causal_intervention,
                effect_direction=effect_direction,
                quality=quality,
                graph_generation=int(self.current_generation()),
            )
        )

    def _existing_proposal(
        self,
        row,
        *,
        prediction_error: float = 0.0,
        transfer_prior: float = 0.0,
        explanatory: float = 0.0,
        future_option: float = 0.0,
        success_sum: float = 0.0,
        cost_sum: float = 0.0,
        attempt_weight: float = 0.0,
        cognitive_state: int = -1,
        validation_state: int = -1,
        parent_uid: MemoryUid | None = None,
        relation_type: RelationType = RelationType.EXPLAINS,
        source_game_hash: int = 0,
    ) -> MemoryProposal:
        return MemoryProposal(
            uid=row.uid,
            fingerprint=int(row.fingerprint),
            event_id=self._event_id(),
            watermark=int(self.current_watermark()),
            level=MemoryLevel(int(row.level)),
            memory_type=MemoryType(int(row.memory_type)),
            key_parts=tuple(int(v) for v in row.key_parts),
            support_delta=0,
            prediction_error_sum=float(prediction_error),
            transfer_prior_sum=float(transfer_prior),
            explanatory_sum=float(explanatory),
            future_option_sum=float(future_option),
            score_weight=0.0,
            success_sum=float(success_sum),
            cost_sum=float(cost_sum),
            attempt_weight=float(attempt_weight),
            parent_uid=MemoryUid.zero() if parent_uid is None else parent_uid,
            relation_type=relation_type,
            source_game_hash=int(source_game_hash),
            cognitive_state=int(cognitive_state),
            validation_state=int(validation_state),
        )

    def _submit(self, proposal: MemoryProposal) -> None:
        self.submit_proposal(proposal)
        self._proposals += 1

    def _parallel_analyses(self, nodes, edges):
        with ThreadPoolExecutor(max_workers=9, thread_name_prefix="v8-peer") as pool:
            futures = {
                "prediction": pool.submit(self.prediction.evaluate, nodes),
                "context": pool.submit(self.context.propose, nodes),
                "roles": pool.submit(self.roles.propose, nodes),
                "future": pool.submit(self.future_options.evaluate, nodes),
                "compression": pool.submit(self.compression.evaluate, nodes, edges),
                "similarity": pool.submit(self.similarity.evaluate, nodes, edges),
                "transfer": pool.submit(
                    self.transfer.candidates,
                    nodes,
                    provenance=self.read_view.source_games,
                ),
                "world": pool.submit(self.world_model.propose, nodes),
                "replay": pool.submit(
                    self.replay.candidates,
                    nodes,
                    budget=self.candidate_budget,
                ),
            }
            return {name: future.result() for name, future in futures.items()}

    def run_once(self) -> None:
        if not self._run_lock.acquire(blocking=False):
            return
        try:
            def cancelled() -> bool:
                event = getattr(self, "_v841_peer_cancel", None)
                return bool(event is not None and event.is_set())

            if cancelled():
                return
            nodes = self.read_view.node_records()
            edges = self.read_view.edge_records()
            if cancelled():
                return
            by_uid = {row.uid: row for row in nodes}
            analyses = self._parallel_analyses(nodes, edges)
            if cancelled():
                return
            attended = {item.uid for item in analyses["replay"]}

            for row_index, row in enumerate(nodes):
                if row_index % 256 == 0 and cancelled():
                    return
                if row.support_count < 2:
                    continue
                kind = {
                    int(MemoryLevel.M1): "contingency_recurrence",
                    int(MemoryLevel.M2): "family_recurrence",
                    int(MemoryLevel.M4): "concept_candidate",
                    int(MemoryLevel.M5): "consequence_structure",
                    int(MemoryLevel.M6): "outcome_equivalence",
                    int(MemoryLevel.M7): "strategy_reuse",
                }.get(int(row.level))
                if int(row.level) == int(MemoryLevel.M3):
                    kind = (
                        "carrier_candidate"
                        if int(row.memory_type) == int(MemoryType.CARRIER)
                        else "role_candidate"
                    )
                if kind and self._fresh(kind, row.uid, row.updated_watermark):
                    self._append_evidence(
                        kind,
                        row,
                        min(1.0, row.support_count / 4.0),
                    )
                if (
                    int(row.level) == int(MemoryLevel.M3)
                    and int(row.memory_type) == int(MemoryType.CARRIER)
                    and self._fresh(
                        "carrier_emergence", row.uid, row.updated_watermark
                    )
                ):
                    self._append_evidence(
                        "carrier_emergence",
                        row,
                        min(1.0, row.support_count / 4.0),
                    )
                if (
                    int(row.memory_type) == int(MemoryType.CONTEXTUAL_ROLE)
                    and row.support_count >= 2
                    and self._fresh("context_gain", row.uid, row.updated_watermark)
                ):
                    self._append_evidence(
                        "context_refinement_gain",
                        row,
                        max(row.significance, row.learning_value),
                    )

            if cancelled():
                return
            for replay in analyses["replay"]:
                row = by_uid.get(replay.uid)
                if row is not None and self._fresh(
                    "replay", row.uid, row.updated_watermark
                ):
                    self._append_evidence(
                        "replay_priority", row, replay.priority
                    )

            if cancelled():
                return
            for evidence_index, evidence in enumerate(analyses["prediction"]):
                if evidence_index % 256 == 0 and cancelled():
                    return
                uid = MemoryUid(evidence.uid_hi, evidence.uid_lo)
                row = by_uid.get(uid)
                if row is None or not self._fresh(
                    "prediction", uid, row.updated_watermark
                ):
                    continue
                self._append_evidence("supported_prediction", row, 1.0)
                if evidence.error > 0.0:
                    self._append_evidence(
                        "prediction_violation", row, evidence.error
                    )

            if cancelled():
                return
            for refinement in analyses["context"][: self.candidate_budget]:
                source = by_uid.get(refinement.source_uid)
                if source is None or not self._fresh(
                    "context",
                    refinement.candidate_uid,
                    source.updated_watermark,
                ):
                    continue
                proposal = MemoryProposal(
                    uid=refinement.candidate_uid,
                    fingerprint=proposal_fingerprint(
                        MemoryLevel.M3,
                        MemoryType.CONTEXTUAL_ROLE,
                        refinement.key_parts,
                    ),
                    event_id=self._event_id(),
                    watermark=int(self.current_watermark()),
                    level=MemoryLevel.M3,
                    memory_type=MemoryType.CONTEXTUAL_ROLE,
                    key_parts=refinement.key_parts,
                    support_delta=1,
                    significance_sum=refinement.contradiction_rate,
                    learning_value_sum=refinement.contradiction_rate,
                    score_weight=1.0,
                    parent_uid=source.uid,
                    relation_type=RelationType.CONTEXT_REFINES,
                    cognitive_state=int(CognitiveState.PROBATION),
                    validation_state=int(ValidationState.STRUCTURAL),
                )
                self._submit(proposal)
                self._append_evidence(
                    "context_refinement",
                    source,
                    refinement.contradiction_rate,
                )

            if cancelled():
                return
            for candidate in analyses["roles"][: self.candidate_budget]:
                watermarks = [
                    by_uid[uid].updated_watermark
                    for uid in candidate.carriers
                    if uid in by_uid
                ]
                if not watermarks or not self._fresh(
                    "role", candidate.uid, max(watermarks)
                ):
                    continue
                exact_games: set[int] = set()
                for carrier in candidate.carriers:
                    exact_games.update(self.read_view.source_games(carrier))
                proposal = MemoryProposal(
                    uid=candidate.uid,
                    fingerprint=proposal_fingerprint(
                        MemoryLevel.M3,
                        MemoryType.ROLE,
                        candidate.key_parts,
                    ),
                    event_id=self._event_id(),
                    watermark=int(self.current_watermark()),
                    level=MemoryLevel.M3,
                    memory_type=MemoryType.ROLE,
                    key_parts=candidate.key_parts,
                    support_delta=len(candidate.carriers),
                    explanatory_sum=float(len(candidate.carriers)),
                    transfer_prior_sum=min(1.0, len(exact_games) / 2.0),
                    score_weight=1.0,
                    parent_uid=candidate.carriers[0],
                    relation_type=RelationType.EXPLAINS,
                    cognitive_state=int(CognitiveState.ACTIVE),
                    validation_state=int(ValidationState.STRUCTURAL),
                )
                self._submit(proposal)
                source = by_uid.get(candidate.carriers[0])
                if source is not None:
                    self._append_evidence(
                        "role_emergence",
                        source,
                        1.0,
                        validation_state=int(ValidationState.STRUCTURAL),
                    )

            if cancelled():
                return
            for evidence in analyses["future"][: self.candidate_budget]:
                row = by_uid.get(evidence.uid)
                if row is None or not self._fresh(
                    "fo", row.uid, row.updated_watermark
                ):
                    continue
                self._submit(
                    self._existing_proposal(
                        row,
                        future_option=float(evidence.delta),
                    )
                )
                self._append_evidence(
                    "future_option_estimate",
                    row,
                    min(1.0, abs(evidence.delta) / 4.0),
                )

            if cancelled():
                return
            for evidence in analyses["compression"][: self.candidate_budget]:
                row = by_uid.get(evidence.uid)
                if row is None or not self._fresh(
                    "compression", row.uid, row.updated_watermark
                ):
                    continue
                self._submit(
                    self._existing_proposal(
                        row,
                        explanatory=float(evidence.explanatory_reach),
                    )
                )
                kind = (
                    "family_compression"
                    if int(row.level) == int(MemoryLevel.M2)
                    else "compression"
                )
                self._append_evidence(
                    kind,
                    row,
                    min(1.0, evidence.compression_benefit / 4.0),
                )
                for target in evidence.superseded[:8]:
                    self._submit(
                        self._existing_proposal(
                            row,
                            parent_uid=target,
                            relation_type=RelationType.SUPERSEDES,
                        )
                    )

            if cancelled():
                return
            for component in analyses["world"][: self.candidate_budget]:
                wm_watermark = max(
                    (
                        by_uid[uid].updated_watermark
                        for uid in component.consequences
                        if uid in by_uid
                    ),
                    default=0,
                )
                if (
                    not component.consequences
                    or not self._fresh(
                        "world_model", component.uid, wm_watermark
                    )
                ):
                    continue
                first = component.consequences[0]
                proposal = MemoryProposal(
                    uid=component.uid,
                    fingerprint=proposal_fingerprint(
                        MemoryLevel.M5,
                        MemoryType.WORLD_MODEL,
                        component.key_parts,
                    ),
                    event_id=self._event_id(),
                    watermark=int(self.current_watermark()),
                    level=MemoryLevel.M5,
                    memory_type=MemoryType.WORLD_MODEL,
                    key_parts=component.key_parts,
                    support_delta=max(1, component.support),
                    explanatory_sum=float(len(component.consequences)),
                    score_weight=1.0,
                    parent_uid=first,
                    relation_type=RelationType.EXPLAINS,
                    cognitive_state=int(CognitiveState.ACTIVE),
                    validation_state=int(ValidationState.STRUCTURAL),
                )
                self._submit(proposal)
                for parent in component.consequences[1:8]:
                    identity = type(
                        "WorldModelIdentity",
                        (),
                        {
                            "uid": component.uid,
                            "fingerprint": proposal.fingerprint,
                            "level": int(MemoryLevel.M5),
                            "memory_type": int(MemoryType.WORLD_MODEL),
                            "key_parts": component.key_parts,
                        },
                    )()
                    self._submit(
                        self._existing_proposal(identity, parent_uid=parent)
                    )
                source = by_uid.get(first)
                if source is not None:
                    self._append_evidence(
                        "world_model_component",
                        source,
                        min(1.0, len(component.consequences) / 4.0),
                    )

            # Similarity is candidate evidence only.  It creates a canonical
            # SIMILAR_TO edge and a transfer prior, never identity merging or
            # validation. Cross-game empirical validation remains a separate trial.
            if cancelled():
                return
            for evidence in analyses["similarity"][: self.candidate_budget]:
                source = by_uid.get(evidence.source_uid)
                target = by_uid.get(evidence.target_uid)
                if source is None or target is None:
                    continue
                freshness_kind = f"similarity:{target.uid.hex()}"
                if not self._fresh(
                    freshness_kind,
                    source.uid,
                    evidence.evidence_watermark,
                ):
                    continue
                self._submit(
                    self._existing_proposal(
                        source,
                        transfer_prior=evidence.score,
                        parent_uid=target.uid,
                        relation_type=RelationType.SIMILAR_TO,
                    )
                )
                self._submit(
                    self._existing_proposal(
                        target,
                        transfer_prior=evidence.score,
                    )
                )
                games = tuple(
                    sorted(
                        self.read_view.source_games(source.uid)
                        | self.read_view.source_games(target.uid)
                    )
                )
                self._append_evidence(
                    "structural_similarity",
                    source,
                    evidence.score,
                    validation_state=int(ValidationState.STRUCTURAL),
                    provenance_games=games,
                )

            if cancelled():
                return
            for candidate in analyses["transfer"][: self.candidate_budget]:
                row = by_uid.get(candidate.uid)
                if row is None or not self._fresh(
                    "transfer", row.uid, row.updated_watermark
                ):
                    continue
                self._submit(
                    self._existing_proposal(
                        row,
                        transfer_prior=candidate.structural_score,
                        validation_state=int(ValidationState.STRUCTURAL),
                    )
                )
                self._append_evidence(
                    "transfer_structural",
                    row,
                    candidate.structural_score,
                    validation_state=int(ValidationState.STRUCTURAL),
                    provenance_games=candidate.formation_games,
                )

            if cancelled():
                return
            classes = self.outcomes.rebuild(nodes)
            if cancelled():
                return
            m7_rows = [
                row
                for row in nodes
                if int(row.level) == int(MemoryLevel.M7)
                and len(row.key_parts) >= 4
            ]
            for outcome in classes[: self.candidate_budget]:
                revision = self.outcomes.merge_revision(outcome)
                if revision is None:
                    row = by_uid.get(outcome.uid)
                    if (
                        row is not None
                        and row.support_count >= 2
                        and self._fresh(
                            "outcome", row.uid, row.updated_watermark
                        )
                    ):
                        self._append_evidence(
                            "outcome_equivalence",
                            row,
                            min(1.0, row.support_count / 4.0),
                        )
                    continue
                max_wm = max(
                    (
                        by_uid[uid].updated_watermark
                        for uid in revision.sources
                        if uid in by_uid
                    ),
                    default=0,
                )
                if not self._fresh(
                    "outcome_merge", revision.target, max_wm
                ):
                    continue
                target_fingerprint = proposal_fingerprint(
                    MemoryLevel.M6,
                    MemoryType.OUTCOME,
                    revision.descriptor,
                )
                for index, source_uid in enumerate(revision.sources):
                    source = by_uid.get(source_uid)
                    if source is None:
                        continue
                    self._submit(
                        MemoryProposal(
                            uid=revision.target,
                            fingerprint=target_fingerprint,
                            event_id=self._event_id(),
                            watermark=int(self.current_watermark()),
                            level=MemoryLevel.M6,
                            memory_type=MemoryType.OUTCOME,
                            key_parts=tuple(revision.descriptor),
                            support_delta=(
                                max(1, outcome.support) if index == 0 else 0
                            ),
                            explanatory_sum=1.0 if index == 0 else 0.0,
                            score_weight=1.0 if index == 0 else 0.0,
                            parent_uid=source_uid,
                            relation_type=RelationType.SUPERSEDES,
                            cognitive_state=int(CognitiveState.ACTIVE),
                            validation_state=int(ValidationState.STRUCTURAL),
                        )
                    )
                    self._append_evidence("outcome_merge", source, 1.0)
                for strategy in m7_rows:
                    member_outcome = MemoryUid(
                        int(strategy.key_parts[1]),
                        int(strategy.key_parts[2]),
                    )
                    if member_outcome not in revision.sources:
                        continue
                    key = (
                        int(strategy.key_parts[0]),
                        int(revision.target.hi),
                        int(revision.target.lo),
                        int(strategy.key_parts[3]),
                    )
                    merged_uid = MemoryUid.from_key(
                        MemoryLevel.M7,
                        MemoryType.STRATEGY,
                        key,
                    )
                    self._submit(
                        MemoryProposal(
                            uid=merged_uid,
                            fingerprint=proposal_fingerprint(
                                MemoryLevel.M7,
                                MemoryType.STRATEGY,
                                key,
                            ),
                            event_id=self._event_id(),
                            watermark=int(self.current_watermark()),
                            level=MemoryLevel.M7,
                            memory_type=MemoryType.STRATEGY,
                            key_parts=key,
                            support_delta=max(
                                1, int(strategy.support_count)
                            ),
                            significance_sum=float(
                                strategy.significance_sum
                            ),
                            learning_value_sum=float(
                                strategy.learning_value_sum
                            ),
                            future_option_sum=float(
                                strategy.future_option_sum
                            ),
                            score_weight=max(
                                1.0, float(strategy.score_weight)
                            ),
                            success_sum=float(strategy.success_sum),
                            cost_sum=float(strategy.cost_sum),
                            attempt_weight=float(strategy.attempt_weight),
                            parent_uid=revision.target,
                            relation_type=RelationType.LEADS_TO,
                            cognitive_state=int(strategy.cognitive_state),
                            validation_state=int(strategy.validation_state),
                        )
                    )

            if cancelled():
                return
            by_outcome = self.strategies.by_outcome(nodes)
            if cancelled():
                return
            for _outcome_uid, alternatives in by_outcome.items():
                if len(alternatives) < 2:
                    continue
                for strategy in alternatives:
                    row = by_uid.get(strategy.uid)
                    if row is None:
                        continue
                    if self._fresh(
                        "alternative", row.uid, row.updated_watermark
                    ):
                        self._append_evidence(
                            "alternative_strategy",
                            row,
                            min(1.0, len(alternatives) / 3.0),
                        )
                    if (
                        row.attempt_weight > 0
                        and self._fresh(
                            "efficiency", row.uid, row.updated_watermark
                        )
                    ):
                        efficiency = 1.0 / max(
                            1e-9, strategy.mean_cost
                        )
                        self._append_evidence(
                            "strategy_efficiency",
                            row,
                            min(1.0, efficiency),
                        )

            if cancelled():
                return
            lifecycle_rows = tuple(
                row
                for row in nodes
                if not attended
                or row.uid in attended
                or int(row.cognitive_state)
                >= int(CognitiveState.QUARANTINED)
            )
            for row in lifecycle_rows[: self.candidate_budget * 2]:
                decision = self.lifecycle.decide(row)
                if decision is None or not self._fresh(
                    "lifecycle", row.uid, row.updated_watermark
                ):
                    continue
                self._submit(
                    self._existing_proposal(
                        row,
                        cognitive_state=decision.cognitive_state,
                        validation_state=decision.validation_state,
                    )
                )
            if cancelled():
                return
            protected = {
                candidate.uid: candidate.protected_by_dependencies
                for candidate in self.pruning.candidates(nodes, edges)
            }
            if cancelled():
                return
            for row_index, row in enumerate(nodes):
                if row_index % 256 == 0 and cancelled():
                    return
                if row.uid not in protected:
                    continue
                decision = self.lifecycle.finalize_retirement(
                    row,
                    protected_by_dependencies=protected[row.uid],
                )
                if decision is None or not self._fresh(
                    "retire", row.uid, row.updated_watermark
                ):
                    continue
                self._submit(
                    self._existing_proposal(
                        row,
                        cognitive_state=decision.cognitive_state,
                        validation_state=decision.validation_state,
                    )
                )
                self._append_evidence("memory_retired", row, 1.0)

            self._cycles += 1
        finally:
            self._run_lock.release()

    def record_strategy_statistics(
        self,
        uid: MemoryUid,
        *,
        attempts: int,
        successes: int,
        cost: float,
        source_game_hash: int,
    ) -> bool:
        row = getattr(self.read_view, "_node_by_uid", {}).get(uid)
        if (
            row is None
            or int(getattr(row, "level", -1)) != int(MemoryLevel.M7)
            or attempts <= 0
        ):
            return False
        self._submit(
            self._existing_proposal(
                row,
                success_sum=float(max(0, successes)),
                cost_sum=float(max(0.0, cost)),
                attempt_weight=float(attempts),
                source_game_hash=int(source_game_hash),
            )
        )
        self._append_evidence(
            "strategy_efficiency",
            row,
            min(1.0, float(attempts) / max(1.0, float(cost))),
            unique=True,
            provenance_games=(int(source_game_hash),),
        )
        return True

    def record_transfer_trial(
        self,
        uid: MemoryUid,
        *,
        target_game_hash: int,
        metric_on: float,
        metric_off: float,
        formation_games: tuple[int, ...] = (),
        intervention: str = "matched_memory_ablation",
    ) -> TransferTrial:
        trial = self.transfer.record_trial(
            uid,
            target_game_hash=target_game_hash,
            metric_on=metric_on,
            metric_off=metric_off,
            formation_games=formation_games,
            intervention=intervention,
        )
        row = next(
            (r for r in self.read_view.node_records() if r.uid == uid),
            None,
        )
        if row is None:
            return trial
        if trial.passed:
            self._submit(
                self._existing_proposal(
                    row,
                    validation_state=int(ValidationState.VALIDATED),
                )
            )
            self._append_evidence(
                "transfer_trial_pass",
                row,
                trial.effect,
                validation_state=int(ValidationState.VALIDATED),
                unique=True,
                target_game_hash=trial.target_game_hash,
                provenance_games=trial.formation_games,
                causal_intervention=trial.intervention,
                effect_direction=1,
            )
            if int(row.level) == int(MemoryLevel.M4):
                self._append_evidence(
                    "concept_transfer_pass",
                    row,
                    trial.effect,
                    validation_state=int(ValidationState.VALIDATED),
                    unique=True,
                    target_game_hash=trial.target_game_hash,
                    provenance_games=trial.formation_games,
                    causal_intervention=trial.intervention,
                    effect_direction=1,
                )
        return trial

    def record_preference_probe(
        self,
        *,
        outcome_a: MemoryUid,
        outcome_b: MemoryUid,
        context_bucket: int,
        chosen_outcome: MemoryUid,
        both_reachable: bool,
        preference_influenced: bool,
    ) -> bool:
        accepted = self.preference.record_probe(
            outcome_a=outcome_a,
            outcome_b=outcome_b,
            context_bucket=context_bucket,
            chosen_outcome=chosen_outcome,
            both_reachable=both_reachable,
            preference_influenced=preference_influenced,
        )
        if not accepted:
            return False
        rows = {
            row.uid: row
            for row in self.read_view.node_records(level=MemoryLevel.M6)
        }
        chosen = rows.get(chosen_outcome)
        if chosen is not None:
            self._append_evidence(
                "preference_probe",
                chosen,
                1.0,
                unique=True,
                causal_intervention="clean_choice_probe",
            )
        for evidence in self.preference.evaluate():
            if evidence.state != "STABLE":
                continue
            preferred = rows.get(evidence.preferred)
            other = rows.get(evidence.other)
            if preferred is None or other is None:
                continue
            stable_key = (
                f"stable-preference:{preferred.uid.hex()}:"
                f"{other.uid.hex()}:{evidence.context_bucket}"
            )
            if self.ledger.contains(stable_key):
                continue
            self._submit(
                self._existing_proposal(
                    preferred,
                    parent_uid=other.uid,
                    relation_type=RelationType.PREFERENCE,
                )
            )
            watermark = int(self.current_watermark())
            games = tuple(
                sorted(
                    self.read_view.source_games(preferred)
                    | self.read_view.source_games(other)
                )
            )
            self.ledger.append(
                EvidenceRecord.for_uid(
                    stable_key,
                    preferred.uid,
                    evidence_kind="stable_preference_probe",
                    watermark=watermark,
                    raw_value=abs(evidence.strength),
                    normalized_value=min(1.0, abs(evidence.strength)),
                    developmental_stage=int(MemoryLevel.M6),
                    validation_state=int(ValidationState.VALIDATED),
                    provenance_games=games,
                    causal_intervention="clean_choice_probe",
                    effect_direction=1,
                    graph_generation=int(self.current_generation()),
                )
            )
        return True

    def record_replanning_trial(
        self,
        *,
        primary_strategy_uid: MemoryUid,
        alternative_strategy_uid: MemoryUid,
        outcome_uid: MemoryUid,
        primary_invalidated: bool,
        alternative_selected: bool,
        recovery_succeeded: bool,
    ) -> ReplanningTrial:
        rows = {
            row.uid: row
            for row in self.read_view.node_records(level=MemoryLevel.M7)
        }
        primary = rows.get(primary_strategy_uid)
        alternative = rows.get(alternative_strategy_uid)
        primary_outcome = None
        alternative_outcome = None
        if primary is not None and len(primary.key_parts) >= 3:
            primary_outcome = MemoryUid(
                int(primary.key_parts[1]), int(primary.key_parts[2])
            )
        if alternative is not None and len(alternative.key_parts) >= 3:
            alternative_outcome = MemoryUid(
                int(alternative.key_parts[1]),
                int(alternative.key_parts[2]),
            )
        outcome_preserved = bool(
            primary_outcome == outcome_uid
            and alternative_outcome == outcome_uid
            and primary_strategy_uid != alternative_strategy_uid
        )
        trial = self.replanning.record_trial(
            primary_strategy_uid=primary_strategy_uid,
            alternative_strategy_uid=alternative_strategy_uid,
            outcome_uid=outcome_uid,
            primary_invalidated=primary_invalidated,
            alternative_selected=alternative_selected,
            outcome_preserved=outcome_preserved,
            recovery_succeeded=recovery_succeeded,
        )
        if trial.valid_recovery and alternative is not None:
            self._append_evidence(
                "replanning_recovery_trial",
                alternative,
                1.0,
                unique=True,
                causal_intervention="strategy_ablation_recovery",
                effect_direction=1,
            )
        return trial

    def state_dict(self) -> dict[str, object]:
        return {
            "version": 2,
            "sequence": self._sequence,
            "evidence_sequence": self._evidence_sequence,
            "seen": [
                {
                    "kind": kind,
                    "hi": hi,
                    "lo": lo,
                    "watermark": watermark,
                }
                for (kind, hi, lo), watermark in self._seen.items()
            ],
            "ledger": self.ledger.state_dict(),
            "transfer": self.transfer.state_dict(),
            "preference": self.preference.state_dict(),
            "lifecycle": self.lifecycle.state_dict(),
            "outcomes": self.outcomes.state_dict(),
            "similarity": self.similarity.state_dict(),
        }

    def load_state(self, state: dict[str, object] | None) -> None:
        if not state:
            return
        self._sequence = max(
            self._sequence, int(state.get("sequence", 0))
        )
        self._evidence_sequence = max(
            self._evidence_sequence,
            int(state.get("evidence_sequence", 0)),
        )
        for raw in state.get("seen", []):
            if not isinstance(raw, dict):
                continue
            key = (
                str(raw.get("kind", "")),
                int(raw.get("hi", 0)),
                int(raw.get("lo", 0)),
            )
            self._seen[key] = max(
                self._seen.get(key, -1),
                int(raw.get("watermark", 0)),
            )
        self.ledger.load_state(
            state.get("ledger")
            if isinstance(state.get("ledger"), dict)
            else None
        )
        self.transfer.load_state(
            state.get("transfer")
            if isinstance(state.get("transfer"), dict)
            else None
        )
        self.preference.load_state(
            state.get("preference")
            if isinstance(state.get("preference"), dict)
            else None
        )
        self.lifecycle.load_state(
            state.get("lifecycle")
            if isinstance(state.get("lifecycle"), dict)
            else None
        )
        self.outcomes.load_state(
            state.get("outcomes")
            if isinstance(state.get("outcomes"), dict)
            else None
        )
        self.similarity.load_state(
            state.get("similarity")
            if isinstance(state.get("similarity"), dict)
            else None
        )

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            if self._pause.is_set():
                continue
            try:
                self.run_once()
            except BaseException as exc:
                self._failures += 1
                self._last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(min(1.0, self.interval_seconds))
