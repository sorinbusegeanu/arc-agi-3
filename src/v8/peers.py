from __future__ import annotations

import threading
import time
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
from v8.replanning import ReplanningController, ReplanningTrial
from v8.roles import FunctionalRoleEstimator
from v8.strategies import StrategyEstimator
from v8.transfer import TransferValidator


@dataclass(frozen=True, slots=True)
class PeerMetrics:
    cycles: int
    proposals: int
    evidence_records: int
    interval_seconds: float


class DevelopmentalPeerSupervisor:
    """Independent read-snapshot operators that publish only canonical mutation proposals."""

    def __init__(
        self,
        *,
        read_view,
        submit_proposal: Callable[[MemoryProposal], None],
        watermark: Callable[[], int],
        interval_seconds: float = 0.5,
    ) -> None:
        self.read_view = read_view
        self.submit_proposal = submit_proposal
        self.current_watermark = watermark
        self.interval_seconds = float(interval_seconds)
        self.prediction = PredictionEstimator()
        self.context = ContextRefiner()
        self.roles = FunctionalRoleEstimator()
        self.future_options = FutureOptionEstimator()
        self.compression = CompressionEstimator()
        self.transfer = TransferValidator()
        self.outcomes = OutcomeEquivalenceEstimator()
        self.strategies = StrategyEstimator()
        self.preference = PreferenceEstimator()
        self.replanning = ReplanningController()
        self.lifecycle = LifecycleController()
        self.ledger = EvidenceLedger()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sequence = 0
        self._evidence_sequence = 0
        self._seen: dict[tuple[str, int, int], int] = {}
        self._cycles = 0
        self._proposals = 0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="v8-developmental-peers", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def set_interval(self, seconds: float) -> None:
        self.interval_seconds = max(0.05, float(seconds))

    def metrics(self) -> PeerMetrics:
        return PeerMetrics(
            self._cycles,
            self._proposals,
            len(self.ledger.cut(self.current_watermark())),
            self.interval_seconds,
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
    ) -> None:
        watermark = int(self.current_watermark())
        suffix = ""
        if unique:
            self._evidence_sequence += 1
            suffix = f":{self._evidence_sequence}"
        evidence_id = f"{kind}:{row.uid.hex()}:{watermark}{suffix}"
        self.ledger.append(
            EvidenceRecord.for_uid(
                evidence_id,
                row.uid,
                evidence_kind=kind,
                watermark=watermark,
                raw_value=float(value),
                normalized_value=max(0.0, min(1.0, float(value))),
                developmental_stage=int(row.level),
                validation_state=int(
                    row.validation_state if validation_state is None else validation_state
                ),
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
        cognitive_state: int = -1,
        validation_state: int = -1,
        parent_uid: MemoryUid | None = None,
        relation_type: RelationType = RelationType.EXPLAINS,
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
            parent_uid=MemoryUid.zero() if parent_uid is None else parent_uid,
            relation_type=relation_type,
            cognitive_state=int(cognitive_state),
            validation_state=int(validation_state),
        )

    def _submit(self, proposal: MemoryProposal) -> None:
        self.submit_proposal(proposal)
        self._proposals += 1

    def run_once(self) -> None:
        nodes = self.read_view.node_records()
        edges = self.read_view.edge_records()
        by_uid = {row.uid: row for row in nodes}

        # Base developmental evidence.
        for row in nodes:
            if row.support_count < 2:
                continue
            kind = {
                int(MemoryLevel.M1): "contingency_recurrence",
                int(MemoryLevel.M2): "family_recurrence",
                int(MemoryLevel.M3): "carrier_candidate"
                if int(row.memory_type) == int(MemoryType.CARRIER)
                else "role_candidate",
                int(MemoryLevel.M4): "concept_candidate",
                int(MemoryLevel.M5): "consequence_structure",
                int(MemoryLevel.M6): "outcome_equivalence",
                int(MemoryLevel.M7): "strategy_reuse",
            }.get(int(row.level))
            if kind and self._fresh(kind, row.uid, row.updated_watermark):
                self._append_evidence(kind, row, min(1.0, row.support_count / 4.0))

        # Prediction violation estimation.
        for evidence in self.prediction.evaluate(nodes):
            uid = MemoryUid(evidence.uid_hi, evidence.uid_lo)
            row = by_uid.get(uid)
            if row is None or not self._fresh("prediction", uid, row.updated_watermark):
                continue
            self._submit(self._existing_proposal(row, prediction_error=evidence.error))
            self._append_evidence("supported_prediction", row, 1.0)
            if evidence.error > 0.0:
                self._append_evidence("prediction_violation", row, evidence.error)

        # Contradictions first propose contextual refinement rather than replacement.
        for refinement in self.context.propose(nodes):
            source = by_uid.get(refinement.source_uid)
            if source is None or not self._fresh(
                "context", refinement.candidate_uid, source.updated_watermark
            ):
                continue
            proposal = MemoryProposal(
                uid=refinement.candidate_uid,
                fingerprint=proposal_fingerprint(
                    MemoryLevel.M3, MemoryType.CONTEXTUAL_ROLE, refinement.key_parts
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
                "context_refinement", source, refinement.contradiction_rate
            )

        # Carrier -> cross-carrier role formation.
        for candidate in self.roles.propose(nodes):
            candidate_watermarks = [
                by_uid[u].updated_watermark for u in candidate.carriers if u in by_uid
            ]
            if not candidate.carriers or not candidate_watermarks:
                continue
            if not self._fresh("role", candidate.uid, max(candidate_watermarks)):
                continue
            proposal = MemoryProposal(
                uid=candidate.uid,
                fingerprint=proposal_fingerprint(
                    MemoryLevel.M3, MemoryType.ROLE, candidate.key_parts
                ),
                event_id=self._event_id(),
                watermark=int(self.current_watermark()),
                level=MemoryLevel.M3,
                memory_type=MemoryType.ROLE,
                key_parts=candidate.key_parts,
                support_delta=len(candidate.carriers),
                explanatory_sum=float(len(candidate.carriers)),
                transfer_prior_sum=min(1.0, candidate.game_evidence_count / 2.0),
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

        # Learned bounded future-option evidence over the M1 transition graph.
        for evidence in self.future_options.evaluate(nodes):
            row = by_uid.get(evidence.uid)
            if row is None or not self._fresh("fo", row.uid, row.updated_watermark):
                continue
            self._submit(
                self._existing_proposal(row, future_option=float(evidence.delta))
            )
            self._append_evidence(
                "future_option_estimate",
                row,
                min(1.0, abs(evidence.delta) / 4.0),
            )

        # Explanatory reach and supersession evidence; no physical deletion here.
        for evidence in self.compression.evaluate(nodes, edges):
            row = by_uid.get(evidence.uid)
            if row is None or not self._fresh(
                "compression", row.uid, row.updated_watermark
            ):
                continue
            self._submit(
                self._existing_proposal(
                    row, explanatory=float(evidence.explanatory_reach)
                )
            )
            kind = (
                "family_compression"
                if int(row.level) == int(MemoryLevel.M2)
                else "compression"
            )
            self._append_evidence(
                kind, row, min(1.0, evidence.compression_benefit / 4.0)
            )
            for target in evidence.superseded[:8]:
                self._submit(
                    self._existing_proposal(
                        row,
                        parent_uid=target,
                        relation_type=RelationType.SUPERSEDES,
                    )
                )

        # Structural transfer candidacy remains distinct from empirical trials.
        for candidate in self.transfer.candidates(nodes):
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
            )

        # Outcome and strategy structure. Preference is deliberately NOT inferred from
        # strategy frequency here; only explicit causally clean preference probes may
        # produce preference evidence or PREFERENCE edges.
        classes = self.outcomes.rebuild(nodes)
        for outcome in classes:
            row = by_uid.get(outcome.uid)
            if (
                row is not None
                and row.support_count >= 2
                and self._fresh("outcome", row.uid, row.updated_watermark)
            ):
                self._append_evidence(
                    "outcome_equivalence", row, min(1.0, row.support_count / 4.0)
                )
        by_outcome = self.strategies.by_outcome(nodes)
        for _outcome_uid, alternatives in by_outcome.items():
            if len(alternatives) >= 2:
                for strategy in alternatives:
                    row = by_uid.get(strategy.uid)
                    if row is not None and self._fresh(
                        "alternative", row.uid, row.updated_watermark
                    ):
                        self._append_evidence(
                            "alternative_strategy",
                            row,
                            min(1.0, len(alternatives) / 3.0),
                        )

        # Hysteretic lifecycle decisions are proposals owned by canonical shards.
        for row in nodes:
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

        self._cycles += 1

    def record_transfer_trial(
        self,
        uid: MemoryUid,
        *,
        target_game_hash: int,
        metric_on: float,
        metric_off: float,
    ) -> None:
        trial = self.transfer.record_trial(
            uid,
            target_game_hash=target_game_hash,
            metric_on=metric_on,
            metric_off=metric_off,
        )
        row = next((r for r in self.read_view.node_records() if r.uid == uid), None)
        if row is None:
            return
        if trial.passed:
            self._submit(
                self._existing_proposal(
                    row, validation_state=int(ValidationState.VALIDATED)
                )
            )
            self._append_evidence(
                "transfer_trial_pass",
                row,
                min(1.0, max(0.0, trial.effect)),
                validation_state=int(ValidationState.VALIDATED),
                unique=True,
            )
            if int(row.level) == int(MemoryLevel.M4):
                self._append_evidence(
                    "concept_transfer_pass",
                    row,
                    min(1.0, max(0.0, trial.effect)),
                    validation_state=int(ValidationState.VALIDATED),
                    unique=True,
                )

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
        rows = {row.uid: row for row in self.read_view.node_records(level=MemoryLevel.M6)}
        chosen = rows.get(chosen_outcome)
        if chosen is not None:
            self._append_evidence("preference_probe", chosen, 1.0, unique=True)

        for evidence in self.preference.evaluate():
            if evidence.state != "STABLE":
                continue
            preferred = rows.get(evidence.preferred)
            other = rows.get(evidence.other)
            if preferred is None or other is None:
                continue
            stable_key = f"stable-preference:{preferred.uid.hex()}:{other.uid.hex()}:{evidence.context_bucket}"
            if stable_key in self.ledger._ids:
                continue
            self._submit(
                self._existing_proposal(
                    preferred,
                    parent_uid=other.uid,
                    relation_type=RelationType.PREFERENCE,
                )
            )
            watermark = int(self.current_watermark())
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
        rows = {row.uid: row for row in self.read_view.node_records(level=MemoryLevel.M7)}
        primary = rows.get(primary_strategy_uid)
        alternative = rows.get(alternative_strategy_uid)
        primary_outcome = None
        alternative_outcome = None
        if primary is not None and len(primary.key_parts) >= 3:
            primary_outcome = MemoryUid(int(primary.key_parts[1]), int(primary.key_parts[2]))
        if alternative is not None and len(alternative.key_parts) >= 3:
            alternative_outcome = MemoryUid(
                int(alternative.key_parts[1]), int(alternative.key_parts[2])
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
                "replanning_recovery_trial", alternative, 1.0, unique=True
            )
        return trial

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.run_once()
            except Exception:
                # Peer diagnostics must not crash live cognition. Runtime health reports
                # peer failures separately through missing evidence and cycle metrics.
                time.sleep(min(1.0, self.interval_seconds))
