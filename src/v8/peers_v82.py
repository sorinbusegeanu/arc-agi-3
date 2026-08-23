from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from v8.carriers import CarrierEstimator
from v8.concept_validation import ConceptValidator
from v8.developmental_cut import DevelopmentalGenerationCut, capture_developmental_cut
from v8.model import (
    CognitiveState,
    MemoryLevel,
    MemoryProposal,
    MemoryType,
    MemoryUid,
    RelationType,
    ValidationState,
    proposal_fingerprint,
)
from v8.peers import DevelopmentalPeerSupervisor
from v8.promotion import EvidenceGatedPromotionEngine, FormationCandidate
from v8.structural_correspondence import StructuralCorrespondenceEstimator
from v8.transfer import TransferTrial


_LINEAGE = {
    int(RelationType.PROVENANCE),
    int(RelationType.EXPLAINS),
    int(RelationType.LEADS_TO),
    int(RelationType.CONTEXT_REFINES),
    int(RelationType.SUPERSEDES),
    int(RelationType.DEPENDS_ON),
}


class _FrozenCutReadView:
    """Read-only adapter so one peer interval consumes one declared shard-vector cut."""

    def __init__(
        self,
        cut: DevelopmentalGenerationCut,
        *,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self.cut = cut
        self._nodes = cut.nodes
        self._edges = cut.edges
        self._cancel_event = cancel_event
        self._by_uid = {row.uid: row for row in self._nodes}
        self._direct_games: dict[MemoryUid, set[int]] = {}
        self._parents: dict[MemoryUid, set[MemoryUid]] = {}
        self.cancelled = False
        for edge_index, edge in enumerate(self._edges):
            if (
                edge_index % 256 == 0
                and cancel_event is not None
                and cancel_event.is_set()
            ):
                self.cancelled = True
                return
            relation = int(edge.relation_type)
            if relation == int(RelationType.GAME_PROVENANCE) and int(edge.target_uid.hi) == 0:
                self._direct_games.setdefault(edge.source_uid, set()).add(int(edge.target_uid.lo))
            elif relation in _LINEAGE:
                self._parents.setdefault(edge.source_uid, set()).add(edge.target_uid)

    def node_records(self, *, level: MemoryLevel | int | None = None):
        if level is None:
            return self._nodes
        wanted = int(level)
        return tuple(row for row in self._nodes if int(row.level) == wanted)

    def edge_records(self):
        return self._edges

    def source_games(self, uid: MemoryUid, *, max_depth: int = 8) -> frozenset[int]:
        if self._cancel_event is not None and self._cancel_event.is_set():
            self.cancelled = True
            return frozenset()
        games = set(self._direct_games.get(uid, ()))
        frontier = {uid}
        visited = {uid}
        for _depth in range(max(0, int(max_depth))):
            following: set[MemoryUid] = set()
            for current in frontier:
                if self._cancel_event is not None and self._cancel_event.is_set():
                    self.cancelled = True
                    return frozenset()
                for parent in self._parents.get(current, ()):
                    games.update(self._direct_games.get(parent, ()))
                    if parent not in visited:
                        visited.add(parent)
                        following.add(parent)
            if not following:
                break
            frontier = following
        return frozenset(games)


@dataclass(frozen=True, slots=True)
class _FormationIdentity:
    uid: MemoryUid
    fingerprint: int
    level: int
    memory_type: int
    key_parts: tuple[int, ...]


class V82DevelopmentalPeerSupervisor(DevelopmentalPeerSupervisor):
    """Paper-v0.5.2 semantics over the v8.1 RAM-authoritative runtime."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.promotion = EvidenceGatedPromotionEngine()
        self.carrier_activation = CarrierEstimator()
        self.correspondence = StructuralCorrespondenceEstimator()
        self.concepts = ConceptValidator(self.transfer)
        self._v82_run_lock = threading.Lock()
        self._last_developmental_cut: DevelopmentalGenerationCut | None = None

    @property
    def last_developmental_cut(self) -> DevelopmentalGenerationCut | None:
        return self._last_developmental_cut

    def wait_idle(self, timeout: float) -> bool:
        """Wait for the complete frozen-cut cycle, including v8.2 formation work."""
        deadline = time.monotonic() + max(0.0, float(timeout))
        acquired = self._v82_run_lock.acquire(timeout=max(0.0, deadline - time.monotonic()))
        if not acquired:
            return False
        try:
            return super().wait_idle(max(0.0, deadline - time.monotonic()))
        finally:
            self._v82_run_lock.release()

    def _fresh(self, kind: str, uid: MemoryUid, watermark: int) -> bool:
        # Carrier recurrence alone is candidate evidence.  It becomes emergence
        # evidence only after explanatory/compression utility is positive.
        if kind == "carrier_emergence":
            row = next(
                (
                    value
                    for value in self.read_view.node_records(level=MemoryLevel.M3)
                    if value.uid == uid
                ),
                None,
            )
            if row is None:
                return False
            hypotheses = self.carrier_activation.evaluate((row,))
            if not hypotheses or not hypotheses[0].activatable:
                return False
        return super()._fresh(kind, uid, watermark)

    @staticmethod
    def _relation_for(candidate: FormationCandidate, *, extra_parent: bool = False) -> RelationType:
        if candidate.level == MemoryLevel.M7:
            return RelationType.DEPENDS_ON if extra_parent else RelationType.LEADS_TO
        return RelationType.EXPLAINS

    def _formation_identity(self, candidate: FormationCandidate) -> _FormationIdentity:
        return _FormationIdentity(
            candidate.uid,
            proposal_fingerprint(candidate.level, candidate.memory_type, candidate.key_parts),
            int(candidate.level),
            int(candidate.memory_type),
            candidate.key_parts,
        )

    @staticmethod
    def _formation_future_option(
        candidate: FormationCandidate,
        by_uid: dict[MemoryUid, object],
    ) -> float:
        """Preserve an already-learned role FO bucket when forming a concept.

        Base role construction stores the structurally learned future-option bucket in
        the role canonical key.  That bucket is causal input to later consequence and
        outcome structure even when no score delta was accumulated on the role row.
        """
        if candidate.level == MemoryLevel.M4 and candidate.parents:
            parent = by_uid.get(candidate.parents[0])
            if (
                parent is not None
                and int(parent.memory_type) == int(MemoryType.ROLE)
                and len(parent.key_parts) >= 2
            ):
                return float(max(-1, min(1, int(parent.key_parts[1]))))
        return float(candidate.future_option_delta)

    def _process_formation(
        self,
        cut: DevelopmentalGenerationCut,
        frozen: _FrozenCutReadView,
    ) -> None:
        by_uid = {row.uid: row for row in cut.nodes}
        for candidate in self.promotion.propose(
            cut.nodes,
            cut.edges,
            budget=self.candidate_budget,
        ):
            parent_watermark = max(
                (by_uid[uid].updated_watermark for uid in candidate.parents if uid in by_uid),
                default=cut.watermark,
            )
            freshness = f"v82-formation:{int(candidate.level)}:{int(candidate.memory_type)}"
            if not self._fresh(freshness, candidate.uid, parent_watermark):
                continue
            identity = self._formation_identity(candidate)
            weight = max(1.0, float(candidate.support))
            first_parent = candidate.parents[0] if candidate.parents else MemoryUid.zero()
            future_option_delta = self._formation_future_option(candidate, by_uid)
            proposal = MemoryProposal(
                uid=candidate.uid,
                fingerprint=identity.fingerprint,
                event_id=self._event_id(),
                watermark=int(cut.watermark),
                level=candidate.level,
                memory_type=candidate.memory_type,
                key_parts=candidate.key_parts,
                support_delta=max(1, int(candidate.support)),
                significance_sum=float(candidate.significance) * weight,
                learning_value_sum=float(candidate.learning_value) * weight,
                transfer_prior_sum=float(candidate.transfer_prior) * weight,
                explanatory_sum=float(candidate.explanatory_reach) * weight,
                future_option_sum=future_option_delta * weight,
                score_weight=weight,
                parent_uid=first_parent,
                relation_type=self._relation_for(candidate),
                cognitive_state=int(candidate.cognitive_state),
                validation_state=int(candidate.validation_state),
            )
            self._submit(proposal)
            for parent in candidate.parents[1:8]:
                self._submit(
                    self._existing_proposal(
                        identity,
                        parent_uid=parent,
                        relation_type=self._relation_for(candidate, extra_parent=True),
                    )
                )
            provenance_games: set[int] = set()
            for parent in candidate.parents:
                provenance_games.update(frozen.source_games(parent))
            self._append_evidence(
                candidate.evidence_kind,
                candidate,
                candidate.evidence_value,
                validation_state=int(candidate.validation_state),
                provenance_games=tuple(sorted(provenance_games)),
            )

    def _process_correspondence(
        self,
        cut: DevelopmentalGenerationCut,
        frozen: _FrozenCutReadView,
    ) -> None:
        by_uid = {row.uid: row for row in cut.nodes}
        for evidence in self.correspondence.evaluate(
            cut.nodes,
            cut.edges,
            budget=self.candidate_budget,
        ):
            source = by_uid.get(evidence.source_uid)
            target = by_uid.get(evidence.target_uid)
            if source is None or target is None:
                continue
            freshness = f"v82-correspondence:{target.uid.hex()}"
            if not self._fresh(freshness, source.uid, evidence.evidence_watermark):
                continue
            score = max(0.0, min(1.0, 1.0 - float(evidence.epsilon_struct)))
            games = tuple(sorted(frozen.source_games(source.uid) | frozen.source_games(target.uid)))
            if evidence.admissible:
                self._submit(
                    self._existing_proposal(
                        source,
                        transfer_prior=score,
                        parent_uid=target.uid,
                        relation_type=RelationType.TRANSFER_CORRESPONDENCE,
                    )
                )
                self._append_evidence(
                    "structural_correspondence",
                    source,
                    score,
                    validation_state=int(ValidationState.STRUCTURAL),
                    provenance_games=games,
                )
            else:
                self._append_evidence(
                    "structural_correspondence_fail",
                    source,
                    max(0.0, min(1.0, evidence.epsilon_struct)),
                    validation_state=int(ValidationState.STRUCTURAL),
                    provenance_games=games,
                    effect_direction=-1,
                )

    def run_once(self) -> None:
        if not self._v82_run_lock.acquire(blocking=False):
            return
        live_read_view = self.read_view
        try:
            cancel = getattr(self, "_v841_peer_cancel", None)

            def cancelled() -> bool:
                return bool(cancel is not None and cancel.is_set())

            if cancelled():
                return
            cut = capture_developmental_cut(
                live_read_view,
                generation=int(self.current_generation()),
                watermark=int(self.current_watermark()),
            )
            if cancelled():
                return
            self._last_developmental_cut = cut
            frozen = _FrozenCutReadView(cut, cancel_event=cancel)
            if frozen.cancelled or cancelled():
                return
            self.read_view = frozen
            self._process_formation(cut, frozen)
            if cancelled():
                return
            self._process_correspondence(cut, frozen)
            if cancelled():
                return
            # Base peer operators run against the exact same frozen cut.  They can
            # emit proposals, but none of those writes become inputs to this interval.
            super().run_once()
        finally:
            self.read_view = live_read_view
            self._v82_run_lock.release()

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
        row = next((value for value in self.read_view.node_records() if value.uid == uid), None)
        if row is None:
            return trial

        if not trial.passed:
            self._append_evidence(
                "transfer_trial_fail",
                row,
                abs(trial.effect),
                unique=True,
                target_game_hash=trial.target_game_hash,
                provenance_games=trial.formation_games,
                causal_intervention=trial.intervention,
                effect_direction=-1,
            )
            if int(row.level) == int(MemoryLevel.M4):
                self._append_evidence(
                    "concept_transfer_fail",
                    row,
                    abs(trial.effect),
                    unique=True,
                    target_game_hash=trial.target_game_hash,
                    provenance_games=trial.formation_games,
                    causal_intervention=trial.intervention,
                    effect_direction=-1,
                )
            return trial

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
            validation = self.concepts.evaluate(
                uid,
                row=row,
                edges=tuple(self.read_view.edge_records()),
            )
            if not validation.validated:
                return trial
            self._submit(
                self._existing_proposal(
                    row,
                    cognitive_state=int(CognitiveState.VALIDATED),
                    validation_state=int(ValidationState.VALIDATED),
                )
            )
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
        else:
            self._submit(
                self._existing_proposal(
                    row,
                    validation_state=int(ValidationState.VALIDATED),
                )
            )
        return trial
