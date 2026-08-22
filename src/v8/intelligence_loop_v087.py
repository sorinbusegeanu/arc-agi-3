from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from v8.arena import EdgeRecord, NodeRecord
from v8.compression import CompressionEstimator
from v8.model import (
    CognitiveState,
    MemoryLevel,
    MemoryProposal,
    MemoryType,
    MemoryUid,
    RelationType,
    ValidationState,
    proposal_fingerprint,
    stable_u64,
)
from v8.normalized_memory_v086 import _M2N_MARKER, is_normalized_contingency
from v8.promotion import FormationCandidate
from v8.roles import FunctionalRoleEstimator, RoleCandidate
from v8.similarity import BoundedNeighborhoodSimilarity
from v8.structural_events import normalized_family_key


_INSTALLED = False


@dataclass(frozen=True, slots=True)
class CompressionProposal:
    uid: MemoryUid
    key_parts: tuple[int, ...]
    parents: tuple[MemoryUid, ...]
    support: int
    compression_benefit: float
    explanatory_reach: float
    contradiction: float
    future_option_delta: float


class V087GenerativeCompressionEstimator(CompressionEstimator):
    """Discover new normalized abstractions as well as score existing ones."""

    def __init__(self, *, min_reach: int = 2, min_support: int = 3, min_members: int = 2, min_benefit: float = 1.0) -> None:
        super().__init__(min_reach=min_reach)
        self.min_support = int(min_support)
        self.min_members = int(min_members)
        self.min_benefit = float(min_benefit)

    def discover(self, nodes: Iterable[NodeRecord], edges: Iterable[EdgeRecord] = (), *, budget: int = 256) -> tuple[CompressionProposal, ...]:
        del edges
        limit = max(0, int(budget))
        if limit <= 0:
            return ()
        grouped: dict[tuple[int, int], list[NodeRecord]] = defaultdict(list)
        for row in nodes:
            if not is_normalized_contingency(row) or int(row.support_count) < self.min_support:
                continue
            grouped[normalized_family_key(int(row.key_parts[0]))].append(row)
        result: list[CompressionProposal] = []
        for family_key, members in sorted(grouped.items()):
            members = list({row.uid: row for row in members}.values())
            if len(members) < self.min_members:
                continue
            total_support = sum(max(0, int(row.support_count)) for row in members)
            benefit = float(max(0, total_support - len(members)))
            if benefit <= self.min_benefit:
                continue
            kind, variant = map(int, family_key)
            key = (int(_M2N_MARKER | kind), int(variant))
            uid = MemoryUid.from_key(MemoryLevel.M2, MemoryType.FAMILY, key)
            future = sum(float(row.future_option_delta) * max(0, int(row.support_count)) for row in members) / max(1, total_support)
            result.append(CompressionProposal(uid, key, tuple(sorted(row.uid for row in members)), total_support, benefit, float(len(members)), 0.0, float(future)))
            if len(result) >= limit:
                break
        return tuple(result)


class V087RelationalRoleEstimator(FunctionalRoleEstimator):
    """Induce roles from recurring graph position rather than family identity."""

    _RELATIONS = {
        int(RelationType.EXPLAINS), int(RelationType.DEPENDS_ON), int(RelationType.ENABLES),
        int(RelationType.BLOCKS), int(RelationType.SIMILAR_TO), int(RelationType.TRANSFER_CORRESPONDENCE),
    }

    @staticmethod
    def _future_bucket(value: float) -> int:
        return 1 if float(value) > 1e-9 else -1 if float(value) < -1e-9 else 0

    @classmethod
    def _descriptor(cls, row: NodeRecord, *, edges: tuple[EdgeRecord, ...], by_uid: dict[MemoryUid, NodeRecord]) -> tuple[int, int, int, int]:
        relations: Counter[tuple[int, int, int, int]] = Counter()
        dependency: list[int] = []
        consequence: list[int] = []
        for edge in edges:
            relation = int(edge.relation_type)
            if relation not in cls._RELATIONS:
                continue
            direction = 0
            neighbor = None
            if edge.source_uid == row.uid:
                direction, neighbor = 1, by_uid.get(edge.target_uid)
            elif edge.target_uid == row.uid:
                direction, neighbor = -1, by_uid.get(edge.source_uid)
            if neighbor is None:
                continue
            relations[(direction, relation, int(neighbor.level), int(neighbor.memory_type))] += max(1, int(edge.support_count))
            if relation in {int(RelationType.DEPENDS_ON), int(RelationType.ENABLES), int(RelationType.BLOCKS)}:
                dependency.extend((direction, relation, int(neighbor.level), int(neighbor.memory_type)))
            if int(neighbor.level) >= int(MemoryLevel.M5):
                consequence.extend((int(neighbor.uid.hi), int(neighbor.uid.lo)))
        relation_parts: list[int] = []
        for key, count in sorted(relations.items()):
            relation_parts.extend((*key, min(15, int(count))))
        relation_signature = stable_u64(*relation_parts, person=b"v8.7-role-rel") if relation_parts else 0
        dependency_signature = stable_u64(*dependency, person=b"v8.7-role-dep") if dependency else 0
        consequence_signature = stable_u64(*sorted(consequence), person=b"v8.7-role-conseq") if consequence else 0
        return int(relation_signature), int(dependency_signature), int(cls._future_bucket(row.future_option_delta)), int(consequence_signature)

    def propose_relational(self, rows: Iterable[NodeRecord], edges: Iterable[EdgeRecord]) -> tuple[RoleCandidate, ...]:
        rows, edges = tuple(rows), tuple(edges)
        by_uid = {row.uid: row for row in rows}
        grouped: dict[tuple[int, int, int, int], list[NodeRecord]] = defaultdict(list)
        for row in rows:
            if int(row.level) != int(MemoryLevel.M3) or int(row.memory_type) != int(MemoryType.CARRIER) or len(row.key_parts) < 3:
                continue
            grouped[self._descriptor(row, edges=edges, by_uid=by_uid)].append(row)
        result: list[RoleCandidate] = []
        for descriptor, members in sorted(grouped.items()):
            carriers = {int(row.key_parts[1]) for row in members}
            member_uids = {row.uid for row in members}
            lower_support = {
                edge.target_uid for edge in edges
                if edge.source_uid in member_uids and int(edge.relation_type) == int(RelationType.EXPLAINS)
                and edge.target_uid in by_uid and int(by_uid[edge.target_uid].level) < int(MemoryLevel.M3)
            }
            if len(carriers) < self.min_carriers or len(lower_support) < 2:
                continue
            key = tuple(int(value) for value in descriptor)
            uid = MemoryUid.from_key(MemoryLevel.M3, MemoryType.ROLE, key)
            mask = 0
            for row in members:
                mask |= int(row.game_mask)
            result.append(RoleCandidate(uid, key, tuple(sorted(row.uid for row in members)), int(mask).bit_count()))
        return tuple(result)

    def propose(self, rows: tuple[NodeRecord, ...]) -> tuple[RoleCandidate, ...]:
        return super().propose(rows)


def _compression_to_candidate(proposal: CompressionProposal) -> FormationCandidate:
    support = max(1, int(proposal.support))
    learning = min(1.0, float(proposal.compression_benefit) / max(1.0, support))
    consistency = min(1.0, support / max(1.0, 2.0 * len(proposal.parents)))
    return FormationCandidate(proposal.uid, MemoryLevel.M2, MemoryType.FAMILY, proposal.key_parts, proposal.parents, support, consistency, learning, 0.0, float(proposal.explanatory_reach), float(proposal.future_option_delta), int(CognitiveState.PROBATION), int(ValidationState.STRUCTURAL), "generative_compression", learning)


def _validated_concept_parent(candidate: FormationCandidate, by_uid: dict[MemoryUid, NodeRecord]) -> bool:
    if int(candidate.level) != int(MemoryLevel.M5):
        return True
    for parent_uid in candidate.parents:
        parent = by_uid.get(parent_uid)
        if parent is None or int(parent.level) != int(MemoryLevel.M4) or int(parent.memory_type) != int(MemoryType.CONCEPT):
            continue
        return bool(int(parent.validation_state) == int(ValidationState.VALIDATED) and int(parent.cognitive_state) in {int(CognitiveState.VALIDATED), int(CognitiveState.REACTIVATED)})
    return False


def _install_promotion_and_abstraction() -> None:
    from v8 import behavior_recovery, compression as compression_module, peers as peers_module, peers_v82, promotion as promotion_module, roles as roles_module
    current_engine = promotion_module.EvidenceGatedPromotionEngine

    class V087PromotionEngine(current_engine):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.generative_compression = V087GenerativeCompressionEstimator(min_support=int(self.min_contingency_support), min_members=int(self.min_family_members), min_benefit=float(self.min_family_compression))

        def propose(self, nodes, edges, *, budget: int = 256):
            limit = max(0, int(budget))
            if limit <= 0:
                return ()
            nodes, edges = tuple(nodes), tuple(edges)
            by_uid = {row.uid: row for row in nodes}
            base = list(super().propose(nodes, edges, budget=limit))
            discovered = [_compression_to_candidate(item) for item in self.generative_compression.discover(nodes, edges, budget=max(1, limit // 3))]
            discovered_uids = {candidate.uid for candidate in discovered}
            normalized_present = any(is_normalized_contingency(row) for row in nodes)
            combined: list[FormationCandidate] = []
            seen: set[MemoryUid] = set()
            for candidate in discovered + base:
                if candidate.uid in seen:
                    continue
                if int(candidate.level) == int(MemoryLevel.M2) and normalized_present and candidate.uid not in discovered_uids:
                    continue
                if not _validated_concept_parent(candidate, by_uid):
                    continue
                seen.add(candidate.uid)
                combined.append(candidate)
                if len(combined) >= limit:
                    break
            return tuple(combined)

    compression_module.CompressionEstimator = V087GenerativeCompressionEstimator
    peers_module.CompressionEstimator = V087GenerativeCompressionEstimator
    roles_module.FunctionalRoleEstimator = V087RelationalRoleEstimator
    peers_module.FunctionalRoleEstimator = V087RelationalRoleEstimator
    promotion_module.EvidenceGatedPromotionEngine = V087PromotionEngine
    peers_v82.EvidenceGatedPromotionEngine = V087PromotionEngine
    behavior_recovery.CausalEvidenceGatedPromotionEngine = V087PromotionEngine
    base_parallel = peers_module.DevelopmentalPeerSupervisor._parallel_analyses

    def parallel_analyses(self, nodes, edges):
        analyses = base_parallel(self, nodes, edges)
        relational = getattr(self.roles, "propose_relational", None)
        if callable(relational):
            analyses["roles"] = relational(tuple(nodes), tuple(edges))
        return analyses

    peers_module.DevelopmentalPeerSupervisor._parallel_analyses = parallel_analyses


def _install_lifecycle_failure_semantics() -> None:
    from v8 import lifecycle as lifecycle_module
    base_fitness = lifecycle_module.LifecycleController.fitness
    base_decide = lifecycle_module.LifecycleController.decide

    def fitness(self, row):
        value = float(base_fitness(self, row))
        return max(0.0, value - 0.10) if int(row.validation_state) == int(ValidationState.FAILED) else value

    def decide(self, row):
        if int(row.level) >= int(MemoryLevel.M4) and int(row.validation_state) == int(ValidationState.FAILED) and int(row.cognitive_state) in {int(CognitiveState.ACTIVE), int(CognitiveState.VALIDATED), int(CognitiveState.REACTIVATED)}:
            return lifecycle_module.LifecycleDecision(row.uid, int(CognitiveState.QUARANTINED), int(ValidationState.FAILED), self.fitness(row), "failed empirical validation")
        decision = base_decide(self, row)
        if decision is not None and int(row.validation_state) == int(ValidationState.FAILED) and int(decision.cognitive_state) == int(CognitiveState.VALIDATED):
            return lifecycle_module.LifecycleDecision(decision.uid, int(CognitiveState.QUARANTINED), int(ValidationState.FAILED), decision.fitness, "failed evidence cannot promote")
        return decision

    lifecycle_module.LifecycleController.fitness = fitness
    lifecycle_module.LifecycleController.decide = decide


def _dependent_m5_rows(supervisor, concept_uid: MemoryUid) -> tuple[NodeRecord, ...]:
    nodes = tuple(supervisor.read_view.node_records())
    by_uid = {row.uid: row for row in nodes}
    result = []
    for edge in supervisor.read_view.edge_records():
        if edge.target_uid == concept_uid and int(edge.relation_type) == int(RelationType.EXPLAINS):
            row = by_uid.get(edge.source_uid)
            if row is not None and int(row.level) == int(MemoryLevel.M5):
                result.append(row)
    return tuple(result)


def _install_transfer_feedback() -> None:
    from v8 import peers_v82
    base_record = peers_v82.V82DevelopmentalPeerSupervisor.record_transfer_trial

    def record_transfer_trial(self, uid: MemoryUid, *, target_game_hash: int, metric_on: float, metric_off: float, formation_games: tuple[int, ...] = (), intervention: str = "matched_memory_ablation"):
        trial = base_record(self, uid, target_game_hash=target_game_hash, metric_on=metric_on, metric_off=metric_off, formation_games=formation_games, intervention=intervention)
        row = next((value for value in self.read_view.node_records() if value.uid == uid), None)
        if row is None or trial.passed:
            return trial
        trials = tuple(self.transfer.trials(uid))
        failed_targets = {item.target_game_hash for item in trials if not item.passed}
        passed_targets = {item.target_game_hash for item in trials if item.passed}
        failures, successes = len(failed_targets), len(passed_targets)
        penalty = min(1.0, 0.25 + abs(float(trial.effect)))
        if failures >= 3 and successes == 0:
            cognitive, validation = int(CognitiveState.QUARANTINED), int(ValidationState.FAILED)
        elif failures >= 2:
            cognitive, validation = int(CognitiveState.PROBATION), int(ValidationState.FAILED)
        else:
            cognitive = int(CognitiveState.PROBATION) if int(row.level) == int(MemoryLevel.M4) else int(row.cognitive_state)
            validation = int(ValidationState.TESTED)
        self._submit(self._existing_proposal(row, transfer_prior=-penalty, cognitive_state=cognitive, validation_state=validation))
        self._append_evidence("transfer_persistent_penalty", row, penalty, validation_state=validation, unique=True, target_game_hash=int(target_game_hash), provenance_games=tuple(formation_games), causal_intervention=str(intervention), effect_direction=-1)
        if int(row.level) == int(MemoryLevel.M4) and validation == int(ValidationState.FAILED):
            for dependent in _dependent_m5_rows(self, row.uid):
                self._submit(self._existing_proposal(dependent, cognitive_state=int(CognitiveState.QUARANTINED), validation_state=int(ValidationState.FAILED)))
                self._append_evidence("concept_descendant_review", dependent, 1.0, validation_state=int(ValidationState.FAILED), unique=True, effect_direction=-1)
        return trial

    peers_v82.V82DevelopmentalPeerSupervisor.record_transfer_trial = record_transfer_trial


def _local_cohort(row: NodeRecord, nodes: tuple[NodeRecord, ...], edges: tuple[EdgeRecord, ...], *, limit: int = 64) -> tuple[tuple[NodeRecord, ...], tuple[EdgeRecord, ...]]:
    adjacent = {row.uid}
    for edge in edges:
        if edge.source_uid == row.uid:
            adjacent.add(edge.target_uid)
        elif edge.target_uid == row.uid:
            adjacent.add(edge.source_uid)
    same_kind = [item for item in nodes if int(item.level) == int(row.level) and int(item.memory_type) == int(row.memory_type) and item.uid != row.uid]
    same_kind.sort(key=lambda item: (-int(item.support_count), item.uid))
    adjacent.update(item.uid for item in same_kind[: max(0, int(limit) - len(adjacent))])
    local_nodes = tuple(item for item in nodes if item.uid in adjacent)[:limit]
    local_uids = {item.uid for item in local_nodes}
    local_edges = tuple(edge for edge in edges if edge.source_uid in local_uids and edge.target_uid in local_uids)
    return local_nodes, local_edges


def _submit_formation(supervisor, candidate: FormationCandidate, by_uid) -> bool:
    if candidate.uid in by_uid:
        return False
    first_parent = candidate.parents[0] if candidate.parents else MemoryUid.zero()
    proposal = MemoryProposal(uid=candidate.uid, fingerprint=proposal_fingerprint(candidate.level, candidate.memory_type, candidate.key_parts), event_id=supervisor._event_id(), watermark=int(supervisor.current_watermark()), level=candidate.level, memory_type=candidate.memory_type, key_parts=candidate.key_parts, support_delta=max(1, int(candidate.support)), significance_sum=float(candidate.significance), learning_value_sum=float(candidate.learning_value), transfer_prior_sum=float(candidate.transfer_prior), explanatory_sum=float(candidate.explanatory_reach), future_option_sum=float(candidate.future_option_delta), score_weight=1.0, parent_uid=first_parent, relation_type=RelationType.EXPLAINS, cognitive_state=int(candidate.cognitive_state), validation_state=int(candidate.validation_state))
    supervisor._submit(proposal)
    for parent in candidate.parents[1:8]:
        supervisor._submit(MemoryProposal(uid=candidate.uid, fingerprint=proposal.fingerprint, event_id=supervisor._event_id(), watermark=int(supervisor.current_watermark()), level=candidate.level, memory_type=candidate.memory_type, key_parts=candidate.key_parts, support_delta=0, score_weight=0.0, parent_uid=parent, relation_type=RelationType.EXPLAINS, cognitive_state=int(candidate.cognitive_state), validation_state=int(candidate.validation_state)))
    return True


def process_replay_cognition(supervisor) -> dict[str, int]:
    """Run bounded extra cognition over ISF-selected memories without environment I/O."""
    nodes = tuple(supervisor.read_view.node_records())
    edges = tuple(supervisor.read_view.edge_records())
    by_uid = {row.uid: row for row in nodes}
    candidates = tuple(supervisor.replay.candidates(nodes, budget=min(max(1, int(supervisor.candidate_budget)), 16)))
    selected = processed = new_memories = revisions = correspondences = 0
    watermark = int(supervisor.current_watermark())
    for replay in candidates:
        row = by_uid.get(replay.uid)
        if row is None:
            continue
        selected += 1
        if not supervisor._fresh("replay_cognition", row.uid, watermark):
            continue
        processed += 1
        supervisor._append_evidence("replay_cognition", row, replay.priority)
        local_nodes, local_edges = _local_cohort(row, nodes, edges)
        for prediction in supervisor.prediction.evaluate(local_nodes):
            if MemoryUid(prediction.uid_hi, prediction.uid_lo) == row.uid:
                supervisor._append_evidence("replay_prediction", row, max(0.0, float(prediction.error)))
                break
        if int(row.level) <= int(MemoryLevel.M3):
            for candidate in supervisor.promotion.propose(nodes, edges, budget=min(16, int(supervisor.candidate_budget))):
                if row.uid not in candidate.parents:
                    continue
                if _submit_formation(supervisor, candidate, by_uid):
                    by_uid[candidate.uid] = candidate
                    new_memories += 1
        if int(row.level) in {int(MemoryLevel.M3), int(MemoryLevel.M4)}:
            similarity = BoundedNeighborhoodSimilarity(max_candidates=16, top_results=2, threshold=float(getattr(supervisor.similarity, "threshold", 0.65)))
            for evidence in similarity.evaluate(local_nodes, local_edges):
                source = next((item for item in local_nodes if item.uid == evidence.source_uid), None)
                target = next((item for item in local_nodes if item.uid == evidence.target_uid), None)
                if source is None or target is None:
                    continue
                supervisor._submit(supervisor._existing_proposal(source, transfer_prior=float(evidence.score), parent_uid=target.uid, relation_type=RelationType.SIMILAR_TO))
                correspondences += 1
        decision = supervisor.lifecycle.decide(row)
        if decision is not None:
            supervisor._submit(supervisor._existing_proposal(row, cognitive_state=int(decision.cognitive_state), validation_state=int(decision.validation_state)))
            revisions += 1
    return {"selected": selected, "processed": processed, "new_memories": new_memories, "revisions": revisions, "correspondences": correspondences}


def _install_cognitive_replay() -> None:
    from v8 import peers_v82
    base_run_once = peers_v82.V82DevelopmentalPeerSupervisor.run_once
    base_state_dict = peers_v82.V82DevelopmentalPeerSupervisor.state_dict
    base_load_state = peers_v82.V82DevelopmentalPeerSupervisor.load_state

    def run_once(self):
        base_run_once(self)
        cancel = getattr(self, "_v841_peer_cancel", None)
        if cancel is not None and cancel.is_set():
            return
        metrics = process_replay_cognition(self)
        totals = getattr(self, "_v87_replay_totals", {"selected": 0, "processed": 0, "new_memories": 0, "revisions": 0, "correspondences": 0})
        for key, value in metrics.items():
            totals[key] = int(totals.get(key, 0)) + int(value)
        self._v87_replay_totals = totals

    def replay_metrics(self):
        return dict(getattr(self, "_v87_replay_totals", {"selected": 0, "processed": 0, "new_memories": 0, "revisions": 0, "correspondences": 0}))

    def state_dict(self):
        state = dict(base_state_dict(self))
        state["version"] = max(3, int(state.get("version", 0)))
        state["v87_replay_totals"] = replay_metrics(self)
        return state

    def load_state(self, state):
        base_load_state(self, state)
        if isinstance(state, dict) and isinstance(state.get("v87_replay_totals"), dict):
            self._v87_replay_totals = {key: int(value) for key, value in state["v87_replay_totals"].items()}

    peers_v82.V82DevelopmentalPeerSupervisor.run_once = run_once
    peers_v82.V82DevelopmentalPeerSupervisor.replay_metrics = replay_metrics
    peers_v82.V82DevelopmentalPeerSupervisor.state_dict = state_dict
    peers_v82.V82DevelopmentalPeerSupervisor.load_state = load_state


def install_intelligence_loop_v087() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_promotion_and_abstraction()
    _install_lifecycle_failure_semantics()
    _install_transfer_feedback()
    _install_cognitive_replay()
    _INSTALLED = True
