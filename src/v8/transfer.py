from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Callable

from v8.arena import EdgeRecord, NodeRecord
from v8 import information_flow_diagnostics as flow
from v8.model import MemoryLevel, MemoryUid, RelationType


@dataclass(frozen=True, slots=True)
class TransferCandidate:
    uid: MemoryUid
    game_evidence_count: int
    structural_score: float
    formation_games: tuple[int, ...] = ()
    correspondence_uid: MemoryUid = MemoryUid(0, 0)
    correspondence_games: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class TransferTrial:
    uid: MemoryUid
    target_game_hash: int
    metric_on: float
    metric_off: float
    effect: float
    passed: bool
    formation_games: tuple[int, ...] = ()
    intervention: str = "matched_memory_ablation"


class TransferValidator:
    """Separate prospective reuse, structural admissibility and empirical transfer."""

    def __init__(self, *, effect_threshold: float = 0.0) -> None:
        self.effect_threshold = float(effect_threshold)
        self._trials: dict[MemoryUid, list[TransferTrial]] = {}

    @staticmethod
    def _provenance_from_edges(
        uids: tuple[MemoryUid, ...],
        edges: tuple[EdgeRecord, ...],
        *,
        max_depth: int = 8,
        cancel_event=None,
    ) -> dict[MemoryUid, tuple[int, ...]] | None:
        direct: dict[MemoryUid, set[int]] = {}
        parents: dict[MemoryUid, set[MemoryUid]] = {}
        lineage = {
            int(RelationType.PROVENANCE),
            int(RelationType.EXPLAINS),
            int(RelationType.CONTEXT_REFINES),
            int(RelationType.SUPERSEDES),
            int(RelationType.LEADS_TO),
        }
        for index, edge in enumerate(edges):
            if (
                index % 4096 == 0
                and cancel_event is not None
                and cancel_event.is_set()
            ):
                return None
            relation = int(edge.relation_type)
            if relation == int(RelationType.GAME_PROVENANCE) and int(edge.target_uid.hi) == 0:
                direct.setdefault(edge.source_uid, set()).add(int(edge.target_uid.lo))
            elif relation in lineage:
                parents.setdefault(edge.source_uid, set()).add(edge.target_uid)

        result: dict[MemoryUid, tuple[int, ...]] = {}
        for index, uid in enumerate(uids):
            if (
                index % 4096 == 0
                and cancel_event is not None
                and cancel_event.is_set()
            ):
                return None
            found = set(direct.get(uid, ()))
            frontier = {uid}
            visited = {uid}
            for _depth in range(max(0, int(max_depth))):
                if cancel_event is not None and cancel_event.is_set():
                    return None
                following: set[MemoryUid] = set()
                for current in frontier:
                    if cancel_event is not None and cancel_event.is_set():
                        return None
                    for parent in parents.get(current, ()):
                        found.update(direct.get(parent, ()))
                        if parent not in visited:
                            visited.add(parent)
                            following.add(parent)
                if not following:
                    break
                frontier = following
            result[uid] = tuple(sorted(found))
        return result

    @staticmethod
    def _provenance_from_indexes(
        uids: tuple[MemoryUid, ...],
        direct: dict[MemoryUid, set[int]],
        parents: dict[MemoryUid, set[MemoryUid]],
        *,
        max_depth: int = 8,
        cancel_event=None,
    ) -> dict[MemoryUid, tuple[int, ...]] | None:
        result: dict[MemoryUid, tuple[int, ...]] = {}
        for index, uid in enumerate(uids):
            if (
                index % 4096 == 0
                and cancel_event is not None
                and cancel_event.is_set()
            ):
                return None
            found = set(direct.get(uid, ()))
            frontier = {uid}
            visited = {uid}
            for _depth in range(max(0, int(max_depth))):
                if cancel_event is not None and cancel_event.is_set():
                    return None
                following: set[MemoryUid] = set()
                for current in frontier:
                    if cancel_event is not None and cancel_event.is_set():
                        return None
                    for parent in parents.get(current, ()):
                        found.update(direct.get(parent, ()))
                        if parent not in visited:
                            visited.add(parent)
                            following.add(parent)
                if not following:
                    break
                frontier = following
            result[uid] = tuple(sorted(found))
        return result

    def candidates(
        self,
        rows: tuple[NodeRecord, ...],
        edges: tuple[EdgeRecord, ...] = (),
        *,
        provenance: Callable[[MemoryUid], frozenset[int]] | None = None,
        cancel_event=None,
    ) -> tuple[TransferCandidate, ...]:
        if cancel_event is not None and cancel_event.is_set():
            flow.emit(
                "transfer", "candidate_selection", input_count=0, output_count=0,
                rejection_counts={"cancelled": 1},
            )
            return ()
        eligible = {
            row.uid: row
            for row in rows
            if int(row.level) in {int(MemoryLevel.M3), int(MemoryLevel.M4)}
        }
        if not eligible:
            flow.emit(
                "transfer", "candidate_selection", input_count=len(rows), output_count=0,
                rejection_counts={"m3_or_m4_unavailable": len(rows)},
                fields={"m3_available": False, "m4_available": False},
            )
            return ()
        m3_available = any(int(row.level) == int(MemoryLevel.M3) for row in eligible.values())
        m4_available = any(int(row.level) == int(MemoryLevel.M4) for row in eligible.values())

        bound_read_view = None
        cached_direct = None
        cached_parents = None
        if provenance is not None:
            owner = getattr(provenance, "__self__", None)
            edge_records = getattr(owner, "edge_records", None)
            if callable(edge_records):
                bound_read_view = owner
                transfer_version = tuple(
                    getattr(bound_read_view, "_v839_transfer_version", ())
                )
                provenance_version = tuple(
                    getattr(bound_read_view, "_v839_provenance_version", ())
                )
                direct = getattr(bound_read_view, "_v839_direct_games", None)
                parents = getattr(
                    bound_read_view, "_v839_provenance_parents", None
                )
                transfer_edges = getattr(
                    bound_read_view, "_v839_transfer_edges", None
                )
                if (
                    transfer_version
                    and transfer_version == provenance_version
                    and isinstance(direct, dict)
                    and isinstance(parents, dict)
                    and isinstance(transfer_edges, tuple)
                ):
                    if not edges:
                        edges = transfer_edges
                    cached_direct = direct
                    cached_parents = parents
                elif not edges:
                    edges = tuple(edge_records())
                if cancel_event is not None and cancel_event.is_set():
                    return ()

        # Compatibility fallback for pure unit use without a graph.  In the live
        # runtime, where edges are present, formal TRANSFER_CORRESPONDENCE is required.
        if not edges and provenance is None:
            result = []
            rejected: Counter[str] = Counter()
            examples = []
            for row in eligible.values():
                games = int(row.game_evidence_count)
                recurrence = min(1.0, games / 4.0) * min(
                    1.0, max(1, row.support_count) / 8.0
                )
                if games >= 2 and recurrence > 0.0:
                    result.append(TransferCandidate(row.uid, games, recurrence))
                    decision = "admissible"
                else:
                    decision = "insufficient_game_recurrence"
                    rejected[decision] += 1
                if len(examples) < flow.MAX_EXAMPLES:
                    examples.append({
                        "source_world": None,
                        "candidate_target_world": None,
                        "candidate_uid": flow.uid_text(row.uid),
                        "correspondence_score": recurrence,
                        "provenance_distinct": None,
                        "m3_available": m3_available,
                        "m4_available": m4_available,
                        "held_out_eligibility": None,
                        "scheduler_decision": "not_reached",
                        "rejection_reason": None if decision == "admissible" else decision,
                    })
            output = tuple(result)
            flow.add_counters("transfer", admissible_candidates=len(output))
            flow.emit(
                "transfer", "candidate_selection", input_count=len(eligible),
                output_count=len(output), rejection_counts=rejected, examples=examples,
                fields={"m3_available": m3_available, "m4_available": m4_available,
                        "compatibility_fallback": True},
            )
            return output

        relevant_uids: set[MemoryUid] = set()
        for index, edge in enumerate(edges):
            if (
                index % 4096 == 0
                and cancel_event is not None
                and cancel_event.is_set()
            ):
                return ()
            if int(edge.relation_type) != int(RelationType.TRANSFER_CORRESPONDENCE):
                continue
            if edge.source_uid in eligible:
                relevant_uids.add(edge.source_uid)
            if edge.target_uid in eligible:
                relevant_uids.add(edge.target_uid)
        if not relevant_uids:
            count = sum(
                int(int(edge.relation_type) == int(RelationType.TRANSFER_CORRESPONDENCE))
                for edge in edges
            )
            flow.emit(
                "transfer", "candidate_selection", input_count=count, output_count=0,
                rejection_counts={"source_or_target_not_m3_or_m4": count} if count else {},
                fields={"m3_available": m3_available, "m4_available": m4_available},
            )
            return ()

        requested_uids = tuple(sorted(relevant_uids))
        graph_games = None
        if cached_direct is not None and cached_parents is not None:
            graph_games = self._provenance_from_indexes(
                requested_uids,
                cached_direct,
                cached_parents,
                cancel_event=cancel_event,
            )
            if graph_games is None:
                return ()
        elif bound_read_view is not None:
            graph_games = self._provenance_from_edges(
                requested_uids, edges, cancel_event=cancel_event
            )
            if graph_games is None:
                return ()

        def games(uid: MemoryUid) -> tuple[int, ...]:
            row = eligible.get(uid)
            if row is None:
                return ()
            if graph_games is not None:
                return graph_games.get(uid, ())
            if provenance is not None:
                return tuple(sorted(provenance(uid)))
            mask = int(row.game_mask)
            return tuple(index for index in range(64) if mask & (1 << index))

        best: dict[MemoryUid, TransferCandidate] = {}
        considered = 0
        rejected: Counter[str] = Counter()
        examples: list[dict[str, object]] = []

        def example(edge, *, uid=None, own_games=(), other_games=(), reason=None) -> None:
            if len(examples) >= flow.MAX_EXAMPLES:
                return
            examples.append(
                {
                    "source_world": list(own_games),
                    "candidate_target_world": list(other_games),
                    "candidate_uid": flow.uid_text(uid if uid is not None else edge.source_uid),
                    "correspondence_uid": flow.uid_text(edge.target_uid),
                    "correspondence_score": float(edge.score),
                    "provenance_distinct": bool(own_games and other_games and set(own_games) != set(other_games)),
                    "m3_available": m3_available,
                    "m4_available": m4_available,
                    "held_out_eligibility": None,
                    "scheduler_decision": "not_reached",
                    "rejection_reason": reason,
                }
            )

        for index, edge in enumerate(edges):
            if (
                index % 4096 == 0
                and cancel_event is not None
                and cancel_event.is_set()
            ):
                return ()
            if int(edge.relation_type) != int(RelationType.TRANSFER_CORRESPONDENCE):
                continue
            if edge.source_uid not in eligible or edge.target_uid not in eligible:
                considered += 1
                rejected["source_or_target_not_m3_or_m4"] += 1
                example(edge, reason="source_or_target_not_m3_or_m4")
                continue
            score = float(edge.score)
            if score <= 0.0:
                considered += 1
                rejected["nonpositive_correspondence_score"] += 1
                example(edge, reason="nonpositive_correspondence_score")
                continue
            left_games = games(edge.source_uid)
            right_games = games(edge.target_uid)
            if not left_games:
                considered += 1
                rejected["source_provenance_missing"] += 1
                example(edge, own_games=left_games, other_games=right_games,
                        reason="source_provenance_missing")
                continue
            if not right_games:
                considered += 1
                rejected["target_provenance_missing"] += 1
                example(edge, own_games=left_games, other_games=right_games,
                        reason="target_provenance_missing")
                continue
            left_set, right_set = set(left_games), set(right_games)
            if left_set == right_set:
                considered += 1
                rejected["provenance_not_distinct"] += 1
                example(edge, own_games=left_games, other_games=right_games,
                        reason="provenance_not_distinct")
                continue

            for uid, own_games, other_uid, other_games in (
                (edge.source_uid, left_games, edge.target_uid, right_games),
                (edge.target_uid, right_games, edge.source_uid, left_games),
            ):
                considered += 1
                if not (set(other_games) - set(own_games)):
                    rejected["no_new_target_world"] += 1
                    example(edge, uid=uid, own_games=own_games, other_games=other_games,
                            reason="no_new_target_world")
                    continue
                candidate = TransferCandidate(
                    uid=uid,
                    game_evidence_count=len(own_games),
                    structural_score=score,
                    formation_games=own_games,
                    correspondence_uid=other_uid,
                    correspondence_games=other_games,
                )
                prior = best.get(uid)
                if prior is None or (
                    candidate.structural_score,
                    candidate.correspondence_uid,
                ) > (
                    prior.structural_score,
                    prior.correspondence_uid,
                ):
                    if prior is not None:
                        rejected["superseded_by_stronger_correspondence"] += 1
                    best[uid] = candidate
                else:
                    rejected["superseded_by_stronger_correspondence"] += 1
                    example(edge, uid=uid, own_games=own_games, other_games=other_games,
                            reason="superseded_by_stronger_correspondence")
        output = tuple(best[uid] for uid in sorted(best))
        for candidate in output:
            if len(examples) >= flow.MAX_EXAMPLES:
                break
            examples.append(
                {
                    "source_world": list(candidate.formation_games),
                    "candidate_target_world": list(candidate.correspondence_games),
                    "candidate_uid": flow.uid_text(candidate.uid),
                    "correspondence_uid": flow.uid_text(candidate.correspondence_uid),
                    "correspondence_score": candidate.structural_score,
                    "provenance_distinct": set(candidate.formation_games) != set(candidate.correspondence_games),
                    "m3_available": m3_available,
                    "m4_available": m4_available,
                    "held_out_eligibility": None,
                    "scheduler_decision": "candidate_admissible",
                    "rejection_reason": None,
                }
            )
        flow.add_counters("transfer", admissible_candidates=len(output))
        flow.emit(
            "transfer", "candidate_selection", input_count=considered,
            output_count=len(output), rejection_counts=rejected, examples=examples,
            fields={"m3_available": m3_available, "m4_available": m4_available},
        )
        return output

    def record_trial(
        self,
        uid: MemoryUid,
        *,
        target_game_hash: int,
        metric_on: float,
        metric_off: float,
        formation_games: tuple[int, ...] = (),
        intervention: str = "matched_memory_ablation",
    ) -> TransferTrial:
        formation = tuple(sorted(set(int(value) for value in formation_games)))
        target = int(target_game_hash)
        held_out = not formation or target not in formation
        effect = float(metric_on) - float(metric_off)
        trial = TransferTrial(
            uid,
            target,
            float(metric_on),
            float(metric_off),
            effect,
            bool(held_out and effect > self.effect_threshold),
            formation,
            str(intervention),
        )
        self._trials.setdefault(uid, []).append(trial)
        return trial

    def trials(self, uid: MemoryUid) -> tuple[TransferTrial, ...]:
        return tuple(self._trials.get(uid, ()))

    def empirically_validated(self, uid: MemoryUid, *, min_targets: int = 1) -> bool:
        passed_targets = {trial.target_game_hash for trial in self._trials.get(uid, ()) if trial.passed}
        return len(passed_targets) >= int(min_targets)

    def state_dict(self) -> dict[str, object]:
        rows = []
        for trials in self._trials.values():
            for trial in trials:
                raw = asdict(trial)
                raw["uid"] = [trial.uid.hi, trial.uid.lo]
                rows.append(raw)
        return {"effect_threshold": self.effect_threshold, "trials": rows}

    def load_state(self, state: dict[str, object] | None) -> None:
        if not state:
            return
        for raw in state.get("trials", []):
            if not isinstance(raw, dict):
                continue
            uid_raw = raw.get("uid", [0, 0])
            uid = MemoryUid(int(uid_raw[0]), int(uid_raw[1]))
            trial = TransferTrial(
                uid,
                int(raw.get("target_game_hash", 0)),
                float(raw.get("metric_on", 0.0)),
                float(raw.get("metric_off", 0.0)),
                float(raw.get("effect", 0.0)),
                bool(raw.get("passed", False)),
                tuple(int(v) for v in raw.get("formation_games", ())),
                str(raw.get("intervention", "matched_memory_ablation")),
            )
            self._trials.setdefault(uid, []).append(trial)
