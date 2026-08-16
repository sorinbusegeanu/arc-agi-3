from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

from v8.arena import EdgeRecord, NodeRecord
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
    ) -> dict[MemoryUid, tuple[int, ...]]:
        direct: dict[MemoryUid, set[int]] = {}
        parents: dict[MemoryUid, set[MemoryUid]] = {}
        lineage = {
            int(RelationType.PROVENANCE),
            int(RelationType.EXPLAINS),
            int(RelationType.CONTEXT_REFINES),
            int(RelationType.SUPERSEDES),
            int(RelationType.LEADS_TO),
        }
        for edge in edges:
            relation = int(edge.relation_type)
            if relation == int(RelationType.GAME_PROVENANCE) and int(edge.target_uid.hi) == 0:
                direct.setdefault(edge.source_uid, set()).add(int(edge.target_uid.lo))
            elif relation in lineage:
                parents.setdefault(edge.source_uid, set()).add(edge.target_uid)

        result: dict[MemoryUid, tuple[int, ...]] = {}
        for uid in uids:
            found = set(direct.get(uid, ()))
            frontier = {uid}
            visited = {uid}
            for _depth in range(max(0, int(max_depth))):
                following: set[MemoryUid] = set()
                for current in frontier:
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
    ) -> tuple[TransferCandidate, ...]:
        eligible = {
            row.uid: row
            for row in rows
            if int(row.level) in {int(MemoryLevel.M3), int(MemoryLevel.M4)}
        }
        if not eligible:
            return ()

        bound_read_view = None
        if provenance is not None:
            owner = getattr(provenance, "__self__", None)
            edge_records = getattr(owner, "edge_records", None)
            if callable(edge_records):
                bound_read_view = owner
                if not edges:
                    edges = tuple(edge_records())

        # Compatibility fallback for pure unit use without a graph.  In the live
        # runtime, where edges are present, formal TRANSFER_CORRESPONDENCE is required.
        if not edges and provenance is None:
            result = []
            for row in eligible.values():
                games = int(row.game_evidence_count)
                recurrence = min(1.0, games / 4.0) * min(
                    1.0, max(1, row.support_count) / 8.0
                )
                if games >= 2 and recurrence > 0.0:
                    result.append(TransferCandidate(row.uid, games, recurrence))
            return tuple(result)

        graph_games = (
            self._provenance_from_edges(tuple(eligible), edges)
            if bound_read_view is not None
            else None
        )

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
        for edge in edges:
            if int(edge.relation_type) != int(RelationType.TRANSFER_CORRESPONDENCE):
                continue
            if edge.source_uid not in eligible or edge.target_uid not in eligible:
                continue
            score = float(edge.score)
            if score <= 0.0:
                continue
            left_games = games(edge.source_uid)
            right_games = games(edge.target_uid)
            if not left_games or not right_games:
                continue
            left_set, right_set = set(left_games), set(right_games)
            if left_set == right_set:
                continue

            for uid, own_games, other_uid, other_games in (
                (edge.source_uid, left_games, edge.target_uid, right_games),
                (edge.target_uid, right_games, edge.source_uid, left_games),
            ):
                if not (set(other_games) - set(own_games)):
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
                    best[uid] = candidate
        return tuple(best[uid] for uid in sorted(best))

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
