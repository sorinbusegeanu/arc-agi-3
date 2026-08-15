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
    """Separate prospective structural reuse from empirical held-out intervention."""

    def __init__(self, *, effect_threshold: float = 0.0) -> None:
        self.effect_threshold = float(effect_threshold)
        self._trials: dict[MemoryUid, list[TransferTrial]] = {}

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

        def games(uid: MemoryUid) -> tuple[int, ...]:
            row = eligible.get(uid)
            if row is None:
                return ()
            if provenance is not None:
                return tuple(sorted(provenance(uid)))
            mask = int(row.game_mask)
            return tuple(index for index in range(64) if mask & (1 << index))

        best: dict[MemoryUid, TransferCandidate] = {}
        for edge in edges:
            if int(edge.relation_type) != int(RelationType.SIMILAR_TO):
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
