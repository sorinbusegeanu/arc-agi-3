from __future__ import annotations

from dataclasses import dataclass

from v8.arena import NodeRecord
from v8.model import MemoryLevel, MemoryUid


@dataclass(frozen=True, slots=True)
class TransferCandidate:
    uid: MemoryUid
    game_evidence_count: int
    structural_score: float


@dataclass(frozen=True, slots=True)
class TransferTrial:
    uid: MemoryUid
    target_game_hash: int
    metric_on: float
    metric_off: float
    effect: float
    passed: bool


class TransferValidator:
    """Separate prospective structural reuse from empirical held-out intervention."""

    def __init__(self, *, effect_threshold: float = 0.0) -> None:
        self.effect_threshold = float(effect_threshold)
        self._trials: dict[MemoryUid, list[TransferTrial]] = {}

    def candidates(self, rows: tuple[NodeRecord, ...]) -> tuple[TransferCandidate, ...]:
        result = []
        for row in rows:
            if int(row.level) not in {int(MemoryLevel.M3), int(MemoryLevel.M4)}:
                continue
            games = int(row.game_evidence_count)
            if games < 2:
                continue
            structural = min(1.0, games / 4.0) * min(1.0, max(1, row.support_count) / 8.0)
            result.append(TransferCandidate(row.uid, games, structural))
        return tuple(result)

    def record_trial(
        self,
        uid: MemoryUid,
        *,
        target_game_hash: int,
        metric_on: float,
        metric_off: float,
    ) -> TransferTrial:
        effect = float(metric_on) - float(metric_off)
        trial = TransferTrial(
            uid,
            int(target_game_hash),
            float(metric_on),
            float(metric_off),
            effect,
            effect > self.effect_threshold,
        )
        self._trials.setdefault(uid, []).append(trial)
        return trial

    def trials(self, uid: MemoryUid) -> tuple[TransferTrial, ...]:
        return tuple(self._trials.get(uid, ()))

    def empirically_validated(self, uid: MemoryUid, *, min_targets: int = 1) -> bool:
        passed_targets = {trial.target_game_hash for trial in self._trials.get(uid, ()) if trial.passed}
        return len(passed_targets) >= int(min_targets)
