from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

from v8.arena import NodeRecord
from v8.model import MemoryLevel, MemoryUid


@dataclass(frozen=True, slots=True)
class TransferCandidate:
    uid: MemoryUid
    game_evidence_count: int
    structural_score: float
    formation_games: tuple[int, ...] = ()


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
        *,
        provenance: Callable[[MemoryUid], frozenset[int]] | None = None,
    ) -> tuple[TransferCandidate, ...]:
        result = []
        for row in rows:
            if int(row.level) not in {int(MemoryLevel.M3), int(MemoryLevel.M4)}:
                continue
            formation = tuple(sorted(provenance(row.uid))) if provenance is not None else ()
            games = len(formation) if formation else int(row.game_evidence_count)
            recurrence_prior = min(1.0, games / 4.0) * min(
                1.0, max(1, row.support_count) / 8.0
            )
            structural = max(float(row.transfer_prior), recurrence_prior)
            # A bounded graph-similarity correspondence can nominate a memory
            # formed in only one game for a held-out probe.  It remains only a
            # prospective prior; validation still requires record_trial().
            if games < 2 and structural <= 0.0:
                continue
            result.append(TransferCandidate(row.uid, games, structural, formation))
        return tuple(result)

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
