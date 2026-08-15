from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from v8.model import MemoryLevel, MemoryUid
from v8.provenance import exact_game_provenance


@dataclass(frozen=True, slots=True)
class TransferInterventionRequest:
    uid: MemoryUid
    target_game_hash: int


@dataclass(frozen=True, slots=True)
class ReplanningInterventionRequest:
    outcome_uid: MemoryUid
    primary_strategy_uid: MemoryUid
    alternative_strategy_uid: MemoryUid


class ExperimentalController:
    """Deterministically schedule bounded causal tests from current graph state.

    The controller schedules tests; environment execution remains delegated to the
    runtime/actor harness. Requests are never themselves treated as validation.
    """

    def transfer_requests(self, nodes, edges, *, max_requests: int = 8) -> tuple[TransferInterventionRequest, ...]:
        provenance = exact_game_provenance(nodes, edges)
        all_games = sorted({game for games in provenance.values() for game in games})
        requests: list[TransferInterventionRequest] = []
        for row in sorted(nodes, key=lambda r: r.uid):
            if int(row.level) not in {int(MemoryLevel.M3), int(MemoryLevel.M4)}:
                continue
            formation = provenance.get(row.uid, frozenset())
            if not formation:
                continue
            for game in all_games:
                if game not in formation:
                    requests.append(TransferInterventionRequest(row.uid, game))
                    break
            if len(requests) >= int(max_requests):
                break
        return tuple(requests)

    def replanning_requests(self, strategies_by_outcome, *, max_requests: int = 8) -> tuple[ReplanningInterventionRequest, ...]:
        requests: list[ReplanningInterventionRequest] = []
        for outcome_uid, strategies in sorted(strategies_by_outcome.items(), key=lambda item: item[0]):
            ordered = sorted(strategies, key=lambda row: (-row.reliability, row.uid))
            if len(ordered) < 2:
                continue
            requests.append(ReplanningInterventionRequest(outcome_uid, ordered[0].uid, ordered[1].uid))
            if len(requests) >= int(max_requests):
                break
        return tuple(requests)
