from __future__ import annotations

"""Apply v8.56 target-local backoff to the final grounded transfer authority."""


_INSTALLED = False
_BASE_GROUNDED_TRANSFER = None


def _adjust_m7_rows(game_id: str, m7):
    from v8 import adaptive_memory_transfer_integrity_v856 as v856

    adjusted = {}
    for action, rows in m7.items():
        kept = []
        for score, strategy_uid, origin in rows:
            failures = max(
                0,
                int(
                    v856._TARGET_TRANSFER_FAILURES.get(
                        v856._transfer_failure_key(str(game_id), strategy_uid),
                        0,
                    )
                ),
            )
            if failures >= v856._TRANSFER_BACKOFF_LIMIT:
                continue
            kept.append((float(score) - 0.15 * failures, strategy_uid, origin))
        if kept:
            kept.sort(key=lambda item: (-float(item[0]), item[1]))
            adjusted[int(action)] = tuple(kept)
    return adjusted


def _grounded_transfer_v856(view, game_id: str):
    m7, m1n = _BASE_GROUNDED_TRANSFER(view, game_id)
    return _adjust_m7_rows(str(game_id), m7), m1n


def _grounded_m7_v856(view, game_id: str):
    return _grounded_transfer_v856(view, game_id)[0]


def install_adaptive_memory_transfer_grounding_v856() -> None:
    global _INSTALLED, _BASE_GROUNDED_TRANSFER
    if _INSTALLED:
        return

    from v8 import environment_neutrality_v837 as v837
    from v8 import sampling_transfer_v833 as transfer

    _BASE_GROUNDED_TRANSFER = v837._grounded_transfer_index
    v837._grounded_transfer_index = _grounded_transfer_v856
    v837._grounded_m7_index_v837 = _grounded_m7_v856

    # v8.33's helper is retained for compatibility/direct callers. Point it at the
    # same final grounded authority so target-local penalties are applied exactly once.
    transfer._lineage_transfer_index = _grounded_m7_v856

    _INSTALLED = True
