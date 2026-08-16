from __future__ import annotations

from collections import defaultdict

from v8.model import MemoryLevel, signed_u64


_INSTALLED = False


def _cached_legacy_exact_click_actions(view, context_signature: int) -> tuple[int, ...]:
    """Index old absolute-click M1 memories once per coherent graph version."""
    from v8 import action_targeting_v810 as targeting

    view._refresh_strategy_cache()
    version = tuple(getattr(view, "_strategy_version", ()))
    if getattr(view, "_v810_legacy_click_index_version", None) != version:
        grouped: dict[int, dict[int, list[float]]] = defaultdict(dict)
        for row in getattr(view, "_node_by_uid", {}).values():
            if int(row.level) != int(MemoryLevel.M1) or len(row.key_parts) < 2:
                continue
            action = signed_u64(int(row.key_parts[1]))
            if targeting._legacy_coordinate_payload(action) is None:
                continue
            context = int(row.key_parts[0])
            valence = float(getattr(row, "expected_primary_valence", 0.0)) * float(
                getattr(row, "primary_valence_confidence", 0.0)
            )
            support = max(0, int(row.support_count))
            bucket = grouped[context].setdefault(action, [0.0, 0.0])
            bucket[0] += float(support)
            bucket[1] = max(float(bucket[1]), valence)

        index: dict[int, tuple[int, ...]] = {}
        for context, actions in grouped.items():
            ranked = sorted(
                actions.items(),
                key=lambda item: (-float(item[1][1]), -float(item[1][0]), int(item[0])),
            )
            index[int(context)] = tuple(
                int(action)
                for action, _stats in ranked[: targeting._MAX_LEGACY_EXACT_CLICK_TARGETS]
            )
        view._v810_legacy_click_index = index
        view._v810_legacy_click_index_version = version

    return tuple(
        getattr(view, "_v810_legacy_click_index", {}).get(int(context_signature), ())
    )


def install_action_targeting_v810_fixups() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from v8 import action_targeting_v810 as targeting

    targeting._legacy_exact_click_actions = _cached_legacy_exact_click_actions
    _INSTALLED = True
