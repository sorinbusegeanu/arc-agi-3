from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class _TransferScopeEvent:
    interaction_id: str
    game_key: object
    context_key: object
    carrier_signature: str
    role_signature: str
    family_signature: str
    step: object

    @property
    def sort_key(self) -> tuple[int, str]:
        return (
            9_223_372_036_854_775_807
            if self.step is None
            else int(self.step),
            self.interaction_id,
        )


@dataclass
class _TransferScopeIndex:
    connection_total_changes: int
    by_interaction: dict[str, tuple[_TransferScopeEvent, ...]]
    by_carrier: dict[str, tuple[_TransferScopeEvent, ...]]
    by_role: dict[str, tuple[_TransferScopeEvent, ...]]
    by_family: dict[str, tuple[_TransferScopeEvent, ...]]
    families_by_carrier: dict[str, tuple[str, ...]]


_SCOPE_INDEX_BY_CONNECTION: dict[
    sqlite3.Connection,
    _TransferScopeIndex,
] = {}

_ORIGINAL_TRANSFER_SCOPE_CANDIDATES: Callable[..., list[dict[str, object]]] | None = None


def _freeze_event_lists(
    values: dict[str, list[_TransferScopeEvent]],
) -> dict[str, tuple[_TransferScopeEvent, ...]]:
    return {
        key: tuple(sorted(items, key=lambda item: item.sort_key))
        for key, items in values.items()
    }


def _build_transfer_scope_index(
    connection: sqlite3.Connection,
) -> _TransferScopeIndex:
    by_interaction_mutable: dict[
        str,
        list[_TransferScopeEvent],
    ] = defaultdict(list)
    by_carrier_mutable: dict[
        str,
        list[_TransferScopeEvent],
    ] = defaultdict(list)
    by_role_mutable: dict[
        str,
        list[_TransferScopeEvent],
    ] = defaultdict(list)
    by_family_mutable: dict[
        str,
        list[_TransferScopeEvent],
    ] = defaultdict(list)

    rows = connection.execute(
        """
        SELECT
            source_interaction_id,
            COALESCE(source_game_id, game) AS game_key,
            COALESCE(
                source_context_signature,
                context_key
            ) AS context_key,
            source_carrier_id,
            source_role_id,
            source_family_id,
            first_seen_global_step
        FROM future_option_events
        WHERE source_interaction_id IS NOT NULL
        """
    )

    for row in rows:
        interaction_id = str(row["source_interaction_id"])
        carrier_signature = str(row["source_carrier_id"] or "")
        role_signature = str(row["source_role_id"] or "")
        family_signature = str(row["source_family_id"] or "")

        event = _TransferScopeEvent(
            interaction_id=interaction_id,
            game_key=row["game_key"],
            context_key=row["context_key"],
            carrier_signature=carrier_signature,
            role_signature=role_signature,
            family_signature=family_signature,
            step=row["first_seen_global_step"],
        )
        by_interaction_mutable[interaction_id].append(event)
        if carrier_signature:
            by_carrier_mutable[carrier_signature].append(event)
        if role_signature:
            by_role_mutable[role_signature].append(event)
        if family_signature:
            by_family_mutable[family_signature].append(event)

    families_by_carrier_mutable: dict[str, set[str]] = defaultdict(set)
    for row in connection.execute(
        """
        SELECT carrier_signature, linked_key
        FROM carrier_links
        WHERE linked_type = 'family'
        ORDER BY carrier_signature ASC, linked_key ASC
        """
    ):
        carrier_signature = str(row["carrier_signature"] or "")
        family_signature = str(row["linked_key"] or "")
        if carrier_signature and family_signature:
            families_by_carrier_mutable[carrier_signature].add(
                family_signature
            )

    return _TransferScopeIndex(
        connection_total_changes=int(connection.total_changes),
        by_interaction=_freeze_event_lists(by_interaction_mutable),
        by_carrier=_freeze_event_lists(by_carrier_mutable),
        by_role=_freeze_event_lists(by_role_mutable),
        by_family=_freeze_event_lists(by_family_mutable),
        families_by_carrier={
            carrier: tuple(sorted(families))
            for carrier, families in families_by_carrier_mutable.items()
        },
    )


def _get_transfer_scope_index(
    connection: sqlite3.Connection,
) -> _TransferScopeIndex:
    cached = _SCOPE_INDEX_BY_CONNECTION.get(connection)
    current_total_changes = int(connection.total_changes)
    if (
        cached is None
        or cached.connection_total_changes != current_total_changes
    ):
        cached = _build_transfer_scope_index(connection)
        _SCOPE_INDEX_BY_CONNECTION[connection] = cached
    return cached


def _candidate_from_event(
    event: _TransferScopeEvent,
    *,
    origin: str,
) -> dict[str, object]:
    return {
        "interaction_id": event.interaction_id,
        "game_key": event.game_key,
        "context_key": event.context_key,
        "origin": origin,
        "step": event.step,
    }


def _append_unique(
    target: list[dict[str, object]],
    seen_interactions: set[str],
    events: tuple[_TransferScopeEvent, ...],
    *,
    origin: str,
) -> None:
    for event in events:
        if event.interaction_id in seen_interactions:
            continue
        seen_interactions.add(event.interaction_id)
        target.append(
            _candidate_from_event(
                event,
                origin=origin,
            )
        )


def _indexed_transfer_scope_candidates(
    connection: sqlite3.Connection,
    *,
    interaction_id: str | None,
    carrier_signature: str | None,
    role_signature: str | None,
) -> list[dict[str, object]]:
    try:
        index = _get_transfer_scope_index(connection)
    except sqlite3.Error:
        if _ORIGINAL_TRANSFER_SCOPE_CANDIDATES is None:
            return []
        return _ORIGINAL_TRANSFER_SCOPE_CANDIDATES(
            connection,
            interaction_id=interaction_id,
            carrier_signature=carrier_signature,
            role_signature=role_signature,
        )

    normalized_interaction = (
        None
        if interaction_id in (None, "")
        else str(interaction_id)
    )
    normalized_carrier = (
        None
        if carrier_signature in (None, "")
        else str(carrier_signature)
    )
    normalized_role = (
        None
        if role_signature in (None, "")
        else str(role_signature)
    )

    candidates: list[dict[str, object]] = []
    seen_interactions: set[str] = set()

    if normalized_interaction is not None:
        _append_unique(
            candidates,
            seen_interactions,
            index.by_interaction.get(
                normalized_interaction,
                (),
            ),
            origin="interaction",
        )

    if normalized_carrier is not None:
        _append_unique(
            candidates,
            seen_interactions,
            index.by_carrier.get(
                normalized_carrier,
                (),
            ),
            origin="carrier_interaction",
        )

    if normalized_role is not None:
        _append_unique(
            candidates,
            seen_interactions,
            index.by_role.get(
                normalized_role,
                (),
            ),
            origin="role_interaction",
        )

    # Preserve the original behavior: family fallback is considered only
    # when exact interaction/carrier/role matching produced no candidates.
    if candidates or normalized_carrier is None:
        return candidates

    for family_signature in index.families_by_carrier.get(
        normalized_carrier,
        (),
    ):
        _append_unique(
            candidates,
            seen_interactions,
            index.by_family.get(
                family_signature,
                (),
            ),
            origin="family_interaction",
        )

    return candidates


def clear_transfer_scope_indexes() -> None:
    """Discard cached indexes, mainly for tests and explicit long-lived reuse."""
    _SCOPE_INDEX_BY_CONNECTION.clear()


def apply_patch() -> None:
    """Replace repeated per-attempt SQLite queries with one indexed scan."""
    global _ORIGINAL_TRANSFER_SCOPE_CANDIDATES

    from v6 import higher_order_substrate

    if getattr(
        higher_order_substrate,
        "_FAST_TRANSFER_SCOPE_INDEX_APPLIED",
        False,
    ):
        return

    _ORIGINAL_TRANSFER_SCOPE_CANDIDATES = (
        higher_order_substrate._transfer_scope_candidates
    )
    higher_order_substrate._transfer_scope_candidates = (
        _indexed_transfer_scope_candidates
    )
    higher_order_substrate._FAST_TRANSFER_SCOPE_INDEX_APPLIED = True
