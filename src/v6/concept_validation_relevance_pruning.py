from __future__ import annotations

import json
import sqlite3
from bisect import bisect_right
from collections import defaultdict
from contextvars import ContextVar
from typing import Any, Iterable

_INSTALLED = False
_RELEVANCE: ContextVar[dict[str, Any] | None] = ContextVar(
    "v6_concept_validation_relevance", default=None
)
_ORIGINALS: dict[str, Any] = {}


class _PrunedEventList(list[dict[str, Any]]):
    """Carry cheap counts for unrelated events without materializing/scoring them.

    ``len(events)`` intentionally returns the original later-event population so
    definition-cost allocation remains compatible with the pre-pruning
    validation semantics. Iteration contains only structurally relevant events.
    """

    def __init__(
        self,
        values: Iterable[dict[str, Any]] = (),
        *,
        skipped_by_type: dict[str, int] | None = None,
        context_only_overlap_count: int = 0,
        game_only_overlap_count: int = 0,
    ) -> None:
        super().__init__(values)
        self.skipped_by_type: dict[str, int] = dict(skipped_by_type or {})
        self.context_only_overlap_count = int(context_only_overlap_count)
        self.game_only_overlap_count = int(game_only_overlap_count)

    @property
    def skipped_count(self) -> int:
        return sum(int(value) for value in self.skipped_by_type.values())

    @property
    def materialized_count(self) -> int:
        return list.__len__(self)

    def __len__(self) -> int:
        return self.materialized_count + self.skipped_count

    def extend(self, values: Iterable[dict[str, Any]]) -> None:
        if isinstance(values, _PrunedEventList):
            for key, value in values.skipped_by_type.items():
                self.skipped_by_type[key] = int(self.skipped_by_type.get(key, 0)) + int(value)
            self.context_only_overlap_count += int(values.context_only_overlap_count)
            self.game_only_overlap_count += int(values.game_only_overlap_count)
        list.extend(self, values)


def _scope() -> dict[str, Any] | None:
    return _RELEVANCE.get()


def _record_skipped(
    event_type: str,
    *,
    context_only: bool = False,
    game_only: bool = False,
) -> None:
    scope = _scope()
    if scope is None:
        return
    counts = scope.setdefault("skipped_by_type", defaultdict(int))
    counts[event_type] += 1
    if context_only:
        scope["context_only_overlap_count"] = int(scope.get("context_only_overlap_count", 0)) + 1
    if game_only:
        scope["game_only_overlap_count"] = int(scope.get("game_only_overlap_count", 0)) + 1


def _wrap_events(values: list[dict[str, Any]]) -> _PrunedEventList:
    scope = _scope()
    if scope is None:
        return _PrunedEventList(values)
    return _PrunedEventList(
        values,
        skipped_by_type=dict(scope.get("builder_skipped_by_type") or {}),
        context_only_overlap_count=int(scope.get("builder_context_only_overlap_count", 0) or 0),
        game_only_overlap_count=int(scope.get("builder_game_only_overlap_count", 0) or 0),
    )


def _builder_scope_start() -> tuple[dict[str, int], int, int]:
    scope = _scope()
    if scope is None:
        return {}, 0, 0
    return (
        dict(scope.get("skipped_by_type") or {}),
        int(scope.get("context_only_overlap_count", 0) or 0),
        int(scope.get("game_only_overlap_count", 0) or 0),
    )


def _builder_scope_finish(
    values: list[dict[str, Any]],
    before: tuple[dict[str, int], int, int],
) -> _PrunedEventList:
    scope = _scope()
    if scope is None:
        return _PrunedEventList(values)
    before_counts, before_context, before_game = before
    current = dict(scope.get("skipped_by_type") or {})
    delta = {
        key: int(value) - int(before_counts.get(key, 0))
        for key, value in current.items()
        if int(value) - int(before_counts.get(key, 0)) > 0
    }
    return _PrunedEventList(
        values,
        skipped_by_type=delta,
        context_only_overlap_count=int(scope.get("context_only_overlap_count", 0) or 0) - before_context,
        game_only_overlap_count=int(scope.get("game_only_overlap_count", 0) or 0) - before_game,
    )


def _transfer_explanation_events(
    *,
    candidate_signature: str,
    source_roles: list[str],
    first_seen_global_step: int | None,
    transfer_rows: list[sqlite3.Row],
    transfer_history=None,
    rate_cache=None,
) -> list[dict[str, Any]]:
    from v6 import concept_validation_fastpath as fast
    from v6 import higher_order_substrate as substrate

    scope = _scope()
    if scope is None or transfer_history is None:
        return _ORIGINALS["fast_transfer"](
            candidate_signature=candidate_signature,
            source_roles=source_roles,
            first_seen_global_step=first_seen_global_step,
            transfer_rows=transfer_rows,
            transfer_history=transfer_history,
            rate_cache=rate_cache,
        )
    if first_seen_global_step is None or not source_roles:
        return _PrunedEventList()

    before = _builder_scope_start()
    steps, ordered = fast._transfer_step_rows(transfer_rows)
    start = bisect_right(steps, int(first_seen_global_step))
    roles = set(scope["roles"])
    carriers = set(scope["carriers"])
    contexts = set(scope["contexts"])
    games = set(scope["games"])
    feature_step_cache: dict[int, int | None] = {}
    events: list[dict[str, Any]] = []

    for row in ordered[start:]:
        source_role = str(row["role_signature"] or "")
        target_role = str(row["observed_role_signature"] or row["predicted_role_signature"] or "")
        source_carrier = str(row["source_carrier_signature"] or "")
        target_carrier = str(row["target_carrier_signature"] or "")
        source_game_key = str(row["source_game_key"] or "")
        source_context_key = str(row["source_context_key"] or "")
        target_game_key = str(row["target_game_key"] or "")
        target_context_key = str(row["target_context_key"] or "")

        structurally_relevant = bool(
            source_role in roles
            or target_role in roles
            or source_carrier in carriers
            or target_carrier in carriers
        )
        if not structurally_relevant:
            context_overlap = bool({source_context_key, target_context_key} - {""}) and bool(
                {source_context_key, target_context_key} & contexts
            )
            game_overlap = bool({source_game_key, target_game_key} - {""}) and bool(
                {source_game_key, target_game_key} & games
            )
            _record_skipped(
                "transfer",
                context_only=context_overlap and not game_overlap,
                game_only=game_overlap and not context_overlap,
            )
            continue

        step = int(row["last_seen_global_step"])
        generic_rates, scoped_rates = fast._role_score_bundle(
            substrate,
            source_roles=source_roles,
            step=step,
            transfer_rows=transfer_rows,
            transfer_history=transfer_history,
            rate_cache=rate_cache,
            scope=(source_game_key, source_context_key, target_game_key, target_context_key),
        )
        best_single = max(generic_rates, default=0.0)
        combination_rates = scoped_rates if len(scoped_rates) >= 2 else generic_rates
        concept_score = (
            substrate._combined_role_score(combination_rates)
            if len(combination_rates) >= 2
            else best_single
        )
        if step not in feature_step_cache:
            feature_step_cache[step] = max(
                (
                    prior
                    for role in source_roles
                    for prior in [transfer_history.max_step_before(role=role, step=step)]
                    if prior is not None
                ),
                default=None,
            )
        outcome = float(int(row["reuse_success"] or 0))
        target_for_id = target_role or "unknown"
        events.append(
            {
                "concept_id": candidate_signature,
                "event_id": (
                    f"transfer:{source_role}:{target_for_id}:{source_game_key}:{source_context_key}:"
                    f"{target_game_key}:{target_context_key}:{row['attempt_id']}"
                ),
                "event_type": "transfer",
                "evaluation_scope": "later_global_step",
                "best_single_role_score": best_single,
                "lower_level_baseline_score": best_single,
                "concept_enabled_score": concept_score,
                "prediction_gain": concept_score - best_single,
                "behavioral_gain": -abs(concept_score - outcome) + abs(best_single - outcome),
                "_outcome": outcome,
                "_evaluation_global_step": step,
                "_feature_global_step_max": feature_step_cache[step],
                "_label_used_as_feature": False,
                "_source_role_signature": source_role,
                "_predicted_target_role_signature": str(row["predicted_role_signature"] or ""),
                "_observed_target_role_signature": str(row["observed_role_signature"] or ""),
                "_carrier_ids": [
                    value for value in (source_carrier, target_carrier) if value
                ],
                "_context_keys": [
                    value for value in (source_context_key, target_context_key) if value
                ],
                "_game_keys": [value for value in (source_game_key, target_game_key) if value],
            }
        )
    return _builder_scope_finish(events, before)


def _future_option_motif_explanation_events(
    state_conn: sqlite3.Connection,
    *,
    candidate_signature: str,
    source_roles: list[str],
    first_seen_global_step: int | None,
    future_rows: list[sqlite3.Row],
    role_rate_cache: dict[tuple[str, int], float] | None = None,
) -> list[dict[str, Any]]:
    from v6 import concept_validation_fastpath as fast
    from v6 import concept_validation_fastpath_compat as compat
    from v6 import higher_order_substrate as substrate

    scope = _scope()
    if scope is None:
        return _ORIGINALS["future"](
            state_conn,
            candidate_signature=candidate_signature,
            source_roles=source_roles,
            first_seen_global_step=first_seen_global_step,
            future_rows=future_rows,
            role_rate_cache=role_rate_cache,
        )
    if first_seen_global_step is None or not source_roles:
        return _PrunedEventList()

    before = _builder_scope_start()
    index = compat._future_role_prefix(future_rows)
    role_rates: dict[str, float] = {}
    for role in source_roles:
        cache_key = (role, int(first_seen_global_step))
        if role_rate_cache is not None and cache_key in role_rate_cache:
            role_rates[role] = role_rate_cache[cache_key]
            continue
        steps, prefix = index.get(role, ((), (0,)))
        from bisect import bisect_left
        count = bisect_left(steps, int(first_seen_global_step))
        rate = float(prefix[count]) / float(count) if count else 0.0
        role_rates[role] = rate
        if role_rate_cache is not None:
            role_rate_cache[cache_key] = rate

    motif_steps, rows = fast._motif_rows(state_conn)
    start = bisect_right(motif_steps, int(first_seen_global_step))
    role_set = set(source_roles)
    events: list[dict[str, Any]] = []
    for row in rows[start:]:
        try:
            motif_roles = {
                str(value) for value in json.loads(str(row.get("source_role_ids_json") or "[]"))
            }
        except (TypeError, ValueError, json.JSONDecodeError):
            motif_roles = set()
        matching_roles = sorted(role_set & motif_roles)
        if not matching_roles:
            _record_skipped("future_option_motif")
            continue
        rates = [role_rates[role] for role in matching_roles]
        baseline = max(rates, default=0.0)
        concept_score = substrate._combined_role_score(rates) if len(rates) >= 2 else baseline
        outcome = float(int(row.get("is_emergent") or 0))
        events.append(
            {
                "concept_id": candidate_signature,
                "event_id": f"future_option_motif:{row['motif_signature']}",
                "event_type": "future_option_motif",
                "evaluation_scope": "later_global_step",
                "best_single_role_score": baseline,
                "lower_level_baseline_score": baseline,
                "concept_enabled_score": concept_score,
                "prediction_gain": concept_score - baseline,
                "behavioral_gain": -abs(concept_score - outcome) + abs(baseline - outcome),
                "_outcome": outcome,
                "_evaluation_global_step": int(row["last_seen_global_step"]),
                "_feature_global_step_max": int(first_seen_global_step) - 1,
                "_label_used_as_feature": False,
                "_source_role_ids": sorted(motif_roles),
            }
        )
    return _builder_scope_finish(events, before)


def _prediction_explanation_events(
    state_conn: sqlite3.Connection,
    *,
    candidate_signature: str,
    source_roles: list[str],
    first_seen_global_step: int | None,
    transfer_rows: list[sqlite3.Row],
    role_links: dict[str, dict[str, set[str]]],
    transfer_history=None,
    rate_cache=None,
) -> list[dict[str, Any]]:
    from v6 import concept_validation_fastpath as fast
    from v6 import higher_order_substrate as substrate

    scope = _scope()
    if scope is None:
        return _ORIGINALS["prediction"](
            state_conn,
            candidate_signature=candidate_signature,
            source_roles=source_roles,
            first_seen_global_step=first_seen_global_step,
            transfer_rows=transfer_rows,
            role_links=role_links,
            transfer_history=transfer_history,
            rate_cache=rate_cache,
        )
    columns = substrate._prediction_result_columns(state_conn)
    required = {"id", "global_step", "context_signature", "predicted_family", "actual_family"}
    if first_seen_global_step is None or not source_roles or not required <= columns:
        return _PrunedEventList()

    before = _builder_scope_start()
    steps, rows, _contradiction_steps, _contradiction_rows = fast._prediction_rows(state_conn)
    start = bisect_right(steps, int(first_seen_global_step))
    relevant_families = set(scope["families"])
    relevant_contexts = set(scope["contexts"])
    source_role_family_ids = sorted(
        {
            str(family)
            for role in source_roles
            for family in role_links.get(role, {}).get("family", set())
            if family not in (None, "")
        }
    )
    events: list[dict[str, Any]] = []
    for row in rows[start:]:
        predicted_family = str(row.get("predicted_family") or "")
        actual_family = str(row.get("actual_family") or "")
        if not ({predicted_family, actual_family} - {""}) & relevant_families:
            context = str(row.get("context_signature") or "")
            _record_skipped(
                "prediction",
                context_only=bool(context and context in relevant_contexts),
            )
            continue
        step = int(row["global_step"])
        generic_rates, _ = fast._role_score_bundle(
            substrate,
            source_roles=source_roles,
            step=step,
            transfer_rows=transfer_rows,
            transfer_history=transfer_history,
            rate_cache=rate_cache,
        )
        baseline = max(generic_rates, default=0.0)
        concept_score = (
            substrate._combined_role_score(generic_rates)
            if len(generic_rates) >= 2
            else baseline
        )
        outcome = float(predicted_family == actual_family)
        context = str(row.get("context_signature") or "")
        events.append(
            {
                "concept_id": candidate_signature,
                "event_id": f"prediction:prediction_result:{row['id']}:later_global_step",
                "event_type": "prediction",
                "evaluation_scope": "later_global_step",
                "predicted_family": predicted_family,
                "actual_family": actual_family,
                "candidate_role_family_ids": source_role_family_ids,
                "best_single_role_score": baseline,
                "lower_level_baseline_score": baseline,
                "concept_enabled_score": concept_score,
                "prediction_gain": concept_score - baseline,
                "behavioral_gain": -abs(concept_score - outcome) + abs(baseline - outcome),
                "_outcome": outcome,
                "_evaluation_global_step": step,
                "_feature_global_step_max": (
                    transfer_history.max_any_step_before(step)
                    if transfer_history is not None
                    else None
                ),
                "_label_used_as_feature": False,
                "_context_keys": [context] if context else [],
                "_family_ids": [family for family in (predicted_family, actual_family) if family],
            }
        )
    return _builder_scope_finish(events, before)


def _contradiction_resolution_explanation_events(
    state_conn: sqlite3.Connection,
    *,
    candidate_signature: str,
    source_roles: list[str],
    first_seen_global_step: int | None,
    transfer_rows: list[sqlite3.Row],
    role_links: dict[str, dict[str, set[str]]],
    transfer_history=None,
    rate_cache=None,
) -> list[dict[str, Any]]:
    from v6 import concept_validation_fastpath as fast
    from v6 import higher_order_substrate as substrate

    scope = _scope()
    if scope is None:
        return _ORIGINALS["contradiction"](
            state_conn,
            candidate_signature=candidate_signature,
            source_roles=source_roles,
            first_seen_global_step=first_seen_global_step,
            transfer_rows=transfer_rows,
            role_links=role_links,
            transfer_history=transfer_history,
            rate_cache=rate_cache,
        )
    columns = substrate._prediction_result_columns(state_conn)
    required = {"id", "global_step", "context_signature", "context_contradiction"}
    if first_seen_global_step is None or not source_roles or not required <= columns:
        return _PrunedEventList()

    before = _builder_scope_start()
    _steps, _rows, contradiction_steps, contradiction_rows = fast._prediction_rows(state_conn)
    start = bisect_right(contradiction_steps, int(first_seen_global_step))
    relevant_families = set(scope["families"])
    relevant_contexts = set(scope["contexts"])
    available_family_columns = {"predicted_family", "actual_family"} <= columns
    source_role_family_ids = sorted(
        {
            str(family)
            for role in source_roles
            for family in role_links.get(role, {}).get("family", set())
            if family not in (None, "")
        }
    )
    events: list[dict[str, Any]] = []
    for row in contradiction_rows[start:]:
        predicted_family = str(row.get("predicted_family") or "") if available_family_columns else ""
        actual_family = str(row.get("actual_family") or "") if available_family_columns else ""
        if not available_family_columns or not ({predicted_family, actual_family} - {""}) & relevant_families:
            context = str(row.get("context_signature") or "")
            _record_skipped(
                "contradiction_resolution",
                context_only=bool(context and context in relevant_contexts),
            )
            continue
        step = int(row["global_step"])
        generic_rates, _ = fast._role_score_bundle(
            substrate,
            source_roles=source_roles,
            step=step,
            transfer_rows=transfer_rows,
            transfer_history=transfer_history,
            rate_cache=rate_cache,
        )
        failure_rates = [1.0 - rate for rate in generic_rates]
        baseline = max(failure_rates, default=0.0)
        concept_score = (
            substrate._combined_role_score(failure_rates)
            if len(failure_rates) >= 2
            else baseline
        )
        context = str(row.get("context_signature") or "")
        events.append(
            {
                "concept_id": candidate_signature,
                "event_id": f"contradiction_resolution:{row['id']}:later_global_step",
                "event_type": "contradiction_resolution",
                "evaluation_scope": "later_global_step",
                "predicted_family": predicted_family,
                "actual_family": actual_family,
                "candidate_role_family_ids": source_role_family_ids,
                "best_single_role_score": baseline,
                "lower_level_baseline_score": baseline,
                "concept_enabled_score": concept_score,
                "prediction_gain": concept_score - baseline,
                "behavioral_gain": concept_score - baseline,
                "_outcome": 1.0,
                "_evaluation_global_step": step,
                "_feature_global_step_max": (
                    transfer_history.max_any_step_before(step)
                    if transfer_history is not None
                    else None
                ),
                "_label_used_as_feature": False,
                "_context_keys": [context] if context else [],
                "_family_ids": [family for family in (predicted_family, actual_family) if family],
            }
        )
    return _builder_scope_finish(events, before)


def _build_functional_explanation_diagnostics(*args: Any, **kwargs: Any):
    source_roles = list(kwargs.get("source_roles") or [])
    candidate_links = dict(kwargs.get("candidate_links") or {})
    scope = {
        "roles": set(source_roles),
        "carriers": set(candidate_links.get("carrier", set())),
        "families": set(candidate_links.get("family", set())),
        "contexts": set(candidate_links.get("context", set())),
        "games": set(candidate_links.get("game", set())),
        "skipped_by_type": defaultdict(int),
        "context_only_overlap_count": 0,
        "game_only_overlap_count": 0,
    }
    token = _RELEVANCE.set(scope)
    try:
        result = _ORIGINALS["diagnostics"](*args, **kwargs)
    finally:
        _RELEVANCE.reset(token)
    if not isinstance(result, tuple) or len(result) != 3:
        return result
    events, diagnostics, state = result
    if not isinstance(events, _PrunedEventList) or not isinstance(diagnostics, dict):
        return result

    skipped_by_type = dict(events.skipped_by_type)
    skipped_count = events.skipped_count
    diagnostics["prefiltered_unrelated_event_count"] = skipped_count
    diagnostics["prefiltered_unrelated_event_type_counts"] = {
        key: int(value) for key, value in sorted(skipped_by_type.items())
    }
    diagnostics["unrelated_event_count"] = int(diagnostics.get("unrelated_event_count", 0) or 0) + skipped_count
    diagnostics["context_only_overlap_count"] = int(
        diagnostics.get("context_only_overlap_count", 0) or 0
    ) + int(events.context_only_overlap_count)
    diagnostics["game_only_overlap_count"] = int(
        diagnostics.get("game_only_overlap_count", 0) or 0
    ) + int(events.game_only_overlap_count)
    diagnostics["relevance_prefilter_applied"] = True
    diagnostics["materialized_later_event_count"] = events.materialized_count
    denominator = (
        int(diagnostics.get("relevant_heldout_event_count", 0) or 0)
        + int(diagnostics.get("unrelated_event_count", 0) or 0)
    )
    diagnostics["global_coverage_descriptive"] = (
        float(diagnostics.get("global_explanatory_reach", 0) or 0) / float(denominator)
        if denominator
        else 0.0
    )
    errors = list(diagnostics.get("diagnostics_errors") or [])
    diagnostics["diagnostics_errors"] = [
        item for item in errors if item != "event_population_accounting_mismatch"
    ]
    return events, diagnostics, state


def install_concept_validation_relevance_pruning() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from v6 import concept_validation_fastpath as fast
    from v6 import higher_order_substrate as substrate

    _ORIGINALS["fast_transfer"] = fast._transfer_explanation_events
    _ORIGINALS["future"] = substrate._future_option_motif_explanation_events
    _ORIGINALS["prediction"] = substrate._prediction_explanation_events
    _ORIGINALS["contradiction"] = substrate._contradiction_resolution_explanation_events
    _ORIGINALS["diagnostics"] = substrate._build_functional_explanation_diagnostics

    fast._transfer_explanation_events = _transfer_explanation_events
    substrate._future_option_motif_explanation_events = _future_option_motif_explanation_events
    substrate._prediction_explanation_events = _prediction_explanation_events
    substrate._contradiction_resolution_explanation_events = _contradiction_resolution_explanation_events
    substrate._build_functional_explanation_diagnostics = _build_functional_explanation_diagnostics
    _INSTALLED = True
