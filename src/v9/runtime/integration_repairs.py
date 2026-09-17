from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import replace
from typing import Any, Iterable

from v9.cognition.compression import form_families as _canonical_form_families

from .parallel_memory_coordinator import MemoryPipelineService


_CROSS_MODAL_PRIORITY = {
    "CROSS_MODAL_CORRESPONDENCE": 0,
    "SYMBOL_INTERACTION_ALIGNMENT": 1,
    "SYMBOL_COINCIDENT_WITH_OUTCOME": 2,
    "SYMBOL_NEAR_BOUNDARY": 3,
    "SYMBOL_COINCIDENT_WITH_PROGRESS": 4,
    "SYMBOL_PRECEDES_ACTION": 5,
    "SYMBOL_FOLLOWS_ACTION": 6,
    "SYMBOL_TO_INTERACTION_PREDICTION": 7,
    "INTERACTION_TO_SYMBOL_GENERALIZATION": 8,
    "SYMBOL_PRECEDES_NORMALIZED_CHANGE": 9,
    "SYMBOL_FOLLOWS_NORMALIZED_CHANGE": 10,
}


class _PreviousInteractionCursor:
    """Replay the exact ordered previous-interaction state inside one true batch."""

    def __init__(self, rows: dict[tuple[int, int], deque[Any | None]]) -> None:
        self._rows = rows

    def get(self, key: tuple[int, int], default: Any = None) -> Any:
        values = self._rows.get(key)
        if not values:
            return default
        return values.popleft()


def _ordered_previous_interactions(self: Any, prepared_rows: Iterable[Any]) -> _PreviousInteractionCursor:
    with self._lock:
        latest = dict(self._latest_interaction_grounding)
    rows: dict[tuple[int, int], deque[Any | None]] = defaultdict(deque)
    for prepared in prepared_rows:
        grounding = getattr(prepared, "m1g", None)
        if grounding is None:
            continue
        key = (int(grounding.environment_instance_id), int(grounding.episode_id))
        rows[key].append(latest.get(key))
        latest[key] = grounding
    return _PreviousInteractionCursor(rows)


def _block_for_result(self: MemoryPipelineService, *, timeout: float = 0.05) -> bool:
    """Bounded blocking wait used while draining ingestion/derivation at epoch boundaries."""
    progressed = self.pump_ingest_tasks()
    progressed = self.pump_derivation_tasks() or progressed

    ingest_outstanding = bool(
        self.ingested < self.sampled
        or self.pending_ingest
        or self.ingest_results
    )
    if ingest_outstanding:
        progressed = self.drain_ingest_results(block=True, timeout=float(timeout)) or progressed
        progressed = self.apply_ingest_ready() or progressed
        progressed = self.pump_ingest_tasks() or progressed
        progressed = self.pump_derivation_tasks() or progressed
        return progressed

    derivation_outstanding = bool(
        self.pending_derivation
        or self.inflight
        or self.derive_results
    )
    if derivation_outstanding:
        progressed = self.drain_derivation_results(block=True, timeout=float(timeout)) or progressed
        progressed = self.apply_derivation_ready() or progressed
        progressed = self.pump_derivation_tasks() or progressed
    return progressed


def _canonical_public_batch(self: Any, rows: Iterable[Any]) -> tuple[tuple[int, ...], ...]:
    """Use the same canonical commit implementation as the multiprocess pipeline."""
    from .canonical_commit import apply_canonical_commit_batch
    from .memory_pipeline import build_commit_plan

    prepared_rows = tuple(rows)
    if not prepared_rows:
        return ()

    telemetry = self.unified_telemetry
    curriculum_before = dict(telemetry.curriculum_counts)
    gauge_names = ("curriculum_step", "environment_family", "game_scenario")
    gauges_before = {
        name: (name in telemetry.gauges, telemetry.gauges.get(name))
        for name in gauge_names
    }

    plans = tuple(build_commit_plan(row) for row in prepared_rows)
    result = apply_canonical_commit_batch(self, plans)

    telemetry.curriculum_counts.clear()
    telemetry.curriculum_counts.update(curriculum_before)
    for name, (present, value) in gauges_before.items():
        if present:
            telemetry.gauges[name] = value
        else:
            telemetry.gauges.pop(name, None)
    return result.signature_rows


def _expanded_concept_evidence(self: Any, concept: Any) -> tuple[Any, ...]:
    pending = deque(tuple(concept.provenance.evidence) + tuple(concept.provenance.parents))
    seen: set[Any] = set()
    while pending and len(seen) < 4096:
        uid = pending.popleft()
        if uid in seen:
            continue
        seen.add(uid)
        payload = self._evidence_payload(uid)
        if payload is None:
            continue
        for raw in payload.get("parents", ()) or ():
            if isinstance(raw, (list, tuple)) and len(raw) == 2:
                from v9.memory.identity import MemoryUid
                parent = MemoryUid(int(raw[0]), int(raw[1]))
                if parent not in seen:
                    pending.append(parent)
    return tuple(sorted(seen))


def _accumulate_action_payload(payload: dict[str, Any], action_stats: dict[int, list[int]], contexts: set[int]) -> None:
    if payload.get("context_signature") is not None:
        contexts.add(int(payload["context_signature"]))
    if payload.get("action_id") is None:
        return
    action = int(payload["action_id"])
    stats = action_stats.setdefault(action, [0, 0, 0])
    valence = int(payload.get("primary_valence", 0))
    stats[0] += int(valence > 0)
    stats[1] += 1
    stats[2] += int(valence < 0)


def _transfer_validation_candidates(self: Any, *, limit: int = 32) -> tuple[dict[str, object], ...]:
    with self._lock:
        rows: list[dict[str, object]] = []
        concepts = sorted(
            self._m4.values(),
            key=lambda row: (
                bool(row.validated),
                -float(row.compression_benefit),
                -int(row.explanatory_reach),
                row.uid,
            ),
        )
        for concept in concepts:
            if concept.validated:
                continue
            evidence_uids = _expanded_concept_evidence(self, concept)
            scope = self._evidence_environment_scope(evidence_uids) or tuple(
                int(value) for value in concept.provenance.formation_scope
            )
            source_types: set[str] = set()
            for environment_id in scope:
                try:
                    source_types.add(str(self.environments.resolve(environment_id).environment_type))
                except KeyError:
                    pass

            action_stats: dict[int, list[int]] = {}
            contexts: set[int] = set()
            for uid in evidence_uids:
                payload = self._evidence_payload(uid)
                if payload is not None:
                    _accumulate_action_payload(payload, action_stats, contexts)

            if not action_stats and scope:
                scope_set = {int(value) for value in scope}
                for payload in self.graph.payloads.values():
                    environment_id = payload.get("environment_instance_id")
                    if environment_id is None or int(environment_id) not in scope_set:
                        continue
                    _accumulate_action_payload(payload, action_stats, contexts)
                for _node, payload, _evidence in self._deferred_base_nodes.values():
                    environment_id = payload.get("environment_instance_id")
                    if environment_id is None or int(environment_id) not in scope_set:
                        continue
                    _accumulate_action_payload(payload, action_stats, contexts)

            actions = tuple(
                action
                for action, _ in sorted(
                    action_stats.items(),
                    key=lambda item: (-item[1][0], -item[1][1], item[1][2], item[0]),
                )
            )
            if not actions:
                continue
            rows.append(
                {
                    "concept_uid": concept.uid,
                    "formation_scope": tuple(sorted(scope)),
                    "source_environment_types": tuple(sorted(source_types)),
                    "actions": actions,
                    "contexts": tuple(sorted(contexts)),
                    "positive_evidence": int(sum(stats[0] for stats in action_stats.values())),
                    "negative_evidence": int(sum(stats[2] for stats in action_stats.values())),
                    "support": int(sum(stats[1] for stats in action_stats.values())),
                    "validated": bool(concept.validated),
                }
            )
        rows.sort(
            key=lambda row: (
                -int(row["positive_evidence"]),
                -int(row["support"]),
                int(row["negative_evidence"]),
                row["concept_uid"],
            )
        )
        return tuple(rows[: max(1, int(limit))])


def _legacy_compatible_form_families(records: tuple[Any, ...], *, minimum_recurrence: int = 2):
    formed = _canonical_form_families(records, minimum_recurrence=minimum_recurrence)
    if formed or len(records) < int(minimum_recurrence):
        return formed
    structural = {int(row.structural_signature) for row in records}
    channels = {str(row.channel.value) for row in records if getattr(row, "channel", None) is not None}
    if len(structural) != 1 or channels != {"WORLD"}:
        return formed
    fallback_family = next(iter(structural))
    migrated = tuple(replace(row, family_signature=fallback_family) for row in records)
    return _canonical_form_families(migrated, minimum_recurrence=minimum_recurrence)


def _prioritized_symbolic_derivation(original: Any):
    def repaired(*args: Any, **kwargs: Any):
        limit = kwargs.get("max_cross_modal_facts")
        unbounded = dict(kwargs)
        unbounded["max_cross_modal_facts"] = None
        rows = tuple(original(*args, **unbounded))
        symbolic = [row for row in rows if str(row.relation.channel.value) != "CROSS_MODAL"]
        cross_modal = [row for row in rows if str(row.relation.channel.value) == "CROSS_MODAL"]
        indexed = list(enumerate(cross_modal))
        indexed.sort(key=lambda item: (_CROSS_MODAL_PRIORITY.get(str(item[1].relation_kind), 50), item[0]))
        selected = tuple(row for _index, row in indexed)
        if limit is not None:
            selected = selected[: max(0, int(limit))]
        return tuple(symbolic) + selected

    return repaired


def _prioritize_commit_plan(plan: Any) -> Any:
    derived = tuple(getattr(plan, "derived_relations", ()) or ())
    if not derived:
        return plan
    ordered = tuple(
        sorted(
            derived,
            key=lambda item: _CROSS_MODAL_PRIORITY.get(
                str(item.write.payload.get("symbol_relation", item.relation.observable_relation)),
                50,
            ),
        )
    )
    return replace(plan, derived_relations=ordered)


def _canonical_commit_without_implicit_grounding(original: Any):
    def repaired(runtime: Any, rows: Iterable[Any]):
        plans = tuple(_prioritize_commit_plan(row) for row in rows)
        states_before = dict(runtime.grounding.states)
        promotions_before = int(runtime.telemetry.get("grounding_promotions", 0))
        result = original(runtime, plans)
        runtime.grounding.states.clear()
        runtime.grounding.states.update(states_before)
        runtime.telemetry["grounding_promotions"] = promotions_before
        return result

    return repaired


def _flush_preserving_symbolic_payload(original: Any):
    def repaired(runtime: Any, *args: Any, **kwargs: Any):
        preserved: dict[Any, dict[str, Any]] = {
            uid: dict(payload)
            for uid, payload in runtime.graph.payloads.items()
            if payload.get("symbol_relation") is not None
        }
        for uid, (_node, payload, _evidence) in runtime._deferred_base_nodes.items():
            if payload.get("symbol_relation") is not None:
                preserved[uid] = {**preserved.get(uid, {}), **dict(payload)}
        result = original(runtime, *args, **kwargs)
        for uid, payload in preserved.items():
            current = runtime.graph.payloads.get(uid)
            if current is None:
                continue
            for key, value in payload.items():
                current.setdefault(key, value)
        return result

    return repaired


def install_integration_repairs(runtime_cls: type[Any]) -> None:
    """Install compatibility repairs required by the unified v9.7.8/v9.7.9 runtime."""
    runtime_cls._previous_interactions = _ordered_previous_interactions
    runtime_cls.apply_prepared_ingestion_batch = _canonical_public_batch
    runtime_cls.transfer_validation_candidates = _transfer_validation_candidates
    MemoryPipelineService.block_for_result = _block_for_result

    from . import runtime as runtime_module
    runtime_module.form_families = _legacy_compatible_form_families

    from .final_runtime import FinalContinuousMemoryRuntime
    if not getattr(FinalContinuousMemoryRuntime, "_integration_flush_repair", False):
        original_record_interaction = FinalContinuousMemoryRuntime.record_interaction

        def record_interaction(self: Any, *args: Any, **kwargs: Any):
            result = original_record_interaction(self, *args, **kwargs)
            self.flush_deferred_memory_updates()
            return result

        FinalContinuousMemoryRuntime.record_interaction = record_interaction
        FinalContinuousMemoryRuntime._integration_flush_repair = True

    from v9.memory import symbolic_relations as symbolic_module
    if not getattr(symbolic_module, "_integration_priority_repair", False):
        repaired = _prioritized_symbolic_derivation(symbolic_module.derive_symbolic_relations)
        symbolic_module.derive_symbolic_relations = repaired
        symbolic_module._integration_priority_repair = True

        from . import completed_runtime, final_runtime, memory_pipeline
        completed_runtime.derive_symbolic_relations = repaired
        final_runtime.derive_symbolic_relations = repaired
        memory_pipeline.derive_symbolic_relations = repaired

    from . import canonical_commit, parallel_memory_coordinator
    if not getattr(canonical_commit, "_integration_grounding_repair", False):
        repaired_commit = _canonical_commit_without_implicit_grounding(canonical_commit.apply_canonical_commit_batch)
        canonical_commit.apply_canonical_commit_batch = repaired_commit
        parallel_memory_coordinator.apply_canonical_commit_batch = repaired_commit
        canonical_commit._integration_grounding_repair = True

    if not getattr(runtime_cls, "_integration_symbolic_flush_repair", False):
        runtime_cls.flush_deferred_memory_updates = _flush_preserving_symbolic_payload(
            runtime_cls.flush_deferred_memory_updates
        )
        runtime_cls._integration_symbolic_flush_repair = True
