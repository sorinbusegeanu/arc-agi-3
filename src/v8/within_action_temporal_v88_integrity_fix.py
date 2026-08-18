from __future__ import annotations

"""Learning-integrity closure for v8.8 within-action temporal evidence.

The fixes here keep macro structural facts lossless, preserve temporal mechanism
identity through M2, and make temporal prediction use the shared canonical graph in
real actors.  The small local tracker remains only as a bootstrap/test fallback when
no live read view exists.
"""

from collections import Counter, defaultdict
from dataclasses import replace


_INSTALLED = False
_BASE_TEMPORAL_FACT_TOKENS = None
_BASE_MERGE_TEMPORAL_FACTS = None
_BASE_NORMALIZED_FAMILY_KEY = None
_BASE_NORMALIZED_M2 = None
_BASE_NORMALIZED_DERIVE = None
_BASE_ACTOR_LIVE_READ_VIEW = None
_ACTOR_READ_VIEW = None

_MIN_PREDICTION_SUPPORT = 2


def _temporal_family_fingerprint(value: int) -> int:
    from v8 import structural_events

    return int(value) & int(structural_events._M1N_HASH_MASK)


def temporal_family_token_v88(family_signature: int) -> int:
    """Recoverable bounded M1N identity for one learned temporal mechanism family."""
    from v8 import structural_events

    payload = _temporal_family_fingerprint(int(family_signature))
    return int(
        structural_events._M1N_MARKER
        | (payload << 8)
        | (int(structural_events.NormalizedPrimitive.AUTONOMOUS_CHANGE) & 0xFF)
    )


def _temporal_fact_tokens_v88(descriptor) -> tuple[int, ...]:
    if not bool(getattr(descriptor, "has_internal_evolution", False)):
        return ()
    family = int(getattr(descriptor, "family_signature", 0))
    if family == 0:
        return ()
    # Trace/carrier detail already lives on V88ExperienceEvent.  M1N receives one
    # stable family identity so repetition strengthens a mechanism instead of
    # fragmenting it by animation length or individual trace.
    return (temporal_family_token_v88(family),)


def _merge_temporal_facts_v88(existing, temporal) -> tuple[int, ...]:
    """Never evict macro structural evidence to make room for temporal evidence."""
    from v8.structural_events import MAX_NORMALIZED_FACTS_PER_EVENT

    kept = list(dict.fromkeys(int(value) for value in existing))[:MAX_NORMALIZED_FACTS_PER_EVENT]
    if len(kept) >= MAX_NORMALIZED_FACTS_PER_EVENT:
        return tuple(kept)
    for value in dict.fromkeys(int(value) for value in temporal):
        if value in kept:
            continue
        kept.append(value)
        if len(kept) >= MAX_NORMALIZED_FACTS_PER_EVENT:
            break
    return tuple(kept)


def _normalized_family_key_v88(value: int) -> tuple[int, int]:
    from v8.structural_events import NormalizedPrimitive, normalized_fact_kind

    kind = normalized_fact_kind(int(value))
    if kind == NormalizedPrimitive.AUTONOMOUS_CHANGE:
        # v8.8 temporal tokens place the family fingerprint directly in the 55-bit
        # normalized payload.  Old v8.8 tokens remain separated instead of all
        # collapsing into a single AUTONOMOUS_CHANGE family.
        return (int(kind), (int(value) >> 8) & ((1 << 55) - 1))
    return _BASE_NORMALIZED_FAMILY_KEY(int(value))


def _derive_normalized_proposals_v88(pipeline, grounded):
    """Publish temporal-family M1N even when all eight macro fact slots are occupied."""
    from v8 import normalized_memory_v086 as normalized
    from v8.model import MemoryLevel, MemoryType, MemoryUid

    base = tuple(_BASE_NORMALIZED_DERIVE(pipeline, grounded))
    experience = pipeline.experience
    if int(getattr(experience, "temporal_transition_count", 0)) <= 1:
        return base
    family = int(getattr(experience, "temporal_family_signature", 0))
    if family == 0:
        return base

    token = temporal_family_token_v88(family)
    uid = MemoryUid.from_key(MemoryLevel.M1, MemoryType.CONTINGENCY, (int(token),))
    if any(row.uid == uid for row in base):
        return base

    temporal_pipeline = replace(pipeline, normalized_facts=(int(token),))
    extra = tuple(_BASE_NORMALIZED_DERIVE(temporal_pipeline, grounded))
    seen = {row.uid for row in base}
    return base + tuple(row for row in extra if row.uid not in seen)


def _normalized_m2_candidates_v88(engine, nodes, *, limit: int):
    """Form distinct M2 temporal families from stable family fingerprints.

    Ordinary M1N families keep the v8.6 distinct-member rule.  A temporal family is
    already a compressed mechanism identity, so repeated support on one canonical
    M1N node is sufficient evidence for M2 once the same minimum-family threshold is
    met.
    """
    from v8 import normalized_memory_v086 as normalized
    from v8.model import CognitiveState, MemoryLevel, MemoryType, MemoryUid, ValidationState
    from v8.promotion import FormationCandidate
    from v8.structural_events import NormalizedPrimitive, normalized_fact_kind

    if int(limit) <= 0:
        return ()
    stable = [
        row
        for row in nodes
        if normalized.is_normalized_contingency(row)
        and int(row.support_count) >= int(engine.min_contingency_support)
        and engine._admissible(row)
    ]
    grouped = defaultdict(list)
    for row in stable:
        grouped[_normalized_family_key_v88(int(row.key_parts[0]))].append(row)

    result = []
    for family_key, members in sorted(grouped.items()):
        kind, variant = map(int, family_key)
        total_support = sum(max(0, int(row.support_count)) for row in members)
        temporal_family = kind == int(NormalizedPrimitive.AUTONOMOUS_CHANGE)
        if temporal_family:
            if total_support < max(
                int(engine.min_contingency_support), int(engine.min_family_members)
            ):
                continue
        elif len(members) < int(engine.min_family_members):
            continue

        compression = float(total_support - len(members))
        if compression <= float(engine.min_family_compression):
            continue
        key = (int(normalized._M2N_MARKER | kind), int(variant))
        uid = MemoryUid.from_key(MemoryLevel.M2, MemoryType.FAMILY, key)
        consistency = min(1.0, total_support / max(1.0, 2.0 * len(members)))
        result.append(
            FormationCandidate(
                uid,
                MemoryLevel.M2,
                MemoryType.FAMILY,
                key,
                tuple(sorted(row.uid for row in members)),
                total_support,
                consistency,
                min(1.0, compression / max(1.0, total_support)),
                0.0,
                float(len(members)),
                sum(row.future_option_delta * row.support_count for row in members)
                / max(1, total_support),
                int(CognitiveState.PROBATION),
                int(ValidationState.STRUCTURAL),
                "temporal_family_support" if temporal_family else "normalized_family_compression",
                min(1.0, compression / max(1.0, total_support)),
            )
        )
        if len(result) >= int(limit):
            break
    return tuple(result)


def _temporal_prediction_index(view):
    """Build a bounded-staleness predictor from shared M1 lineage/provenance edges."""
    from v8 import normalized_memory_v086 as normalized
    from v8.model import MemoryLevel, RelationType
    from v8.structural_events import NormalizedPrimitive, normalized_fact_kind

    refresh = getattr(view, "_refresh_strategy_cache", None)
    if callable(refresh):
        refresh()
    version = tuple(int(value) for value in getattr(view, "_strategy_version", ()))
    cached = getattr(view, "_v88_temporal_prediction_cache", None)
    if cached is not None and cached[0] == version:
        return cached[1]

    nodes = {
        row.uid: row for row in tuple(view.node_records(level=MemoryLevel.M1))
    }
    edges = tuple(view.edge_records())
    provenance: dict[object, set[int]] = defaultdict(set)
    for edge in edges:
        if (
            int(edge.relation_type) == int(RelationType.GAME_PROVENANCE)
            and int(edge.target_uid.hi) == 0
        ):
            provenance[edge.source_uid].add(int(edge.target_uid.lo))

    temporal: dict[object, int] = {}
    for uid, row in nodes.items():
        if not normalized.is_normalized_contingency(row):
            continue
        try:
            kind = normalized_fact_kind(int(row.key_parts[0]))
        except ValueError:
            continue
        if kind != NormalizedPrimitive.AUTONOMOUS_CHANGE:
            continue
        temporal[uid] = (int(row.key_parts[0]) >> 8) & ((1 << 55) - 1)

    index: dict[tuple[int, int, int], Counter[int]] = defaultdict(Counter)
    for edge in edges:
        if int(edge.relation_type) != int(RelationType.EXPLAINS):
            continue
        family = temporal.get(edge.source_uid)
        if family is None:
            continue
        parent = nodes.get(edge.target_uid)
        if parent is None or not normalized.is_grounded_contingency(parent):
            continue
        context, action = int(parent.key_parts[0]), int(parent.key_parts[1])
        games = set(provenance.get(edge.source_uid, ()))
        games.update(provenance.get(parent.uid, ()))
        if not games:
            continue
        weight = max(1, int(getattr(edge, "support_count", 0)))
        for game in games:
            index[(int(game), context, action)][int(family)] += weight

    frozen = dict(index)
    try:
        view._v88_temporal_prediction_cache = (version, frozen)
    except BaseException:
        pass
    return frozen


def temporal_prediction_error_from_view_v88(
    view,
    source_scope: int,
    context_signature: int,
    action_id: int,
    temporal_family_signature: int,
    *,
    minimum_support: int = _MIN_PREDICTION_SUPPORT,
) -> float:
    counts = _temporal_prediction_index(view).get(
        (int(source_scope), int(context_signature), int(action_id)), Counter()
    )
    if sum(int(value) for value in counts.values()) < max(1, int(minimum_support)):
        return 0.0
    expected, _count = min(counts.items(), key=lambda item: (-item[1], item[0]))
    actual = _temporal_family_fingerprint(int(temporal_family_signature))
    return 0.0 if int(expected) == int(actual) else 1.0


def _install_shared_prediction() -> None:
    global _BASE_ACTOR_LIVE_READ_VIEW, _ACTOR_READ_VIEW

    from v8 import actor as actor_module
    from v8 import within_action_temporal_v88 as temporal

    _BASE_ACTOR_LIVE_READ_VIEW = actor_module.LiveReadView

    class V88ActorLiveReadView(_BASE_ACTOR_LIVE_READ_VIEW):
        def __init__(self, *args, **kwargs):
            global _ACTOR_READ_VIEW
            super().__init__(*args, **kwargs)
            _ACTOR_READ_VIEW = self

        def close(self):
            global _ACTOR_READ_VIEW
            if _ACTOR_READ_VIEW is self:
                _ACTOR_READ_VIEW = None
            return super().close()

    V88ActorLiveReadView.__name__ = "V88ActorLiveReadView"
    actor_module.LiveReadView = V88ActorLiveReadView

    class SharedTemporalPredictionTracker(temporal.TemporalPredictionTracker):
        def prediction_error(
            self,
            source_scope: int,
            context_signature: int,
            action_id: int,
            temporal_family_signature: int,
        ) -> float:
            if _ACTOR_READ_VIEW is not None:
                return temporal_prediction_error_from_view_v88(
                    _ACTOR_READ_VIEW,
                    source_scope,
                    context_signature,
                    action_id,
                    temporal_family_signature,
                    minimum_support=self.minimum_support,
                )
            return super().prediction_error(
                source_scope,
                context_signature,
                action_id,
                temporal_family_signature,
            )

        def observe(
            self,
            source_scope: int,
            context_signature: int,
            action_id: int,
            temporal_family_signature: int,
        ) -> None:
            # In a real actor the canonical graph is the authority and the current
            # event will reach it through the normal M0/M1 pipeline.  Do not maintain
            # a divergent actor-private evidence count.
            if _ACTOR_READ_VIEW is None:
                super().observe(
                    source_scope,
                    context_signature,
                    action_id,
                    temporal_family_signature,
                )

    temporal._TEMPORAL_PREDICTIONS = SharedTemporalPredictionTracker(
        minimum_support=_MIN_PREDICTION_SUPPORT
    )


def install_within_action_temporal_v88_integrity_fix() -> None:
    global _INSTALLED
    global _BASE_TEMPORAL_FACT_TOKENS, _BASE_MERGE_TEMPORAL_FACTS
    global _BASE_NORMALIZED_FAMILY_KEY, _BASE_NORMALIZED_M2, _BASE_NORMALIZED_DERIVE
    if _INSTALLED:
        return

    from v8 import normalized_memory_v086 as normalized
    from v8 import within_action_temporal_v88 as temporal

    _BASE_TEMPORAL_FACT_TOKENS = temporal.temporal_fact_tokens
    _BASE_MERGE_TEMPORAL_FACTS = temporal._merge_temporal_facts
    _BASE_NORMALIZED_FAMILY_KEY = normalized.normalized_family_key
    _BASE_NORMALIZED_M2 = normalized._normalized_m2_candidates
    _BASE_NORMALIZED_DERIVE = normalized.derive_normalized_proposals

    temporal.temporal_fact_tokens = _temporal_fact_tokens_v88
    temporal._merge_temporal_facts = _merge_temporal_facts_v88
    normalized.normalized_family_key = _normalized_family_key_v88
    normalized._normalized_m2_candidates = _normalized_m2_candidates_v88
    normalized.derive_normalized_proposals = _derive_normalized_proposals_v88

    _install_shared_prediction()
    _INSTALLED = True
