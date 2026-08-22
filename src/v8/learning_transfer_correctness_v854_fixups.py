from __future__ import annotations

"""Authority-preserving integration for v8.54 learning/transfer correctness."""

import inspect
from collections import defaultdict

_INSTALLED = False
_PRIOR_RESTART_BASE_RESET = None


def _clear_view_state(view) -> None:
    if view is None:
        return
    view._v854_session_transitions = []
    view._v815_session_trajectory = []
    active = getattr(view, "_v854_transfer_active", None)
    if isinstance(active, dict):
        active.clear()


def _restart_base_reset_v854(self, *args, **kwargs):
    try:
        from v8 import behavior_recovery as behavior

        _clear_view_state(getattr(behavior, "_CURRENT_ACTOR_VIEW", None))
    except BaseException:
        pass
    return _PRIOR_RESTART_BASE_RESET(self, *args, **kwargs)


def _contextual_cross_game_transfer(sampler, actions):
    """Use the actual v8.33 caller context without replacing its public hooks."""
    from v8 import behavior_recovery as behavior
    from v8 import learning_transfer_correctness_v854 as v854

    available = {int(value) for value in actions}
    view = getattr(behavior, "_CURRENT_ACTOR_VIEW", None)
    if view is not None and available:
        caller = inspect.currentframe().f_back
        context = None if caller is None else caller.f_locals.get("context")
        if context is not None:
            ordered = v854._ordered_action(
                view,
                str(getattr(sampler, "game_id", "")),
                int(context),
                available,
            )
            if ordered is not None:
                return ordered
    return v854._BASE_CROSS_GAME(sampler, tuple(sorted(available)))


def _locked_record_transfer_trial(self, *args, **kwargs):
    """Keep v8.45 snapshot serialization authority around v8.54 mutation."""
    from v8 import learning_transfer_correctness_v854 as v854

    lock = getattr(self, "_v845_state_lock", None)
    if lock is None:
        return v854._record_trial_v854(self, *args, **kwargs)
    with lock:
        return v854._record_trial_v854(self, *args, **kwargs)


def _similarity_v854_fixup(self, nodes, edges):
    """Reserve structurally ranked cross-world candidates before bucket truncation."""
    from v8 import learning_transfer_correctness_v854 as v854
    from v8.model import stable_u64

    nodes, edges = tuple(nodes), tuple(edges)
    descriptors = self.descriptors(nodes, edges)
    by_uid, _parents, games = v854._graph_index(nodes, edges)
    index, fallback = defaultdict(list), defaultdict(list)
    for descriptor in descriptors.values():
        index[(
            descriptor.level,
            descriptor.memory_type,
            self._relation_bucket(descriptor),
            descriptor.future_option_bucket,
        )].append(descriptor)
        fallback[(descriptor.level, descriptor.memory_type)].append(descriptor)
    dirty = [
        descriptor
        for descriptor in descriptors.values()
        if descriptor.descriptor_version > self._processed_versions.get(descriptor.uid, -1)
    ]
    dirty.sort(key=lambda descriptor: (
        descriptor.level,
        descriptor.memory_type,
        descriptor.descriptor_version,
        descriptor.uid,
    ))
    results = {}

    def is_cross(left_uid, right_uid) -> bool:
        left, right = games(left_uid), games(right_uid)
        if left and right:
            return left != right
        left_row, right_row = by_uid.get(left_uid), by_uid.get(right_uid)
        left_mask = 0 if left_row is None else int(left_row.game_mask)
        right_mask = 0 if right_row is None else int(right_row.game_mask)
        return bool(left_mask and right_mask and left_mask != right_mask)

    for source in dirty:
        relation_bucket = self._relation_bucket(source)
        bucket_candidates = []
        for delta in (0, -1, 1):
            bucket_candidates.extend(index.get((
                source.level,
                source.memory_type,
                relation_bucket,
                source.future_option_bucket + delta,
            ), ()))

        type_pool = fallback.get((source.level, source.memory_type), ())
        cross_pool = [
            candidate
            for candidate in type_pool
            if candidate.uid != source.uid and is_cross(source.uid, candidate.uid)
        ]
        cross_pool.sort(key=lambda candidate: (
            -v854._cheap_rank(source, candidate),
            stable_u64(
                int(source.uid.hi),
                int(source.uid.lo),
                int(candidate.uid.hi),
                int(candidate.uid.lo),
                person=b"v854-sim-pair",
            ),
        ))

        # Cross-world evidence is a separate requirement, not merely a fallback
        # when the local structural bucket happens to be sparse.
        candidates = list(bucket_candidates)
        candidates.extend(cross_pool[: self.max_candidates])
        if len(candidates) < self.max_candidates:
            candidates.extend(type_pool)
        unique = {
            candidate.uid: candidate
            for candidate in candidates
            if candidate.uid != source.uid
        }
        ranked_candidates = sorted(
            unique.values(),
            key=lambda candidate: (
                0 if is_cross(source.uid, candidate.uid) else 1,
                -v854._cheap_rank(source, candidate),
                stable_u64(
                    int(source.uid.hi),
                    int(source.uid.lo),
                    int(candidate.uid.hi),
                    int(candidate.uid.lo),
                    person=b"v854-sim-pair",
                ),
            ),
        )[: self.max_candidates]

        scored = []
        for candidate in ranked_candidates:
            self.candidate_comparisons += 1
            evidence = self.score(source, candidate)
            if evidence.score >= self.threshold:
                scored.append(evidence)
        scored.sort(key=lambda evidence: (
            -evidence.score,
            evidence.source_uid,
            evidence.target_uid,
        ))
        cross = [
            evidence
            for evidence in scored
            if is_cross(evidence.source_uid, evidence.target_uid)
        ]
        reserve = max(1, self.top_results // 2)
        chosen = cross[:reserve]
        seen = {(evidence.source_uid, evidence.target_uid) for evidence in chosen}
        for evidence in scored:
            pair = (evidence.source_uid, evidence.target_uid)
            if pair in seen:
                continue
            chosen.append(evidence)
            seen.add(pair)
            if len(chosen) >= self.top_results:
                break
        for evidence in chosen[: self.top_results]:
            results[(evidence.source_uid, evidence.target_uid)] = evidence
        self._processed_versions[source.uid] = source.descriptor_version
        self.processed_descriptors += 1
    return tuple(results[key] for key in sorted(results))


def install_learning_transfer_correctness_v854_fixups() -> None:
    global _INSTALLED, _PRIOR_RESTART_BASE_RESET
    if _INSTALLED:
        return

    from v7.environment.arc_adapter import ArcGridEnvironment
    from v8 import learning_transfer_correctness_v854 as v854
    from v8 import restart_memory_v815 as restart
    from v8 import sampling_transfer_v833 as transfer_sampling
    from v8.peers_v82 import V82DevelopmentalPeerSupervisor
    from v8.similarity import BoundedNeighborhoodSimilarity

    # Restore historical public sampler identities. Their global transfer helper is
    # dynamic, so ordered v8.54 transfer can compose underneath without changing the
    # v8.33/v8.47 authority chain.
    transfer_sampling._forced_action_v833 = v854._BASE_FORCED
    transfer_sampling._discovery_action_v833 = v854._BASE_DISCOVERY
    transfer_sampling._cross_game_transfer_action = _contextual_cross_game_transfer

    # Restore v8.22 as the public reset authority. v8.15's reset wrapper calls its
    # module-global base dynamically, which is the safe point for v8.54 cleanup.
    ArcGridEnvironment.reset = v854._BASE_RESET
    _PRIOR_RESTART_BASE_RESET = restart._BASE_ENV_RESET
    restart._BASE_ENV_RESET = _restart_base_reset_v854

    # v8.54 replaced the peer method after v8.45 was installed. Reapply the same
    # serialization lock around the corrected relation-scoped mutation.
    V82DevelopmentalPeerSupervisor.record_transfer_trial = _locked_record_transfer_trial

    # The first v8.54 evaluator still allowed a full same-world structural bucket to
    # starve cross-world candidates before ranking. Replace both the module helper and
    # installed evaluator so direct tests and production use identical semantics.
    v854._similarity_v854 = _similarity_v854_fixup
    BoundedNeighborhoodSimilarity.evaluate = _similarity_v854_fixup

    _INSTALLED = True
