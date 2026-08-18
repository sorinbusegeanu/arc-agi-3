from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Iterable

from v8.model import CognitiveState, MemoryLevel, MemoryType, MemoryUid, RelationType, ValidationState, stable_u64


_INSTALLED = False
_OLD_PROXY_TRANSFER_INTERVENTION = "leave_one_memory_out_correspondence_ablation"
_BEHAVIORAL_TRANSFER_INTERVENTION = "matched_context_cross_game_transfer"
_H13_HOLDOUT_INTERVENTION = "leave_one_game_out_outcome_rebuild"
_H12_COMPARISON_INTERVENTION = "outcome_comparable_strategy_efficiency"

_BASE_EVALUATE = None
_BASE_CROSS_GAME_TRANSFER_ACTION = None
_BASE_SAMPLER_BEGIN_LEASE = None
_BASE_SAMPLER_EXTERNAL_RESET = None
_BASE_SAMPLER_OBSERVE = None
_BASE_LEARNING_BATCH = None
_BASE_PUBLISH_LEARNING = None
_BASE_PEER_RUN_ONCE = None


def _contract_dependencies() -> dict[str, tuple[str, ...]]:
    return {
        "H01": (),
        "H02": ("H01",),
        "H03": ("H01",),
        "H04": ("H03",),
        "H05": ("H04",),
        "H06": ("H05",),
        "H07": ("H06",),
        "H08": ("H07",),
        "H09": ("H01",),
        "H10": ("H02",),
        "H11": ("H06",),
        "H12": ("H13",),
        "H13": ("H08",),
        "H14": ("H13",),
        "H15": ("H13",),
    }


def _deprecated_evidence(row) -> bool:
    kind = str(getattr(row, "evidence_kind", ""))
    intervention = str(getattr(row, "causal_intervention", ""))
    evidence_id = str(getattr(row, "evidence_id", ""))
    if kind in {"transfer_trial_pass", "concept_transfer_pass"} and intervention == _OLD_PROXY_TRANSFER_INTERVENTION:
        return True
    if kind == "strategy_efficiency" and evidence_id.startswith("strategy_efficiency:") and int(getattr(row, "effect_direction", 0)) <= 0:
        return True
    if kind == "outcome_consistency_holdout" and evidence_id.startswith("outcome_consistency_holdout:") and intervention != _H13_HOLDOUT_INTERVENTION:
        return True
    return False


def _evaluate_v842(self, evidence: Iterable[object]):
    """Two-pass dependency evaluation so forward dependencies such as H12->H13 work."""
    from v8 import evaluation as evaluation_module

    rows = tuple(row for row in evidence if not _deprecated_evidence(row))
    kinds: dict[str, list[object]] = {}
    for row in rows:
        kinds.setdefault(str(row.evidence_kind), []).append(row)

    prelim: dict[str, dict[str, object]] = {}
    contracts = tuple(evaluation_module.CONTRACTS)
    for contract in contracts:
        required_raw = [row for kind in contract.required_kinds for row in kinds.get(kind, ())]
        partial_raw = [row for kind in contract.partial_kinds for row in kinds.get(kind, ())]
        negative_raw = [row for kind in contract.negative_kinds for row in kinds.get(kind, ())]
        required = [row for row in required_raw if self._admissible(row, contract)]
        negative = [row for row in negative_raw if row.quality_valid()]

        if negative and not self._enough(required, contract):
            raw = "INVALID"
        elif self._enough(required_raw, contract):
            raw = "VALID"
        elif partial_raw:
            raw = "PARTIALLY_VALID"
        else:
            raw = "INSUFFICIENT_EVIDENCE"

        quality_pass = self._enough(required, contract)
        if raw == "INVALID" and negative:
            quality_pass = True
        quality_gate = "PASS" if quality_pass else ("NO_EVIDENCE" if not rows else "FAIL")

        prelim[contract.hypothesis_id] = {
            "contract": contract,
            "required_raw": required_raw,
            "partial_raw": partial_raw,
            "negative": negative,
            "required": required,
            "raw": raw,
            "quality_gate": quality_gate,
        }

    resolving: set[str] = set()
    resolved: dict[str, tuple[str, str, tuple[str, ...]]] = {}

    def resolve(hypothesis_id: str) -> tuple[str, str, tuple[str, ...]]:
        cached = resolved.get(hypothesis_id)
        if cached is not None:
            return cached
        if hypothesis_id in resolving:
            raise RuntimeError(f"cyclic hypothesis dependency at {hypothesis_id}")
        resolving.add(hypothesis_id)
        item = prelim[hypothesis_id]
        contract = item["contract"]
        blocked: list[str] = []
        for dep in contract.dependencies:
            dep_final, _dep_gate, _dep_blocked = resolve(dep)
            if not self._dependency_satisfied(dep_final, contract.dependency_min_status):
                blocked.append(dep)
        dependency_gate = "PASS" if not blocked else "BLOCKED"
        raw = str(item["raw"])
        quality_gate = str(item["quality_gate"])
        required_raw = item["required_raw"]
        partial_raw = item["partial_raw"]
        if raw == "INVALID" and quality_gate == "PASS" and dependency_gate == "PASS":
            final = "INVALID"
        elif raw == "VALID" and quality_gate == "PASS" and dependency_gate == "PASS":
            final = "VALID"
        elif partial_raw or required_raw:
            final = "PARTIALLY_VALID"
        else:
            final = "INSUFFICIENT_EVIDENCE"
        resolving.remove(hypothesis_id)
        resolved[hypothesis_id] = (final, dependency_gate, tuple(blocked))
        return resolved[hypothesis_id]

    decisions = []
    for contract in contracts:
        item = prelim[contract.hypothesis_id]
        final, dependency_gate, blocked_dependencies = resolve(contract.hypothesis_id)
        required = item["required"]
        required_raw = item["required_raw"]
        partial_raw = item["partial_raw"]
        raw = str(item["raw"])
        quality_gate = str(item["quality_gate"])
        blockers: list[str] = []
        if raw == "VALID" and not self._enough(required, contract):
            blockers.append("required evidence failed quality/causality/held-out gate")
        elif raw == "PARTIALLY_VALID":
            blockers.append("missing required evidence: " + ",".join(contract.required_kinds))
        elif raw == "INSUFFICIENT_EVIDENCE":
            blockers.append("missing evidence contract fields: " + ",".join(contract.partial_kinds))
        if contract.min_distinct_targets > 0:
            targets = {row.target_game_hash for row in required if row.target_game_hash}
            if len(targets) < contract.min_distinct_targets:
                blockers.append(f"requires {contract.min_distinct_targets} distinct held-out targets; has {len(targets)}")
        if blocked_dependencies:
            blockers.append("blocked dependencies: " + ",".join(blocked_dependencies))
        decisions.append(
            evaluation_module.HypothesisDecision(
                contract.hypothesis_id,
                raw,
                quality_gate,
                dependency_gate,
                final,
                len(required) if required else len(partial_raw),
                "; ".join(blockers),
            )
        )
    return tuple(decisions)


def _install_contracts_and_evaluator() -> None:
    global _BASE_EVALUATE
    from v8 import evaluation as evaluation_module
    from v8.evaluation_v82 import V82ScientificHypothesisEvaluator

    deps = _contract_dependencies()
    updated = []
    for contract in evaluation_module.CONTRACTS:
        kwargs = {"dependencies": deps[contract.hypothesis_id]}
        if contract.hypothesis_id in {"H14", "H15"}:
            kwargs["dependency_min_status"] = "PARTIALLY_VALID"
        updated.append(replace(contract, **kwargs))
    evaluation_module.CONTRACTS = tuple(updated)

    base_evaluator = V82ScientificHypothesisEvaluator.__mro__[1]
    _BASE_EVALUATE = base_evaluator.evaluate
    base_evaluator.evaluate = _evaluate_v842


def _lineage_nodes(view, uid: MemoryUid) -> tuple[MemoryUid, ...]:
    try:
        view._refresh_strategy_cache()
    except BaseException:
        pass
    parents = getattr(view, "_parents", {})
    visited = {uid}
    frontier = {uid}
    for _depth in range(8):
        following = set()
        for current in frontier:
            for parent in parents.get(current, ()):
                if parent not in visited:
                    visited.add(parent)
                    following.add(parent)
        if not following:
            break
        frontier = following
    return tuple(sorted(visited))


def _validation_source_for_strategy(sampler, strategy_uid: MemoryUid) -> MemoryUid | None:
    from v8 import sampling_transfer_v833 as transfer

    view = transfer._current_view()
    if view is None:
        return None
    current_game = int(stable_u64(str(sampler.game_id), person=b"v8-game"))
    try:
        view._refresh_strategy_cache()
    except BaseException:
        pass
    version = tuple(getattr(view, "_strategy_version", ()))
    cache = getattr(sampler, "_v842_validation_source_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        sampler._v842_validation_source_cache = cache
    key = (strategy_uid, current_game, version)
    if key in cache:
        return cache[key]

    nodes = dict(getattr(view, "_node_by_uid", {}))
    if not nodes:
        nodes = {row.uid: row for row in view.node_records()}
    lineage = _lineage_nodes(view, strategy_uid)
    correspondences: dict[MemoryUid, list[tuple[MemoryUid, float]]] = defaultdict(list)
    for edge in view.edge_records():
        if int(edge.relation_type) != int(RelationType.TRANSFER_CORRESPONDENCE):
            continue
        score = max(0.0, float(edge.score))
        if score <= 0.0:
            continue
        correspondences[edge.source_uid].append((edge.target_uid, score))
        correspondences[edge.target_uid].append((edge.source_uid, score))

    best: tuple[float, MemoryUid] | None = None
    for candidate_uid in lineage:
        row = nodes.get(candidate_uid)
        if row is None or int(row.level) not in {int(MemoryLevel.M3), int(MemoryLevel.M4)}:
            continue
        formation_games = set(int(v) for v in view.source_games(candidate_uid))
        if not formation_games or current_game in formation_games:
            continue
        for other_uid, score in correspondences.get(candidate_uid, ()):
            if current_game not in set(int(v) for v in view.source_games(other_uid)):
                continue
            candidate = (float(score), candidate_uid)
            if best is None or (candidate[0], -candidate[1].hi, -candidate[1].lo) > (best[0], -best[1].hi, -best[1].lo):
                best = candidate
    result = None if best is None else best[1]
    cache[key] = result
    if len(cache) > 256:
        for stale in tuple(cache)[:-256]:
            cache.pop(stale, None)
    return result


def _cross_game_transfer_action_v842(sampler, actions):
    selected = _BASE_CROSS_GAME_TRANSFER_ACTION(sampler, actions)
    sampler._v842_transfer_validation_uid = None
    if selected is None:
        return None
    action, origin, uid = selected
    if str(origin) == "M7" and uid is not None:
        sampler._v842_transfer_validation_uid = _validation_source_for_strategy(sampler, uid)
    return int(action), str(origin), uid


def _sampler_begin_lease_v842(self, seed: int) -> None:
    _BASE_SAMPLER_BEGIN_LEASE(self, int(seed))
    self._v842_transfer_validation_uid = None
    self._v842_transfer_baselines = {}
    self._v842_transfer_stats = {}
    self._v842_validation_source_cache = {}


def _sampler_external_reset_v842(self) -> None:
    _BASE_SAMPLER_EXTERNAL_RESET(self)
    self._v842_transfer_validation_uid = None


def _sampler_observe_v842(self, **kwargs) -> None:
    intervention = getattr(self.base, "current", None)
    kind = "" if intervention is None else str(getattr(intervention, "kind", ""))
    validation_uid = getattr(self, "_v842_transfer_validation_uid", None)
    before_level = int(kwargs.get("before_level", 0))
    before_context = int(kwargs.get("before_context", 0))
    baseline_key = (before_level, before_context)
    success = bool(kwargs.get("level_advanced", False) or str(kwargs.get("terminal_state", "")) == "WIN")

    _BASE_SAMPLER_OBSERVE(self, **kwargs)

    baselines = getattr(self, "_v842_transfer_baselines", None)
    if not isinstance(baselines, dict):
        baselines = {}
        self._v842_transfer_baselines = baselines
    stats = getattr(self, "_v842_transfer_stats", None)
    if not isinstance(stats, dict):
        stats = {}
        self._v842_transfer_stats = stats

    if kind == "CROSS_GAME_TRANSFER" and validation_uid is not None:
        baseline = baselines.get(baseline_key)
        if baseline is not None and float(baseline[0]) > 0.0:
            baseline_rate = float(baseline[1]) / float(baseline[0])
            row = stats.setdefault(validation_uid, [0.0, 0.0, 0.0])
            row[0] += 1.0
            row[1] += 1.0 if success else 0.0
            row[2] += max(0.0, min(1.0, baseline_rate))
        self._v842_transfer_validation_uid = None
        return

    if kind in {"RANDOM", "RANDOM_WALK", "DISCOVERY", "SEQUENCE"}:
        baseline = baselines.setdefault(baseline_key, [0.0, 0.0])
        baseline[0] += 1.0
        baseline[1] += 1.0 if success else 0.0
    self._v842_transfer_validation_uid = None


def _transfer_stats_for_job(job) -> tuple[tuple[MemoryUid, int, int, float], ...]:
    try:
        from v8 import decision_point_sampling_v821 as sampling
        from v8.sampling_portfolio_v831 import PortfolioSampler
    except BaseException:
        return ()
    sampler = sampling._SAMPLERS.get((int(job.actor_id), str(job.game_id)))
    if not isinstance(sampler, PortfolioSampler):
        return ()
    stats = getattr(sampler, "_v842_transfer_stats", {})
    result = []
    for uid, values in sorted(stats.items()):
        attempts = max(0, int(round(float(values[0]))))
        if attempts <= 0:
            continue
        successes = max(0, min(attempts, int(round(float(values[1])))))
        result.append((uid, attempts, successes, float(values[2])))
    return tuple(result)


def _clear_transfer_stats_for_job(job) -> None:
    try:
        from v8 import decision_point_sampling_v821 as sampling
        from v8.sampling_portfolio_v831 import PortfolioSampler
    except BaseException:
        return
    sampler = sampling._SAMPLERS.get((int(job.actor_id), str(job.game_id)))
    if isinstance(sampler, PortfolioSampler):
        sampler._v842_transfer_stats = {}


def _learning_batch_v842(*, job, strategy_stats, preference_probes, replanning_trials):
    from v8 import actor as actor_module

    batch = _BASE_LEARNING_BATCH(
        job=job,
        strategy_stats=strategy_stats,
        preference_probes=preference_probes,
        replanning_trials=replanning_trials,
    )
    extras = tuple(
        actor_module.StrategyRunStat(uid, attempts, successes, baseline_sum)
        for uid, attempts, successes, baseline_sum in _transfer_stats_for_job(job)
    )
    if not extras:
        return batch
    if batch is None:
        return actor_module.ActorLearningBatch(
            int(job.actor_id),
            str(job.game_id),
            extras,
            (),
            (),
            0,
        )
    return replace(batch, strategy_stats=tuple(batch.strategy_stats) + extras)


def _publish_learning_v842(progress_queue, *, job, strategy_stats, preference_probes, replanning_trials) -> bool:
    accepted = bool(
        _BASE_PUBLISH_LEARNING(
            progress_queue,
            job=job,
            strategy_stats=strategy_stats,
            preference_probes=preference_probes,
            replanning_trials=replanning_trials,
        )
    )
    if accepted:
        _clear_transfer_stats_for_job(job)
    return accepted


def _record_strategy_statistics_v842(
    self,
    uid: MemoryUid,
    *,
    attempts: int,
    successes: int,
    cost: float,
    source_game_hash: int,
) -> bool:
    attempts = max(0, int(attempts))
    if attempts <= 0:
        return False

    row = next(
        (
            value
            for value in self.read_view.node_records(level=MemoryLevel.M7)
            if value.uid == uid
        ),
        None,
    )
    if row is not None:
        self._submit(
            self._existing_proposal(
                row,
                success_sum=float(max(0, successes)),
                cost_sum=float(max(0.0, cost)),
                attempt_weight=float(attempts),
                source_game_hash=int(source_game_hash),
            )
        )
        return True

    transfer_row = None
    for level in (MemoryLevel.M3, MemoryLevel.M4):
        transfer_row = next((value for value in self.read_view.node_records(level=level) if value.uid == uid), None)
        if transfer_row is not None:
            break
    if transfer_row is None:
        return False
    target_game = int(source_game_hash)
    formation_games = tuple(sorted(int(v) for v in self.read_view.source_games(uid)))
    if target_game == 0 or not formation_games or target_game in set(formation_games):
        return False
    metric_on = max(0.0, min(1.0, float(successes) / float(attempts)))
    metric_off = max(0.0, min(1.0, float(cost) / float(attempts)))
    self.record_transfer_trial(
        uid,
        target_game_hash=target_game,
        metric_on=metric_on,
        metric_off=metric_off,
        formation_games=formation_games,
        intervention=_BEHAVIORAL_TRANSFER_INTERVENTION,
    )
    return True


def _emit_strategy_efficiency_comparisons(self) -> None:
    nodes = tuple(self.read_view.node_records())
    by_uid = {row.uid: row for row in nodes}
    for outcome_uid, alternatives in self.strategies.by_outcome(nodes).items():
        observed = [
            strategy
            for strategy in alternatives
            if (row := by_uid.get(strategy.uid)) is not None and float(row.attempt_weight) > 0.0
        ]
        if len(observed) < 2:
            continue
        scored = sorted(
            (
                (float(strategy.reliability) / max(1e-9, float(strategy.mean_cost)), strategy)
                for strategy in observed
            ),
            key=lambda item: (item[0], item[1].uid),
        )
        worst_score, worst = scored[0]
        best_score, best = scored[-1]
        gain = float(best_score) - float(worst_score)
        if gain <= 1e-12 or best.uid == worst.uid:
            continue
        best_row = by_uid.get(best.uid)
        worst_row = by_uid.get(worst.uid)
        if best_row is None or worst_row is None:
            continue
        watermark = max(int(best_row.updated_watermark), int(worst_row.updated_watermark))
        freshness = f"strategy-efficiency-comparison:{outcome_uid.hex()}:{worst.uid.hex()}"
        if not self._fresh(freshness, best.uid, watermark):
            continue
        games = tuple(sorted(self.read_view.source_games(best.uid) | self.read_view.source_games(worst.uid)))
        self._append_evidence(
            "strategy_efficiency",
            best_row,
            min(1.0, gain),
            unique=True,
            provenance_games=games,
            causal_intervention=_H12_COMPARISON_INTERVENTION,
            effect_direction=1,
        )


def _peer_run_once_v842(self) -> None:
    _BASE_PEER_RUN_ONCE(self)
    _emit_strategy_efficiency_comparisons(self)


def _no_proxy_auto_transfer_trials(self, nodes, edges) -> None:
    return None


def _fresh_outcome_estimator(source):
    from v8.outcomes import OutcomeEquivalenceEstimator

    return OutcomeEquivalenceEstimator(
        min_support=int(source.min_support),
        stability_threshold=float(source.stability_threshold),
        context_consistency_threshold=float(source.context_consistency_threshold),
        max_diameter=float(source.max_diameter),
        interchangeability_threshold=float(source.interchangeability_threshold),
    )


def _auto_outcome_holdout_v842(self, nodes, edges) -> None:
    """Rebuild each M6 class without the target game before evaluating that target."""
    from v8 import hypothesis_validation_v054 as v054

    nodes = tuple(nodes)
    games = v054._provenance_lookup(tuple(edges))
    m6 = tuple(
        row
        for row in nodes
        if int(row.level) == int(MemoryLevel.M6)
        and int(row.memory_type) == int(MemoryType.OUTCOME)
        and len(row.key_parts) >= 2
    )
    all_games = sorted({int(game) for row in m6 for game in games(row.uid)})
    emitted = 0
    budget = max(1, min(16, int(getattr(self, "candidate_budget", 16))))
    for target_game in all_games:
        formation_nodes = tuple(
            row
            for row in m6
            if (row_games := set(int(v) for v in games(row.uid)))
            and int(target_game) not in row_games
        )
        target_rows = tuple(row for row in m6 if int(target_game) in set(int(v) for v in games(row.uid)))
        if not formation_nodes or not target_rows:
            continue
        estimator = _fresh_outcome_estimator(self.outcomes)
        classes = tuple(estimator.rebuild(formation_nodes))
        target_by_descriptor: dict[tuple[int, int], list[object]] = defaultdict(list)
        for row in target_rows:
            target_by_descriptor[(int(row.key_parts[0]), int(row.key_parts[1]))].append(row)
        formation_by_uid = {row.uid: row for row in formation_nodes}
        for outcome in sorted(classes, key=lambda item: item.uid):
            if not bool(outcome.persistent):
                continue
            matches = target_by_descriptor.get(tuple(int(v) for v in outcome.descriptor), ())
            if not matches:
                continue
            formation_games: set[int] = set()
            for member_uid in outcome.members:
                if member_uid not in formation_by_uid:
                    continue
                formation_games.update(int(v) for v in games(member_uid))
            formation_games.discard(int(target_game))
            if not formation_games:
                continue
            score = min(
                float(outcome.stability),
                float(outcome.context_consistency),
                float(outcome.predictive_interchangeability),
            )
            if score <= 0.0:
                continue
            row = sorted(matches, key=lambda item: (-int(item.support_count), item.uid))[0]
            freshness = f"outcome-holdout-v842:{int(target_game)}:{int(outcome.descriptor[0])}:{int(outcome.descriptor[1])}"
            if not self._fresh(freshness, row.uid, row.updated_watermark):
                continue
            self._append_evidence(
                "outcome_consistency_holdout",
                row,
                score,
                validation_state=int(ValidationState.VALIDATED),
                unique=True,
                target_game_hash=int(target_game),
                provenance_games=tuple(sorted(formation_games)),
                causal_intervention=_H13_HOLDOUT_INTERVENTION,
                effect_direction=1,
            )
            emitted += 1
            if emitted >= budget:
                return


def _similarity_evaluate_v842(self, nodes, edges):
    """Reserve bounded similarity slots for different-game candidates."""
    nodes = tuple(nodes)
    edges = tuple(edges)
    descriptors = self.descriptors(nodes, edges)
    by_uid = {row.uid: row for row in nodes}
    index: dict[tuple[int, int, int, int], list[object]] = defaultdict(list)
    fallback: dict[tuple[int, int], list[object]] = defaultdict(list)
    for descriptor in sorted(descriptors.values(), key=lambda item: item.uid):
        index[(descriptor.level, descriptor.memory_type, self._relation_bucket(descriptor), descriptor.future_option_bucket)].append(descriptor)
        fallback[(descriptor.level, descriptor.memory_type)].append(descriptor)

    dirty = [
        descriptor
        for descriptor in sorted(descriptors.values(), key=lambda item: item.uid)
        if descriptor.descriptor_version > self._processed_versions.get(descriptor.uid, -1)
    ]
    results = {}
    for descriptor in dirty:
        candidates = []
        relation_bucket = self._relation_bucket(descriptor)
        for future_delta in (0, -1, 1):
            candidates.extend(index.get((descriptor.level, descriptor.memory_type, relation_bucket, descriptor.future_option_bucket + future_delta), ()))
        if len(candidates) < self.max_candidates:
            candidates.extend(fallback.get((descriptor.level, descriptor.memory_type), ()))
        unique_candidates = {candidate.uid: candidate for candidate in candidates if candidate.uid != descriptor.uid}
        source_row = by_uid.get(descriptor.uid)
        source_mask = 0 if source_row is None else int(source_row.game_mask)

        def candidate_rank(candidate):
            target_row = by_uid.get(candidate.uid)
            target_mask = 0 if target_row is None else int(target_row.game_mask)
            cross_game = bool(source_mask and target_mask and source_mask != target_mask)
            overlap = (source_mask & target_mask).bit_count() if cross_game else 64
            return (0 if cross_game else 1, overlap, candidate.uid)

        scored = []
        for candidate in sorted(unique_candidates.values(), key=candidate_rank)[: self.max_candidates]:
            self.candidate_comparisons += 1
            evidence = self.score(descriptor, candidate)
            if evidence.score >= self.threshold:
                scored.append(evidence)
        ranked = sorted(scored, key=lambda item: (-item.score, item.source_uid, item.target_uid))

        def is_cross_game(evidence) -> bool:
            left = by_uid.get(evidence.source_uid)
            right = by_uid.get(evidence.target_uid)
            if left is None or right is None:
                return False
            left_mask, right_mask = int(left.game_mask), int(right.game_mask)
            return bool(left_mask and right_mask and left_mask != right_mask)

        cross = [row for row in ranked if is_cross_game(row)]
        chosen = cross[: max(1, self.top_results // 2)] if cross else []
        chosen_keys = {(row.source_uid, row.target_uid) for row in chosen}
        for row in ranked:
            key = (row.source_uid, row.target_uid)
            if key in chosen_keys:
                continue
            chosen.append(row)
            chosen_keys.add(key)
            if len(chosen) >= self.top_results:
                break
        for row in chosen[: self.top_results]:
            results[(row.source_uid, row.target_uid)] = row
        self._processed_versions[descriptor.uid] = descriptor.descriptor_version
        self.processed_descriptors += 1
    return tuple(results[key] for key in sorted(results, key=lambda pair: (pair[0], pair[1])))


def _hypothesis_status_line_v842(evidence_rows, watermark_value: int) -> str:
    from v8.diagnostics import format_hypothesis_line
    from v8.evaluation_v82 import V82ScientificHypothesisEvaluator

    cut = tuple(
        row
        for row in evidence_rows
        if int(getattr(row, "watermark", getattr(row, "evidence_available_watermark", 0))) <= int(watermark_value)
    )
    evaluator = V82ScientificHypothesisEvaluator()
    statuses = evaluator.status_map(evaluator.evaluate(cut))
    return format_hypothesis_line(statuses)


def install_hypothesis_validation_v842() -> None:
    global _INSTALLED
    global _BASE_CROSS_GAME_TRANSFER_ACTION, _BASE_SAMPLER_BEGIN_LEASE
    global _BASE_SAMPLER_EXTERNAL_RESET, _BASE_SAMPLER_OBSERVE
    global _BASE_LEARNING_BATCH, _BASE_PUBLISH_LEARNING, _BASE_PEER_RUN_ONCE
    if _INSTALLED:
        return

    _install_contracts_and_evaluator()

    from v8 import actor as actor_module
    from v8 import hypothesis_validation_v054 as v054
    from v8 import runtime_observability_v836 as observability
    from v8 import sampling_transfer_v833 as transfer
    from v8.evaluation_v82 import V82ScientificHypothesisEvaluator
    from v8.sampling_portfolio_v831 import PortfolioSampler
    from v8.similarity import BoundedNeighborhoodSimilarity

    _BASE_CROSS_GAME_TRANSFER_ACTION = transfer._cross_game_transfer_action
    transfer._cross_game_transfer_action = _cross_game_transfer_action_v842

    _BASE_SAMPLER_BEGIN_LEASE = PortfolioSampler.begin_lease
    _BASE_SAMPLER_EXTERNAL_RESET = PortfolioSampler.on_external_reset
    _BASE_SAMPLER_OBSERVE = PortfolioSampler.observe_transition
    PortfolioSampler.begin_lease = _sampler_begin_lease_v842
    PortfolioSampler.on_external_reset = _sampler_external_reset_v842
    PortfolioSampler.observe_transition = _sampler_observe_v842

    _BASE_LEARNING_BATCH = actor_module._learning_batch
    _BASE_PUBLISH_LEARNING = actor_module._publish_learning
    actor_module._learning_batch = _learning_batch_v842
    actor_module._publish_learning = _publish_learning_v842

    from v8.peers_v82 import V82DevelopmentalPeerSupervisor
    peer_base = V82DevelopmentalPeerSupervisor.__mro__[1]
    peer_base.record_strategy_statistics = _record_strategy_statistics_v842

    _BASE_PEER_RUN_ONCE = V82DevelopmentalPeerSupervisor.run_once
    V82DevelopmentalPeerSupervisor.run_once = _peer_run_once_v842

    v054._auto_transfer_trials = _no_proxy_auto_transfer_trials
    v054._auto_outcome_holdout = _auto_outcome_holdout_v842

    BoundedNeighborhoodSimilarity.evaluate = _similarity_evaluate_v842
    observability._hypothesis_status_line = _hypothesis_status_line_v842

    _INSTALLED = True
