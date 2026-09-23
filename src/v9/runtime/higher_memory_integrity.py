from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

from v9.memory.provenance import DerivationProvenance
from v9.mutation.proposals import MutationKind, ProposalClass


def _merge_provenance(left: DerivationProvenance, right: DerivationProvenance) -> DerivationProvenance:
    return DerivationProvenance(
        tuple(sorted(set(left.parents) | set(right.parents))),
        tuple(sorted(set(left.evidence) | set(right.evidence))),
        tuple(sorted(set(left.formation_scope) | set(right.formation_scope))),
    )


def _republish_outcome(runtime: Any, outcome: Any) -> None:
    node = runtime.graph.nodes.get(outcome.uid)
    payload = runtime.graph.payloads.get(outcome.uid)
    if node is None or payload is None:
        return
    runtime._publish(
        node,
        {
            **payload,
            "class_signature": list(outcome.class_signature),
            "class_version": int(outcome.class_version),
            "equivalence_trials": int(outcome.equivalence_trials),
            "equivalence_successes": int(outcome.equivalence_successes),
            "contexts_observed": list(outcome.contexts_observed),
            "environments_observed": list(outcome.environments_observed),
            "primary_valence_sum": int(outcome.primary_valence_sum),
            "preference_trials": int(outcome.preference_trials),
            "parents": [[uid.hi, uid.lo] for uid in outcome.provenance.parents],
        },
        outcome.provenance.evidence,
        proposal_class=ProposalClass.STATEFUL,
        mutation_kind=MutationKind.UPDATE_VALIDATION,
    )


def _republish_strategy(runtime: Any, strategy: Any) -> None:
    node = runtime.graph.nodes.get(strategy.uid)
    payload = runtime.graph.payloads.get(strategy.uid)
    if node is None or payload is None:
        return
    runtime._publish(
        node,
        {
            **payload,
            "target_outcome": [strategy.target_outcome.hi, strategy.target_outcome.lo],
            "target_environment_id": int(strategy.target_environment_id),
            "native_actions": list(strategy.native_actions),
            "reliability_successes": int(strategy.reliability_successes),
            "reliability_trials": int(strategy.reliability_trials),
            "primary_valence_sum": int(strategy.primary_valence_sum),
            "realized_cost_sum": int(strategy.realized_cost_sum),
            "parents": [[uid.hi, uid.lo] for uid in strategy.provenance.parents],
        },
        strategy.provenance.evidence,
        proposal_class=ProposalClass.STATEFUL,
        mutation_kind=MutationKind.UPDATE_VALIDATION,
    )


def install(runtime_class: type) -> None:
    if getattr(runtime_class, "_higher_memory_integrity_installed", False):
        return

    original_record: Callable[..., Any] = runtime_class.record_transfer_validation

    def record_transfer_validation(self: Any, concept_uid: Any, **kwargs: Any) -> None:
        # Formation of a new M5 member can legitimately expand an existing M6
        # outcome class. Keep the empirical state already learned for that class
        # and for any existing strategies targeting it.
        before_m6 = dict(getattr(self, "_m6", {}))
        before_m7 = dict(getattr(self, "_m7", {}))
        original_record(self, concept_uid, **kwargs)

        preserved_outcomes = 0
        expanded_outcomes = 0
        for uid, old in before_m6.items():
            current = getattr(self, "_m6", {}).get(uid)
            if current is None:
                continue
            old_members = set(old.members)
            current_members = set(current.members)
            members = tuple(sorted(old_members | current_members))
            expanded = current_members - old_members
            merged = replace(
                current,
                members=members,
                provenance=_merge_provenance(old.provenance, current.provenance),
                class_version=max(int(old.class_version), int(current.class_version)) + int(bool(expanded)),
                equivalence_trials=max(int(old.equivalence_trials), int(current.equivalence_trials)),
                equivalence_successes=max(int(old.equivalence_successes), int(current.equivalence_successes)),
                contexts_observed=tuple(sorted(set(old.contexts_observed) | set(current.contexts_observed))),
                environments_observed=tuple(sorted(set(old.environments_observed) | set(current.environments_observed))),
                primary_valence_sum=(
                    int(old.primary_valence_sum)
                    if int(old.preference_trials) >= int(current.preference_trials)
                    else int(current.primary_valence_sum)
                ),
                preference_trials=max(int(old.preference_trials), int(current.preference_trials)),
            )
            if merged != current:
                self._m6[uid] = merged
                _republish_outcome(self, merged)
                preserved_outcomes += 1
                expanded_outcomes += int(bool(expanded))

        preserved_strategies = 0
        for uid, old in before_m7.items():
            current = getattr(self, "_m7", {}).get(uid)
            if current is None:
                continue
            if int(old.reliability_trials) > int(current.reliability_trials):
                empirical = old
            elif int(old.reliability_trials) < int(current.reliability_trials):
                empirical = current
            else:
                empirical = old if int(old.reliability_successes) >= int(current.reliability_successes) else current
            merged = replace(
                current,
                reliability_successes=int(empirical.reliability_successes),
                reliability_trials=int(empirical.reliability_trials),
                primary_valence_sum=int(empirical.primary_valence_sum),
                realized_cost_sum=int(empirical.realized_cost_sum),
                provenance=_merge_provenance(old.provenance, current.provenance),
            )
            if merged != current:
                self._m7[uid] = merged
                _republish_strategy(self, merged)
                preserved_strategies += 1

        if preserved_outcomes:
            self.set_telemetry_gauge("m6_state_preservations", preserved_outcomes)
        if expanded_outcomes:
            self.set_telemetry_gauge("m6_class_expansions", expanded_outcomes)
        if preserved_strategies:
            self.set_telemetry_gauge("m7_state_preservations", preserved_strategies)

    runtime_class.record_transfer_validation = record_transfer_validation
    runtime_class._higher_memory_integrity_installed = True
