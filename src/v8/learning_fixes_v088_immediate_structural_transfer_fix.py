from __future__ import annotations

import copy

from v8.persistent_identity import world_id


_INSTALLED = False
_BRANCH_STRUCTURAL: dict[int, dict[str, dict[str, object]]] = {}


def _expected_profile(diagnostic: dict[str, object]) -> dict[str, object] | None:
    evidence = diagnostic.get("target_interaction_grounding_evidence")
    if not isinstance(evidence, dict):
        return None
    m2_key = evidence.get("target_ephemeral_m2_key")
    if not isinstance(m2_key, (list, tuple)) or len(m2_key) != 2:
        return None
    family = evidence.get("target_ephemeral_family_token")
    future = evidence.get("target_ephemeral_future_bucket")
    transform = evidence.get("observed_target_transformation_family_signature")
    if family is None or future is None or transform is None:
        return None
    return {
        "m2_key": tuple(int(v) for v in m2_key),
        "family_token": int(family),
        "future_bucket": int(future),
        "transformation_family_signature": int(transform),
    }


def _observed_profile(environment, action: int) -> dict[str, object] | None:
    from v7.environment.encoding import (
        changed_cell_count,
        structural_grid_signature,
        transformation_family_signature,
        transition_signature,
    )
    from v8.learning_fixes_v088_target_ephemeral_production_grounding_fix import (
        _production_ephemeral_identity,
    )

    env = copy.deepcopy(environment)
    before = env.observe()
    before_actions = tuple(sorted(set(int(v) for v in env.available_actions())))
    if int(action) not in set(before_actions):
        return None
    env.step(int(action))
    after = env.observe()
    after_actions = tuple(sorted(set(int(v) for v in env.available_actions())))
    context = int(structural_grid_signature(before))
    next_context = int(structural_grid_signature(after))
    outcome = int(transition_signature(before, after))
    future_delta = float(len(after_actions) - len(before_actions))
    identity = _production_ephemeral_identity(
        context_signature=context,
        action_id=int(action),
        outcome_signature=outcome,
        next_context_signature=next_context,
        future_option_delta=future_delta,
    )
    return {
        "action_id": int(action),
        "context_signature": context,
        "next_context_signature": next_context,
        "m2_key": tuple(int(v) for v in identity["m2_key"]),
        "family_token": int(identity["family_token"]),
        "future_bucket": int(identity["future_bucket"]),
        "transformation_family_signature": int(
            transformation_family_signature(before, after)
        ),
        "changed_cell_count": int(changed_cell_count(before, after)),
        "future_option_delta": future_delta,
    }


def _structural_alignment(
    observed: dict[str, object] | None,
    expected: dict[str, object] | None,
) -> float | None:
    """Production-role alignment without semantic labels or terminal outcomes."""
    if not isinstance(observed, dict) or not isinstance(expected, dict):
        return None
    # M2 family identity and functional transformation carry most of the signal;
    # future-option sign is a smaller independent structural consequence.
    return (
        0.40 * float(tuple(observed.get("m2_key", ())) == tuple(expected["m2_key"]))
        + 0.40
        * float(
            int(observed.get("transformation_family_signature", -1))
            == int(expected["transformation_family_signature"])
        )
        + 0.20
        * float(
            int(observed.get("future_bucket", 99))
            == int(expected["future_bucket"])
        )
    )


def _install_immediate_structural_transfer() -> None:
    from v8 import learning_fixes_v088 as learning
    from v8 import information_flow_diagnostics as flow
    from v8.transfer import TransferTrial, TransferValidator

    current_probe = learning._probe_policy_v088
    current_record_trial = TransferValidator.record_trial

    def probe_policy_v088(**kwargs):
        environment = kwargs.get("environment")
        start = copy.deepcopy(environment) if environment is not None else None
        metric, used = current_probe(**kwargs)
        diagnostic = kwargs.get("diagnostic")
        if not isinstance(diagnostic, dict) or start is None:
            return metric, used

        executed = tuple(int(v) for v in diagnostic.get("executed_target_actions", ()))
        if not executed:
            return metric, used
        observed = _observed_profile(start, executed[0])
        target_hash = int(world_id(str(kwargs.get("game_id", ""))))
        branch = "on" if kwargs.get("required_ancestor") is not None else "off"

        expected = _expected_profile(diagnostic)
        state = _BRANCH_STRUCTURAL.setdefault(target_hash, {})
        if expected is None and branch == "off":
            prior_on = state.get("on", {})
            candidate = prior_on.get("expected") if isinstance(prior_on, dict) else None
            if isinstance(candidate, dict):
                expected = dict(candidate)
        alignment = _structural_alignment(observed, expected)
        state[branch] = {
            "observed": observed or {},
            "expected": expected or {},
            "alignment": alignment,
        }
        diagnostic["immediate_structural_consequence"] = observed
        diagnostic["immediate_structural_expected_role"] = expected
        diagnostic["immediate_structural_alignment"] = alignment
        return metric, used

    def record_trial(
        self,
        uid,
        *,
        target_game_hash: int,
        metric_on: float,
        metric_off: float,
        formation_games=(),
        intervention: str = "matched_memory_ablation",
    ):
        if str(intervention) != "matched_arc_target_memory_vs_memory_free":
            return current_record_trial(
                self,
                uid,
                target_game_hash=target_game_hash,
                metric_on=metric_on,
                metric_off=metric_off,
                formation_games=formation_games,
                intervention=intervention,
            )

        formation = tuple(sorted(set(int(v) for v in formation_games)))
        target = int(target_game_hash)
        held_out = not formation or target not in formation
        long_effect = float(metric_on) - float(metric_off)
        branches = _BRANCH_STRUCTURAL.pop(target, {})
        on = branches.get("on", {}) if isinstance(branches, dict) else {}
        off = branches.get("off", {}) if isinstance(branches, dict) else {}
        on_alignment = on.get("alignment") if isinstance(on, dict) else None
        off_alignment = off.get("alignment") if isinstance(off, dict) else None
        if on_alignment is None or off_alignment is None:
            immediate_gain = 0.0
            immediate_available = False
        else:
            immediate_gain = float(on_alignment) - float(off_alignment)
            immediate_available = True

        # Two-timescale effect: retain the existing long-horizon causal score and
        # add only a directional structural gain relative to the grounded role.
        effect = long_effect + immediate_gain
        trial = TransferTrial(
            uid,
            target,
            float(metric_on),
            float(metric_off),
            float(effect),
            bool(held_out and effect > self.effect_threshold),
            formation,
            str(intervention),
        )
        self._trials.setdefault(uid, []).append(trial)
        flow.emit(
            "transfer",
            "immediate_structural_transfer_effect",
            input_count=1,
            output_count=int(immediate_available),
            rejection_counts=(
                {} if immediate_available else {"immediate_structural_profile_unavailable": 1}
            ),
            fields={
                "target_world_hash": target,
                "immediate_structural_available": bool(immediate_available),
                "intervention_structural_similarity": on_alignment,
                "control_structural_similarity": off_alignment,
                "immediate_structural_gain": float(immediate_gain),
                "long_horizon_effect": float(long_effect),
                "two_timescale_transfer_effect": float(effect),
                "effect_rule": "long_horizon_effect + immediate_structural_gain",
                "structural_alignment_components": [
                    "production_m2_family",
                    "transformation_family",
                    "future_option_bucket",
                ],
            },
        )
        return trial

    learning._probe_policy_v088 = probe_policy_v088
    TransferValidator.record_trial = record_trial

    # _install_transfer_experiments captured the prior probe function into the
    # experiments module, so refresh that public hook as well.
    try:
        from v8 import experiments as experiments_module
        experiments_module._probe_policy = probe_policy_v088
    except BaseException:
        pass


def install_immediate_structural_transfer_fix() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_immediate_structural_transfer()
    _INSTALLED = True
