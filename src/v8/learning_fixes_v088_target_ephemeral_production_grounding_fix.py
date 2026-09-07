from __future__ import annotations

import copy
from collections import Counter, defaultdict
from dataclasses import replace
from random import Random

from v8.model import MemoryLevel, MemoryType, MemoryUid, RelationType, stable_u64
from v8.persistent_identity import world_id
from v8.similarity import BoundedNeighborhoodSimilarity, NeighborhoodDescriptor
from v8.structural_correspondence import StructuralCorrespondenceEstimator


_INSTALLED = False
_TARGET_GROUNDING_MAX_STEPS = 8


def _future_bucket(value: float) -> int:
    return 1 if float(value) > 1e-9 else -1 if float(value) < -1e-9 else 0


def _production_ephemeral_identity(
    *,
    context_signature: int,
    action_id: int,
    outcome_signature: int,
    next_context_signature: int,
    future_option_delta: float,
) -> dict[str, object]:
    """Mirror production M1->M2->M3 canonical identity without publishing memory."""
    m1_key = (
        int(context_signature),
        int(action_id),
        int(outcome_signature),
        int(next_context_signature),
    )
    m2_key = (int(action_id), int(outcome_signature) & 0xFFFF)
    family_uid = MemoryUid.from_key(MemoryLevel.M2, MemoryType.FAMILY, m2_key)
    family_token = stable_u64(
        int(family_uid.hi), int(family_uid.lo), person=b"v8.2-family"
    )
    carrier_token = stable_u64(
        int(context_signature),
        int(next_context_signature),
        person=b"v8.2-carrier",
    )
    future = _future_bucket(float(future_option_delta))
    return {
        "m1_key": m1_key,
        "m2_key": m2_key,
        "family_uid": family_uid,
        "family_token": int(family_token),
        "carrier_token": int(carrier_token),
        "future_bucket": int(future),
        "role_group": (int(family_token), int(future)),
    }


def _target_role_descriptor(reference, *, family_token: int, future_bucket: int, carrier_count: int, step_index: int):
    count = max(1, int(carrier_count))
    uid = MemoryUid.from_key(
        MemoryLevel.M3,
        MemoryType(int(reference.memory_type)),
        (int(family_token), int(future_bucket)),
    )
    return NeighborhoodDescriptor(
        uid=uid,
        level=int(MemoryLevel.M3),
        memory_type=int(reference.memory_type),
        incoming_relations=(),
        outgoing_relations=((int(RelationType.EXPLAINS), count),),
        neighbor_levels=((int(MemoryLevel.M3), count),),
        neighbor_types=((int(MemoryType.CARRIER), count),),
        dependency_signature=0,
        enable_block_signature=0,
        future_option_bucket=0,
        consequence_bucket=0,
        context_bucket=0,
        descriptor_version=int(step_index) + 1,
    )


def _target_structural_counter(carrier_count: int) -> Counter[tuple[int, int, int, int]]:
    return Counter(
        {
            (
                1,
                int(RelationType.EXPLAINS),
                int(MemoryLevel.M3),
                int(MemoryType.CARRIER),
            ): max(1, int(carrier_count))
        }
    )


def _structural_match(reference_row: dict[str, object], target, carrier_count: int) -> dict[str, object]:
    similarity = BoundedNeighborhoodSimilarity()
    correspondence = StructuralCorrespondenceEstimator()
    reference = reference_row["descriptor"]
    evidence = similarity.score(reference, target)
    left = reference_row["structural_counter"]
    right = _target_structural_counter(carrier_count)
    preserved_lr, mismatched_lr, mapping_lr, epsilon_lr = correspondence._error(left, right)
    preserved_rl, mismatched_rl, mapping_rl, epsilon_rl = correspondence._error(right, left)
    mapping_size = min(mapping_lr, mapping_rl)
    epsilon = max(epsilon_lr, epsilon_rl)
    preserved = min(preserved_lr, preserved_rl)
    mismatched = max(mismatched_lr, mismatched_rl)
    accepted = bool(
        float(evidence.score) >= float(similarity.threshold)
        and int(mapping_size) > 0
        and float(epsilon) < float(correspondence.theta_struct)
    )
    return {
        "reference_uid": str(reference_row["uid"]),
        "similarity_score": float(evidence.score),
        "similarity_threshold": float(similarity.threshold),
        "epsilon_struct": float(epsilon),
        "theta_struct": float(correspondence.theta_struct),
        "mapping_size": int(mapping_size),
        "preserved_edges": int(preserved),
        "mismatched_edges": int(mismatched),
        "accepted": accepted,
    }


def _install_ephemeral_production_grounding() -> None:
    from v7.environment.encoding import (
        carrier_signature,
        grid_signature,
        structural_grid_signature,
        transformation_family_signature,
        transition_signature,
    )
    from v8 import learning_fixes_v088 as learning
    from v8 import learning_fixes_v088_target_grounding_fix as grounding

    current_capture = learning._capture_target_probe_state

    def capture_target_probe_state(*, game_id: str, env_root: str | None, seed: int):
        target_hash = int(world_id(game_id))
        template = grounding._PENDING_TARGET_TEMPLATES.pop(target_hash, None)
        references = grounding._PENDING_TARGET_STRUCTURES.pop(target_hash, ())
        if not isinstance(template, dict) or not references:
            return current_capture(game_id=game_id, env_root=env_root, seed=seed)

        # Obtain the exact base capture while preventing the older target-grounding
        # wrapper from consuming the pending template a second time.
        captured = current_capture(game_id=game_id, env_root=env_root, seed=seed)
        env = copy.deepcopy(captured.environment)
        rng = Random(int(seed) ^ 0x54A2)
        matched = None
        grounding_steps = 0
        attempts: list[dict[str, object]] = []
        carriers_by_role_group: dict[tuple[int, int], set[int]] = defaultdict(set)

        for step_index in range(_TARGET_GROUNDING_MAX_STEPS):
            before = env.observe()
            before_actions = tuple(sorted(set(int(value) for value in env.available_actions())))
            if not before_actions:
                env.reset()
                continue
            action = learning._memory_free_action(before_actions, rng)
            env.step(action)
            grounding_steps += 1
            after = env.observe()
            after_actions = tuple(sorted(set(int(value) for value in env.available_actions())))

            context = int(structural_grid_signature(before))
            next_context = int(structural_grid_signature(after))
            outcome = int(transition_signature(before, after))
            family_diagnostic = int(transformation_family_signature(before, after))
            raw_carrier = carrier_signature(before, after)
            future_delta = float(len(after_actions) - len(before_actions))
            identity = _production_ephemeral_identity(
                context_signature=context,
                action_id=int(action),
                outcome_signature=outcome,
                next_context_signature=next_context,
                future_option_delta=future_delta,
            )
            role_group = identity["role_group"]
            assert isinstance(role_group, tuple)
            carriers_by_role_group[role_group].add(int(identity["carrier_token"]))
            carrier_count = len(carriers_by_role_group[role_group])

            scored: list[dict[str, object]] = []
            if carrier_count >= 2:
                for reference_row in references:
                    reference = reference_row["descriptor"]
                    if int(reference.level) != int(MemoryLevel.M3):
                        continue
                    if int(reference.memory_type) not in {
                        int(MemoryType.ROLE),
                        int(MemoryType.CONTEXTUAL_ROLE),
                    }:
                        continue
                    target = _target_role_descriptor(
                        reference,
                        family_token=int(identity["family_token"]),
                        future_bucket=int(identity["future_bucket"]),
                        carrier_count=carrier_count,
                        step_index=step_index,
                    )
                    scored.append(_structural_match(reference_row, target, carrier_count))
            scored.sort(
                key=lambda row: (
                    not bool(row["accepted"]),
                    -float(row["similarity_score"]),
                    float(row["epsilon_struct"]),
                    str(row["reference_uid"]),
                )
            )
            best = scored[0] if scored else None
            attempts.append(
                {
                    "step_index": int(step_index),
                    "action_id": int(action),
                    "observed_target_transformation_family_signature": family_diagnostic,
                    "observed_target_carrier_signature": None if raw_carrier is None else int(raw_carrier),
                    "ephemeral_m1_key": [int(v) for v in identity["m1_key"]],
                    "ephemeral_m2_key": [int(v) for v in identity["m2_key"]],
                    "ephemeral_family_token": int(identity["family_token"]),
                    "ephemeral_carrier_token": int(identity["carrier_token"]),
                    "ephemeral_future_bucket": int(identity["future_bucket"]),
                    "distinct_carriers_for_target_family": int(carrier_count),
                    "best_structural_match": None if best is None else dict(best),
                }
            )
            if best is None or not bool(best["accepted"]):
                continue

            evidence_hash = stable_u64(
                str(game_id),
                int(seed),
                int(step_index),
                int(action),
                int(identity["family_token"]),
                int(identity["carrier_token"]),
                person=b"v8-xfer-ground",
            )
            matched = {
                **template,
                "derived_target_action": int(action),
                "observed_target_transformation_family_signature": family_diagnostic,
                "observed_target_carrier_signature": None if raw_carrier is None else int(raw_carrier),
                "target_ephemeral_m1_key": [int(v) for v in identity["m1_key"]],
                "target_ephemeral_m2_key": [int(v) for v in identity["m2_key"]],
                "target_ephemeral_family_token": int(identity["family_token"]),
                "target_ephemeral_carrier_token": int(identity["carrier_token"]),
                "target_ephemeral_future_bucket": int(identity["future_bucket"]),
                "target_interaction_evidence_id": f"{evidence_hash:016x}",
                "correspondence_conditioned_mapping": True,
                "grounding_step_index": int(step_index),
                "grounding_identity_rule": "production_m1_m2_m3_canonical_identity",
                "structural_match": dict(best),
            }
            break

        if matched is None:
            grounding._CAPTURE_GROUNDINGS[captured.capture_id] = {
                "evidence": None,
                "grounding_prefix_steps": int(grounding_steps),
                "grounding_policy": "shared_memory_free_prefix",
                "grounding_attempts": attempts,
                "failure_reason": "no_target_transition_passed_production_structural_gates",
            }
            return captured

        state = env.observe()
        actions = tuple(sorted(set(int(value) for value in env.available_actions())))
        signature = int(grid_signature(state))
        capture_hash = stable_u64(
            str(game_id),
            int(seed),
            signature,
            *actions,
            grounding_steps,
            person=b"v8-xfer-grounded-state",
        )
        capture_id = f"{capture_hash:016x}"
        grounding._CAPTURE_GROUNDINGS[capture_id] = {
            "evidence": matched,
            "grounding_prefix_steps": int(grounding_steps),
            "grounding_policy": "shared_memory_free_prefix",
            "grounding_attempts": attempts,
            "failure_reason": None,
        }
        return replace(
            captured,
            environment=env,
            capture_id=capture_id,
            initial_state_signature=signature,
            initial_available_actions=actions,
        )

    learning._capture_target_probe_state = capture_target_probe_state


def install_target_ephemeral_production_grounding_fix() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_ephemeral_production_grounding()
    _INSTALLED = True
