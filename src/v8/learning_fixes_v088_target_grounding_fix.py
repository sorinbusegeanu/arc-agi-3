from __future__ import annotations

import copy
from collections import Counter, defaultdict
from dataclasses import replace
from random import Random

from v8.model import MemoryLevel, MemoryType, MemoryUid, RelationType, stable_u64
from v8.persistent_identity import world_id


_INSTALLED = False
_TARGET_GROUNDING_MAX_STEPS = 8
_PENDING_TARGET_TEMPLATES: dict[int, dict[str, object]] = {}
_PENDING_TARGET_STRUCTURES: dict[int, tuple[dict[str, object], ...]] = {}
_CAPTURE_GROUNDINGS: dict[str, dict[str, object]] = {}


def _uid_text(uid: MemoryUid) -> str:
    return uid.hex()


def _structural_family(row) -> int | None:
    if row is None or int(row.level) not in {int(MemoryLevel.M3), int(MemoryLevel.M4)}:
        return None
    if not row.key_parts:
        return None
    return int(row.key_parts[0])


def _install_target_local_transfer_grounding() -> None:
    from v7.environment.encoding import (
        carrier_signature,
        grid_signature,
        transformation_family_signature,
    )
    from v8 import learning_fixes_v088 as learning
    from v8.similarity import BoundedNeighborhoodSimilarity, NeighborhoodDescriptor
    from v8.structural_correspondence import StructuralCorrespondenceEstimator

    current_transfer_execution = learning._transfer_execution_evidence
    current_capture = learning._capture_target_probe_state
    current_probe = learning._probe_policy_v088
    similarity = BoundedNeighborhoodSimilarity()
    correspondence = StructuralCorrespondenceEstimator()

    def target_descriptor(
        *,
        reference,
        family: int,
        carrier_count: int,
        step_index: int,
    ):
        level = int(reference.level)
        memory_type = int(reference.memory_type)
        if level == int(MemoryLevel.M3) and memory_type in {
            int(MemoryType.ROLE),
            int(MemoryType.CONTEXTUAL_ROLE),
        }:
            count = max(1, int(carrier_count))
            outgoing = ((int(RelationType.EXPLAINS), count),)
            neighbor_levels = ((int(MemoryLevel.M3), count),)
            neighbor_types = ((int(MemoryType.CARRIER), count),)
        elif level == int(MemoryLevel.M4):
            outgoing = ((int(RelationType.EXPLAINS), 1),)
            neighbor_levels = ((int(MemoryLevel.M3), 1),)
            neighbor_types = ((int(MemoryType.ROLE), 1),)
        else:
            return None
        uid = MemoryUid.from_key(
            level,
            memory_type,
            (int(family), int(step_index), int(carrier_count)),
        )
        return NeighborhoodDescriptor(
            uid=uid,
            level=level,
            memory_type=memory_type,
            incoming_relations=(),
            outgoing_relations=outgoing,
            neighbor_levels=neighbor_levels,
            neighbor_types=neighbor_types,
            dependency_signature=0,
            enable_block_signature=0,
            future_option_bucket=0,
            consequence_bucket=0,
            context_bucket=0,
            descriptor_version=int(step_index) + 1,
        )

    def target_structural_counter(descriptor) -> Counter[tuple[int, int, int, int]]:
        result: Counter[tuple[int, int, int, int]] = Counter()
        if int(descriptor.level) == int(MemoryLevel.M3):
            count = dict(descriptor.outgoing_relations).get(int(RelationType.EXPLAINS), 0)
            if count:
                result[(
                    1,
                    int(RelationType.EXPLAINS),
                    int(MemoryLevel.M3),
                    int(MemoryType.CARRIER),
                )] = int(count)
        elif int(descriptor.level) == int(MemoryLevel.M4):
            result[(
                1,
                int(RelationType.EXPLAINS),
                int(MemoryLevel.M3),
                int(MemoryType.ROLE),
            )] = 1
        return result

    def structural_match(reference_row: dict[str, object], target):
        reference = reference_row["descriptor"]
        evidence = similarity.score(reference, target)
        left = reference_row["structural_counter"]
        right = target_structural_counter(target)
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

    def transfer_execution_evidence(
        read_view,
        nodes,
        edges,
        candidate,
        source_trajectories,
        *,
        target_game_hash: int,
    ):
        result = current_transfer_execution(
            read_view,
            nodes,
            edges,
            candidate,
            source_trajectories,
            target_game_hash=target_game_hash,
        )
        if result.get("correspondence_conditioned_mapping") is True:
            return result

        by_uid = getattr(read_view, "_node_by_uid", None)
        if not isinstance(by_uid, dict):
            by_uid = {row.uid: row for row in nodes}
        source = by_uid.get(candidate.uid)
        correspondence_row = by_uid.get(candidate.correspondence_uid)
        if source is None or correspondence_row is None or candidate.correspondence_uid.is_zero:
            return result

        graph_edges = tuple(edges or ())
        descriptors = BoundedNeighborhoodSimilarity.descriptors(nodes, graph_edges)
        reference_structures = []
        for row in (source, correspondence_row):
            descriptor = descriptors.get(row.uid)
            if descriptor is None:
                continue
            reference_structures.append(
                {
                    "uid": _uid_text(row.uid),
                    "descriptor": descriptor,
                    "structural_counter": StructuralCorrespondenceEstimator._descriptor(
                        row.uid, graph_edges, by_uid
                    ),
                }
            )
        if not reference_structures:
            return result

        families = tuple(
            sorted(
                {
                    value
                    for value in (
                        _structural_family(source),
                        _structural_family(correspondence_row),
                    )
                    if value is not None
                }
            )
        )
        template = {
            "source_structural_memory_uid": _uid_text(candidate.uid),
            "correspondence_uid": _uid_text(candidate.correspondence_uid),
            "source_role_entity": {
                "memory_uid": _uid_text(candidate.uid),
                "memory_level": int(source.level),
                "structural_key": [int(v) for v in source.key_parts],
            },
            "target_role_entity": {
                "correspondence_anchor_uid": _uid_text(candidate.correspondence_uid),
                "memory_level": int(correspondence_row.level),
                "structural_key": [int(v) for v in correspondence_row.key_parts],
            },
            "known_transformation_family_signatures": list(families),
            "mapping_kind": "explicit_structural_role_to_target_interaction_grounding",
            "similarity_threshold": float(similarity.threshold),
            "theta_struct": float(correspondence.theta_struct),
            "grounding_rule": "production_radius1_similarity_plus_structural_error_gate",
        }
        _PENDING_TARGET_TEMPLATES[int(target_game_hash)] = template
        _PENDING_TARGET_STRUCTURES[int(target_game_hash)] = tuple(reference_structures)
        result = dict(result)
        result["target_grounding_template"] = template
        result["resolution_status"] = "pending_target_local_structural_grounding"
        result["failure_reason"] = "pending_target_local_structural_grounding"
        result["lower_level_resolution_failures"] = [
            "pending_target_local_structural_grounding"
        ]
        return result

    def capture_target_probe_state(*, game_id: str, env_root: str | None, seed: int):
        captured = current_capture(game_id=game_id, env_root=env_root, seed=seed)
        target_hash = int(world_id(game_id))
        template = _PENDING_TARGET_TEMPLATES.pop(target_hash, None)
        references = _PENDING_TARGET_STRUCTURES.pop(target_hash, ())
        if not isinstance(template, dict) or not references:
            return captured

        env = copy.deepcopy(captured.environment)
        rng = Random(int(seed) ^ 0x54A2)
        matched = None
        matched_environment = None
        grounding_steps = 0
        attempts: list[dict[str, object]] = []
        carriers_by_family: dict[int, set[int]] = defaultdict(set)
        for step_index in range(_TARGET_GROUNDING_MAX_STEPS):
            before = env.observe()
            actions = tuple(sorted(set(int(value) for value in env.available_actions())))
            if not actions:
                env.reset()
                continue
            pre_action_environment = copy.deepcopy(env)
            action = learning._memory_free_action(actions, rng)
            env.step(action)
            grounding_steps += 1
            after = env.observe()
            family = int(transformation_family_signature(before, after))
            carrier = carrier_signature(before, after)
            if carrier is not None:
                carriers_by_family[family].add(int(carrier))
            carrier_count = len(carriers_by_family.get(family, ()))

            scored: list[dict[str, object]] = []
            for reference_row in references:
                reference = reference_row["descriptor"]
                if (
                    int(reference.level) == int(MemoryLevel.M3)
                    and carrier_count < 2
                ):
                    continue
                target = target_descriptor(
                    reference=reference,
                    family=family,
                    carrier_count=carrier_count,
                    step_index=step_index,
                )
                if target is None:
                    continue
                scored.append(structural_match(reference_row, target))
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
                    "observed_target_transformation_family_signature": int(family),
                    "observed_target_carrier_signature": (
                        None if carrier is None else int(carrier)
                    ),
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
                int(family),
                person=b"v8-xfer-ground",
            )
            matched = {
                **template,
                "derived_target_action": int(action),
                "observed_target_transformation_family_signature": int(family),
                "observed_target_carrier_signature": (
                    None if carrier is None else int(carrier)
                ),
                "target_interaction_evidence_id": f"{evidence_hash:016x}",
                "target_grounding_context_signature": int(grid_signature(before)),
                "target_grounding_next_context_signature": int(grid_signature(after)),
                "correspondence_conditioned_mapping": True,
                "grounding_step_index": int(step_index),
                "structural_match": dict(best),
            }
            matched_environment = pre_action_environment
            break

        if matched is None or matched_environment is None:
            _CAPTURE_GROUNDINGS[captured.capture_id] = {
                "evidence": None,
                "grounding_prefix_steps": int(grounding_steps),
                "grounding_policy": "shared_memory_free_prefix",
                "grounding_attempts": attempts,
                "failure_reason": "no_target_transition_passed_production_structural_gates",
            }
            return captured

        state = matched_environment.observe()
        actions = tuple(
            sorted(set(int(value) for value in matched_environment.available_actions()))
        )
        signature = int(grid_signature(state))
        capture_hash = stable_u64(
            str(game_id),
            int(seed),
            signature,
            *actions,
            max(0, grounding_steps - 1),
            person=b"v8-xfer-grounded-state",
        )
        capture_id = f"{capture_hash:016x}"
        _CAPTURE_GROUNDINGS[capture_id] = {
            "evidence": matched,
            "grounding_prefix_steps": max(0, int(grounding_steps) - 1),
            "grounding_policy": "shared_memory_free_prefix",
            "grounding_attempts": attempts,
            "failure_reason": None,
        }
        return replace(
            captured,
            environment=matched_environment,
            capture_id=capture_id,
            initial_state_signature=signature,
            initial_available_actions=actions,
        )

    def mapped_evidence_action(target_action_evidence, available_actions, cursor):
        valid = tuple(
            row
            for row in target_action_evidence
            if isinstance(row, dict)
            and row.get("correspondence_conditioned_mapping") is True
            and row.get("source_structural_memory_uid")
            and row.get("correspondence_uid")
            and isinstance(row.get("source_role_entity"), dict)
            and isinstance(row.get("target_role_entity"), dict)
            and row.get("mapping_kind")
            and (
                row.get("target_grounded_memory_uid")
                or row.get("target_executable_structure_uid")
                or row.get("target_interaction_evidence_id")
            )
            and row.get("derived_target_action") is not None
        )
        if not valid:
            return None, int(cursor), None
        available = {int(value) for value in available_actions}
        start = int(cursor) % len(valid)
        for offset in range(len(valid)):
            index = (start + offset) % len(valid)
            row = valid[index]
            action = int(row["derived_target_action"])
            if action in available:
                return action, (index + 1) % len(valid), dict(row)
        return None, start, None

    def probe_policy_v088(**kwargs):
        execution = kwargs.get("execution_evidence")
        capture_id = kwargs.get("target_state_capture_id")
        required_ancestor = kwargs.get("required_ancestor")
        grounding = _CAPTURE_GROUNDINGS.get(str(capture_id))
        if (
            required_ancestor is not None
            and isinstance(execution, dict)
            and isinstance(grounding, dict)
            and isinstance(grounding.get("evidence"), dict)
        ):
            evidence = dict(grounding["evidence"])
            execution["kind"] = "target_conditioned_structural_action_mapping"
            execution["action_ids"] = [int(evidence["derived_target_action"])]
            execution["mapped_action_evidence"] = [evidence]
            execution["target_action_evidence"] = [evidence]
            execution["action_sequence_candidates"] = [
                {
                    "kind": "target_conditioned_structural_action_mapping",
                    "action_ids": [int(evidence["derived_target_action"])],
                    "target_action_evidence": [evidence],
                    "correspondence_conditioned_mapping": True,
                    "grounding_prefix_steps": int(
                        grounding.get("grounding_prefix_steps", 0)
                    ),
                    "grounding_policy": grounding.get("grounding_policy"),
                }
            ]
            execution["correspondence_conditioned_mapping"] = True
            execution["resolution_status"] = "target_local_structural_grounding_resolved"
            execution["failure_reason"] = None
            execution["lower_level_resolution_failures"] = []
        metric, used = current_probe(**kwargs)
        diagnostic = kwargs.get("diagnostic")
        if isinstance(diagnostic, dict) and isinstance(grounding, dict):
            diagnostic["target_grounding_prefix_steps"] = int(
                grounding.get("grounding_prefix_steps", 0)
            )
            diagnostic["target_grounding_policy"] = grounding.get("grounding_policy")
            diagnostic["target_grounding_attempts"] = list(
                grounding.get("grounding_attempts", ())
            )[:_TARGET_GROUNDING_MAX_STEPS]
            diagnostic["target_grounding_failure_reason"] = grounding.get(
                "failure_reason"
            )
            evidence = grounding.get("evidence")
            diagnostic["target_interaction_grounding_evidence"] = (
                dict(evidence) if isinstance(evidence, dict) else None
            )
        return metric, used

    learning._transfer_execution_evidence = transfer_execution_evidence
    learning._capture_target_probe_state = capture_target_probe_state
    learning._mapped_evidence_action = mapped_evidence_action
    learning._probe_policy_v088 = probe_policy_v088


def install_target_local_transfer_grounding_fix() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_target_local_transfer_grounding()
    _INSTALLED = True
