from __future__ import annotations

import copy
from dataclasses import replace
from random import Random

from v8.model import MemoryLevel, MemoryUid, stable_u64
from v8.persistent_identity import world_id


_INSTALLED = False
_TARGET_GROUNDING_MAX_STEPS = 8
_PENDING_TARGET_TEMPLATES: dict[int, dict[str, object]] = {}
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
    from v7.environment.encoding import grid_signature, transformation_family_signature
    from v8 import learning_fixes_v088 as learning

    current_transfer_execution = learning._transfer_execution_evidence
    current_capture = learning._capture_target_probe_state
    current_probe = learning._probe_policy_v088

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
        correspondence = by_uid.get(candidate.correspondence_uid)
        families = tuple(
            sorted(
                {
                    value
                    for value in (
                        _structural_family(source),
                        _structural_family(correspondence),
                    )
                    if value is not None
                }
            )
        )
        if not families or candidate.correspondence_uid.is_zero:
            return result

        template = {
            "source_structural_memory_uid": _uid_text(candidate.uid),
            "correspondence_uid": _uid_text(candidate.correspondence_uid),
            "source_role_entity": {
                "memory_uid": _uid_text(candidate.uid),
                "memory_level": None if source is None else int(source.level),
                "structural_key": [] if source is None else [int(v) for v in source.key_parts],
            },
            "target_role_entity": {
                "correspondence_anchor_uid": _uid_text(candidate.correspondence_uid),
                "memory_level": None if correspondence is None else int(correspondence.level),
                "structural_key": [] if correspondence is None else [int(v) for v in correspondence.key_parts],
                "expected_transformation_family_signatures": list(families),
            },
            "expected_transformation_family_signatures": list(families),
            "mapping_kind": "explicit_structural_role_to_target_interaction_grounding",
        }
        _PENDING_TARGET_TEMPLATES[int(target_game_hash)] = template
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
        template = _PENDING_TARGET_TEMPLATES.pop(int(world_id(game_id)), None)
        if not isinstance(template, dict):
            return captured

        expected = {
            int(value)
            for value in template.get("expected_transformation_family_signatures", ())
        }
        if not expected:
            return captured

        env = copy.deepcopy(captured.environment)
        rng = Random(int(seed) ^ 0x54A2)
        matched = None
        grounding_steps = 0
        for step_index in range(_TARGET_GROUNDING_MAX_STEPS):
            before = env.observe()
            actions = tuple(sorted(set(int(value) for value in env.available_actions())))
            if not actions:
                env.reset()
                continue
            action = learning._memory_free_action(actions, rng)
            env.step(action)
            grounding_steps += 1
            after = env.observe()
            family = int(transformation_family_signature(before, after))
            if family not in expected:
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
                "target_interaction_evidence_id": f"{evidence_hash:016x}",
                "correspondence_conditioned_mapping": True,
                "grounding_step_index": int(step_index),
            }
            break

        if matched is None:
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
        _CAPTURE_GROUNDINGS[capture_id] = {
            "evidence": matched,
            "grounding_prefix_steps": int(grounding_steps),
            "grounding_policy": "shared_memory_free_prefix",
        }
        return replace(
            captured,
            environment=env,
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
