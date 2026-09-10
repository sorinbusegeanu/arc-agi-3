from __future__ import annotations


_INSTALLED = False


def _install_target_mapping_lifetime() -> None:
    from v8 import learning_fixes_v088 as learning

    def mapped_evidence_action(target_action_evidence, available_actions, cursor):
        """Consume each grounded target-local action at most once.

        A target-local mapping is evidence for the exact pre-action state captured
        during grounding. It is not a stationary policy and must not cycle across
        later target states.
        """
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
                (
                    row.get("target_interaction_evidence_id")
                    and row.get("target_grounding_context_signature") is not None
                )
                # A persisted target-local grounded contingency is already a
                # structural target observation; it need not masquerade as the
                # ephemeral exact-state grounding form.
                or row.get("target_grounded_memory_uid")
            )
            and row.get("derived_target_action") is not None
        )
        position = max(0, int(cursor))
        if not valid or position >= len(valid):
            return None, position, None

        available = {int(value) for value in available_actions}
        for index in range(position, len(valid)):
            row = valid[index]
            action = int(row["derived_target_action"])
            next_cursor = index + 1
            if action in available:
                return action, next_cursor, dict(row)
        return None, len(valid), None

    learning._mapped_evidence_action = mapped_evidence_action


def install_target_mapping_lifetime_fix() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_target_mapping_lifetime()
    _INSTALLED = True
