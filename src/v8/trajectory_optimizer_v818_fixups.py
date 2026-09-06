from __future__ import annotations


_INSTALLED = False


def install_trajectory_optimizer_v818_fixups() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from v8 import trajectory_optimizer_v814 as optimizer
    from v8 import trajectory_optimizer_v818 as v818
    from v8.trajectory_validation_v814 import validate_arc_candidate

    base_game_validate = v818._GameReplayValidator.validate

    def game_validate(self, candidate):
        validator = getattr(self.service, "validator", None)
        if validator is not None and validator is not validate_arc_candidate:
            raw = validator(candidate)
            success = bool(getattr(raw, "success", False))
            attempts = max(1, int(getattr(raw, "attempts", 1)))
            successes = int(getattr(raw, "successes", int(success)))
            return v818.V818ValidationResult(
                success,
                int(getattr(raw, "actions_executed", len(candidate.actions))),
                str(getattr(raw, "reason", "target_preserved" if success else "target_not_reached")),
                str(getattr(raw, "terminal_state", "")),
                int(getattr(raw, "levels_completed", 0)),
                attempts,
                successes,
                tuple(self.service._v818_prefix_for(candidate)),
                int(getattr(raw, "terminal_context", 0)),
                int(getattr(raw, "terminal_action", 0)),
                int(getattr(raw, "outcome_signature", 0)),
            )
        return base_game_validate(self, candidate)

    def select_variant(rows, *, source_id, seed=None, action_history=(), attempted=None):
        # Seed is execution-only in v8.18. Exact action-prefix replay remains a
        # valid compatibility path even when an older validated row has no M6 UID;
        # the broader target-compatible path still requires a resolved M6 target.
        del seed
        history = tuple(int(value) for value in action_history)
        blocked = attempted or set()
        candidates = [
            row
            for row in rows
            if row.variant_id not in blocked
            and row.anchor.source_id == str(source_id)
            and tuple(row.anchor.prefix_actions) == history
            and row.actions
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda row: (-row.saved_actions, row.cost, -row.successes, row.variant_id),
        )

    def reset_capture(job=None) -> None:
        # Actor seeds remain available to the actor/environment RNG, but the
        # trajectory capture channel itself has no seed identity in v8.18.
        optimizer._CAPTURE_ACTIVE = job is not None
        optimizer._CAPTURE_SOURCE_ID = "" if job is None else str(job.game_id)
        optimizer._CAPTURE_SEED = 0
        optimizer._CAPTURE_ENV_ROOT = None if job is None else job.env_root
        optimizer._CAPTURE_PREFIX = []
        optimizer._CAPTURE_SEGMENT = []
        optimizer._ACTOR_ACTION_HISTORY = []
        optimizer._ACTOR_RESET_EPOCH += 1

    v818._GameReplayValidator.validate = game_validate
    optimizer.select_validated_variant = select_variant
    optimizer._reset_capture = reset_capture
    _INSTALLED = True
