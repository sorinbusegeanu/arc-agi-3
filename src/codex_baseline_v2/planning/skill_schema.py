from __future__ import annotations

SUPPORTED_SKILL_TYPES = {
    "go_to_region",
    "contact_poi",
    "probe_hidden_trigger",
    "cross_transition",
    "dwell_on_region",
    "perform_action_at_region",
    "verify_mechanic",
    "return_to_anchor",
}

# Optional short-term fields carried on SkillSpecV1 records for same-round planning.
CURRENT_ROUND_MEMORY_FIELDS = {
    "executions_this_round",
    "latest_termination_reason_this_round",
    "repeated_contact_no_effect_count_this_round",
    "target_x_this_round",
    "target_y_this_round",
}
