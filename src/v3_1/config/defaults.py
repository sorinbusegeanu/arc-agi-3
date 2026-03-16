from __future__ import annotations

from v3_1.config.schema import ExecutionSection, StorageSection, V31Config
from v3_1.config.hypothesis_generation import HypothesisGenerationSection


DEFAULT_CONFIG = V31Config(
    planning=type(V31Config().planning)(
        **{
            **V31Config().planning.__dict__,
            "probe_branch_count": 1,
            "directed_trial_count": 1,
        }
    ),
    execution=ExecutionSection(
        move_terminal_distance_cells=0,
        interact_terminal_distance_cells=0,
        click_terminal_distance_cells=0,
    ),
    storage=StorageSection(
        max_blackboard_consequences=100,
        enable_persistent_memory=True,
        persistent_memory_db_path_override=None,
        persistent_memory_flush_every_n_rounds=0,
        load_persistent_priors_on_session_start=True,
        persist_skill_stats=True,
        persist_candidate_outcomes=True,
        persist_failure_patterns=True,
        persist_recovery_patterns=True,
        persist_poi_patterns=True,
        persist_trigger_patterns=True,
        persist_consequence_patterns=True,
        persist_entity_signatures=True,
        persist_area_signatures=True,
        persist_mechanic_hypotheses=True,
        persist_ranker_state=True,
    ),
    hypothesis_generation=HypothesisGenerationSection(),
)
