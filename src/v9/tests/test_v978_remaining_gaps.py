from __future__ import annotations

from v9.memory.relations import RelationType
from v9.research.grounding_h16 import (
    evaluate_h16,
    load_h16_trials,
    run_synthetic_h16_controls,
    save_h16_trials,
)


def test_symbolic_relation_vocabulary_is_explicit() -> None:
    assert RelationType.SYMBOL_OCCURRENCE.value == "SYMBOL_OCCURRENCE"
    assert RelationType.SYMBOL_PRECEDES_SYMBOL.value == "SYMBOL_PRECEDES_SYMBOL"
    assert RelationType.SYMBOL_FOLLOWS_SYMBOL.value == "SYMBOL_FOLLOWS_SYMBOL"
    assert RelationType.CROSS_MODAL_CORRESPONDENCE.value == "CROSS_MODAL_CORRESPONDENCE"


def test_h16_saved_trials_are_reproducible(tmp_path) -> None:
    trials = run_synthetic_h16_controls(
        seeds=(3, 5),
        environment_config_id=11,
        interaction_budget=24,
        evaluation_id=7,
    )
    path = tmp_path / "h16-trials.json"
    save_h16_trials(path, trials)
    restored = load_h16_trials(path)
    assert restored == trials
    before = evaluate_h16(trials)
    after = evaluate_h16(restored)
    assert after.means == before.means
    assert after.metric_means == before.metric_means
    assert after.causal_effect == before.causal_effect
    assert after.matched == before.matched
