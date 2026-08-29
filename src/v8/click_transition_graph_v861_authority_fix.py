from __future__ import annotations

"""Place v8.61 beneath the historically pinned v8.60 sampler hooks."""

_INSTALLED = False


def install_click_transition_graph_v861_authority_fix() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from v8 import click_transition_exploration_v860 as v860
    from v8 import click_transition_graph_v861 as v861
    from v8 import sampling_evidence_frontier_v847_fixups as frontier_fixups

    # v8.61 initially installs at the deep frontier seam. Rewire the same delegates
    # one level lower so the established frontier -> v8.60 identity remains exact.
    old_forced = v861._BASE_FORCED_ACTION
    if old_forced is v860._forced_action_v860:
        old_forced = v860._BASE_FORCED_ACTION
    v861._BASE_FORCED_ACTION = old_forced
    v860._BASE_FORCED_ACTION = v861._forced_action_v861
    frontier_fixups._BASE_LOWER_FORCED = v860._forced_action_v860

    old_observe = v861._BASE_OBSERVE_TRANSITION
    if old_observe is v860._observe_transition_v860:
        old_observe = v860._BASE_OBSERVE_TRANSITION
    v861._BASE_OBSERVE_TRANSITION = old_observe
    v860._BASE_OBSERVE_TRANSITION = v861._observe_transition_v861
    frontier_fixups._BASE_LOWER_OBSERVE = v860._observe_transition_v860
    _INSTALLED = True
