from __future__ import annotations

"""v8.26 repair for planner authority and unsolved-episode continuity.

v8.24 made decision-point discovery authoritative before the learned planner and
bounded every unsolved lease to 2048 actions.  The first change could suppress an
already executable learned policy; the second made a scheduler quantum an ARC
episode boundary because each lease invokes a fresh actor/environment.

Restore the v8.17/v8.21 control contract without removing newer trajectory,
adaptive-allocation, transfer, lifecycle, or reporting layers:

* REPLAY/VERIFICATION remain forced interventions.
* The composed learned planner gets first authority for ordinary discovery.
* Decision-point discovery runs only when the planner returns no executable plan.
* An unsolved game receives its remaining per-game sampling budget in one actor
  lease.  A WIN may still end the lease early, releasing the unused credits for
  adaptive reallocation.  This keeps one unsolved ARC episode, trajectory capture,
  decision sampler, local overlay, and actor-local strategy state continuous.
"""


_INSTALLED = False
_BASE_PLAN_CANDIDATES = None


def episode_aligned_unsolved_lease_steps_v826(
    *,
    available: int,
    base_steps: int,
    initial_probe: bool,
    worker_count: int,
    game_count: int,
) -> int:
    """Do not turn the allocator's short quantum into an environment reset.

    The adaptive worker already stops a lease immediately on a real WIN and returns
    the unused reservation to the global budget.  For a still-unsolved game, keep
    the actor/environment alive for the full remaining per-game budget so long
    first solutions can cross the former 2048-step boundary.
    """

    del initial_probe, worker_count, game_count
    return min(max(1, int(available)), max(1, int(base_steps)))


def _plan_candidates_v826(self, context_signature, action_ids, **kwargs):
    """Give learned control first authority without weakening v8.24 transfer gates.

    The captured v8.24 planner remains the semantic authority because it also
    enforces foreign provenance in TRANSFER mode.  Suppress only the transient
    v8.22 `before_plan` discovery flag while invoking it.  If no plan is returned,
    restore the flag so the v8.21 actor naturally falls through to decision-point
    discovery.
    """

    from v8 import runtime_repair_v822 as v822

    prior_probe = bool(getattr(v822._PROBE_STATE, "before_plan", False))
    if prior_probe:
        v822._PROBE_STATE.before_plan = False
    try:
        return _BASE_PLAN_CANDIDATES(self, context_signature, action_ids, **kwargs)
    finally:
        v822._PROBE_STATE.before_plan = prior_probe


def install_learning_control_continuity_v826() -> None:
    global _INSTALLED, _BASE_PLAN_CANDIDATES
    if _INSTALLED:
        return

    from v8 import adaptive_learning_allocation_v819_performance_fix as perf
    from v8 import runtime_repair_v822 as v822
    from v8 import sampling_control_repair_v823 as v823
    from v8.publication import LiveReadView

    # Preserve adaptive allocation and its early-WIN credit release, but remove the
    # hard 2048-action cutoff for final composed runtime behavior.  Keep the v8.24
    # helper itself unchanged so historical-layer tests continue to describe v8.24.
    perf.__dict__["_v823_initial_unsolved_lease_steps"] = (
        episode_aligned_unsolved_lease_steps_v826
    )

    # Capture the final v8.24 planner chain (including TRANSFER provenance rules),
    # then make planner-first behavior the final composed control authority.  Keep
    # the v8.24 function object intact and wrap it rather than rewriting history.
    _BASE_PLAN_CANDIDATES = LiveReadView.plan_candidates
    v822._BASE_PLAN_CANDIDATES = _plan_candidates_v826
    LiveReadView.plan_candidates = _plan_candidates_v826

    # Verification is a bounded intervention, not ordinary discovery.  Keep the
    # v8.23 one-repeat contract explicit so verification may force an action while
    # discovery never suppresses a usable learned plan.
    from v8 import decision_point_sampling_v821 as sampling

    sampling._VERIFICATION_REPEATS = 1

    assert v823._INSTALLED
    _INSTALLED = True
