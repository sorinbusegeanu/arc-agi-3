from __future__ import annotations

"""v8.28 restore the last known-good v8.19 discovery runtime.

PR #99 was the last main revision observed solving games with long first-win
trajectories. v8.21 later replaced the no-plan actor path with decision-point
reset/replay probing. Later repairs restored planner-first control and long leases,
but cold-start DISCOVERY still fell back into the v8.21 sampler.

Keep all later memory, optimizer, trajectory, transfer and lifecycle layers, while
restoring two runtime properties from PR #99:

* ordinary DISCOVERY uses the pre-v8.21 actor exploration path;
* --actors remains a minimum lane request, so a game set still gets at least one
  concurrent worker per game by default.
"""

import argparse


_INSTALLED = False


def _decision_mode_enabled_v828() -> bool:
    # v8.21's custom decision-point actor is no longer the final DISCOVERY policy.
    # Its classes remain available for experiments/tests, but production actor
    # dispatch falls through to the pre-v8.21 worker captured by that layer.
    return False


def _requested_actor_pool_v828(values: list[str]) -> int:
    """Restore pre-v8.23 actor semantics: at least one lane per selected game."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--actors", type=int, default=8)
    parser.add_argument("--games", default=None)
    parser.add_argument("--env-root", default=None)
    parsed, _unknown = parser.parse_known_args(values)
    requested = int(parsed.actors)
    if requested <= 0:
        raise ValueError("--actors must be positive")
    if not parsed.games:
        return requested

    from v7.game_sets import resolve_game_selector

    games = tuple(resolve_game_selector(parsed.games, parsed.env_root))
    return max(requested, len(games))


def install_sampling_baseline_recovery_v828() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from v8 import cli_v819
    from v8 import decision_point_sampling_v821 as sampling

    sampling._decision_mode_enabled = _decision_mode_enabled_v828
    cli_v819._requested_actor_pool = _requested_actor_pool_v828
    _INSTALLED = True
