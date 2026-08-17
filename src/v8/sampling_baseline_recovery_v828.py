from __future__ import annotations

"""v8.28 restore the last known-good v8.19 discovery runtime.

PR #99 was the last main revision observed solving games with long first-win
trajectories. v8.21 later replaced the no-plan actor path with decision-point
reset/replay probing. Later repairs restored planner-first control and long leases,
but cold-start DISCOVERY still fell back into the v8.21 sampler.

Keep all later memory, optimizer, trajectory, transfer, reporting and lifecycle
layers while restoring two PR #99 runtime properties:

* ordinary DISCOVERY uses the pre-v8.21 actor exploration path;
* --actors remains a minimum lane request, so a game set still gets at least one
  concurrent worker per game by default.
"""

import argparse
import os


_INSTALLED = False
_BASE_ACTOR_DELEGATE = None


def _requested_actor_pool_v828(values: list[str]) -> int:
    """Restore pre-v8.23 actor semantics without validating the game registry early."""

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

    from v7.game_sets import V7_GAME_PRESETS

    selector = str(parsed.games).strip()
    if selector in V7_GAME_PRESETS:
        game_count = len(tuple(dict.fromkeys(V7_GAME_PRESETS[selector])))
    elif selector == "all":
        try:
            from v7.environment.arc_adapter import registered_game_ids

            game_count = len(
                tuple(game for game in registered_game_ids(parsed.env_root) if game != "gc01")
            )
        except BaseException:
            game_count = 0
    else:
        game_count = len(
            tuple(dict.fromkeys(item.strip() for item in selector.split(",") if item.strip()))
        )
    return max(requested, int(game_count))


def _actor_delegate_v828(*, job, **kwargs):
    """Route cold-start DISCOVERY to PR #99; retain later modes unchanged."""

    from v8 import decision_point_sampling_v821 as sampling

    mode = os.environ.get(sampling._SAMPLING_MODE_ENV, "DISCOVERY").strip().upper()
    if mode in {"", "DISCOVERY"}:
        # v8.21 captured this function before replacing actor_worker. It is the
        # continuous planner/unseen-action exploration chain used by PR #99: no
        # decision-point reset/replay controller and no artificial shallow anchor.
        return sampling._BASE_ACTOR_WORKER(job=job, **kwargs)
    return _BASE_ACTOR_DELEGATE(job=job, **kwargs)


def install_sampling_baseline_recovery_v828() -> None:
    global _INSTALLED, _BASE_ACTOR_DELEGATE
    if _INSTALLED:
        return

    from v8 import cli_v819
    from v8 import progress_runtime_fix_v822 as progress

    # Keep the v8.22 solve-metric wrapper as the public/final actor authority.
    # Change only what that wrapper calls underneath for ordinary DISCOVERY.
    _BASE_ACTOR_DELEGATE = progress._BASE_ACTOR_WORKER
    progress._BASE_ACTOR_WORKER = _actor_delegate_v828
    cli_v819._requested_actor_pool = _requested_actor_pool_v828
    _INSTALLED = True
