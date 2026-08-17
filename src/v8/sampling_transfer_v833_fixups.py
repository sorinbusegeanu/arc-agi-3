from __future__ import annotations

"""v8.33 compatibility wiring beneath the v8.32 public sampler authority.

v8.32 owns the public begin/reset/forced/observe methods. v8.33 extends the lower
delegates used by those methods, while keeping only discovery_action as the new
public entry point because v8.32 never owned that method.
"""

_INSTALLED = False


def install_sampling_transfer_v833_fixups() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from v8 import sampling_persistence_v832 as persistence
    from v8 import sampling_portfolio_v831 as portfolio
    from v8 import sampling_transfer_v833 as transfer

    cls = portfolio.PortfolioSampler

    # Capture the v8.31 delegates that v8.32 wrapped before v8.33 was installed.
    base_begin = persistence._BASE_BEGIN_LEASE
    base_reset = persistence._BASE_ON_EXTERNAL_RESET
    base_forced = persistence._BASE_FORCED_ACTION
    base_observe = persistence._BASE_OBSERVE_TRANSITION

    # Retarget the v8.33 wrappers to those lower delegates, then compose them
    # underneath v8.32. This preserves both semantics and historical identities.
    transfer._BASE_BEGIN_LEASE = base_begin
    transfer._BASE_ON_EXTERNAL_RESET = base_reset
    transfer._BASE_FORCED_ACTION = base_forced
    transfer._BASE_OBSERVE_TRANSITION = base_observe

    persistence._BASE_BEGIN_LEASE = transfer._begin_lease_v833
    persistence._BASE_ON_EXTERNAL_RESET = transfer._on_external_reset_v833
    persistence._BASE_FORCED_ACTION = transfer._forced_action_v833
    persistence._BASE_OBSERVE_TRANSITION = transfer._observe_transition_v833

    cls.begin_lease = persistence._begin_lease_v832
    cls.on_external_reset = persistence._on_external_reset_v832
    cls.forced_action = persistence._forced_action_v832
    cls.observe_transition = persistence._observe_transition_v832

    # v8.32 did not override discovery_action, so v8.33 remains its authority.
    cls.discovery_action = transfer._discovery_action_v833

    _INSTALLED = True
