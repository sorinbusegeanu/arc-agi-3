from __future__ import annotations

import os


LIFECYCLE_ENV = "ARC_AGI3_V8_LIFECYCLE"
_INSTALLED = False
_BASE_SUPERVISOR_START = None


def lifecycle_enabled() -> bool:
    return str(os.environ.get(LIFECYCLE_ENV, "on")).strip().lower() != "off"


def _supervisor_start_v827(self) -> None:
    if lifecycle_enabled():
        return _BASE_SUPERVISOR_START(self)

    # Start the developmental peer thread directly, bypassing only v8.13's
    # dedicated lifecycle-thread wrapper. Prediction/promotion/replay peers remain on.
    from v8.peers import DevelopmentalPeerSupervisor

    return DevelopmentalPeerSupervisor.start(self)


def install_lifecycle_switch_v827() -> None:
    global _INSTALLED, _BASE_SUPERVISOR_START
    if _INSTALLED:
        return

    from v8.peers_v82 import V82DevelopmentalPeerSupervisor

    _BASE_SUPERVISOR_START = V82DevelopmentalPeerSupervisor.start
    V82DevelopmentalPeerSupervisor.start = _supervisor_start_v827
    _INSTALLED = True

    from v8.lifecycle_competence_integration_v827_fixups import (
        install_lifecycle_competence_integration_v827_fixups,
    )

    install_lifecycle_competence_integration_v827_fixups()
