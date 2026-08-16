from __future__ import annotations

from v8.model import stable_u64
from v8.normalized_memory_v086 import stage_worker_v086
from v8.structural_events import native_action_set_signature


_INSTALLED = False
_CURRENT_ACTION_SET_SIGNATURE = 0


def install_normalized_memory_v086_fixups() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from v7.environment import arc_adapter as adapter
    from v7.environment import encoding
    from v8 import development
    from v8 import runtime

    base_available = adapter.ArcGridEnvironment.available_actions
    base_structural = encoding.structural_grid_signature

    def available_actions(self):
        global _CURRENT_ACTION_SET_SIGNATURE
        values = list(base_available(self))
        _CURRENT_ACTION_SET_SIGNATURE = native_action_set_signature(values)
        return values

    def structural_grid_signature(grid):
        return stable_u64(
            int(base_structural(grid)),
            int(_CURRENT_ACTION_SET_SIGNATURE),
            person=b"v8.6-action-context",
        )

    adapter.ArcGridEnvironment.available_actions = available_actions
    encoding.structural_grid_signature = structural_grid_signature

    # Runtime already points at the v8.6 worker. Keep the development entry point
    # consistent for direct tests/tools and future runtime constructors.
    development.stage_worker = stage_worker_v086
    runtime.stage_worker = stage_worker_v086

    _INSTALLED = True
