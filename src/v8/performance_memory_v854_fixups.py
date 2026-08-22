from __future__ import annotations

"""Compatibility fixes for the v8.54 performance layer."""

import os
import time
from pathlib import Path


_INSTALLED = False
_VARIANT_REFRESH_SECONDS = 1.0


def _file_signature(path: Path) -> tuple[int | None, int | None]:
    try:
        stat = path.stat()
        return int(stat.st_mtime_ns), int(stat.st_size)
    except OSError:
        return None, None


def _variant_input_signature() -> tuple[object, ...]:
    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import trajectory_optimizer_v814 as optimizer

    root_raw = str(os.environ.get(optimizer._TRAJECTORY_ROOT_ENV, "")).strip()
    source_id = str(getattr(optimizer, "_CAPTURE_SOURCE_ID", ""))
    mode = str(
        os.environ.get(
            v819._SAMPLING_MODE_ENV,
            v819.SamplingMode.DISCOVERY.value,
        )
    )
    if not root_raw:
        return source_id, mode, None, None, None, None
    root = Path(root_raw)
    validated = _file_signature(root / "validated.json")
    best = _file_signature(root / "best_successful.json")
    return source_id, mode, *validated, *best


def _refresh_view_variants_v854_fixup(view) -> None:
    """Cache around the composed v8.44+ loader without bypassing VERIFY semantics."""
    from v8 import performance_memory_v854 as v854

    now = time.monotonic()
    if now < float(getattr(view, "_v814_next_refresh", 0.0)):
        return
    signature = _variant_input_signature()
    if signature == getattr(view, "_v854_variant_input_signature", None):
        view._v814_next_refresh = now + _VARIANT_REFRESH_SECONDS
        return

    # The composed historical loader has its own one-second gate. We are the
    # outer gate now, so open the inner gate for this genuine input change.
    view._v814_next_refresh = 0.0
    v854._BASE_REFRESH_VARIANTS(view)

    source_id = str(signature[0])
    if source_id:
        view._v814_variants = tuple(
            row
            for row in tuple(getattr(view, "_v814_variants", ()))
            if str(row.anchor.source_id) == source_id
        )
    else:
        view._v814_variants = ()
    view._v854_variant_input_signature = signature
    view._v814_next_refresh = now + _VARIANT_REFRESH_SECONDS


def _actor_graph_check_v854_fixup(
    read_view,
    *,
    completed_steps: int,
    next_check_step: int,
    check_interval_steps: int = 1_000,
) -> int:
    """Keep historical generic-view semantics while optimizing real actor views."""
    interval = int(check_interval_steps)
    if interval <= 0:
        raise ValueError("graph check interval must be positive")
    if int(completed_steps) < int(next_check_step):
        return int(next_check_step)

    nodes = getattr(read_view, "_nodes", None)
    edges = getattr(read_view, "_edges", None)
    if not isinstance(nodes, (tuple, list)) or not isinstance(edges, (tuple, list)):
        read_view.invalidate_strategy_cache()
        return (int(completed_steps) // interval + 1) * interval

    from v8 import performance_memory_v854 as v854

    return v854._actor_graph_check_v854(
        read_view,
        completed_steps=completed_steps,
        next_check_step=next_check_step,
        check_interval_steps=interval,
    )


def install_performance_memory_v854_fixups() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from v8 import actor as actor_module
    from v8 import trajectory_optimizer_v814 as optimizer

    optimizer._refresh_view_variants = _refresh_view_variants_v854_fixup
    actor_module._refresh_actor_graph_if_due = _actor_graph_check_v854_fixup
    _INSTALLED = True
