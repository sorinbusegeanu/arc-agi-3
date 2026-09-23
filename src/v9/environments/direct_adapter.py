from __future__ import annotations

from typing import Any


def make_adapter(
    spec: Any,
    *,
    seed: int,
    env_root: str | None,
    alfred_backend_factory: str | None = None,
) -> Any:
    # Lazy import keeps actor processes on the normal CLI adapter path without
    # adding the passive symbol-capture wrapper around observe().
    from v9.cli import make_adapter as base_factory

    return base_factory(
        spec,
        seed=seed,
        env_root=env_root,
        alfred_backend_factory=alfred_backend_factory,
    )
