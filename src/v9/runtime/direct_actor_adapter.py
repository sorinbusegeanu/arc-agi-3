from __future__ import annotations

from typing import Any


_DIRECT_ACTOR_FACTORY = "v9.environments.direct_adapter:make_adapter"


def install(process_topology_cls: type) -> None:
    if getattr(process_topology_cls, "_direct_actor_adapter_installed", False):
        return

    original_start_actor = process_topology_cls.start_actor

    def start_actor(self: Any, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("adapter_factory_path") == "v9.cli:make_adapter":
            kwargs = dict(kwargs)
            kwargs["adapter_factory_path"] = _DIRECT_ACTOR_FACTORY
        return original_start_actor(self, *args, **kwargs)

    process_topology_cls.start_actor = start_actor
    process_topology_cls._direct_actor_adapter_installed = True
