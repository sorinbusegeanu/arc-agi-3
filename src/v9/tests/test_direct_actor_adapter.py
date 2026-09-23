from __future__ import annotations

from v9.runtime.direct_actor_adapter import _DIRECT_ACTOR_FACTORY, install


class _Topology:
    def start_actor(self, *args, **kwargs):
        self.adapter_factory_path = kwargs["adapter_factory_path"]
        return "started"


def test_actor_cli_factory_bypasses_passive_capture_wrapper() -> None:
    install(_Topology)
    topology = _Topology()

    assert topology.start_actor(adapter_factory_path="v9.cli:make_adapter") == "started"
    assert topology.adapter_factory_path == _DIRECT_ACTOR_FACTORY


def test_non_cli_factory_is_preserved() -> None:
    install(_Topology)
    topology = _Topology()

    assert topology.start_actor(adapter_factory_path="custom.module:make") == "started"
    assert topology.adapter_factory_path == "custom.module:make"
