from __future__ import annotations

from dataclasses import dataclass

from v4_5.contracts.errors import PluginRegistrationError
from v4_5.plugins.base import PlannerPlugin


@dataclass
class PluginRegistry:
    _plugins: dict[str, PlannerPlugin]

    def register(self, plugin: PlannerPlugin) -> None:
        name = str(plugin.plugin_name)
        if not name:
            raise PluginRegistrationError("plugin name is required")
        self._plugins[name] = plugin

    def get(self, name: str) -> PlannerPlugin:
        try:
            return self._plugins[name]
        except KeyError as exc:
            raise PluginRegistrationError(f"unknown plugin: {name}") from exc

    def all(self) -> tuple[PlannerPlugin, ...]:
        return tuple(self._plugins[name] for name in sorted(self._plugins))


def default_registry() -> PluginRegistry:
    from v4_5.plugins.clickPlugin import ClickPlugin
    from v4_5.plugins.compositionPlugin import CompositionPlugin
    from v4_5.plugins.memoryPlugin import MemoryHiddenPlugin
    from v4_5.plugins.movementPlugin import MovementPlugin
    from v4_5.plugins.rulePlugin import RuleSwitchPlugin
    from v4_5.plugins.temporalPlugin import TemporalPlugin

    registry = PluginRegistry(_plugins={})
    for plugin in (
        MovementPlugin(),
        ClickPlugin(),
        MemoryHiddenPlugin(),
        RuleSwitchPlugin(),
        TemporalPlugin(),
        CompositionPlugin(),
    ):
        registry.register(plugin)
    return registry
