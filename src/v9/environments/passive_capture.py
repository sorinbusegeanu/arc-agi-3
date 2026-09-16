from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class PassiveSymbolCaptureAdapter:
    """Transparent adapter that preserves when passive symbols were observed.

    Plain symbol streams are sampled at observation time and after the action.
    Adapters that already emit structured timing records retain their explicit
    phase/macro/micro metadata unchanged.
    """

    def __init__(self, adapter: Any) -> None:
        self._adapter = adapter
        self._macro_step = 0
        self._pending: list[object] = []
        self._pre_sampled = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._adapter, name)

    @staticmethod
    def _structured(raw: object, *, phase: str, macro_step: int, micro_step: int) -> object:
        if isinstance(raw, Mapping):
            row = dict(raw)
            row.setdefault("phase", phase)
            row.setdefault("macro_step", int(macro_step))
            row.setdefault("micro_step", int(micro_step))
            return row
        if hasattr(raw, "token") or hasattr(raw, "value"):
            token = getattr(raw, "token", getattr(raw, "value", raw))
            row: dict[str, object] = {
                "token": token,
                "phase": str(getattr(raw, "phase", phase)),
                "macro_step": int(getattr(raw, "macro_step", macro_step)),
                "micro_step": int(getattr(raw, "micro_step", micro_step)),
            }
            causal = getattr(raw, "causal_watermark", None)
            if causal is not None:
                row["causal_watermark"] = int(causal)
            source = getattr(raw, "source_sequence", None)
            if source is not None:
                row["source_sequence"] = int(source)
            return row
        return {
            "token": raw,
            "phase": phase,
            "macro_step": int(macro_step),
            "micro_step": int(micro_step),
        }

    def _sample(self, phase: str) -> list[object]:
        source = getattr(self._adapter, "optional_symbol_stream", None)
        if not callable(source):
            return []
        rows = tuple(source() or ())
        return [
            self._structured(raw, phase=phase, macro_step=self._macro_step, micro_step=index)
            for index, raw in enumerate(rows)
        ]

    def reset(self) -> Any:
        result = self._adapter.reset()
        self._macro_step = 0
        self._pending.clear()
        self._pre_sampled = False
        return result

    def observe(self) -> Any:
        observation = self._adapter.observe()
        # One pre-action sample belongs to the next action opportunity. Repeated
        # observe() calls before step() do not duplicate passive evidence.
        if not self._pre_sampled:
            self._pending.extend(self._sample("BEFORE_ACTION"))
            self._pre_sampled = True
        return observation

    def step(self, native_action: Any) -> Any:
        result = self._adapter.step(native_action)
        progress = getattr(self._adapter, "task_progress", None)
        boundary = getattr(self._adapter, "boundary_event", None)
        terminal = False
        if callable(progress):
            row = progress()
            terminal = bool(
                getattr(row, "terminal", False)
                or getattr(row, "success", False)
                or getattr(row, "failure", False)
                or getattr(row, "truncated", False)
            )
        if callable(boundary):
            row = boundary()
            terminal = terminal or not bool(getattr(row, "continuation", True))
        phase = "AFTER_OUTCOME" if terminal else "AFTER_ACTION"
        self._pending.extend(self._sample(phase))
        self._macro_step += 1
        self._pre_sampled = False
        return result

    def optional_symbol_stream(self) -> tuple[object, ...]:
        rows = tuple(self._pending)
        self._pending.clear()
        return rows


def wrap_adapter(adapter: Any) -> Any:
    if isinstance(adapter, PassiveSymbolCaptureAdapter):
        return adapter
    return PassiveSymbolCaptureAdapter(adapter)


def make_adapter(spec: Any, *, seed: int, env_root: str | None, alfred_backend_factory: str | None = None) -> Any:
    # Lazy import avoids a package-import cycle in spawned actor processes.
    from v9.cli import make_adapter as base_factory

    return wrap_adapter(
        base_factory(
            spec,
            seed=seed,
            env_root=env_root,
            alfred_backend_factory=alfred_backend_factory,
        )
    )


def install_cli_adapter_factory(cli_module: Any) -> None:
    if getattr(cli_module, "_v978_passive_capture_installed", False):
        return
    original = cli_module.make_adapter

    def make_adapter_wrapped(spec: Any, *, seed: int, env_root: str | None, alfred_backend_factory: str | None = None) -> Any:
        return wrap_adapter(
            original(
                spec,
                seed=seed,
                env_root=env_root,
                alfred_backend_factory=alfred_backend_factory,
            )
        )

    cli_module.make_adapter = make_adapter_wrapped
    cli_module._v978_passive_capture_installed = True


def install_process_factory_route(multiprocess_module: Any) -> None:
    topology = multiprocess_module.ProcessTopology
    if getattr(topology, "_v978_passive_capture_installed", False):
        return
    original = topology.start_actor

    def start_actor(self: Any, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("adapter_factory_path") == "v9.cli:make_adapter":
            kwargs["adapter_factory_path"] = "v9.environments.passive_capture:make_adapter"
        return original(self, *args, **kwargs)

    topology.start_actor = start_actor
    topology._v978_passive_capture_installed = True
