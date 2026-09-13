from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class EnvironmentSpec:
    adapter: str
    game_id: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    condition: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    curriculum_step: str | None = None
    validation_mode: str | None = None

    @property
    def display_name(self) -> str:
        suffix = ""
        if self.kwargs:
            suffix = "[" + ",".join(f"{key}={self.kwargs[key]}" for key in sorted(self.kwargs)) + "]"
        if self.condition:
            suffix += f"[{self.condition}]"
        return f"{self.game_id}{suffix}"


@dataclass(frozen=True, slots=True)
class CurriculumSelection:
    selector: str
    specs: tuple[EnvironmentSpec, ...]
    validation_mode: str | None = None


def default_curriculum_path() -> Path:
    return Path(__file__).resolve().parents[2] / "docs" / "v9" / "curriculum.yaml"


def load_curriculum(path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path) if path is not None else default_curriculum_path()
    if not target.exists():
        raise FileNotFoundError(f"curriculum config not found: {target}")
    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or int(raw.get("version", 0)) <= 0:
        raise ValueError("invalid curriculum config")
    if not isinstance(raw.get("steps"), dict):
        raise ValueError("curriculum requires steps")
    return raw


def _resolve_candidate(game: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if "id" in game:
        return str(game["id"]), dict(game.get("kwargs", {}))
    candidates = tuple(str(value) for value in game.get("id_candidates", ()))
    if not candidates:
        raise ValueError("curriculum game mapping requires id or id_candidates")
    try:
        import gymnasium as gym
    except ImportError:
        return candidates[0], {}
    for candidate in candidates:
        try:
            gym.spec(candidate)
            return candidate, {}
        except Exception:
            continue
    return candidates[0], {}


def _game_spec(
    *,
    adapter: str,
    raw_game: Any,
    condition: str | None,
    options: dict[str, Any],
    step: str | None,
    validation_mode: str | None,
) -> EnvironmentSpec:
    if isinstance(raw_game, str):
        game_id, kwargs = raw_game, {}
    elif isinstance(raw_game, dict):
        game_id, kwargs = _resolve_candidate(raw_game)
    else:
        raise ValueError(f"unsupported curriculum game entry: {raw_game!r}")
    return EnvironmentSpec(
        adapter=str(adapter),
        game_id=str(game_id),
        kwargs=dict(kwargs),
        condition=None if condition is None else str(condition),
        options=dict(options),
        curriculum_step=step,
        validation_mode=validation_mode,
    )


def _step_specs(config: dict[str, Any], step_name: str) -> tuple[EnvironmentSpec, ...]:
    steps = dict(config["steps"])
    if step_name not in steps:
        raise ValueError(f"unknown curriculum step: {step_name}")
    step = dict(steps[step_name])
    default_validation = str(config.get("defaults", {}).get("validation_mode", "learning_only"))
    validation_mode = str(step.get("validation_mode", default_validation))
    step_options = dict(step.get("environment_options", {}))
    specs: list[EnvironmentSpec] = []
    for environment in step.get("environments", ()):
        row = dict(environment)
        adapter = str(row["adapter"])
        condition = row.get("condition")
        options = {**step_options, **dict(row.get("environment_options", {}))}
        for game in row.get("games", ()):
            specs.append(
                _game_spec(
                    adapter=adapter,
                    raw_game=game,
                    condition=condition,
                    options=options,
                    step=step_name,
                    validation_mode=validation_mode,
                )
            )
    if not specs:
        raise ValueError(f"curriculum step {step_name} resolves to no environments")
    return tuple(specs)


def resolve_curriculum_selector(selector: str, *, path: str | Path | None = None) -> CurriculumSelection | None:
    normalized = selector.strip().lower()
    config = load_curriculum(path)
    steps = dict(config["steps"])
    if normalized in steps:
        specs = _step_specs(config, normalized)
        return CurriculumSelection(normalized, specs, specs[0].validation_mode)

    presets = dict(config.get("presets", {}))
    if normalized not in presets:
        return None

    preset = dict(presets[normalized])
    allowed_adapters = set(str(value) for value in preset.get("adapters", ()))
    specs: list[EnvironmentSpec] = []
    for step_name in preset.get("include_steps", ()):
        for spec in _step_specs(config, str(step_name)):
            if allowed_adapters and spec.adapter not in allowed_adapters:
                continue
            specs.append(spec)
    if not specs:
        raise ValueError(f"curriculum preset {normalized} resolves to no environments")
    modes = {spec.validation_mode for spec in specs}
    validation_mode = modes.pop() if len(modes) == 1 else None
    return CurriculumSelection(normalized, tuple(specs), validation_mode)
