from __future__ import annotations

from v9.cli import build_parser, make_adapter, resolve_game_specs, resolve_games, run_continuous
from v9.curriculum import load_curriculum, resolve_curriculum_selector
from v9.environments import GymDiscreteAdapter, SokobanAdapter, SyntheticSymbolicEnvironment


def test_all_fourteen_curriculum_steps_resolve() -> None:
    config = load_curriculum()
    assert tuple(config["steps"]) == tuple(f"step{index}" for index in range(1, 15))
    for index in range(1, 15):
        selection = resolve_curriculum_selector(f"step{index}")
        assert selection is not None
        assert selection.specs
        assert all(spec.curriculum_step == f"step{index}" for spec in selection.specs)


def test_step1_is_deterministic_causal_curriculum() -> None:
    specs = resolve_game_specs("step1")
    assert len(specs) == 12
    assert {spec.adapter for spec in specs} == {"synthetic_causal"}
    assert specs[0].game_id == "syn_move"
    assert specs[-1].game_id == "syn_delayed_effect"
    assert {spec.validation_mode for spec in specs} == {"learning_only"}

    adapter = make_adapter(specs[0], seed=3, env_root=None)
    assert isinstance(adapter, SyntheticSymbolicEnvironment)
    assert adapter.identity().environment_type == "syn_move"
    assert adapter.optional_symbol_stream() == ()


def test_step2_preserves_frozenlake_kwargs() -> None:
    specs = resolve_game_specs("step2")
    frozen = [spec for spec in specs if spec.game_id == "FrozenLake-v1"]
    assert len(frozen) == 2
    assert {spec.kwargs["map_name"] for spec in frozen} == {"4x4", "8x8"}
    assert all(spec.kwargs["is_slippery"] is False for spec in frozen)

    adapter = make_adapter(frozen[0], seed=1, env_root=None)
    try:
        assert isinstance(adapter, GymDiscreteAdapter)
        assert adapter.make_kwargs["is_slippery"] is False
    finally:
        adapter.close()


def test_step3_uses_native_sokoban_adapter() -> None:
    specs = resolve_game_specs("step3")
    sokoban = next(spec for spec in specs if spec.adapter == "sokoban")
    adapter = make_adapter(sokoban, seed=0, env_root=None)
    assert isinstance(adapter, SokobanAdapter)
    assert adapter.identity().family == "sokoban"
    assert adapter.available_actions() == (0, 1, 2, 3)


def test_curriculum_presets_resolve_structured_specs() -> None:
    semantics = resolve_curriculum_selector("semantics")
    assert semantics is not None
    assert {spec.curriculum_step for spec in semantics.specs} == {"step5", "step6", "step7", "step8", "step9"}
    assert any(spec.adapter == "babyai" for spec in semantics.specs)

    cross_family = resolve_curriculum_selector("cross_family")
    assert cross_family is not None
    assert {spec.curriculum_step for spec in cross_family.specs} == {"step12"}
    assert len({spec.adapter for spec in cross_family.specs}) >= 8


def test_legacy_game_selectors_remain_supported() -> None:
    assert resolve_games("gp03") == ("gp03",)
    mix = resolve_game_specs("mix")
    assert {spec.adapter for spec in mix} == {"auto"}
    assert len(mix) == 5


def test_step1_continuous_run_executes_curriculum(tmp_path) -> None:
    root = tmp_path / "step1"
    args = build_parser().parse_args([
        "continuous-run",
        "--root", str(root),
        "--games", "step1",
        "--steps-per-game", "1",
        "--actors", "2",
        "--shards", "2",
        "--stage-workers", "1",
        "--no-peers",
        "--no-dashboard",
    ])
    assert run_continuous(args) == 0

    import json
    summary = json.loads((root / "v9_run_summary.json").read_text(encoding="utf-8"))
    assert len(summary["games"]) == 12
    assert len(summary["actors"]) == 12
    assert summary["automatic_transfer_experiments"]["mode"] == "learning_only"
    counts = summary["metrics"]["telemetry_diagnostics"]["curriculum_counts"]
    assert sum(counts.values()) == 12
    assert all(key.startswith("step1|synthetic|") for key in counts)


def test_broad_preset_includes_all_runnable_babyai_language_games() -> None:
    selection = resolve_curriculum_selector("broad")
    assert selection is not None
    assert len(selection.specs) == 47
    assert {spec.adapter for spec in selection.specs} == {
        "synthetic_causal",
        "gym_discrete",
        "gym_structured",
        "sokoban",
        "minigrid",
        "babyai",
        "arc",
    }
    assert all(spec.curriculum_step == "broad" for spec in selection.specs)
    babyai = tuple(spec.game_id for spec in selection.specs if spec.adapter == "babyai")
    assert len(babyai) == 21
    assert "BabyAI-OpenDoorsOrder-v0" not in babyai
    assert "BabyAI-PutNext-v0" not in babyai
    assert tuple(spec.game_id for spec in selection.specs if spec.adapter == "arc") == (
        "g50t",
        "ls20",
        "re86",
        "tr87",
        "tu93",
        "wa30",
    )
