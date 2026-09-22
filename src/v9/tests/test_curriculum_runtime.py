from __future__ import annotations

from v9.cli import _runtime_config, build_parser, make_adapter, resolve_game_specs, resolve_games, run_continuous
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
    assert summary["automatic_transfer_experiments"]["attempted"] == 0
    assert summary["automatic_transfer_experiments"]["blocker"] == "automatic transfer validation disabled"
    assert summary["epochs"][0]["training"]["transfer_validation"]["blocker"] == "automatic transfer validation disabled"
    counts = summary["metrics"]["telemetry_diagnostics"]["curriculum_counts"]
    assert sum(counts.values()) == 1200
    assert all(key.startswith("step1|synthetic|") for key in counts)


def test_broad_preset_spans_all_runnable_language_and_symbolic_families() -> None:
    selection = resolve_curriculum_selector("broad")
    assert selection is not None
    assert len(selection.specs) == 111
    assert {spec.adapter for spec in selection.specs} == {
        "synthetic_causal",
        "synthetic_symbolic",
        "gym_discrete",
        "gym_structured",
        "sokoban",
        "minigrid",
        "chess",
        "sudoku",
        "babyai",
        "alfred",
        "arc",
    }
    assert all(spec.curriculum_step == "broad" for spec in selection.specs)
    assert {spec.validation_mode for spec in selection.specs} == {"validation_budgeted"}

    symbolic = tuple(spec for spec in selection.specs if spec.adapter == "synthetic_symbolic")
    assert len(symbolic) == 16
    assert {spec.condition for spec in symbolic} == {"C0", "C1", "C2", "C3"}

    frozen = tuple(spec for spec in selection.specs if spec.game_id == "FrozenLake-v1")
    assert len(frozen) == 4
    assert {(spec.kwargs["map_name"], spec.kwargs["is_slippery"]) for spec in frozen} == {
        ("4x4", False),
        ("8x8", False),
        ("4x4", True),
        ("8x8", True),
    }

    babyai = tuple(spec.game_id for spec in selection.specs if spec.adapter == "babyai")
    assert len(babyai) == 21
    assert "BabyAI-OpenDoorsOrder-v0" not in babyai
    assert "BabyAI-PutNext-v0" not in babyai

    assert tuple(spec.game_id for spec in selection.specs if spec.adapter == "alfred") == (
        "pick_and_place_simple",
        "pick_clean_then_place_in_recep",
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "look_at_obj_in_light",
        "pick_two_obj_and_place",
    )
    assert tuple(spec.game_id for spec in selection.specs if spec.adapter == "chess") == (
        "chess_first_white",
        "chess_random_white",
    )
    assert tuple(spec.game_id for spec in selection.specs if spec.adapter == "sudoku") == (
        "sudoku_clues_45",
        "sudoku_clues_36",
        "sudoku_clues_30",
    )
    arc_games = tuple(spec.game_id for spec in selection.specs if spec.adapter == "arc")
    assert len(arc_games) == 42
    assert arc_games[:4] == ("gp01", "gp02", "ez01", "ez02")
    assert arc_games[-4:] == ("wk01", "rf01", "mo01", "zq01")


def test_continuous_cli_inherits_scientific_transfer_defaults(tmp_path) -> None:
    args = build_parser().parse_args([
        "continuous-run",
        "--root", str(tmp_path / "fresh"),
        "--games", "step1",
        "--no-dashboard",
    ])
    config = _runtime_config(args)
    assert config.scientific.transfer_validation_trials_per_interval == 900
    assert config.scientific.transfer_validation_workers == 30
    assert config.scientific.transfer_validation_time_budget_seconds == 300.0


def test_environment_spec_has_no_runtime_identity_contract() -> None:
    spec = resolve_game_specs("step1")[0]
    assert not hasattr(spec, "instance_id")
    adapter = make_adapter(spec, seed=0, env_root=None)
    assert adapter.identity().instance_id.value > 0


def test_viability_epoch_path_uses_runtime_environment_identity(tmp_path) -> None:
    root = tmp_path / "viability_identity"
    args = build_parser().parse_args([
        "continuous-run",
        "--root", str(root),
        "--games", "step1",
        "--steps-per-game", "1",
        "--actors", "2",
        "--shards", "2",
        "--stage-workers", "1",
        "--epochs", "1",
        "--no-peers",
        "--no-dashboard",
        "--no-automatic-experiments",
    ])
    assert run_continuous(args) == 0
    assert (root / "environment_viability.log").exists()
    rows = [
        __import__("json").loads(line)
        for line in (root / "environment_viability.log").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    epoch_rows = [row for row in rows if "game" in row]
    assert epoch_rows
    assert all("evidence_confidence" in row for row in epoch_rows)
