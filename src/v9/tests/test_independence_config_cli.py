from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from v9.cli import build_parser, resolve_games
from v9.runtime.config import ScientificConfig, write_scientific_config_manifest

PACKAGE = Path(__file__).resolve().parents[1]


def test_production_source_has_no_historical_import_or_reference() -> None:
    for path in PACKAGE.rglob("*.py"):
        if "tests" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith("v8") for alias in node.names), path
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("v8"), path
        assert "v8." not in source, path


def test_v9_imports_and_help_with_only_v9_on_pythonpath(tmp_path: Path) -> None:
    shutil.copytree(PACKAGE, tmp_path / "v9")
    environment = dict(os.environ, PYTHONPATH=str(tmp_path))
    result = subprocess.run([sys.executable, "-m", "v9", "--help"], cwd=tmp_path, env=environment, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "arc-agi3-v9" in result.stdout


def test_scientific_config_id_is_deterministic_and_complete() -> None:
    left = ScientificConfig()
    right = ScientificConfig()
    assert left.config_id == right.config_id
    assert left.as_dict()["scientific_config_id"] == left.config_id.value
    assert left.research_contract_version == "0.6.3.1"
    assert left.design_version == "9.5"


def test_scientific_config_changes_change_identity() -> None:
    assert ScientificConfig().config_id != ScientificConfig(candidates_per_radius=8).config_id


def test_scientific_config_is_immutable_and_manifest_rejects_condition_drift(tmp_path: Path) -> None:
    config = ScientificConfig()
    with pytest.raises(FrozenInstanceError):
        config.design_version = "changed"  # type: ignore[misc]
    target = write_scientific_config_manifest(tmp_path, config)
    assert json.loads(target.read_text())["scientific_config_id"] == config.config_id.value
    with pytest.raises(RuntimeError, match="different immutable"):
        write_scientific_config_manifest(tmp_path, ScientificConfig(candidates_per_radius=8))


def test_cli_keeps_commands_and_v9_native_defaults() -> None:
    parser = build_parser()
    smoke = parser.parse_args(["smoke"])
    assert smoke.root == "runs/v9/continuous"
    continuous = parser.parse_args(["continuous-run", "--games", "synthetic"])
    assert continuous.steps_per_game == 1000
    assert continuous.actors == 8


def test_mixed_selectors_are_native() -> None:
    assert "FrozenLake-v1" in resolve_games("mix")
    assert "tp01" in resolve_games("research_1")
    assert resolve_games("a,b") == ("a", "b")
    with pytest.raises(ValueError):
        resolve_games(" , ")
