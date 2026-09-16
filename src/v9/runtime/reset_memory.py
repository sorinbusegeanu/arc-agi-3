from __future__ import annotations

import json
import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

RESET_MEMORY_ENV = "ARC_AGI3_V9_RESET_MEMORY"


def _nested_scores(raw: object) -> dict[int, dict[int, float]]:
    return {
        int(environment): {int(action): float(score) for action, score in dict(actions).items()}
        for environment, actions in dict(raw or {}).items()
    }


def _nested_context_scores(raw: object) -> dict[int, dict[int, dict[int, float]]]:
    return {
        int(environment): {
            int(context): {int(action): float(score) for action, score in dict(actions).items()}
            for context, actions in dict(contexts).items()
        }
        for environment, contexts in dict(raw or {}).items()
    }


def _load_checkpoint_scores(checkpoint_path: Path) -> tuple[dict[int, dict[int, float]], dict[int, dict[int, dict[int, float]]]]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "--reset-memory needs PyTorch when the last accepted HGT model differs from the manifest's active model"
        ) from exc
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"accepted HGT checkpoint is invalid: {checkpoint_path}")
    return (
        _nested_scores(checkpoint.get("action_scores", {})),
        _nested_context_scores(checkpoint.get("context_action_scores", {})),
    )


def resolve_last_accepted_hgt(root: str | Path) -> dict[str, Any]:
    root_path = Path(root)
    manifest_path = root_path / "models" / "hgt_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("--reset-memory requires an existing HGT manifest with an accepted model")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    accepted_version = manifest.get("last_accepted_model_version") or manifest.get("accepted_model_version")
    if not accepted_version:
        raise RuntimeError("--reset-memory requires a previously accepted HGT model")
    checkpoint_rel = manifest.get("accepted_checkpoint") or f"models/{accepted_version}.pt"
    checkpoint_path = root_path / str(checkpoint_rel)
    if not checkpoint_path.is_file():
        raise RuntimeError(f"accepted HGT checkpoint is missing: {checkpoint_path}")

    current_version = manifest.get("current_model_version")
    candidate_version = manifest.get("candidate_model_version")
    candidate_status = str(manifest.get("candidate_status") or "")
    manifest_scores_are_accepted = (
        str(current_version or "") == str(accepted_version)
        and (not candidate_version or candidate_status == "PROMOTED")
    )
    if manifest_scores_are_accepted:
        action_scores = _nested_scores(manifest.get("action_scores", {}))
        context_action_scores = _nested_context_scores(manifest.get("context_action_scores", {}))
    else:
        action_scores, context_action_scores = _load_checkpoint_scores(checkpoint_path)

    return {
        "manifest_path": manifest_path,
        "manifest": manifest,
        "accepted_version": str(accepted_version),
        "checkpoint_rel": str(checkpoint_rel),
        "checkpoint_path": checkpoint_path,
        "action_scores": action_scores,
        "context_action_scores": context_action_scores,
    }


def reset_persistent_memory(root: str | Path) -> None:
    root_path = Path(root)
    for name in ("snapshots", "snapshot_chunks"):
        path = root_path / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    for name in ("scientific_config.json", "v9_auxiliary_state.json"):
        path = root_path / name
        if path.exists():
            path.unlink()


def apply_last_accepted_hgt(runtime: Any, bootstrap: dict[str, Any]) -> None:
    version = str(bootstrap["accepted_version"])
    checkpoint_rel = str(bootstrap["checkpoint_rel"])
    runtime.set_hgt_action_scores(
        dict(bootstrap["action_scores"]),
        context_action_scores=dict(bootstrap["context_action_scores"]),
    )
    runtime.unified_telemetry.model_version = version
    runtime.set_telemetry_gauge("reset_memory", 1)
    runtime.set_telemetry_gauge("reset_memory_hgt_model", version)

    manifest_path = Path(bootstrap["manifest_path"])
    manifest = dict(bootstrap["manifest"])
    manifest.update(
        {
            "current_model_version": version,
            "current_checkpoint": checkpoint_rel,
            "accepted_model_version": version,
            "accepted_checkpoint": checkpoint_rel,
            "last_accepted_model_version": version,
            "candidate_model_version": None,
            "candidate_status": "RESET_TO_ACCEPTED",
            "parent_model_version": None,
            "action_scores": {
                str(environment): {str(action): float(score) for action, score in actions.items()}
                for environment, actions in dict(bootstrap["action_scores"]).items()
            },
            "context_action_scores": {
                str(environment): {
                    str(context): {str(action): float(score) for action, score in actions.items()}
                    for context, actions in contexts.items()
                }
                for environment, contexts in dict(bootstrap["context_action_scores"]).items()
            },
        }
    )
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)


def install_reset_memory(runtime_cls: type) -> None:
    if getattr(runtime_cls, "_reset_memory_installed", False):
        return
    original_init = runtime_cls.__init__

    def runtime_init(self: Any, config: Any) -> None:
        requested = os.environ.pop(RESET_MEMORY_ENV, "0") == "1"
        if not requested:
            original_init(self, config)
            return
        bootstrap = resolve_last_accepted_hgt(config.root)
        reset_persistent_memory(config.root)
        original_init(self, replace(config, restore=False, reset_persistent_identity=False))
        apply_last_accepted_hgt(self, bootstrap)

    runtime_cls.__init__ = runtime_init
    runtime_cls._reset_memory_installed = True
