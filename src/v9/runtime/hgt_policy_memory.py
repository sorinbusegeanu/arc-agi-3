from __future__ import annotations

import gc
import json
import os
import pickle
from pathlib import Path
from typing import Any, Callable


POLICY_SIDECAR_SCHEMA = 1


def _sidecar_path(root: str | Path, model_version: str) -> Path:
    return Path(root) / "models" / f"{model_version}.policy.pkl"


def _write_policy_sidecar(
    root: str | Path,
    model_version: str,
    scores: dict[int, dict[int, float]],
    context_scores: dict[int, dict[int, dict[int, float]]],
) -> Path:
    path = _sidecar_path(root, model_version)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pkl.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(
            {
                "schema": POLICY_SIDECAR_SCHEMA,
                "model_version": str(model_version),
                "scores": scores,
                "context_scores": context_scores,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


def _load_policy_sidecar(root: str | Path, model_version: str) -> tuple[dict, dict] | None:
    path = _sidecar_path(root, model_version)
    if not path.exists():
        return None
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or int(payload.get("schema", 0)) != POLICY_SIDECAR_SCHEMA:
        raise RuntimeError(f"invalid HGT policy sidecar: {path}")
    if str(payload.get("model_version")) != str(model_version):
        raise RuntimeError(f"HGT policy sidecar version mismatch: {path}")
    scores = payload.get("scores")
    context_scores = payload.get("context_scores")
    if not isinstance(scores, dict) or not isinstance(context_scores, dict):
        raise RuntimeError(f"invalid HGT policy payload: {path}")
    return scores, context_scores


def _manifest(root: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(root) / "models" / "hgt_manifest.json"
    if not path.exists():
        return path, {}
    return path, json.loads(path.read_text(encoding="utf-8"))


def _atomic_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _assign_policy(runtime: Any, model_version: str, scores: dict, context_scores: dict) -> str:
    # Policy sidecars already contain normalized integer-keyed maps. Assigning
    # them directly avoids the second full deep copy performed by
    # set_hgt_action_scores during model switching.
    with runtime._lock:
        runtime._hgt_action_scores = scores
        runtime._hgt_context_action_scores = context_scores
        runtime.unified_telemetry.model_version = str(model_version)
        runtime._actor_policy_generation += 1
    return str(model_version)


def _policy_from_current_manifest(root: str | Path, model_version: str) -> tuple[dict, dict] | None:
    _path, manifest = _manifest(root)
    if str(manifest.get("current_model_version") or "") != str(model_version):
        return None
    scores = manifest.get("action_scores")
    context_scores = manifest.get("context_action_scores")
    if not isinstance(scores, dict) or not isinstance(context_scores, dict):
        return None
    normalized_scores = {
        int(environment): {int(action): float(score) for action, score in actions.items()}
        for environment, actions in scores.items()
    }
    normalized_context = {
        int(environment): {
            int(context): {int(action): float(score) for action, score in actions.items()}
            for context, actions in contexts.items()
        }
        for environment, contexts in context_scores.items()
    }
    return normalized_scores, normalized_context


def _legacy_policy_from_checkpoint(root: str | Path, model_version: str) -> tuple[dict, dict] | None:
    checkpoint_path = Path(root) / "models" / f"{model_version}.pt"
    if not checkpoint_path.exists():
        return None
    try:
        import torch
    except ImportError:
        return None
    # mmap keeps model/optimizer tensor storage file-backed while the small
    # behavior policy dictionaries are extracted from legacy checkpoints.
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        return None
    scores = {
        int(environment): {int(action): float(score) for action, score in actions.items()}
        for environment, actions in dict(checkpoint.get("action_scores", {})).items()
    }
    context_scores = {
        int(environment): {
            int(context): {int(action): float(score) for action, score in actions.items()}
            for context, actions in contexts.items()
        }
        for environment, contexts in dict(checkpoint.get("context_action_scores", {})).items()
    }
    del checkpoint
    gc.collect()
    _write_policy_sidecar(root, model_version, scores, context_scores)
    return scores, context_scores


def load_hgt_policy_version(runtime: Any, *, root: str | Path, model_version: str) -> str | None:
    selected = str(model_version)
    loaded = _load_policy_sidecar(root, selected)
    if loaded is None:
        loaded = _policy_from_current_manifest(root, selected)
        if loaded is not None:
            _write_policy_sidecar(root, selected, loaded[0], loaded[1])
    if loaded is None:
        loaded = _legacy_policy_from_checkpoint(root, selected)
    if loaded is None:
        return None
    return _assign_policy(runtime, selected, loaded[0], loaded[1])


def rollback_hgt_model(runtime: Any, *, root: str | Path) -> str | None:
    manifest_path, manifest = _manifest(root)
    if not manifest:
        return None
    parent_version = manifest.get("parent_model_version") or manifest.get("last_accepted_model_version")
    current_version = manifest.get("current_model_version")
    if not current_version:
        return None
    if not parent_version:
        with runtime._lock:
            runtime._hgt_action_scores = {}
            runtime._hgt_context_action_scores = {}
            runtime.unified_telemetry.model_version = "untrained"
            runtime._actor_policy_generation += 1
        manifest.update(
            {
                "current_model_version": None,
                "current_checkpoint": None,
                "parent_model_version": None,
                "rollback_from_model_version": str(current_version),
                "rollback_to_model_version": "untrained",
                "accepted_model_version": None,
                "accepted_checkpoint": None,
            }
        )
        _atomic_manifest(manifest_path, manifest)
        runtime.set_telemetry_gauge("hgt_behavior_rollback", 1)
        runtime.set_telemetry_gauge("hgt_rollback_from_model", str(current_version))
        runtime.set_telemetry_gauge("hgt_rollback_to_model", "untrained")
        return "untrained"
    if str(parent_version) == str(current_version):
        return str(current_version)
    restored = load_hgt_policy_version(runtime, root=root, model_version=str(parent_version))
    if restored is None:
        return None
    checkpoint_rel = f"models/{parent_version}.pt"
    manifest.update(
        {
            "current_model_version": str(parent_version),
            "current_checkpoint": checkpoint_rel,
            "parent_model_version": None,
            "accepted_model_version": str(parent_version),
            "accepted_checkpoint": checkpoint_rel,
            "rollback_from_model_version": str(current_version),
        }
    )
    _atomic_manifest(manifest_path, manifest)
    runtime.set_telemetry_gauge("hgt_behavior_rollback", 1)
    runtime.set_telemetry_gauge("hgt_rollback_from_model", str(current_version))
    runtime.set_telemetry_gauge("hgt_rollback_to_model", str(parent_version))
    return restored


def resolve_hgt_behavior_test(runtime: Any, *, root: str | Path, accepted: bool) -> str | None:
    manifest_path, manifest = _manifest(root)
    if not manifest:
        return None
    candidate = manifest.get("candidate_model_version")
    if not candidate or manifest.get("candidate_status") != "TESTING_PENDING_BEHAVIOR":
        return None
    if accepted:
        accepted_checkpoint = f"models/{candidate}.pt"
        manifest.update(
            {
                "candidate_status": "PROMOTED",
                "last_accepted_model_version": str(candidate),
                "accepted_model_version": str(candidate),
                "accepted_checkpoint": accepted_checkpoint,
                "current_model_version": str(candidate),
                "current_checkpoint": accepted_checkpoint,
                "parent_model_version": None,
                "candidate_model_version": None,
            }
        )
        _atomic_manifest(manifest_path, manifest)
        runtime.set_telemetry_gauge("hgt_behavior_test_result", "PROMOTED")
        return str(candidate)
    rolled_back = rollback_hgt_model(runtime, root=root)
    _path, manifest = _manifest(root)
    manifest["candidate_status"] = "REJECTED_BEHAVIOR_GATE"
    manifest["candidate_model_version"] = None
    _atomic_manifest(manifest_path, manifest)
    runtime.set_telemetry_gauge("hgt_behavior_test_result", "REJECTED_BEHAVIOR_GATE")
    return rolled_back


def _capture_policy_state_shallow(self: Any) -> dict[str, Any]:
    # set_hgt_action_scores replaces the policy maps instead of mutating them,
    # so retaining references here is safe and avoids a full nested deep copy.
    with self._lock:
        return {
            "scores": self._hgt_action_scores,
            "context_scores": self._hgt_context_action_scores,
            "model_version": self.unified_telemetry.model_version,
        }


def _restore_hgt_checkpoint_lightweight(self: Any) -> None:
    manifest_path, manifest = _manifest(self.root)
    if not manifest:
        return
    selected = manifest.get("current_model_version") or manifest.get("accepted_model_version")
    if not selected:
        return
    try:
        load_hgt_policy_version(self, root=self.root, model_version=str(selected))
    except (OSError, RuntimeError, ValueError, pickle.PickleError):
        return


def install(
    runtime_class: type,
    hgt_training_module: Any,
    hgt_package: Any,
    epoch_runner_module: Any,
) -> None:
    original_train: Callable[..., Any] = hgt_training_module.train_hgt_epoch

    def train_hgt_epoch(runtime: Any, **kwargs):
        root = kwargs["root"]
        current = str(runtime.unified_telemetry.model_version or "")
        if current and current not in {"None", "untrained"} and not _sidecar_path(root, current).exists():
            _write_policy_sidecar(
                root,
                current,
                runtime._hgt_action_scores,
                runtime._hgt_context_action_scores,
            )
        result = original_train(runtime, **kwargs)
        version = str(getattr(result, "model_version", "") or "")
        if version and version not in {"None", "untrained"}:
            _write_policy_sidecar(
                root,
                version,
                runtime._hgt_action_scores,
                runtime._hgt_context_action_scores,
            )
        return result

    runtime_class.capture_hgt_policy_state = _capture_policy_state_shallow
    runtime_class._restore_hgt_checkpoint = _restore_hgt_checkpoint_lightweight
    hgt_training_module.load_hgt_policy_version = load_hgt_policy_version
    hgt_training_module.rollback_hgt_model = rollback_hgt_model
    hgt_training_module.resolve_hgt_behavior_test = resolve_hgt_behavior_test
    hgt_training_module.train_hgt_epoch = train_hgt_epoch
    hgt_package.load_hgt_policy_version = load_hgt_policy_version
    hgt_package.rollback_hgt_model = rollback_hgt_model
    hgt_package.resolve_hgt_behavior_test = resolve_hgt_behavior_test
    hgt_package.train_hgt_epoch = train_hgt_epoch
    epoch_runner_module.load_hgt_policy_version = load_hgt_policy_version
    epoch_runner_module.resolve_hgt_behavior_test = resolve_hgt_behavior_test
    epoch_runner_module.train_hgt_epoch = train_hgt_epoch
