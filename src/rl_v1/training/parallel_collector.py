from __future__ import annotations

import io
import multiprocessing as mp

import torch

from rl_v1.training.worker import _build_model_signature, rollout_worker_main


class ParallelRolloutManager:
    def __init__(self, cfg, action_selector_cfg_source) -> None:
        self.cfg = cfg
        self.action_selector_cfg_source = action_selector_cfg_source
        self._started = False
        self._ctx = None
        self._workers: list[dict] = []
        self._worker_debug: dict[int, dict] = {}
        self._training_episode_cursor_by_game: dict[str, int] = {}

    def start(self) -> None:
        if self._started:
            return
        try:
            torch.multiprocessing.set_sharing_strategy("file_system")
        except Exception:
            pass
        self._ctx = mp.get_context(self.cfg.env.mp_start_method)
        # env.num_workers is deprecated for multiprocessing; runtime.rollout_processes is the active knob.
        partitions = _build_worker_game_assignments(
            self.cfg.env.game_ids,
            self.cfg.env.game_episode_multipliers,
            int(self.cfg.runtime.rollout_processes),
        )
        try:
            for worker_id in range(int(self.cfg.runtime.rollout_processes)):
                parent_conn, child_conn = self._ctx.Pipe()
                process = self._ctx.Process(
                    target=rollout_worker_main,
                    args=(child_conn,),
                    daemon=True,
                )
                process.start()
                child_conn.close()
                assigned_game_ids = partitions[worker_id] if worker_id < len(partitions) else []
                worker = {
                    "id": worker_id,
                    "process": process,
                    "conn": parent_conn,
                    "game_ids": assigned_game_ids,
                }
                self._workers.append(worker)
                self._worker_debug[worker_id] = {
                    "assigned_game_ids": list(assigned_game_ids),
                    "init_succeeded": False,
                }
                worker["conn"].send(
                    {
                        "type": "init",
                        "payload": {
                            "config_dict": self.cfg.to_dict(),
                            "worker_id": int(worker_id),
                            "game_ids": list(assigned_game_ids),
                        },
                    }
                )
            for worker in self._workers:
                response = worker["conn"].recv()
                if not response.get("ok"):
                    raise RuntimeError(f"parallel worker {worker['id']} init failed: {response.get('error')}")
                self._worker_debug[int(worker["id"])]["init_succeeded"] = True
            self._started = True
        except Exception:
            self.close()
            raise

    def collect(self, model, game_episode_counts: dict[str, int], deterministic: bool, evaluation: bool, acting_mode: str | None = None) -> list:
        if not self._started:
            self.start()
        model_buffer = io.BytesIO()
        cpu_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        torch.save(cpu_state, model_buffer)
        model_state_bytes = model_buffer.getvalue()
        base_offsets = (
            {str(game_id): 0 for game_id in game_episode_counts}
            if evaluation
            else {str(game_id): int(self._training_episode_cursor_by_game.get(str(game_id), 0)) for game_id in game_episode_counts}
        )
        worker_loads = _split_episode_counts_across_workers(self._workers, game_episode_counts, base_offsets=base_offsets)
        if not evaluation:
            for game_id, total in game_episode_counts.items():
                gid = str(game_id)
                self._training_episode_cursor_by_game[gid] = int(self._training_episode_cursor_by_game.get(gid, 0)) + int(total)
        for worker in self._workers:
            worker_plan = worker_loads.get(int(worker["id"]), {"counts": {}, "offsets": {}})
            worker["conn"].send(
                {
                    "type": "collect",
                    "payload": {
                        "model_state_bytes": model_state_bytes,
                        "model_signature": _build_model_signature(self.cfg),
                        "per_game_episode_counts": worker_plan["counts"],
                        "per_game_episode_offsets": worker_plan["offsets"],
                        "deterministic": bool(deterministic),
                        "evaluation": bool(evaluation),
                        "acting_mode": acting_mode,
                        "worker_id": int(worker["id"]),
                    },
                }
            )
        sequences = []
        for worker in self._workers:
            response = worker["conn"].recv()
            if not response.get("ok"):
                raise RuntimeError(f"parallel worker {worker['id']} collect failed: {response.get('error')}")
            if "sequences_bytes" in response:
                loaded = torch.load(io.BytesIO(response["sequences_bytes"]), map_location="cpu", weights_only=False)
                sequences.extend(loaded)
            else:
                sequences.extend(response.get("sequences", []))
        return sequences

    def close(self) -> None:
        if not self._started:
            return
        for worker in self._workers:
            try:
                worker["conn"].send({"type": "close"})
            except Exception:
                pass
        for worker in self._workers:
            try:
                worker["conn"].recv()
            except Exception:
                pass
            worker["process"].join(timeout=5.0)
            if worker["process"].is_alive():
                worker["process"].terminate()
            worker["conn"].close()
        self._workers = []
        self._worker_debug = {}
        self._training_episode_cursor_by_game = {}
        self._started = False


def _partition_game_ids(game_ids, process_count: int):
    count = max(1, int(process_count))
    buckets = [[] for _ in range(count)]
    for idx, game_id in enumerate(game_ids):
        buckets[idx % count].append(game_id)
    return buckets


def _build_worker_game_assignments(game_ids, multipliers, process_count: int):
    count = max(1, int(process_count))
    configured = [str(game_id) for game_id in game_ids]
    if not configured:
        return [[] for _ in range(count)]
    buckets = [[] for _ in range(count)]
    # First pass: guarantee coverage so every configured game is collected at least once.
    for idx, game_id in enumerate(configured):
        buckets[idx % count].append(game_id)

    # Second pass: spread weighted replicas across workers without dropping any game.
    extras = []
    for game_id in configured:
        weight = int((multipliers or {}).get(game_id, 1))
        extras.extend([game_id] * max(0, weight - 1))
    for game_id in extras:
        worker_ids = sorted(range(count), key=lambda wid: (len(buckets[wid]), wid))
        placed = False
        for wid in worker_ids:
            if game_id not in buckets[wid]:
                buckets[wid].append(game_id)
                placed = True
                break
        if not placed:
            buckets[worker_ids[0]].append(game_id)
    # Ensure each worker has unique game ids; duplicate game entries on one worker
    # can cause repeated collection passes and episode-identity collisions.
    for idx, worker_games in enumerate(buckets):
        buckets[idx] = list(dict.fromkeys(worker_games))
    return buckets


def _split_episode_counts_across_workers(workers, game_episode_counts: dict[str, int], base_offsets: dict[str, int] | None = None) -> dict[int, dict]:
    by_game_workers: dict[str, list[int]] = {}
    for worker in workers:
        wid = int(worker["id"])
        for game_id in set(worker["game_ids"]):
            by_game_workers.setdefault(str(game_id), []).append(wid)
    result: dict[int, dict] = {
        int(worker["id"]): {"counts": {}, "offsets": {}}
        for worker in workers
    }
    for game_id, total in game_episode_counts.items():
        ids = by_game_workers.get(str(game_id), [])
        if not ids:
            continue
        total_count = int(total)
        base = total_count // len(ids)
        rem = total_count % len(ids)
        offset = int((base_offsets or {}).get(str(game_id), 0))
        for idx, wid in enumerate(ids):
            share = base + (1 if idx < rem else 0)
            result[wid]["counts"][str(game_id)] = int(share)
            result[wid]["offsets"][str(game_id)] = int(offset)
            offset += share
    return result
