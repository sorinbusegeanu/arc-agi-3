from __future__ import annotations

import io
import multiprocessing as mp
from multiprocessing.connection import wait

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
                worker = {
                    "id": worker_id,
                    "process": process,
                    "conn": parent_conn,
                }
                self._workers.append(worker)
                self._worker_debug[worker_id] = {
                    "assigned_game_ids": [],
                    "init_succeeded": False,
                }
                worker["conn"].send(
                    {
                        "type": "init",
                        "payload": {
                            "config_dict": self.cfg.to_dict(),
                            "worker_id": int(worker_id),
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

    def collect(
        self,
        model,
        game_episode_counts: dict[str, int],
        deterministic: bool,
        evaluation: bool,
        acting_mode: str | None = None,
        collection_mode: str = "rl",
    ) -> list:
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
        game_tasks = _build_game_tasks(game_episode_counts, base_offsets=base_offsets)
        if not evaluation:
            for game_id, total in game_episode_counts.items():
                gid = str(game_id)
                self._training_episode_cursor_by_game[gid] = int(self._training_episode_cursor_by_game.get(gid, 0)) + int(total)
        for worker in self._workers:
            worker["conn"].send(
                {
                    "type": "set_model",
                    "payload": {
                        "model_state_bytes": model_state_bytes,
                        "model_signature": _build_model_signature(self.cfg),
                        "acting_mode": acting_mode,
                        "worker_id": int(worker["id"]),
                    },
                }
            )
        for worker in self._workers:
            response = worker["conn"].recv()
            if not response.get("ok"):
                raise RuntimeError(f"parallel worker {worker['id']} set_model failed: {response.get('error')}")
        tasks_queue = list(sorted(game_tasks, key=lambda t: int(t["episodes"]), reverse=True))
        conn_to_worker = {worker["conn"]: worker for worker in self._workers}
        in_flight = 0
        for worker in self._workers:
            if not tasks_queue:
                break
            task = tasks_queue.pop(0)
            worker["conn"].send(
                {
                    "type": "collect_task",
                    "payload": {
                        "task": task,
                        "deterministic": bool(deterministic),
                        "evaluation": bool(evaluation),
                        "collection_mode": str(collection_mode),
                        "worker_id": int(worker["id"]),
                    },
                }
            )
            in_flight += 1
        sequences = []
        while in_flight > 0:
            ready = wait(list(conn_to_worker.keys()))
            if not ready:
                continue
            conn = ready[0]
            worker = conn_to_worker[conn]
            response = conn.recv()
            if not response.get("ok"):
                raise RuntimeError(f"parallel worker {worker['id']} collect failed: {response.get('error')}")
            if "sequences_bytes" in response:
                loaded = torch.load(io.BytesIO(response["sequences_bytes"]), map_location="cpu", weights_only=False)
                sequences.extend(loaded)
            else:
                sequences.extend(response.get("sequences", []))
            if tasks_queue:
                task = tasks_queue.pop(0)
                conn.send(
                    {
                        "type": "collect_task",
                        "payload": {
                            "task": task,
                            "deterministic": bool(deterministic),
                            "evaluation": bool(evaluation),
                            "collection_mode": str(collection_mode),
                            "worker_id": int(worker["id"]),
                        },
                    }
                )
            else:
                in_flight -= 1
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


def _build_game_tasks(game_episode_counts: dict[str, int], base_offsets: dict[str, int] | None = None) -> list[dict]:
    tasks: list[dict] = []
    for game_id, total in game_episode_counts.items():
        episodes = int(total)
        if episodes <= 0:
            continue
        start = int((base_offsets or {}).get(str(game_id), 0))
        for local_idx in range(episodes):
            tasks.append(
                {
                    "game_id": str(game_id),
                    "episodes": 1,
                    "episode_offset": int(start + local_idx),
                }
            )
    return tasks
