"""env_factory.py — Picklable env factory for multiprocessing workers.

The standard closure returned by _build_env_factory() in run.py is not picklable
(it captures an Arcade instance). EnvFactory stores only the plain parameters and
creates its own Arcade lazily inside each worker process.
"""
from __future__ import annotations


class EnvFactory:
    """Picklable callable that creates ARC envs.

    Safe to pass to worker processes via multiprocessing/ProcessPoolExecutor.
    The Arcade instance is created lazily on first call inside each process
    so it is never included in the pickle payload.
    """

    def __init__(self, game_id: str, op_mode: str, base_seed: int):
        self.game_id = game_id
        self.op_mode = op_mode
        self.base_seed = base_seed
        self._arcade = None   # not pickled — recreated per process

    def __call__(self, ep_idx: int):
        if self._arcade is None:
            from arc_agi import Arcade, OperationMode
            self._arcade = Arcade(operation_mode=OperationMode(self.op_mode))
        env_seed = self.base_seed + ep_idx
        env = self._arcade.make(self.game_id, seed=env_seed)
        if env is None:
            raise RuntimeError(f"arcade.make failed for game_id={self.game_id!r}")
        return env, self.game_id, env_seed

    def __reduce__(self):
        # Serialize only the plain parameters — Arcade is NOT pickled
        return (self.__class__, (self.game_id, self.op_mode, self.base_seed))
