"""Environment adapters and local benchmark environments for v8."""

from v8.environments.chess_env import (
    CHESS_GYM_ID,
    ChessAdapter,
    ChessGymEnv,
    decode_chess_move,
    encode_chess_move,
    register_chess_gym,
)
from v8.environments.gym_adapter import GymDiscreteAdapter
from v8.environments.schemas import (
    ActionSchema,
    DiscreteActionCodec,
    DiscreteObservationCodec,
    EnvironmentIdentity,
    ObservationSchema,
)
from v8.environments.sudoku_env import (
    SUDOKU_ENV_ID,
    SudokuAdapter,
    decode_sudoku_action,
    encode_sudoku_action,
)

register_chess_gym()

__all__ = [
    "ActionSchema",
    "CHESS_GYM_ID",
    "ChessAdapter",
    "ChessGymEnv",
    "DiscreteActionCodec",
    "DiscreteObservationCodec",
    "EnvironmentIdentity",
    "GymDiscreteAdapter",
    "ObservationSchema",
    "SUDOKU_ENV_ID",
    "SudokuAdapter",
    "decode_chess_move",
    "decode_sudoku_action",
    "encode_chess_move",
    "encode_sudoku_action",
    "register_chess_gym",
]
