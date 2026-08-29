from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from v8.environment_contract import (
    BoundaryEvent,
    BoundaryScope,
    EnvironmentStepResult,
    EnvironmentTransition,
    WithinActionFrame,
    WithinActionTrace,
)
from v8.environments.schemas import ActionSchema, EnvironmentIdentity, ObservationSchema
from v8.model import stable_u64
from v8.structural_events import (
    MAX_NORMALIZED_FACTS_PER_EVENT,
    NormalizedPrimitive,
    StructuralFact,
)


CHESS_GYM_ID = "ArcAgi/Chess-v0"
CHESS_ACTION_COUNT = 64 * 64 * 5


def _imports():
    try:
        import chess
        import gymnasium as gym
    except ImportError as exc:  # pragma: no cover - dependency error is explicit
        raise RuntimeError("chess environment requires gymnasium and chess") from exc
    return chess, gym


def encode_chess_move(move) -> int:
    chess, _gym = _imports()
    promotion_codes = {
        None: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
    }
    if move.promotion not in promotion_codes:
        raise ValueError(f"unsupported promotion piece {move.promotion!r}")
    return ((int(move.from_square) * 64 + int(move.to_square)) * 5) + promotion_codes[move.promotion]


def decode_chess_move(token: int):
    chess, _gym = _imports()
    value = int(token)
    if not 0 <= value < CHESS_ACTION_COUNT:
        raise ValueError(f"chess action token {value} out of range")
    square_pair, promotion_code = divmod(value, 5)
    from_square, to_square = divmod(square_pair, 64)
    promotions = {
        0: None,
        1: chess.KNIGHT,
        2: chess.BISHOP,
        3: chess.ROOK,
        4: chess.QUEEN,
    }
    return chess.Move(from_square, to_square, promotion=promotions[promotion_code])


def _piece_code(piece) -> int:
    if piece is None:
        return 0
    base = int(piece.piece_type)
    return base if bool(piece.color) else base + 6


def _castling_mask(board) -> int:
    chess, _gym = _imports()
    return (
        int(board.has_kingside_castling_rights(chess.WHITE))
        | (int(board.has_queenside_castling_rights(chess.WHITE)) << 1)
        | (int(board.has_kingside_castling_rights(chess.BLACK)) << 2)
        | (int(board.has_queenside_castling_rights(chess.BLACK)) << 3)
    )


def _repetition_count(board) -> int:
    for count in (5, 4, 3, 2):
        if board.is_repetition(count):
            return count
    return 1


class ChessGymEnv:
    """Single-agent local chess environment with an automatic local opponent.

    One Gym step is one agent move plus, when the game continues, one opponent move.
    The intermediate board is returned in info so the v8 adapter can retain micro-time.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        *,
        opponent: str = "random",
        agent_color: str = "white",
        initial_fen: str | None = None,
    ) -> None:
        chess, gym = _imports()
        if opponent not in {"random", "first"}:
            raise ValueError("opponent must be 'random' or 'first'")
        if agent_color not in {"white", "black"}:
            raise ValueError("agent_color must be 'white' or 'black'")
        self.opponent = opponent
        self.agent_color = chess.WHITE if agent_color == "white" else chess.BLACK
        self.initial_fen = initial_fen
        self.action_space = gym.spaces.Discrete(CHESS_ACTION_COUNT)
        self.observation_space = gym.spaces.MultiDiscrete(
            np.asarray([13] * 64 + [2, 16, 65, 151, 512, 6], dtype=np.int64)
        )
        self.board = chess.Board(initial_fen) if initial_fen else chess.Board()
        self.np_random = np.random.default_rng()

    def _observation(self) -> np.ndarray:
        chess, _gym = _imports()
        values = [_piece_code(self.board.piece_at(square)) for square in chess.SQUARES]
        values.extend(
            [
                int(self.board.turn),
                _castling_mask(self.board),
                64 if self.board.ep_square is None else int(self.board.ep_square),
                min(150, max(0, int(self.board.halfmove_clock))),
                min(511, max(0, int(self.board.fullmove_number))),
                min(5, _repetition_count(self.board)),
            ]
        )
        return np.asarray(values, dtype=np.int64)

    def _outcome(self):
        return self.board.outcome(claim_draw=False)

    def _reward(self, outcome) -> float:
        if outcome is None or outcome.winner is None:
            return 0.0
        return 1.0 if bool(outcome.winner) == bool(self.agent_color) else -1.0

    def legal_action_tokens(self) -> tuple[int, ...]:
        if self._outcome() is not None or bool(self.board.turn) != bool(self.agent_color):
            return ()
        return tuple(sorted(encode_chess_move(move) for move in self.board.legal_moves))

    def _opponent_move(self) -> None:
        if self._outcome() is not None or bool(self.board.turn) == bool(self.agent_color):
            return
        moves = list(self.board.legal_moves)
        if not moves:
            return
        if self.opponent == "first":
            move = min(moves, key=lambda row: row.uci())
        else:
            move = moves[int(self.np_random.integers(0, len(moves)))]
        self.board.push(move)

    def reset(self, *, seed: int | None = None, options=None):
        del options
        if seed is not None:
            self.np_random = np.random.default_rng(int(seed))
        chess, _gym = _imports()
        self.board = chess.Board(self.initial_fen) if self.initial_fen else chess.Board()
        if bool(self.board.turn) != bool(self.agent_color) and self._outcome() is None:
            self._opponent_move()
        return self._observation(), {"legal_actions": self.legal_action_tokens()}

    def step(self, action: int):
        if self._outcome() is not None:
            raise RuntimeError("chess episode is terminal; call reset()")
        if bool(self.board.turn) != bool(self.agent_color):
            raise RuntimeError("chess environment is not at the agent decision point")
        move = decode_chess_move(action)
        if move not in self.board.legal_moves:
            raise ValueError(f"illegal chess move {move.uci()}")
        self.board.push(move)
        intermediate = self._observation()
        outcome = self._outcome()
        if outcome is None:
            self._opponent_move()
            outcome = self._outcome()
        observation = self._observation()
        terminated = outcome is not None
        reward = self._reward(outcome)
        info = {
            "legal_actions": self.legal_action_tokens(),
            "result": None if outcome is None else outcome.result(),
            "intermediate_observation": intermediate,
        }
        return observation, reward, terminated, False, info

    def render(self):
        return str(self.board)

    def close(self) -> None:
        return None


def register_chess_gym() -> None:
    _chess, gym = _imports()
    if CHESS_GYM_ID in gym.registry:
        return
    gym.register(
        id=CHESS_GYM_ID,
        entry_point="v8.environments.chess_env:ChessGymEnv",
        max_episode_steps=512,
    )


@dataclass(frozen=True, slots=True)
class ChessStepTelemetry:
    reward: float = 0.0
    terminated: bool = False
    result: str | None = None


class ChessAdapter:
    def __init__(
        self,
        *,
        seed: int = 0,
        opponent: str = "random",
        agent_color: str = "white",
        initial_fen: str | None = None,
    ) -> None:
        self.seed = int(seed)
        self.env = ChessGymEnv(
            opponent=opponent,
            agent_color=agent_color,
            initial_fen=initial_fen,
        )
        self.observation_schema = ObservationSchema("vector", "chess-board-v1")
        self.action_schema = ActionSchema("parameterized-discrete", "chess-move-v1")
        self.identity = EnvironmentIdentity(
            "chess",
            CHESS_GYM_ID,
            f"opponent={opponent},agent_color={agent_color}",
            f"seed={self.seed}",
        )
        self._episode = 0
        self._observation = np.zeros(70, dtype=np.int64)
        self._boundary = BoundaryEvent()
        self._last_trace: WithinActionTrace | None = None
        self._last_step_result: EnvironmentStepResult | None = None
        self._telemetry = ChessStepTelemetry()
        self.reset()

    @property
    def telemetry(self) -> ChessStepTelemetry:
        return self._telemetry

    def close(self) -> None:
        self.env.close()

    def observe(self) -> np.ndarray:
        return np.asarray(self._observation, dtype=np.int64).copy()

    def reset(self) -> np.ndarray:
        observation, _info = self.env.reset(seed=self.seed + self._episode)
        self._episode += 1
        self._observation = np.asarray(observation, dtype=np.int64)
        self._boundary = BoundaryEvent()
        self._last_trace = None
        self._last_step_result = None
        self._telemetry = ChessStepTelemetry()
        return self.observe()

    def available_actions(self) -> tuple[int, ...]:
        if not self._boundary.continuation:
            return ()
        return self.env.legal_action_tokens()

    def observation_signature(self, observation) -> int:
        array = np.asarray(observation, dtype=np.int64)
        return stable_u64(
            self.observation_schema.schema_id,
            array.tobytes(),
            person=b"v8.58-chess-observation",
        )

    def step(self, action: int) -> np.ndarray:
        before = self.observe()
        observation, reward, terminated, _truncated, info = self.env.step(int(action))
        self._observation = np.asarray(observation, dtype=np.int64)
        after = self.observe()
        if terminated:
            valence = 1 if reward > 0 else -1 if reward < 0 else 0
            self._boundary = BoundaryEvent(BoundaryScope.EPISODE, valence, False)
        else:
            self._boundary = BoundaryEvent()
        intermediate = np.asarray(info.get("intermediate_observation", after), dtype=np.int64)
        frames = [WithinActionFrame(intermediate.copy(), 0)]
        if not np.array_equal(intermediate, after):
            frames.append(WithinActionFrame(after.copy(), 1))
        trace = WithinActionTrace(before, tuple(frames), after)
        self._last_trace = trace
        self._last_step_result = EnvironmentStepResult(
            after,
            trace,
            self.available_actions(),
            self._boundary.primary_valence,
            self._boundary.scope,
            self._boundary.continuation,
        )
        self._telemetry = ChessStepTelemetry(float(reward), bool(terminated), info.get("result"))
        return after

    def cognitive_boundary_event(self) -> BoundaryEvent:
        return self._boundary

    def cognitive_context_signature(self) -> int:
        return self.observation_signature(self.observe())

    def cognitive_transition_signature(self, before, after) -> int:
        return stable_u64(
            self.observation_signature(before),
            self.observation_signature(after),
            person=b"v8.58-chess-transition",
        )

    def cognitive_family_signature(self, before, after) -> int:
        left = np.asarray(before, dtype=np.int64)[:64]
        right = np.asarray(after, dtype=np.int64)[:64]
        changed = int(np.count_nonzero(left != right))
        material_delta = int(np.count_nonzero(right) - np.count_nonzero(left))
        return stable_u64(changed, material_delta, person=b"v8.58-chess-family")

    def cognitive_changed_extent(self, before, after) -> int:
        return int(
            np.count_nonzero(
                np.asarray(before, dtype=np.int64) != np.asarray(after, dtype=np.int64)
            )
        )

    def cognitive_subepisode_index(self) -> int:
        return 0

    def cognitive_within_action_trace(self) -> WithinActionTrace | None:
        return self._last_trace

    def cognitive_step_result(self) -> EnvironmentStepResult | None:
        return self._last_step_result

    def normalized_fact_tokens(
        self,
        before,
        after,
        *,
        before_actions=(),
        after_actions=(),
    ) -> tuple[int, ...]:
        left = np.asarray(before, dtype=np.int64)[:64]
        right = np.asarray(after, dtype=np.int64)[:64]
        changed = int(np.count_nonzero(left != right))
        material_delta = int(np.count_nonzero(right) - np.count_nonzero(left))
        if changed == 0:
            kind = NormalizedPrimitive.NO_CHANGE
        elif material_delta < 0:
            kind = NormalizedPrimitive.COMPONENT_REMOVED
        elif material_delta == 0:
            kind = NormalizedPrimitive.COMPONENT_RELOCATED
        else:
            kind = NormalizedPrimitive.COMPONENT_ATTRIBUTE_CHANGED
        facts = [
            StructuralFact(
                kind,
                stable_u64(changed, person=b"v8.58-chess-change"),
                stable_u64(material_delta, person=b"v8.58-chess-material"),
                0,
                min(5, changed),
            )
        ]
        before_set = set(map(int, before_actions))
        after_set = set(map(int, after_actions))
        if after_set - before_set and len(facts) < MAX_NORMALIZED_FACTS_PER_EVENT:
            facts.append(
                StructuralFact(
                    NormalizedPrimitive.ACTION_BECAME_AVAILABLE,
                    stable_u64(len(before_set), len(after_set), person=b"v8.58-chess-actions"),
                    0,
                    0,
                    min(5, len(after_set - before_set)),
                )
            )
        if before_set - after_set and len(facts) < MAX_NORMALIZED_FACTS_PER_EVENT:
            facts.append(
                StructuralFact(
                    NormalizedPrimitive.ACTION_BECAME_UNAVAILABLE,
                    stable_u64(len(before_set), len(after_set), person=b"v8.58-chess-actions"),
                    0,
                    0,
                    min(5, len(before_set - after_set)),
                )
            )
        return tuple(fact.token for fact in facts[:MAX_NORMALIZED_FACTS_PER_EVENT])

    def cognitive_transition(
        self,
        *,
        before_observation,
        after_observation,
        action_token: int,
        available_actions_before,
        available_actions_after,
    ) -> EnvironmentTransition:
        before_context = self.observation_signature(before_observation)
        after_context = self.observation_signature(after_observation)
        changed = self.cognitive_changed_extent(before_observation, after_observation)
        return EnvironmentTransition(
            before_observation,
            after_observation,
            int(action_token),
            tuple(map(int, available_actions_before)),
            tuple(map(int, available_actions_after)),
            {
                "transition_signature": self.cognitive_transition_signature(
                    before_observation, after_observation
                ),
                "changed_extent": changed,
            },
            self._boundary,
            before_context,
            after_context,
            bool(changed),
            self._last_trace,
        )

    def cognitive_target_reached(self, target, outcome_uid=None) -> bool:
        del target, outcome_uid
        return bool(self._boundary.positive)
