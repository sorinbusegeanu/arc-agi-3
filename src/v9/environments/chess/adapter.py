from __future__ import annotations

from typing import Any

from v9.environments.base import StructuralAdapter
from v9.environments.contract import BoundaryEvent, BoundaryScope, WithinActionFrame, WithinActionTrace
from v9.environments.schemas import ActionSchema, EnvironmentIdentity, ObservationSchema

ACTION_COUNT = 64 * 64 * 5


def _chess():
    try:
        import chess
    except ImportError as exc:
        raise RuntimeError("ChessAdapter requires the chess dependency") from exc
    return chess


def encode_move(move: Any) -> int:
    chess = _chess()
    promotions = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}
    return ((int(move.from_square) * 64 + int(move.to_square)) * 5) + promotions[move.promotion]


def decode_move(value: int) -> Any:
    chess = _chess()
    if not 0 <= int(value) < ACTION_COUNT:
        raise ValueError("chess action is out of range")
    pair, promotion = divmod(int(value), 5)
    source, target = divmod(pair, 64)
    return chess.Move(source, target, promotion={0: None, 1: chess.KNIGHT, 2: chess.BISHOP, 3: chess.ROOK, 4: chess.QUEEN}[promotion])


class ChessAdapter(StructuralAdapter):
    def __init__(self, *, seed: int = 0, opponent: str = "first", agent_color: str = "white", initial_fen: str | None = None) -> None:
        chess = _chess()
        if opponent not in {"first", "random"} or agent_color not in {"white", "black"}:
            raise ValueError("invalid chess opponent or agent color")
        self.seed, self.opponent = int(seed), opponent
        self.agent_color = chess.WHITE if agent_color == "white" else chess.BLACK
        self.initial_fen = initial_fen
        self._identity = EnvironmentIdentity("chess", "Chess-v0", f"opponent={opponent},color={agent_color}", f"seed={seed}")
        self._observation_schema = ObservationSchema("vector", "python-chess-board")
        self._action_schema = ActionSchema("parameterized-discrete", f"n={ACTION_COUNT}")
        self.board = chess.Board(initial_fen) if initial_fen else chess.Board()
        self._boundary = BoundaryEvent()
        self._last_trace = None
        self._rng_state = self.seed
        self.reset()

    def _observation(self) -> tuple[int, ...]:
        chess = _chess()
        pieces = []
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            pieces.append(0 if piece is None else int(piece.piece_type) + (0 if piece.color else 6))
        return tuple(pieces + [int(self.board.turn), int(self.board.halfmove_clock), int(self.board.fullmove_number)])

    def reset(self) -> tuple[int, ...]:
        chess = _chess()
        self.board = chess.Board(self.initial_fen) if self.initial_fen else chess.Board()
        self._boundary = BoundaryEvent()
        self._last_trace = None
        if self.board.turn != self.agent_color:
            self._opponent_move()
        return self.observe()

    def observe(self) -> tuple[int, ...]:
        return self._observation()

    def available_actions(self) -> tuple[int, ...]:
        if self.board.outcome(claim_draw=False) is not None or self.board.turn != self.agent_color:
            return ()
        return tuple(sorted(encode_move(move) for move in self.board.legal_moves))

    def _opponent_move(self) -> None:
        moves = sorted(self.board.legal_moves, key=lambda move: move.uci())
        if not moves:
            return
        if self.opponent == "random":
            self._rng_state = (1103515245 * self._rng_state + 12345) & 0x7FFFFFFF
            move = moves[self._rng_state % len(moves)]
        else:
            move = moves[0]
        self.board.push(move)

    def step(self, native_action: Any) -> tuple[int, ...]:
        before = self.observe()
        move = decode_move(int(native_action))
        if move not in self.board.legal_moves:
            raise ValueError("illegal target-local chess action")
        self.board.push(move)
        intermediate = self.observe()
        if self.board.outcome(claim_draw=False) is None:
            self._opponent_move()
        after = self.observe()
        outcome = self.board.outcome(claim_draw=False)
        if outcome is None:
            self._boundary = BoundaryEvent()
        else:
            valence = 0 if outcome.winner is None else (1 if outcome.winner == self.agent_color else -1)
            self._boundary = BoundaryEvent(BoundaryScope.EPISODE, valence, False)
        frames = (WithinActionFrame(intermediate, 0), WithinActionFrame(after, 1))
        self._last_trace = WithinActionTrace(before, frames, after)
        return after

