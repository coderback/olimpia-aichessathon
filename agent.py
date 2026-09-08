"""The submission entrypoint. The platform imports this file and calls get_move.

An alpha-beta search over a material and piece-square evaluation:

  iterative deepening   so there is always a move to return when the clock runs out
  MVV-LVA ordering      alpha-beta only pays off when good moves come first
  quiescence            leaves are never scored in the middle of an exchange
  killer moves          quiet refutations tried early at the same ply

The process is suspended while the opponent moves, so nothing runs between our own moves
and there is no pondering to do. State that survives within a game is used for one thing
only: remembering the positions we have already been asked about, so the search knows
when a line repeats one of them.
"""

import contextlib
import time

import chess

# --- Evaluation -------------------------------------------------------------------

PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}
PHASE_WEIGHT = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
}
TOTAL_PHASE = 24


def _table(rows: str) -> list[int]:
    """Read a table written rank 8 first into python-chess square order, where a1 is 0."""
    values = [int(value) for value in rows.split()]
    if len(values) != 64:
        raise ValueError(f"a piece-square table needs 64 entries, got {len(values)}")
    return [values[(7 - rank) * 8 + file] for rank in range(8) for file in range(8)]


PAWN_TABLE = _table("""
     0   0   0   0   0   0   0   0
    50  50  50  50  50  50  50  50
    10  10  20  30  30  20  10  10
     5   5  10  25  25  10   5   5
     0   0   0  20  20   0   0   0
     5  -5 -10   0   0 -10  -5   5
     5  10  10 -20 -20  10  10   5
     0   0   0   0   0   0   0   0
""")
KNIGHT_TABLE = _table("""
   -50 -40 -30 -30 -30 -30 -40 -50
   -40 -20   0   0   0   0 -20 -40
   -30   0  10  15  15  10   0 -30
   -30   5  15  20  20  15   5 -30
   -30   0  15  20  20  15   0 -30
   -30   5  10  15  15  10   5 -30
   -40 -20   0   5   5   0 -20 -40
   -50 -40 -30 -30 -30 -30 -40 -50
""")
BISHOP_TABLE = _table("""
   -20 -10 -10 -10 -10 -10 -10 -20
   -10   0   0   0   0   0   0 -10
   -10   0   5  10  10   5   0 -10
   -10   5   5  10  10   5   5 -10
   -10   0  10  10  10  10   0 -10
   -10  10  10  10  10  10  10 -10
   -10   5   0   0   0   0   5 -10
   -20 -10 -10 -10 -10 -10 -10 -20
""")
ROOK_TABLE = _table("""
     0   0   0   0   0   0   0   0
     5  10  10  10  10  10  10   5
    -5   0   0   0   0   0   0  -5
    -5   0   0   0   0   0   0  -5
    -5   0   0   0   0   0   0  -5
    -5   0   0   0   0   0   0  -5
    -5   0   0   0   0   0   0  -5
     0   0   0   5   5   0   0   0
""")
QUEEN_TABLE = _table("""
   -20 -10 -10  -5  -5 -10 -10 -20
   -10   0   0   0   0   0   0 -10
   -10   0   5   5   5   5   0 -10
    -5   0   5   5   5   5   0  -5
     0   0   5   5   5   5   0  -5
   -10   5   5   5   5   5   0 -10
   -10   0   5   0   0   0   0 -10
   -20 -10 -10  -5  -5 -10 -10 -20
""")
KING_MIDDLEGAME = _table("""
   -30 -40 -40 -50 -50 -40 -40 -30
   -30 -40 -40 -50 -50 -40 -40 -30
   -30 -40 -40 -50 -50 -40 -40 -30
   -30 -40 -40 -50 -50 -40 -40 -30
   -20 -30 -30 -40 -40 -30 -30 -20
   -10 -20 -20 -20 -20 -20 -20 -10
    20  20   0   0   0   0  20  20
    20  30  10   0   0  10  30  20
""")
KING_ENDGAME = _table("""
   -50 -40 -30 -20 -20 -30 -40 -50
   -30 -20 -10   0   0 -10 -20 -30
   -30 -10  20  30  30  20 -10 -30
   -30 -10  30  40  40  30 -10 -30
   -30 -10  30  40  40  30 -10 -30
   -30 -10  20  30  30  20 -10 -30
   -30 -30   0   0   0   0 -30 -30
   -50 -30 -30 -30 -30 -30 -30 -50
""")
PIECE_SQUARE = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE,
    chess.QUEEN: QUEEN_TABLE,
}
SCORED = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
MIRROR = [chess.square_mirror(square) for square in range(64)]

# value and placement folded into one lookup per piece, already flipped for black
WHITE_SCORE = [
    [PIECE_VALUE[piece] + PIECE_SQUARE[piece][square] for square in range(64)]
    if piece in PIECE_SQUARE
    else [0] * 64
    for piece in range(7)
]
BLACK_SCORE = [
    [PIECE_VALUE[piece] + PIECE_SQUARE[piece][MIRROR[square]] for square in range(64)]
    if piece in PIECE_SQUARE
    else [0] * 64
    for piece in range(7)
]
KING_MG_BLACK = [KING_MIDDLEGAME[MIRROR[square]] for square in range(64)]
KING_EG_BLACK = [KING_ENDGAME[MIRROR[square]] for square in range(64)]


def evaluate(board: chess.Board) -> int:
    """Static score in centipawns, from the point of view of the side to move.

    This walks the bitboards directly. Going through board.piece_map() builds a Piece
    object per occupied square and cost about half of all search time.
    """
    white = board.occupied_co[chess.WHITE]
    black = board.occupied_co[chess.BLACK]
    balance = 0
    phase = 0

    bitboards = (board.pawns, board.knights, board.bishops, board.rooks, board.queens)
    for piece, bitboard in zip(SCORED, bitboards, strict=True):
        phase += PHASE_WEIGHT[piece] * bitboard.bit_count()
        table = WHITE_SCORE[piece]
        mine = bitboard & white
        while mine:
            balance += table[(mine & -mine).bit_length() - 1]
            mine &= mine - 1
        table = BLACK_SCORE[piece]
        theirs = bitboard & black
        while theirs:
            balance -= table[(theirs & -theirs).bit_length() - 1]
            theirs &= theirs - 1

    # The king wants shelter while the queens are on and the centre once they are gone.
    # The blend is integer so the score never depends on float rounding.
    phase = min(phase, TOTAL_PHASE)
    king = board.king(chess.WHITE)
    if king is not None:
        balance += (
            KING_MIDDLEGAME[king] * phase + KING_ENDGAME[king] * (TOTAL_PHASE - phase)
        ) // TOTAL_PHASE
    king = board.king(chess.BLACK)
    if king is not None:
        balance -= (
            KING_MG_BLACK[king] * phase + KING_EG_BLACK[king] * (TOTAL_PHASE - phase)
        ) // TOTAL_PHASE

    return balance if board.turn == chess.WHITE else -balance


# --- Search -----------------------------------------------------------------------

MATE = 30_000
MATE_THRESHOLD = MATE - 1_000
MAX_DEPTH = 64
MAX_PLY = 128

INCREMENT_MS = 500
OVERHEAD_MS = 200.0
MIN_BUDGET_MS = 5.0
# A node costs hundreds of microseconds and time.monotonic() costs well under one, so
# checking often is nearly free. It bounds how far past the deadline we can run, which
# on the platform is slower per node than a dev machine: validation showed a 3.5 s move
# against a 3.1 s budget at CHECK_INTERVAL 128.
CHECK_INTERVAL = 16


class TimeUp(Exception):
    """Raised inside the search when the move budget is gone."""


def _mvv_lva(board: chess.Board, move: chess.Move) -> int:
    """Most valuable victim, least valuable attacker."""
    victim = board.piece_type_at(move.to_square)
    attacker = board.piece_type_at(move.from_square)
    victim_value = PIECE_VALUE[victim] if victim is not None else PIECE_VALUE[chess.PAWN]
    attacker_value = PIECE_VALUE[attacker] if attacker is not None else 0
    return victim_value * 16 - attacker_value


def _key(board: chess.Board) -> object:
    """Identity of the position, ignoring the move counters, as docs/IDEAS.md suggests."""
    return board._transposition_key()


class Search:
    """One search, bounded by a wall-clock deadline."""

    def __init__(self, deadline: float, history: set[object]) -> None:
        self.deadline = deadline
        self.history = history
        self.nodes = 0
        self.killers: list[list[chess.Move]] = [[] for _ in range(MAX_PLY)]

    def _tick(self) -> None:
        self.nodes += 1
        if self.nodes % CHECK_INTERVAL == 0 and time.monotonic() >= self.deadline:
            raise TimeUp

    def ordered(self, board: chess.Board, ply: int, first: chess.Move | None) -> list[chess.Move]:
        killers = self.killers[ply] if ply < MAX_PLY else []
        # a direct bitboard test beats board.is_capture here; it misses en passant, which
        # only costs a little ordering on a rare move
        theirs = board.occupied_co[not board.turn]
        squares = chess.BB_SQUARES

        def rank(move: chess.Move) -> int:
            if move == first:
                return 1_000_000
            if move.promotion is not None:
                return 500_000 + PIECE_VALUE[move.promotion]
            if squares[move.to_square] & theirs:
                return 100_000 + _mvv_lva(board, move)
            if move in killers:
                return 50_000
            return 0

        return sorted(board.legal_moves, key=rank, reverse=True)

    def root(self, board: chess.Board, depth: int, first: chess.Move) -> tuple[int, chess.Move]:
        alpha = -MATE
        best = first
        for move in self.ordered(board, 0, first):
            if time.monotonic() >= self.deadline:
                raise TimeUp
            board.push(move)
            score = -self._negamax(board, depth - 1, -MATE, -alpha, 1)
            board.pop()
            if score > alpha:
                alpha, best = score, move
        return alpha, best

    def _negamax(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
        self._tick()

        # a line back to a position from earlier in the game is a draw the referee claims
        if self.history and _key(board) in self.history:
            return 0
        if board.halfmove_clock >= 100:
            return 0
        # is_insufficient_material scans both sides, so only ask once a board is nearly bare
        if chess.popcount(board.occupied) <= 4 and board.is_insufficient_material():
            return 0

        if depth <= 0:
            if not board.is_check():
                return self._quiesce(board, alpha, beta)
            depth = 1  # never score a position while in check

        moves = self.ordered(board, ply, None)
        if not moves:
            return -MATE + ply if board.is_check() else 0

        for move in moves:
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            if score >= beta:
                # only ask about the capture on a cutoff; asking for every move costs more
                if not board.is_capture(move):
                    self._remember(move, ply)
                return beta
            alpha = max(alpha, score)
        return alpha

    def _quiesce(self, board: chess.Board, alpha: int, beta: int) -> int:
        self._tick()
        stand_pat = evaluate(board)
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)

        captures = sorted(board.generate_legal_captures(), key=lambda m: _mvv_lva(board, m))
        for move in reversed(captures):
            board.push(move)
            score = -self._quiesce(board, -beta, -alpha)
            board.pop()
            if score >= beta:
                return beta
            alpha = max(alpha, score)
        return alpha

    def _remember(self, move: chess.Move, ply: int) -> None:
        if ply >= MAX_PLY:
            return
        slot = self.killers[ply]
        if move not in slot:
            slot.insert(0, move)
            del slot[2:]


# --- Time management --------------------------------------------------------------


def _budget_ms(time_left_ms: int, board: chess.Board) -> float:
    """How long this move may take. A flag is the most common self-inflicted loss.

    get_move is told the clock but never the increment, so the platform's 0.5s is a
    constant here. A local arena run at a different increment will be a little off.
    """
    moves_left = max(18, 46 - board.fullmove_number)
    budget = time_left_ms / moves_left + INCREMENT_MS * 0.6
    budget = min(budget, time_left_ms * 0.35)
    return max(budget - OVERHEAD_MS, MIN_BUDGET_MS)


# --- Entry point ------------------------------------------------------------------

_history: set[object] = set()


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation."""
    board = chess.Board(fen)
    _history.add(_key(board))

    budget_s = _budget_ms(time_left_ms, board) / 1000.0
    started = time.monotonic()
    search = Search(started + budget_s, _history)

    moves = search.ordered(board, 0, None)
    if not moves:
        return "0000"  # the referee ends the game before asking, so this is only a guard
    best = moves[0]

    for depth in range(1, MAX_DEPTH + 1):
        # each depth costs several times the last, so do not open one we cannot finish
        if time.monotonic() - started > budget_s * 0.45:
            break
        try:
            score, move = search.root(board, depth, best)
        except TimeUp:
            break
        best = move
        if abs(score) >= MATE_THRESHOLD:
            break

    return best.uci()


# Warm the search once at import so the first move on the clock pays no start-up cost.
# Import time has a 90 second budget; a move does not.
_warm = Search(time.monotonic() + 1.0, set())
with contextlib.suppress(TimeUp):
    _warm.root(chess.Board(), 2, chess.Move.from_uci("e2e4"))
