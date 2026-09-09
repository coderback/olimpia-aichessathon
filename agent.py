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

# How much each piece contributes to hunting the enemy king, which is not the same as how
# much it contributes to the phase of the game. A lone queen is a mating attack; a lone
# rook is not, and against only minor pieces a king belongs in the centre.
THREAT_WEIGHT = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 3,
    chess.QUEEN: 6,
}
TOTAL_THREAT = 10


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

# how far a square is from the middle four, in king moves along ranks and files
CENTRE_DISTANCE = [
    max(3 - chess.square_file(square), chess.square_file(square) - 4, 0)
    + max(3 - chess.square_rank(square), chess.square_rank(square) - 4, 0)
    for square in range(64)
]
MOP_UP_PHASE = 6
MOP_UP_MARGIN = 400
EDGE_WEIGHT = 12
APPROACH_WEIGHT = 5


def evaluate(board: chess.Board) -> int:
    """Static score in centipawns, from the point of view of the side to move.

    This walks the bitboards directly. Going through board.piece_map() builds a Piece
    object per occupied square and cost about half of all search time.
    """
    white = board.occupied_co[chess.WHITE]
    black = board.occupied_co[chess.BLACK]
    balance = 0
    phase = 0
    white_threat = 0  # what white still has to attack the black king with
    black_threat = 0  # and what black has to attack white with

    bitboards = (board.pawns, board.knights, board.bishops, board.rooks, board.queens)
    for piece, bitboard in zip(SCORED, bitboards, strict=True):
        phase += PHASE_WEIGHT[piece] * bitboard.bit_count()
        threat = THREAT_WEIGHT[piece]
        table = WHITE_SCORE[piece]
        mine = bitboard & white
        white_threat += threat * mine.bit_count()
        while mine:
            balance += table[(mine & -mine).bit_length() - 1]
            mine &= mine - 1
        table = BLACK_SCORE[piece]
        theirs = bitboard & black
        black_threat += threat * theirs.bit_count()
        while theirs:
            balance -= table[(theirs & -theirs).bit_length() - 1]
            theirs &= theirs - 1

    # A king shelters while the OTHER side still has the pieces to hunt it, and walks to
    # the centre once they are gone. Blending on total material instead got this backwards:
    # trading our own pieces away lowered the phase, shifted our king towards the endgame
    # table, and paid it to march up the board while an enemy queen was still on. Four of
    # five rated losses were our king three to five ranks advanced with their queen alive.
    phase = min(phase, TOTAL_PHASE)
    king = board.king(chess.WHITE)
    if king is not None:
        danger = min(black_threat, TOTAL_THREAT)
        balance += (
            KING_MIDDLEGAME[king] * danger + KING_ENDGAME[king] * (TOTAL_THREAT - danger)
        ) // TOTAL_THREAT
    king = board.king(chess.BLACK)
    if king is not None:
        danger = min(white_threat, TOTAL_THREAT)
        balance -= (
            KING_MG_BLACK[king] * danger + KING_EG_BLACK[king] * (TOTAL_THREAT - danger)
        ) // TOTAL_THREAT

    # With a decisive edge and almost nothing left, material and placement give the search
    # no reason to make progress, so it shuffles until the game is drawn. Push the bare
    # king to the edge and walk our own king towards it.
    if phase <= MOP_UP_PHASE and not -MOP_UP_MARGIN <= balance <= MOP_UP_MARGIN:
        strong = chess.WHITE if balance > 0 else chess.BLACK
        winner = board.king(strong)
        loser = board.king(not strong)
        if winner is not None and loser is not None:
            drive = (
                CENTRE_DISTANCE[loser] * EDGE_WEIGHT
                + (14 - chess.square_manhattan_distance(winner, loser)) * APPROACH_WEIGHT
            )
            balance += drive if strong == chess.WHITE else -drive

    return balance if board.turn == chess.WHITE else -balance


# --- Search -----------------------------------------------------------------------

MATE = 30_000
MATE_THRESHOLD = MATE - 1_000
MAX_DEPTH = 64
MAX_PLY = 128
PASS_GROWTH = 1.5
FUTILITY_DEPTH = 2
FUTILITY_MARGIN = (0, 150, 300)
DELTA_MARGIN = 150
DELTA_MIN_PIECES = 8
LMR_MIN_DEPTH = 3
LMR_MIN_MOVE = 3
LMR_LATE_MOVE = 6
NULL_MIN_DEPTH = 3
NULL_REDUCTION = 2

INCREMENT_MS = 500
OVERHEAD_MS = 200.0
MIN_BUDGET_MS = 5.0
PANIC_MS = 100
# A node costs hundreds of microseconds and time.perf_counter() costs well under one, so
# checking often is nearly free. It bounds how far past the deadline we can run, which
# on the platform is slower per node than a dev machine: validation showed a 3.5 s move
# against a 3.1 s budget at CHECK_INTERVAL 128.
CHECK_INTERVAL = 16


EXACT, LOWER, UPPER = 0, 1, 2
TableEntry = tuple[int, int, int, "chess.Move | None"]


class TimeUp(Exception):
    """Raised inside the search when the move budget is gone."""


def _store_score(score: int, ply: int) -> int:
    """Make a mate score independent of where in the tree it was found."""
    if score >= MATE_THRESHOLD:
        return score + ply
    if score <= -MATE_THRESHOLD:
        return score - ply
    return score


def _load_score(score: int, ply: int) -> int:
    """Put a stored mate score back into this node, at this distance from the root."""
    if score >= MATE_THRESHOLD:
        return score - ply
    if score <= -MATE_THRESHOLD:
        return score + ply
    return score


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
        # quiet moves that caused a cutoff anywhere, scored by how deep the cutoff was
        self.quiet_history: list[list[int]] = [[0] * 64 for _ in range(64)]
        self.root_best: chess.Move | None = None
        # One table per move. It carries between deepening passes, which is where most of
        # the gain is, and starts empty each move so a stored score can never have been
        # computed under a shorter game history than the one we now have.
        self.table: dict[object, TableEntry] = {}

    def _tick(self) -> None:
        self.nodes += 1
        if self.nodes % CHECK_INTERVAL == 0 and time.perf_counter() >= self.deadline:
            raise TimeUp

    def ordered(self, board: chess.Board, ply: int, first: chess.Move | None) -> list[chess.Move]:
        killers = self.killers[ply] if ply < MAX_PLY else []
        # a direct bitboard test beats board.is_capture here; it misses en passant, which
        # only costs a little ordering on a rare move
        theirs = board.occupied_co[not board.turn]
        squares = chess.BB_SQUARES

        history = self.quiet_history

        def rank(move: chess.Move) -> int:
            if move == first:
                return 1_000_000
            if move.promotion is not None:
                return 500_000 + PIECE_VALUE[move.promotion]
            if squares[move.to_square] & theirs:
                return 100_000 + _mvv_lva(board, move)
            if move in killers:
                return 50_000
            # a quiet move that has cut elsewhere in this search is worth trying early
            return min(history[move.from_square][move.to_square], 49_000)

        return sorted(board.legal_moves, key=rank, reverse=True)

    def root(self, board: chess.Board, depth: int, first: chess.Move) -> tuple[int, chess.Move]:
        alpha = -MATE
        best = first
        principal = True
        self.root_best = None  # what this pass has proved so far, if it does not finish
        for move in self.ordered(board, 0, first):
            if time.perf_counter() >= self.deadline:
                raise TimeUp
            board.push(move)
            if principal:
                score = -self._negamax(board, depth - 1, -MATE, -alpha, 1)
            else:
                score = -self._negamax(board, depth - 1, -alpha - 1, -alpha, 1)
                if score > alpha:  # the narrow window was wrong, so pay for the real one
                    score = -self._negamax(board, depth - 1, -MATE, -alpha, 1)
            board.pop()
            principal = False
            if score > alpha:
                alpha, best = score, move
                self.root_best = move
        return alpha, best

    def _negamax(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        allow_null: bool = True,
    ) -> int:
        self._tick()

        key = _key(board)
        # a line back to a position from earlier in the game is a draw the referee claims
        if self.history and key in self.history:
            return 0
        if board.halfmove_clock >= 100:
            return 0
        # is_insufficient_material scans both sides, so only ask once a board is nearly bare
        if chess.popcount(board.occupied) <= 4 and board.is_insufficient_material():
            return 0

        entry = self.table.get(key)
        best_move = None if entry is None else entry[3]
        if entry is not None and entry[0] >= depth:
            score = _load_score(entry[1], ply)
            flag = entry[2]
            if flag == EXACT:
                return score
            if flag == LOWER:
                if score >= beta:
                    return beta
            elif score <= alpha:
                return alpha

        if depth <= 0:
            if not board.is_check():
                return self._quiesce(board, alpha, beta)
            depth = 1  # never score a position while in check

        # Null move: hand the opponent a free move. If the position still beats beta after
        # that, the real move list will almost certainly beat it too, so cut without
        # searching. Skipped in check, and skipped when the side to move has nothing but
        # pawns, because that is where zugzwang lives and having to move is a liability
        # rather than the free gift this assumes.
        if (
            allow_null
            and depth >= NULL_MIN_DEPTH
            and not board.is_check()
            and board.occupied_co[board.turn]
            & (board.knights | board.bishops | board.rooks | board.queens)
        ):
            board.push(chess.Move.null())
            score = -self._negamax(
                board, depth - 1 - NULL_REDUCTION, -beta, -beta + 1, ply + 1, False
            )
            board.pop()
            if score >= beta:
                return beta

        # the stored move is the best ordering hint there is, even at a shallower depth
        moves = self.ordered(board, ply, best_move)
        if not moves:
            return -MATE + ply if board.is_check() else 0

        found = None
        principal = True
        in_check = board.is_check()

        # Futility. Close to the leaves, a quiet move rarely rescues a position that is
        # already this far below alpha, so once the static score plus a generous margin
        # still cannot reach it, quiet moves are skipped. Never while in check, never on
        # a mate score, and never before at least one move has actually been searched.
        futile = False
        if not in_check and depth <= FUTILITY_DEPTH and abs(alpha) < MATE_THRESHOLD:
            futile = evaluate(board) + FUTILITY_MARGIN[depth] <= alpha

        searched = 0
        for index, move in enumerate(moves):
            quiet = move.promotion is None and not board.is_capture(move)
            board.push(move)
            if futile and quiet and searched and not board.is_check():
                board.pop()
                continue
            searched += 1

            # Late move reduction. The ordering already put the moves worth believing
            # first, so a quiet move this far down the list is unlikely to be best.
            # Search it shallower, and only pay full depth if it beats alpha anyway.
            # Never reduce out of check, into check, or a capture: those are the moves
            # a shallow search is most likely to misjudge.
            reduction = 0
            if (
                quiet
                and not principal
                and depth >= LMR_MIN_DEPTH
                and index >= LMR_MIN_MOVE
                and not in_check
                and not board.is_check()
            ):
                reduction = 1 if index < LMR_LATE_MOVE else 2

            if principal:
                score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            else:
                # Trust the ordering: assume everything after the first move fails low and
                # prove it with a one-point window, which is far cheaper. Only a move that
                # beats alpha costs a full re-search. In a window that is already one point
                # wide the probe is the real search, so nothing is repeated.
                score = -self._negamax(
                    board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1
                )
                if reduction and score > alpha:
                    # the reduction was wrong about this move, so pay the full depth
                    score = -self._negamax(board, depth - 1, -alpha - 1, -alpha, ply + 1)
                if alpha < score < beta:
                    score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            principal = False
            if score >= beta:
                if quiet:
                    self._remember(move, ply)
                    self.quiet_history[move.from_square][move.to_square] += depth * depth
                self.table[key] = (depth, _store_score(beta, ply), LOWER, move)
                return beta
            if score > alpha:
                alpha = score
                found = move
        self.table[key] = (
            depth,
            _store_score(alpha, ply),
            EXACT if found is not None else UPPER,
            found,
        )
        return alpha

    def _quiesce(self, board: chess.Board, alpha: int, beta: int) -> int:
        self._tick()
        stand_pat = evaluate(board)
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)

        captures = sorted(board.generate_legal_captures(), key=lambda m: _mvv_lva(board, m))
        for move in reversed(captures):
            # Delta pruning. Take the most optimistic view of this capture, that the piece
            # is won outright for nothing, and if even that does not reach alpha there is
            # no point searching it. Skipped once the board is nearly bare, where a single
            # capture swings the game and the optimistic view is not optimistic enough.
            if board.occupied.bit_count() > DELTA_MIN_PIECES:
                victim = board.piece_type_at(move.to_square)
                gain = PIECE_VALUE[victim] if victim is not None else PIECE_VALUE[chess.PAWN]
                if move.promotion is not None:
                    gain += PIECE_VALUE[move.promotion]
                if stand_pat + gain + DELTA_MARGIN <= alpha:
                    continue
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
    started = time.perf_counter()
    search = Search(started + budget_s, _history)

    moves = search.ordered(board, 0, None)
    if not moves:
        return "0000"  # the referee ends the game before asking, so this is only a guard
    best = moves[0]

    # Out of clock. One search pass costs about 16 ms however small the budget says it is,
    # because a board, a move list and one ply have to happen at all, and the referee flags
    # the moment the elapsed time exceeds the clock rather than the budget. Ordering alone
    # costs a fraction of a millisecond, so hand back its first move and keep the game.
    if time_left_ms < PANIC_MS:
        return best.uci()

    pass_cost = 0.0
    for depth in range(1, MAX_DEPTH + 1):
        # Each pass costs a multiple of the one before it. Predicting the next one from the
        # last one measured adapts to the position, where a fixed fraction of the budget
        # stopped early in quiet positions and still overcommitted in sharp ones.
        elapsed = time.perf_counter() - started
        if pass_cost and elapsed + pass_cost * PASS_GROWTH > budget_s:
            break
        pass_started = time.perf_counter()
        try:
            score, move = search.root(board, depth, best)
        except TimeUp:
            # A root move that already beat every move before it at this depth was proved
            # by a complete search of that move, so it is a better answer than the one the
            # last finished pass returned. An unfinished pass is not a wasted one.
            if search.root_best is not None:
                best = search.root_best
            break
        pass_cost = time.perf_counter() - pass_started
        best = move
        if abs(score) >= MATE_THRESHOLD:
            break

    return best.uci()


# Warm the search once at import so the first move on the clock pays no start-up cost.
# Import time has a 90 second budget; a move does not.
_warm = Search(time.perf_counter() + 1.0, set())
with contextlib.suppress(TimeUp):
    _warm.root(chess.Board(), 2, chess.Move.from_uci("e2e4"))
