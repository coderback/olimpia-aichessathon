"""The submission entrypoint. The platform imports this file and calls get_move.

An alpha-beta search over a material and piece-square evaluation, with the board, the
move generator and the search itself compiled by numba:

  bitboards + magic lookups   sliding attacks are one multiply and one table read
  iterative deepening         so there is always a move to return when the clock runs out
  transposition table         kept for the whole game, replaced on collision
  MVV-LVA, killers, history   alpha-beta only pays off when good moves come first
  principal variation search  narrow window after the first move
  null move, LMR, futility    search less where the ordering says less is needed
  quiescence + delta pruning  leaves are never scored in the middle of an exchange

The pure Python version of this engine reached about 13,000 nodes a second; python-chess
is not built for engine use. Compiling at import spends the numba build inside the 90
second init budget instead of on the clock. Should the compiled path ever fail, a
python-chess search of the same shape takes over so a move is always returned.

The process is suspended while the opponent moves, so there is no pondering. State that
survives within a game: the transposition table, and the hashes of the positions already
played, so a line that returns to one of them scores as the draw the referee will claim.
"""

# mypy: disable-error-code="call-arg, var-annotated, no-untyped-call"
# The compiled functions take numba array types mypy cannot follow; everything else is checked.

import contextlib
import math
import time

import chess
import numpy as np
from llvmlite import ir  # type: ignore[import-untyped]
from numba import boolean, float64, int32, int64, njit, objmode, uint64, void
from numba.core import types
from numba.extending import intrinsic

# --- Bit tricks ---------------------------------------------------------------------
# numba has no bit-scan or popcount; these bind the LLVM intrinsics directly.


@intrinsic
def ctz(typingctx, x):  # type: ignore[no-untyped-def]
    """Index of the lowest set bit. 64 when nothing is set."""
    sig = types.int64(types.uint64)

    def codegen(context, builder, signature, args):  # type: ignore[no-untyped-def]
        i64, i1 = ir.IntType(64), ir.IntType(1)
        fn = builder.module.declare_intrinsic(
            "llvm.cttz", [i64], ir.FunctionType(i64, [i64, i1])
        )
        return builder.call(fn, [args[0], ir.Constant(i1, 0)])

    return sig, codegen


@intrinsic
def popcount(typingctx, x):  # type: ignore[no-untyped-def]
    sig = types.int64(types.uint64)

    def codegen(context, builder, signature, args):  # type: ignore[no-untyped-def]
        i64 = ir.IntType(64)
        fn = builder.module.declare_intrinsic("llvm.ctpop", [i64], ir.FunctionType(i64, [i64]))
        return builder.call(fn, [args[0]])

    return sig, codegen


# Every compiled function declares its signature. Without one, numba compiles a fresh
# copy for each distinct literal argument at each call site, which made the import take
# three times as long as it needed to.
U64 = uint64[::1]
I64 = int64[::1]
U64_2D = uint64[:, ::1]
I32_2D = int32[:, ::1]
SEARCH_ARGS = (U64, I64, U64_2D, I32_2D, U64, I64)

# --- Board constants ----------------------------------------------------------------
# Squares are a1 = 0 ... h8 = 63, the same order python-chess uses. Piece codes are
# white pawn 0 ... white king 5, black pawn 6 ... black king 11, empty 12.

PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 0, 1, 2, 3, 4, 5
WHITE, BLACK = 0, 1
EMPTY = 12
OCC_WHITE, OCC_BLACK, OCC_ALL, HASH = 12, 13, 14, 15  # slots after the 12 piece bitboards

U0 = np.uint64(0)
U1 = np.uint64(1)
U8 = np.uint64(8)
UFULL = np.uint64(0xFFFFFFFFFFFFFFFF)

BIT = np.array([1 << s for s in range(64)], dtype=np.uint64)
FILE_A = np.uint64(0x0101010101010101)
FILE_H = np.uint64(0x8080808080808080)
RANK_1 = np.uint64(0x00000000000000FF)
RANK_3 = np.uint64(0x0000000000FF0000)
RANK_6 = np.uint64(0x0000FF0000000000)
RANK_8 = np.uint64(0xFF00000000000000)

# Castling rights: 1 white kingside, 2 white queenside, 4 black kingside, 8 black queenside.
# A move from or to one of these squares removes the rights it touches.
CASTLE_MASK = np.full(64, 15, dtype=np.int64)
CASTLE_MASK[0], CASTLE_MASK[4], CASTLE_MASK[7] = 13, 12, 14
CASTLE_MASK[56], CASTLE_MASK[60], CASTLE_MASK[63] = 7, 3, 11

# A move is from | to << 6 | promotion piece << 12 | flags. The promotion field holds the
# piece type, so knight = 1 ... queen = 4 and zero means none.
PROMO_SHIFT = 12
PROMO_MASK = 7 << PROMO_SHIFT
F_CAPTURE = 1 << 15
F_EP = 1 << 16
F_CASTLE = 1 << 17
F_DOUBLE = 1 << 18
TACTICAL = F_CAPTURE | PROMO_MASK


def _leaper(square: int, steps: list[tuple[int, int]]) -> int:
    file, rank = square & 7, square >> 3
    mask = 0
    for df, dr in steps:
        f, r = file + df, rank + dr
        if 0 <= f < 8 and 0 <= r < 8:
            mask |= 1 << (r * 8 + f)
    return mask


KNIGHT_ATT = np.array(
    [
        _leaper(s, [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)])
        for s in range(64)
    ],
    dtype=np.uint64,
)
KING_ATT = np.array(
    [
        _leaper(s, [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)])
        for s in range(64)
    ],
    dtype=np.uint64,
)
# squares a pawn of each colour attacks from a square
PAWN_ATT = np.array(
    [
        [_leaper(s, [(-1, 1), (1, 1)]) for s in range(64)],
        [_leaper(s, [(-1, -1), (1, -1)]) for s in range(64)],
    ],
    dtype=np.uint64,
)

# --- Magic bitboards ----------------------------------------------------------------
# A sliding piece's attacks depend only on the occupied squares along its rays. Those
# squares, masked and multiplied by a per-square magic number, hash to a table index that
# holds the attack set. The magics are found at import by random trial, which takes well
# under a second once compiled.

_ROOK_STEPS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
_BISHOP_STEPS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]


def _slider_mask(square: int, bishop: bool) -> int:
    """Ray squares whose occupancy matters. The board edge never blocks anything further."""
    file, rank = square & 7, square >> 3
    mask = 0
    for df, dr in _BISHOP_STEPS if bishop else _ROOK_STEPS:
        f, r = file + df, rank + dr
        while 0 <= f < 8 and 0 <= r < 8:
            # the last square along a ray is an edge square, and it never blocks anything
            if not (0 <= f + df < 8 and 0 <= r + dr < 8):
                break
            mask |= 1 << (r * 8 + f)
            f, r = f + df, r + dr
    return mask


def _slider_attacks(square: int, occupied: int, bishop: bool) -> int:
    file, rank = square & 7, square >> 3
    attacks = 0
    for df, dr in _BISHOP_STEPS if bishop else _ROOK_STEPS:
        f, r = file + df, rank + dr
        while 0 <= f < 8 and 0 <= r < 8:
            bit = 1 << (r * 8 + f)
            attacks |= bit
            if occupied & bit:
                break
            f, r = f + df, r + dr
    return attacks


@njit(cache=False)
def _find_magic(mask, shift, occupancies, references, table, offset, seed):  # type: ignore[no-untyped-def]
    size = occupancies.shape[0]
    used = np.zeros(size, dtype=np.uint64)
    while True:
        # three xorshift draws ANDed together give the sparse magic that tends to work
        magic = UFULL
        for _ in range(3):
            seed ^= seed << uint64(13)
            seed ^= seed >> uint64(7)
            seed ^= seed << uint64(17)
            magic &= seed
        if popcount((mask * magic) >> uint64(56)) < 6:
            continue
        for i in range(size):
            used[i] = U0
        ok = True
        for i in range(size):
            index = (occupancies[i] * magic) >> shift
            if used[index] == U0:
                used[index] = references[i]
            elif used[index] != references[i]:
                ok = False
                break
        if ok:
            for i in range(size):
                table[offset + uint64(i)] = used[i]
            return magic, seed


def _build_magics(bishop: bool) -> tuple[np.ndarray, ...]:
    masks = np.array([_slider_mask(s, bishop) for s in range(64)], dtype=np.uint64)
    bits = np.array([int(m).bit_count() for m in masks], dtype=np.int64)
    shifts = (64 - bits).astype(np.uint64)
    sizes = 1 << bits
    offsets = np.zeros(64, dtype=np.uint64)
    offsets[1:] = np.cumsum(sizes)[:-1]
    table = np.zeros(int(sizes.sum()), dtype=np.uint64)
    magics = np.zeros(64, dtype=np.uint64)
    seed = np.uint64(0x9E3779B97F4A7C15 if bishop else 0xD1B54A32D192ED03)
    for square in range(64):
        # every occupancy subset of the mask, with its true attack set, for the trial to check
        mask = int(masks[square])
        subsets = []
        subset = 0
        while True:
            subsets.append(subset)
            subset = (subset - mask) & mask
            if subset == 0:
                break
        occupancies = np.array(subsets, dtype=np.uint64)
        references = np.array(
            [_slider_attacks(square, o, bishop) for o in subsets], dtype=np.uint64
        )
        magic, seed = _find_magic(
            masks[square], shifts[square], occupancies, references, table, offsets[square], seed
        )
        magics[square] = magic
        seed = np.uint64(seed)  # compiled code hands back a Python int; keep the type stable
    return masks, magics, shifts, offsets, table


ROOK_MASK, ROOK_MAGIC, ROOK_SHIFT, ROOK_OFFSET, ROOK_ATTACKS = _build_magics(False)
BISHOP_MASK, BISHOP_MAGIC, BISHOP_SHIFT, BISHOP_OFFSET, BISHOP_ATTACKS = _build_magics(True)


@njit(cache=False)
def rook_attacks(square, occupied):  # type: ignore[no-untyped-def]
    index = ((occupied & ROOK_MASK[square]) * ROOK_MAGIC[square]) >> ROOK_SHIFT[square]
    return ROOK_ATTACKS[ROOK_OFFSET[square] + index]


@njit(cache=False)
def bishop_attacks(square, occupied):  # type: ignore[no-untyped-def]
    index = ((occupied & BISHOP_MASK[square]) * BISHOP_MAGIC[square]) >> BISHOP_SHIFT[square]
    return BISHOP_ATTACKS[BISHOP_OFFSET[square] + index]


# --- Zobrist hashing ----------------------------------------------------------------

_rng = np.random.default_rng(20240907)
_U64_MAX = np.iinfo(np.uint64).max
Z_PIECE = _rng.integers(0, _U64_MAX, size=(13, 64), dtype=np.uint64, endpoint=True)
Z_CASTLE = _rng.integers(0, _U64_MAX, size=16, dtype=np.uint64, endpoint=True)
Z_EP = _rng.integers(0, _U64_MAX, size=8, dtype=np.uint64, endpoint=True)
Z_SIDE = np.uint64(_rng.integers(0, _U64_MAX, dtype=np.uint64, endpoint=True))

# --- Evaluation tables --------------------------------------------------------------

PIECE_VALUE = np.array([100, 320, 330, 500, 900, 0], dtype=np.int64)
PHASE_WEIGHT = np.array([0, 1, 1, 2, 4, 0], dtype=np.int64)
TOTAL_PHASE = 24


def _table(rows: str) -> list[int]:
    """Read a table written rank 8 first into square order, where a1 is 0."""
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

MIRROR = [chess.square_mirror(square) for square in range(64)]
_TABLES = (PAWN_TABLE, KNIGHT_TABLE, BISHOP_TABLE, ROOK_TABLE, QUEEN_TABLE)

# value and placement folded into one lookup per piece code, already flipped for black
PST = np.zeros((13, 64), dtype=np.int64)
for _piece, _placement in enumerate(_TABLES):
    for _square in range(64):
        PST[_piece, _square] = int(PIECE_VALUE[_piece]) + _placement[_square]
        PST[6 + _piece, _square] = int(PIECE_VALUE[_piece]) + _placement[MIRROR[_square]]
KING_MG = np.array(
    [KING_MIDDLEGAME, [KING_MIDDLEGAME[MIRROR[s]] for s in range(64)]], dtype=np.int64
)
KING_EG = np.array([KING_ENDGAME, [KING_ENDGAME[MIRROR[s]] for s in range(64)]], dtype=np.int64)

# how far a square is from the middle four, in king moves along ranks and files
CENTRE_DISTANCE = np.array(
    [max(3 - (s & 7), (s & 7) - 4, 0) + max(3 - (s >> 3), (s >> 3) - 4, 0) for s in range(64)],
    dtype=np.int64,
)
MOP_UP_PHASE = 6
MOP_UP_MARGIN = 400
EDGE_WEIGHT = 12
APPROACH_WEIGHT = 5

# Terms beyond material and placement, each with a middlegame and an endgame weight. The
# values are about half of what the textbooks give: the first attempt at these, on the
# slow engine, lost 102 elo, and a search that sees this deep needs less help from them.
FILE_MASK = np.array([FILE_A << np.uint64(f) for f in range(8)], dtype=np.uint64)
NEIGHBOUR_FILES = np.array(
    [
        (FILE_MASK[f - 1] if f > 0 else 0) | (FILE_MASK[f + 1] if f < 7 else 0)
        for f in range(8)
    ],
    dtype=np.uint64,
)
# squares a pawn must pass through, on its own and the neighbouring files
PASSED_MASK = np.zeros((2, 64), dtype=np.uint64)
for _square in range(64):
    _file, _rank = _square & 7, _square >> 3
    _files = int(FILE_MASK[_file] | NEIGHBOUR_FILES[_file])
    _ahead_white = sum(1 << (r * 8 + f) for r in range(_rank + 1, 8) for f in range(8))
    _ahead_black = sum(1 << (r * 8 + f) for r in range(_rank) for f in range(8))
    PASSED_MASK[WHITE, _square] = _files & _ahead_white
    PASSED_MASK[BLACK, _square] = _files & _ahead_black
# the three squares in front of a king, where its pawn shelter lives
SHIELD_MASK = np.zeros((2, 64), dtype=np.uint64)
for _square in range(64):
    _file, _rank = _square & 7, _square >> 3
    for _df in (-1, 0, 1):
        if 0 <= _file + _df < 8:
            if _rank < 7:
                SHIELD_MASK[WHITE, _square] |= np.uint64(1 << ((_rank + 1) * 8 + _file + _df))
            if _rank > 0:
                SHIELD_MASK[BLACK, _square] |= np.uint64(1 << ((_rank - 1) * 8 + _file + _df))
PASSED_BONUS = np.array([0, 0, 8, 15, 30, 55, 90, 0], dtype=np.int64)  # by rank from home
DOUBLED_MG, DOUBLED_EG = -10, -15
ISOLATED_MG, ISOLATED_EG = -10, -15
BISHOP_PAIR_MG, BISHOP_PAIR_EG = 25, 40
ROOK_SEMI_OPEN_MG, ROOK_OPEN_MG = 12, 20
MOBILITY_MG = np.array([0, 3, 3, 2, 1, 0], dtype=np.int64)  # per reachable square, by piece
MOBILITY_EG = np.array([0, 3, 3, 4, 2, 0], dtype=np.int64)
SHIELD_MG = 10  # per pawn beyond two in front of the king

# --- Search constants ---------------------------------------------------------------

MATE = 30_000
MATE_THRESHOLD = MATE - 1_000
MAX_DEPTH = 64
MAX_PLY = 128
FUTILITY_DEPTH = 2
FUTILITY_MARGIN = np.array([0, 150, 300], dtype=np.int64)
DELTA_MARGIN = 150
DELTA_MIN_PIECES = 8
LMR_MIN_DEPTH = 3
LMR_MIN_MOVE = 2
# how much shallower a late quiet move is searched, by depth and by its place in the list
LMR_TABLE = np.zeros((64, 64), dtype=np.int64)
for _depth in range(1, 64):
    for _index in range(1, 64):
        LMR_TABLE[_depth, _index] = int(0.75 + math.log(_depth) * math.log(_index) / 2.25)
NULL_MIN_DEPTH = 3
NULL_REDUCTION = 2
NULL_REDUCTION_DEEP = 3
NULL_DEEP_DEPTH = 6
ASPIRATION = 40
ASPIRATION_DEPTH = 4
HISTORY_LIMIT = 16_384
# Pruning that acts on a guess is confined to nodes proving a bound, never the line we
# mean to play. Applied everywhere it measured -178 elo; guarded it measured +127.
RFP_DEPTH = 6
RFP_MARGIN = 120
LMP_DEPTH = 4
LMP_COUNT = np.array([0, 5, 8, 13, 20], dtype=np.int64)
IIR_DEPTH = 4
TT_SLACK = 2
CHECK_INTERVAL = 1023  # nodes between clock reads, as a mask

EXACT, LOWER, UPPER = 0, 1, 2
TT_BITS = 21
TT_MASK = np.uint64((1 << TT_BITS) - 1)
SCORE_BIAS = 32_768

# The search state lives in a few flat arrays so the recursive functions pay nothing to
# pass it around. Layout of st, the int64 state array: the mailbox in 0..63, then scalars.
SIDE, CASTLE, EP, HALFMOVE = 64, 65, 66, 67
ROOT, NODES, ABORT, MAX_NODES, ROOT_BEST, ROOT_SCORE = 68, 69, 70, 71, 72, 73
S_SIZE = 80
# Per-ply undo records, uint64: the hash before the move, the move, the captured piece,
# castling rights, en passant square plus one, and the halfmove clock. Positions already
# played in the game occupy the rows below ROOT so one scan finds every repetition.
U_HASH, U_MOVE, U_CAPTURED, U_CASTLE, U_EP, U_HALFMOVE = 0, 1, 2, 3, 4, 5
STACK_ROWS = 1024 + MAX_PLY + 8
# Move lists per ply: moves in the first half of a row, ordering scores in the second.
LIST_WIDTH = 256
# Killers at 2 * ply, then history indexed by side, from and to.
KILLER_BASE = 0
HISTORY_BASE = 2 * MAX_PLY
HH_SIZE = HISTORY_BASE + 2 * 64 * 64

# --- Compiled board -----------------------------------------------------------------


@njit(float64(), cache=False)
def now():  # type: ignore[no-untyped-def]
    with objmode(t="f8"):
        t = time.perf_counter()
    return t


@njit(boolean(U64, int64, int64), cache=False)
def is_attacked(bb, square, by):  # type: ignore[no-untyped-def]
    """Whether any piece of colour `by` attacks the square."""
    base = by * 6
    if PAWN_ATT[by ^ 1, square] & bb[base + PAWN]:
        return True
    if KNIGHT_ATT[square] & bb[base + KNIGHT]:
        return True
    if KING_ATT[square] & bb[base + KING]:
        return True
    occupied = bb[OCC_ALL]
    if bishop_attacks(square, occupied) & (bb[base + BISHOP] | bb[base + QUEEN]):
        return True
    return bool(rook_attacks(square, occupied) & (bb[base + ROOK] | bb[base + QUEEN]))


@njit(int64(U64, int64), cache=False)
def king_square(bb, side):  # type: ignore[no-untyped-def]
    return ctz(bb[side * 6 + KING])


@njit(void(U64), cache=False)
def refresh_occupancy(bb):  # type: ignore[no-untyped-def]
    bb[OCC_WHITE] = bb[0] | bb[1] | bb[2] | bb[3] | bb[4] | bb[5]
    bb[OCC_BLACK] = bb[6] | bb[7] | bb[8] | bb[9] | bb[10] | bb[11]
    bb[OCC_ALL] = bb[OCC_WHITE] | bb[OCC_BLACK]


@njit(uint64(U64, I64), cache=False)
def compute_hash(bb, st):  # type: ignore[no-untyped-def]
    h = U0
    for square in range(64):
        piece = st[square]
        if piece != EMPTY:
            h ^= Z_PIECE[piece, square]
    h ^= Z_CASTLE[st[CASTLE]]
    if st[EP] >= 0:
        h ^= Z_EP[st[EP] & 7]
    if st[SIDE] == BLACK:
        h ^= Z_SIDE
    return h


@njit(int64(I32_2D, int64, int64, int64, uint64, int64), cache=False)
def add_moves(ml, ply, count, source, targets, flags):  # type: ignore[no-untyped-def]
    """Append one move per target bit."""
    while targets:
        to = ctz(targets)
        targets &= targets - U1
        ml[ply, count] = source | (to << 6) | flags
        count += 1
    return count


@njit(int64(I32_2D, int64, int64, uint64, int64, int64, boolean), cache=False)
def add_pawn_moves(ml, ply, count, targets, step, flags, promotions):  # type: ignore[no-untyped-def]
    """Pawn moves land on `to` from `to - step`. Promotions are expanded to all four pieces."""
    while targets:
        to = ctz(targets)
        targets &= targets - U1
        move = (to - step) | (to << 6) | flags
        if promotions:
            ml[ply, count] = move | (QUEEN << PROMO_SHIFT)
            ml[ply, count + 1] = move | (KNIGHT << PROMO_SHIFT)
            ml[ply, count + 2] = move | (ROOK << PROMO_SHIFT)
            ml[ply, count + 3] = move | (BISHOP << PROMO_SHIFT)
            count += 4
        else:
            ml[ply, count] = move
            count += 1
    return count


@njit(int64(U64, I64, I32_2D, int64, boolean), cache=False)
def generate(bb, st, ml, ply, tactical_only):  # type: ignore[no-untyped-def]
    """Pseudo-legal moves into ml[ply]; make_move rejects the ones that leave the king en prise.

    With tactical_only, just captures and promotions, which is what quiescence needs.
    """
    side = st[SIDE]
    base = side * 6
    theirs = bb[OCC_BLACK - side]
    occupied = bb[OCC_ALL]
    empty = ~occupied
    count = 0

    pawns = bb[base + PAWN]
    if side == WHITE:
        single = (pawns << U8) & empty
        count = add_pawn_moves(ml, ply, count, single & RANK_8, 8, 0, True)
        left = ((pawns & ~FILE_A) << uint64(7)) & theirs
        right = ((pawns & ~FILE_H) << uint64(9)) & theirs
        count = add_pawn_moves(ml, ply, count, left & RANK_8, 7, F_CAPTURE, True)
        count = add_pawn_moves(ml, ply, count, right & RANK_8, 9, F_CAPTURE, True)
        count = add_pawn_moves(ml, ply, count, left & ~RANK_8, 7, F_CAPTURE, False)
        count = add_pawn_moves(ml, ply, count, right & ~RANK_8, 9, F_CAPTURE, False)
        if not tactical_only:
            count = add_pawn_moves(ml, ply, count, single & ~RANK_8, 8, 0, False)
            double = ((single & RANK_3) << U8) & empty
            count = add_pawn_moves(ml, ply, count, double, 16, F_DOUBLE, False)
    else:
        single = (pawns >> U8) & empty
        count = add_pawn_moves(ml, ply, count, single & RANK_1, -8, 0, True)
        left = ((pawns & ~FILE_A) >> uint64(9)) & theirs
        right = ((pawns & ~FILE_H) >> uint64(7)) & theirs
        count = add_pawn_moves(ml, ply, count, left & RANK_1, -9, F_CAPTURE, True)
        count = add_pawn_moves(ml, ply, count, right & RANK_1, -7, F_CAPTURE, True)
        count = add_pawn_moves(ml, ply, count, left & ~RANK_1, -9, F_CAPTURE, False)
        count = add_pawn_moves(ml, ply, count, right & ~RANK_1, -7, F_CAPTURE, False)
        if not tactical_only:
            count = add_pawn_moves(ml, ply, count, single & ~RANK_1, -8, 0, False)
            double = ((single & RANK_6) >> U8) & empty
            count = add_pawn_moves(ml, ply, count, double, -16, F_DOUBLE, False)
    if st[EP] >= 0:
        ep = st[EP]
        attackers = PAWN_ATT[side ^ 1, ep] & pawns
        while attackers:
            source = ctz(attackers)
            attackers &= attackers - U1
            ml[ply, count] = source | (ep << 6) | F_CAPTURE | F_EP
            count += 1

    pieces = bb[base + KNIGHT]
    while pieces:
        source = ctz(pieces)
        pieces &= pieces - U1
        attacks = KNIGHT_ATT[source]
        count = add_moves(ml, ply, count, source, attacks & theirs, F_CAPTURE)
        if not tactical_only:
            count = add_moves(ml, ply, count, source, attacks & empty, 0)
    pieces = bb[base + BISHOP] | bb[base + QUEEN]
    while pieces:
        source = ctz(pieces)
        pieces &= pieces - U1
        attacks = bishop_attacks(source, occupied)
        count = add_moves(ml, ply, count, source, attacks & theirs, F_CAPTURE)
        if not tactical_only:
            count = add_moves(ml, ply, count, source, attacks & empty, 0)
    pieces = bb[base + ROOK] | bb[base + QUEEN]
    while pieces:
        source = ctz(pieces)
        pieces &= pieces - U1
        attacks = rook_attacks(source, occupied)
        count = add_moves(ml, ply, count, source, attacks & theirs, F_CAPTURE)
        if not tactical_only:
            count = add_moves(ml, ply, count, source, attacks & empty, 0)
    king = ctz(bb[base + KING])
    attacks = KING_ATT[king]
    count = add_moves(ml, ply, count, king, attacks & theirs, F_CAPTURE)
    if not tactical_only:
        count = add_moves(ml, ply, count, king, attacks & empty, 0)
        rights = st[CASTLE] >> (2 * side)
        if rights & 3 and not is_attacked(bb, king, side ^ 1):
            home = 56 * side
            if (
                rights & 1
                and not occupied & (BIT[home + 5] | BIT[home + 6])
                and not is_attacked(bb, home + 5, side ^ 1)
                and not is_attacked(bb, home + 6, side ^ 1)
            ):
                ml[ply, count] = king | ((home + 6) << 6) | F_CASTLE
                count += 1
            if (
                rights & 2
                and not occupied & (BIT[home + 1] | BIT[home + 2] | BIT[home + 3])
                and not is_attacked(bb, home + 3, side ^ 1)
                and not is_attacked(bb, home + 2, side ^ 1)
            ):
                ml[ply, count] = king | ((home + 2) << 6) | F_CASTLE
                count += 1
    return count


@njit(void(U64, I64, U64_2D, int64), cache=False)
def unmake_move(bb, st, stack, row):  # type: ignore[no-untyped-def]
    move = int64(stack[row, U_MOVE])
    side = st[SIDE] ^ 1
    source = move & 63
    to = (move >> 6) & 63
    promotion = (move >> PROMO_SHIFT) & 7
    captured = int64(stack[row, U_CAPTURED])

    piece = st[to]
    if promotion:
        bb[piece] ^= BIT[to]
        piece = side * 6 + PAWN
        bb[piece] ^= BIT[source]
    else:
        bb[piece] ^= BIT[source] | BIT[to]
    st[source] = piece
    st[to] = EMPTY

    if move & F_EP:
        victim = to - 8 if side == WHITE else to + 8
        bb[captured] ^= BIT[victim]
        st[victim] = captured
    elif captured != EMPTY:
        bb[captured] ^= BIT[to]
        st[to] = captured

    if move & F_CASTLE:
        rook = side * 6 + ROOK
        if to > source:
            rook_from, rook_to = source + 3, source + 1
        else:
            rook_from, rook_to = source - 4, source - 1
        bb[rook] ^= BIT[rook_from] | BIT[rook_to]
        st[rook_from] = rook
        st[rook_to] = EMPTY

    st[SIDE] = side
    st[CASTLE] = int64(stack[row, U_CASTLE])
    st[EP] = int64(stack[row, U_EP]) - 1
    st[HALFMOVE] = int64(stack[row, U_HALFMOVE])
    bb[HASH] = stack[row, U_HASH]
    refresh_occupancy(bb)


@njit(boolean(U64, I64, U64_2D, int64, int64), cache=False)
def make_move(bb, st, stack, row, move):  # type: ignore[no-untyped-def]
    """Play the move, recording what unmake needs in stack[row].

    Returns False, with the move already taken back, when it leaves the mover's own king
    attacked. That is the only legality test there is; generation is pseudo-legal.
    """
    side = st[SIDE]
    them = side ^ 1
    source = move & 63
    to = (move >> 6) & 63
    promotion = (move >> PROMO_SHIFT) & 7
    piece = st[source]
    captured = st[to]
    h = bb[HASH]

    stack[row, U_HASH] = h
    stack[row, U_MOVE] = uint64(move)
    stack[row, U_CASTLE] = uint64(st[CASTLE])
    stack[row, U_EP] = uint64(st[EP] + 1)
    stack[row, U_HALFMOVE] = uint64(st[HALFMOVE])

    if st[EP] >= 0:
        h ^= Z_EP[st[EP] & 7]
    h ^= Z_CASTLE[st[CASTLE]]

    bb[piece] ^= BIT[source] | BIT[to]
    st[source] = EMPTY
    st[to] = piece
    h ^= Z_PIECE[piece, source] ^ Z_PIECE[piece, to]

    if move & F_EP:
        victim = to - 8 if side == WHITE else to + 8
        captured = st[victim]
        bb[captured] ^= BIT[victim]
        st[victim] = EMPTY
        h ^= Z_PIECE[captured, victim]
    elif captured != EMPTY:
        bb[captured] ^= BIT[to]
        h ^= Z_PIECE[captured, to]
    stack[row, U_CAPTURED] = uint64(captured)

    if promotion:
        promoted = side * 6 + promotion
        bb[piece] ^= BIT[to]
        bb[promoted] ^= BIT[to]
        st[to] = promoted
        h ^= Z_PIECE[piece, to] ^ Z_PIECE[promoted, to]
    elif move & F_CASTLE:
        rook = side * 6 + ROOK
        if to > source:
            rook_from, rook_to = source + 3, source + 1
        else:
            rook_from, rook_to = source - 4, source - 1
        bb[rook] ^= BIT[rook_from] | BIT[rook_to]
        st[rook_from] = EMPTY
        st[rook_to] = rook
        h ^= Z_PIECE[rook, rook_from] ^ Z_PIECE[rook, rook_to]

    st[CASTLE] &= CASTLE_MASK[source] & CASTLE_MASK[to]
    h ^= Z_CASTLE[st[CASTLE]]

    st[EP] = -1
    if move & F_DOUBLE:
        # only an en passant square that can actually be captured on, so hashes stay canonical
        ep = (source + to) >> 1
        if PAWN_ATT[side, ep] & bb[them * 6 + PAWN]:
            st[EP] = ep
            h ^= Z_EP[ep & 7]

    if piece % 6 == PAWN or captured != EMPTY:
        st[HALFMOVE] = 0
    else:
        st[HALFMOVE] += 1

    refresh_occupancy(bb)
    st[SIDE] = them
    bb[HASH] = h ^ Z_SIDE

    if is_attacked(bb, king_square(bb, side), them):
        unmake_move(bb, st, stack, row)
        return False
    return True


@njit(void(U64, I64, U64_2D, int64), cache=False)
def make_null(bb, st, stack, row):  # type: ignore[no-untyped-def]
    stack[row, U_HASH] = bb[HASH]
    stack[row, U_EP] = uint64(st[EP] + 1)
    stack[row, U_HALFMOVE] = uint64(st[HALFMOVE])
    h = bb[HASH] ^ Z_SIDE
    if st[EP] >= 0:
        h ^= Z_EP[st[EP] & 7]
    st[EP] = -1
    st[HALFMOVE] += 1
    st[SIDE] ^= 1
    bb[HASH] = h


@njit(void(U64, I64, U64_2D, int64), cache=False)
def unmake_null(bb, st, stack, row):  # type: ignore[no-untyped-def]
    st[SIDE] ^= 1
    st[EP] = int64(stack[row, U_EP]) - 1
    st[HALFMOVE] = int64(stack[row, U_HALFMOVE])
    bb[HASH] = stack[row, U_HASH]


# --- Compiled evaluation ------------------------------------------------------------


@njit(types.UniTuple(int64, 2)(U64, int64), cache=False)
def structure(bb, colour):  # type: ignore[no-untyped-def]
    """Pawn structure, bishop pair, rook files, mobility and king shelter for one colour.

    Returns a middlegame and an endgame score; the caller blends them by phase.
    """
    mg = 0
    eg = 0
    base = colour * 6
    pawns = bb[base + PAWN]
    their_pawns = bb[(colour ^ 1) * 6 + PAWN]
    own = bb[OCC_WHITE + colour]
    occupied = bb[OCC_ALL]

    pieces = pawns
    while pieces:
        square = ctz(pieces)
        pieces &= pieces - U1
        file = square & 7
        if not PASSED_MASK[colour, square] & their_pawns:
            rank = square >> 3 if colour == WHITE else 7 - (square >> 3)
            mg += PASSED_BONUS[rank] // 2
            eg += PASSED_BONUS[rank]
        if FILE_MASK[file] & pawns & ~BIT[square]:
            mg += DOUBLED_MG
            eg += DOUBLED_EG
        if not NEIGHBOUR_FILES[file] & pawns:
            mg += ISOLATED_MG
            eg += ISOLATED_EG

    pieces = bb[base + KNIGHT]
    while pieces:
        square = ctz(pieces)
        pieces &= pieces - U1
        reach = popcount(KNIGHT_ATT[square] & ~own)
        mg += reach * MOBILITY_MG[KNIGHT]
        eg += reach * MOBILITY_EG[KNIGHT]
    pieces = bb[base + BISHOP]
    if popcount(pieces) >= 2:
        mg += BISHOP_PAIR_MG
        eg += BISHOP_PAIR_EG
    while pieces:
        square = ctz(pieces)
        pieces &= pieces - U1
        reach = popcount(bishop_attacks(square, occupied) & ~own)
        mg += reach * MOBILITY_MG[BISHOP]
        eg += reach * MOBILITY_EG[BISHOP]
    pieces = bb[base + ROOK]
    while pieces:
        square = ctz(pieces)
        pieces &= pieces - U1
        reach = popcount(rook_attacks(square, occupied) & ~own)
        mg += reach * MOBILITY_MG[ROOK]
        eg += reach * MOBILITY_EG[ROOK]
        if not FILE_MASK[square & 7] & pawns:
            mg += ROOK_OPEN_MG if not FILE_MASK[square & 7] & their_pawns else ROOK_SEMI_OPEN_MG
    pieces = bb[base + QUEEN]
    while pieces:
        square = ctz(pieces)
        pieces &= pieces - U1
        reach = popcount(
            (bishop_attacks(square, occupied) | rook_attacks(square, occupied)) & ~own
        )
        mg += reach * MOBILITY_MG[QUEEN]
        eg += reach * MOBILITY_EG[QUEEN]

    king = king_square(bb, colour)
    mg += (popcount(SHIELD_MASK[colour, king] & pawns) - 2) * SHIELD_MG
    return mg, eg


@njit(int64(int64, int64), cache=False)
def scaled(value, weight):  # type: ignore[no-untyped-def]
    """value * weight / TOTAL_PHASE, rounded towards zero so the score mirrors exactly."""
    product = value * weight
    if product < 0:
        return -((-product) // TOTAL_PHASE)
    return product // TOTAL_PHASE


@njit(int64(U64, I64), cache=False)
def evaluate(bb, st):  # type: ignore[no-untyped-def]
    """Static score in centipawns, from the point of view of the side to move."""
    balance = 0
    phase = 0
    for piece in range(12):
        kind = piece % 6
        if kind == KING:
            continue
        pieces = bb[piece]
        phase += PHASE_WEIGHT[kind] * popcount(pieces)
        if piece < 6:
            while pieces:
                balance += PST[piece, ctz(pieces)]
                pieces &= pieces - U1
        else:
            while pieces:
                balance -= PST[piece, ctz(pieces)]
                pieces &= pieces - U1

    # The king wants shelter while the queens are on and the centre once they are gone.
    # The blend is integer so the score never depends on float rounding.
    if phase > TOTAL_PHASE:
        phase = TOTAL_PHASE
    white_king = king_square(bb, WHITE)
    black_king = king_square(bb, BLACK)
    white_mg, white_eg = structure(bb, WHITE)
    black_mg, black_eg = structure(bb, BLACK)
    middlegame = KING_MG[WHITE, white_king] - KING_MG[BLACK, black_king] + white_mg - black_mg
    endgame = KING_EG[WHITE, white_king] - KING_EG[BLACK, black_king] + white_eg - black_eg
    balance += scaled(middlegame, phase) + scaled(endgame, TOTAL_PHASE - phase)

    # With a decisive edge and almost nothing left, material and placement give the search
    # no reason to make progress, so it shuffles until the game is drawn. Push the bare
    # king to the edge and walk our own king towards it.
    if phase <= MOP_UP_PHASE and (balance > MOP_UP_MARGIN or balance < -MOP_UP_MARGIN):
        if balance > 0:
            winner, loser = white_king, black_king
        else:
            winner, loser = black_king, white_king
        distance = abs((winner & 7) - (loser & 7)) + abs((winner >> 3) - (loser >> 3))
        drive = CENTRE_DISTANCE[loser] * EDGE_WEIGHT + (14 - distance) * APPROACH_WEIGHT
        balance += drive if balance > 0 else -drive

    # A score that has not been converted for many moves is worth less, and drifts
    # towards the draw the fifty-move rule will make of it. That gives the stronger side
    # a reason to make progress rather than shuffle.
    balance -= scaled(balance, st[HALFMOVE] * TOTAL_PHASE // 200)

    return balance if st[SIDE] == WHITE else -balance


@njit(boolean(U64), cache=False)
def insufficient_material(bb):  # type: ignore[no-untyped-def]
    if bb[PAWN] | bb[6 + PAWN] | bb[ROOK] | bb[6 + ROOK] | bb[QUEEN] | bb[6 + QUEEN]:
        return False
    return popcount(bb[KNIGHT] | bb[BISHOP] | bb[6 + KNIGHT] | bb[6 + BISHOP]) <= 1


# --- Compiled search ----------------------------------------------------------------


@njit(int64(int64, int64), cache=False)
def store_score(score, ply):  # type: ignore[no-untyped-def]
    """Make a mate score independent of where in the tree it was found."""
    if score >= MATE_THRESHOLD:
        return score + ply
    if score <= -MATE_THRESHOLD:
        return score - ply
    return score


@njit(int64(int64, int64), cache=False)
def load_score(score, ply):  # type: ignore[no-untyped-def]
    """Put a stored mate score back into this node, at this distance from the root."""
    if score >= MATE_THRESHOLD:
        return score - ply
    if score <= -MATE_THRESHOLD:
        return score + ply
    return score


@njit(void(U64, uint64, int64, int64, int64, int64), cache=False)
def tt_store(tt, key, depth, score, flag, move):  # type: ignore[no-untyped-def]
    slot = int64(key & TT_MASK) << 1
    # a deeper result for another position keeps its slot unless this one is nearly as deep
    if tt[slot] != key and int64((tt[slot + 1] >> uint64(16)) & uint64(127)) > depth + TT_SLACK:
        return
    tt[slot] = key
    tt[slot + 1] = (
        uint64(score + SCORE_BIAS)
        | (uint64(depth) << uint64(16))
        | (uint64(flag) << uint64(23))
        | (uint64(move) << uint64(25))
    )


@njit(int64(I64, int64), cache=False)
def victim_value(st, move):  # type: ignore[no-untyped-def]
    victim = st[(move >> 6) & 63]
    if victim == EMPTY:  # en passant
        return PIECE_VALUE[PAWN]
    return PIECE_VALUE[victim % 6]


@njit(void(I64, I32_2D, I64, int64, int64, int64), cache=False)
def score_moves(st, ml, hh, ply, count, first):  # type: ignore[no-untyped-def]
    """Ordering scores: the table move, promotions, captures by MVV-LVA, killers, history."""
    side = st[SIDE]
    for i in range(count):
        move = int64(ml[ply, i])
        if move == first:
            score = 1_000_000
        elif move & PROMO_MASK:
            score = 500_000 + PIECE_VALUE[(move >> PROMO_SHIFT) & 7]
        elif move & F_CAPTURE:
            attacker = st[move & 63] % 6
            score = 100_000 + victim_value(st, move) * 16 - PIECE_VALUE[attacker]
        elif move == hh[KILLER_BASE + 2 * ply] or move == hh[KILLER_BASE + 2 * ply + 1]:
            score = 50_000
        else:
            score = HISTORY_LIMIT + hh[
                HISTORY_BASE + side * 4096 + (move & 63) * 64 + ((move >> 6) & 63)
            ]
        ml[ply, LIST_WIDTH + i] = score


@njit(int64(I32_2D, int64, int64, int64), cache=False)
def pick(ml, ply, i, count):  # type: ignore[no-untyped-def]
    """Swap the best remaining move into position i and return it."""
    best = i
    for j in range(i + 1, count):
        if ml[ply, LIST_WIDTH + j] > ml[ply, LIST_WIDTH + best]:
            best = j
    if best != i:
        move, score = ml[ply, best], ml[ply, LIST_WIDTH + best]
        ml[ply, best], ml[ply, LIST_WIDTH + best] = ml[ply, i], ml[ply, LIST_WIDTH + i]
        ml[ply, i], ml[ply, LIST_WIDTH + i] = move, score
    return int64(ml[ply, i])


@njit(void(I64, int64, int64, int64), cache=False)
def history_credit(hh, side, move, bonus):  # type: ignore[no-untyped-def]
    index = HISTORY_BASE + side * 4096 + (move & 63) * 64 + ((move >> 6) & 63)
    hh[index] += bonus - hh[index] * abs(bonus) // HISTORY_LIMIT


@njit(boolean(I64, float64), cache=False)
def out_of_time(st, deadline):  # type: ignore[no-untyped-def]
    st[NODES] += 1
    if st[NODES] & CHECK_INTERVAL == 0 and (st[NODES] >= st[MAX_NODES] or now() >= deadline):
        st[ABORT] = 1
        return True
    return False


@njit(int64(*SEARCH_ARGS, int64, int64, int64, float64), cache=False)
def quiesce(bb, st, stack, ml, tt, hh, alpha, beta, ply, deadline):  # type: ignore[no-untyped-def]
    if st[ABORT] or out_of_time(st, deadline):
        return 0
    stand_pat = evaluate(bb, st)
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat
    if ply >= MAX_PLY - 1:
        return alpha

    count = generate(bb, st, ml, ply, True)
    for i in range(count):
        move = int64(ml[ply, i])
        attacker = st[move & 63] % 6
        ml[ply, LIST_WIDTH + i] = victim_value(st, move) * 16 - PIECE_VALUE[attacker]
    # Delta pruning. Take the most optimistic view of a capture, that the piece is won
    # outright for nothing, and if even that does not reach alpha there is no point
    # searching it. Skipped once the board is nearly bare, where a single capture swings
    # the game and the optimistic view is not optimistic enough.
    prune = popcount(bb[OCC_ALL]) > DELTA_MIN_PIECES
    row = st[ROOT] + ply
    for i in range(count):
        move = pick(ml, ply, i, count)
        if prune:
            gain = victim_value(st, move) if move & F_CAPTURE else 0
            if move & PROMO_MASK:
                gain += PIECE_VALUE[(move >> PROMO_SHIFT) & 7]
            if stand_pat + gain + DELTA_MARGIN <= alpha:
                continue
        if not make_move(bb, st, stack, row, move):
            continue
        score = -quiesce(bb, st, stack, ml, tt, hh, -beta, -alpha, ply + 1, deadline)
        unmake_move(bb, st, stack, row)
        if st[ABORT]:
            return 0
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


@njit(int64(*SEARCH_ARGS, int64, int64, int64, int64, boolean, float64), cache=False)
def negamax(bb, st, stack, ml, tt, hh, depth, alpha, beta, ply, allow_null, deadline):  # type: ignore[no-untyped-def]
    if st[ABORT] or out_of_time(st, deadline):
        return 0
    row = st[ROOT] + ply
    key = bb[HASH]

    # a line back to a position already on the board, or earlier in this line, is a draw
    if st[HALFMOVE] >= 100:
        return 0
    earliest = row - st[HALFMOVE]
    i = row - 2
    while i >= earliest and i >= 0:
        if stack[i, U_HASH] == key:
            return 0
        i -= 2
    if ply >= MAX_PLY - 1:
        return evaluate(bb, st)
    if popcount(bb[OCC_ALL]) <= 4 and insufficient_material(bb):
        return 0

    slot = int64(key & TT_MASK) << 1
    table_move = 0
    if tt[slot] == key:
        data = tt[slot + 1]
        table_move = int64(data >> uint64(25))
        if int64((data >> uint64(16)) & uint64(127)) >= depth:
            score = load_score(int64(data & uint64(0xFFFF)) - SCORE_BIAS, ply)
            flag = int64((data >> uint64(23)) & uint64(3))
            if flag == EXACT:
                return score
            if flag == LOWER:
                if score >= beta:
                    return beta
            elif score <= alpha:
                return alpha

    side = st[SIDE]
    in_check = is_attacked(bb, king_square(bb, side), side ^ 1)
    if in_check:
        depth += 1  # a forcing line is searched a ply deeper, and never scored while in check
    if depth <= 0:
        return quiesce(bb, st, stack, ml, tt, hh, alpha, beta, ply, deadline)
    pv_node = beta - alpha > 1

    static = 0
    if not in_check:
        static = evaluate(bb, st)
        # Reverse futility. A position this far above beta with the move in hand fails
        # high on almost anything, so do not search it.
        if (
            not pv_node
            and depth <= RFP_DEPTH
            and beta < MATE_THRESHOLD
            and static - RFP_MARGIN * depth >= beta
        ):
            return beta
    # no table move means the ordering here is guesswork; a ply less keeps it cheap
    if depth >= IIR_DEPTH and table_move == 0:
        depth -= 1

    # Null move: hand the opponent a free move. If the position still beats beta after
    # that, the real move list will almost certainly beat it too, so cut without
    # searching. Skipped in check, and skipped when the side to move has nothing but
    # pawns, because that is where zugzwang lives and having to move is a liability
    # rather than the free gift this assumes.
    base = side * 6
    if (
        allow_null
        and depth >= NULL_MIN_DEPTH
        and not in_check
        and (bb[base + KNIGHT] | bb[base + BISHOP] | bb[base + ROOK] | bb[base + QUEEN])
    ):
        reduction = NULL_REDUCTION_DEEP if depth >= NULL_DEEP_DEPTH else NULL_REDUCTION
        make_null(bb, st, stack, row)
        score = -negamax(
            bb, st, stack, ml, tt, hh, depth - 1 - reduction, -beta, -beta + 1,
            ply + 1, False, deadline,
        )
        unmake_null(bb, st, stack, row)
        if st[ABORT]:
            return 0
        if score >= beta:
            return beta

    count = generate(bb, st, ml, ply, False)
    score_moves(st, ml, hh, ply, count, table_move)

    # Futility. Close to the leaves, a quiet move rarely rescues a position that is
    # already this far below alpha, so once the static score plus a generous margin
    # still cannot reach it, quiet moves are skipped. Never while in check, never on
    # a mate score, and never before at least one move has actually been searched.
    futile = False
    if not in_check and depth <= FUTILITY_DEPTH and abs(alpha) < MATE_THRESHOLD:
        futile = static + FUTILITY_MARGIN[depth] <= alpha
    # Late move pruning. Near the leaves, once this many quiet moves have been searched
    # without beating alpha, the rest are skipped rather than reduced.
    prune_late = not pv_node and not in_check and depth <= LMP_DEPTH

    best_move = 0
    flag = UPPER
    legal = 0
    searched = 0
    for i in range(count):
        move = pick(ml, ply, i, count)
        quiet = not move & TACTICAL
        if not make_move(bb, st, stack, row, move):
            continue
        legal += 1
        gives_check = is_attacked(bb, king_square(bb, side ^ 1), side)
        if (
            quiet
            and searched
            and not gives_check
            and (futile or (prune_late and searched >= LMP_COUNT[depth]))
        ):
            unmake_move(bb, st, stack, row)
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
            and searched > 1
            and depth >= LMR_MIN_DEPTH
            and i >= LMR_MIN_MOVE
            and not in_check
            and not gives_check
        ):
            reduction = LMR_TABLE[min(depth, 63), min(i, 63)]
            if pv_node:
                reduction -= 1  # the line we mean to play deserves more of the depth
            reduction = max(0, min(reduction, depth - 1))

        if searched == 1:
            score = -negamax(
                bb, st, stack, ml, tt, hh, depth - 1, -beta, -alpha, ply + 1, True, deadline
            )
        else:
            # Trust the ordering: assume everything after the first move fails low and
            # prove it with a one-point window, which is far cheaper. Only a move that
            # beats alpha costs a full re-search.
            score = -negamax(
                bb, st, stack, ml, tt, hh, depth - 1 - reduction, -alpha - 1, -alpha,
                ply + 1, True, deadline,
            )
            if reduction and score > alpha and not st[ABORT]:
                score = -negamax(
                    bb, st, stack, ml, tt, hh, depth - 1, -alpha - 1, -alpha, ply + 1, True,
                    deadline,
                )
            if alpha < score < beta and not st[ABORT]:
                score = -negamax(
                    bb, st, stack, ml, tt, hh, depth - 1, -beta, -alpha, ply + 1, True, deadline
                )
        unmake_move(bb, st, stack, row)
        if st[ABORT]:
            return 0
        if score >= beta:
            if quiet:
                if hh[KILLER_BASE + 2 * ply] != move:
                    hh[KILLER_BASE + 2 * ply + 1] = hh[KILLER_BASE + 2 * ply]
                    hh[KILLER_BASE + 2 * ply] = move
                # The cutoff move is rewarded and the quiet moves tried before it are
                # penalised, each pulled towards the limit so old evidence fades.
                bonus = min(depth * depth, 400)
                history_credit(hh, side, move, bonus)
                for j in range(i):
                    earlier = int64(ml[ply, j])
                    if not earlier & TACTICAL:
                        history_credit(hh, side, earlier, -bonus)
            tt_store(tt, key, depth, store_score(beta, ply), LOWER, move)
            return beta
        if score > alpha:
            alpha = score
            best_move = move
            flag = EXACT

    if legal == 0:
        return -MATE + ply if in_check else 0
    tt_store(tt, key, depth, store_score(alpha, ply), flag, best_move)
    return alpha


@njit(
    types.UniTuple(int64, 2)(*SEARCH_ARGS, int64, int64, int64, int64, float64), cache=False
)
def search_root(bb, st, stack, ml, tt, hh, depth, first, alpha, beta, deadline):  # type: ignore[no-untyped-def]
    """One deepening pass inside a window. Returns the score and move; ROOT_BEST tracks the
    pass as it goes."""
    st[ROOT_BEST] = 0
    row = st[ROOT]
    count = generate(bb, st, ml, 0, False)
    score_moves(st, ml, hh, 0, count, first)
    best = first
    legal = 0
    for i in range(count):
        move = pick(ml, 0, i, count)
        if not make_move(bb, st, stack, row, move):
            continue
        legal += 1
        if legal == 1:
            score = -negamax(bb, st, stack, ml, tt, hh, depth - 1, -beta, -alpha, 1, True, deadline)
        else:
            score = -negamax(
                bb, st, stack, ml, tt, hh, depth - 1, -alpha - 1, -alpha, 1, True, deadline
            )
            if score > alpha and not st[ABORT]:  # the narrow window was wrong, pay for the real one
                score = -negamax(
                    bb, st, stack, ml, tt, hh, depth - 1, -beta, -alpha, 1, True, deadline
                )
        unmake_move(bb, st, stack, row)
        if st[ABORT]:
            break
        if score > alpha:
            alpha = score
            best = move
            # a move proved best so far by a complete search is a better answer than the
            # last finished pass gave, even if this pass never finishes
            st[ROOT_BEST] = move
            st[ROOT_SCORE] = score
            if score >= beta:
                break
    return alpha, best


@njit(int64(U64, I64, U64_2D, I32_2D, int64, int64), cache=False)
def perft(bb, st, stack, ml, depth, ply):  # type: ignore[no-untyped-def]
    """Leaf count to a depth. Only used by the tests, against python-chess."""
    if depth == 0:
        return 1
    count = generate(bb, st, ml, ply, False)
    total = 0
    for i in range(count):
        move = int64(ml[ply, i])
        if make_move(bb, st, stack, st[ROOT] + ply, move):
            total += perft(bb, st, stack, ml, depth - 1, ply + 1)
            unmake_move(bb, st, stack, st[ROOT] + ply)
    return total


# --- Bridging python-chess and the compiled board -----------------------------------

_CASTLE_BITS = ((chess.BB_H1, 1), (chess.BB_A1, 2), (chess.BB_H8, 4), (chess.BB_A8, 8))


def load_position(board: chess.Board, bb: np.ndarray, st: np.ndarray) -> None:
    """Write a python-chess board into the compiled representation."""
    bb[:] = 0
    st[:64] = EMPTY
    for square, piece in board.piece_map().items():
        code = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        bb[code] |= BIT[square]
        st[square] = code
    st[SIDE] = WHITE if board.turn == chess.WHITE else BLACK
    rights = board.clean_castling_rights()
    st[CASTLE] = sum(flag for mask, flag in _CASTLE_BITS if rights & mask)
    st[EP] = board.ep_square if board.has_legal_en_passant() else -1
    st[HALFMOVE] = min(board.halfmove_clock, 100)
    refresh_occupancy(bb)
    bb[HASH] = compute_hash(bb, st)


def to_chess_move(move: int) -> chess.Move:
    promotion = (move >> PROMO_SHIFT) & 7
    return chess.Move(move & 63, (move >> 6) & 63, promotion + 1 if promotion else None)


class Engine:
    """The compiled search plus the state that lives for the whole game."""

    def __init__(self) -> None:
        self.bb = np.zeros(16, dtype=np.uint64)
        self.st = np.zeros(S_SIZE, dtype=np.int64)
        self.stack = np.zeros((STACK_ROWS, 6), dtype=np.uint64)
        self.ml = np.zeros((MAX_PLY + 4, 2 * LIST_WIDTH), dtype=np.int32)
        self.tt = np.zeros(2 << TT_BITS, dtype=np.uint64)
        self.hh = np.zeros(HH_SIZE, dtype=np.int64)
        self.history: list[int] = []  # hashes of the positions this game has been through
        self.nodes = 0
        self.depth = 0

    def remember(self, board: chess.Board) -> None:
        """Record a position the game has reached. An irreversible move empties the past."""
        load_position(board, self.bb, self.st)
        if board.halfmove_clock == 0:
            self.history.clear()
        self.history.append(int(self.bb[HASH]))
        del self.history[:-1000]

    def search(self, board: chess.Board, deadline: float, max_nodes: int = 1 << 60) -> chess.Move:
        st, stack = self.st, self.stack
        load_position(board, self.bb, st)
        # the positions already played sit under the root so the repetition scan sees them
        past = self.history
        if past and past[-1] == int(self.bb[HASH]):
            past = past[:-1]
        for i, h in enumerate(past):
            stack[i, U_HASH] = h
        st[ROOT] = len(past)
        st[NODES] = 0
        st[ABORT] = 0
        st[MAX_NODES] = max_nodes
        # killers belong to one search; history carries over, faded, as ordering advice
        self.hh[:HISTORY_BASE] = 0
        self.hh[HISTORY_BASE:] //= 2

        best = 0
        score = 0
        pass_cost = 0.0
        growth = 2.0
        started = time.perf_counter()
        for depth in range(1, MAX_DEPTH + 1):
            # Each pass costs a multiple of the one before it. Predicting the next one from
            # the last one measured adapts to the position.
            elapsed = time.perf_counter() - started
            if pass_cost and elapsed + pass_cost * growth > deadline - started:
                break
            pass_started = time.perf_counter()
            # Aspiration: search in a narrow window around the last score, which cuts off
            # far more, and widen only when the score lands outside it.
            window = ASPIRATION
            alpha, beta = -MATE, MATE
            if depth >= ASPIRATION_DEPTH:
                alpha, beta = score - window, score + window
            while True:
                found, move = search_root(
                    self.bb, st, stack, self.ml, self.tt, self.hh, depth, best, alpha, beta,
                    deadline,
                )
                if st[ABORT] or alpha < found < beta:
                    break
                window *= 2
                if found <= alpha:
                    alpha = max(-MATE, alpha - window)
                else:
                    beta = min(MATE, beta + window)
            if st[ABORT]:
                if st[ROOT_BEST]:
                    best = int(st[ROOT_BEST])
                break
            score = int(found)
            cost = time.perf_counter() - pass_started
            if pass_cost:
                growth = min(4.0, max(1.5, cost / pass_cost))
            pass_cost = cost
            best = int(move)
            self.depth = depth
            if abs(score) >= MATE_THRESHOLD:
                break
        self.nodes = int(st[NODES])
        return to_chess_move(best)


# --- Fallback search on python-chess ------------------------------------------------
# Only runs if the compiled engine raises or hands back an illegal move. It is the shape of
# the search above without the pruning, and it has never been needed.

_FALLBACK_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


def _fallback_evaluate(board: chess.Board) -> int:
    bb = np.zeros(16, dtype=np.uint64)
    st = np.zeros(S_SIZE, dtype=np.int64)
    load_position(board, bb, st)
    return int(evaluate(bb, st))


def _fallback_order(board: chess.Board) -> list[chess.Move]:
    def rank(move: chess.Move) -> int:
        victim = board.piece_type_at(move.to_square)
        if victim is None:
            return 0
        attacker = board.piece_type_at(move.from_square)
        return 16 * _FALLBACK_VALUE[victim] - _FALLBACK_VALUE.get(attacker or 0, 0) + 1000

    return sorted(board.legal_moves, key=rank, reverse=True)


def _fallback_search(
    board: chess.Board, depth: int, alpha: int, beta: int, deadline: float
) -> int:
    if time.perf_counter() >= deadline:
        raise TimeoutError
    if board.is_repetition(2) or board.halfmove_clock >= 100:
        return 0
    if depth <= 0:
        stand = _fallback_evaluate(board)
        if stand >= beta:
            return beta
        alpha = max(alpha, stand)
        for move in _fallback_order(board):
            if not board.is_capture(move):
                break
            board.push(move)
            score = -_fallback_search(board, 0, -beta, -alpha, deadline)
            board.pop()
            if score >= beta:
                return beta
            alpha = max(alpha, score)
        return alpha
    moves = _fallback_order(board)
    if not moves:
        return -MATE if board.is_check() else 0
    for move in moves:
        board.push(move)
        score = -_fallback_search(board, depth - 1, -beta, -alpha, deadline)
        board.pop()
        if score >= beta:
            return beta
        alpha = max(alpha, score)
    return alpha


def fallback_move(board: chess.Board, deadline: float) -> chess.Move:
    moves = _fallback_order(board)
    best = moves[0]
    for depth in range(1, 6):
        alpha = -MATE
        current = best
        try:
            for move in moves:
                board.push(move)
                try:
                    score = -_fallback_search(board, depth - 1, -MATE, -alpha, deadline)
                finally:
                    board.pop()
                if score > alpha:
                    alpha, current = score, move
        except TimeoutError:
            break
        best = current
    return best


# --- Time management ----------------------------------------------------------------

INCREMENT_MS = 500
OVERHEAD_MS = 150.0
MIN_BUDGET_MS = 5.0
PANIC_MS = 100


def _budget_ms(time_left_ms: int, board: chess.Board) -> float:
    """How long this move may take. A flag is the most common self-inflicted loss.

    get_move is told the clock but never the increment, so the platform's 0.5s is a
    constant here. A local arena run at a different increment will be a little off.
    """
    moves_left = max(18, 46 - board.fullmove_number)
    budget = time_left_ms / moves_left + INCREMENT_MS * 0.6
    budget = min(budget, time_left_ms * 0.35)
    return max(budget - OVERHEAD_MS, MIN_BUDGET_MS)


# --- Entry point --------------------------------------------------------------------

_engine = Engine()


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation."""
    started = time.perf_counter()
    board = chess.Board(fen)
    if not any(board.legal_moves):
        return "0000"  # the referee ends the game before asking, so this is only a guard
    deadline = started + _budget_ms(time_left_ms, board) / 1000.0

    move: chess.Move | None = None
    if time_left_ms >= PANIC_MS and board.is_valid():
        try:
            _engine.remember(board)
            move = _engine.search(board, deadline)
            if not board.is_legal(move):
                move = None
        except Exception:  # any failure here must still produce a move
            move = None
    if move is None:
        if time_left_ms >= PANIC_MS:
            move = fallback_move(board, deadline)
        else:
            move = _fallback_order(board)[0]

    # the position after our reply is one the game has reached too
    board.push(move)
    with contextlib.suppress(Exception):
        _engine.remember(board)
    return move.uci()


# Compile and warm everything at import, inside the 90 second init budget rather than on
# the clock. numba compiles per signature, so this runs the real call path once.
_warm = Engine()
_warm.remember(chess.Board())
_warm.search(chess.Board(), time.perf_counter() + 30.0, max_nodes=20_000)
_warm.search(
    chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    time.perf_counter() + 30.0,
    max_nodes=20_000,
)
del _warm
