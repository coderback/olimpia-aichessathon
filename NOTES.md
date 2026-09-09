# Olimpia — build notes

What we built for AI Chessathon, why it is shaped this way, what we measured, and
what we got wrong. Written so anyone on the team can pick this up cold.

The submission is `agent.py` alone; nothing else ships. Section 7 says which commit is
live and which is waiting on a gate.

---

## 1. What the agent is

A classical alpha-beta engine whose board, move generator, evaluation and search are
compiled by numba at import. The first two days were the same engine in pure Python on
`python-chess` at ~13,000 nodes a second, depth 4–5 per move. The compiled engine runs
at **1.4–2.0 million nodes a second**, depth 13–16 in the middlegame, 24 in endgames.
Speed was the lever with the right order of magnitude (§7 of the previous notes said
so; it was right) and it is worth more than everything else on this page combined.

**Board** — bitboards, one per piece and colour, plus a 64-square mailbox. Sliding
attacks are magic lookups: the occupancy along a piece's rays, masked and multiplied by
a per-square constant, indexes a table of attack sets. The magics are found at import
by random trial in ~2 s. Move generation is pseudo-legal; `make_move` plays the move
and takes it straight back if the mover's king is attacked, and that is the only
legality test. Zobrist hashes are kept incrementally.

**Search**

| piece | what it does |
|---|---|
| negamax + alpha-beta | the whole game |
| iterative deepening | always a move to return when the clock runs out |
| aspiration windows | from depth 4, a ±40 window around the last score, widened on failure |
| transposition table | 2M entries, kept for the whole game, deeper entries keep their slot |
| MVV-LVA, killers, history | history has gravity, penalises the quiet moves tried before a cutoff, and carries between moves at half strength |
| principal variation search | narrow window after the first move |
| check extension | a position in check is searched a ply deeper, wherever it is |
| null-move pruning | R = 2, 3 from depth 6; never in check, never with pawns only |
| late move reductions | a table in log(depth) × log(move index), one less on the PV |
| reverse futility, late move pruning | **only at nodes proving a bound** (§3) |
| internal iterative reduction | a ply less when the table has no move to order by |
| futility, quiescence, delta pruning | as before |
| contempt | a draw scores −20 for the root side, so it plays on when level |

**Evaluation** — material and piece-square tables, the king table blended between
middlegame and endgame by phase, a mop-up term for bare-king endgames, and now:
passed pawns by rank, doubled and isolated pawns, the bishop pair, rooks on open and
half-open files, mobility per reachable square, pawns in front of the king. Each new
term has a middlegame and endgame weight blended by phase. The score drifts towards
zero as the halfmove clock climbs, so a side that is ahead makes progress.

**Time management** — `_budget_ms` sizes each move from the clock; the compiled search
reads the wall clock every 1024 nodes through numba's `objmode` (1 µs a read) and
aborts cleanly, salvaging the best root move the aborted pass had already proved.
Below a 100 ms clock the move is returned without searching.

**Fallback** — a python-chess search of the same shape runs only if the compiled path
raises or returns an illegal move. It has never been needed.

---

## 2. Measured results

Head-to-head, alternating colours from the 19 curated starting positions the platform
used in our rated games (`openings.txt`). Elo figures carry 95% intervals.

**Compiled engine, 2026-09-09**

| change | build | fast clock (10s + 0.1) | real clock (120s + 0.5) |
|---|---|---|---|
| numba port, same logic as v6 | nb1 `628b58a` | 40–0 vs greedy, 30–0 vs v6 | **+46 =2 −0 vs v6** (48 games) |
| search tuned for depth 13 (A) | `c9a9f2b` | **+98 ±79** vs nb1 (80 games) | — |
| non-PV pruning, IIR, fifty-move drift (B) | `91cf8e6` | +41 ±78 vs A (77 games) | — |
| evaluation terms (C) | `f7cd53d` | **+167 ±88** vs B (74 games) | gate running: 4–0 so far |
| contempt (D) | `be8e53b` | screen running | — |

The evaluation terms are the same idea that lost 102 Elo on the slow engine. At depth
13 they are the largest single gain. The earlier result was a depth artefact, not a
verdict on the terms.

**Pure Python engine, 2026-09-07/08** (for the record; all superseded)

| change | fast (5s) | real (120s) | outcome |
|---|---|---|---|
| TT + PVS + history + null-move | +37 | — | shipped |
| six evaluation terms | −102 | — | rejected, see above |
| late move reductions | +124 | — | shipped |
| futility + delta pruning | +36 | — | shipped |
| whole build vs v1 | +69 | +280 | — |
| king danger blend | +38 | −46 | reverted |
| reverse futility + LMP at every node | −178 | — | rejected |
| the same, non-PV only | +127 | −44 | rejected |

**Robustness:** every compiled build so far has finished every game it played. Perft
matches python-chess exactly on ten positions covering castling, en passant, promotions
and both colours; the evaluation mirrors exactly with colours swapped on 500 random
positions. Ruff and mypy clean.

---

## 3. Architectural decisions, and why

**numba, not a rewrite of the search.** The search logic had ~5,000 games of evidence
behind it. The port kept it line for line and only replaced the substrate, so the first
measurement isolated the speed gain. Then the search was retuned for the depth it now
reaches, as its own measured step.

**Constant tables are module globals; mutable state travels as six flat arrays.**
numba freezes global arrays into the compiled code at no runtime cost, even the 800 KB
rook table. Passing a *tuple* of arrays through a recursive call instead measured **70×
slower** — numba increfs every element on every call. Nine separate array arguments
cost 56 ns a call, acceptable against a 1–2 µs node.

**Every compiled function declares its signature.** Without one, numba compiles a fresh
copy per distinct *literal* argument at each call site: `add_pawn_moves` was compiled
14 times and `negamax` three times (for `True`, `False` and plain bool). Import went
from 46 s to 22 s. The platform allows 90 s.

**Bit scans and popcount are LLVM intrinsics.** numba has neither; `llvm.cttz` and
`llvm.ctpop` bound through `numba.extending.intrinsic` compile to single instructions.

**`uint64` discipline.** numba turns `uint64 & 255`, `uint64 + 1` and `uint64 - i` into
`int64`, and unifying that with a `uint64` variable produces `float64`. Every mask is an
explicit `np.uint64`; single bits come from a `BIT[sq]` table rather than a shift by a
signed index. The one time this slipped (a Python `int` handed back from compiled code
and passed in again) it broke the build with a `float64 & int64` error.

**Pseudo-legal generation with the check in `make_move`.** Simplest correct design;
perft made every bug loud within minutes.

**Transposition table lives for the game.** The Python engine cleared it per move
because a stored score could contradict the repetition history. The compiled engine
checks repetition before probing the table, so the table can stay.

**Repetitions.** The platform hands us a bare FEN. Every position we are asked about,
and every position after our reply, is recorded; they sit under the root of the undo
stack so one backward scan finds any repetition in the game or the line. Any earlier
occurrence counts as a draw.

**Pruning on a guess only where a bound is being proved.** Reverse futility and late
move pruning applied at every node measured −178 on the slow engine; guarded to
one-point windows, positive. The guard is the whole story.

**Contempt.** In the screens every repetition draw had the stronger build level or
behind on material — the weaker side escaping. Most ladder opponents are now weaker
than this build, and a draw against them is a lost half point.

**No neural network.** With 44 hours to the upload deadline the training pipeline
alone would have consumed it. The compiled substrate is what a net would need anyway;
it is the first thing to try if this continues.

---

## 4. What we tried and rejected

Nothing today; every batch screened positive. The previous engine's rejections are in
§2. One lesson from them survived intact: fast-clock results were not trusted for any
upload, only the 120 s + 0.5 s gate.

---

## 5. How to measure

**The rule:** a build ships only if it is not negative at **120 s + 0.5 s**. Fast
screens filter candidates; they do not gate. On the Python engine fast results
diverged from real ones three times in both directions.

**The self-play driver** (`scratchpad/selfplay.py`, not in the repo) imports two frozen
builds once and plays them in one process. The arena spawns two fresh processes per
game, which at a 10 s clock spends more time compiling numba than playing chess. Six
shards of the driver play ~240 fast games an hour.

**Freeze builds first.** Copy each build to its own directory. The arena and the driver
import `agent.py` at start; editing the working tree during a run corrupted a result
once.

**Openings.** `openings.txt` holds the 19 curated FENs from our rated games. Two
deterministic engines from the start position replay the same game.

**Power.** The dev laptop throttles to 1.9 GHz on battery. A benchmark taken at 8%
charge read 5× slow, and last session's mid-run shutdown was probably the battery. Check
the charge before trusting any number.

**Sample sizes.** ±80 Elo needs ~50 games, ±40 needs ~200. Node counts and depths are
not evidence; only games at the right clock decide.

---

## 6. Constraints that shaped this

From `aichessathon.com/docs` — authoritative and they change, so re-fetch:

- `agent.py` at the zip root, `get_move(fen, time_left_ms) -> str`, UCI out
- 120 s + 0.5 s per game, one core of an EPYC 9V74, 2 GB, no network, no GPU
- 90 s init budget — the compiled build uses ~22 s of it locally
- Only torch, numpy, python-chess, onnxruntime, numba are available
- Game drawn at 600 plies; flag draws when the other side cannot mate
- No third-party engine or published network; source a judge can read

---

## 7. Where we stand

Before today: ladder rating 1518, rank #241 of 418, record 8–8–3 over the rated rounds,
~520–690 Elo short of a London seat.

The compiled build nb1 (`628b58a`) beat the live build 46–0–2 at the real clock. Its
`agent.zip` is at the repo root. The full stack through C is ~+300 Elo over nb1 at the
fast clock and is in its real-clock gate now; D is in its fast screen.

Uploads close **11 September 11:00**. Order of business: upload nb1 and read the
platform's init time from the validation log; gate C; upload C; gate D on top if there
is time. Then leave it alone.

---

## 8. Running things

```
make play                                   # one game against a baseline, real clock
make arena                                  # 20 fast games with a score
make zip                                    # build the submission
make gate                                   # ruff, mypy, two games that must finish
uv run python -m harness.arena --agent <frozen> --opponent <frozen> --games 48 \
    --base-ms 120000 --increment-ms 500 --openings openings.txt --pgn-dir games/
```

The harness is local only; nothing in it ships. The platform's validation log on the
dashboard is the authority on whether an upload is accepted.
