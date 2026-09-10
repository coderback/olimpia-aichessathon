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
| evaluation terms (C) | `f7cd53d` | **+167 ±88** vs B (74 games) | **+31 =4 −1 vs nb1** (36 games, +417 ±205) |
| contempt (D) | `68c63c6` | +44 ±77 vs C (80 games) | **+15 =14 −7 vs C** (36 games, +79 ±116) |
| soft/hard time limits, 8M table (E) | `7af899e` | −35 ±77 vs D (80 games) | **−10 ±114 vs D** (36 games) — **reverted** |
| king attack (F) | `e88d0ce` | +16 ±44 vs D (240 games) | **−29 ±114 vs D** (36 games) — **reverted** |
| check evasion in quiescence (G) | `96f3263` | — | **+6 ±44 vs D** (240 games) — level, shipped on correctness |

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

**Stockfish review of the rated games.** `harness/review.py` runs Stockfish 17 at depth
16 over a folder of PGNs and reports what the dashboard shows: accuracy, centipawn loss
and move labels for both sides, plus each of our mistakes as a position. Over the 19
rated games the Python engine played, it scored 87–98% accuracy and lost games to
sixteen concrete mistakes rather than to being outplayed. **The compiled build replays
thirteen of the sixteen correctly, including all six blunders.** The three it still
misplayed share one cause: pieces bearing on a king, with our static score far tamer
than Stockfish (+142 against +592 in one). That is what batch F adds, and with it the
engine finds the attacking move in one of the three at 3 s. The review is committed at
`Chess results/stockfish-review-rounds-64-82.txt`.

**Known weakness:** rook endgames. In one reviewed position (rook and pawn each, a
passed pawn on the seventh) the compiled build reads both the drawing and the losing
king move as level at depth 21; which one it plays depends on the tree. No endgame
knowledge beyond the mop-up term exists yet.

**On the ladder.** Rounds 87 and 88 are the first two games by the compiled build, both
wins by checkmate, and the review agrees with the arena:

| | round 87 | round 88 | the Python engine (83–86) |
|---|---|---|---|
| accuracy | 97.9% | 93.4% | 85–95% |
| centipawn loss per move | 8 | 28 | 46–108 |
| inaccuracies / mistakes / blunders | 0 / 0 / 0 | 0 / 0 / 0 | 2–6 / 0–2 / 0 |
| clock left at the end | 90.8 s | 71.9 s | 11.5–59.6 s |
| init on the ladder machine | 18.4 s | 18.0 s | 0.5 s |

Zero flagged moves in 53 moves of play, where the old engine averaged three or four a
game. Two games is two games, both as White against weaker opponents, so this is
consistent with the arena rather than independent confirmation of it.

**Absolute strength.** Everything else here is self-play, which measures what beats us
rather than what beats the field and gives no absolute number. `harness/spar.py` plays a
build against Stockfish with `UCI_LimitStrength` set to a known Elo, both sides on the
same clock. The first brackets were set at 1900 / 2200 / 2500 and the build won its first
five games across all three, so the handicap was checked directly: Stockfish at 2500 beat
Stockfish at 1320 four–nil, so the knob works and the brackets were simply too low. They
are now **2500 / 2850 / 3190** (3190 is Stockfish's maximum).

Final result over 178 games at 120 s + 0.5 s:

| opponent | result | score | implied |
|---|---|---|---|
| Stockfish 2500 | +38 =11 −11 | 72.5% | 2668 ±98 |
| Stockfish 2850 | +5 =35 −18 | 38.8% | 2771 ±92 |
| Stockfish 3190 | +0 =26 −34 | 21.7% | 2967 ±107 |

**The three brackets do not agree, and that is the finding.** The implied rating climbs
with the opponent's label and the outer two intervals do not overlap at all, which means
Stockfish's `UCI_Elo` scale is compressed: the real gap between its 2500 and 3190 settings
is far smaller than 690 points. So the anchor gives a range, roughly **2700–2800**, and
cannot give a point estimate. An earlier reading at half the sample appeared to agree on
2700–2780; with tighter intervals it no longer does.

What is solid regardless of the scale: the build beats the 2500 setting clearly, holds the
2850 setting to 35 draws in 58, and loses to the maximum setting without ever being swept
— 26 draws in 60. This is a hard engine to beat rather than one that collapses.

Read it as a bracket, not a rating. `UCI_LimitStrength` weakens Stockfish by making it
choose deliberately inferior moves, which are human-shaped errors that another engine
punishes harder than the label implies, and the scale is calibrated against human
ratings.

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

**Soft/hard time limits (E).** Letting a deepening pass run past its budget, to spend the
11–75 s the rated games left on the clock, measured −35 at the fast clock and −10 at the
real one. The unused time is real, but a pass that overruns takes it from later moves;
the way to use it is a larger budget per move, not a longer overrun, and that was not
tried. Reverted on the branch so the attempt and the evidence stay on the record.

**King attack (F).** Attack units on the squares around the enemy king, squared,
middlegame only; chosen because all three positions the compiled build still misplays
are king attacks its score cannot see. +16 at the fast clock, −29 at the real one, and
reverted. This is the second king-safety term to pass a fast screen and fail the real
clock (the first was on the Python engine). The diagnosis stands; the term does not.
The next attempt should be tested only at the real clock, and should probably scale
with the defender's missing shelter rather than stand alone.

Otherwise every batch today screened positive. The previous engine's rejections are in
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

**Sample sizes, and what they rule out.** ±80 Elo needs ~50 games, ±40 needs ~200.
A 36-game real-clock gate resolves ±114, which is how E (−10) and F (−29) came to be
decided on numbers smaller than their own error bars. A 240-game gate resolves ±40 and
takes four shards about seven hours. **Search parameters — the LMR divisor, null-move R,
aspiration width, futility margins — are typically worth 10–30 Elo each, which is below
what we can resolve in the time available, so tuning them individually would be reading
noise.** Screen many variants at the fast clock, gate only the best one or two.

Node counts and depths are not evidence; only games at the right clock decide.

**Sparring.** `uv run python -m harness.spar <build> --elo 2500 --games 60
--base-ms 120000 --increment-ms 500` plays a build against Stockfish at a fixed strength
with both sides on the same clock, and prints the implied rating. Use several brackets at
once rather than guessing which one is right.

**Reviewing games.** `uv run python -m harness.review "Chess results"` needs Stockfish
(`winget install Stockfish.Stockfish`; the default path is where winget puts it). Every
flagged position can then be handed to a build to see whether it still misplays it —
that is how the king-attack term was chosen, and it is the cheapest diagnostic we have.
Analysing our own games with an engine is allowed; only shipping one is not.

---

## 6. Constraints that shaped this

From `aichessathon.com/docs` — authoritative and they change, so re-fetch:

- `agent.py` at the zip root, `get_move(fen, time_left_ms) -> str`, UCI out
- 120 s + 0.5 s per game, one core of an EPYC 9V74, 2 GB, no network, no GPU
- 90 s init budget — the compiled build uses ~22 s of it locally
- Only torch, numpy, python-chess, onnxruntime, numba are available
- Game drawn at 600 plies; flag draws when the other side cannot mate
- No third-party engine or published network; source a judge can read

**Using an engine off the board is allowed, and this was checked against the live docs
rather than assumed.** The ban reads "Third party engines are prohibited. That covers
Stockfish, Lc0, Maia, any wrapper around one and any port or translation of one", and it
governs what ships; the same page says "training it on positions an existing engine
labelled is allowed". Generating training labels is the more aggressive use, so analysing
our games and sparring against Stockfish are plainly inside the line. Verified rather than
assumed: `agent.zip` holds one file, `agent.py` has no reference to any engine and no
`subprocess` or `Popen`, and `harness/package.py` ships only `agent.py`, so the two files
that know Stockfish's path can never reach a submission. The boundary to keep is "port or
translation": the piece-square tables are the published simplified-evaluation set and the
search techniques are general chess-programming knowledge, not transcribed code.

---

## 7. Where we stand

Before today: ladder rating 1518, rank #241 of 418, record 8–8–3 over the rated rounds,
~520–690 Elo short of a London seat.

The compiled build nb1 (`628b58a`) beat the live build 46–0–2 at the real clock, and
build C (`f7cd53d`) beat nb1 31–1–4 at the real clock. **C was uploaded and validated
on 9 September at 17:01Z: platform init 21.9 s and 27.5 s of the 90 s budget.** **D was
uploaded and validated at 17:17Z** (init 25.5 s and 23.0 s) after passing its real-clock
check on top of C (15–7–14); `main` is D. E was rejected at both clocks and reverted.

Uploads close **11 September 11:00**; the dashboard caps uploads at **10 per 24 hours**.
**The live build is D**, and `main` is D. E and F were rejected at the real clock and
reverted.

Running overnight on 9 September, seven workers at 120 s + 0.5 s:

- **G against D**, four shards, 240 games — the shipping decision at ±40 rather than ±114
- **Stockfish at 2500 / 2850 / 3190**, three shards — an absolute bracket, and a corpus of
  losses to an engine that plays nothing like us

Both finished on the morning of 10 September. G measured **+6 ±44** — indistinguishable
from D. It ships anyway, on the tie-break that when measurement cannot separate two
builds, the one that is correct by construction wins: a side in check genuinely cannot
stand pat, and answering a check with captures only is simply wrong. 240 games, no
failure of any kind.

Three changes in a row (E, F, G) have now measured flat or negative at the real clock,
each one well motivated. The engine is at the point where reasoning about it no longer
produces gains, which is the argument for fitting the evaluation to data rather than
picking another idea by hand.

**Next, if there is time: tune the evaluation weights against data.** The piece-square
tables are textbook constants and the structure weights were set at "about half the
textbook value" by hand; neither was ever fitted to this search. Logistic regression over
the ~900 game PGNs already on disk is a few hundred parameters, trains in minutes, costs
nothing at runtime and cannot break correctness, because it changes constants and not
code. That is worth more than any search knob, all of which are worth less than we can
measure.

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
