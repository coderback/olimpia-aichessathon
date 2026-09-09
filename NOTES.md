# Olimpia — build notes

What we built for AI Chessathon, why it is shaped this way, what we measured, and
what we got wrong. Written so anyone on the team can pick this up cold.

Live build: `579889a`. `main` carries a revert on top of it, so `main` and the live
build are byte-identical. The submission is `agent.py` alone; nothing else ships.

---

## 1. What the agent is

A classical alpha-beta engine in pure Python on top of `python-chess`. No neural
network — the rules allow a classical search as a full entry, and see §6 for why we
never got near one.

**Search**

| piece | what it does |
|---|---|
| negamax + alpha-beta | the whole game |
| iterative deepening | always a move to return when the clock runs out |
| transposition table | per move, carried across deepening passes |
| MVV-LVA ordering | captures first, most valuable victim first |
| killer moves | quiet refutations tried early at the same ply |
| history heuristic | quiet moves scored by how deep their cutoffs were |
| principal variation search | narrow window after the first move |
| null-move pruning | guarded on zugzwang: not in check, not pawns-only |
| late move reductions | quiet moves far down the list searched shallower |
| quiescence | captures only, so leaves are never scored mid-exchange |
| delta pruning | skip captures that cannot reach alpha even winning outright |
| futility pruning | skip quiet moves near the leaves that cannot reach alpha |

**Evaluation** — material plus piece-square tables, tapered between a middlegame and
an endgame king table by game phase, plus a mop-up term that drives a bare king to
the edge in a won endgame. That is all. Every attempt to add more measured worse
(§4).

**Time management** — `_budget_ms` sizes each move from the clock it is handed; the
search checks its deadline every 16 nodes; `get_move` predicts whether another
deepening pass can finish from the measured cost of the last one; an aborted pass
still contributes its best root move. Below a 100 ms clock it returns the
best-ordered move without searching at all (§3).

---

## 2. Measured results

Everything below is our build against our own previous build, alternating colours.
Elo figures carry 95% intervals.

| change | fast clock (5s) | real clock (120s) | outcome |
|---|---|---|---|
| TT + PVS + history + null-move | +37 | — | shipped |
| pass prediction + partial-pass salvage | ~0 | — | shipped on curve shape |
| **six evaluation terms** | **−102** | — | **rejected** |
| late move reductions | +124 | — | shipped |
| LMR parameter sweep | unresolved | — | no change made |
| futility + delta pruning | +36 (1008 games) | — | shipped |
| whole build vs v1 | +69 | **+280** | — |
| **king danger blend** | **+38** (1008 games) | **−46** | **reverted** |
| **reverse futility + LMP** | **+127** | **−44** | **rejected** |

Roughly 5,000 arena games. Four of the last five changes attempted were rejected on
evidence.

**Robustness:** 5,184 hardening positions and 132 real-clock games with zero
crashes, illegal moves, or flags. Ruff and mypy strict clean.

---

## 3. Architectural decisions, and why

**No pondering.** The platform suspends our process while the opponent moves. The
starter's `AGENTS.md` says the opposite; the live docs are authoritative and the
validation log states it explicitly. Building pondering would have been wasted.

**Transposition table is per move, not per game.** A score stored under a shorter
game history can contradict the repetition check once that history grows. Carrying
it across deepening passes is where nearly all the value is anyway.

**Repetition awareness.** The platform hands us a bare FEN with no history, so
`_history` accumulates the positions we have been asked about and a line returning
to one scores as the draw the referee will claim. This is deliberate and it has
already saved half a point: in round 64 we were down 200cp and drew by threefold.

**`evaluate` walks bitboards, not `board.piece_map()`.** The latter builds a `Piece`
object per occupied square. Cost of the naive version was a large fraction of search
time.

**The king-table blend is integer.** A float blend produced off-by-one scores in 4
of 8,656 positions through truncation. Integer arithmetic makes the evaluation exact
and reproducible.

**Panic guard below a 100 ms clock.** One search pass costs ~16 ms however small the
budget says it is, because a board, a move list and one ply have to happen at all.
`MIN_BUDGET_MS = 5` was a promise the code could not keep. The referee flags the
moment elapsed exceeds the *clock*, not the budget — the 500 ms watchdog grace only
governs the hard kill. Under a corrected check the pre-guard build lost on time in
197 of 864 low-clock positions.

**`time.perf_counter`, not `time.monotonic`.** On Windows `monotonic` resolves to
15.6 ms, so every local timing measurement was quantised to one tick. Linux is
nanosecond-resolution so this never affected the platform, but it made our own
numbers meaningless until it was fixed.

**Harness corrections.** `harness/` was changed only to match the published rules,
which the starter had drifted from: 600-ply draw rather than 300-ply material
adjudication, a flag draws when the other side cannot mate, and a failed game is
blamed on the side that actually failed rather than always on us.

---

## 4. What we tried and rejected

**Six evaluation terms** (passed/doubled/isolated pawns, bishop pair, rook files,
king shield). Textbook-correct, verified against hand-built positions and 1,800
symmetry checks, cheap in nodes. **−102 Elo over 204 games.** Not slow — the build
reached the same depth. Simply miscalibrated. The likely culprit is the passed-pawn
scale stacking on a pawn table that already rewards advancement.

**King danger blend.** Motivated by real evidence: four of five rated losses had our
king three to five ranks advanced with the enemy queen alive. The diagnosis was
sound and is probably still correct — `phase` counted material for *both* sides, so
trading our own pieces away shifted our king towards the endgame table and paid it
to march into a live queen. Shipped on +38 over 1,008 fast games under a decision
rule fixed in advance, then measured **−46 at the real clock** and reverted.

**Reverse futility + late move pruning.** Applied at every node, **−178 Elo**.
Restricted to non-PV nodes, **+127** at the fast clock and **−44** at the real one.
Rejected. Kept on branch `rejected/shallow-pruning`.

---

## 5. How to measure, and the trap we fell into

**The trap:** we ran essentially all of day one at 5 s + 0.1 s for throughput.
Fast-clock results **do not predict real-clock results**, in either direction:

- whole build vs v1: +69 fast, **+280** real
- king danger: +38 fast, **−46** real
- shallow pruning: +127 fast, **−44** real

We first believed fast testing was conservative for search changes and safe to rely
on. It is not. Both of the changes we shipped and later reverted were validated at a
clock the platform never uses.

**The rule that follows:** a change ships only if it is not negative at
**120 s + 0.5 s**. Fast runs are a screen for filtering candidates, never a gate.

**Practical mechanics**

- **Freeze the builds first.** `harness.arena` defaults to `--agent .`, which imports
  `agent.py` from the live working tree at every game. Editing during a run corrupts
  it — this produced four phantom "crash" terminations before we noticed. Copy each
  build to its own directory and point the arena at those.
- **Shard across cores.** The machine has 8 physical cores and a single arena uses
  one. Six parallel shards is ~6× throughput. Contention slows both sides equally so
  head-to-head comparison stays fair; it only corrupts *absolute* timing numbers.
- **Detach long runs.** `nohup ... &` plus `disown`. Tracked background tasks were
  being stopped mid-run and taking the agent processes with them.
- **Fix the decision rule before seeing the data**, and make it specify the *time
  control*, not just the sample size. Ours specified games and forgot the clock,
  which is exactly how the king change shipped.
- **Sample sizes.** ±80 Elo needs ~50 games, ±40 needs ~200, ±30 needs ~400. A
  real-clock experiment costs roughly 2 hours for ±80.

**Node counts are not evidence of strength.** They pointed the wrong way three
times. Only games at the right clock decide.

---

## 6. Constraints that shaped this

From the live docs at `aichessathon.com/docs` — authoritative and they change, so
re-fetch rather than trusting notes:

- `agent.py` at the zip root, `get_move(fen, time_left_ms) -> str`, UCI out
- 120 s + 0.5 s per move, one core of an EPYC 9V74, 2 GB, no network, no GPU
- 90 s init budget — **we use 0.6 s of it**, which is a large unspent asset
- Only torch, numpy, python-chess, onnxruntime, numba are available
- Game drawn at 600 plies; flag draws when the other side cannot mate
- No third-party engine or published network, no shipped table of another engine's
  moves or evaluations. Labelling training positions with an engine *is* allowed
- Source a judge can read; obfuscation is disqualifying

**Why no neural network.** It is downstream of speed, not an alternative to it. Our
node costs ~76 µs of which evaluation is ~16 µs; an NNUE in numpy lands at 20–40 µs,
so a net would cost depth to buy evaluation quality — the exact trade our two
evaluation experiments lost. And the training literature puts a net at roughly a
month of CPU work. It only becomes viable on top of a much faster substrate.

---

## 7. Where we stand and what is left

Ladder rating 1518, rank #241 of 418. Roughly 82% of the field is Swiss-eligible, so
50 London seats fill down to about overall rank 40–60, or a rating near 2040–2205.
**We are 520–690 Elo short of a seat.** That gap is not closeable by tuning.

**The one lever with the right order of magnitude is nodes per second.** We run
~13,000. Published numba engines report 200× that, and their search feature lists
are almost identical to ours — so our search logic is competitive and the entire gap
is the board representation. `python-chess` is explicitly not built for engine use.

If anyone picks this up with real time available, the order is:

1. Own board representation and move generator, numba-compiled, validated by
   **perft against python-chess** (this makes movegen bugs loud, not silent — the
   risk we wrongly treated as decisive)
2. Port this search onto it unchanged
3. Only then consider a learned evaluation

Init compile of 20–45 s fits inside the 90 s budget we are barely using.

---

## 8. Running things

```
make play                                   # one game against a baseline, real clock
make arena                                  # 20 fast games with a score
make zip                                    # build the submission
make gate                                   # ruff, mypy, two games that must finish
uv run python -m harness.arena --agent <frozen-build> --opponent <frozen-build> \
    --games 42 --base-ms 120000 --increment-ms 500     # the real gate
```

The harness is local only; nothing in it ships. The platform's validation log on the
dashboard is the authority on whether an upload is accepted.
