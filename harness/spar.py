"""Play a build against a UCI engine at a fixed strength, for an absolute measurement.

Every other measurement here is self-play against our own previous build, which says what
beats us, not what beats the field, and gives no absolute rating at all. Stockfish with
UCI_LimitStrength set to a known Elo is an anchor: the score against it converts to a
rating with the usual formula.

  uv run python -m harness.spar <build> --elo 1800 --games 40 --base-ms 120000 \
      --increment-ms 500 --openings openings.txt --pgn-dir DIR

Both sides get the same clock. Analysing and sparring with an engine is allowed; shipping
one is not, and nothing here ships.
"""

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path
from types import ModuleType

import chess
import chess.engine
import chess.pgn

from harness.review import DEFAULT_ENGINE

PLY_CAP = 600


def load(build: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, build / "agent.py")
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {build / 'agent.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def play(
    agent: ModuleType,
    engine: chess.engine.SimpleEngine,
    agent_white: bool,
    fen: str,
    base_ms: int,
    increment_ms: int,
) -> tuple[str, str, str]:
    """One game. Returns the result from white's side, the termination, and the PGN."""
    board = chess.Board(fen)
    clocks = {chess.WHITE: float(base_ms), chess.BLACK: float(base_ms)}
    agent_colour = chess.WHITE if agent_white else chess.BLACK
    agent._engine = agent.Engine()  # type: ignore[attr-defined]  # fresh state per game
    termination, result = "ply_cap", "draw"
    while True:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            termination = outcome.termination.name.lower()
            result = {True: "white", False: "black", None: "draw"}[outcome.winner]
            break
        if board.ply() >= PLY_CAP:
            break
        side = board.turn
        started = time.perf_counter()
        if side == agent_colour:
            try:
                uci = agent.get_move(board.fen(), int(clocks[side]))
            except Exception as error:  # any failure by the agent loses it the game
                termination = "crash"
                result = "black" if side == chess.WHITE else "white"
                print("crash:", repr(error), file=sys.stderr)
                break
            move = chess.Move.from_uci(uci)
        else:
            limit = chess.engine.Limit(
                white_clock=clocks[chess.WHITE] / 1000.0,
                black_clock=clocks[chess.BLACK] / 1000.0,
                white_inc=increment_ms / 1000.0,
                black_inc=increment_ms / 1000.0,
            )
            played = engine.play(board, limit)
            if played.move is None:
                termination = "engine_resigned"
                result = "white" if side == chess.BLACK else "black"
                break
            move = played.move
        clocks[side] -= (time.perf_counter() - started) * 1000.0
        if clocks[side] < 0:
            termination = "flag"
            result = "black" if side == chess.WHITE else "white"
            break
        clocks[side] += increment_ms
        if move not in board.legal_moves:
            termination = "illegal"
            result = "black" if side == chess.WHITE else "white"
            break
        board.push(move)
    game = chess.pgn.Game.from_board(board)
    game.headers["Result"] = {"white": "1-0", "black": "0-1", "draw": "1/2-1/2"}[result]
    game.headers["Termination"] = termination
    game.headers["White"] = "Karvaxis" if agent_white else "Stockfish"
    game.headers["Black"] = "Stockfish" if agent_white else "Karvaxis"
    return result, termination, str(game)


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a build against a UCI engine.")
    parser.add_argument("build", type=Path)
    parser.add_argument("--elo", type=int, default=1800)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--base-ms", type=int, default=120_000)
    parser.add_argument("--increment-ms", type=int, default=500)
    parser.add_argument("--openings", type=Path)
    parser.add_argument("--pgn-dir", type=Path)
    parser.add_argument("--engine", default=DEFAULT_ENGINE)
    args = parser.parse_args()

    openings = [chess.STARTING_FEN]
    if args.openings:
        lines = args.openings.read_text().splitlines()
        openings = [line.strip() for line in lines if line.strip()]

    agent = load(args.build, "agent_under_test")
    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    # one thread and a small table, so the anchor is the rating and not the hardware
    engine.configure(
        {"Threads": 1, "Hash": 64, "UCI_LimitStrength": True, "UCI_Elo": args.elo}
    )
    print(f"loaded build and stockfish at UCI_Elo {args.elo}", flush=True)

    wins = draws = losses = 0
    failures = 0
    for game in range(args.start, args.start + args.games):
        agent_white = game % 2 == 0
        fen = openings[(game // 2) % len(openings)]
        result, termination, pgn = play(
            agent, engine, agent_white, fen, args.base_ms, args.increment_ms
        )
        if result == "draw":
            draws += 1
        elif (result == "white") == agent_white:
            wins += 1
        else:
            losses += 1
        if termination in ("crash", "flag", "illegal"):
            failures += 1
        if args.pgn_dir:
            args.pgn_dir.mkdir(parents=True, exist_ok=True)
            (args.pgn_dir / f"game_{game + 1:04d}.pgn").write_text(pgn + "\n")
        colour = "white" if agent_white else "black"
        print(f"game {game + 1}: {result} by {termination}  [A {colour}]", flush=True)

    engine.quit()
    total = wins + draws + losses
    score = (wins + draws / 2) / total if total else 0.0
    print(f"\nRESULT +{wins} ={draws} -{losses} failures {failures}")
    if 0 < score < 1:
        implied = args.elo + 400 * math.log10(score / (1 - score))
        print(f"score {score:.1%} vs Elo {args.elo} -> implied {implied:.0f}")


if __name__ == "__main__":
    main()
