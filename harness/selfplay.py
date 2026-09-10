"""Fast head-to-head between two frozen builds inside one process.

Each build is imported once, so the numba compile is paid once rather than once per
game; the arena spawns two fresh processes per game, which at a fast clock spends more
time compiling than playing. The referee is python-chess with real clocks: a flag
loses, 600 plies draws. Usage:

  uv run python -m harness.selfplay <build_a> <build_b> --games N --start K
      --base-ms 10000 --increment-ms 100 --openings openings.txt --pgn-dir DIR
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType

import chess
import chess.pgn

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
    white: ModuleType, black: ModuleType, fen: str, base_ms: int, increment_ms: int
) -> tuple[str, str, str]:
    board = chess.Board(fen)
    clocks = {chess.WHITE: float(base_ms), chess.BLACK: float(base_ms)}
    agents = {chess.WHITE: white, chess.BLACK: black}
    for module in (white, black):
        module._engine = module.Engine()  # type: ignore[attr-defined]  # fresh state per game
    termination = "ply_cap"
    result = "draw"
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
        try:
            uci = agents[side].get_move(board.fen(), int(clocks[side]))
        except Exception as error:  # any failure by an agent loses it the game
            termination, result = "crash", ("black" if side == chess.WHITE else "white")
            print("crash:", repr(error), file=sys.stderr)
            break
        elapsed = (time.perf_counter() - started) * 1000.0
        clocks[side] -= elapsed
        if clocks[side] < 0:
            termination, result = "flag", ("black" if side == chess.WHITE else "white")
            break
        clocks[side] += increment_ms
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            termination, result = "illegal", ("black" if side == chess.WHITE else "white")
            break
        board.push(move)
    game = chess.pgn.Game.from_board(board)
    game.headers["Result"] = {"white": "1-0", "black": "0-1", "draw": "1/2-1/2"}[result]
    game.headers["Termination"] = termination
    return result, termination, str(game)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--base-ms", type=int, default=10_000)
    parser.add_argument("--increment-ms", type=int, default=100)
    parser.add_argument("--openings", type=Path)
    parser.add_argument("--pgn-dir", type=Path)
    args = parser.parse_args()
    openings = [chess.STARTING_FEN]
    if args.openings:
        lines = args.openings.read_text().splitlines()
        openings = [line.strip() for line in lines if line.strip()]
    a = load(args.a, "agent_a")
    b = load(args.b, "agent_b")
    print("loaded both builds", flush=True)
    wins = draws = losses = 0
    failures = 0
    for game in range(args.start, args.start + args.games):
        a_white = game % 2 == 0
        fen = openings[(game // 2) % len(openings)]
        white, black = (a, b) if a_white else (b, a)
        result, termination, pgn = play(white, black, fen, args.base_ms, args.increment_ms)
        if result == "draw":
            draws += 1
        elif (result == "white") == a_white:
            wins += 1
        else:
            losses += 1
        if termination in ("crash", "flag", "illegal"):
            failures += 1
        if args.pgn_dir:
            args.pgn_dir.mkdir(parents=True, exist_ok=True)
            (args.pgn_dir / f"game_{game + 1:04d}.pgn").write_text(pgn + "\n")
        colour = "white" if a_white else "black"
        print(f"game {game + 1}: {result} by {termination}  [A {colour}]", flush=True)
    print(f"\nRESULT +{wins} ={draws} -{losses} failures {failures}")


if __name__ == "__main__":
    main()
