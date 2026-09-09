import argparse
from pathlib import Path

import chess

from harness.referee import FAILED_TERMINATIONS, play_match
from harness.rules import PLY_CAP
from harness.sandbox import local

FAST_BASE_MS = 10_000
FAST_INCREMENT_MS = 100


def main() -> None:
    parser = argparse.ArgumentParser(description="Score an agent over several games.")
    parser.add_argument("--agent", type=Path, default=Path("."))
    parser.add_argument("--opponent", type=Path, default=Path("baselines/greedy"))
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--base-ms", type=int, default=FAST_BASE_MS)
    parser.add_argument("--increment-ms", type=int, default=FAST_INCREMENT_MS)
    parser.add_argument("--ply-cap", type=int, default=PLY_CAP)
    parser.add_argument(
        "--openings",
        type=Path,
        help="file of FENs, one per line; each is played twice with colours swapped",
    )
    parser.add_argument(
        "--start", type=int, default=0, help="index of the first game, so shards differ"
    )
    parser.add_argument("--pgn-dir", type=Path, help="write each game as a PGN here")
    arguments = parser.parse_args()
    openings = [chess.STARTING_FEN]
    if arguments.openings:
        lines = arguments.openings.read_text().splitlines()
        openings = [line.strip() for line in lines if line.strip() and not line.startswith("#")]

    agent = arguments.agent.resolve()
    opponent = arguments.opponent.resolve()
    wins = draws = losses = 0
    terminations: dict[str, int] = {}
    broken: dict[str, int] = {}

    for game in range(arguments.start, arguments.start + arguments.games):
        plays_white = game % 2 == 0
        start_fen = openings[(game // 2) % len(openings)]
        white, black = (agent, opponent) if plays_white else (opponent, agent)
        outcome = play_match(
            local(white),
            local(black),
            arguments.base_ms,
            arguments.increment_ms,
            ply_cap=arguments.ply_cap,
            start_fen=start_fen,
        )
        terminations[outcome.termination] = terminations.get(outcome.termination, 0) + 1
        undecided = outcome.result == "draw" or outcome.result == "void"
        if undecided:
            draws += 1
        elif (outcome.result == "white") == plays_white:
            wins += 1
        else:
            losses += 1
        # a failed termination is only ours when we lost by it, or when both sides failed
        if outcome.termination in FAILED_TERMINATIONS and (
            outcome.result == "void"
            or (not undecided and (outcome.result == "white") != plays_white)
        ):
            broken[outcome.termination] = broken.get(outcome.termination, 0) + 1
        if arguments.pgn_dir:
            arguments.pgn_dir.mkdir(parents=True, exist_ok=True)
            (arguments.pgn_dir / f"game_{game + 1:04d}.pgn").write_text(outcome.pgn + "\n")
        print(f"game {game + 1}: {outcome.result} by {outcome.termination}", flush=True)

    score = (wins + draws / 2) / arguments.games
    print(f"\n{arguments.agent} vs {arguments.opponent} over {arguments.games} games")
    print(f"+{wins} ={draws} -{losses}, score {score:.1%}")
    print("terminations: " + ", ".join(f"{name} {count}" for name, count in terminations.items()))
    if broken:
        raise SystemExit(
            "your agent failed to finish a game: "
            + ", ".join(f"{name} {count}" for name, count in broken.items())
        )


if __name__ == "__main__":
    main()
