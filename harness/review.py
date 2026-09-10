"""Stockfish review of a folder of PGNs, in the shape the dashboard shows.

For every position: the evaluation before and after each move, from the mover's side.
Accuracy and the move labels follow the lichess definitions (win probability from the
centipawn score, accuracy from the drop in win probability). Each of our mistakes and
blunders is printed with its position, so it can be replayed by a build.

  uv run python -m harness.review "Chess results" --depth 16 --engine <stockfish.exe>

Analysing our own games with an engine is allowed; shipping one is not, and nothing here
ships. Stockfish installs with `winget install Stockfish.Stockfish`.
"""

import argparse
import glob
import math
import os

import chess
import chess.engine
import chess.pgn

DEFAULT_ENGINE = os.path.join(
    os.environ.get("LOCALAPPDATA", ""),
    "Microsoft",
    "WinGet",
    "Packages",
    "Stockfish.Stockfish_Microsoft.Winget.Source_8wekyb3d8bbwe",
    "stockfish",
    "stockfish-windows-x86-64-universal.exe",
)
US = "Karvaxis"


def win_pct(cp: int) -> float:
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)


def accuracy(before: float, after: float) -> float:
    drop = max(0.0, before - after)
    return max(0.0, min(100.0, 103.1668 * math.exp(-0.04354 * drop) - 3.1669))


def label(drop: float) -> str:
    if drop >= 30:
        return "??"
    if drop >= 20:
        return "?"
    if drop >= 10:
        return "?!"
    return ""


def cp_for(info: chess.engine.InfoDict, colour: chess.Color) -> int:
    score = info["score"].pov(colour).score(mate_score=2000)
    return 0 if score is None else score


def review(
    path: str, engine: chess.engine.SimpleEngine, depth: int
) -> tuple[str, list[str], list[str]]:
    with open(path, encoding="utf-8") as handle:
        game = chess.pgn.read_game(handle)
    if game is None:
        return f"{path}: not a game", [], []
    board = game.board()
    us = chess.WHITE if game.headers["White"] == US else chess.BLACK
    moves = list(game.mainline_moves())
    stats: dict[chess.Color, list[tuple[float, int, str]]] = {chess.WHITE: [], chess.BLACK: []}
    notes = []
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    for move in moves:
        mover = board.turn
        before_cp = cp_for(info, mover)
        best = info.get("pv", [None])[0]
        san = board.san(move)
        fen = board.fen()
        number = f"{board.fullmove_number}{'.' if mover == chess.WHITE else '...'}"
        board.push(move)
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        after_cp = cp_for(info, mover)
        drop = win_pct(before_cp) - win_pct(after_cp)
        loss = max(0, before_cp - after_cp)
        stats[mover].append(
            (accuracy(win_pct(before_cp), win_pct(after_cp)), min(loss, 1000), label(drop))
        )
        if mover == us and label(drop) in ("?", "??"):
            best_san = chess.Board(fen).san(best) if best else "?"
            notes.append(
                f"    {label(drop)} {number} {san}  (best {best_san})  "
                f"{before_cp:+d} -> {after_cp:+d}  {fen}"
            )
    out = []
    sides = ((chess.WHITE, game.headers["White"]), (chess.BLACK, game.headers["Black"]))
    for colour, name in sides:
        s = stats[colour]
        if not s:
            continue
        acc = sum(a for a, _, _ in s) / len(s)
        acpl = sum(loss for _, loss, _ in s) / len(s)
        labels = {k: sum(1 for _, _, x in s if x == k) for k in ("?!", "?", "??")}
        tag = " <- us" if colour == us else ""
        out.append(
            f"  {name:22s} accuracy {acc:5.1f}%  acpl {acpl:4.0f}  "
            f"?! {labels['?!']}  ? {labels['?']}  ?? {labels['??']}{tag}"
        )
    header = (
        f"{game.headers.get('Round')} {game.headers['White']} v {game.headers['Black']} "
        f"{game.headers['Result']} ({game.headers.get('Termination')}, {len(moves)} plies)"
    )
    return header, out, notes


def main() -> None:
    parser = argparse.ArgumentParser(description="Review a folder of PGNs with Stockfish.")
    parser.add_argument("folder")
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--engine", default=DEFAULT_ENGINE)
    args = parser.parse_args()
    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    engine.configure({"Threads": 1, "Hash": 256})
    for path in sorted(glob.glob(f"{args.folder}/*.pgn")):
        header, out, notes = review(path, engine, args.depth)
        print(header)
        print("\n".join(out))
        if notes:
            print("\n".join(notes))
        print(flush=True)
    engine.quit()


if __name__ == "__main__":
    main()
