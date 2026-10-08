"""Position conversion for the shared test protocol and exact development split."""
from __future__ import annotations

import json
import math
from collections import deque
from itertools import islice
from pathlib import Path
from typing import Any, Iterator

import chess
import pyarrow as pa
import pyarrow.parquet as pq

from .upstream import ASSETS, HELDOUT_SHA256, verify_hash


def string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    text = str(value).strip()
    return [str(item) for item in json.loads(text)] if text.startswith("[") else text.split()


def phase_for_ply(ply_index: int) -> str:
    move_number = ply_index // 2 + 1
    return "opening" if move_number <= 10 else "middlegame" if move_number <= 40 else "endgame"


def validate_position(row: dict[str, Any], *, replay_prefix: bool = False) -> chess.Board:
    board = chess.Board(row["fen_before"])
    legal = string_list(row["legal_moves_uci"])
    expected = [move.uci() for move in board.legal_moves]
    if len(legal) != len(set(legal)) or set(legal) != set(expected):
        raise ValueError(f"Legal-move mismatch in game {row['game_id']}")
    if row["human_move_uci"] not in legal:
        raise ValueError(f"Illegal target in game {row['game_id']}")
    if replay_prefix:
        replayed = chess.Board()
        for move in string_list(row["previous_moves_uci"]):
            replayed.push_uci(move)
        if replayed.fen() != board.fen():
            raise ValueError(f"History/FEN mismatch in game {row['game_id']}")
    return board


def rows_for_game(game: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Keep ply 11 onward, stopping before either player's clock drops below 30s.

    Clock checks also run during the first ten plies, matching the original
    converter. A clock of exactly 30 seconds passes. Times are durations, so
    clock_after = clock_before + increment - duration.
    """
    moves = str(game["moves-uci"]).split()
    durations = game["moves-seconds"]
    if len(moves) != len(durations):
        raise ValueError(f"Move/time length mismatch: {game['game-id']}")
    base, increment = map(int, str(game["time-control"]).split("+"))
    if base < 0 or increment < 0:
        raise ValueError("Invalid time control")
    clocks = [float(base), float(base)]
    white, black = int(game["white-elo"]), int(game["black-elo"])
    board = chess.Board()
    past_fens: deque[str] = deque(maxlen=7)
    for ply, (move, duration) in enumerate(zip(moves, durations, strict=True)):
        side = 0 if board.turn == chess.WHITE else 1
        before = clocks[side]
        if before < 30:
            break
        # Preserve finite source durations, including negative clock corrections;
        # the paper converter did not clamp or filter them.
        if not math.isfinite(float(duration)):
            raise ValueError(f"Invalid move duration in {game['game-id']}")
        if chess.Move.from_uci(move) not in board.legal_moves:
            raise ValueError(f"Illegal move at ply {ply}: {game['game-id']}")
        if ply >= 10:
            player, opponent = (white, black) if side == 0 else (black, white)
            legal = [candidate.uci() for candidate in board.legal_moves]
            yield {
                "game_id": str(game["game-id"]),
                "fen_before": board.fen(),
                "human_move_uci": move,
                "legal_moves_uci": json.dumps(legal),
                "num_legal_moves": len(legal),
                "player_elo": player,
                "opponent_elo": opponent,
                "white_elo": white,
                "black_elo": black,
                "rating_diff": player - opponent,
                "rating_bucket": f"{player // 200 * 200}-{player // 200 * 200 + 200}",
                "player_color": "white" if side == 0 else "black",
                "move_number": board.fullmove_number,
                "ply_index": ply,
                "target_ply_index": ply,
                "time_control": str(game["time-control"]),
                "initial_time_seconds": float(base),
                "increment_seconds": float(increment),
                "clock_before_seconds": before,
                "time_spent_seconds": float(duration),
                "past_fens": json.dumps(list(past_fens)),
                "previous_moves_uci": " ".join(moves[:ply]),
                "previous_move_seconds": [int(value) for value in durations[:ply]],
                "phase": phase_for_ply(ply),
            }
        past_fens.append(board.fen())
        board.push_uci(move)
        clocks[side] = before + increment - float(duration)


def iter_parquet(path: Path, batch_size: int = 8192) -> Iterator[dict[str, Any]]:
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def heldout_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Preserve published row order and MAIA3 history, adding Allie's zero-time prefix.

    Original heldout ply_index is one-based; Allie's target index is instead
    derived from the complete move prefix. Original past_fens may have eight
    entries; MAIA3 consumes the last seven before the current position.
    """
    keys = (
        "game_id", "fen_before", "human_move_uci", "legal_moves_uci", "num_legal_moves",
        "player_elo", "opponent_elo", "white_elo", "black_elo", "rating_diff",
        "rating_bucket", "player_color", "move_number", "ply_index", "time_control",
        "initial_time_seconds", "increment_seconds", "clock_before_seconds",
        "time_spent_seconds", "past_fens", "previous_moves_uci", "phase",
    )
    for row in iter_parquet(path):
        validate_position(row, replay_prefix=True)
        result = {key: row[key] for key in keys}
        target_ply = len(string_list(row["previous_moves_uci"]))
        result["target_ply_index"] = target_ply
        result["previous_move_seconds"] = [0] * target_ply
        yield result


def prepare_data(input_path: Path, output_path: Path, source_format: str,
                 max_rows: int | None = None, expected_rows: int | None = None,
                 verify_source: bool = True) -> dict[str, Any]:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must differ")
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be positive")
    construction = {"source_positions": 0, "source_games": 0,
                    "after_excluding_plies_1_to_10_positions": 0,
                    "after_excluding_plies_1_to_10_games": 0}
    if source_format == "allie-jsonl":
        if verify_source:
            verify_hash(input_path, ASSETS["allie-test.jsonl"][-1])

        def source_rows():
            seen = set()
            with input_path.open(encoding="utf-8") as handle:
                for line in handle:
                    game = json.loads(line)
                    if game["game-id"] in seen:
                        raise ValueError(f"Duplicate game: {game['game-id']}")
                    seen.add(game["game-id"])
                    count = len(str(game["moves-uci"]).split())
                    construction["source_positions"] += count
                    construction["source_games"] += 1
                    construction["after_excluding_plies_1_to_10_positions"] += max(count - 10, 0)
                    construction["after_excluding_plies_1_to_10_games"] += count > 10
                    yield from rows_for_game(game)
        source = source_rows()
        default_expected = 884049
    elif source_format == "heldout-parquet":
        if verify_source:
            verify_hash(input_path, HELDOUT_SHA256)
        source = heldout_rows(input_path)
        default_expected = 500000
    else:
        raise ValueError(f"Unknown format: {source_format}")
    if expected_rows is None:
        expected_rows = max_rows if max_rows is not None else default_expected
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    writer = None
    rows_written = 0
    game_ids = set()
    pending = []
    try:
        for row in islice(source, max_rows):
            row["row_id"] = rows_written
            pending.append(row)
            game_ids.add(row["game_id"])
            rows_written += 1
            if len(pending) == 8192:
                table = pa.Table.from_pylist(pending)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
                writer.write_table(table)
                pending.clear()
        if pending:
            table = pa.Table.from_pylist(pending)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
        if rows_written != expected_rows:
            raise ValueError(f"Expected {expected_rows} positions, got {rows_written}")
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(output_path)
    summary = {"source_format": source_format, "rows": rows_written, "games": len(game_ids),
               "smoke_subset": max_rows is not None, "source_hash_verified": verify_source,
               "previous_times": "actual" if source_format == "allie-jsonl" else "zeros (paper development protocol)"}
    if source_format == "allie-jsonl":
        summary["construction"] = {**construction,
                                   "after_clock_truncation_positions": rows_written,
                                   "after_clock_truncation_games": len(game_ids)}
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
