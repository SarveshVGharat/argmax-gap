#!/usr/bin/env python3
"""Reproduce Appendix F's source-versus-retained filtering sensitivity (Table 30)."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import chess
import numpy as np
import pandas as pd
from tqdm import tqdm

from argmax_gap.data import phase_for_ply
from argmax_gap.upstream import ASSETS, verify_hash


CONTINUOUS = [
    "player_elo", "opponent_elo", "move_number", "initial_time_seconds",
    "increment_seconds", "time_spent_seconds", "clock_before_seconds", "num_legal_moves",
]
CATEGORICAL = ["phase", "time_control", "initial_time_seconds", "increment_seconds", "player_color"]
SOURCE_ROWS = 1_239_353
RETAINED_ROWS = 884_049


def reconstruct_source(path: Path, retained: pd.DataFrame):
    """Replay every source move, including the opening and all low-clock tails."""
    values = np.empty((SOURCE_ROWS, len(CONTINUOUS)), dtype=np.float64)
    categories = {key: Counter() for key in CATEGORICAL}
    expected_keys = iter(zip(retained.game_id.astype(str), retained.ply_index.astype(int),
                             retained.human_move_uci.astype(str), strict=True))
    seen_games, retained_games = set(), set()
    count = kept = after_opening = after_opening_games = 0
    with path.open(encoding="utf-8") as handle:
        for line in tqdm(handle, total=18_239, desc="source games", mininterval=5):
            game = json.loads(line)
            game_id = str(game["game-id"])
            if game_id in seen_games:
                raise ValueError(f"Duplicate source game: {game_id}")
            seen_games.add(game_id)
            moves, durations = game["moves-uci"].split(), game["moves-seconds"]
            if len(moves) != len(durations):
                raise ValueError(f"Move/time length mismatch: {game_id}")
            base, increment = map(int, game["time-control"].split("+"))
            clocks = [float(base), float(base)]
            board = chess.Board()
            truncated = False
            after_opening_games += len(moves) > 10
            for ply, (move, duration) in enumerate(zip(moves, durations, strict=True)):
                side = 0 if board.turn == chess.WHITE else 1
                before = clocks[side]
                legal_moves = [candidate.uci() for candidate in board.legal_moves]
                if move not in legal_moves:
                    raise ValueError(f"Illegal source move: {game_id}, ply {ply}")
                truncated |= before < 30.0
                if ply >= 10:
                    after_opening += 1
                    if not truncated:
                        if next(expected_keys, None) != (game_id, ply, move):
                            raise ValueError(f"Retained key/order mismatch: {game_id}, ply {ply}")
                        kept += 1
                        retained_games.add(game_id)
                player = int(game["white-elo"] if side == 0 else game["black-elo"])
                opponent = int(game["black-elo"] if side == 0 else game["white-elo"])
                if count >= SOURCE_ROWS:
                    raise ValueError("Unexpected source position count")
                values[count] = [player, opponent, board.fullmove_number, base, increment,
                                 float(duration), before, len(legal_moves)]
                for key, value in {
                    "phase": phase_for_ply(ply), "time_control": str(game["time-control"]),
                    "initial_time_seconds": str(float(base)), "increment_seconds": str(float(increment)),
                    "player_color": "white" if side == 0 else "black",
                }.items():
                    categories[key][value] += 1
                count += 1
                # Negative clock corrections and zero-base controls remain in
                # the unfiltered source, exactly as in the original audit.
                clocks[side] = before + increment - float(duration)
                board.push_uci(move)
    if count != SOURCE_ROWS or kept != RETAINED_ROWS or next(expected_keys, None) is not None:
        raise ValueError(f"Unexpected source/retained counts: {count}/{kept}")
    summary = {
        "source_positions": count, "source_games": len(seen_games),
        "after_opening_exclusion_positions": after_opening,
        "after_opening_exclusion_games": after_opening_games,
        "retained_positions": kept, "retained_games": len(retained_games),
        "retained_keys_and_order_match": True,
    }
    return values, categories, summary


def continuous_summary(source: pd.Series, retained: pd.Series, variable: str):
    source = pd.to_numeric(source, errors="coerce").dropna().astype(float)
    retained = pd.to_numeric(retained, errors="coerce").dropna().astype(float)
    source_sd, retained_sd = float(source.std(ddof=1)), float(retained.std(ddof=1))
    pooled_sd = math.sqrt((source_sd ** 2 + retained_sd ** 2) / 2.0)
    row = {"variable": variable, "type": "continuous"}
    for name, values in [("source", source), ("retained", retained)]:
        row.update({f"{name}_n": len(values), f"{name}_mean": float(values.mean()),
                    f"{name}_sd": float(values.std(ddof=1)), f"{name}_median": float(values.median()),
                    f"{name}_p10": float(values.quantile(.1)), f"{name}_p90": float(values.quantile(.9))})
    row["standardized_mean_difference"] = float((retained.mean() - source.mean()) / pooled_sd) if pooled_sd else 0.0
    return row


def compare_reference(frame: pd.DataFrame, reference: Path) -> pd.DataFrame:
    expected = pd.read_csv(reference)
    expected = expected[expected["table"].astype(str) == "30"]
    actual = {}
    for row in frame.to_dict("records"):
        item = row["variable"] if row["type"] == "continuous" else f"{row['variable']}/{row['level']}"
        for metric, value in row.items():
            actual[item, metric] = value
    rows = []
    for row in expected.to_dict("records"):
        value = float(actual[row["item"], row["metric"]])
        difference = value - float(row["value"])
        rows.append({"item": row["item"], "metric": row["metric"], "computed": value,
                     "reference": row["value"], "difference": difference})
        if not math.isclose(value, float(row["value"]), rel_tol=1e-12, abs_tol=5e-10):
            raise ValueError(f"Table 30 reference mismatch: {row['item']}, {row['metric']}")
    if len(rows) != 21:
        raise ValueError(f"Expected 21 Table 30 reference values, found {len(rows)}")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--retained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Optional reference/provenance.csv validation")
    args = parser.parse_args()
    verify_hash(args.source_jsonl, ASSETS["allie-test.jsonl"][-1])
    columns = list(dict.fromkeys(["game_id", "ply_index", "human_move_uci", *CONTINUOUS, *CATEGORICAL]))
    retained = pd.read_parquet(args.retained, columns=columns)
    if len(retained) != RETAINED_ROWS:
        raise ValueError(f"Expected {RETAINED_ROWS} retained positions")
    source, categories, summary = reconstruct_source(args.source_jsonl, retained)
    rows = [continuous_summary(pd.Series(source[:, i]), retained[key], key)
            for i, key in enumerate(CONTINUOUS)]
    for key in CATEGORICAL:
        source_counts = categories[key]
        retained_counts = retained[key].fillna("NA").astype(str).value_counts()
        levels = sorted(set(name for name, _ in source_counts.most_common(20)) | set(retained_counts.head(20).index))
        for level in levels:
            source_share = source_counts[level] / SOURCE_ROWS
            retained_share = int(retained_counts.get(level, 0)) / RETAINED_ROWS
            rows.append({"variable": key, "level": level, "type": "categorical",
                         "source_share_percent": 100 * source_share,
                         "retained_share_percent": 100 * retained_share,
                         "absolute_percentage_point_change": 100 * (retained_share - source_share)})
    frame = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.reference:
        comparison = compare_reference(frame, args.reference)
        comparison.to_csv(args.output / "reference_comparison.csv", index=False)
        summary["table30_reference_values_matched"] = len(comparison)
        summary["table30_max_absolute_difference"] = float(comparison.difference.abs().max())
    frame.to_csv(args.output / "filtering_sensitivity.csv", index=False)
    (args.output / "source_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
