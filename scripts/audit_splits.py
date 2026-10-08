#!/usr/bin/env python3
"""Audit evaluation/development independence and optional legal distributions.

Rows are identified by (game ID, target ply, target move), rather than by a
split-local row number. Canonical positions use the first four FEN fields.
Game-context keys add game ID and target ply to the canonical position.
Repeated canonical positions are reported, not discarded. If both files have
past_fens, history-context overlap additionally compares that stored history.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
from typing import Any, Iterator

import chess
import numpy as np
import pyarrow.parquet as pq


def as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return json.loads(value) if value.lstrip().startswith("[") else value.split()
    return list(value)


def digest(value: Any) -> bytes:
    return hashlib.blake2b(json.dumps(value, separators=(",", ":")).encode(), digest_size=16).digest()


def canonical_fen(fen: str) -> str:
    fields = fen.split()
    if len(fields) != 6:
        raise ValueError("Expected a complete six-field pre-move FEN")
    return " ".join(fields[:4])


def records(path: Path, columns: list[str] | None = None) -> Iterator[dict]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=4096, columns=columns):
        yield from batch.to_pylist()


@dataclass
class Split:
    name: str
    row_keys: list[bytes]
    games: list[str]
    positions: list[bytes]
    contexts: list[bytes]
    histories: list[bytes | None]
    targets: list[str]
    issues: Counter

    def subset(self, name: str, indices: np.ndarray) -> "Split":
        return Split(name, *[[values[int(i)] for i in indices] for values in
                     [self.row_keys, self.games, self.positions, self.contexts,
                      self.histories, self.targets]], Counter())

    def summary(self) -> dict:
        position_moves: dict[bytes, set[str]] = {}
        for position, target in zip(self.positions, self.targets):
            position_moves.setdefault(position, set()).add(target)
        return {
            "rows": len(self.row_keys), "games": len(set(self.games)),
            "canonical_positions": len(position_moves),
            "duplicate_row_keys": len(self.row_keys) - len(set(self.row_keys)),
            "positions_with_multiple_targets": sum(len(moves) > 1 for moves in position_moves.values()),
            "history_available_rows": sum(key is not None for key in self.histories),
            "issues": dict(self.issues),
        }


def scan_split(path: Path, name: str) -> Split:
    required = {"game_id", "ply_index", "fen_before", "human_move_uci", "legal_moves_uci"}
    available = set(pq.ParquetFile(path).schema_arrow.names)
    if missing := required - available:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    wanted = sorted(required | ({"past_fens"} & available))
    split = Split(name, [], [], [], [], [], [], Counter())
    for index, row in enumerate(records(path, wanted)):
        if any(row[key] is None for key in required):
            raise ValueError(f"{name}: null required field at row {index}")
        game, ply, target = str(row["game_id"]), int(row["ply_index"]), str(row["human_move_uci"])
        fen = canonical_fen(str(row["fen_before"]))
        board = chess.Board(str(row["fen_before"]))
        legal = [str(move) for move in as_list(row["legal_moves_uci"])]
        actual = {move.uci() for move in board.legal_moves}
        split.issues["invalid_board"] += int(not board.is_valid())
        split.issues["duplicate_legal_moves"] += int(len(legal) != len(set(legal)))
        split.issues["legal_set_mismatch"] += int(set(legal) != actual)
        split.issues["target_not_legal"] += int(target not in actual or target not in legal)
        split.issues["empty_game_id"] += int(not game.strip())
        split.issues["negative_ply"] += int(ply < 0)
        split.row_keys.append(digest([game, ply, target]))
        split.games.append(game)
        split.positions.append(digest(fen))
        split.contexts.append(digest([game, ply, fen]))
        split.targets.append(target)
        history = row.get("past_fens")
        split.histories.append(digest([fen, [canonical_fen(str(item)) for item in as_list(history)]])
                               if history is not None else None)
    split.issues = +split.issues
    if not split.row_keys:
        split.issues["empty_dataset"] += 1
    return split


def overlaps(first: Split, second: Split) -> dict:
    result = {"first": first.name, "second": second.name}
    for field, label in [("row_keys", "row_overlap"), ("games", "game_overlap"),
                         ("positions", "canonical_overlap"), ("contexts", "game_context_overlap")]:
        result[label] = len(set(getattr(first, field)) & set(getattr(second, field)))
    history_a = {value for value in first.histories if value is not None}
    history_b = {value for value in second.histories if value is not None}
    result["stored_history_context_overlap"] = len(history_a & history_b) if history_a and history_b else None
    return result


def audit_predictions(dataset: Path, predictions: Path, tolerance: float = 1e-5) -> dict:
    """Require prediction row_id to identify the original source row ordinal."""
    required = {"row_id", "human_move_uci", "legal_moves_uci", "legal_probs"}
    available = set(pq.ParquetFile(predictions).schema_arrow.names)
    if missing := required - available:
        raise ValueError(f"Predictions missing columns {sorted(missing)}")
    issues: Counter = Counter()
    seen: set[int] = set()
    maximum_error = 0.0
    rows = 0
    optional = {"game_id", "p_human", "p_top1", "top1_move_uci", "human_rank", "legal_logits"} & available
    source = records(dataset, ["game_id", "human_move_uci", "legal_moves_uci"])
    for index, (row, pred) in enumerate(zip_longest(source, records(predictions, sorted(required | optional)))):
        if row is None or pred is None:
            issues["row_count_mismatch"] += 1
            continue
        rows += 1
        row_id = pred["row_id"]
        issues["duplicate_row_id"] += int(row_id in seen)
        seen.add(row_id)
        issues["row_id_order_mismatch"] += int(row_id != index)
        issues["target_mismatch"] += int(pred["human_move_uci"] != row["human_move_uci"])
        if "game_id" in pred:
            issues["game_id_mismatch"] += int(pred["game_id"] != row["game_id"])
        moves = [str(move) for move in as_list(pred["legal_moves_uci"])]
        probs = np.asarray(as_list(pred["legal_probs"]), dtype=np.float64)
        issues["duplicate_legal_moves"] += int(len(moves) != len(set(moves)))
        issues["legal_set_mismatch"] += int(set(moves) != set(as_list(row["legal_moves_uci"])))
        if not moves or probs.shape != (len(moves),):
            issues["probability_shape"] += 1
            continue
        if not np.isfinite(probs).all() or (probs < 0).any():
            issues["invalid_probabilities"] += 1
            continue
        error = abs(float(probs.sum()) - 1.0)
        maximum_error = max(maximum_error, error)
        issues["probability_sum"] += int(error > tolerance)
        target = str(row["human_move_uci"])
        if target not in moves:
            issues["target_not_legal"] += 1
            continue
        human_p = float(probs[moves.index(target)])
        issues["zero_human_probability"] += int(human_p == 0.0)
        for field, expected in [("p_human", human_p), ("p_top1", float(probs.max()))]:
            if field in pred:
                issues[field + "_mismatch"] += int(not np.isclose(float(pred[field]), expected, atol=tolerance, rtol=0))
        if "top1_move_uci" in pred:
            move = pred["top1_move_uci"]
            issues["top1_mismatch"] += int(move not in moves or probs[moves.index(move)] != probs.max())
        if "human_rank" in pred:
            scores = np.asarray(as_list(pred["legal_logits"]), dtype=np.float64) if "legal_logits" in pred else probs
            if scores.shape != probs.shape or not np.isfinite(scores).all():
                issues["invalid_logits"] += 1
            else:
                expected_rank = 1 + int((scores > scores[moves.index(target)]).sum())
                issues["rank_mismatch"] += int(int(pred["human_rank"]) != expected_rank)
    return {"rows": rows, "max_probability_sum_error": maximum_error, "issues": dict(+issues)}


def selector_indices(path: Path, total: int, family: str = "selector") -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        train = np.asarray(data[f"{family}_train"] if f"{family}_train" in data else data["train"])
        val = np.asarray(data[f"{family}_val"] if f"{family}_val" in data else data["validation"])
    for name, values in [(f"{family}_train", train), (f"{family}_val", val)]:
        if values.ndim != 1 or values.dtype.kind not in "iu":
            raise ValueError(f"{name} must be a one-dimensional integer array")
        if len(set(values.tolist())) != len(values) or (values < 0).any() or (values >= total).any():
            raise ValueError(f"{name} contains duplicate or out-of-bounds indices")
    if set(train.tolist()) & set(val.tolist()):
        raise ValueError(f"{family.capitalize()} training and validation row indices overlap")
    return train, val


def build_report(args: argparse.Namespace) -> dict:
    test = scan_split(args.test, "test")
    heldout = scan_split(args.heldout, "heldout")
    splits = [test, heldout]
    development_pairs = []
    for family in ("selector", "refiner"):
        split_path = getattr(args, f"{family}_split", None)
        if split_path is not None:
            train, val = selector_indices(split_path, len(heldout.row_keys), family)
            pair = (heldout.subset(f"{family}_train", train), heldout.subset(f"{family}_val", val))
            splits.extend(pair)
            development_pairs.append(pair)
    comparisons = [overlaps(test, split) for split in splits[1:]]
    comparisons.extend(overlaps(train, val) for train, val in development_pairs)
    predictions = {}
    for split_name, path in [("test", args.test), ("heldout", args.heldout)]:
        for model in ["maia3", "allie"]:
            prediction = getattr(args, f"{split_name}_{model}", None)
            if prediction is not None:
                predictions[f"{split_name}_{model}"] = audit_predictions(path, prediction)
    summaries = {split.name: split.summary() for split in splits}
    passed = not any(summary["issues"] or summary["duplicate_row_keys"] for summary in summaries.values())
    passed = passed and not any(row["row_overlap"] or row["game_overlap"] for row in comparisons if row["first"] == "test")
    passed = passed and not any(result["issues"] for result in predictions.values())
    return {
        "passed": bool(passed), "splits": summaries, "overlaps": comparisons,
        "predictions": predictions,
        "definitions": {
            "row": "game_id, ply_index, human_move_uci",
            "canonical_position": "first four FEN fields",
            "game_context": "game_id, ply_index, canonical_position",
            "stored_history_context": "canonical_position plus stored past_fens; not necessarily the complete game history",
            "independence": "test shares no row or game with development; repeated cross-game positions are allowed",
            "selector_train_validation": "disjoint rows; game overlap reported, as the paper uses a row split",
            "refiner_train_validation": "disjoint rows; game overlap reported, as the paper uses a row split",
            "prediction_alignment": "ordered row_id equals source row ordinal; legal move order may differ",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--selector-split", type=Path, help="NPZ with train/validation (or selector_train/selector_val) row indices")
    parser.add_argument("--refiner-split", type=Path, help="NPZ with train/validation (or refiner_train/refiner_val) row indices")
    parser.add_argument("--output", type=Path, required=True)
    for split in ["test", "heldout"]:
        for model in ["maia3", "allie"]:
            parser.add_argument(f"--{split}-{model}", type=Path, help="Optional complete legal-distribution parquet")
    args = parser.parse_args()
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {args.output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
