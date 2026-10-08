"""Audited pre-move candidate features and explicitly opt-in duration diagnostics."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

TIME_EDGES = [1.0, 2.0, 4.0, 7.0]


TIME_LABELS = ["[0,1]", "(1,2]", "(2,4]", "(4,7]", "(7,inf)"]


DIAGNOSTIC_FEATURES = [
    "player_elo",
    "opponent_elo",
    "elo_diff",
    "time_spent_seconds",
    "log_time_spent",
    "time_bucket_code",
    "move_number",
    "phase_code",
    "num_legal_moves",
    "maia3_p_top1",
    "maia3_entropy",
    "maia3_top1_top2_margin",
    "maia3_top3_mass",
    "maia3_top5_mass",
    "maia3_top10_mass",
    "allie_p_top1",
    "allie_entropy",
    "allie_top1_top2_margin",
    "allie_top3_mass",
    "allie_top5_mass",
    "allie_top10_mass",
    "models_agree_top1",
    "p_top1_diff_maia3_minus_allie",
    "entropy_diff_maia3_minus_allie",
    "margin_diff_maia3_minus_allie",
    "maia3_top1_rank_in_allie",
    "allie_top1_rank_in_maia3",
    "top1s_in_each_other_top5",
    "top1s_in_each_other_top10",
    "candidate_model_code",
    "candidate_source_file",
    "candidate_source_rank",
    "candidate_dest_file",
    "candidate_dest_rank",
    "candidate_file_delta",
    "candidate_rank_delta",
    "candidate_is_promotion",
    "candidate_promotion_code",
    "candidate_is_castle_like",
    "candidate_rank_source",
    "candidate_prob_source",
    "candidate_log_prob_source",
    "candidate_rank_other",
    "candidate_prob_other",
    "candidate_log_prob_other",
    "candidate_prob_diff_source_minus_other",
    "candidate_in_other_top5",
    "candidate_in_other_top10",
]


def time_bucket_code(value: Any) -> int:
    try:
        x = float(value)
    except Exception:  # noqa: BLE001
        return -1
    if not math.isfinite(x) or x < 0:
        return -1
    for idx, edge in enumerate(TIME_EDGES):
        if x <= edge:
            return idx
    return len(TIME_EDGES)


def phase_code(value: Any) -> int:
    text = str(value).lower()
    if "opening" in text:
        return 0
    if "middle" in text:
        return 1
    if "end" in text:
        return 2
    return -1


def model_code(name: str) -> int:
    return 0 if name == "maia3" else 1


def source_model_name(code: int) -> str:
    return "maia3" if int(code) == 0 else "allie"


PROMOTION_CODES = {"": 0, "q": 1, "r": 2, "b": 3, "n": 4}


def move_features(move: str) -> dict[str, float]:
    move = str(move)
    files = "abcdefgh"
    if len(move) < 4 or move[0] not in files or move[2] not in files:
        return {
            "candidate_source_file": -1,
            "candidate_source_rank": -1,
            "candidate_dest_file": -1,
            "candidate_dest_rank": -1,
            "candidate_file_delta": 0,
            "candidate_rank_delta": 0,
            "candidate_is_promotion": 0,
            "candidate_promotion_code": 0,
            "candidate_is_castle_like": 0,
        }
    src_file = files.index(move[0])
    dst_file = files.index(move[2])
    try:
        src_rank = int(move[1])
        dst_rank = int(move[3])
    except ValueError:
        src_rank = -1
        dst_rank = -1
    promo = move[4].lower() if len(move) >= 5 else ""
    castle_like = move in {"e1g1", "e1c1", "e8g8", "e8c8"}
    return {
        "candidate_source_file": src_file,
        "candidate_source_rank": src_rank,
        "candidate_dest_file": dst_file,
        "candidate_dest_rank": dst_rank,
        "candidate_file_delta": dst_file - src_file,
        "candidate_rank_delta": dst_rank - src_rank,
        "candidate_is_promotion": int(bool(promo)),
        "candidate_promotion_code": PROMOTION_CODES.get(promo, 0),
        "candidate_is_castle_like": int(castle_like),
    }


def stable_top_indices(probs: np.ndarray, k: int) -> np.ndarray:
    if len(probs) == 0:
        return np.zeros(0, dtype=np.int32)
    order = np.argsort(-np.asarray(probs, dtype=np.float64), kind="mergesort")
    return order[: min(k, len(order))].astype(np.int32)


def strict_rank(probs: np.ndarray, index: int) -> int:
    values = np.asarray(probs, dtype=np.float64)
    p = float(values[index])
    return int((values > p).sum() + 1)


def prob_at_move(
    candidate_move: str,
    source_moves: list[str],
    other_moves: list[str],
    other_probs: np.ndarray,
    source_index: int | None = None,
) -> tuple[int, float, int]:
    if source_index is not None and len(source_moves) == len(other_moves) and source_moves == other_moves:
        other_index = int(source_index)
    else:
        try:
            other_index = other_moves.index(candidate_move)
        except ValueError:
            return -1, 0.0, 999
    prob = float(other_probs[other_index])
    rank = strict_rank(other_probs, other_index)
    return other_index, prob, rank


@dataclass
class DistributionBundle:
    frame: pd.DataFrame
    top_indices: np.ndarray
    top_probs: np.ndarray
    top3_mass: np.ndarray
    top5_mass: np.ndarray
    top10_mass: np.ndarray
    top1_top2_margin: np.ndarray


def summarize_distribution(frame: pd.DataFrame, max_k: int = 10) -> DistributionBundle:
    n = len(frame)
    top_indices = np.full((n, max_k), -1, dtype=np.int32)
    top_probs = np.zeros((n, max_k), dtype=np.float32)
    top3_mass = np.zeros(n, dtype=np.float32)
    top5_mass = np.zeros(n, dtype=np.float32)
    top10_mass = np.zeros(n, dtype=np.float32)
    margin = np.zeros(n, dtype=np.float32)
    legal_probs = frame["legal_probs"].tolist()
    for i, probs_obj in enumerate(legal_probs):
        probs = np.asarray(probs_obj, dtype=np.float64)
        order = stable_top_indices(probs, max_k)
        if len(order):
            top_indices[i, : len(order)] = order
            top_probs[i, : len(order)] = probs[order].astype(np.float32)
        top3_mass[i] = float(top_probs[i, : min(3, len(order))].sum())
        top5_mass[i] = float(top_probs[i, : min(5, len(order))].sum())
        top10_mass[i] = float(top_probs[i, : min(10, len(order))].sum())
        if len(order) > 1:
            margin[i] = float(probs[order[0]] - probs[order[1]])
        elif len(order) == 1:
            margin[i] = float(probs[order[0]])
    return DistributionBundle(frame, top_indices, top_probs, top3_mass, top5_mass, top10_mass, margin)


def common_row_features(
    i: int,
    maia: DistributionBundle,
    allie: DistributionBundle,
    maia_top1_rank_in_allie: int,
    allie_top1_rank_in_maia: int,
) -> dict[str, float]:
    row = maia.frame.iloc[i]
    player_elo = float(row["player_elo"])
    opponent_elo = float(row["opponent_elo"])
    maia_p_top1 = float(row["p_top1"])
    allie_p_top1 = float(allie.frame.iloc[i]["p_top1"])
    maia_entropy = float(row["entropy"])
    allie_entropy = float(allie.frame.iloc[i]["entropy"])
    return {
        "player_elo": player_elo,
        "opponent_elo": opponent_elo,
        "elo_diff": player_elo - opponent_elo,
        "move_number": float(row["move_number"]),
        "phase_code": phase_code(row["phase"]),
        "num_legal_moves": float(row["num_legal_moves"]),
        "maia3_p_top1": maia_p_top1,
        "maia3_entropy": maia_entropy,
        "maia3_top1_top2_margin": float(maia.top1_top2_margin[i]),
        "maia3_top3_mass": float(maia.top3_mass[i]),
        "maia3_top5_mass": float(maia.top5_mass[i]),
        "maia3_top10_mass": float(maia.top10_mass[i]),
        "allie_p_top1": allie_p_top1,
        "allie_entropy": allie_entropy,
        "allie_top1_top2_margin": float(allie.top1_top2_margin[i]),
        "allie_top3_mass": float(allie.top3_mass[i]),
        "allie_top5_mass": float(allie.top5_mass[i]),
        "allie_top10_mass": float(allie.top10_mass[i]),
        "models_agree_top1": int(row["top1_move_uci"] == allie.frame.iloc[i]["top1_move_uci"]),
        "p_top1_diff_maia3_minus_allie": maia_p_top1 - allie_p_top1,
        "entropy_diff_maia3_minus_allie": maia_entropy - allie_entropy,
        "margin_diff_maia3_minus_allie": float(maia.top1_top2_margin[i] - allie.top1_top2_margin[i]),
        "maia3_top1_rank_in_allie": maia_top1_rank_in_allie,
        "allie_top1_rank_in_maia3": allie_top1_rank_in_maia,
        "top1s_in_each_other_top5": int(maia_top1_rank_in_allie <= 5 and allie_top1_rank_in_maia <= 5),
        "top1s_in_each_other_top10": int(maia_top1_rank_in_allie <= 10 and allie_top1_rank_in_maia <= 10),
    }


def candidate_record(
    *,
    row_id: int,
    split: str,
    task: str,
    source_model: str,
    topk: int,
    candidate_slot: int,
    candidate_move: str,
    candidate_index_source: int,
    source_probs: np.ndarray,
    source_moves: list[str],
    other_probs: np.ndarray,
    other_moves: list[str],
    candidate_correct: bool,
    base_correct: bool,
    oracle_correct: bool,
    outside_topk: bool,
    common: dict[str, float],
) -> dict[str, Any]:
    source_rank = strict_rank(source_probs, candidate_index_source)
    source_prob = float(source_probs[candidate_index_source])
    _, other_prob, other_rank = prob_at_move(candidate_move, source_moves, other_moves, other_probs, candidate_index_source)
    rec: dict[str, Any] = {
        "row_id": int(row_id),
        "candidate_slot": int(candidate_slot),
        "candidate_move_uci": candidate_move,
        "candidate_correct": bool(candidate_correct),
        "base_correct": bool(base_correct),
        "oracle_correct": bool(oracle_correct),
        "outside_topk": bool(outside_topk),
        "task": task,
        "source_model": source_model,
        "topk": int(topk),
        "split": split,
        "candidate_model_code": model_code(source_model),
        "candidate_rank_source": int(source_rank),
        "candidate_prob_source": source_prob,
        "candidate_log_prob_source": float(math.log(max(source_prob, 1e-45))),
        "candidate_rank_other": int(other_rank),
        "candidate_prob_other": float(other_prob),
        "candidate_log_prob_other": float(math.log(max(other_prob, 1e-45))),
        "candidate_prob_diff_source_minus_other": float(source_prob - other_prob),
        "candidate_in_other_top5": int(other_rank <= 5),
        "candidate_in_other_top10": int(other_rank <= 10),
    }
    rec.update(common)
    rec.update(move_features(candidate_move))
    return rec

POST_DECISION_FEATURES = {"time_spent_seconds", "log_time_spent", "time_bucket_code"}
PREMOVE_FEATURES = [f for f in DIAGNOSTIC_FEATURES if f not in POST_DECISION_FEATURES]
INPUT_FEATURES = PREMOVE_FEATURES
TASK_SPECS = {"cross_model": {"kind": "cross", "base": "maia3", "topk": 1}, "maia3_self_top10": {"kind": "self", "base": "maia3", "topk": 10}, "allie_self_top5": {"kind": "self", "base": "allie", "topk": 5}}
BATCH_SOURCE_ROWS = 2000

class _Rows:
    """Column-backed row access avoids constructing millions of pandas Series."""
    def __init__(self, frame):
        self.columns = {c: frame[c].to_numpy(copy=False) for c in frame.columns}
        self.iloc = self

    def __getitem__(self, i):
        return _Row(self.columns, i)


class _Row:
    def __init__(self, columns, i):
        self.columns, self.i = columns, i

    def __getitem__(self, name):
        return self.columns[name][self.i]


def _rank_in_other_for_top1(i: int, source: DistributionBundle, other: DistributionBundle) -> int:
    source_idx = int(source.top_indices[i, 0])
    source_moves = list(source.frame.iloc[i]["legal_moves_uci"])
    other_moves = list(other.frame.iloc[i]["legal_moves_uci"])
    move = source_moves[source_idx]
    other_probs = np.asarray(other.frame.iloc[i]["legal_probs"], dtype=np.float64)
    _, _, rank = prob_at_move(move, source_moves, other_moves, other_probs, source_idx)
    return rank


def _record_chunks_for_task(
    indices: np.ndarray,
    split: str,
    task: str,
    maia: DistributionBundle,
    allie: DistributionBundle,
    max_candidates: int | None = None,
) -> Iterable[list[dict[str, Any]]]:
    spec = TASK_SPECS[task]
    if max_candidates is not None and spec["kind"] == "self":
        spec = {**spec, "topk": min(int(spec["topk"]), max_candidates)}
    for start in range(0, len(indices), BATCH_SOURCE_ROWS):
        batch_indices = indices[start : start + BATCH_SOURCE_ROWS]
        records: list[dict[str, Any]] = []
        for i_raw in batch_indices:
            i = int(i_raw)
            row_id = int(maia.frame.iloc[i]["row_id"])
            human = str(maia.frame.iloc[i]["human_move_uci"])
            maia_moves = list(maia.frame.iloc[i]["legal_moves_uci"])
            allie_moves = list(allie.frame.iloc[i]["legal_moves_uci"])
            maia_probs = np.asarray(maia.frame.iloc[i]["legal_probs"], dtype=np.float64)
            allie_probs = np.asarray(allie.frame.iloc[i]["legal_probs"], dtype=np.float64)
            maia_top1_rank_in_allie = _rank_in_other_for_top1(i, maia, allie)
            allie_top1_rank_in_maia = _rank_in_other_for_top1(i, allie, maia)
            common = common_row_features(i, maia, allie, maia_top1_rank_in_allie, allie_top1_rank_in_maia)
            maia_base_correct = bool(maia.frame.iloc[i]["is_top1"])
            allie_base_correct = bool(allie.frame.iloc[i]["is_top1"])
            if spec["kind"] == "cross":
                candidates = [
                    ("maia3", int(maia.top_indices[i, 0]), maia_probs, maia_moves, allie_probs, allie_moves),
                    ("allie", int(allie.top_indices[i, 0]), allie_probs, allie_moves, maia_probs, maia_moves),
                ]
                oracle = maia_base_correct or allie_base_correct
                for slot, (source_model, candidate_idx, source_probs, source_moves, other_probs, other_moves) in enumerate(candidates):
                    candidate_move = str(source_moves[candidate_idx])
                    records.append(
                        candidate_record(
                            row_id=row_id,
                            split=split,
                            task=task,
                            source_model=source_model,
                            topk=1,
                            candidate_slot=slot,
                            candidate_move=candidate_move,
                            candidate_index_source=candidate_idx,
                            source_probs=source_probs,
                            source_moves=source_moves,
                            other_probs=other_probs,
                            other_moves=other_moves,
                            candidate_correct=candidate_move == human,
                            base_correct=maia_base_correct,
                            oracle_correct=oracle,
                            outside_topk=not oracle,
                            common=common,
                        )
                    )
            else:
                source_name = str(spec["base"])
                k = int(spec["topk"])
                source = maia if source_name == "maia3" else allie
                other = allie if source_name == "maia3" else maia
                source_probs = maia_probs if source_name == "maia3" else allie_probs
                other_probs = allie_probs if source_name == "maia3" else maia_probs
                source_moves = maia_moves if source_name == "maia3" else allie_moves
                other_moves = allie_moves if source_name == "maia3" else maia_moves
                base_correct = maia_base_correct if source_name == "maia3" else allie_base_correct
                top_indices = source.top_indices[i, :k]
                top_indices = top_indices[top_indices >= 0]
                top_moves = [str(source_moves[int(idx)]) for idx in top_indices]
                oracle = human in top_moves
                for slot, candidate_idx_raw in enumerate(top_indices):
                    candidate_idx = int(candidate_idx_raw)
                    candidate_move = str(source_moves[candidate_idx])
                    records.append(
                        candidate_record(
                            row_id=row_id,
                            split=split,
                            task=task,
                            source_model=source_name,
                            topk=k,
                            candidate_slot=slot,
                            candidate_move=candidate_move,
                            candidate_index_source=candidate_idx,
                            source_probs=source_probs,
                            source_moves=source_moves,
                            other_probs=other_probs,
                            other_moves=other_moves,
                            candidate_correct=candidate_move == human,
                            base_correct=base_correct,
                            oracle_correct=oracle,
                            outside_topk=not oracle,
                            common=common,
                        )
                    )
        yield records


def deterministic_split(n_rows: int, seed: int = 20260605, *, smoke: bool = False):
    """Sorted position indices, preserving the paper's candidate training order."""
    if not smoke and n_rows != 500_000:
        raise ValueError(f"Paper protocol requires 500,000 held-out rows, got {n_rows}; use --smoke only for a smoke test")
    if n_rows < 2:
        raise ValueError("Need at least two held-out positions")
    n_train = 400_000 if not smoke else max(1, int(0.8 * n_rows))
    perm = np.random.default_rng(seed).permutation(n_rows)
    return np.sort(perm[:n_train]), np.sort(perm[n_train:])


def validate_alignment(maia: pd.DataFrame, allie: pd.DataFrame) -> None:
    for frame in (maia, allie):
        if not frame.row_id.is_unique:
            raise ValueError("row_id must be unique")
    if len(maia) != len(allie) or not np.array_equal(maia.row_id, allie.row_id):
        raise ValueError("MAIA3 and Allie rows must align one-to-one by row_id")
    if not np.array_equal(maia.human_move_uci, allie.human_move_uci):
        raise ValueError("Human-move labels differ across model distributions")
    for m, a in zip(maia.legal_moves_uci, allie.legal_moves_uci, strict=True):
        if len(m) != len(set(m)) or set(m) != set(a):
            raise ValueError("Both policies must cover the same complete legal-move set")


def candidate_frame(maia: pd.DataFrame, allie: pd.DataFrame, task: str,
                    indices=None, *, diagnostic_time: bool = False, max_candidates=None) -> pd.DataFrame:
    """Build candidate rows; labels are separate from PREMOVE_FEATURES."""
    validate_alignment(maia, allie)
    if indices is None:
        indices = np.arange(len(maia))
    mb, ab = summarize_distribution(maia), summarize_distribution(allie)
    mb.frame, ab.frame = _Rows(maia), _Rows(allie)
    records = [r for chunk in _record_chunks_for_task(indices, "provided", task, mb, ab, max_candidates) for r in chunk]
    out = pd.DataFrame.from_records(records)
    if diagnostic_time:
        times = maia.set_index("row_id")["time_spent_seconds"]
        out["time_spent_seconds"] = out.row_id.map(times).astype(float)
        out["log_time_spent"] = np.log1p(out.time_spent_seconds.clip(lower=0))
        out["time_bucket_code"] = out.time_spent_seconds.map(time_bucket_code)
    return out


def write_candidate_parquet(path: Path, maia: pd.DataFrame, allie: pd.DataFrame,
                            task: str, indices, *, diagnostic_time=False) -> None:
    """Bound candidate-memory usage while retaining the original row-group order."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    validate_alignment(maia, allie)
    mb, ab = summarize_distribution(maia), summarize_distribution(allie)
    mb.frame, ab.frame = _Rows(maia), _Rows(allie)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    times = maia.set_index("row_id")["time_spent_seconds"] if diagnostic_time else None
    try:
        for records in _record_chunks_for_task(indices, "provided", task, mb, ab):
            if diagnostic_time:
                for rec in records:
                    t = float(times.loc[rec["row_id"]])
                    rec.update(time_spent_seconds=t, log_time_spent=math.log1p(max(t, 0)), time_bucket_code=time_bucket_code(t))
            table = pa.Table.from_pylist(records)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
