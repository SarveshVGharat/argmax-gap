"""Calibrated trust-region residual scorer and validation search."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import time_bucket_code, phase_code, move_features, stable_top_indices
from .calibration import softmax_temperature, calibration_temperature_for_row

FORBIDDEN_INPUT_FEATURES = {
    "human_move_uci",
    "correct_top1",
    "human_rank",
    "p_human",
    "top1_human_margin",
    "candidate_correct",
    "base_correct",
    "oracle_correct",
    "human_candidate_slot",
    "human_base_prob",
    "human_cal_prob",
    "outside_topK",
}


PREMOVE_FEATURES = [
    "player_elo",
    "opponent_elo",
    "elo_diff",
    "move_number",
    "phase_code",
    "num_legal_moves",
    "base_p_top1",
    "base_entropy",
    "base_top1_top2_margin",
    "base_top3_mass",
    "base_top5_mass",
    "base_top10_mass",
    "base_top20_mass",
    "candidate_rank",
    "candidate_is_base_top1",
    "candidate_prob_base",
    "candidate_prob_cal",
    "candidate_log_prob_base",
    "candidate_log_prob_cal",
    "candidate_prob_over_topk_mass",
    "candidate_source_file",
    "candidate_source_rank",
    "candidate_dest_file",
    "candidate_dest_rank",
    "candidate_file_delta",
    "candidate_rank_delta",
    "candidate_is_promotion",
    "candidate_promotion_code",
    "candidate_is_castle_like",
]


@dataclass
class DistributionBundle:
    frame: pd.DataFrame
    top_indices: np.ndarray
    top_probs: np.ndarray
    top3_mass: np.ndarray
    top5_mass: np.ndarray
    top10_mass: np.ndarray
    top20_mass: np.ndarray
    top1_top2_margin: np.ndarray


def summarize_distribution(frame: pd.DataFrame, max_k: int = 20) -> DistributionBundle:
    n = len(frame)
    top_indices = np.full((n, max_k), -1, dtype=np.int32)
    top_probs = np.zeros((n, max_k), dtype=np.float32)
    top3_mass = np.zeros(n, dtype=np.float32)
    top5_mass = np.zeros(n, dtype=np.float32)
    top10_mass = np.zeros(n, dtype=np.float32)
    top20_mass = np.zeros(n, dtype=np.float32)
    margin = np.zeros(n, dtype=np.float32)
    for i, probs_obj in enumerate(frame["legal_probs"]):
        probs = np.asarray(probs_obj, dtype=np.float64)
        order = stable_top_indices(probs, max_k)
        if len(order):
            top_indices[i, : len(order)] = order
            top_probs[i, : len(order)] = probs[order].astype(np.float32)
        top3_mass[i] = float(top_probs[i, : min(3, len(order))].sum())
        top5_mass[i] = float(top_probs[i, : min(5, len(order))].sum())
        top10_mass[i] = float(top_probs[i, : min(10, len(order))].sum())
        top20_mass[i] = float(top_probs[i, : min(20, len(order))].sum())
        if len(order) > 1:
            margin[i] = float(probs[order[0]] - probs[order[1]])
        elif len(order) == 1:
            margin[i] = float(probs[order[0]])
    return DistributionBundle(frame, top_indices, top_probs, top3_mass, top5_mass, top10_mass, top20_mass, margin)


def row_common_features(row: pd.Series, bundle: DistributionBundle, i: int) -> dict[str, float]:
    player_elo = float(row["player_elo"])
    opponent_elo = float(row["opponent_elo"])
    return {
        "player_elo": player_elo,
        "opponent_elo": opponent_elo,
        "elo_diff": player_elo - opponent_elo,
        "move_number": float(row["move_number"]),
        "phase_code": phase_code(row["phase"]),
        "num_legal_moves": float(row["num_legal_moves"]),
        "base_p_top1": float(row["p_top1"]),
        "base_entropy": float(row["entropy"]),
        "base_top1_top2_margin": float(bundle.top1_top2_margin[i]),
        "base_top3_mass": float(bundle.top3_mass[i]),
        "base_top5_mass": float(bundle.top5_mass[i]),
        "base_top10_mass": float(bundle.top10_mass[i]),
        "base_top20_mass": float(bundle.top20_mass[i]),
    }


def candidate_record(
    *,
    row_pos: int,
    row: pd.Series,
    bundle: DistributionBundle,
    base_model: str,
    split: str,
    topk: int,
    candidate_slot: int,
    candidate_index: int,
    cal_probs: np.ndarray,
    topk_cal_mass: float,
    outside_mass_cal: float,
    outside_max_log_prob_cal: float,
    outside_max_correct: bool,
    human_candidate_slot: int,
    human_cal_prob: float,
    oracle_correct: bool,
) -> dict[str, Any]:
    moves = list(row["legal_moves_uci"])
    probs = np.asarray(row["legal_probs"], dtype=np.float64)
    candidate_move = str(moves[candidate_index])
    candidate_prob_base = float(probs[candidate_index])
    candidate_prob_cal = float(cal_probs[candidate_index])
    rec: dict[str, Any] = {
        "row_id": int(row["row_id"]),
        "row_pos": int(row_pos),
        "split": split,
        "base_model": base_model,
        "topk": int(topk),
        "candidate_slot": int(candidate_slot),
        "candidate_index": int(candidate_index),
        "candidate_move_uci": candidate_move,
        "candidate_correct": bool(candidate_move == str(row["human_move_uci"])),
        "base_correct": bool(row["is_top1"]),
        "oracle_correct": bool(oracle_correct),
        "outside_topK": bool(not oracle_correct),
        "human_candidate_slot": int(human_candidate_slot),
        "human_base_prob": float(row["p_human"]),
        "human_cal_prob": float(human_cal_prob),
        "outside_mass_cal": float(outside_mass_cal),
        "outside_max_log_prob_cal": float(outside_max_log_prob_cal),
        "outside_max_correct": bool(outside_max_correct),
        "candidate_rank": int(candidate_slot + 1),
        "candidate_is_base_top1": int(candidate_slot == 0),
        "candidate_prob_base": candidate_prob_base,
        "candidate_prob_cal": candidate_prob_cal,
        "candidate_log_prob_base": float(math.log(max(candidate_prob_base, 1e-45))),
        "candidate_log_prob_cal": float(math.log(max(candidate_prob_cal, 1e-45))),
        "candidate_prob_over_topk_mass": float(candidate_prob_cal / max(topk_cal_mass, 1e-45)),
    }
    rec.update(row_common_features(row, bundle, row_pos))
    rec.update(move_features(candidate_move))
    return rec


def feature_matrix(frame: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    if frame.columns.duplicated().any():
        frame = frame.loc[:, ~frame.columns.duplicated()].copy()
    return frame[feature_names].to_numpy(dtype=np.float32, copy=False)


def residual_scores(frame: pd.DataFrame, model: dict[str, Any]) -> np.ndarray:
    names = list(model["feature_names"])
    x = feature_matrix(frame, names)
    z = (x - np.asarray(model["mean"], dtype=np.float32)) / np.asarray(model["std"], dtype=np.float32)
    return np.clip(z @ np.asarray(model["weights"], dtype=np.float32) + float(model["bias"]), -8.0, 8.0)

TOPK_VARIANTS = [5, 10, 20]


ALPHA_GRID = [0.05, 0.1, 0.2, 0.5]


LOSS_PROFILES = [
    {
        "profile": "conservative",
        "lambda_rescue": 0.05,
        "lambda_break": 1.0,
        "lambda_kl": 0.1,
        "lambda_l2": 1e-4,
        "margin": 0.05,
        "epochs": 2,
        "learning_rate": 0.08,
    },
    {
        "profile": "balanced",
        "lambda_rescue": 0.1,
        "lambda_break": 0.5,
        "lambda_kl": 0.05,
        "lambda_l2": 1e-4,
        "margin": 0.05,
        "epochs": 2,
        "learning_rate": 0.08,
    },
    {
        "profile": "rescue_weighted",
        "lambda_rescue": 0.2,
        "lambda_break": 0.3,
        "lambda_kl": 0.01,
        "lambda_l2": 1e-4,
        "margin": 0.0,
        "epochs": 2,
        "learning_rate": 0.08,
    },
]


SGD_BATCH_CANDIDATES = 262_144


def train_weighted_logistic(
    frame: pd.DataFrame,
    feature_names: list[str],
    profile: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    if frame.columns.duplicated().any():
        frame = frame.loc[:, ~frame.columns.duplicated()].copy()
    x = frame[feature_names].to_numpy(dtype=np.float32, copy=False)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    z = (x - mean) / std
    y = frame["candidate_correct"].to_numpy(dtype=np.float32)
    base_correct = frame["base_correct"].to_numpy(dtype=bool)
    is_top1 = frame["candidate_is_base_top1"].to_numpy(dtype=bool)
    outside = frame["outside_topK"].to_numpy(dtype=bool)
    weights = np.ones(len(frame), dtype=np.float32)
    pos = y > 0.5
    if pos.any():
        weights[pos] *= min(10.0, max(1.0, float((~pos).sum()) / max(float(pos.sum()), 1.0)))
    weights[outside] *= 0.25
    weights[(~base_correct) & pos] *= 1.0 + 5.0 * float(profile["lambda_rescue"])
    weights[base_correct & is_top1] *= 1.0 + 2.0 * float(profile["lambda_break"])
    weights[base_correct & (~is_top1)] *= 1.0 + float(profile["lambda_break"])
    rng = np.random.default_rng(seed)
    w = np.zeros(z.shape[1], dtype=np.float32)
    b = 0.0
    lr = float(profile["learning_rate"])
    l2 = float(profile["lambda_l2"])
    order = np.arange(len(frame), dtype=np.int64)
    for _epoch in range(int(profile["epochs"])):
        rng.shuffle(order)
        for start in range(0, len(order), SGD_BATCH_CANDIDATES):
            idx = order[start : start + SGD_BATCH_CANDIDATES]
            xb = z[idx]
            yb = y[idx]
            wb = weights[idx]
            pred = 1.0 / (1.0 + np.exp(-np.clip(xb @ w + b, -40.0, 40.0)))
            err = (pred - yb) * wb
            denom = max(float(wb.sum()), 1.0)
            grad_w = xb.T @ err / denom + l2 * w
            grad_b = float(err.sum() / denom)
            w -= lr * grad_w.astype(np.float32)
            b -= lr * grad_b
    return {
        "weights": w,
        "bias": float(b),
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "feature_names": feature_names,
        "metadata": {
            "model_family": "linear_residual_scorer_numpy",
            "training_loss": "weighted candidate-level CE surrogate with rescue/no-break weights and L2",
            "profile": profile,
            "paper_test_used": False,
            "forbidden_input_features_used": sorted(set(feature_names).intersection(FORBIDDEN_INPUT_FEATURES)),
        },
    }


def evaluate_candidate_refiner(frame: pd.DataFrame, scores: np.ndarray, alpha: float) -> dict[str, Any]:
    row_id = frame["row_id"].to_numpy()
    starts = np.r_[0, np.flatnonzero(row_id[1:] != row_id[:-1]) + 1]
    ends = np.r_[starts[1:], len(frame)]
    log_p = frame["candidate_log_prob_cal"].to_numpy(dtype=np.float64)
    p_cal = frame["candidate_prob_cal"].to_numpy(dtype=np.float64)
    cand_correct = frame["candidate_correct"].to_numpy(dtype=bool)
    base_correct_arr = frame["base_correct"].to_numpy(dtype=bool)
    human_slot_arr = frame["human_candidate_slot"].to_numpy(dtype=np.int32)
    human_cal_arr = frame["human_cal_prob"].to_numpy(dtype=np.float64)
    human_base_arr = frame["human_base_prob"].to_numpy(dtype=np.float64)
    outside_mass_arr = frame["outside_mass_cal"].to_numpy(dtype=np.float64)
    outside_log_arr = frame["outside_max_log_prob_cal"].to_numpy(dtype=np.float64)
    outside_correct_arr = frame["outside_max_correct"].to_numpy(dtype=bool)
    oracle_arr = frame["oracle_correct"].to_numpy(dtype=bool)
    correct = np.zeros(len(starts), dtype=bool)
    nll = np.zeros(len(starts), dtype=np.float64)
    kl = np.zeros(len(starts), dtype=np.float64)
    base_correct = np.zeros(len(starts), dtype=bool)
    cal_nll = np.zeros(len(starts), dtype=np.float64)
    base_nll = np.zeros(len(starts), dtype=np.float64)
    oracle = np.zeros(len(starts), dtype=bool)
    for r, (start, end) in enumerate(zip(starts, ends, strict=True)):
        logits = log_p[start:end] + alpha * scores[start:end]
        best_rel = int(np.argmax(logits))
        best_abs = start + best_rel
        outside_wins = bool(outside_log_arr[start] > float(logits[best_rel]))
        correct[r] = bool(outside_correct_arr[start] if outside_wins else cand_correct[best_abs])
        exp_factor = np.exp(np.clip(alpha * scores[start:end], -30.0, 30.0))
        denom = float(outside_mass_arr[start] + np.dot(p_cal[start:end], exp_factor))
        human_slot = int(human_slot_arr[start])
        if human_slot >= 0:
            numerator = float(p_cal[start + human_slot] * exp_factor[human_slot])
        else:
            numerator = float(human_cal_arr[start])
        nll[r] = -math.log(max(numerator / max(denom, 1e-45), 1e-45))
        kl[r] = math.log(max(denom, 1e-45)) - alpha * float(np.dot(p_cal[start:end], scores[start:end]))
        base_correct[r] = bool(base_correct_arr[start])
        cal_nll[r] = -math.log(max(float(human_cal_arr[start]), 1e-45))
        base_nll[r] = -math.log(max(float(human_base_arr[start]), 1e-45))
        oracle[r] = bool(oracle_arr[start])
    rescues = correct & ~base_correct
    breaks = ~correct & base_correct
    return {
        "rows": int(len(starts)),
        "Top1": float(correct.mean()),
        "Top1_percent": float(correct.mean() * 100.0),
        "base_Top1": float(base_correct.mean()),
        "base_Top1_percent": float(base_correct.mean() * 100.0),
        "delta_Top1": float(correct.mean() - base_correct.mean()),
        "delta_Top1_pp": float((correct.mean() - base_correct.mean()) * 100.0),
        "NLL": float(nll.mean()),
        "calibrated_base_NLL": float(cal_nll.mean()),
        "base_NLL": float(base_nll.mean()),
        "delta_NLL_vs_base": float(nll.mean() - base_nll.mean()),
        "delta_NLL_vs_calibrated": float(nll.mean() - cal_nll.mean()),
        "mean_kl_pcal_to_refiner": float(kl.mean()),
        "rescues": int(rescues.sum()),
        "breaks": int(breaks.sum()),
        "net": int(rescues.sum() - breaks.sum()),
        "rescue_rate": float(rescues.mean()),
        "break_rate": float(breaks.mean()),
        "oracle_candidate_set_Top1": float(oracle.mean()),
        "coverage": float(oracle.mean()),
    }


def candidate_chunks(frame, params, base, indices, topk):
    """Generate refiner candidates in original sorted position order."""
    bundle = summarize_distribution(frame, max_k=20)
    for start in range(0, len(indices), 2000):
        records = []
        for row_pos in indices[start:start + 2000]:
            row_pos = int(row_pos)
            row = frame.iloc[row_pos]
            moves = list(row.legal_moves_uci)
            human_idx = moves.index(row.human_move_uci)
            temp = calibration_temperature_for_row(params, row)
            cal = softmax_temperature(row.get("legal_logits"), row.legal_probs, temp)
            order = bundle.top_indices[row_pos]
            order = order[order >= 0]
            candidates = order[:topk]
            slots = np.flatnonzero(candidates == human_idx)
            human_slot = int(slots[0]) if len(slots) else -1
            mass = float(cal[candidates].sum())
            # Preserve the published Top20-cached validation rule. Final inference
            # below normalizes over every legal move.
            outside = [int(i) for i in order if i not in set(candidates)]
            outside_log = math.log(max(cal[outside[0]], 1e-45)) if outside else -1e30
            for slot, idx in enumerate(candidates):
                records.append(candidate_record(row_pos=row_pos, row=row, bundle=bundle,
                    base_model=base, split="provided", topk=topk, candidate_slot=slot,
                    candidate_index=int(idx), cal_probs=cal, topk_cal_mass=mass,
                    outside_mass_cal=max(1 - mass, 0), outside_max_log_prob_cal=outside_log,
                    outside_max_correct=bool(outside and outside[0] == human_idx),
                    human_candidate_slot=human_slot, human_cal_prob=cal[human_idx], oracle_correct=human_slot >= 0))
        yield records


def write_candidates(path, frame, params, base, indices, topk):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        for records in candidate_chunks(frame, params, base, indices, topk):
            table = pa.Table.from_pylist(records)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def fit_refiner(frame, params, base, train_idx, val_idx, cache_dir, *, seed_offset=0):
    from .selectors import save_model
    rows, best, best_key = [], None, None
    for topk in TOPK_VARIANTS:
        trpath, vapath = (cache_dir / f"{base}_refiner_{s}_top{topk}.parquet" for s in ("train", "val"))
        write_candidates(trpath, frame, params, base, train_idx, topk)
        write_candidates(vapath, frame, params, base, val_idx, topk)
        train_cols = list(dict.fromkeys(PREMOVE_FEATURES + ["candidate_correct", "base_correct", "candidate_is_base_top1", "outside_topK"]))
        train = pd.read_parquet(trpath, columns=train_cols)
        val = pd.read_parquet(vapath)
        for profile in LOSS_PROFILES:
            model = train_weighted_logistic(train, PREMOVE_FEATURES, profile,
                                            seed=13_000 + topk + seed_offset + len(rows))
            scores = residual_scores(val, model)
            for alpha in ALPHA_GRID:
                result = evaluate_candidate_refiner(val, scores, alpha)
                record = {"base": base, "topk": topk, "alpha": alpha, "profile": profile["profile"], **result}
                rows.append(record)
                key = (record["Top1"], -record["delta_NLL_vs_base"], -record["break_rate"])
                if best_key is None or key > best_key:
                    best_key = key
                    best = {"model": model, "topk": topk, "alpha": alpha, "calibration": params,
                            "base": base, "validation": record}
        del train, val
    return best, rows


def apply_refiner(frame, fitted):
    from .calibration import distribution_frame
    indices = np.arange(len(frame))
    candidates = pd.DataFrame.from_records([r for chunk in candidate_chunks(
        frame, fitted["calibration"], fitted["base"], indices, fitted["topk"]) for r in chunk])
    scores = residual_scores(candidates, fitted["model"])
    candidates["residual"] = scores
    groups = {int(i): g for i, g in candidates.groupby("row_pos", sort=False)}
    probabilities = []
    for i, (_, row) in enumerate(frame.iterrows()):
        temp = calibration_temperature_for_row(fitted["calibration"], row)
        p = softmax_temperature(row.get("legal_logits"), row.legal_probs, temp)
        z = np.log(np.maximum(p, 1e-45))
        group = groups[i]
        z[group.candidate_index.to_numpy(int)] += fitted["alpha"] * group.residual.to_numpy()
        p = np.exp(z - z.max())
        probabilities.append(p / p.sum())
    return distribution_frame(frame, probabilities)
