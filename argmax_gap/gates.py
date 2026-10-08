"""Conservative rank-2 gate with train-only fitting and validation-only thresholds."""
from __future__ import annotations

import math
from typing import Any
import numpy as np
import pandas as pd
from .features import PREMOVE_FEATURES

SEED = 20260701


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x.astype(np.float32, copy=False) - mean.astype(np.float32)) / std.astype(np.float32)


def score_linear_model(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    family = model["metadata"]["family"]
    z = standardize_apply(x, model["mean"], model["std"])
    if family == "ridge_delta_numpy":
        return z @ model["weights"] + float(model["bias"][0])
    if family == "logistic_delta_ovr_numpy":
        return sigmoid(z @ model["weights_pos"] + float(model["bias_pos"][0])) - sigmoid(
            z @ model["weights_neg"] + float(model["bias_neg"][0])
        )
    raise ValueError(f"Unknown model family {family}")


def train_ridge(x: np.ndarray, y: np.ndarray, features: list[str], model_id: str, l2: float) -> dict[str, Any]:
    mean, std = standardize_fit(x)
    z = standardize_apply(x, mean, std).astype(np.float64)
    design = np.hstack([z, np.ones((len(z), 1), dtype=np.float64)])
    reg = np.eye(design.shape[1], dtype=np.float64) * float(l2)
    reg[-1, -1] = 0.0
    beta = np.linalg.solve(design.T @ design + reg, design.T @ y.astype(np.float64))
    return {
        "weights": beta[:-1].astype(np.float32),
        "bias": np.asarray([beta[-1]], dtype=np.float32),
        "mean": mean,
        "std": std,
        "feature_names": features,
        "metadata": {
            "family": "ridge_delta_numpy",
            "model_id": model_id,
            "l2": float(l2),
            "trained_on": "selector_train only",
            "paper_test_used_for_training": False,
            "paper_test_used_for_selection": False,
        },
    }


def _train_binary_logistic(z: np.ndarray, y: np.ndarray, *, l2: float, lr: float, epochs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, d = z.shape
    weights = np.zeros(d, dtype=np.float32)
    pos = float(y.sum())
    neg = float(n - pos)
    bias = np.asarray([math.log((pos + 1.0) / (neg + 1.0))], dtype=np.float32)
    pos_w = n / max(2.0 * pos, 1.0)
    neg_w = n / max(2.0 * neg, 1.0)
    batch = 32_768
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch):
            idx = order[start : start + batch]
            xb = z[idx]
            yb = y[idx].astype(np.float32)
            pred = sigmoid(xb @ weights + float(bias[0]))
            sample_weight = np.where(yb > 0.5, pos_w, neg_w).astype(np.float32)
            err = (pred - yb) * sample_weight
            weights -= float(lr) * (((xb.T @ err) / len(idx)) + float(l2) * weights).astype(np.float32)
            bias -= float(lr) * np.asarray([err.mean()], dtype=np.float32)
    return weights.astype(np.float32), bias.astype(np.float32)


def train_logistic_ovr(
    x: np.ndarray,
    delta: np.ndarray,
    features: list[str],
    model_id: str,
    *,
    l2: float = 1e-4,
    lr: float = 0.05,
    epochs: int = 3,
) -> dict[str, Any]:
    mean, std = standardize_fit(x)
    z = standardize_apply(x, mean, std)
    w_pos, b_pos = _train_binary_logistic(z, (delta > 0).astype(np.float32), l2=l2, lr=lr, epochs=epochs, seed=20260630 + 1)
    w_neg, b_neg = _train_binary_logistic(z, (delta < 0).astype(np.float32), l2=l2, lr=lr, epochs=epochs, seed=20260630 + 2)
    return {
        "weights_pos": w_pos,
        "bias_pos": b_pos,
        "weights_neg": w_neg,
        "bias_neg": b_neg,
        "mean": mean,
        "std": std,
        "feature_names": features,
        "metadata": {
            "family": "logistic_delta_ovr_numpy",
            "model_id": model_id,
            "l2": float(l2),
            "learning_rate": float(lr),
            "epochs": int(epochs),
            "trained_on": "selector_train only",
            "paper_test_used_for_training": False,
            "paper_test_used_for_selection": False,
        },
    }


def exact_best_threshold(scores: np.ndarray, delta: np.ndarray, base_correct: np.ndarray) -> dict[str, Any]:
    scores = np.asarray(scores, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.int8)
    base_correct = np.asarray(base_correct, dtype=bool)
    n = len(scores)
    if n == 0:
        raise ValueError("Threshold selection requires validation positions")
    base_accuracy = float(base_correct.mean())
    order = np.argsort(-scores, kind="mergesort")
    ss = scores[order]
    dd = delta[order]
    ends = np.r_[np.flatnonzero(ss[1:] != ss[:-1]) + 1, n]
    rows = [
        {
            "threshold": float(np.nextafter(np.nanmax(scores), np.inf)),
            "Top1": base_accuracy,
            "delta_vs_MAIA3": 0.0,
            "rescues": 0,
            "breaks": 0,
            "net": 0,
            "switch_rate": 0.0,
        }
    ]
    rescues = breaks = switched = 0
    start = 0
    for end_raw in ends:
        end = int(end_raw)
        block = dd[start:end]
        rescues += int((block > 0).sum())
        breaks += int((block < 0).sum())
        switched += int(end - start)
        net = rescues - breaks
        rows.append(
            {
                "threshold": float(np.nextafter(ss[end - 1], -np.inf)),
                "Top1": float(base_accuracy + net / n),
                "delta_vs_MAIA3": float(net / n),
                "rescues": rescues,
                "breaks": breaks,
                "net": net,
                "switch_rate": float(switched / n),
            }
        )
        start = end
    return sorted(rows, key=lambda r: (r["Top1"], r["net"], -r["breaks"], -r["switch_rate"]), reverse=True)[0]


def metrics_from_switch(delta: np.ndarray, base_correct: np.ndarray, switch: np.ndarray) -> dict[str, Any]:
    delta = np.asarray(delta, dtype=np.int8)
    base_correct = np.asarray(base_correct, dtype=bool)
    switch = np.asarray(switch, dtype=bool)
    rescues = int(((delta > 0) & switch).sum())
    breaks = int(((delta < 0) & switch).sum())
    net = rescues - breaks
    n = len(base_correct)
    return {
        "rows": n,
        "Top1": float(base_correct.mean() + net / n),
        "delta_vs_MAIA3": float(net / n),
        "rescues": rescues,
        "breaks": breaks,
        "net": net,
        "switch_rate": float(switch.mean()),
    }


def threshold_metrics(scores: np.ndarray, delta: np.ndarray, base: np.ndarray, theta: float) -> dict[str, Any]:
    out = metrics_from_switch(delta, base, np.asarray(scores) > float(theta))
    out["threshold"] = float(theta)
    return out


def select_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(rows, key=lambda r: (r["Top1"], r["net"], -r["breaks"], -r["switch_rate"]), reverse=True)[0]


TOP1_GEOMETRY_FEATURES = [
    "top1_source_file",
    "top1_source_rank",
    "top1_dest_file",
    "top1_dest_rank",
    "top1_file_delta",
    "top1_rank_delta",
    "top1_is_promotion",
    "top1_promotion_code",
    "top1_is_castle_like",
]


RANK_EXTRA_FEATURES = [
    "candidate_maia_rank",
    "candidate_maia_position",
    "maia3_rank_candidate_prob",
    "maia3_top1_minus_candidate_prob_margin",
    "maia3_top1_minus_candidate_logprob",
    "allie_rank_candidate_rank",
    "allie_rank_candidate_prob",
    "allie_rank_candidate_logprob",
    "candidate_is_allie_top1",
    "maia3_top20_mass",
    "allie_top20_mass",
] + TOP1_GEOMETRY_FEATURES


RANK_FEATURES = PREMOVE_FEATURES + RANK_EXTRA_FEATURES


def add_rank_features(frame: pd.DataFrame, maia20: dict[int, float], allie20: dict[int, float]) -> pd.DataFrame:
    frame = frame.copy()
    base = frame.loc[frame["candidate_slot"].eq(0)].drop_duplicates("row_id").set_index("row_id")
    top1_cols = {
        "candidate_source_file": "top1_source_file",
        "candidate_source_rank": "top1_source_rank",
        "candidate_dest_file": "top1_dest_file",
        "candidate_dest_rank": "top1_dest_rank",
        "candidate_file_delta": "top1_file_delta",
        "candidate_rank_delta": "top1_rank_delta",
        "candidate_is_promotion": "top1_is_promotion",
        "candidate_promotion_code": "top1_promotion_code",
        "candidate_is_castle_like": "top1_is_castle_like",
        "candidate_log_prob_source": "top1_log_prob_source",
    }
    frame = frame.join(base[list(top1_cols)].rename(columns=top1_cols), on="row_id")
    frame["candidate_maia_rank"] = frame["candidate_rank_source"].astype(np.float32)
    frame["candidate_maia_position"] = (frame["candidate_slot"].astype(np.float32) + 1.0)
    frame["maia3_rank_candidate_prob"] = frame["candidate_prob_source"].astype(np.float32)
    frame["maia3_top1_minus_candidate_prob_margin"] = (
        frame["maia3_p_top1"].astype(np.float32) - frame["candidate_prob_source"].astype(np.float32)
    )
    frame["maia3_top1_minus_candidate_logprob"] = (
        frame["top1_log_prob_source"].astype(np.float32) - frame["candidate_log_prob_source"].astype(np.float32)
    )
    frame["allie_rank_candidate_rank"] = frame["candidate_rank_other"].astype(np.float32)
    frame["allie_rank_candidate_prob"] = frame["candidate_prob_other"].astype(np.float32)
    frame["allie_rank_candidate_logprob"] = frame["candidate_log_prob_other"].astype(np.float32)
    frame["candidate_is_allie_top1"] = frame["candidate_rank_other"].eq(1).astype(np.float32)
    frame["maia3_top20_mass"] = frame["row_id"].map(maia20).astype(np.float32)
    frame["allie_top20_mass"] = frame["row_id"].map(allie20).astype(np.float32)
    return frame


def train_models(x: np.ndarray, delta: np.ndarray, prefix: str, features: list[str]) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for l2 in [0.01, 0.1, 1.0]:
        model = train_ridge(x, delta, features, f"{prefix}_ridge_l2_{l2:g}", l2)
        model["metadata"]["seed"] = SEED
        models.append(model)
    for l2 in [1e-4, 1e-3]:
        model = train_logistic_ovr(x, delta, features, f"{prefix}_logistic_l2_{l2:g}", l2=l2, lr=0.05, epochs=3)
        model["metadata"]["seed"] = SEED
        models.append(model)
    return models


def base_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["candidate_maia_position"].eq(1), ["row_id", "base_correct"]].sort_values("row_id").reset_index(drop=True)


def single_full_arrays(base: pd.DataFrame, alt: pd.DataFrame, scores: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_ids = base["row_id"].to_numpy(np.int64)
    row_to_pos = {int(row_id): pos for pos, row_id in enumerate(row_ids)}
    base_correct = base["base_correct"].to_numpy(bool)
    full_scores = np.full(len(base), -1e9, dtype=np.float64)
    full_delta = np.zeros(len(base), dtype=np.int8)
    if len(alt):
        pos = np.asarray([row_to_pos[int(row_id)] for row_id in alt["row_id"].to_numpy()], dtype=np.int64)
        full_delta[pos] = alt["candidate_correct"].to_numpy(np.int8) - alt["base_correct"].to_numpy(np.int8)
        if scores is not None:
            full_scores[pos] = np.asarray(scores, dtype=np.float64)
    return full_scores, full_delta, base_correct


def rank_frame(candidates, maia, allie):
    def masses(frame):
        result = {}
        for row in frame.itertuples(index=False):
            p = np.asarray(row.legal_probs, np.float64)
            k = min(20, len(p))
            result[int(row.row_id)] = float(np.partition(p, -k)[-k:].sum())
        return result
    return add_rank_features(candidates, masses(maia), masses(allie))


def fit_single_rank(train, validation, rank=2):
    alt = train[train.candidate_maia_position.eq(rank)]
    if alt.empty:
        raise ValueError(f"No training candidates at rank {rank}")
    x = alt[RANK_FEATURES].to_numpy(np.float32)
    delta = alt.candidate_correct.to_numpy(np.int8) - alt.base_correct.to_numpy(np.int8)
    va = validation[validation.candidate_maia_position.eq(rank)]
    models = train_models(x, delta, f"single_rank{rank}", RANK_FEATURES)
    rows = []
    for model in models:
        scores = score_linear_model(model, va[RANK_FEATURES].to_numpy(np.float32))
        scores, delta, base = single_full_arrays(base_rows(validation), va, scores)
        rows.append({"model_id": model["metadata"]["model_id"], "rank": rank,
                     **exact_best_threshold(scores, delta, base)})
    best = select_from_rows(rows)
    model = next(m for m in models if m["metadata"]["model_id"] == best["model_id"])
    return {"model": model, "threshold": best["threshold"], "rank": rank, "kind": "single"}, rows


def predict_single_rank(frame, fitted):
    base = frame[frame.candidate_maia_position.eq(1)].sort_values("row_id").set_index("row_id")
    alt = frame[frame.candidate_maia_position.eq(fitted["rank"])].copy()
    scores = score_linear_model(fitted["model"], alt[RANK_FEATURES].to_numpy(np.float32))
    alt["switch"] = scores.astype(np.float64) > fitted["threshold"]
    switch_ids = alt.loc[alt.switch, "row_id"]
    out = base[["candidate_move_uci", "candidate_correct", "base_correct"]].copy()
    # The paper retains the cached strict-rank base correctness on untouched rows.
    out["candidate_correct"] = out["base_correct"].to_numpy(bool)
    out["switch"] = False
    replacements = alt.set_index("row_id")
    out.loc[switch_ids, ["candidate_move_uci", "candidate_correct"]] = replacements.loc[
        switch_ids, ["candidate_move_uci", "candidate_correct"]]
    out.loc[switch_ids, "switch"] = True
    return out.reset_index().rename(columns={"candidate_move_uci": "top1_move_uci", "candidate_correct": "correct_top1"})
