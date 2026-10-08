"""Class-weighted linear selectors; preprocessing is fitted on training rows."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

import json
import pyarrow.parquet as pq
from .features import PREMOVE_FEATURES, DIAGNOSTIC_FEATURES, POST_DECISION_FEATURES
TRAIN_BATCH_ROWS = 200_000
LOGREG_GRID = [{"model_id": "logistic_l2_1e-4", "l2": 1e-4, "lr": 0.05, "epochs": 3}, {"model_id": "logistic_l2_1e-3", "l2": 1e-3, "lr": 0.05, "epochs": 3}]

def iter_batches(path: Path, columns: list[str], batch_size: int = TRAIN_BATCH_ROWS):
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        yield batch.to_pandas()


def compute_standardization(path: Path, features: list[str]) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    sums = np.zeros(len(features), dtype=np.float64)
    sqs = np.zeros(len(features), dtype=np.float64)
    n = 0
    positives = 0
    for frame in iter_batches(path, features + ["candidate_correct"]):
        x = frame[features].to_numpy(dtype=np.float64, copy=False)
        y = frame["candidate_correct"].to_numpy(dtype=bool, copy=False)
        sums += x.sum(axis=0)
        sqs += np.square(x).sum(axis=0)
        n += len(frame)
        positives += int(y.sum())
    mean = sums / max(n, 1)
    var = np.maximum(sqs / max(n, 1) - mean**2, 1e-8)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32), n, positives, n - positives


def train_logistic(path: Path, task: str, variant: str, features: list[str], params: dict[str, Any]) -> dict[str, Any]:
    mean, std, n, positives, negatives = compute_standardization(path, features)
    weights = np.zeros(len(features), dtype=np.float32)
    bias = np.float32(math.log((positives + 1.0) / (negatives + 1.0)))
    pos_weight = n / max(2.0 * positives, 1.0)
    neg_weight = n / max(2.0 * negatives, 1.0)
    lr = float(params["lr"])
    l2 = float(params["l2"])
    epochs = int(params["epochs"])
    for epoch in range(epochs):
        loss_sum = 0.0
        seen = 0
        for frame in iter_batches(path, features + ["candidate_correct"]):
            x = frame[features].to_numpy(dtype=np.float32, copy=False)
            y = frame["candidate_correct"].to_numpy(dtype=np.float32, copy=False)
            z = (x - mean) / std
            logits = z @ weights + bias
            pred = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
            sample_weight = np.where(y > 0.5, pos_weight, neg_weight).astype(np.float32)
            err = (pred - y) * sample_weight
            denom = float(len(y))
            grad_w = (z.T @ err) / denom + l2 * weights
            grad_b = float(err.mean())
            weights -= lr * grad_w.astype(np.float32)
            bias = np.float32(bias - lr * grad_b)
            clipped = np.clip(pred, 1e-6, 1 - 1e-6)
            loss = -(sample_weight * (y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped))).mean()
            loss_sum += float(loss) * len(y)
            seen += len(y)
        print(f"{variant} {task} {params['model_id']} epoch {epoch + 1}/{epochs} loss={loss_sum / max(seen, 1):.6f}", flush=True)
    return {
        "weights": weights,
        "bias": float(bias),
        "mean": mean,
        "std": std,
        "feature_names": features,
        "metadata": {
            "task": task,
            "variant": variant,
            "model_id": params["model_id"],
            "model_family": "logistic_regression_numpy",
            "l2": l2,
            "learning_rate": lr,
            "epochs": epochs,
            "train_rows_candidate_level": n,
            "train_positive_candidates": positives,
            "train_negative_candidates": negatives,
            "trained_on": "selector-training candidates only",
            "paper_test_used_for_training": False,
            "paper_test_used_for_model_selection": False,
            "excluded_features": sorted(POST_DECISION_FEATURES - set(features)),
            "diagnostic_only": bool(POST_DECISION_FEATURES & set(features)),
        },
    }


def selector_scores(frame: pd.DataFrame, model: dict[str, Any]) -> np.ndarray:
    features = list(model["feature_names"])
    x = frame[features].to_numpy(dtype=np.float32, copy=False)
    z = (x - np.asarray(model["mean"], dtype=np.float32)) / np.asarray(model["std"], dtype=np.float32)
    logits = z @ np.asarray(model["weights"], dtype=np.float32) + float(model["bias"])
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def select_top_candidate(frame: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    keep = [
        "row_id",
        "candidate_slot",
        "candidate_move_uci",
        "candidate_correct",
        "base_correct",
        "oracle_correct",
        "outside_topk",
    ]
    work = frame[keep].copy()
    work["selector_score"] = scores
    work = work.sort_values(["row_id", "selector_score", "candidate_slot"], ascending=[True, False, True])
    return work.drop_duplicates("row_id", keep="first").sort_values("row_id").reset_index(drop=True)

def save_npz_model(path: Path, model: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        weights=np.asarray(model["weights"], dtype=np.float32),
        bias=np.asarray([float(model["bias"])], dtype=np.float32),
        mean=np.asarray(model["mean"], dtype=np.float32),
        std=np.asarray(model["std"], dtype=np.float32),
        feature_names=np.asarray(model["feature_names"], dtype=object),
        metadata=np.asarray([json.dumps(model["metadata"], sort_keys=True)], dtype=object),
    )


def load_npz_model(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {
            "weights": data["weights"].astype(np.float32),
            "bias": float(data["bias"][0]),
            "mean": data["mean"].astype(np.float32),
            "std": data["std"].astype(np.float32),
            "feature_names": [str(x) for x in data["feature_names"].tolist()],
            "metadata": json.loads(str(data["metadata"][0])),
        }


def selection_metrics(selected: pd.DataFrame) -> dict[str, Any]:
    correct = selected.candidate_correct.to_numpy(bool)
    base = selected.base_correct.to_numpy(bool)
    rescues = int((correct & ~base).sum())
    breaks = int((~correct & base).sum())
    return {"rows": len(correct), "Top1": float(correct.mean()),
            "rescues": rescues, "breaks": breaks, "net": rescues - breaks}


def fit_selector(train_path: Path, validation: pd.DataFrame, task: str,
                 *, diagnostic_time=False):
    features = DIAGNOSTIC_FEATURES if diagnostic_time else PREMOVE_FEATURES
    variant = "diagnostic_realized_duration" if diagnostic_time else "premove"
    rows, models = [], []
    for params in LOGREG_GRID:
        model = train_logistic(train_path, task, variant, features, params)
        selected = select_top_candidate(validation, selector_scores(validation, model))
        rows.append({"task": task, "variant": variant, **params, **selection_metrics(selected)})
        models.append(model)
    best = max(range(len(rows)), key=lambda i: (rows[i]["Top1"], rows[i]["net"], -rows[i]["breaks"]))
    rows[best]["selected"] = True
    return models[best], rows


def hard_predictions(selected: pd.DataFrame) -> pd.DataFrame:
    return selected[["row_id", "candidate_move_uci", "candidate_correct", "base_correct"]].rename(
        columns={"candidate_move_uci": "top1_move_uci", "candidate_correct": "correct_top1"})


def save_model(path: Path, model: dict[str, Any]) -> None:
    """Store numerical heads without executable pickle objects."""
    payload = {k: v for k, v in model.items() if isinstance(v, np.ndarray)}
    if "bias" in model:
        payload["bias"] = np.atleast_1d(model["bias"]).astype(np.float32)
    payload["feature_names"] = np.asarray(model["feature_names"], dtype=str)
    payload["metadata"] = np.asarray([json.dumps(model["metadata"], sort_keys=True)])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
