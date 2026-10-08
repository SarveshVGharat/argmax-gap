"""Exact deterministic small-MLP and XGBoost selector baselines."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .features import PREMOVE_FEATURES
SEED = 20260711
MLP_TRAIN_ROW_LIMIT = 80_000
MLP_VAL_LOSS_ROW_LIMIT = 80_000
MLP_MAX_EPOCHS = 4
MLP_BATCH_SIZE = 8192
TRAIN_SAMPLE_LIMIT = 80_000
MAX_POSITIVE_SAMPLE = 40_000

MLP_CONFIGS: list[dict[str, Any]] = [
    {"hidden": (32,), "dropout": 0.0, "weight_decay": 0.0},
    {"hidden": (32,), "dropout": 0.1, "weight_decay": 1e-4},
    {"hidden": (32,), "dropout": 0.1, "weight_decay": 1e-3},
    {"hidden": (64,), "dropout": 0.0, "weight_decay": 0.0},
    {"hidden": (64,), "dropout": 0.1, "weight_decay": 1e-4},
    {"hidden": (64,), "dropout": 0.1, "weight_decay": 1e-3},
    {"hidden": (128,), "dropout": 0.0, "weight_decay": 0.0},
    {"hidden": (128,), "dropout": 0.1, "weight_decay": 1e-4},
    {"hidden": (128,), "dropout": 0.1, "weight_decay": 1e-3},
    {"hidden": (64, 32), "dropout": 0.0, "weight_decay": 0.0},
    {"hidden": (64, 32), "dropout": 0.1, "weight_decay": 1e-4},
    {"hidden": (64, 32), "dropout": 0.1, "weight_decay": 1e-3},
    {"hidden": (128, 64), "dropout": 0.0, "weight_decay": 0.0},
    {"hidden": (128, 64), "dropout": 0.1, "weight_decay": 1e-4},
    {"hidden": (128, 64), "dropout": 0.1, "weight_decay": 1e-3},
]


def deterministic_subset(n: int, limit: int, seed_offset: int) -> np.ndarray:
    if n <= limit:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(SEED + seed_offset)
    return np.sort(rng.choice(np.arange(n, dtype=np.int64), size=limit, replace=False))


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x.astype(np.float32, copy=False) - mean.astype(np.float32)) / std.astype(np.float32)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = input_dim
        for width in hidden:
            layers.append(nn.Linear(dim, int(width)))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            dim = int(width)
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def mlp_id(prefix: str, config: dict[str, Any]) -> str:
    hidden = "x".join(str(x) for x in config["hidden"])
    drop = str(config["dropout"]).replace(".", "p")
    wd = f"{config['weight_decay']:.0e}".replace("-", "m")
    return f"{prefix}_mlp_h{hidden}_d{drop}_wd{wd}"


def train_mlp(
    *,
    x_full: np.ndarray,
    y_full: np.ndarray,
    x_val_loss: np.ndarray,
    y_val_loss: np.ndarray,
    config: dict[str, Any],
    model_id: str,
    seed_offset: int,
) -> dict[str, Any]:
    torch.manual_seed(SEED + seed_offset)
    np.random.seed(SEED + seed_offset)
    train_idx = deterministic_subset(len(x_full), MLP_TRAIN_ROW_LIMIT, 10_000 + seed_offset)
    x_train = x_full[train_idx].astype(np.float32, copy=False)
    y_train = y_full[train_idx].astype(np.float32, copy=False)
    mean, std = standardize_fit(x_train)
    z_train = standardize_apply(x_train, mean, std)
    val_idx = deterministic_subset(len(x_val_loss), MLP_VAL_LOSS_ROW_LIMIT, 20_000 + seed_offset)
    z_val = standardize_apply(x_val_loss[val_idx].astype(np.float32, copy=False), mean, std)
    y_val = y_val_loss[val_idx].astype(np.float32, copy=False)
    model = MLP(z_train.shape[1], tuple(config["hidden"]), float(config["dropout"]))
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=float(config["weight_decay"]))
    dataset = TensorDataset(torch.from_numpy(z_train), torch.from_numpy(y_train))
    generator = torch.Generator().manual_seed(SEED + seed_offset)
    loader = DataLoader(dataset, batch_size=MLP_BATCH_SIZE, shuffle=True, generator=generator)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_run = 0
    for epoch in range(MLP_MAX_EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(torch.from_numpy(z_val))
            val_loss = float(loss_fn(val_logits, torch.from_numpy(y_val)).item())
        epochs_run = epoch + 1
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "kind": "mlp",
        "model_id": model_id,
        "config": dict(config),
        "model": model.eval(),
        "mean": mean,
        "std": std,
        "epochs_run": epochs_run,
        "best_val_loss": best_loss,
        "train_rows_used": int(len(train_idx)),
        "features": list(PREMOVE_FEATURES),
    }


def mlp_scores(model_obj: dict[str, Any], frame_or_x: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(frame_or_x, pd.DataFrame):
        x = frame_or_x[model_obj["features"]].to_numpy(np.float32, copy=False)
    else:
        x = frame_or_x.astype(np.float32, copy=False)
    z = standardize_apply(x, model_obj["mean"], model_obj["std"])
    scores: list[np.ndarray] = []
    model: nn.Module = model_obj["model"]
    model.eval()
    with torch.no_grad():
        for start in range(0, len(z), 131_072):
            batch = torch.from_numpy(z[start : start + 131_072])
            scores.append(torch.sigmoid(model(batch)).cpu().numpy().astype(np.float32))
    return np.concatenate(scores) if scores else np.asarray([], dtype=np.float32)


def candidate_selection(frame: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    work = frame[["row_id", "candidate_slot", "candidate_move_uci", "candidate_correct", "base_correct"]].copy()
    work["score"] = np.asarray(scores, dtype=np.float64)
    base = work[work["candidate_slot"].eq(0)][["row_id", "candidate_move_uci", "base_correct"]].rename(
        columns={"candidate_move_uci": "base_move", "base_correct": "reference_correct"}
    )
    work = work.sort_values(["row_id", "score", "candidate_slot"], ascending=[True, False, True])
    selected = work.drop_duplicates("row_id", keep="first").merge(base, on="row_id", how="left", validate="one_to_one")
    selected["changed"] = selected["candidate_move_uci"].to_numpy(str) != selected["base_move"].to_numpy(str)
    return selected.sort_values("row_id").reset_index(drop=True)


def selection_metrics(selected: pd.DataFrame, *, reference_name: str, maia_correct: np.ndarray | None = None) -> dict[str, Any]:
    correct = selected["candidate_correct"].to_numpy(bool)
    ref = selected["reference_correct"].to_numpy(bool)
    rescues = int(((correct == 1) & (ref == 0)).sum())
    breaks = int(((correct == 0) & (ref == 1)).sum())
    out = {
        "rows": int(len(selected)),
        "Top1": float(correct.mean()),
        "delta_vs_reference": float(correct.mean() - ref.mean()),
        "reference": reference_name,
        "rescues": rescues,
        "breaks": breaks,
        "net": int(rescues - breaks),
        "switch_change_rate": float(selected["changed"].mean()),
    }
    if maia_correct is not None:
        out["delta_vs_MAIA3"] = float(correct.mean() - np.asarray(maia_correct, dtype=bool).mean())
    return out


def select_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        rows,
        key=lambda r: (
            r["Top1"],
            r.get("net", 0),
            -r.get("breaks", 0),
            -r.get("switch_change_rate", r.get("switch_rate", 0.0)),
            1 if r.get("model_family") == "linear" else 0,
        ),
        reverse=True,
    )[0]

CHECKPOINTS = (100, 300)


LEARNING_RATES = (0.03, 0.1)


MAX_DEPTHS = (2, 3, 4)


SUBSAMPLES = (0.8, 1.0)


COLSAMPLES = (0.8, 1.0)


REG_LAMBDAS = (1, 5, 10)


MIN_CHILD_WEIGHTS = (1, 10)


def xgb_grid() -> Iterable[dict[str, Any]]:
    for lr in LEARNING_RATES:
        for depth in MAX_DEPTHS:
            for subsample in SUBSAMPLES:
                for colsample in COLSAMPLES:
                    for reg_lambda in REG_LAMBDAS:
                        for min_child_weight in MIN_CHILD_WEIGHTS:
                            yield {"learning_rate": lr, "max_depth": depth,
                                "subsample": subsample, "colsample_bytree": colsample,
                                "reg_lambda": reg_lambda, "min_child_weight": min_child_weight}


def deterministic_choice(indices: np.ndarray, size: int, seed_offset: int) -> np.ndarray:
    if len(indices) <= size:
        return np.asarray(indices, dtype=np.int64)
    rng = np.random.default_rng(SEED + seed_offset)
    return np.sort(rng.choice(indices, size=size, replace=False).astype(np.int64))


def sample_training_rows(y: np.ndarray, seed_offset: int) -> tuple[np.ndarray, dict[str, Any]]:
    y = np.asarray(y)
    n = len(y)
    positive = np.flatnonzero(y > 0)
    nonpositive = np.flatnonzero(y <= 0)
    if n <= TRAIN_SAMPLE_LIMIT:
        idx = np.arange(n, dtype=np.int64)
        return idx, {
            "sample_rule": "used_all_training_rows",
            "train_rows_total": int(n),
            "positive_rows_total": int(len(positive)),
            "train_rows_used": int(len(idx)),
            "positive_rows_used": int(len(positive)),
        }
    pos_keep = deterministic_choice(positive, min(MAX_POSITIVE_SAMPLE, len(positive)), seed_offset + 1)
    neg_keep = deterministic_choice(nonpositive, TRAIN_SAMPLE_LIMIT - len(pos_keep), seed_offset + 2)
    idx = np.sort(np.concatenate([pos_keep, neg_keep]).astype(np.int64))
    return idx, {
        "sample_rule": (
            "deterministic train-only sample: keep all positive rows if <= half the cap; "
            "otherwise sample positives up to half the cap, then fill with nonpositive rows"
        ),
        "train_rows_total": int(n),
        "positive_rows_total": int(len(positive)),
        "train_rows_used": int(len(idx)),
        "positive_rows_used": int(np.isin(idx, positive, assume_unique=False).sum()),
    }


def fit_mlp_selector(train, validation, task, *, linear_model=None, smoke=False):
    """Search the MLP grid against the pre-move linear baseline on validation.

    The original shortlist searches selected the existing linear head. Retain
    that outcome explicitly instead of labeling a linear prediction as an MLP.
    """
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    x = train[PREMOVE_FEATURES].to_numpy(np.float32)
    y = train.candidate_correct.to_numpy(np.float32)
    vx = validation[PREMOVE_FEATURES].to_numpy(np.float32)
    vy = validation.candidate_correct.to_numpy(np.float32)
    best, best_row, rows = None, None, []
    if linear_model is not None:
        from .selectors import selector_scores
        result = selection_metrics(candidate_selection(validation, selector_scores(validation, linear_model)),
                                   reference_name=task)
        best_row = {"task": task, "model_id": "existing_premove_linear_logistic",
                    "model_family": "linear", **result}
        rows.append(best_row)
        best = {"kind": "linear", "model": linear_model, "model_id": best_row["model_id"]}
    for idx, config in enumerate(MLP_CONFIGS[:1] if smoke else MLP_CONFIGS):
        obj = train_mlp(x_full=x, y_full=y, x_val_loss=vx, y_val_loss=vy,
                        config=config, model_id=mlp_id(task, config),
                        seed_offset=1000 + 37 * idx + len(task))
        result = selection_metrics(candidate_selection(validation, mlp_scores(obj, validation)), reference_name=task)
        row = {"task": task, "model_id": obj["model_id"], "model_family": "mlp", **config,
               "epochs_run": obj["epochs_run"], "best_val_loss": obj["best_val_loss"],
               "train_rows_used": obj["train_rows_used"], **result}
        rows.append(row)
        if best_row is None or select_best([best_row, row]) is row:
            best, best_row = obj, row
    for row in rows:
        row["selected_by_validation"] = row is best_row
    best["validation"] = best_row
    return best, rows


def fit_xgboost_selector(train, validation, task, *, smoke=False):
    import xgboost as xgb
    if xgb.__version__ != "3.2.0":
        raise RuntimeError(f"Paper reproduction requires xgboost==3.2.0; installed {xgb.__version__}")
    x = train[PREMOVE_FEATURES].to_numpy(np.float32)
    y = train.candidate_correct.to_numpy(np.float32)
    vx = validation[PREMOVE_FEATURES].to_numpy(np.float32)
    best, best_key, rows = None, None, []
    grid = list(xgb_grid())
    for i, config in enumerate(grid[:1] if smoke else grid):
        idx, sample_meta = sample_training_rows(y, 1000 + 37 * i + len(task))
        estimator = xgb.XGBClassifier(n_estimators=300, objective="binary:logistic",
            tree_method="hist", random_state=SEED, n_jobs=8, verbosity=0, **config)
        estimator.fit(x[idx], y[idx])
        for n_trees in CHECKPOINTS:
            # The published implementation uses hard class predictions, including ties.
            scores = estimator.predict(vx, iteration_range=(0, n_trees)).astype(np.float32)
            result = selection_metrics(candidate_selection(validation, scores), reference_name=task)
            row = {"task": task, "model_family": "xgboost", "n_estimators": n_trees, **config, **sample_meta, **result}
            rows.append(row)
            key = (row["Top1"], row["net"], -row["breaks"], -row["switch_change_rate"], -n_trees)
            if best_key is None or key > best_key:
                best_key = key
                best = {"model": estimator, "n_estimators": n_trees, "config": config,
                        "features": PREMOVE_FEATURES, "metadata": row}
    return best, rows


def xgboost_scores(fitted, frame):
    return fitted["model"].predict(frame[PREMOVE_FEATURES].to_numpy(np.float32),
                                  iteration_range=(0, fitted["n_estimators"])).astype(np.float32)


def save_mlp(path, fitted):
    torch.save({"state_dict": fitted["model"].state_dict(), "config": fitted["config"],
                "mean": torch.from_numpy(fitted["mean"]), "std": torch.from_numpy(fitted["std"]),
                "features": fitted["features"], "model_id": fitted["model_id"]}, path)
