"""Held-out temperature calibration and probability ensembles."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import time_bucket_code, TIME_LABELS

def softmax_temperature(logits_obj: Any, probs_obj: Any, temperature: float) -> np.ndarray:
    temp = max(float(temperature), 1e-6)
    if logits_obj is not None:
        try:
            logits = np.asarray(logits_obj, dtype=np.float64)
            if np.isfinite(logits).all() and len(logits):
                z = logits / temp
                z = z - float(z.max())
                exp = np.exp(z)
                return exp / float(exp.sum())
        except Exception:  # noqa: BLE001
            pass
    probs = np.clip(np.asarray(probs_obj, dtype=np.float64), 1e-45, 1.0)
    z = np.power(probs, 1.0 / temp)
    return z / float(z.sum())


def calibration_temperature_for_row(params: dict[str, Any], row: pd.Series) -> float:
    selected = params["selected"]
    if selected == "time_bucket":
        bucket = str(time_bucket_code(row["time_spent_seconds"]))
        return float(params["time_bucket"]["temperatures"].get(bucket, params["global"]["temperature"]))
    return float(params["global"]["temperature"])

TEMPERATURE_GRID = [0.75, 0.85, 0.95, 1.0, 1.05, 1.1, 1.2, 1.35]


def temperature_cache(frame: pd.DataFrame) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    scores: list[np.ndarray] = []
    human_indices = np.zeros(len(frame), dtype=np.int32)
    codes = np.zeros(len(frame), dtype=np.int32)
    for i, row in enumerate(frame.itertuples(index=False)):
        probs = np.asarray(getattr(row, "legal_probs"), dtype=np.float64)
        logits = np.asarray(getattr(row, "legal_logits"), dtype=np.float64)
        if len(logits) == len(probs) and np.isfinite(logits).all():
            base_scores = logits
        else:
            base_scores = np.log(np.clip(probs, 1e-45, 1.0))
        scores.append(base_scores.astype(np.float64))
        moves = list(getattr(row, "legal_moves_uci"))
        human_indices[i] = moves.index(str(getattr(row, "human_move_uci")))
        codes[i] = int(time_bucket_code(getattr(row, "time_spent_seconds")))
    return scores, human_indices, codes


def nll_values_for_temperature(scores: list[np.ndarray], human_indices: np.ndarray, temperature: float) -> np.ndarray:
    temp = max(float(temperature), 1e-6)
    out = np.empty(len(scores), dtype=np.float64)
    for i, values in enumerate(scores):
        z = values / temp
        m = float(z.max())
        logsum = m + math.log(float(np.exp(z - m).sum()))
        out[i] = logsum - float(z[int(human_indices[i])])
    return out


def fit_calibration_for_base(base: str, frame: pd.DataFrame) -> dict[str, Any]:
    scores, human_indices, codes = temperature_cache(frame)
    grid_rows = []
    bucket_rows: dict[str, list[dict[str, float]]] = {str(code): [] for code in range(len(TIME_LABELS))}
    for temp in TEMPERATURE_GRID:
        values = nll_values_for_temperature(scores, human_indices, temp)
        grid_rows.append({"temperature": float(temp), "heldout_nll": float(values.mean())})
        for code in range(len(TIME_LABELS)):
            mask = codes == code
            bucket_rows[str(code)].append(
                {
                    "temperature": float(temp),
                    "heldout_nll": float(values[mask].mean()) if mask.any() else float(values.mean()),
                }
            )
    global_fit = min(grid_rows, key=lambda row: (row["heldout_nll"], row["temperature"]))
    bucket_temps: dict[str, float] = {}
    bucket_nlls: dict[str, float] = {}
    bucket_counts: dict[str, int] = {}
    weighted = 0.0
    for code in range(len(TIME_LABELS)):
        idx = np.flatnonzero(codes == code).astype(np.int64)
        fit = min(bucket_rows[str(code)], key=lambda row: (row["heldout_nll"], row["temperature"])) if len(idx) else global_fit
        bucket_temps[str(code)] = float(fit["temperature"])
        bucket_nlls[str(code)] = float(fit["heldout_nll"])
        bucket_counts[str(code)] = int(len(idx))
        weighted += float(fit["heldout_nll"]) * int(len(idx))
    weighted_nll = weighted / max(len(frame), 1)
    selected = "time_bucket" if weighted_nll < float(global_fit["heldout_nll"]) - 1e-9 else "global"
    return {
        "base": base,
        "temperature_grid": TEMPERATURE_GRID,
        "global": {**global_fit, "grid": grid_rows},
        "time_bucket": {
            "temperatures": bucket_temps,
            "heldout_nll_by_bucket": bucket_nlls,
            "rows_by_bucket": bucket_counts,
            "heldout_nll": float(weighted_nll),
        },
        "selected": selected,
        "selected_heldout_nll": float(weighted_nll if selected == "time_bucket" else global_fit["heldout_nll"]),
        "fit_scope": "5M heldout only",
    }


def fit_fine_calibration(frame: pd.DataFrame):
    """Standalone calibration: 31 log-spaced temperatures, then 31 linear points."""
    scores, hidx, codes = temperature_cache(frame)
    # Padding permits batched numerical operations while preserving legal support.
    width = max(map(len, scores))
    logits = np.full((len(scores), width), -np.inf)
    for i, values in enumerate(scores):
        logits[i, :len(values)] = values
    def fit(indices):
        values = logits[indices]
        hi = hidx[indices]
        def nll(temp):
            total = 0.0
            for start in range(0, len(values), 8192):
                z = values[start:start + 8192] / temp
                z = z - z.max(axis=1, keepdims=True)
                p = np.exp(z)
                p /= p.sum(axis=1, keepdims=True)
                ph = p[np.arange(len(z)), hi[start:start + 8192]]
                total += -np.log(np.clip(ph, 1e-45, 1.0)).sum()
            return float(total / len(indices))
        grid = np.exp(np.linspace(math.log(0.35), math.log(3.5), 31))
        vals = [nll(t) for t in grid]
        best = float(grid[int(np.argmin(vals))])
        refine = np.linspace(max(0.1, best / 1.5), min(8.0, best * 1.5), 31)
        fine = [nll(t) for t in refine]
        i = int(np.argmin(fine))
        return {"temperature": float(refine[i]), "heldout_nll": fine[i],
                "coarse_temperature": best, "coarse_nll": min(vals)}
    global_fit = fit(np.arange(len(frame)))
    buckets = {}
    for code in range(5):
        indices = np.flatnonzero(codes == code)
        buckets[str(code)] = {**(fit(indices) if len(indices) >= 100 else global_fit),
                              "row_count": int(len(indices)), "fallback": len(indices) < 100}
    return {"global": global_fit, "time_bucket": buckets,
            "protocol": "standalone_fine_grid", "fit_rows": len(frame)}


def distribution_frame(frame: pd.DataFrame, probabilities) -> pd.DataFrame:
    """Recompute all label-dependent summaries after changing probabilities."""
    out = frame.copy()
    records = []
    probs_list = []
    for row, probs in zip(frame.itertuples(index=False), probabilities, strict=True):
        p = np.asarray(probs, np.float64)
        if not np.isfinite(p).all() or np.any(p < 0) or p.sum() <= 0:
            raise ValueError("Invalid legal distribution")
        p = p / p.sum()
        moves = list(row.legal_moves_uci)
        h = moves.index(row.human_move_uci)
        top = int(np.argmax(p))
        rank = int(np.count_nonzero(p > p[h]) + 1)
        rec = {"p_human": p[h], "p_top1": p[top], "top1_move_uci": moves[top],
               "human_rank": rank, "entropy": float(-(p * np.log(np.clip(p, 1e-45, 1))).sum()),
               "nll": -math.log(max(p[h], 1e-45)), "brier": float(np.square(p).sum() - 2*p[h] + 1),
               "num_legal_moves": len(p), "correct_top1": top == h, "is_top1": rank == 1}
        rec.update({f"is_top{k}": rank <= k for k in (3, 5, 10, 20)})
        records.append(rec)
        probs_list.append(p)
    for name in records[0]:
        out[name] = [r[name] for r in records]
    out["legal_probs"] = probs_list
    out["legal_logits"] = [np.log(np.clip(p, 1e-45, 1)) for p in probs_list]
    return out


def apply_calibration(frame, params, variant="global", *, coarse=False):
    probabilities = []
    for _, row in frame.iterrows():
        if coarse:
            temp = calibration_temperature_for_row(params, row)
        elif variant == "global":
            temp = params["global"]["temperature"]
        else:
            temp = params["time_bucket"][str(time_bucket_code(row.time_spent_seconds))]["temperature"]
        probabilities.append(softmax_temperature(row.get("legal_logits"), row.legal_probs, temp))
    return distribution_frame(frame, probabilities)


def mix_probs(pm, pa, kind, alpha):
    if kind == "convex":
        p = alpha * pm + (1 - alpha) * pa
    elif kind == "geometric":
        z = alpha * np.log(np.maximum(pm, 1e-12)) + (1 - alpha) * np.log(np.maximum(pa, 1e-12))
        p = np.exp(z - z.max())
    else:
        raise ValueError(f"Unknown mixture {kind}")
    return p / p.sum()


def aligned_probability_pairs(maia, allie):
    from .features import validate_alignment
    validate_alignment(maia, allie)
    for m, a in zip(maia.itertuples(index=False), allie.itertuples(index=False), strict=True):
        moves = list(m.legal_moves_uci)
        lookup = dict(zip(a.legal_moves_uci, a.legal_probs, strict=True))
        pm = np.asarray(m.legal_probs, np.float64)
        pa = np.asarray([lookup[move] for move in moves], np.float64)
        yield pm / pm.sum(), pa / pa.sum(), moves.index(m.human_move_uci)


def fit_ensembles(maia, allie):
    rows = ensemble_sweep(maia, allie)
    selected = {}
    for kind in ("convex", "geometric"):
        selected[kind] = min([r for r in rows if r["kind"] == kind], key=lambda r: (r["NLL"], r["alpha"]))
    return selected, rows


def ensemble_sweep(maia, allie):
    """Fixed complete alpha grid; use held-out NLL alone for selection."""
    alphas = np.round(np.arange(0, 1.0001, 0.05), 2)
    accumulators = {kind: {"nll_sum": np.zeros(len(alphas)), "correct": np.zeros(len(alphas), np.int64),
                          "rescues": np.zeros(len(alphas), np.int64), "breaks": np.zeros(len(alphas), np.int64)}
                    for kind in ("convex", "geometric")}
    base_flags = maia.is_top1.to_numpy(bool)
    for i, (pm, pa, h) in enumerate(aligned_probability_pairs(maia, allie)):
        base_correct = bool(base_flags[i])
        for kind, acc in accumulators.items():
            if kind == "convex":
                mix = alphas[:, None] * pm + (1 - alphas[:, None]) * pa
            else:
                z = alphas[:, None] * np.log(np.maximum(pm, 1e-12)) + (1 - alphas[:, None]) * np.log(np.maximum(pa, 1e-12))
                mix = np.exp(z - z.max(axis=1, keepdims=True))
                mix /= mix.sum(axis=1, keepdims=True)
            ph = np.maximum(mix[:, h], 1e-12)
            correct = np.count_nonzero(mix > ph[:, None] + 1e-15, axis=1) == 0
            acc["nll_sum"] -= np.log(ph)
            acc["correct"] += correct
            acc["rescues"] += (not base_correct) & correct
            acc["breaks"] += base_correct & (~correct)
    rows = []
    for kind, acc in accumulators.items():
        for i, alpha in enumerate(alphas):
            row = {"kind": kind, "alpha": float(alpha), "rows": len(maia),
                   **{key: values[i].item() for key, values in acc.items()}}
            row.update(NLL=row["nll_sum"] / len(maia), Top1=row["correct"] / len(maia))
            rows.append(row)
    return rows


def apply_ensemble(maia, allie, kind, alpha):
    return distribution_frame(maia, (mix_probs(pm, pa, kind, alpha)
                                     for pm, pa, _ in aligned_probability_pairs(maia, allie)))
