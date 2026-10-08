"""Tie-aware ranking, calibration, and paired uncertainty estimates."""

from __future__ import annotations

import math
import hashlib
import numpy as np


def strict_rank(probabilities, human_index):
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
        raise ValueError("Expected a nonempty, finite, nonnegative distribution")
    return 1 + int(np.count_nonzero(p > p[human_index]))


def ece(confidence, correct, bins=10):
    confidence, correct = np.asarray(confidence), np.asarray(correct, dtype=float)
    if len(confidence) != len(correct):
        raise ValueError("Confidence and correctness must have equal lengths")
    if not len(confidence):
        return math.nan
    if not np.isfinite(confidence).all() or ((confidence < 0) | (confidence > 1)).any():
        raise ValueError("Confidence must be finite and in [0, 1]")
    ids = np.minimum((confidence * bins).astype(int), bins - 1)
    return sum(float((ids == b).mean()) * abs(float(correct[ids == b].mean() - confidence[ids == b].mean()))
               for b in range(bins) if (ids == b).any())


def summarize(frame):
    rank = frame.human_rank.to_numpy()
    if not len(rank) or (rank < 1).any():
        raise ValueError("No valid ranked predictions")
    result = {"rows": len(rank), "NLL": float(frame.nll.mean()),
              **{f"Top{k}": float(100 * (rank <= k).mean()) for k in (1, 3, 5, 10, 20)},
              "MRR": float((1 / rank).mean()),
              **{f"NDCG@{k}": float(np.where(rank <= k, 1 / np.log2(rank + 1), 0).mean()) for k in (3, 5, 10)},
              "ECE": ece(frame.p_top1.to_numpy(), rank == 1)}
    result["gap_pp"] = result["Top5"] - result["Top1"]
    return result


def paired_top1(base, alternative, reps=1000, seed=20260605, rng=None):
    base, alternative = np.asarray(base, dtype=bool), np.asarray(alternative, dtype=bool)
    if base.shape != alternative.shape or not base.size:
        raise ValueError("Paired outcomes must have the same nonzero shape")
    n = base.size
    rescues = int((~base & alternative).sum())
    breaks = int((base & ~alternative).sum())
    rng = np.random.default_rng(seed) if rng is None else rng
    counts = rng.multinomial(n, np.array([rescues, breaks, n - rescues - breaks]) / n, size=reps)
    low, high = np.quantile(100 * (counts[:, 0] - counts[:, 1]) / n, [0.025, 0.975])
    # Continuity-corrected McNemar statistic used by the paper.
    p = math.erfc(math.sqrt((abs(rescues - breaks) - 1) ** 2 / (2 * (rescues + breaks)))) if rescues + breaks else 1.0
    return {"rows": n, "rescues": rescues, "breaks": breaks,
            "delta_pp": 100 * (rescues - breaks) / n, "ci_low_pp": float(low), "ci_high_pp": float(high),
            "mcnemar_p": p}


def paired_mean_ci(difference):
    difference = np.asarray(difference, dtype=float)
    if not len(difference) or not np.isfinite(difference).all():
        raise ValueError("Expected finite paired differences")
    mean = float(difference.mean())
    se = float(difference.std(ddof=1) / np.sqrt(len(difference))) if len(difference) > 1 else 0.0
    return {"delta": mean, "ci_low": mean - 1.959963984540054 * se,
            "ci_high": mean + 1.959963984540054 * se}


def paired_poisson_cis(differences, reps=500, seed=20260606, rng=None):
    """Paired Poisson intervals sharing position weights across the supplied metrics."""
    arrays = {name: np.asarray(values, dtype=np.float64) for name, values in differences.items()}
    if not arrays or len({len(values) for values in arrays.values()}) != 1:
        raise ValueError("Expected aligned paired differences")
    n = len(next(iter(arrays.values())))
    if not n or reps < 1 or any(values.ndim != 1 or not np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("Expected nonempty finite differences and positive bootstrap replicates")
    rng = np.random.default_rng(seed) if rng is None else rng
    samples = {name: np.empty(reps) for name in arrays}
    for replicate in range(reps):
        weights = rng.poisson(1.0, size=n).astype(np.float64)
        denominator = max(float(weights.sum()), 1.0)
        for name, values in arrays.items():
            samples[name][replicate] = np.sum(weights * values) / denominator
    return {name: {'delta': float(values.mean()),
                   'ci_low': float(np.quantile(samples[name], .025)),
                   'ci_high': float(np.quantile(samples[name], .975))}
            for name, values in arrays.items()}


def comparison_seed(comparison, metric):
    offset = int.from_bytes(hashlib.blake2b(f'{comparison}|{metric}'.encode('utf-8'), digest_size=4).digest(), 'little')
    return 20260711 + offset


def clustered_ci(difference, game_ids, reps=10000, seed=20260711):
    """Resample games, retaining all their rows and weighting by resampled rows."""
    difference = np.asarray(difference, dtype=float)
    if len(difference) != len(game_ids) or not len(difference) or not np.isfinite(difference).all():
        raise ValueError("Expected aligned finite differences and game identifiers")
    _, first, inverse = np.unique(np.asarray(game_ids, dtype=str), return_index=True, return_inverse=True)
    # Preserve first-encounter game order, as in the paper's groupby(sort=False).
    order = np.argsort(first)
    counts = np.bincount(inverse)[order]
    sums = np.bincount(inverse, weights=difference)[order]
    rng = np.random.default_rng(seed)
    samples = np.empty(reps)
    for start in range(0, reps, 100):
        draws = rng.integers(0, len(counts), size=(min(100, reps - start), len(counts)))
        samples[start:start + len(draws)] = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"game_ci_low": float(low), "game_ci_high": float(high)}
