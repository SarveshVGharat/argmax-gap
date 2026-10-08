#!/usr/bin/env python3
"""Reproduce the descriptive probability-ensemble sweep in Figure 7.

This reports the complete fixed alpha grid. It does not select an alpha from
test labels; train_methods.py selects the deployed mixtures on held-out NLL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ALPHAS = np.round(np.arange(0.0, 1.0001, 0.05), 2)
EPSILON = 1e-12
RANK_TOLERANCE = 1e-15
COLUMNS = ["row_id", "human_move_uci", "legal_moves_uci", "legal_probs"]


def aligned_arrays(maia: dict, allie: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate and align one batch, preserving every row's legal support."""
    if maia["row_id"] != allie["row_id"] or maia["human_move_uci"] != allie["human_move_uci"]:
        raise ValueError("MAIA3 and Allie row IDs or target moves differ")
    if "game_id" in maia and "game_id" in allie and maia["game_id"] != allie["game_id"]:
        raise ValueError("MAIA3 and Allie game IDs differ")
    lengths = np.asarray([len(moves) for moves in maia["legal_moves_uci"]])
    if not len(lengths) or (lengths == 0).any():
        raise ValueError("Expected nonempty legal move lists")
    pm = np.zeros((len(lengths), int(lengths.max())), dtype=np.float64)
    pa = np.zeros_like(pm)
    human = np.empty(len(lengths), dtype=np.int64)
    legal = np.arange(pm.shape[1])[None, :] < lengths[:, None]
    for i, (moves, other_moves, target, probs, other_probs) in enumerate(zip(
        maia["legal_moves_uci"], allie["legal_moves_uci"], maia["human_move_uci"],
        maia["legal_probs"], allie["legal_probs"], strict=True,
    )):
        if len(moves) != len(set(moves)) or len(other_moves) != len(set(other_moves)):
            raise ValueError(f"Duplicate legal moves at row {maia['row_id'][i]}")
        if set(moves) != set(other_moves) or target not in moves:
            raise ValueError(f"Legal move set or target mismatch at row {maia['row_id'][i]}")
        if len(probs) != len(moves) or len(other_probs) != len(other_moves):
            raise ValueError(f"Probability length mismatch at row {maia['row_id'][i]}")
        aligned = dict(zip(other_moves, other_probs, strict=True))
        pm[i, :len(moves)] = probs
        pa[i, :len(moves)] = [aligned[move] for move in moves]
        human[i] = moves.index(target)
    for values in (pm, pa):
        totals = values.sum(axis=1)
        if not np.isfinite(values).all() or (values < 0).any() or not np.allclose(totals, 1, atol=2e-5, rtol=0):
            raise ValueError("Probabilities must be finite, nonnegative, and normalized")
        values /= totals[:, None]
    return pm, pa, human, legal


def sweep(maia_path: Path, allie_path: Path, expected_rows: int = 884049,
          batch_size: int = 2048) -> pd.DataFrame:
    if expected_rows < 1 or batch_size < 1:
        raise ValueError("Expected row count and batch size must be positive")
    maia_file, allie_file = pq.ParquetFile(maia_path), pq.ParquetFile(allie_path)
    if maia_file.metadata.num_rows != expected_rows or allie_file.metadata.num_rows != expected_rows:
        raise ValueError(f"Both input files must contain exactly {expected_rows} rows")
    shared_optional = {"game_id"} & set(maia_file.schema_arrow.names) & set(allie_file.schema_arrow.names)
    columns = COLUMNS + sorted(shared_optional)
    miter = maia_file.iter_batches(batch_size=batch_size, columns=columns)
    aiter = allie_file.iter_batches(batch_size=batch_size, columns=columns)
    nll = np.zeros((2, len(ALPHAS)), dtype=np.float64)
    top = np.zeros((2, len(ALPHAS), 3), dtype=np.int64)
    rows, previous_id = 0, None
    for maia_batch, allie_batch in zip(miter, aiter, strict=True):
        maia, allie = maia_batch.to_pydict(), allie_batch.to_pydict()
        ids = np.asarray(maia["row_id"], dtype=np.int64)
        if (np.diff(ids) <= 0).any() or (previous_id is not None and ids[0] <= previous_id):
            raise ValueError("Prediction row IDs must be unique and increasing")
        previous_id = int(ids[-1])
        pm, pa, human, legal = aligned_arrays(maia, allie)
        log_m, log_a = np.log(np.maximum(pm, EPSILON)), np.log(np.maximum(pa, EPSILON))
        index = np.arange(len(pm))
        for kind_index, kind in enumerate(("convex", "geometric")):
            for alpha_index, alpha in enumerate(ALPHAS):
                if kind == "convex":
                    mixed = alpha * pm + (1.0 - alpha) * pa
                else:
                    logits = alpha * log_m + (1.0 - alpha) * log_a
                    logits[~legal] = -np.inf
                    mixed = np.exp(logits - logits.max(axis=1, keepdims=True))
                mixed /= mixed.sum(axis=1, keepdims=True)
                human_p = np.maximum(mixed[index, human], EPSILON)
                nll[kind_index, alpha_index] -= np.log(human_p).sum()
                # Preserve the numerical comparison used for the original sweep.
                ranks = 1 + (mixed > human_p[:, None] + RANK_TOLERANCE).sum(axis=1)
                top[kind_index, alpha_index] += [(ranks <= k).sum() for k in (1, 3, 5)]
        rows += len(pm)
    if rows != expected_rows:
        raise ValueError(f"Expected {expected_rows} aligned rows, got {rows}")
    output = []
    for kind_index, kind in enumerate(("convex", "geometric")):
        for i, alpha in enumerate(ALPHAS):
            output.append({"mixture_type": kind, "alpha_maia3": float(alpha), "rows": rows,
                           "NLL": float(nll[kind_index, i] / rows),
                           **{f"Top{k}": float(100 * top[kind_index, i, j] / rows)
                              for j, k in enumerate((1, 3, 5))}})
    return pd.DataFrame(output)


def plot_sweep(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.1))
    for kind, color in [("convex", "#2563eb"), ("geometric", "#f97316")]:
        values = frame.loc[frame.mixture_type.eq(kind)].sort_values("alpha_maia3")
        for ax, metric in zip(axes, ["NLL", "Top1"], strict=True):
            ax.plot(values.alpha_maia3, values[metric], marker="o", markersize=3,
                    color=color, label=kind.capitalize())
    for ax, ylabel in zip(axes, ["NLL", "Top1 (%)"], strict=True):
        ax.set(xlabel="MAIA3 mixture weight", ylabel=ylabel, xlim=(0, 1))
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maia", required=True, type=Path)
    parser.add_argument("--allie", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-rows", type=int, default=884049)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()
    start = time.perf_counter()
    result = sweep(args.maia, args.allie, args.expected_rows, args.batch_size)
    args.output.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output / "ensemble_sweep.csv", index=False)
    plot_sweep(result, args.output / "ensemble_sweep.pdf")
    elapsed = time.perf_counter() - start
    (args.output / "run.json").write_text(json.dumps({
        "rows": args.expected_rows, "mixtures": 2, "alphas": ALPHAS.tolist(),
        "alpha_selected_on_test": False, "probability_floor": EPSILON,
        "rank_comparison_tolerance": RANK_TOLERANCE, "elapsed_seconds": elapsed,
    }, indent=2) + "\n")
    print(f"Wrote ensemble_sweep.csv and ensemble_sweep.pdf for {args.expected_rows:,} rows in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
