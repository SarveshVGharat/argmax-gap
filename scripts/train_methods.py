#!/usr/bin/env python3
"""Train, select on held-out data, then evaluate the paper's lightweight methods."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from argmax_gap import calibration, features, gates, refinement, selectors


TASKS = {
    "cross_model": "cross_model",
    "maia3_self_top10": "maia3_shortlist_top10",
    "allie_self_top5": "allie_shortlist_top5",
}
FAMILIES = ["linear", "mlp", "xgboost", "gates", "calibration", "refiner", "ensembles"]


def read_frame(path, limit=None):
    if limit:
        return next(pq.ParquetFile(path).iter_batches(batch_size=limit)).to_pandas()
    return pd.read_parquet(path)


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, default=json_value) + "\n")


def train(args):
    out = args.output
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output directory is not empty: {out}. Choose a new directory to avoid mixing runs.")
    for sub in ("models", "candidates", "predictions", "distributions"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    limit = args.smoke_rows if args.smoke else None
    maia, allie = read_frame(args.heldout_maia, limit), read_frame(args.heldout_allie, limit)
    features.validate_alignment(maia, allie)
    tr, va = features.deterministic_split(len(maia), smoke=args.smoke)
    np.savez_compressed(out / "selector_split.npz", train=tr, validation=va, seed=20260605)
    models, selections, searches = {}, {}, []
    heads = set(args.families) & {"linear", "mlp", "xgboost", "gates"}
    for task, nonlinear_task in TASKS.items():
        if not heads:
            break
        if heads <= {"gates"} and task != "maia3_self_top10":
            continue
        print(f"Building {task} candidates", flush=True)
        train_path = out / "candidates" / f"{task}_train.parquet"
        val_path = out / "candidates" / f"{task}_validation.parquet"
        features.write_candidate_parquet(train_path, maia, allie, task, tr,
                                           diagnostic_time=args.diagnostic_time)
        features.write_candidate_parquet(val_path, maia, allie, task, va,
                                           diagnostic_time=args.diagnostic_time)
        validation = pd.read_parquet(val_path)
        linear, rows = selectors.fit_selector(train_path, validation, task)
        searches.extend({"family": "linear", **r} for r in rows)
        selectors.save_model(out / "models" / f"{task}_linear.npz", linear)
        models[f"{task}_linear"] = {"kind": "linear", "task": task, "model": linear}
        selections[f"{task}_linear"] = linear["metadata"]
        if args.diagnostic_time:
            fitted, rows = selectors.fit_selector(train_path, validation, task, diagnostic_time=True)
            name = f"{task}_diagnostic_realized_duration"
            selectors.save_model(out / "models" / f"{name}.npz", fitted)
            models[name] = {"kind": "linear", "task": task, "model": fitted, "diagnostic_time": True}
            selections[name] = fitted["metadata"]
            searches.extend({"family": "diagnostic_realized_duration", **r} for r in rows)
        if set(args.families) & {"mlp", "xgboost", "gates"}:
            train_frame = pd.read_parquet(train_path)
        if set(args.families) & {"mlp", "xgboost"}:
            from argmax_gap import nonlinear
            if "mlp" in args.families and task == "cross_model":
                fitted, rows = nonlinear.fit_mlp_selector(train_frame, validation, nonlinear_task, smoke=args.smoke)
                name = f"{task}_mlp"
                nonlinear.save_mlp(out / "models" / f"{name}.pt", fitted)
                models[name] = {"kind": "mlp", "task": task, "model": fitted}
                selections[name] = {"config": fitted["config"], "model_id": fitted["model_id"]}
                searches.extend({"family": "mlp", **r} for r in rows)
            if "xgboost" in args.families:
                fitted, rows = nonlinear.fit_xgboost_selector(train_frame, validation, nonlinear_task, smoke=args.smoke)
                name = f"{task}_xgboost"
                fitted["model"].save_model(out / "models" / f"{name}.ubj")
                models[name] = {"kind": "xgboost", "task": task, "model": fitted}
                selections[name] = fitted["metadata"]
                searches.extend({"family": "xgboost", **r} for r in rows)
        if task == "maia3_self_top10" and "gates" in args.families:
            rank_train = gates.rank_frame(train_frame, maia, allie)
            rank_val = gates.rank_frame(validation, maia, allie)
            fitted, rows = gates.fit_single_rank(rank_train, rank_val, 2)
            name = "maia3_rank2_gate"
            selectors.save_model(out / "models" / f"{name}.npz", fitted["model"])
            models[name] = {"kind": "gate", "task": task, "model": fitted}
            selections[name] = {k: v for k, v in fitted.items() if k != "model"}
            selections[name]["metadata"] = fitted["model"]["metadata"]
            searches.extend({"family": "single_rank_gate", **r} for r in rows)
            del rank_train, rank_val
        del validation
        if "train_frame" in locals():
            del train_frame
        gc.collect()
    if "ensembles" in args.families:
        selected, rows = calibration.fit_ensembles(maia, allie)
        searches.extend({"family": "ensemble", **r} for r in rows)
        for kind, selected_row in selected.items():
            name = f"ensemble_{kind}"
            models[name] = {"kind": "ensemble", "model": selected_row}
            selections[name] = selected_row
    for base, frame in (("maia3", maia), ("allie", allie)):
        if "calibration" in args.families:
            params = calibration.fit_fine_calibration(frame)
            write_json(out / "models" / f"{base}_calibration.json", params)
            for variant in ("time_bucket",):
                name = f"{base}_{variant}_calibration"
                models[name] = {"kind": "calibration", "base": base, "variant": variant, "model": params}
                selections[name] = params
        if "refiner" in args.families:
            params = calibration.fit_calibration_for_base(base, frame)
            rt, rv = features.deterministic_split(len(frame), seed=20260606, smoke=args.smoke)
            np.savez_compressed(out / "refiner_split.npz", train=rt, validation=rv, seed=20260606)
            fitted, rows = refinement.fit_refiner(frame, params, base, rt, rv, out / "candidates",
                                                  seed_offset=0 if base == "maia3" else 36)
            name = f"{base}_refiner"
            selectors.save_model(out / "models" / f"{name}.npz", fitted["model"])
            models[name] = {"kind": "refiner", "base": base, "model": fitted}
            selections[name] = {k: v for k, v in fitted.items() if k != "model"}
            searches.extend({"family": "refiner", **r} for r in rows)
    pd.DataFrame(searches).to_csv(out / "validation_search.csv", index=False)
    write_json(out / "selected.json", {"smoke_test": args.smoke, "heldout_rows": len(maia),
               "selector_seed": 20260605, "refiner_seed": 20260606, "methods": selections})
    del maia, allie
    gc.collect()
    return models


def evaluate(args, models):
    """Test data is first opened after all validation selections have been saved."""
    writers = {}
    total = 0
    miter = pq.ParquetFile(args.test_maia).iter_batches(batch_size=args.batch_size)
    aiter = pq.ParquetFile(args.test_allie).iter_batches(batch_size=args.batch_size)
    try:
        for mb, ab in zip(miter, aiter, strict=True):
            maia, allie = mb.to_pandas(), ab.to_pandas()
            if args.smoke:
                remaining = args.smoke_rows - total
                maia, allie = maia.iloc[:remaining], allie.iloc[:remaining]
            features.validate_alignment(maia, allie)
            candidates, ranked = {}, None
            for name, spec in models.items():
                kind, fitted = spec["kind"], spec["model"]
                if "task" in spec:
                    task = spec["task"]
                    if task not in candidates:
                        candidates[task] = features.candidate_frame(maia, allie, task,
                                                                   diagnostic_time=args.diagnostic_time)
                    cand = candidates[task]
                if kind == "linear":
                    result = selectors.hard_predictions(selectors.select_top_candidate(cand, selectors.selector_scores(cand, fitted)))
                elif kind in ("mlp", "xgboost"):
                    from argmax_gap import nonlinear
                    scores = nonlinear.mlp_scores(fitted, cand) if kind == "mlp" else nonlinear.xgboost_scores(fitted, cand)
                    result = selectors.hard_predictions(nonlinear.candidate_selection(cand, scores))
                elif kind == "gate":
                    if ranked is None:
                        ranked = gates.rank_frame(cand, maia, allie)
                    result = gates.predict_single_rank(ranked, fitted)
                elif kind == "ensemble":
                    result = calibration.apply_ensemble(maia, allie, fitted["kind"], fitted["alpha"])
                else:
                    frame = maia if spec["base"] == "maia3" else allie
                    if kind == "refiner":
                        result = refinement.apply_refiner(frame, fitted)
                    else:
                        result = calibration.apply_calibration(frame, fitted, spec.get("variant", "global"), coarse=kind == "coarse_calibration")
                subdir = "distributions" if "legal_probs" in result else "predictions"
                table = pa.Table.from_pandas(result, preserve_index=False)
                if name not in writers:
                    writers[name] = pq.ParquetWriter(args.output / subdir / f"{name}.parquet", table.schema, compression="zstd")
                writers[name].write_table(table)
            total += len(maia)
            print(f"Evaluated {total:,} test positions", flush=True)
            if args.smoke and total >= args.smoke_rows:
                break
    finally:
        for writer in writers.values():
            writer.close()
    if not args.smoke and total != 884_049:
        raise ValueError(f"Expected 884,049 paper test positions, got {total}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("heldout-maia", "heldout-allie", "test-maia", "test-allie"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=FAMILIES)
    parser.add_argument("--diagnostic-time", action="store_true", help="Also reproduce duration-containing diagnostic selectors (Table 15)")
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--smoke", action="store_true", help="Small end-to-end run; reduced nonlinear search, not paper reproduction")
    parser.add_argument("--smoke-rows", type=int, default=500)
    args = parser.parse_args()
    models = train(args)
    evaluate(args, models)


if __name__ == "__main__":
    main()
