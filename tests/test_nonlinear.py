"""Validation selection and publication of the three nonlinear selector tasks."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from argmax_gap import calibration, features, nonlinear


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train_methods", ROOT / "scripts" / "train_methods.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def positions(n=24):
    moves = ["e2e4", "d2d4", "g1f3", "c2c4", "b1c3", "a2a3"]
    frame = pd.DataFrame([{"row_id": i, "human_move_uci": moves[i % 6],
        "legal_moves_uci": moves, "player_elo": 1500 + i, "opponent_elo": 1600,
        "move_number": 15, "phase": "middlegame", "time_spent_seconds": 3.0}
        for i in range(n)])
    rng = np.random.default_rng(190)
    return tuple(calibration.distribution_frame(frame, rng.dirichlet(np.ones(6), n)) for _ in range(2))


def constant_linear():
    return {"weights": np.zeros(45, np.float32), "mean": np.zeros(45, np.float32),
            "std": np.ones(45, np.float32), "bias": 0., "feature_names": features.PREMOVE_FEATURES}


class NonlinearSelectionTests(unittest.TestCase):
    def test_linear_head_wins_identical_mlp_predictions_without_relabeling(self):
        maia, allie = positions()
        frame = features.candidate_frame(maia, allie, "cross_model")
        with patch.object(nonlinear, "mlp_scores", return_value=np.full(len(frame), .5)):
            selected, rows = nonlinear.fit_mlp_selector(frame, frame, "cross_model",
                linear_model=constant_linear(), smoke=True)
        self.assertEqual(selected["kind"], "linear")
        self.assertEqual(len(rows), 2)
        self.assertEqual([r["model_family"] for r in rows if r["selected_by_validation"]], ["linear"])
        self.assertIn("best_val_loss", rows[1])
        self.assertEqual(rows[1]["epochs_run"], 4)

    def test_mlp_can_replace_linear_baseline(self):
        maia, allie = positions()
        frame = features.candidate_frame(maia, allie, "cross_model")
        with patch.object(nonlinear, "mlp_scores", return_value=frame.candidate_correct.to_numpy(float)):
            selected, rows = nonlinear.fit_mlp_selector(frame, frame, "cross_model",
                linear_model=constant_linear(), smoke=True)
        self.assertEqual(selected["kind"], "mlp")
        self.assertGreater(selected["validation"]["Top1"], rows[0]["Top1"])
        self.assertEqual(sum(r["selected_by_validation"] for r in rows), 1)

    def test_mlp_command_records_all_three_searches_and_selected_artifacts(self):
        maia, allie = positions()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            maia.to_parquet(tmp / "maia.parquet")
            allie.to_parquet(tmp / "allie.parquet")
            args = argparse.Namespace(output=tmp / "methods", smoke=True, smoke_rows=len(maia),
                heldout_maia=tmp / "maia.parquet", heldout_allie=tmp / "allie.parquet",
                families=["mlp"], diagnostic_time=False)
            models = runner.train(args)
            search = pd.read_csv(args.output / "validation_search.csv")
            mlp_search = search[search.family.eq("mlp")]
            self.assertEqual(set(mlp_search.task), set(runner.TASKS.values()))
            for _, rows in mlp_search.groupby("task"):
                self.assertEqual(set(rows.model_family), {"linear", "mlp"})
                self.assertEqual(rows.selected_by_validation.fillna(False).sum(), 1)
            selections = json.loads((args.output / "selected.json").read_text())["methods"]
            for task in runner.TASKS:
                selected = selections[f"{task}_mlp_search"]
                name = selected["selected_method"]
                self.assertIn(name, models)
                self.assertEqual(models[name]["kind"], selected["model_family"])

    def test_gate_only_command_does_not_fit_unrequested_linear_selectors(self):
        maia, allie = positions()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            maia.to_parquet(tmp / "maia.parquet")
            allie.to_parquet(tmp / "allie.parquet")
            args = argparse.Namespace(output=tmp / "methods", smoke=True, smoke_rows=len(maia),
                heldout_maia=tmp / "maia.parquet", heldout_allie=tmp / "allie.parquet",
                families=["gates"], diagnostic_time=False)
            with patch.object(runner.selectors, "fit_selector", side_effect=AssertionError("Unrequested selector fit")):
                models = runner.train(args)
            self.assertEqual(set(models), {"maia3_rank2_gate"})


if __name__ == "__main__":
    unittest.main()
