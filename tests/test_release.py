"""Independence, legal-distribution alignment, and numerical release checks."""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import chess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("audit_splits", ROOT / "scripts" / "audit_splits.py")
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)
check_spec = importlib.util.spec_from_file_location("check_reference", ROOT / "scripts" / "check_reference.py")
reference_check = importlib.util.module_from_spec(check_spec)
check_spec.loader.exec_module(reference_check)


def position(game="test-game", move="e2e4"):
    board = chess.Board()
    return {"game_id": game, "ply_index": 0, "fen_before": board.fen(),
            "human_move_uci": move, "legal_moves_uci": [m.uci() for m in board.legal_moves],
            "past_fens": []}


def prediction(row, row_id=0):
    moves = list(reversed(row["legal_moves_uci"]))
    probabilities = [1.0 / len(moves)] * len(moves)
    return {"row_id": row_id, "game_id": row["game_id"], "human_move_uci": row["human_move_uci"],
            "legal_moves_uci": moves, "legal_probs": probabilities, "p_human": probabilities[0],
            "p_top1": probabilities[0], "human_rank": 1, "top1_move_uci": moves[0]}


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.test = self.write("test", [position()])
        self.heldout = self.write("heldout", [position("development-game")])

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, rows):
        path = self.root / (name + ".parquet")
        pq.write_table(pa.Table.from_pylist(rows), path)
        return path

    def args(self):
        return argparse.Namespace(test=self.test, heldout=self.heldout, selector_split=None, refiner_split=None,
                                  test_maia3=None, test_allie=None, heldout_maia3=None, heldout_allie=None)

    def test_repeated_boards_in_independent_games_are_allowed(self):
        report = audit.build_report(self.args())
        self.assertTrue(report["passed"])
        self.assertEqual(report["overlaps"][0]["canonical_overlap"], 1)
        self.assertEqual(report["overlaps"][0]["game_overlap"], 0)
        self.assertEqual(report["overlaps"][0]["stored_history_context_overlap"], 1)

    def test_shared_game_is_rejected_even_with_different_target(self):
        self.heldout = self.write("heldout", [position(move="d2d4")])
        report = audit.build_report(self.args())
        self.assertFalse(report["passed"])
        self.assertEqual(report["overlaps"][0]["row_overlap"], 0)
        self.assertEqual(report["overlaps"][0]["game_overlap"], 1)
        self.assertEqual(report["overlaps"][0]["game_context_overlap"], 1)

    def test_shared_game_url_and_bare_id_are_rejected(self):
        self.test = self.write("test", [position("https://lichess.org/Ab12Cd34/black?x=1")])
        development = position("Ab12Cd34")
        development.update(ply_index=1, previous_moves_uci="")
        self.heldout = self.write("heldout", [development])
        report = audit.build_report(self.args())
        self.assertFalse(report["passed"])
        pair = report["overlaps"][0]
        self.assertEqual((pair["game_overlap"], pair["row_overlap"], pair["game_context_overlap"]), (1, 1, 1))

    def test_explicit_target_index_normalizes_one_based_metadata(self):
        self.test = self.write("test", [position("https://lichess.org/Ab12Cd34")])
        development = position("Ab12Cd34")
        development.update(ply_index=1, target_ply_index=0)
        self.heldout = self.write("heldout", [development])
        report = audit.build_report(self.args())
        self.assertFalse(report["passed"])
        self.assertEqual(report["overlaps"][0]["row_overlap"], 1)

    def test_target_index_disagreeing_with_prefix_is_rejected(self):
        row = position()
        row.update(target_ply_index=1, previous_moves_uci="")
        self.test = self.write("test", [row])
        with self.assertRaisesRegex(ValueError, "Target ply disagrees"):
            audit.build_report(self.args())

    def test_duplicate_row_and_missing_legal_move_are_rejected(self):
        row = position()
        row["legal_moves_uci"].remove("e2e4")
        self.test = self.write("test", [row, row])
        report = audit.build_report(self.args())
        self.assertFalse(report["passed"])
        self.assertEqual(report["splits"]["test"]["duplicate_row_keys"], 1)
        self.assertEqual(report["splits"]["test"]["issues"]["target_not_legal"], 2)

    def test_legal_move_order_and_ties_are_valid(self):
        path = self.write("predictions", [prediction(position())])
        result = audit.audit_predictions(self.test, path)
        self.assertEqual(result["issues"], {})

    def test_nonfinite_and_misaligned_predictions_are_rejected(self):
        row = prediction(position())
        row["legal_probs"][0] = float("nan")
        row["human_move_uci"] = "d2d4"
        row["row_id"] = 42
        path = self.write("predictions", [row])
        result = audit.audit_predictions(self.test, path)
        self.assertEqual(result["issues"]["invalid_probabilities"], 1)
        self.assertEqual(result["issues"]["target_mismatch"], 1)
        self.assertEqual(result["issues"]["row_id_order_mismatch"], 1)

    def test_nonfinite_logits_are_counted_independently_of_probabilities_and_rank(self):
        row = prediction(position())
        row["legal_probs"][0] = float("nan")
        row["legal_logits"] = [float("nan")] * len(row["legal_probs"])
        del row["human_rank"]
        path = self.write("predictions", [row])
        result = audit.audit_predictions(self.test, path)
        self.assertEqual(result["nonfinite_probability_rows"], 1)
        self.assertEqual(result["nonfinite_logit_rows"], 1)
        self.assertEqual(result["issues"]["invalid_logits"], 1)
        self.assertEqual(result["logit_rows_checked"], 1)

    def test_full_development_aliases_cover_calibration_and_ensembles(self):
        report = audit.build_report(self.args())
        for name in ("calibration_fit", "ensemble_weight_selection"):
            self.assertEqual(report["splits"][name], report["splits"]["heldout"])
            expected = {**report["overlaps"][0], "second": name}
            self.assertIn(expected, report["overlaps"])

    def test_prediction_missing_rows_and_duplicate_ids_are_rejected(self):
        row = prediction(position())
        self.test = self.write("test", [position(), position("second"), position("third")])
        path = self.write("predictions", [row, copy.deepcopy(row)])
        result = audit.audit_predictions(self.test, path)
        self.assertEqual(result["issues"]["row_count_mismatch"], 1)
        self.assertEqual(result["issues"]["duplicate_row_id"], 1)

    def test_selector_split_indices_validate_membership_and_overlap(self):
        path = self.root / "split.npz"
        np.savez(path, selector_train=np.array([0, 1]), selector_val=np.array([2]))
        train, val = audit.selector_indices(path, 3)
        self.assertEqual(train.tolist(), [0, 1])
        self.assertEqual(val.tolist(), [2])
        with self.assertRaisesRegex(ValueError, "out-of-bounds"):
            audit.selector_indices(path, 2)
        np.savez(path, selector_train=np.array([0, 1]), selector_val=np.array([1]))
        with self.assertRaisesRegex(ValueError, "overlap"):
            audit.selector_indices(path, 3)

    def test_refiner_and_selector_subsets_are_both_audited(self):
        self.heldout = self.write("heldout", [position("dev-one"), position("dev-two"), position("dev-three")])
        selector_path, refiner_path = self.root / "selector.npz", self.root / "refiner.npz"
        np.savez(selector_path, train=np.array([0, 1]), validation=np.array([2]))
        np.savez(refiner_path, refiner_train=np.array([1, 2]), refiner_val=np.array([0]))
        args = self.args()
        args.selector_split, args.refiner_split = selector_path, refiner_path
        report = audit.build_report(args)
        self.assertTrue(report["passed"])
        self.assertEqual(report["splits"]["refiner_train"]["rows"], 2)
        self.assertEqual(report["splits"]["refiner_val"]["rows"], 1)
        pairs = {(row["first"], row["second"]): row for row in report["overlaps"]}
        for family in ("selector", "refiner"):
            for subset in ("train", "val"):
                self.assertEqual(pairs[("test", f"{family}_{subset}")]["game_overlap"], 0)
            self.assertEqual(pairs[(f"{family}_train", f"{family}_val")]["row_overlap"], 0)

    def test_refiner_split_overlap_is_rejected(self):
        path = self.root / "refiner.npz"
        np.savez(path, train=np.array([0]), validation=np.array([0]))
        args = self.args()
        args.refiner_split = path
        with self.assertRaisesRegex(ValueError, "Refiner training and validation row indices overlap"):
            audit.build_report(args)

    def test_reference_rescue_break_arithmetic(self):
        with (ROOT / "reference" / "paper_results.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if row["rescues"] and row["breaks"]:
                net = int(row["rescues"]) - int(row["breaks"])
                self.assertEqual(net, int(row["net"]))
                self.assertAlmostEqual(100 * net / int(row["rows"]), float(row["delta_top1_pp"]), places=10)
        gate = next(row for row in rows if row["method"] == "rank-2 correction gate")
        self.assertEqual(int(gate["net"]), 1207)
        self.assertAlmostEqual(float(gate["top1_percent"]), 57.39138893884841)

    def test_figure_one_moves_are_legal_and_distinct(self):
        example = json.loads((ROOT / "reference" / "figure1.json").read_text())
        board = chess.Board(example["fen_before"])
        moves = [example[key] for key in ["human_move_uci", "maia3_top1_move_uci", "allie_top1_move_uci"]]
        self.assertEqual(len(set(moves)), 3)
        for move in moves:
            self.assertIn(chess.Move.from_uci(move), board.legal_moves)
        self.assertEqual(example["maia3_human_rank"], 2)
        self.assertEqual(example["allie_human_rank"], 2)

    def test_reference_checker_fails_on_wrong_counts_and_estimates(self):
        report = self.root / "report"
        report.mkdir()
        path = report / "metrics.csv"
        path.write_text("method,rows,Top1,NLL\nmaia3,884049,57.2548580452,1.2922393641\n")
        reference = ROOT / "reference" / "paper_results.csv"
        self.assertTrue(reference_check.compare(report, reference)["passed"])
        self.assertFalse(reference_check.compare(report, reference, require_all=True)["passed"])
        path.write_text("method,rows,Top1,NLL\nmaia3,4,57.2548580452,1.2922393641\n")
        result = reference_check.compare(report, reference)
        self.assertFalse(result["passed"])
        self.assertEqual(result["failures"][0]["metric"], "rows")
        path.write_text("method,rows,Top1,NLL\nmaia3,884049,56.0,1.2922393641\n")
        self.assertFalse(reference_check.compare(report, reference)["passed"])


if __name__ == "__main__":
    unittest.main()
