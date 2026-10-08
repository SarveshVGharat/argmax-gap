"""Check full-support mixtures and rejected misaligned sweep inputs."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from argmax_gap.calibration import mix_probs

spec = importlib.util.spec_from_file_location("ensemble_sweep", ROOT / "scripts" / "ensemble_sweep.py")
ensemble = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ensemble)


class EnsembleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.maia = self.root / "maia.parquet"
        self.allie = self.root / "allie.parquet"
        self.mrows = [
            {"row_id": 0, "human_move_uci": "e2e4", "legal_moves_uci": ["e2e4", "d2d4"], "legal_probs": [.8, .2]},
            {"row_id": 1, "human_move_uci": "e7e5", "legal_moves_uci": ["e7e5"], "legal_probs": [1.]},
        ]
        self.arows = [
            {"row_id": 0, "human_move_uci": "e2e4", "legal_moves_uci": ["d2d4", "e2e4"], "legal_probs": [.7, .3]},
            {"row_id": 1, "human_move_uci": "e7e5", "legal_moves_uci": ["e7e5"], "legal_probs": [1.]},
        ]
        self.write()

    def tearDown(self):
        self.temp.cleanup()

    def write(self):
        pq.write_table(pa.Table.from_pylist(self.mrows), self.maia, row_group_size=1)
        pq.write_table(pa.Table.from_pylist(self.arows), self.allie, row_group_size=2)

    def test_vectorized_mixtures_match_scalar_implementation(self):
        frame = ensemble.sweep(self.maia, self.allie, expected_rows=2, batch_size=2)
        self.assertEqual(len(frame), 42)
        for row in frame.itertuples(index=False):
            p = mix_probs(np.array([.8, .2]), np.array([.3, .7]), row.mixture_type, row.alpha_maia3)
            self.assertAlmostEqual(row.NLL, -np.log(p[0]) / 2, places=12)
            correct = 1 + int(not (p > p[0] + 1e-15).any())
            self.assertAlmostEqual(row.Top1, 100 * correct / 2)
        # A one-legal-move row must not acquire mass on padded entries.
        allie = frame.loc[(frame.mixture_type == "convex") & (frame.alpha_maia3 == 0)].iloc[0]
        maia = frame.loc[(frame.mixture_type == "convex") & (frame.alpha_maia3 == 1)].iloc[0]
        self.assertEqual(allie.Top1, 50)
        self.assertEqual(maia.Top1, 100)

    def test_bad_probabilities_alignment_and_row_counts_fail(self):
        with self.assertRaisesRegex(ValueError, "exactly"):
            ensemble.sweep(self.maia, self.allie, expected_rows=3)
        self.arows[0]["legal_probs"] = [.7, float("nan")]
        self.write()
        with self.assertRaisesRegex(ValueError, "normalized"):
            ensemble.sweep(self.maia, self.allie, expected_rows=2)
        self.arows[0]["legal_probs"] = [.7, .3]
        self.arows[0]["human_move_uci"] = "d2d4"
        self.write()
        with self.assertRaisesRegex(ValueError, "target moves"):
            ensemble.sweep(self.maia, self.allie, expected_rows=2)

    def test_duplicate_ids_are_rejected_across_batches(self):
        self.mrows[1]["row_id"] = self.arows[1]["row_id"] = 0
        self.write()
        with self.assertRaisesRegex(ValueError, "unique"):
            ensemble.sweep(self.maia, self.allie, expected_rows=2, batch_size=1)


if __name__ == "__main__":
    unittest.main()
