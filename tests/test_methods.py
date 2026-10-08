"""Numerical and leakage checks for the learned-method reproduction."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from argmax_gap import calibration, features, gates, refinement, selectors


def example_frames(n=32):
    rng = np.random.default_rng(80)
    rows = []
    moves = ["e2e4", "d2d4", "g1f3", "c2c4", "b1c3", "a2a3"]
    for i in range(n):
        rows.append({"row_id": i, "human_move_uci": moves[i % len(moves)],
            "legal_moves_uci": moves, "player_elo": 1500 + i * 10, "opponent_elo": 1600,
            "move_number": 12 + i, "phase": "middlegame", "time_spent_seconds": i % 10})
    frame = pd.DataFrame(rows)
    maia = calibration.distribution_frame(frame, rng.dirichlet(np.ones(6), n))
    allie = calibration.distribution_frame(frame, rng.dirichlet(np.ones(6), n))
    return maia, allie


class MethodTests(unittest.TestCase):
    def test_premove_features_independent_of_labels_and_realized_time(self):
        maia, allie = example_frames()
        original = features.candidate_frame(maia, allie, "cross_model")
        maia["time_spent_seconds"] = 1e9
        maia["human_move_uci"] = "e2e4"
        allie["human_move_uci"] = "e2e4"
        changed = features.candidate_frame(maia, allie, "cross_model")
        self.assertEqual(len(features.PREMOVE_FEATURES), 45)
        np.testing.assert_array_equal(original[features.PREMOVE_FEATURES], changed[features.PREMOVE_FEATURES])
        self.assertFalse(set(features.POST_DECISION_FEATURES) & set(original.columns))
        diagnostic = features.candidate_frame(maia, allie, "cross_model", diagnostic_time=True)
        self.assertTrue(set(features.DIAGNOSTIC_FEATURES) <= set(diagnostic.columns))

    def test_position_split_is_reproducible_and_disjoint(self):
        tr, va = features.deterministic_split(500_000)
        self.assertEqual((len(tr), len(va)), (400_000, 100_000))
        self.assertEqual(len(np.intersect1d(tr, va)), 0)
        np.testing.assert_array_equal(tr, features.deterministic_split(500_000)[0])
        self.assertFalse(np.array_equal(tr, features.deterministic_split(500_000, seed=20260606)[0]))

    def test_selector_trains_and_prefers_slot_zero_on_ties(self):
        maia, allie = example_frames()
        train = features.candidate_frame(maia, allie, "cross_model")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.parquet"
            train.to_parquet(path)
            model, rows = selectors.fit_selector(path, train, "cross_model")
            self.assertTrue(np.isfinite(selectors.selector_scores(train, model)).all())
            self.assertEqual(len(rows), 2)
            expected = train[features.PREMOVE_FEATURES].to_numpy().mean(axis=0)
            np.testing.assert_allclose(model["mean"], expected, rtol=1e-6, atol=1e-6)
        chosen = selectors.select_top_candidate(train, np.ones(len(train)))
        self.assertTrue(chosen.candidate_slot.eq(0).all())

    def test_threshold_search_matches_exhaustive_rule_including_ties(self):
        scores = np.array([.2, .2, .1, -.1, -.2, -.2])
        base = np.array([False, True, False, True, False, True])
        delta = np.array([1, -1, 1, -1, 0, 0])
        chosen = gates.exact_best_threshold(scores, delta, base)
        thresholds = [np.inf, *[np.nextafter(v, -np.inf) for v in np.unique(scores)]]
        brute = gates.select_from_rows([gates.threshold_metrics(scores, delta, base, t) for t in thresholds])
        for key in ("Top1", "rescues", "breaks", "net", "switch_rate"):
            self.assertEqual(chosen[key], brute[key])
        self.assertEqual(gates.threshold_metrics(np.array([.2]), np.array([1]), np.array([False]), .2)["switch_rate"], 0)

    def test_rank2_model_has_65_features_and_keeps_legal_candidates(self):
        maia, allie = example_frames()
        candidates = features.candidate_frame(maia, allie, "maia3_self_top10")
        frame = gates.rank_frame(candidates, maia, allie)
        self.assertEqual(len(gates.RANK_FEATURES), 65)
        fitted, rows = gates.fit_single_rank(frame, frame, 2)
        pred = gates.predict_single_rank(frame, fitted)
        self.assertEqual(len(rows), 5)
        for p, (_, row) in zip(pred.top1_move_uci, maia.iterrows()):
            order = np.argsort(-row.legal_probs, kind="stable")[:2]
            self.assertIn(p, [row.legal_moves_uci[i] for i in order])

    def test_temperature_preserves_rank_and_ensemble_aligns_moves(self):
        maia, allie = example_frames()
        for p in maia.legal_probs:
            q = calibration.softmax_temperature(None, p, .7)
            self.assertEqual(np.argmax(q), np.argmax(p))
            self.assertAlmostEqual(q.sum(), 1)
        expected = calibration.apply_ensemble(maia, allie, "convex", .7)
        allie["legal_moves_uci"] = [list(reversed(m)) for m in allie.legal_moves_uci]
        allie["legal_probs"] = [p[::-1] for p in allie.legal_probs]
        actual = calibration.apply_ensemble(maia, allie, "convex", .7)
        for a, b in zip(expected.legal_probs, actual.legal_probs):
            np.testing.assert_array_equal(a, b)

    def test_gate_keeps_strict_rank_base_correctness_on_unswitched_ties(self):
        maia, allie = example_frames(2)
        maia["human_move_uci"] = "d2d4"
        allie["human_move_uci"] = "d2d4"
        maia = calibration.distribution_frame(maia, [[.4, .4, .05, .05, .05, .05]] * 2)
        candidates = features.candidate_frame(maia, allie, "maia3_self_top10")
        frame = gates.rank_frame(candidates, maia, allie)
        model = {"mean": np.zeros(65), "std": np.ones(65), "weights": np.zeros(65),
                 "bias": np.zeros(1), "metadata": {"family": "ridge_delta_numpy"}}
        result = gates.predict_single_rank(frame, {"rank": 2, "model": model, "threshold": 0})
        self.assertTrue(result.correct_top1.all())
        self.assertFalse(result["switch"].any())
        self.assertTrue(result.top1_move_uci.eq("e2e4").all())

    def test_refiner_zero_residual_recovers_calibrated_distribution(self):
        maia, _ = example_frames()
        params = {"selected": "global", "global": {"temperature": .9}}
        model = {"weights": np.zeros(len(refinement.PREMOVE_FEATURES)), "bias": 0.,
                 "mean": np.zeros(len(refinement.PREMOVE_FEATURES)),
                 "std": np.ones(len(refinement.PREMOVE_FEATURES)),
                 "feature_names": refinement.PREMOVE_FEATURES}
        fitted = {"model": model, "calibration": params, "base": "maia3", "topk": 5, "alpha": .1}
        actual = refinement.apply_refiner(maia, fitted)
        expected = calibration.apply_calibration(maia, params, coarse=True)
        for a, b in zip(expected.legal_probs, actual.legal_probs):
            np.testing.assert_allclose(a, b, atol=1e-15)

    def test_refiner_preserves_distinct_tiny_tail_probabilities(self):
        maia, _ = example_frames(1)
        p = np.array([.5, .3, .15, .05, 1e-60, 1e-70])
        maia.at[0, "human_move_uci"] = maia.at[0, "legal_moves_uci"][-1]
        maia.at[0, "legal_probs"] = p
        maia.at[0, "legal_logits"] = np.log(p)
        maia.at[0, "p_human"] = p[-1]
        maia.at[0, "is_top1"] = False
        params = {"selected": "global", "global": {"temperature": .75}}
        dimension = len(refinement.PREMOVE_FEATURES)
        model = {"weights": np.zeros(dimension, np.float32), "bias": 0.,
                 "mean": np.zeros(dimension, np.float32), "std": np.ones(dimension, np.float32),
                 "feature_names": refinement.PREMOVE_FEATURES}
        result = refinement.apply_refiner(maia, {"model": model, "calibration": params,
            "base": "maia3", "topk": 5, "alpha": .05}).iloc[0]
        expected = calibration.softmax_temperature(np.log(p), p, .75)
        np.testing.assert_allclose(result.legal_probs, expected, rtol=1e-14, atol=0)
        self.assertEqual(result.human_rank, 6)

    def test_alignment_rejects_duplicate_or_mismatched_legal_sets(self):
        maia, allie = example_frames()
        allie.loc[0, "row_id"] = 1
        with self.assertRaises(ValueError):
            features.candidate_frame(maia, allie, "cross_model")

    def test_ensemble_search_matches_explicit_mixtures(self):
        maia, allie = example_frames()
        selected, rows = calibration.fit_ensembles(maia, allie)
        self.assertEqual(len(rows), 42)
        for row in rows:
            result = calibration.apply_ensemble(maia, allie, row["kind"], row["alpha"])
            nll = -np.log(np.maximum(result.p_human.to_numpy(), 1e-12)).mean()
            self.assertAlmostEqual(row["NLL"], nll, places=13)
        for kind, result in selected.items():
            self.assertEqual(result["NLL"], min(r["NLL"] for r in rows if r["kind"] == kind))


if __name__ == "__main__":
    unittest.main()
