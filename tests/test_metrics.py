import unittest
import numpy as np
from argmax_gap.metrics import strict_rank, ece, paired_top1, clustered_ci


class MetricsTests(unittest.TestCase):
    def test_tie_aware_rank(self):
        self.assertEqual(strict_rank([.4, .4, .2], 1), 1)
        self.assertEqual(strict_rank([.4, .4, .2], 2), 3)

    def test_invalid_probabilities(self):
        for p in ([0, 0], [np.nan, 1], [-1, 2]):
            with self.assertRaises(ValueError):
                strict_rank(p, 0)

    def test_ece_includes_one_and_bin_edges(self):
        self.assertAlmostEqual(ece(np.array([0., .1, 1.]), np.array([0, 1, 1])), .3)

    def test_rescue_break_identity(self):
        a = np.array([True, True, False, False, False])
        b = np.array([True, False, True, True, False])
        result = paired_top1(a, b)
        self.assertEqual((result['rescues'], result['breaks']), (2, 1))
        self.assertAlmostEqual(result['delta_pp'], 100 * (b.mean() - a.mean()))
        self.assertEqual(paired_top1(a, a)['mcnemar_p'], 1)

    def test_clustered_interval_weights_positions(self):
        result = clustered_ci([.5, .5, .5], ['a', 'a', 'b'], reps=100)
        self.assertEqual(result['game_ci_low'], .5)
        self.assertEqual(result['game_ci_high'], .5)
