import unittest
import numpy as np
from argmax_gap.metrics import strict_rank, ece, paired_top1, paired_poisson_cis, clustered_ci, comparison_seed


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

    def test_clustered_interval_preserves_first_encounter_order(self):
        rng = np.random.default_rng(123)
        draws = rng.integers(0, 3, size=(80, 3))
        sums = np.array([3., -1., 2.])  # z, a, m: first encounter, not alphabetic order
        counts = np.array([2, 1, 1])
        expected = np.quantile(sums[draws].sum(axis=1) / counts[draws].sum(axis=1), [.025, .975])
        result = clustered_ci([1., -1., 2., 2.], ['z', 'a', 'z', 'm'], reps=80, seed=123)
        np.testing.assert_array_equal([result['game_ci_low'], result['game_ci_high']], expected)

    def test_poisson_intervals_share_position_weights(self):
        values = np.array([-1., 2., 0., 3.])
        result = paired_poisson_cis({'forward': values, 'reverse': -values}, reps=40, seed=31)
        rng = np.random.default_rng(31)
        samples = []
        for _ in range(40):
            weights = rng.poisson(1., size=len(values))
            samples.append(np.dot(weights, values) / max(weights.sum(), 1))
        expected = np.quantile(samples, [.025, .975])
        np.testing.assert_array_equal([result['forward']['ci_low'], result['forward']['ci_high']], expected)
        self.assertAlmostEqual(result['reverse']['ci_low'], -result['forward']['ci_high'])
        self.assertAlmostEqual(result['reverse']['ci_high'], -result['forward']['ci_low'])

    def test_shared_top1_rng_and_paper_comparison_seed(self):
        base, alternative = [True, False, False], [False, True, True]
        rng = np.random.default_rng(42)
        first = paired_top1(base, alternative, reps=20, rng=rng)
        second = paired_top1(base, alternative, reps=20, rng=rng)
        rng = np.random.default_rng(42)
        draws = rng.multinomial(3, [2 / 3, 1 / 3, 0], size=40)
        for actual, sample in zip((first, second), (draws[:20], draws[20:])):
            np.testing.assert_array_equal([actual['ci_low_pp'], actual['ci_high_pp']],
                                          np.quantile(100 * (sample[:, 0] - sample[:, 1]) / 3, [.025, .975]))
        self.assertEqual(comparison_seed('MAIA3 calibrated vs MAIA3 base', 'MRR'), 704719658)
