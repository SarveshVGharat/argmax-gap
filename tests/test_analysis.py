import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from argmax_gap.analysis import load_predictions, check_alignment, strata, report


class AnalysisTests(unittest.TestCase):
    def test_ragged_probabilities_and_ties(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'predictions.parquet'
            rows = [dict(row_id=0, human_move_uci='a', legal_moves_uci=['b', 'a', 'c'], legal_probs=[.4, .4, .2]),
                    dict(row_id=1, human_move_uci='x', legal_moves_uci=['x'], legal_probs=[1.])]
            pq.write_table(pa.Table.from_pylist(rows), path)
            frame = load_predictions(path, batch_size=1)
            self.assertEqual(frame.human_rank.tolist(), [1, 1])
            np.testing.assert_allclose(frame.margin, [0, 1])
            self.assertEqual(frame.top1_move_uci.tolist(), ['b', 'x'])
            with self.assertRaises(ValueError):
                check_alignment(frame, frame.iloc[::-1])

    def test_duplicate_move_and_bad_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'predictions.parquet'
            for moves, probabilities in [(['a', 'a'], [.5, .5]), (['a', 'b'], [.5, .4])]:
                pq.write_table(pa.Table.from_pylist([dict(row_id=0, human_move_uci='a',
                    legal_moves_uci=moves, legal_probs=probabilities)]), path)
                with self.assertRaises(ValueError):
                    load_predictions(path)

    def test_fixed_second_boundaries(self):
        times = [0, 2, 2.01, 4, 4.01, 6, 6.01, 8, 8.01, 10, 10.01, 15, 15.01, 30, 30.01]
        positions = pd.DataFrame({'time_spent_seconds': times, 'move_number': 20, 'player_elo': 1500})
        maia = pd.DataFrame({'entropy': np.arange(len(times)), 'num_legal_moves': 25, 'margin': .1})
        groups = strata(positions, maia)['fixed_seconds'].astype(str).tolist()
        self.assertEqual(groups, ['[0,2]', '[0,2]', '(2,4]', '(2,4]', '(4,6]', '(4,6]',
                                  '(6,8]', '(6,8]', '(8,10]', '(8,10]', '(10,15]', '(10,15]',
                                  '(15,30]', '(15,30]', '(30,inf)'])

    def test_report_rank_transitions_and_time_intervals(self):
        def predictions(ranks):
            result = []
            for index, rank in enumerate(ranks):
                moves = ['b', 'c', 'd', 'e', 'f']
                moves.insert(rank - 1, 'a')
                result.append({'row_id': index, 'human_move_uci': 'a', 'legal_moves_uci': moves,
                               'legal_probs': (np.arange(6, 0, -1) / 21).tolist()})
            return pa.Table.from_pylist(result)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            distribution = root / 'methods' / 'distributions'
            distribution.mkdir(parents=True)
            pq.write_table(predictions([6, 1, 2, 1]), root / 'base.parquet')
            pq.write_table(predictions([5, 2, 1, 6]), distribution / 'ensemble_convex.parquet')
            pq.write_table(pa.Table.from_pydict({'game_id': ['g0', 'g0', 'g1', 'g1'],
                'human_move_uci': ['a'] * 4, 'time_spent_seconds': [.5, 1., 8., 9.],
                'move_number': [5, 15, 30, 45], 'player_elo': [1500] * 4}), root / 'positions.parquet')
            output = root / 'report'
            report(root / 'positions.parquet', root / 'base.parquet', root / 'base.parquet', output,
                   methods=root / 'methods', expected_rows=4, bootstrap_reps=0)
            changes = pd.read_csv(output / 'rank_changes.csv').iloc[0]
            self.assertEqual(changes.top5_in, 1)
            self.assertEqual(changes.top5_out, 1)
            self.assertEqual(changes.top1_in, 1)
            self.assertEqual(changes.top1_out, 2)
            time = pd.read_csv(output / 'time_differences.csv')
            top1 = time.loc[(time.model == 'maia3') & (time.metric == 'Top1')].iloc[0]
            self.assertEqual(top1.fast_rows, 2)
            self.assertEqual(top1.slow_rows, 2)
            self.assertEqual(top1.fast_mean, 50)
            self.assertEqual(top1.slow_mean, 50)
            self.assertAlmostEqual(top1.ci_low, -1.959963984540054 * 50)
            self.assertAlmostEqual(top1.ci_high, 1.959963984540054 * 50)
            self.assertTrue((output / 'stratified_gap.pdf').is_file())
