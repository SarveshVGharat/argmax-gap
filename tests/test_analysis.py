import tempfile
import unittest
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from argmax_gap.analysis import load_predictions, check_alignment


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
