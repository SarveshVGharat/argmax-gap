import json
import unittest

from argmax_gap.data import rows_for_game, validate_position


def game(durations=None):
    moves = "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 d7d6"
    return {"game-id": "fixture", "moves-uci": moves, "moves-seconds": durations or [0] * 14,
            "time-control": "30+0", "white-elo": 1800, "black-elo": 1900}


class DataTests(unittest.TestCase):
    def test_first_target_and_clock_boundary(self):
        rows = list(rows_for_game(game()))
        self.assertEqual([row["ply_index"] for row in rows], [10, 11, 12, 13])
        self.assertEqual(rows[0]["clock_before_seconds"], 30)
        self.assertEqual(len(json.loads(rows[0]["past_fens"])), 7)
        self.assertEqual(len(rows[0]["previous_moves_uci"].split()), 10)
        for row in rows:
            validate_position(row, replay_prefix=True)

    def test_low_clock_truncates_all_later_targets(self):
        durations = [0] * 14
        durations[10] = 1
        rows = list(rows_for_game(game(durations)))
        # White falls below 30 after ply 11. Black's next target is retained,
        # then the whole game ends at white's next pre-move clock check.
        self.assertEqual([row["ply_index"] for row in rows], [10, 11])

    def test_long_target_duration_is_allowed(self):
        durations = [0] * 14
        durations[10] = 31
        rows = list(rows_for_game(game(durations)))
        self.assertEqual(rows[0]["time_spent_seconds"], 31)

    def test_target_duration_is_absent_from_prefix(self):
        first = list(rows_for_game(game()))[0]
        durations = [0] * 14
        durations[10] = 15
        changed = list(rows_for_game(game(durations)))[0]
        for field in ("fen_before", "past_fens", "previous_moves_uci", "previous_move_seconds"):
            self.assertEqual(first[field], changed[field])

    def test_source_clock_correction_is_preserved(self):
        durations = [0] * 14
        durations[10] = -1
        rows = list(rows_for_game(game(durations)))
        self.assertEqual(rows[0]["time_spent_seconds"], -1)
        self.assertEqual(rows[2]["clock_before_seconds"], 31)

    def test_zero_base_control_is_truncated(self):
        source = game()
        source["time-control"] = "0+5"
        self.assertEqual(list(rows_for_game(source)), [])

    def test_bad_move_time_alignment_fails(self):
        source = game()
        source["moves-seconds"].pop()
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            list(rows_for_game(source))

    def test_incomplete_legal_set_fails(self):
        row = list(rows_for_game(game()))[0]
        row["legal_moves_uci"] = json.dumps([row["human_move_uci"]])
        with self.assertRaisesRegex(ValueError, "Legal-move mismatch"):
            validate_position(row)
