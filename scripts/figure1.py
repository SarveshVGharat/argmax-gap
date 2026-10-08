#!/usr/bin/env python3
"""Render the paper's illustrative position, identified by its fixed row ID."""
import argparse
import json
from pathlib import Path
import chess
import chess.svg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/figure1.svg'))
    args = parser.parse_args()
    example = json.loads((Path(__file__).resolve().parents[1] / 'reference' / 'figure1.json').read_text())
    board = chess.Board(example['fen_before'])
    # Test row 482262: observed move, MAIA3 Top1, Allie Top1.
    moves = [(example['human_move_uci'], '#1f9d55cc'), (example['maia3_top1_move_uci'], '#2563ebcc'),
             (example['allie_top1_move_uci'], '#f97316cc')]
    arrows = []
    for uci, color in moves:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f'Illegal illustration move: {uci}')
        arrows.append(chess.svg.Arrow(move.from_square, move.to_square, color=color))
    svg = chess.svg.board(board, arrows=arrows, size=600)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg)
    print(args.output)


if __name__ == '__main__':
    main()
