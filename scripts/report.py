#!/usr/bin/env python3
"""Recompute tables and plots from model outputs; no test-set parameter fitting."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from argmax_gap.analysis import report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--positions', type=Path, required=True)
    parser.add_argument('--maia', type=Path, required=True)
    parser.add_argument('--allie', type=Path, required=True)
    parser.add_argument('--methods', type=Path, help='Output directory from train_methods.py')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-rows', type=int, default=884049)
    parser.add_argument('--bootstrap-reps', type=int, default=10000, help='Game bootstrap repetitions; 0 skips game intervals')
    args = parser.parse_args()
    report(args.positions, args.maia, args.allie, args.output, args.methods, args.expected_rows, args.bootstrap_reps)
    print(f'Report written to {args.output}')


if __name__ == '__main__':
    main()
