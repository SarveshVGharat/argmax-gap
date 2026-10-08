#!/usr/bin/env python3
"""Download the pinned public policies and Allie test games."""
import argparse
from pathlib import Path

from argmax_gap.upstream import fetch_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--heldout-file", type=Path, help="Exact heldout.parquet companion release asset")
    args = parser.parse_args()
    fetch_assets(args.output_dir, args.heldout_file)


if __name__ == "__main__":
    main()
