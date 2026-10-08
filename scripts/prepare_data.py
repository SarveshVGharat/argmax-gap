#!/usr/bin/env python3
"""Create aligned position parquet files for either frozen policy."""
import argparse
import json
from pathlib import Path

from argmax_gap.data import prepare_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-format", choices=["allie-jsonl", "heldout-parquet"], required=True)
    parser.add_argument("--max-rows", type=int, help="First N retained rows, for smoke testing only")
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--allow-custom-source", action="store_true", help="Disable paper source hash check for custom datasets")
    args = parser.parse_args()
    print(json.dumps(prepare_data(args.input, args.output, args.source_format, args.max_rows,
                                  args.expected_rows, not args.allow_custom_source), indent=2))


if __name__ == "__main__":
    main()
