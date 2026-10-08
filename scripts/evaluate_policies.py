#!/usr/bin/env python3
"""Evaluate one frozen policy on aligned position rows."""
import argparse
import json
from pathlib import Path

from argmax_gap.inference import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["maia3", "allie"], required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--precision", choices=["float32", "float16", "bfloat16"],
                        help="Default: MAIA3 CUDA float16; Allie and CPU float32")
    parser.add_argument("--max-rows", type=int, help="First N rows, for smoke tests only")
    parser.add_argument("--threads", type=int, default=4, help="CPU thread count")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    print(json.dumps(evaluate(args.model, args.input, args.output, args.upstream_root,
                              args.checkpoint, args.device, args.batch_size, args.precision,
                              args.max_rows), indent=2))


if __name__ == "__main__":
    main()
