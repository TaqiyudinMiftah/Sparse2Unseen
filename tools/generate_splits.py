#!/usr/bin/env python
from __future__ import annotations

import argparse

from sparse2unseen.data.manifest import read_manifest
from sparse2unseen.data.splits import make_random_split, save_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    samples = read_manifest(args.manifest)
    split = make_random_split(samples, args.fraction, args.seed)
    save_split(split, args.output)
    print(
        f"Wrote split: {split['n_labeled']}/{split['n_total']} labeled "
        f"({split['fraction']:.1%}), seed={split['seed']} -> {args.output}"
    )


if __name__ == "__main__":
    main()
