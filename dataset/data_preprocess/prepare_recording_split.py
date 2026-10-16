"""Prepare or audit a recording-disjoint REACT2025 split; no identity CSV required."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from framework.utils.speaker_split import load_split_bundle, prepare_recording_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--data_dir", required=True)
    prepare.add_argument("--output_dir", required=True)
    prepare.add_argument("--source_splits", nargs="+", default=["train", "val"])
    prepare.add_argument("--seed", type=int, default=1234)
    prepare.add_argument("--test_fraction", type=float, default=0.2)
    prepare.add_argument("--val_fraction", type=float, default=0.15)
    audit = commands.add_parser("audit")
    audit.add_argument("--split_dir", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_recording_split(
            args.data_dir, args.output_dir, args.source_splits,
            args.seed, args.test_fraction, args.val_fraction,
        )
    else:
        _, result = load_split_bundle(args.split_dir)
        if result["protocol"] != "recording_disjoint":
            raise ValueError("Not a recording-disjoint split bundle")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
