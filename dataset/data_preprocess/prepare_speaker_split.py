"""Create a real-identity mapping template or prepare an immutable FlowREACT split."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from framework.utils.speaker_split import identity_template, load_split_bundle, prepare_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    template = commands.add_parser("template")
    template.add_argument("--data_dir", required=True)
    template.add_argument("--source_splits", nargs="+", default=["train", "val"])
    template.add_argument("--output", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--data_dir", required=True)
    prepare.add_argument("--identities", required=True)
    prepare.add_argument("--output_dir", required=True)
    prepare.add_argument("--source_splits", nargs="+", default=["train", "val"])
    prepare.add_argument("--seed", type=int, default=1234)
    prepare.add_argument("--unseen_fraction", type=float, default=0.2)
    prepare.add_argument("--val_fraction", type=float, default=0.15)
    prepare.add_argument("--seen_fraction", type=float, default=0.15)
    prepare.add_argument("--unseen_speakers", nargs="+")
    audit = commands.add_parser("audit")
    audit.add_argument("--split_dir", required=True)
    args = parser.parse_args()
    if args.command == "template":
        count = identity_template(args.data_dir, args.output, args.source_splits)
        print("Created {} recording rows at {}. Fill both IDs using actual participant metadata.".format(count, args.output))
    elif args.command == "prepare":
        result = prepare_split(
            args.data_dir, args.identities, args.output_dir, args.source_splits,
            args.seed, args.unseen_fraction, args.val_fraction, args.seen_fraction,
            args.unseen_speakers,
        )
        print(json.dumps(result, indent=2))
    else:
        _, result = load_split_bundle(args.split_dir)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
