#!/usr/bin/env python3
import argparse
import json

from lidarwm.config import load_config
from lidarwm.engine import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    print(json.dumps(evaluate(load_config(args.config, args.set), args.checkpoint, args.split), indent=2))


if __name__ == "__main__":
    main()

