#!/usr/bin/env python3
import argparse

from lidarwm.config import load_config
from lidarwm.engine import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the latent acoustic LiDAR world model")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    Trainer(load_config(args.config, args.set)).fit()


if __name__ == "__main__":
    main()

