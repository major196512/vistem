import os
import argparse

__all__ = ["default_argument_parser"]

def default_argument_parser():
    parser = argparse.ArgumentParser(description="Distributed Setting")
    parser.add_argument("--config-file", type=str)

    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument("--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)")

    parser.add_argument("--dist-ip", default=None)
    parser.add_argument("--dist-port", default=None)

    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--resume", action="store_true", help="resume training")
    return parser
