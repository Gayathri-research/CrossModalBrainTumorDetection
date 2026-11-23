"""Train CrossModalNet (or a baseline).

    python -m crossmodalnet.cli.train --split patient
    python -m crossmodalnet.cli.train --split image
"""
import argparse

from .. import config as C
from ..training import fit


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", choices=C.SPLIT_MODES, required=True)
    ap.add_argument("--model", default="crossmodalnet")
    ap.add_argument("--name", default=None, help="run name (default: model name)")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4, help="backbone learning rate")
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seqs", default=",".join(map(str, range(C.N_SEQS))), help="sequence indices to use")
    ap.add_argument("--no_attention", action="store_true")
    ap.add_argument("--head", choices=["capsule", "mlp"], default="capsule")
    ap.add_argument("--separate_backbones", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dataset", choices=C.DATASETS, default=C.DATASET)
    cfg = vars(ap.parse_args())
    C.use_dataset(cfg["dataset"])
    fit(cfg)


if __name__ == "__main__":
    main()
