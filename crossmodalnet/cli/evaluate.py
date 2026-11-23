"""Evaluate a trained run on the validation and test partitions.

    python -m crossmodalnet.cli.evaluate --split patient
    python -m crossmodalnet.cli.evaluate --split patient --drop_seq 2     # missing-sequence test
"""
import argparse

from .. import config as C
from ..evaluation.report import evaluate_run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", choices=C.SPLIT_MODES, required=True)
    ap.add_argument("--name", default="crossmodalnet")
    ap.add_argument("--drop_seq", default="", help="comma list of sequence indices zeroed at test time")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dataset", choices=C.DATASETS, default=C.DATASET)
    a = ap.parse_args()
    C.use_dataset(a.dataset)
    drop = tuple(int(i) for i in a.drop_seq.split(",") if i != "")
    t = evaluate_run(a.split, a.name, drop, a.n_boot, a.workers)["test"]
    print(f"test: accuracy {t['accuracy']:.4f}  macro-F1 {t['macro_f1']:.4f}  "
          f"macro-AUC {t['macro_auc_ovr']:.4f}  MCC {t['mcc']:.4f}")


if __name__ == "__main__":
    main()
