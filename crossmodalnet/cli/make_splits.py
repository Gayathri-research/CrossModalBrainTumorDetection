"""Build the preprocessing cache and both split manifests.

    python -m crossmodalnet.cli.make_splits [--seed 42]
"""
import argparse

from .. import config as C
from ..data import load_index, make_split, save_split


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--seed", type=int, default=C.SPLIT_SEED)
    ap.add_argument("--dataset", choices=C.DATASETS, default=C.DATASET)
    a = ap.parse_args()
    C.use_dataset(a.dataset)
    df = load_index()
    print(f"{len(df)} slice triplets, {df.patient_id.nunique()} patients")
    for mode in C.SPLIT_MODES:
        print(f"\n[{mode}-wise split]\n{save_split(make_split(df, mode, a.seed), mode).to_string(index=False)}")


if __name__ == "__main__":
    main()
