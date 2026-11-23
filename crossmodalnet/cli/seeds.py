"""Repeat training with several seeds on the same split, then report mean and SD over seeds.

    python -m crossmodalnet.cli.seeds --split patient                    # print the commands only
    python -m crossmodalnet.cli.seeds --split patient --run              # train + evaluate each seed, then aggregate
    python -m crossmodalnet.cli.seeds --split patient --aggregate_only   # aggregate finished seeds

The seed controls the initialisation of the new layers (positional/modality embeddings, attention, capsule
head), dropout and the order of training batches. The split is fixed (see make_splits).
Finished seeds (an existing metrics.json) are skipped unless --force is given.
"""
import argparse
import subprocess
import sys

from .. import config as C
from ..evaluation.seeds import aggregate, run_name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", choices=C.SPLIT_MODES, required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--base", default="crossmodalnet", help="run-name prefix")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--aggregate_only", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dataset", choices=C.DATASETS, default=C.DATASET)
    a = ap.parse_args()
    C.use_dataset(a.dataset)
    seeds = [int(s) for s in a.seeds.split(",")]
    py = sys.executable
    if not a.aggregate_only:
        for s in seeds:
            name = run_name(a.base, s)
            cmd = (f"{py} -m crossmodalnet.cli.train --dataset {a.dataset} --split {a.split} --name {name} --seed {s} && "
                   f"{py} -m crossmodalnet.cli.evaluate --dataset {a.dataset} --split {a.split} --name {name}")
            done = (C.RESULT_DIR / a.split / name / "metrics.json").exists()
            print(("# done, skipped: " if done and not a.force else "") + cmd)
            if a.run and (not done or a.force):
                subprocess.run(cmd, shell=True, check=True)
    if a.run or a.aggregate_only:
        res = aggregate(a.split, a.base, seeds)
        for k, v in res["overall"].items():
            print(f"{k:16s} {v['mean']:.4f} +/- {v['sd']:.4f}")


if __name__ == "__main__":
    main()
