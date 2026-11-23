"""Ablation and baseline commands. Provided as code; not executed by default.

    python -m crossmodalnet.cli.ablations --split patient          # print the commands
    python -m crossmodalnet.cli.ablations --split patient --run    # execute them one after another
"""
import argparse
import subprocess
import sys

from .. import config as C
from ..models import BASELINES

ABLATIONS = {
    **{f"only_{s}": f"--seqs {i}" for i, s in enumerate(C.SEQS)},
    "no_attention": "--no_attention",
    "mlp_head": "--head mlp",
    "no_attention_mlp_head": "--no_attention --head mlp",
    "separate_backbones": "--separate_backbones",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", choices=C.SPLIT_MODES, required=True)
    ap.add_argument("--seeds", default="0", help="comma list of seeds")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--dataset", choices=C.DATASETS, default=C.DATASET)
    a = ap.parse_args()
    py = sys.executable
    for seed in a.seeds.split(","):
        jobs = [(f"abl_{n}_s{seed}", flags) for n, flags in ABLATIONS.items()]
        jobs += [(f"base_{b}_s{seed}", f"--model {b}") for b in BASELINES]
        for name, flags in jobs:
            cmd = (f"{py} -m crossmodalnet.cli.train --dataset {a.dataset} --split {a.split} --name {name} "
                   f"--seed {seed} {flags} && "
                   f"{py} -m crossmodalnet.cli.evaluate --dataset {a.dataset} --split {a.split} --name {name}")
            print(cmd)
            if a.run:
                subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    main()
