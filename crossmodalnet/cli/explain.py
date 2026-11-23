"""Grad-CAM / Eigen-CAM figures and, with --seg_root, comparison with the tumor annotation.

    python -m crossmodalnet.cli.explain --split patient --n_per_class 20
    python -m crossmodalnet.cli.explain --split patient --seg_root <BraTS 2020 training folder> --region whole

    python -m crossmodalnet.cli.explain --split patient --seg_root Data/subset --slices_per_patient 10
        -> annotation scores for every patient in Data/subset, reported per partition (train / val / test)

--seg_root must contain <patient>/<patient>_seg.nii (BraTS 2020). The first two forms need
results/<split>/<name>/predictions.csv (run evaluate first). Outputs go to
results/<split>/<name>/explain/: correct/ and failure/ figures, explain_scores.csv, summary.json
"""
import argparse

from .. import config as C
from ..explain.masks import REGIONS
from ..explain.runner import run_annotation_eval, run_explain


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", choices=C.SPLIT_MODES, required=True)
    ap.add_argument("--name", default="crossmodalnet")
    ap.add_argument("--n_per_class", type=int, default=20, help="correctly classified slices per class")
    ap.add_argument("--n_failures", type=int, default=10, help="misclassified slices per class")
    ap.add_argument("--seg_root", default=None)
    ap.add_argument("--region", choices=list(REGIONS), default="whole")
    ap.add_argument("--target", choices=["pred", "label"], default="pred", help="class whose score is explained")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_figures", action="store_true")
    ap.add_argument("--slices_per_patient", type=int, default=None,
                    help="annotation mode: score up to this many tumor slices of EVERY patient found in --seg_root,\n"
                         "reported separately for train / val / test patients (needs only --seg_root and a trained run)")
    ap.add_argument("--dataset", choices=C.DATASETS, default=C.DATASET)
    a = ap.parse_args()
    C.use_dataset(a.dataset)
    if a.dataset == "subset" and not a.seg_root:
        a.seg_root = str(C.SOURCE_DIR)             # the subset folder holds the segmentation files
    if a.slices_per_patient:
        if not a.seg_root:
            ap.error("--slices_per_patient needs --seg_root")
        run_annotation_eval(a.split, a.name, a.seg_root, a.slices_per_patient, a.region, a.target, a.seed,
                            not a.no_figures)
        return
    run_explain(a.split, a.name, a.n_per_class, a.n_failures, a.seg_root, a.region, a.target, a.seed,
                not a.no_figures)


if __name__ == "__main__":
    main()
