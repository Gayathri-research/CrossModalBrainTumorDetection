"""Cut BraTS volumes into slice triplets under Data/data (see crossmodalnet/data/extract_slices.py).

    python -m crossmodalnet.cli.extract_slices --source Data/source
    python -m crossmodalnet.cli.extract_slices --source Data/source --patients BraTS20_Training_355

After adding patients, delete Data/cache and run make_splits again so the cache and the
split manifests include them.
"""
import argparse
from pathlib import Path

from .. import config as C
from ..data.extract_slices import extract_all


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--source", default=None, help="folder with the BraTS volumes (default: the dataset's source folder)")
    ap.add_argument("--patients", nargs="*", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dataset", choices=C.DATASETS, default=C.DATASET)
    a = ap.parse_args()
    C.use_dataset(a.dataset)
    report = extract_all(Path(a.source) if a.source else C.SOURCE_DIR, a.patients, overwrite=a.overwrite)
    print(report.to_string(index=False) if len(report) else "nothing to extract")
    if (report.status == "skipped").any():
        print("\nSome patients were skipped (see 'reason'); no slices were written for them.")


if __name__ == "__main__":
    main()
