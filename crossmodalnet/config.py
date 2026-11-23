"""Single place for paths, class/sequence names and protocol constants.

Two datasets share the same code and are kept in separate folders:
  full   Data/data, Data/source/name_mapping.csv       -> Data/cache, splits, runs, results
  subset Data/subset (volumes + selected_patients.csv) -> Data/subset_slices, Data/cache_subset,
         splits_subset, runs_subset, results_subset
Select with the --dataset option of every command, or the CMN_DATASET environment variable.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---- classes and sequences ----------------------------------------------------
CLASSES = ["HGG", "LGG", "No Tumor"]
N_CLASSES = len(CLASSES)
HGG, LGG, NO_TUMOR = (CLASSES.index(c) for c in ("HGG", "LGG", "No Tumor"))
SEQS = ["T1", "T2", "FLAIR"]
N_SEQS = len(SEQS)
IMAGE_SIZE = 224

# ---- protocol -----------------------------------------------------------------
SPLIT_MODES = ("patient", "image")
SPLIT_SEED = 42
VAL_FRAC = 0.15
TEST_FRAC = 0.15

DATASETS = ("full", "subset")
DATASET = "full"
# the paths below are (re)assigned by use_dataset()
SOURCE_DIR = DATA_DIR = GRADE_CSV = CACHE_DIR = SPLIT_DIR = RUN_DIR = RESULT_DIR = ROOT


def use_dataset(name: str) -> None:
    """Point every path at the 'full' or the 'subset' dataset."""
    global DATASET, SOURCE_DIR, DATA_DIR, GRADE_CSV, CACHE_DIR, SPLIT_DIR, RUN_DIR, RESULT_DIR
    if name not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}")
    DATASET = name
    d = ROOT / "Data"
    if name == "full":
        SOURCE_DIR, DATA_DIR, GRADE_CSV = d / "source", d / "data", d / "source" / "name_mapping.csv"
        CACHE_DIR, SPLIT_DIR, RUN_DIR, RESULT_DIR = d / "cache", ROOT / "splits", ROOT / "runs", ROOT / "results"
    else:
        SOURCE_DIR, DATA_DIR, GRADE_CSV = d / "subset", d / "subset_slices", d / "subset" / "selected_patients.csv"
        CACHE_DIR, SPLIT_DIR = d / "cache_subset", ROOT / "splits_subset"
        RUN_DIR, RESULT_DIR = ROOT / "runs_subset", ROOT / "results_subset"


use_dataset(os.environ.get("CMN_DATASET", "full"))


def slice_path(cls_name: str, seq: str, patient: str, slice_idx: int) -> Path:
    return DATA_DIR / cls_name / seq / f"{patient}_{slice_idx}_{seq}.png"
