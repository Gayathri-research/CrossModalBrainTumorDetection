"""Slice index: one row per (patient, slice) triplet found on disk."""
import os
import re

import pandas as pd

from .. import config as C
from .preprocessing import build_cache

_NAME = re.compile(r"^(?P<patient>.+)_(?P<slice>\d+)_T1\.png$")


def read_grades() -> dict:
    """patient -> 'HGG'/'LGG' from name_mapping.csv (BraTS) or selected_patients.csv (patient_id, grade)."""
    t = pd.read_csv(C.GRADE_CSV)
    if "BraTS_2020_subject_ID" in t.columns:
        return dict(zip(t["BraTS_2020_subject_ID"], t["Grade"]))
    return dict(zip(t["patient_id"], t["grade"]))


def build_index() -> pd.DataFrame:
    """Scan the slice folder. Columns: sample_id, patient_id, slice_idx, label, grade."""
    grade = read_grades()
    rows = []
    for label, cls in enumerate(C.CLASSES):
        for fname in sorted(os.listdir(C.DATA_DIR / cls / C.SEQS[0])):
            m = _NAME.match(fname)
            if m is None:
                raise ValueError(f"unexpected file name: {fname}")
            p, i = m["patient"], int(m["slice"])
            if not all(C.slice_path(cls, s, p, i).exists() for s in C.SEQS):
                raise FileNotFoundError(f"incomplete triplet for {p} slice {i}")
            rows.append(dict(sample_id=f"{p}_{i}", patient_id=p, slice_idx=i, label=label, grade=grade[p]))
    df = pd.DataFrame(rows)
    if not df.sample_id.is_unique:
        raise ValueError("a slice appears in more than one class folder")
    tumor = df[df.label != C.NO_TUMOR]
    if not (tumor.grade == tumor.label.map(dict(enumerate(C.CLASSES)))).all():
        raise ValueError("tumor-class folders disagree with the patient grade in name_mapping.csv")
    return df.reset_index(drop=True)


def load_index() -> pd.DataFrame:
    """Index + preprocessing cache (built on first use)."""
    df = build_index()
    build_cache(df)
    cached = pd.read_csv(C.CACHE_DIR / "index.csv")
    if not (cached.sample_id.values == df.sample_id.values).all():
        raise RuntimeError("cache does not match Data/data; delete Data/cache and rebuild")
    return df
