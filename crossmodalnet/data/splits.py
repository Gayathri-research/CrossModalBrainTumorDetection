"""Train/val/test partitions: patient-wise (no leakage) and image-wise (leaky reference)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .. import config as C


def make_split(df: pd.DataFrame, mode: str, seed: int = C.SPLIT_SEED,
               val_frac: float = C.VAL_FRAC, test_frac: float = C.TEST_FRAC) -> pd.DataFrame:
    """Return a copy of df with a 'split' column in {train, val, test}.

    patient: patients are stratified by grade and partitioned first; every slice and
             sequence of a patient follows its patient, including its tumor-free slices.
    image  : slices are partitioned at random, stratified by slice class. Slices of one
             patient end up in several partitions, so results are optimistic (reference only).
    """
    df = df.copy()
    rest = val_frac + test_frac
    if mode == "patient":
        pat = df.groupby("patient_id").grade.first().reset_index()
        tr, tmp = train_test_split(pat, test_size=rest, stratify=pat.grade, random_state=seed)
        va, te = train_test_split(tmp, test_size=test_frac / rest, stratify=tmp.grade, random_state=seed)
        part = {**{p: "train" for p in tr.patient_id}, **{p: "val" for p in va.patient_id},
                **{p: "test" for p in te.patient_id}}
        df["split"] = df.patient_id.map(part)
    elif mode == "image":
        idx = np.arange(len(df))
        tr, tmp = train_test_split(idx, test_size=rest, stratify=df.label, random_state=seed)
        va, te = train_test_split(tmp, test_size=test_frac / rest, stratify=df.label.values[tmp], random_state=seed)
        df["split"] = "train"
        df.loc[va, "split"] = "val"
        df.loc[te, "split"] = "test"
    else:
        raise ValueError(f"unknown split mode {mode!r}")
    return df


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Patients and slices per partition, and patients that also occur in another partition."""
    n_parts = df.groupby("patient_id").split.nunique()
    rows = []
    for s in ["train", "val", "test"]:
        d = df[df.split == s]
        grade = d.groupby("patient_id").grade.first()
        row = dict(split=s, patients=grade.size)
        row.update({f"patients_{g}": int((grade == g).sum()) for g in ("HGG", "LGG")})
        row["slices"] = len(d)
        row.update({f"slices_{c.replace(' ', '')}": int((d.label == i).sum()) for i, c in enumerate(C.CLASSES)})
        row["patients_in_other_partitions"] = int((n_parts.loc[d.patient_id.unique()] > 1).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def save_split(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Write manifests to splits/. Raises if a patient-wise split leaks."""
    C.SPLIT_DIR.mkdir(exist_ok=True)
    df.to_csv(C.SPLIT_DIR / f"{mode}_manifest.csv", index=False)
    if mode == "patient":
        if df.groupby("patient_id").split.nunique().max() != 1:
            raise RuntimeError("patient-wise split places a patient in more than one partition")
        (df.groupby("patient_id")
           .agg(grade=("grade", "first"), split=("split", "first"),
                **{f"n_{c.replace(' ', '')}": ("label", lambda s, i=i: int((s == i).sum()))
                   for i, c in enumerate(C.CLASSES)})
           .reset_index().to_csv(C.SPLIT_DIR / "patient_split_manifest.csv", index=False))
    summary = split_summary(df)
    summary.to_csv(C.SPLIT_DIR / f"{mode}_summary.csv", index=False)
    return summary


def load_split(mode: str) -> pd.DataFrame:
    return pd.read_csv(C.SPLIT_DIR / f"{mode}_manifest.csv")
