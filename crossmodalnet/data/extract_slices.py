"""Cut BraTS 2020 volumes into the 2D slice triplets stored in Data/data.

Per patient and axial slice k (same k for every sequence):
  * skip a slice whose T1, T2 and FLAIR each have fewer than MIN_NONZERO non-zero pixels (empty slice)
  * tumor pixels = number of voxels with segmentation label > 0
      0 tumor pixels          -> class "No Tumor"          (tumor-free slice of a glioma patient)
      > TUMOR_THRESHOLD       -> class = patient grade, HGG or LGG, from name_mapping.csv
      1 .. TUMOR_THRESHOLD    -> discarded (too small to label reliably)
  * each sequence: per-slice z-score, resize to 224 x 224, min-max to uint8, saved as PNG

Volumes may be .nii, .nii.gz or a .zip holding one .nii, in or below the source folder.
The segmentation file is <patient>_seg.nii[.gz|.zip]; a differently named file containing 'seg'
(BraTS 2020 names the segmentation of patient 355 differently) is accepted if it is the only match.
"""
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .. import config as C

TUMOR_THRESHOLD = 100
MIN_NONZERO = 6000
_KEYS = {"T1": "t1", "T2": "t2", "FLAIR": "flair"}


def _candidates(root: Path, patient: str, key: str):
    return sorted(p for p in root.rglob(f"{patient}_{key}.nii*") if p.is_file())


def find_seg(root: Path, patient: str):
    """Segmentation file of a patient: <patient>_seg.nii[.gz|.zip] or, as for BraTS20_Training_355,
    <patient>_Segm.nii[.zip] (any case of 'seg'); falls back to the only 'seg' file in the patient's folder."""
    root = Path(root)
    hits = sorted(p for p in root.rglob(f"{patient}_*") if p.is_file()
                  and p.name[len(patient) + 1:].lower().startswith("seg") and ".nii" in p.name.lower())
    if hits:
        return hits[0]
    folder = root / patient
    alt = [p for p in folder.rglob("*") if p.is_file() and "seg" in p.name.lower() and ".nii" in p.name.lower()] \
        if folder.is_dir() else []
    return alt[0] if len(alt) == 1 else None


def find_volume(root: Path, patient: str, key: str):
    if key == "seg":
        return find_seg(root, patient)
    hits = _candidates(root, patient, key)
    return hits[0] if hits else None


def load_volume(path: Path) -> np.ndarray:
    import nibabel as nib
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z, tempfile.TemporaryDirectory() as tmp:
            member = next(n for n in z.namelist() if ".nii" in n.lower())
            return np.asarray(nib.load(z.extract(member, tmp)).get_fdata(), dtype=np.float32)
    return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)


def _slice_to_png(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    sd = img.std()
    if sd > 0:
        img = (img - img.mean()) / sd
    img = cv2.resize(img, (C.IMAGE_SIZE, C.IMAGE_SIZE))
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def extract_patient(patient: str, grade: str, source: Path, out_dir: Path = None):
    """Returns (counts per class, reason or None). reason is set when the patient was skipped.
    out_dir defaults to the active dataset's slice folder."""
    out_dir = Path(out_dir) if out_dir else C.DATA_DIR
    paths = {s: find_volume(source, patient, k) for s, k in _KEYS.items()}
    seg_path = find_volume(source, patient, "seg")
    missing = [n for n, p in {**paths, "seg": seg_path}.items() if p is None]
    if missing:
        return None, f"missing volume(s): {', '.join(missing)}"
    vols = {s: load_volume(p) for s, p in paths.items()}
    seg = load_volume(seg_path)
    if len({v.shape for v in [*vols.values(), seg]}) != 1:
        return None, "volumes have different shapes"
    counts = {c: 0 for c in C.CLASSES}
    for k in range(seg.shape[2]):
        sl = {s: v[:, :, k] for s, v in vols.items()}
        if all(np.count_nonzero(a) < MIN_NONZERO for a in sl.values()):
            continue
        tumor = int((seg[:, :, k] > 0).sum())
        if tumor == 0:
            cls = "No Tumor"
        elif tumor > TUMOR_THRESHOLD:
            cls = grade
        else:
            continue
        for s, a in sl.items():
            path = out_dir / cls / s / f"{patient}_{k}_{s}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), _slice_to_png(a))
        counts[cls] += 1
    return counts, None


def extract_all(source: Path, patients=None, out_dir: Path = None, overwrite: bool = False) -> pd.DataFrame:
    """Extract the listed patients (default: every patient of the grade table with no slices yet)."""
    from .index import read_grades
    out_dir = Path(out_dir) if out_dir else C.DATA_DIR
    grades = read_grades()
    done = {p.name.rsplit("_", 2)[0] for cls in ("HGG", "LGG") for p in (out_dir / cls / "T1").glob("*.png")}
    todo = list(patients) if patients else [p for p in grades if overwrite or p not in done]
    rows = []
    for p in todo:
        counts, reason = extract_patient(p, grades[p], Path(source), out_dir)
        rows.append(dict(patient_id=p, grade=grades[p], status="skipped" if reason else "extracted",
                         reason=reason, **(counts or {})))
    return pd.DataFrame(rows)
