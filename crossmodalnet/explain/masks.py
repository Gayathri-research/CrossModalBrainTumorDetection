"""BraTS segmentation annotations aligned to the 224 x 224 slices.

Slices were produced as seg[:, :, k] followed by cv2.resize to 224 x 224 (same indexing and no
transposition as the images), so the mask is loaded and resized the same way (nearest neighbour).
BraTS 2020 labels: 1 necrotic/non-enhancing core, 2 edema, 4 enhancing tumor.
"""
from pathlib import Path

import cv2
import numpy as np

from .. import config as C
from ..data.extract_slices import find_seg, load_volume

REGIONS = {"whole": (1, 2, 4), "core": (1, 4), "enhancing": (4,)}


def find_seg_file(seg_root, patient: str) -> Path:
    hit = find_seg(Path(seg_root), patient)
    if hit is None:
        raise FileNotFoundError(f"no segmentation file for {patient} under {seg_root}")
    return hit


class MaskReader:
    """Caches one volume at a time (slices of a patient are usually requested together)."""

    def __init__(self, seg_root, region: str = "whole"):
        self.root, self.labels = seg_root, REGIONS[region]
        self._patient, self._vol = None, None

    def __call__(self, patient: str, slice_idx: int) -> np.ndarray:
        if patient != self._patient:            # handles .nii, .nii.gz and zipped files
            self._vol = load_volume(find_seg_file(self.root, patient))
            self._patient = patient
        m = np.isin(self._vol[:, :, slice_idx], self.labels).astype(np.uint8)
        return cv2.resize(m, (C.IMAGE_SIZE, C.IMAGE_SIZE), interpolation=cv2.INTER_NEAREST).astype(bool)
