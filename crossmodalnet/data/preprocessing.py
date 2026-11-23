"""Preprocessing: 1st-99th percentile clipping + CLAHE (cached as uint8), z-score at load time."""
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import torch

from .. import config as C


def preprocess_slice(img: np.ndarray) -> np.ndarray:
    """uint8 slice -> uint8 slice: percentile clipping, rescale to 0-255, CLAHE."""
    lo, hi = np.percentile(img, [1, 99])
    img = np.clip(img.astype(np.float32), lo, hi)
    img = ((img - lo) / max(hi - lo, 1.0) * 255.0).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)


def build_cache(df: pd.DataFrame, workers: int = 8) -> None:
    """Write Data/cache/<SEQ>.npy, uint8 (N, H, W) in index order. No-op if already built."""
    done = C.CACHE_DIR / "DONE"
    if done.exists():
        return
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pats, idxs, labs = df.patient_id.values, df.slice_idx.values, df.label.values
    shape = (len(df), C.IMAGE_SIZE, C.IMAGE_SIZE)
    for seq in C.SEQS:
        mm = np.lib.format.open_memmap(C.CACHE_DIR / f"{seq}.npy", mode="w+", dtype=np.uint8, shape=shape)

        def work(k):
            path = C.slice_path(C.CLASSES[labs[k]], seq, pats[k], idxs[k])
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None or img.shape != shape[1:]:
                raise RuntimeError(f"unreadable or wrongly sized image: {path}")
            mm[k] = preprocess_slice(img)

        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(work, range(len(df)), chunksize=256))
        mm.flush()
        del mm
    df.to_csv(C.CACHE_DIR / "index.csv", index=False)
    done.write_text("ok")


def normalize_batch(x: torch.Tensor) -> torch.Tensor:
    """uint8 (B, S, H, W) -> float32, per-image z-score."""
    x = x.float() / 255.0
    mu = x.mean(dim=(2, 3), keepdim=True)
    sd = x.std(dim=(2, 3), keepdim=True)
    return (x - mu) / (sd + 1e-6)
