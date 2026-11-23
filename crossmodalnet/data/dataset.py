import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .. import config as C


class TripletDataset(Dataset):
    """Item = (uint8 tensor (S, H, W) of T1/T2/FLAIR, label, row position in the index)."""

    def __init__(self, positions: np.ndarray, labels: np.ndarray):
        self.pos, self.labels, self._arr = np.asarray(positions), np.asarray(labels), None

    def __len__(self):
        return len(self.pos)

    def __getitem__(self, k):
        if self._arr is None:      # opened lazily so each dataloader worker owns its memmaps
            self._arr = [np.load(C.CACHE_DIR / f"{s}.npy", mmap_mode="r") for s in C.SEQS]
        p = self.pos[k]
        return torch.from_numpy(np.stack([a[p] for a in self._arr])), int(self.labels[k]), int(p)


def make_loader(df: pd.DataFrame, split: str, batch: int, shuffle: bool, workers: int = 4) -> DataLoader:
    """df must be the full split manifest (row order = cache order)."""
    pos = np.where(df.split.values == split)[0]
    return DataLoader(TripletDataset(pos, df.label.values[pos]), batch_size=batch, shuffle=shuffle,
                      num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
