import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data import normalize_batch
from .metrics import summarize


@torch.no_grad()
def predict(model, loader: DataLoader, device, drop_seq=()):
    """Returns (scores (n, C), labels (n,), index positions (n,), mean loss).
    drop_seq: sequence indices replaced by zeros at test time (missing-sequence test)."""
    model.eval()
    scores, labels, positions, loss = [], [], [], 0.0
    for x, y, pos in loader:
        x = normalize_batch(x.to(device, non_blocking=True))
        for k in drop_seq:
            x[:, k] = 0
        y = y.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
        loss += model.loss(out.float(), y).item() * len(y)
        scores.append(model.scores(out.float()).cpu()); labels.append(y.cpu()); positions.append(pos)
    labels = torch.cat(labels).numpy()
    return torch.cat(scores).numpy(), labels, torch.cat(positions).numpy(), loss / len(labels)


def validation_metrics(model, loader, device) -> dict:
    s, y, _, loss = predict(model, loader, device)
    m = summarize(y, s)
    m["loss"] = loss
    return m
