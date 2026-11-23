"""Evaluate one trained run on validation and test partitions and write the result files."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve

from .. import config as C
from ..data import load_split, make_loader
from ..models import build_model
from .inference import predict
from .metrics import patient_bootstrap, patient_grade_metrics, summarize


def load_run(split: str, name: str, device):
    run = C.RUN_DIR / split / name
    cfg = json.loads((run / "config.json").read_text())
    model = build_model(cfg, pretrained=False).to(device).to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(run / "best.pt", map_location=device))
    return model.eval(), cfg


def evaluate_run(split: str, name: str, drop_seq=(), n_boot: int = 1000, workers: int = 4) -> dict:
    device = torch.device("cuda")
    model, cfg = load_run(split, name, device)
    df = load_split(split)
    tag = "" if not drop_seq else "_drop" + "".join(map(str, drop_seq))
    out_dir = C.RESULT_DIR / split / (name + tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"split_protocol": f"{split}-wise", "model": cfg["model"], "dropped_sequences": list(drop_seq)}
    for part in ["val", "test"]:
        scores, y, pos, _ = predict(model, make_loader(df, part, 32, False, workers), device, drop_seq)
        rows = df.iloc[pos]
        m = summarize(y, scores)
        m["n_slices"], m["n_patients"] = int(len(y)), int(rows.patient_id.nunique())
        if part == "test":
            patients = rows.patient_id.values
            m["ci95_patient_bootstrap"] = patient_bootstrap(y, scores, patients, n_boot)
            m["patient_level_grade"] = patient_grade_metrics(y, scores, patients)
            pred = pd.DataFrame(scores, columns=[f"score_{c}" for c in C.CLASSES])
            pred.insert(0, "sample_id", rows.sample_id.values)
            pred.insert(1, "patient_id", patients)
            pred["label"], pred["pred"] = y, scores.argmax(1)
            pred.to_csv(out_dir / "predictions.csv", index=False)
            _plot_confusion(out_dir / "confusion_matrix.png", m["confusion_matrix"])
            _plot_roc(out_dir / "roc.png", y, scores, m)
        result[part] = m
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


def _plot_confusion(path: Path, cm) -> None:
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(cm)), C.CLASSES); ax.set_yticks(range(len(cm)), C.CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def _plot_roc(path: Path, y, scores, m) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    for c, name in enumerate(C.CLASSES):
        fpr, tpr, _ = roc_curve((y == c).astype(int), scores[:, c])
        ax.plot(fpr, tpr, label=f"{name} (AUC {m['per_class'][name]['auc_ovr']:.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
