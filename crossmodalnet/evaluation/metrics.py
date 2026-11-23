"""Classification metrics with explicit averaging, and patient-level (cluster) bootstrap CIs."""
import numpy as np
from sklearn.metrics import (confusion_matrix, matthews_corrcoef, precision_recall_fscore_support,
                             roc_auc_score)

from .. import config as C

LABELS = list(range(C.N_CLASSES))
SUMMARY_KEYS = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_auc_ovr", "mcc"]


def auc_one_vs_rest(y, s):
    out = []
    for c in LABELS:
        t = (y == c).astype(int)
        out.append(roc_auc_score(t, s[:, c]) if 0 < t.sum() < len(t) else np.nan)
    return np.array(out)


def summarize(y: np.ndarray, s: np.ndarray) -> dict:
    """y: (n,) labels; s: (n, n_classes) class scores.
    'macro_*' are unweighted means over classes; AUC is one-vs-rest; MCC is the multiclass form."""
    pred = s.argmax(1)
    p, r, f, support = precision_recall_fscore_support(y, pred, labels=LABELS, zero_division=0)
    auc = auc_one_vs_rest(y, s)
    return {
        "accuracy": float((pred == y).mean()),
        "macro_precision": float(p.mean()), "macro_recall": float(r.mean()), "macro_f1": float(f.mean()),
        "macro_auc_ovr": float(np.nanmean(auc)), "mcc": float(matthews_corrcoef(y, pred)),
        "per_class": {c: dict(precision=float(p[i]), recall=float(r[i]), f1=float(f[i]),
                              auc_ovr=float(auc[i]), support=int(support[i])) for i, c in enumerate(C.CLASSES)},
        "confusion_matrix": confusion_matrix(y, pred, labels=LABELS).tolist(),
    }


def patient_bootstrap(y, s, patients, n_boot: int = 1000, seed: int = 0) -> dict:
    """95% CI by resampling PATIENTS; all slices of a resampled patient are kept together."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(patients, return_inverse=True)
    groups = [np.where(inv == k)[0] for k in range(len(uniq))]
    draws = {k: [] for k in SUMMARY_KEYS}
    for _ in range(n_boot):
        idx = np.concatenate([groups[k] for k in rng.integers(0, len(uniq), len(uniq))])
        m = summarize(y[idx], s[idx])
        for k in SUMMARY_KEYS:
            draws[k].append(m[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in draws.items()}


def patient_grade_metrics(y, s, patients) -> dict:
    """Secondary patient-level HGG-vs-LGG result. Every patient also has tumor-free slices, so a
    patient cannot be assigned one of the three slice classes; instead, among the slices the model
    calls tumor, average P(HGG) / (P(HGG) + P(LGG)) per patient and compare with the patient grade."""
    truth, grade_score = [], []
    for p in np.unique(patients):
        m = patients == p
        ys, ss = y[m], s[m]
        grades = np.unique(ys[ys != C.NO_TUMOR])
        called_tumor = ss.argmax(1) != C.NO_TUMOR
        if len(grades) != 1 or not called_tumor.any():
            continue
        truth.append(int(grades[0] == C.LGG))
        a, b = ss[called_tumor, C.HGG], ss[called_tumor, C.LGG]
        grade_score.append(float((b / (a + b + 1e-8)).mean()))        # P(LGG)
    if not truth:
        return {}
    t, g = np.array(truth), np.array(grade_score)
    out = {"n_patients": len(t), "n_LGG": int(t.sum()), "n_HGG": int((1 - t).sum()),
           "accuracy": float(((g >= 0.5).astype(int) == t).mean())}
    if 0 < t.sum() < len(t):
        out["auc_lgg_vs_hgg"] = float(roc_auc_score(t, g))
    return out
