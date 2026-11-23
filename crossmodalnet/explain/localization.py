"""Quantitative comparison of a saliency map with the tumor annotation.

For a map m in [0, 1] and a binary mask g (both H x W):
  iou@t / dice@t   overlap of {m >= t} with g, for each threshold t
  auroc            threshold-free: how well m ranks tumor pixels above non-tumor pixels
  energy_in_mask   sum of m inside g / sum of m
  enrichment       energy_in_mask / area fraction of g  (1 = no better than uniform, >1 = concentrated in tumor)
  pointing_hit     the arg-max pixel of m lies inside g
Controls give a reference on the same slices: a uniform-random map and a centred Gaussian
(centre-bias) map. A CAM is informative only to the extent it beats both.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

THRESHOLDS = (0.3, 0.5, 0.7)


def centre_prior(shape, sigma_frac: float = 0.25) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    g = np.exp(-(((yy - h / 2) / (sigma_frac * h)) ** 2 + ((xx - w / 2) / (sigma_frac * w)) ** 2) / 2)
    return g / g.max()


def brain_mask(imgs) -> np.ndarray:
    """Foreground (head) mask from the image stack (S, H, W): pixels above the background level in any
    sequence, holes filled. The background level is estimated per image from its border (contrast
    enhancement lifts the empty background to a small constant, not to exactly zero)."""
    from scipy.ndimage import binary_fill_holes
    fg = np.zeros(np.asarray(imgs[0]).shape, bool)
    for x in imgs:
        x = np.asarray(x).astype(np.int32)
        border = np.concatenate([x[0], x[-1], x[:, 0], x[:, -1]])
        fg |= x > np.median(border) + 3
    fg = binary_fill_holes(fg)
    return fg if fg.any() else np.ones(fg.shape, bool)


def score_map(m: np.ndarray, g: np.ndarray, thresholds=THRESHOLDS, valid: np.ndarray = None) -> dict:
    """m: float (H, W) in [0, 1]; g: bool (H, W) with at least one tumor pixel.
    valid: optional bool (H, W); when given, every metric uses only those pixels (e.g. inside the brain)."""
    if valid is not None:
        g = np.logical_and(g, valid)
        m = np.where(valid, m, -1.0)                 # outside the brain: below every threshold, never the arg-max
    else:
        valid = np.ones(g.shape, bool)
    mv = np.where(valid, m, 0.0)
    out = {}
    area = g.sum() / valid.sum()
    for t in thresholds:
        p = np.logical_and(m >= t, valid)
        inter = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        out[f"iou@{t}"] = inter / union if union else np.nan
        out[f"dice@{t}"] = 2 * inter / (p.sum() + g.sum()) if (p.sum() + g.sum()) else np.nan
    gv, mvv = g[valid], m[valid]
    out["auroc"] = roc_auc_score(gv, mvv) if 0 < gv.sum() < gv.size else np.nan
    energy = mv[g].sum() / (mv.sum() + 1e-8)
    out["energy_in_mask"] = float(energy)
    out["enrichment"] = float(energy / area) if area > 0 else np.nan
    out["pointing_hit"] = float(g[np.unravel_index(m.argmax(), m.shape)])
    return out


def score_slice(maps: dict, g: np.ndarray, rng: np.random.Generator, brain: np.ndarray = None) -> dict:
    """maps: name -> (H, W) map. Adds the two control maps. Returns a flat {name/metric: value} dict.
    With `brain`, the same metrics are also computed on brain pixels only, under keys 'name@brain/metric'
    (the controls are scored the same way)."""
    maps = dict(maps)
    maps["control_random"] = rng.random(g.shape)
    maps["control_centre"] = centre_prior(g.shape)
    row = {}
    for name, m in maps.items():
        for k, v in score_map(m, g).items():
            row[f"{name}/{k}"] = v
        if brain is not None:
            for k, v in score_map(m, g, valid=brain).items():
                row[f"{name}@brain/{k}"] = v
    row["mask_fraction"] = float(g.mean())
    if brain is not None:
        row["brain_fraction"] = float(brain.mean())
    return row


def summarise(rows: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> dict:
    """Mean of every column with a 95% CI from a patient-level bootstrap (needs a 'patient_id' column)."""
    cols = [c for c in rows.columns if "/" in c or c in ("mask_fraction", "brain_fraction")]
    rng = np.random.default_rng(seed)
    by_patient = {p: d[cols].mean() for p, d in rows.groupby("patient_id")}
    table = pd.DataFrame(by_patient).T                        # per-patient means -> clusters
    point = rows[cols].mean()
    boots = np.stack([table.values[rng.integers(0, len(table), len(table))].mean(0) for _ in range(n_boot)])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    return {"n_slices": int(len(rows)), "n_patients": int(len(table)),
            "metrics": {c: dict(mean=float(point[c]), ci95=[float(l), float(h)]) for c, l, h in zip(cols, lo, hi)}}
