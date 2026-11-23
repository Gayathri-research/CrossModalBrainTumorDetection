"""Run Grad-CAM / Eigen-CAM on test slices and (optionally) score them against annotations."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .. import config as C
from ..data import load_split, normalize_batch
from ..evaluation.report import load_run
from .cam import attention_summary, eigen_cam, grad_cam
from .localization import brain_mask, score_slice, summarise
from ..data.extract_slices import find_seg
from .masks import MaskReader


def _select(test: pd.DataFrame, per_class: int, failures_per_class: int, seed: int):
    """Small, reproducible real subset: correctly classified slices plus failures, per class."""
    picks = []
    for c in range(C.N_CLASSES):
        t = test[test.label == c]
        ok, bad = t[t.pred == c], t[t.pred != c]
        picks += [(r, "correct") for _, r in ok.sample(min(per_class, len(ok)), random_state=seed).iterrows()]
        picks += [(r, "failure") for _, r in bad.sample(min(failures_per_class, len(bad)), random_state=seed).iterrows()]
    return picks


def _figure(path: Path, imgs, cams, eig, mask, title: str) -> None:
    """Heat maps are shown inside the head only (the raw maps are strong on the zero background)."""
    brain = brain_mask(imgs)
    fig, axs = plt.subplots(len(C.SEQS), 3, figsize=(7.5, 2.5 * len(C.SEQS)))
    for s, name in enumerate(C.SEQS):
        for k, (m, t) in enumerate([(None, "image"), (cams[s], "Grad-CAM"), (eig[s], "Eigen-CAM")]):
            ax = axs[s, k]
            ax.imshow(imgs[s], cmap="gray")
            if m is not None:
                lo, hi = m[brain].min(), m[brain].max()               # rescale within the head
                ax.imshow(np.ma.masked_where(~brain, (m - lo) / (hi - lo + 1e-8)), cmap="jet", alpha=0.4, vmin=0, vmax=1)
            if mask is not None and mask.any():
                ax.contour(mask, levels=[0.5], colors="w", linewidths=0.8)
            ax.set_title(f"{name} {t}", fontsize=8); ax.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def run_annotation_eval(split: str, name: str, seg_root, slices_per_patient: int = 10, region: str = "whole",
                        target: str = "pred", seed: int = 0, make_figures: bool = False) -> dict:
    """Score Grad-CAM / Eigen-CAM against the tumor annotation for every patient found in seg_root
    (any partition of the split). Results are reported per partition because only validation and test
    patients are unseen in training (validation patients also chose the best epoch):
      train -> seen in training; val -> unseen in training, used for model selection; test -> unseen.
    Only tumor-bearing slices (HGG or LGG label, so the annotation is non-empty) are scored, up to
    slices_per_patient per patient."""
    device = torch.device("cuda")
    model, _ = load_run(split, name, device)
    df = load_split(split)
    root = Path(seg_root)
    avail = [p for p in df.patient_id.unique() if find_seg(root, p) is not None]
    if not avail:
        raise FileNotFoundError(f"no patient of the {split}-wise manifest has a segmentation file under {root}")
    out = C.RESULT_DIR / split / name / "annotation_eval"
    out.mkdir(parents=True, exist_ok=True)
    arrays = [np.load(C.CACHE_DIR / f"{s}.npy", mmap_mode="r") for s in C.SEQS]
    reader = MaskReader(root, region)
    rng = np.random.default_rng(seed)
    rows = []
    for p in sorted(avail):
        d = df[(df.patient_id == p) & (df.label != C.NO_TUMOR)]
        if d.empty:
            continue
        for pos in d.sample(min(slices_per_patient, len(d)), random_state=seed).index:
            r = df.loc[pos]
            mask = reader(p, int(r.slice_idx))
            if not mask.any():
                continue
            x = normalize_batch(torch.from_numpy(np.stack([a[pos] for a in arrays]))[None].to(device))
            with torch.no_grad():
                pred = int(model(x).argmax(1).item())
            cls = pred if target == "pred" else int(r.label)
            cams, _, aux = grad_cam(model, x, torch.tensor([cls], device=device))
            eig = eigen_cam(aux["feats"])
            cams_np, eig_np = cams[0].cpu().numpy(), eig[0].cpu().numpy()
            maps = {f"gradcam_{s}": cams_np[i] for i, s in enumerate(C.SEQS)}
            maps.update({f"eigencam_{s}": eig_np[i] for i, s in enumerate(C.SEQS)})
            maps["gradcam_mean"] = cams_np.mean(0)
            rows.append(dict(sample_id=r.sample_id, patient_id=p, partition=r.split, grade=r.grade,
                             label=C.CLASSES[int(r.label)], pred=C.CLASSES[pred], correct=bool(pred == r.label),
                             **score_slice(maps, mask, rng, brain_mask([a[pos] for a in arrays]))))
            if make_figures:
                (out / "figures").mkdir(exist_ok=True)
                _figure(out / "figures" / f"{r.sample_id}.png", [a[pos] for a in arrays], cams_np, eig_np, mask,
                        f"{r.sample_id} [{r.split}] true={rows[-1]['label']} pred={rows[-1]['pred']}")
    table = pd.DataFrame(rows)
    table.to_csv(out / "annotation_scores.csv", index=False)
    meaning = {"train": "seen in training", "val": "unseen in training; used to pick the best epoch",
               "test": "unseen (held-out test)"}
    summary = {"cam_target": target, "mask_region": region, "slices_per_patient_max": slices_per_patient,
               "patients_found": {s: int(df[df.patient_id.isin(avail)].groupby("split").patient_id.nunique().get(s, 0))
                                  for s in ("train", "val", "test")},
               "by_partition": {}}
    for part in ("test", "val", "train"):
        sub = table[table.partition == part]
        if len(sub):
            entry = {"meaning": meaning[part], "all_slices": summarise(sub)}
            if sub.correct.any():
                entry["correctly_classified_only"] = summarise(sub[sub.correct])
            summary["by_partition"][part] = entry
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_explain(split: str, name: str, n_per_class: int, n_failures: int, seg_root=None, region: str = "whole",
                target: str = "pred", seed: int = 0, make_figures: bool = True) -> dict:
    device = torch.device("cuda")
    model, _ = load_run(split, name, device)
    df = load_split(split)
    out = C.RESULT_DIR / split / name / "explain"
    for sub in ("correct", "failure"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    preds = pd.read_csv(C.RESULT_DIR / split / name / "predictions.csv").set_index("sample_id")["pred"]
    test = df[df.split == "test"].copy()
    test["pred"] = test.sample_id.map(preds)
    arrays = [np.load(C.CACHE_DIR / f"{s}.npy", mmap_mode="r") for s in C.SEQS]
    reader = MaskReader(seg_root, region) if seg_root else None
    rng = np.random.default_rng(seed)

    rows, attn_rows = [], []
    for r, kind in sorted(_select(test, n_per_class, n_failures, seed), key=lambda t: t[0].patient_id):
        pos = int(df.index[df.sample_id == r.sample_id][0])
        raw = torch.from_numpy(np.stack([a[pos] for a in arrays]))[None]
        x = normalize_batch(raw.to(device))
        cls = int(r.pred if target == "pred" else r.label)
        cams, _, aux = grad_cam(model, x, torch.tensor([cls], device=device))
        eig = eigen_cam(aux["feats"])
        cams_np, eig_np = cams[0].cpu().numpy(), eig[0].cpu().numpy()
        info = dict(sample_id=r.sample_id, patient_id=r.patient_id, label=C.CLASSES[int(r.label)],
                    pred=C.CLASSES[int(r.pred)], correct=bool(r.pred == r.label), kind=kind)
        att = attention_summary(aux["attn"], C.N_SEQS)
        if att is not None:
            for a in range(C.N_SEQS):
                for b in range(C.N_SEQS):
                    if a != b:
                        info[f"attn_{C.SEQS[a]}_to_{C.SEQS[b]}"] = float(att[0, a, b])
        mask = None
        if reader is not None:
            mask = reader(r.patient_id, int(r.slice_idx))
            if mask.any():                           # slices with a tumor annotation only
                maps = {f"gradcam_{s}": cams_np[i] for i, s in enumerate(C.SEQS)}
                maps.update({f"eigencam_{s}": eig_np[i] for i, s in enumerate(C.SEQS)})
                maps["gradcam_mean"] = cams_np.mean(0)
                info.update(score_slice(maps, mask, rng, brain_mask([a[pos] for a in arrays])))
        rows.append(info)
        if make_figures:
            _figure(out / kind / f"{r.sample_id}.png", [a[pos] for a in arrays], cams_np, eig_np, mask,
                    f"{r.sample_id}  true={info['label']}  pred={info['pred']}")

    table = pd.DataFrame(rows)
    table.to_csv(out / "explain_scores.csv", index=False)
    summary = {"n_slices_explained": int(len(table)), "cam_target": target, "mask_region": region if seg_root else None}
    scored = table[[c for c in table.columns if "/" in c]].dropna(how="all").index if seg_root else []
    if seg_root and len(scored):
        sc = table.loc[scored]
        summary["all_annotated_slices"] = summarise(sc)
        if sc.correct.any():
            summary["correctly_classified_only"] = summarise(sc[sc.correct])
    else:
        summary["note"] = "no annotations supplied or no annotated slice selected: figures and attention only"
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
