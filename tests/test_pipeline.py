"""Code checks on SYNTHETIC fixtures. They verify behaviour, not performance: no result is produced.

    python -m pytest tests -q          (or)          python tests/test_pipeline.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossmodalnet import config as C
from crossmodalnet.data import make_split, normalize_batch, preprocess_slice, split_summary
from crossmodalnet.evaluation import patient_bootstrap, patient_grade_metrics, summarize
from crossmodalnet.explain import MaskReader, attention_summary, eigen_cam, grad_cam, score_map, score_slice, summarise
from crossmodalnet.models import CapsuleHead, CrossModalNet, build_model, squash


def fake_index(n_patients=60, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_patients):
        grade = "HGG" if i % 4 else "LGG"
        for k in range(int(rng.integers(10, 30))):
            label = C.CLASSES.index(grade) if rng.random() < 0.7 else C.NO_TUMOR
            rows.append(dict(sample_id=f"P{i}_{k}", patient_id=f"P{i}", slice_idx=k, label=label, grade=grade))
    return pd.DataFrame(rows)


def test_patient_split_has_no_leakage_and_image_split_does():
    df = fake_index()
    pw, iw = make_split(df, "patient"), make_split(df, "image")
    assert pw.groupby("patient_id").split.nunique().max() == 1
    assert iw.groupby("patient_id").split.nunique().max() > 1
    assert set(pw.split) == {"train", "val", "test"} == set(iw.split)
    s = split_summary(pw)
    assert (s.patients_in_other_partitions == 0).all() and s.slices.sum() == len(df)
    assert (split_summary(iw).patients_in_other_partitions > 0).all()
    # grade stratification: every partition has both grades
    assert all(((s.patients_HGG > 0) & (s.patients_LGG > 0)))


def test_split_is_deterministic():
    df = fake_index()
    assert (make_split(df, "patient", 7).split.values == make_split(df, "patient", 7).split.values).all()


def test_preprocessing():
    img = (np.random.default_rng(0).random((C.IMAGE_SIZE, C.IMAGE_SIZE)) * 255).astype(np.uint8)
    out = preprocess_slice(img)
    assert out.dtype == np.uint8 and out.shape == img.shape
    z = normalize_batch(torch.from_numpy(np.stack([img] * C.N_SEQS))[None])
    assert abs(z.mean().item()) < 1e-3 and abs(z.std().item() - 1) < 1e-2


def test_capsule_head_and_squash():
    assert squash(torch.randn(5, 4, 16)).norm(dim=-1).max() < 1.0
    h = CapsuleHead(32, C.N_CLASSES, n_types=4)
    lengths, v = h(torch.randn(2, 32, 7, 7))
    assert lengths.shape == (2, C.N_CLASSES) and v.shape == (2, C.N_CLASSES, 16)
    assert ((lengths >= 0) & (lengths < 1)).all()


def _model():
    return CrossModalNet(pretrained=False).eval()


def test_model_shapes_loss_and_variants():
    m = _model()
    x = torch.randn(2, C.N_SEQS, C.IMAGE_SIZE, C.IMAGE_SIZE)
    out, aux = m(x, return_aux=True)
    assert out.shape == (2, C.N_CLASSES) and aux["fused"].shape == (2, 768 * C.N_SEQS, 7, 7)
    assert len(aux["attn"]) == C.N_SEQS and aux["attn"][0].shape == (2, 49, 49 * (C.N_SEQS - 1))
    s = m.scores(out)
    assert torch.allclose(s.sum(1), torch.ones(2), atol=1e-5)
    m.loss(out, torch.tensor([0, 1])).backward()
    for cfg in [dict(seqs="0"), dict(no_attention=True), dict(head="mlp"), dict(separate_backbones=True)]:
        v = build_model(cfg, pretrained=False).eval()
        assert v(x).shape == (2, C.N_CLASSES)


def test_cams_and_attention():
    m = _model()
    x = torch.randn(2, C.N_SEQS, C.IMAGE_SIZE, C.IMAGE_SIZE)
    cams, out, aux = grad_cam(m, x, torch.tensor([0, 2]))
    assert cams.shape == (2, C.N_SEQS, C.IMAGE_SIZE, C.IMAGE_SIZE)
    assert cams.min() >= 0 and cams.max() <= 1 + 1e-6
    # batched Grad-CAM must equal single-image Grad-CAM
    single, _, _ = grad_cam(m, x[1:2], torch.tensor([2]))
    assert torch.allclose(cams[1], single[0], atol=1e-4)
    eig = eigen_cam(aux["feats"])
    assert eig.shape == cams.shape
    att = attention_summary(aux["attn"], C.N_SEQS)
    assert torch.allclose(att.sum(2), torch.ones(2, C.N_SEQS), atol=1e-4)    # each query puts all mass on the others


def test_metrics_and_bootstrap():
    rng = np.random.default_rng(0)
    y = rng.integers(0, C.N_CLASSES, 600)
    s = np.eye(C.N_CLASSES)[y] * 0.8 + rng.random((600, C.N_CLASSES)) * 0.2
    m = summarize(y, s)
    assert m["accuracy"] == 1.0 and m["mcc"] == 1.0 and m["macro_f1"] == 1.0
    patients = rng.integers(0, 40, 600).astype(str)
    ci = patient_bootstrap(y, s, patients, n_boot=20)
    assert all(lo <= hi for lo, hi in ci.values())
    assert isinstance(patient_grade_metrics(y, s, patients), dict)


def test_localization_metrics():
    g = np.zeros((C.IMAGE_SIZE, C.IMAGE_SIZE), bool)
    g[80:140, 90:150] = True
    perfect = score_map(g.astype(float), g)
    assert perfect["iou@0.5"] == 1 and perfect["dice@0.5"] == 1 and perfect["auroc"] == 1
    assert abs(perfect["energy_in_mask"] - 1) < 1e-6 and perfect["pointing_hit"] == 1
    inverse = score_map((~g).astype(float), g)
    assert inverse["iou@0.5"] == 0 and inverse["auroc"] == 0
    rng = np.random.default_rng(0)
    row = score_slice({"cam": g.astype(float)}, g, rng)
    assert row["cam/auroc"] == 1 and 0.4 < row["control_random/auroc"] < 0.6
    assert row["control_random/enrichment"] < 2 < row["cam/enrichment"]
    table = pd.DataFrame([{**row, "patient_id": f"P{i % 3}"} for i in range(6)])
    out = summarise(table, n_boot=10)
    assert out["n_patients"] == 3 and out["metrics"]["cam/auroc"]["mean"] == 1


def test_mask_reader_matches_slice_indexing():
    import nibabel as nib
    vol = np.zeros((240, 240, 155), np.int16)
    vol[100:140, 60:100, 70] = 2          # edema
    vol[110:120, 70:80, 70] = 4           # enhancing
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "PX"
        p.mkdir()
        nib.save(nib.Nifti1Image(vol, np.eye(4)), str(p / "PX_seg.nii"))
        whole, enh = MaskReader(d, "whole")("PX", 70), MaskReader(d, "enhancing")("PX", 70)
        assert whole.shape == (C.IMAGE_SIZE, C.IMAGE_SIZE) and whole.any() and 0 < enh.sum() < whole.sum()
        assert not MaskReader(d, "whole")("PX", 10).any()
        ys, xs = np.where(whole)                                   # same row/column order as the slice images
        assert abs(ys.mean() - 120 * C.IMAGE_SIZE / 240) < 2 and abs(xs.mean() - 80 * C.IMAGE_SIZE / 240) < 2


def test_seed_aggregation():
    import json
    from crossmodalnet.evaluation.seeds import aggregate, run_name
    with tempfile.TemporaryDirectory() as d:
        old = C.RESULT_DIR
        C.RESULT_DIR = Path(d)
        try:
            for s, acc in enumerate([0.8, 0.9, 1.0]):
                f = Path(d) / "patient" / run_name("m", s)
                f.mkdir(parents=True)
                test = {k: acc for k in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_auc_ovr", "mcc"]}
                test["per_class"] = {c: dict(precision=acc, recall=acc, f1=acc, auc_ovr=acc, support=1) for c in C.CLASSES}
                (f / "metrics.json").write_text(json.dumps({"test": test}))
            out = aggregate("patient", "m", (0, 1, 2))
            assert abs(out["overall"]["accuracy"]["mean"] - 0.9) < 1e-9
            assert abs(out["overall"]["accuracy"]["sd"] - 0.1) < 1e-9          # sample SD of 0.8, 0.9, 1.0
            assert (Path(d) / "patient" / "m_seeds" / "seed_summary.csv").exists()
            try:
                aggregate("patient", "m", (0, 1, 2, 3))
                raise AssertionError("missing seed should raise")
            except FileNotFoundError:
                pass
        finally:
            C.RESULT_DIR = old


def test_slice_extraction_rules_zip_input_and_missing_segmentation():
    import zipfile
    import nibabel as nib
    from crossmodalnet.data.extract_slices import extract_patient
    rng = np.random.default_rng(0)
    shape = (240, 240, 6)

    def write(path, vol, zipped=False):
        nii = Path(str(path))
        nib.save(nib.Nifti1Image(vol, np.eye(4)), str(nii))
        if zipped:
            with zipfile.ZipFile(str(nii) + ".zip", "w") as z:
                z.write(nii, nii.name)
            nii.unlink()

    with tempfile.TemporaryDirectory() as d:
        src, out = Path(d) / "src", Path(d) / "out"
        src.mkdir()
        img = (rng.random(shape) * 100 + 1).astype(np.float32)           # every slice non-empty
        seg = np.zeros(shape, np.int16)
        seg[100:120, 100:120, 2] = 1                                      # 400 px  -> grade class
        seg[10:13, 10:13, 3] = 2                                          # 9 px    -> discarded
        for key in ("t1", "t2", "flair", "t1ce"):
            write(src / f"PX_{key}.nii", img, zipped=True)                # zipped volumes
        write(src / "PX_seg.nii", seg)
        counts, reason = extract_patient("PX", "LGG", src, out)
        assert reason is None
        assert counts == {"HGG": 0, "LGG": 1, "No Tumor": 4}              # 6 slices - 1 discarded - 1 tumor
        for s in C.SEQS:
            png = out / "LGG" / s / "PX_2_" f"{s}.png"
            assert png.exists() and len(list((out / "No Tumor" / s).glob("*.png"))) == 4
        # BraTS20_Training_355 ships its segmentation as <patient>_Segm.nii.zip
        for key in ("t1", "t2", "flair"):
            write(src / f"PZ_{key}.nii", img)
        write(src / "PZ_Segm.nii", seg, zipped=True)
        counts, reason = extract_patient("PZ", "HGG", src, out)
        assert reason is None and counts["HGG"] == 1
        # no segmentation -> the patient is skipped, nothing is written
        for key in ("t1", "t2", "flair"):
            write(src / f"PY_{key}.nii", img)
        counts, reason = extract_patient("PY", "HGG", src, out)
        assert counts is None and "seg" in reason and not list(out.rglob("PY_*"))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
