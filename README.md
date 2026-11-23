# CrossModalNet

Multi-sequence brain MRI glioma classification into **HGG / LGG / No Tumor** from T1, T2 and FLAIR slices (BraTS 2020).

ConvNeXt-Small (weights shared across sequences) -> 7x7 tokens -> cross-modal attention (each sequence attends to
the other two) -> channel concatenation -> capsule classifier (dynamic routing, 3 iterations).

## Layout
```
crossmodalnet/         the package
  config.py            paths, class and sequence names, protocol constants
  data/                extract_slices, index, preprocessing, splits, dataset
  models/              crossmodalnet, capsule, baselines
  training/            trainer
  evaluation/          metrics, inference, report, seeds
  explain/             cam, masks, localization, runner
  cli/                 command-line entry points (below)
tests/                 synthetic checks of the code (no results)
scripts/               kaggle_copy_test_seg.py (copy segmentation files on Kaggle); run_history/ (queue scripts used for the full runs)
docs/                  manuscript, revised manuscript (+ its markdown source and figures), reviewer comments, plan, notes
splits/                split manifests written by make_splits (released with the code)
Data/                  inputs and caches, not in git (see below)
runs/ results/ logs/   checkpoints, metrics and figures, logs of the runs, not in git
```

## Data
Expected inputs (not stored in git):
* `Data/data/<class>/<sequence>/<patient>_<slice>_<sequence>.png`, 224 x 224 slices
* `Data/source/name_mapping.csv` (patient -> grade, BraTS 2020) and, to (re)build the slices, the BraTS volumes
* `Data/annotations/<patient>/<patient>_seg.nii`: BraTS segmentation of the patients used for annotation scoring
  (74 patients: all 56 patient-wise test patients and 18 others)
* `Data/cache/`: uint8 cache written by `make_splits`
* `Data/subset/`, `Data/test_seg/`: raw volumes and segmentation files the annotation folder was copied from

Slice rules (`crossmodalnet/data/extract_slices.py`): empty slices are dropped; a slice with no tumor voxel is
**No Tumor** (tumor-free slice of a glioma patient, not a healthy subject); a slice with more than 100 tumor voxels
takes the patient's grade (**HGG** or **LGG**); slices in between are discarded. A patient without a segmentation
file cannot be labelled and is skipped with a message.

## Pipeline (run from the project root)
```
python -m crossmodalnet.cli.extract_slices --source Data/source   # only if Data/data must be (re)built
python -m crossmodalnet.cli.make_splits                           # cache + split manifests
python -m crossmodalnet.cli.train      --split patient            # or --split image
python -m crossmodalnet.cli.evaluate   --split patient            # add --drop_seq 1 to zero a sequence (0=T1, 1=T2, 2=FLAIR)
python -m crossmodalnet.cli.explain    --split patient --n_per_class 20
python -m crossmodalnet.cli.explain    --split patient --seg_root Data/annotations --slices_per_patient 10
python -m crossmodalnet.cli.seeds      --split patient --seeds 0,1,2 --run   # repeated runs, mean +/- SD
python -m crossmodalnet.cli.complexity
python -m crossmodalnet.cli.ablations  --split patient            # prints commands; add --run to execute
python tests/test_pipeline.py                                     # or: python -m pytest tests -q
```
After adding patients to `Data/data`, delete `Data/cache` and run `make_splits` again. Every command accepts
`--dataset full|subset`; `subset` expects a 20-patient folder `Data/subset` and writes to `*_subset` folders.

## Protocols
* **patient-wise**: patients are stratified by grade and split 70/15/15 first; every slice and sequence of a
  patient, including its tumor-free slices, stays in one partition (asserted in code). This is the reported result.
* **image-wise (slice-wise)**: slices are split at random, stratified by class; patients appear in several
  partitions, so the score is optimistic. It is a reference that shows how much leakage inflates results.

Manifests: `splits/<mode>_manifest.csv` (partition of every slice), `splits/patient_split_manifest.csv`
(partition of every patient), `splits/<mode>_summary.csv` (patients and slices per partition).

## Training and evaluation protocol
Preprocessing: 1st-99th percentile clipping and CLAHE (clip 2.0, 8 x 8 tiles), cached as uint8; per-image z-score
at load time. No augmentation. AdamW (backbone lr 1e-4, head lr 1e-3, weight decay 0.05), one-cycle cosine
schedule, batch 16, bf16 autocast, capsule margin loss, at most 6 epochs, early stopping on validation macro-F1
(patience 3). Split seed 42; training seed 0 unless `--seed` is given. The test partition is read only by `evaluate`.
GPU training in bf16 is not bit-for-bit repeatable, so a repeated seed gives close but not identical results.

Seeds: a training seed is called converged if its best validation macro-F1 is at least 0.60 (validation only, never
test). Non-converged seeds are replaced and must be reported; see `docs/Revised_Manuscript.docx`, Section 4.3.

Metrics (`evaluation/metrics.py`): accuracy; class-wise precision, recall, F1 and one-vs-rest AUC; **macro** averages
(unweighted mean over classes); multiclass MCC; 95% CIs from a **patient-level bootstrap** (1,000 resamples; all
slices of a resampled patient move together); and a secondary patient-level HGG-vs-LGG accuracy/AUC computed from
the slices the model calls tumor.

## Explanations
* **Grad-CAM** is taken through the whole network: the target is the class-capsule length, back-propagated through
  routing, fusion and cross-modal attention into the last ConvNeXt stage of each sequence (one map per sequence).
* **Eigen-CAM**: first principal component of the same activations (gradient-free).
* Cross-attention mass between sequences is reported per slice.
* **Against annotations** (`--seg_root`, BraTS `<patient>/<patient>_seg.nii`; `--region whole|core|enhancing`):
  IoU and Dice at thresholds 0.3/0.5/0.7, pixel AUROC, energy inside the tumor, enrichment over uniform, and
  pointing-game hit, each computed over the whole image and **inside the brain only**, with random-map and
  centre-prior controls and patient-level bootstrap CIs. The brain-only scores are the ones to report (the raw maps
  are strong on the zero background). Maps are never multiplied by the mask.

Outputs go to `results/<split>/<name>/` (metrics.json, predictions.csv, confusion_matrix.png, roc.png, explain/,
annotation_eval/). Training logs and checkpoints go to `runs/<split>/<name>/`.
