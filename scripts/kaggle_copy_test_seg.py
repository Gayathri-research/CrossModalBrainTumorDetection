# Kaggle notebook cell: copy ONLY the segmentation files of the 54 remaining test patients
# (the other 2 test patients, 292 and 332, are already in Data/subset). Segmentation files are small.
import os, shutil

SRC = "/kaggle/input/datasets/awsaf49/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
OUT = "/kaggle/working/test_seg"

PATIENTS = [
    "BraTS20_Training_002",
    "BraTS20_Training_009",
    "BraTS20_Training_015",
    "BraTS20_Training_024",
    "BraTS20_Training_044",
    "BraTS20_Training_048",
    "BraTS20_Training_049",
    "BraTS20_Training_050",
    "BraTS20_Training_051",
    "BraTS20_Training_055",
    "BraTS20_Training_059",
    "BraTS20_Training_082",
    "BraTS20_Training_092",
    "BraTS20_Training_095",
    "BraTS20_Training_100",
    "BraTS20_Training_103",
    "BraTS20_Training_106",
    "BraTS20_Training_111",
    "BraTS20_Training_122",
    "BraTS20_Training_130",
    "BraTS20_Training_131",
    "BraTS20_Training_135",
    "BraTS20_Training_150",
    "BraTS20_Training_152",
    "BraTS20_Training_162",
    "BraTS20_Training_164",
    "BraTS20_Training_170",
    "BraTS20_Training_172",
    "BraTS20_Training_175",
    "BraTS20_Training_189",
    "BraTS20_Training_191",
    "BraTS20_Training_192",
    "BraTS20_Training_202",
    "BraTS20_Training_206",
    "BraTS20_Training_234",
    "BraTS20_Training_242",
    "BraTS20_Training_257",
    "BraTS20_Training_258",
    "BraTS20_Training_266",
    "BraTS20_Training_267",
    "BraTS20_Training_276",
    "BraTS20_Training_280",
    "BraTS20_Training_304",
    "BraTS20_Training_305",
    "BraTS20_Training_313",
    "BraTS20_Training_314",
    "BraTS20_Training_326",
    "BraTS20_Training_335",
    "BraTS20_Training_344",
    "BraTS20_Training_345",
    "BraTS20_Training_347",
    "BraTS20_Training_350",
    "BraTS20_Training_355",
    "BraTS20_Training_357"
]

os.makedirs(OUT, exist_ok=True)
missing = []
for p in PATIENTS:
    src = os.path.join(SRC, p)
    seg = [f for f in os.listdir(src) if "seg" in f.lower()] if os.path.isdir(src) else []
    if not seg:
        missing.append(p)
        continue
    os.makedirs(os.path.join(OUT, p), exist_ok=True)
    for f in seg:
        shutil.copy(os.path.join(src, f), os.path.join(OUT, p, f))

print("copied:", len(PATIENTS) - len(missing), "missing:", missing)
shutil.make_archive("/kaggle/working/test_seg", "zip", OUT)   # download test_seg.zip from the Output tab
