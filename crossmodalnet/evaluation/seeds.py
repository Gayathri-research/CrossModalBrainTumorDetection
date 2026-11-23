"""Aggregate repeated runs (different training seeds, same split) into mean and standard deviation."""
import json

import numpy as np
import pandas as pd

from .. import config as C
from .metrics import SUMMARY_KEYS


def run_name(base: str, seed: int) -> str:
    return f"{base}_s{seed}"


def aggregate(split: str, base: str = "crossmodalnet", seeds=(0, 1, 2), part: str = "test") -> dict:
    """Reads results/<split>/<base>_s<seed>/metrics.json for each seed and writes
    results/<split>/<base>_seeds/{seed_summary.json, seed_summary.csv}.
    SD is the sample standard deviation (ddof=1) over seeds."""
    runs = {}
    for s in seeds:
        path = C.RESULT_DIR / split / run_name(base, s) / "metrics.json"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}; run training and evaluation for seed {s} first")
        runs[s] = json.loads(path.read_text())[part]

    def stat(values):
        v = np.asarray(values, float)
        return dict(mean=float(np.nanmean(v)), sd=float(np.nanstd(v, ddof=1)) if len(v) > 1 else float("nan"),
                    per_seed=[float(x) for x in v])

    out = {"split_protocol": f"{split}-wise", "partition": part, "seeds": list(seeds),
           "overall": {k: stat([runs[s][k] for s in seeds]) for k in SUMMARY_KEYS},
           "per_class": {c: {m: stat([runs[s]["per_class"][c][m] for s in seeds])
                             for m in ("precision", "recall", "f1", "auc_ovr")} for c in C.CLASSES}}
    rows = [dict(scope="overall", name=k, mean=v["mean"], sd=v["sd"]) for k, v in out["overall"].items()]
    rows += [dict(scope=c, name=m, mean=v["mean"], sd=v["sd"]) for c, d in out["per_class"].items() for m, v in d.items()]
    folder = C.RESULT_DIR / split / f"{base}_seeds"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "seed_summary.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame(rows).to_csv(folder / "seed_summary.csv", index=False)
    return out
