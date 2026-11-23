"""Training loop. Model selection uses the validation partition only (best macro-F1, early stopping)."""
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .. import config as C
from ..data import load_split, make_loader, normalize_batch
from ..evaluation import validation_metrics
from ..models import build_model


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _param_groups(model, lr: float, head_lr: float):
    is_backbone = lambda n: n.startswith(("backbones", "net."))
    bb = [p for n, p in model.named_parameters() if is_backbone(n)]
    hd = [p for n, p in model.named_parameters() if not is_backbone(n)]
    return [dict(params=bb, lr=lr)] + ([dict(params=hd, lr=head_lr)] if hd else [])


def fit(cfg: dict) -> Path:
    """cfg keys: split, model, name, epochs, patience, batch, lr, head_lr, wd, seed, workers,
    seqs, no_attention, head, separate_backbones, resume."""
    set_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    run = C.RUN_DIR / cfg["split"] / (cfg["name"] or cfg["model"])
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(cfg, indent=2))

    df = load_split(cfg["split"])
    train_loader = make_loader(df, "train", cfg["batch"], True, cfg["workers"])
    val_loader = make_loader(df, "val", cfg["batch"] * 2, False, cfg["workers"])

    model = build_model(cfg).to(device).to(memory_format=torch.channels_last)
    groups = _param_groups(model, cfg["lr"], cfg["head_lr"])
    opt = torch.optim.AdamW(groups, weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in groups], total_steps=cfg["epochs"] * len(train_loader),
        pct_start=0.1, anneal_strategy="cos")
    print(f"{cfg['model']}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters | "
          f"{cfg['split']}-wise | {len(train_loader)} steps/epoch", flush=True)

    best, bad, start, hist = -1.0, 0, 0, []
    last = run / "last.pt"
    if cfg.get("resume") and last.exists():
        ck = torch.load(last, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        best, bad, start, hist = ck["best"], ck["bad"], ck["epoch"], ck["hist"]

    for epoch in range(start, cfg["epochs"]):
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        for step, (x, y, _) in enumerate(train_loader):
            x = normalize_batch(x.to(device, non_blocking=True))
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
            loss = model.loss(out.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            running += loss.item() * len(y); seen += len(y)
            if step % 100 == 0:
                print(f"  epoch {epoch + 1} step {step}/{len(train_loader)} loss {running / seen:.4f}", flush=True)
        m = validation_metrics(model, val_loader, device)
        rec = dict(epoch=epoch + 1, train_loss=running / seen, val_loss=m["loss"], val_acc=m["accuracy"],
                   val_macro_f1=m["macro_f1"], val_mcc=m["mcc"], seconds=time.time() - t0)
        hist.append(rec)
        pd.DataFrame(hist).to_csv(run / "history.csv", index=False)
        print("epoch {epoch}: ".format(**rec) + " ".join(f"{k}={v:.4f}" for k, v in rec.items() if k != "epoch"), flush=True)
        if m["macro_f1"] > best:
            best, bad = m["macro_f1"], 0
            torch.save(model.state_dict(), run / "best.pt")
        else:
            bad += 1
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        best=best, bad=bad, epoch=epoch + 1, hist=hist), last)
        if bad >= cfg["patience"]:
            print("early stopping", flush=True)
            break
    return run
