"""Parameters, FLOPs, latency and peak GPU memory for one slice triplet (batch size 1).

    python -m crossmodalnet.cli.complexity [--model crossmodalnet]
"""
import argparse
import json
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

from .. import config as C
from ..models import build_model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", default="crossmodalnet")
    a = ap.parse_args()
    device = torch.device("cuda")
    model = build_model({"model": a.model}, pretrained=False).to(device).eval()
    x = torch.randn(1, C.N_SEQS, C.IMAGE_SIZE, C.IMAGE_SIZE, device=device)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(10):
            model(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(50):
            model(x)
        torch.cuda.synchronize()
    r = dict(model=a.model, params_M=sum(p.numel() for p in model.parameters()) / 1e6,
             GFLOPs=fc.get_total_flops() / 1e9, latency_ms=(time.time() - t0) / 50 * 1e3,
             peak_memory_MB=torch.cuda.max_memory_allocated() / 2 ** 20, gpu=torch.cuda.get_device_name(0))
    C.RESULT_DIR.mkdir(exist_ok=True)
    (C.RESULT_DIR / f"complexity_{a.model}.json").write_text(json.dumps(r, indent=2))
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
