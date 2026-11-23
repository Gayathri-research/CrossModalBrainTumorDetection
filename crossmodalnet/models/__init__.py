import torch

from .baselines import BASELINES, build_baseline
from .capsule import CapsuleHead, MLPHead, squash
from .crossmodalnet import CrossModalNet


def build_model(cfg: dict, pretrained: bool = True) -> torch.nn.Module:
    """Build a model from a run config (the dict saved in runs/<split>/<name>/config.json)."""
    if cfg.get("model", "crossmodalnet") == "crossmodalnet":
        return CrossModalNet(seqs=[int(i) for i in str(cfg.get("seqs", "0,1,2")).split(",")],
                             attention=not cfg.get("no_attention", False), head=cfg.get("head", "capsule"),
                             shared_backbone=not cfg.get("separate_backbones", False), pretrained=pretrained)
    return build_baseline(cfg["model"], pretrained=pretrained)
