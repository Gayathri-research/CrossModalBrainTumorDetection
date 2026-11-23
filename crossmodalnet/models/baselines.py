"""Comparison models (provided as code; the revision does not run them).
Each receives the (T1, T2, FLAIR) triplet stacked as its three input channels, so every
baseline sees the same information and the same splits as CrossModalNet."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

from .. import config as C


class Baseline(nn.Module):
    loss_kind = "ce"

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x, return_aux: bool = False):
        out = self.net(x.contiguous(memory_format=torch.channels_last))
        return (out, {}) if return_aux else out

    def loss(self, out, y):
        return F.cross_entropy(out, y)

    def scores(self, out):
        return F.softmax(out, 1)


def build_baseline(name: str, pretrained: bool = True, n_classes: int = C.N_CLASSES) -> Baseline:
    w = "DEFAULT" if pretrained else None
    if name == "resnet50":
        net = tvm.resnet50(weights=w); net.fc = nn.Linear(net.fc.in_features, n_classes)
    elif name == "vgg19":
        net = tvm.vgg19(weights=w); net.classifier[6] = nn.Linear(4096, n_classes)
    elif name == "densenet121":
        net = tvm.densenet121(weights=w); net.classifier = nn.Linear(net.classifier.in_features, n_classes)
    elif name == "mobilenet_v2":
        net = tvm.mobilenet_v2(weights=w); net.classifier[1] = nn.Linear(net.last_channel, n_classes)
    elif name == "inception_v3":
        net = tvm.inception_v3(weights=w, aux_logits=pretrained, init_weights=not pretrained)
        net.aux_logits, net.AuxLogits = False, None
        net.fc = nn.Linear(net.fc.in_features, n_classes)
    else:
        raise ValueError(f"unknown baseline {name!r}")
    return Baseline(net)


BASELINES = ["resnet50", "vgg19", "densenet121", "mobilenet_v2", "inception_v3"]
