"""Class-activation maps for CrossModalNet.

Grad-CAM: the target is the class-capsule length of the chosen class. Its gradient is
back-propagated through dynamic routing, fusion and cross-modal attention into the last
ConvNeXt stage of EACH sequence, giving one map per sequence.
Eigen-CAM: gradient-free; first principal component of the same last-stage activations.
All maps are returned at (B, S, H, W), each normalised to [0, 1] per image.
"""
import torch
import torch.nn.functional as F

from .. import config as C


def _upsample_normalise(maps: torch.Tensor) -> torch.Tensor:
    """(B, S, g, g) -> (B, S, H, W) bilinear, min-max normalised per map."""
    B, S = maps.shape[:2]
    up = F.interpolate(maps.reshape(B * S, 1, *maps.shape[2:]), size=(C.IMAGE_SIZE, C.IMAGE_SIZE),
                       mode="bilinear", align_corners=False).reshape(B, S, C.IMAGE_SIZE, C.IMAGE_SIZE)
    lo = up.amin(dim=(2, 3), keepdim=True)
    up = up - lo
    return up / (up.amax(dim=(2, 3), keepdim=True) + 1e-8)


def grad_cam(model, x: torch.Tensor, target: torch.Tensor):
    """x: (B, len(config.SEQS), H, W) z-scored; target: (B,) class index per image.
    Returns (cams (B, S, H, W), class lengths (B, C) detached, aux dict).
    Gradients of different images are independent (eval mode, per-sample LayerNorm), so
    one backward pass over the summed target scores gives every image's own Grad-CAM."""
    model.eval()
    x = x.requires_grad_(False)
    with torch.enable_grad():
        out, aux = model(x, return_aux=True)
        score = out.gather(1, target[:, None]).sum()
        grads = torch.autograd.grad(score, aux["feats"])
        maps = torch.stack([F.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(1)) for a, g in zip(aux["feats"], grads)], 1)
    return _upsample_normalise(maps.detach()), out.detach(), aux


@torch.no_grad()
def eigen_cam(feats) -> torch.Tensor:
    """feats: list of S maps (B, D, g, g) -> (B, S, H, W)."""
    maps = []
    for a in feats:
        B, D, g, _ = a.shape
        t = a.detach().float().flatten(2).transpose(1, 2)            # B, g*g, D
        t = t - t.mean(1, keepdim=True)
        _, _, vh = torch.linalg.svd(t, full_matrices=False)
        p = torch.einsum("bnd,bd->bn", t, vh[:, 0])
        sign = torch.where(p.sum(1, keepdim=True) < 0, -1.0, 1.0)     # SVD sign is arbitrary
        maps.append(F.relu(p * sign).view(B, g, g))
    return _upsample_normalise(torch.stack(maps, 1))


def attention_summary(attn_weights, n_seqs: int):
    """attn_weights: list over sequences of (B, tokens, tokens*(S-1)) -> (B, S, S) mean attention mass
    that sequence k (row) places on sequence j (column); diagonal is 0 (no self-attention)."""
    if attn_weights is None:
        return None
    B = attn_weights[0].shape[0]
    out = torch.zeros(B, n_seqs, n_seqs)
    for k, w in enumerate(attn_weights):
        others = [j for j in range(n_seqs) if j != k]
        per = w.detach().float().cpu().mean(1).reshape(B, len(others), -1).sum(-1)
        for c, j in enumerate(others):
            out[:, k, j] = per[:, c]
    return out
