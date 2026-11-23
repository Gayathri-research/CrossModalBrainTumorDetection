import torch
import torch.nn as nn
import torch.nn.functional as F


def squash(s: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    n2 = (s ** 2).sum(dim, keepdim=True)
    return n2 / (1.0 + n2) * s / torch.sqrt(n2 + eps)


class CapsuleHead(nn.Module):
    """Convolutional primary capsules (n_types x d_prim at every spatial position) routed to
    one capsule per class (d_out dims) by dynamic routing. Transformation matrices are shared
    across spatial positions, as in the original CapsNet. Returns (class lengths, class vectors)."""

    def __init__(self, in_ch: int, n_classes: int, n_types: int = 32, d_prim: int = 8,
                 d_out: int = 16, iters: int = 3):
        super().__init__()
        self.n_types, self.d_prim, self.d_out, self.n_classes, self.iters = n_types, d_prim, d_out, n_classes, iters
        self.primary = nn.Conv2d(in_ch, n_types * d_prim, kernel_size=1)
        self.W = nn.Parameter(0.05 * torch.randn(n_types, n_classes, d_out, d_prim))

    def forward(self, x: torch.Tensor):
        B, _, H, W = x.shape
        u = squash(self.primary(x).view(B, self.n_types, self.d_prim, H * W).permute(0, 1, 3, 2))   # B,t,N,dp
        u_hat = torch.einsum("tcod,btnd->btnco", self.W, u).reshape(B, -1, self.n_classes, self.d_out)
        logits = torch.zeros(B, u_hat.shape[1], self.n_classes, device=x.device, dtype=x.dtype)
        for r in range(self.iters):
            coupling = F.softmax(logits, dim=2)                                   # over classes
            v = squash((coupling.unsqueeze(-1) * u_hat).sum(1))                   # B,C,d_out
            if r < self.iters - 1:
                logits = logits + (u_hat * v.unsqueeze(1)).sum(-1)               # agreement
        return v.norm(dim=-1), v


class MLPHead(nn.Module):
    """Non-capsule head used by the ablation variants."""

    def __init__(self, in_ch: int, n_classes: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_ch, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x: torch.Tensor):
        return self.net(x.mean(dim=(2, 3))), None
