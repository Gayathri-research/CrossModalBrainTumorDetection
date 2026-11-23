"""CrossModalNet: shared ConvNeXt-Small backbone -> tokens -> cross-modal attention ->
channel concatenation -> capsule classifier."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .. import config as C
from .capsule import CapsuleHead, MLPHead

FEAT_CH = 768      # ConvNeXt-Small last-stage channels
GRID = 7           # feature grid side at 224 x 224 input


def convnext_features(pretrained: bool) -> nn.Module:
    w = torchvision.models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
    return torchvision.models.convnext_small(weights=w).features


class CrossModalNet(nn.Module):
    """
    Args
      seqs            indices into config.SEQS that are used (ablation: single / paired sequences)
      attention       False -> plain concatenation of backbone features (ablation)
      head            'capsule' | 'mlp'
      shared_backbone False -> a separate backbone per sequence (ablation)
    """

    def __init__(self, n_classes: int = C.N_CLASSES, seqs=tuple(range(C.N_SEQS)), attention: bool = True,
                 head: str = "capsule", shared_backbone: bool = True, pretrained: bool = True, n_heads: int = 8):
        super().__init__()
        self.seqs = list(seqs)
        self.S = len(self.seqs)
        self.use_attention = attention and self.S > 1
        n_bb = 1 if shared_backbone else self.S
        self.backbones = nn.ModuleList([convnext_features(pretrained) for _ in range(n_bb)])
        n_tokens = GRID * GRID
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, FEAT_CH))
        self.mod_embed = nn.Parameter(torch.zeros(self.S, 1, FEAT_CH))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mod_embed, std=0.02)
        if self.use_attention:    # sequence k (query) attends to the tokens of all other sequences
            self.attn = nn.ModuleList([nn.MultiheadAttention(FEAT_CH, n_heads, batch_first=True) for _ in range(self.S)])
            self.norm = nn.ModuleList([nn.LayerNorm(FEAT_CH) for _ in range(self.S)])
        head_cls = CapsuleHead if head == "capsule" else MLPHead
        self.head = head_cls(FEAT_CH * self.S, n_classes)
        self.loss_kind = "margin" if head == "capsule" else "ce"

    # ------------------------------------------------------------------ stages
    def encode(self, x: torch.Tensor):
        """x (B, len(config.SEQS), H, W), z-scored -> list of S feature maps (B, 768, 7, 7)."""
        x = x[:, self.seqs]
        feats = []
        for k in range(self.S):
            img = x[:, k:k + 1].expand(-1, 3, -1, -1).contiguous(memory_format=torch.channels_last)
            feats.append(self.backbones[k % len(self.backbones)](img))
        return feats

    def fuse(self, feats):
        """Tokenise, cross-attend, concatenate along channels. Returns (B, S*768, 7, 7), attention weights."""
        B = feats[0].shape[0]
        toks = [f.flatten(2).transpose(1, 2) + self.pos_embed + self.mod_embed[k] for k, f in enumerate(feats)]
        attn_w = None
        if self.use_attention:
            refined, attn_w = [], []
            for k in range(self.S):
                ctx = torch.cat([t for j, t in enumerate(toks) if j != k], dim=1)
                out, w = self.attn[k](toks[k], ctx, ctx, need_weights=True, average_attn_weights=True)
                refined.append(self.norm[k](toks[k] + out))
                attn_w.append(w)                                  # (B, 49, 49 * (S-1))
            toks = refined
        fused = torch.cat(toks, dim=2)
        return fused.transpose(1, 2).reshape(B, self.S * FEAT_CH, GRID, GRID), attn_w

    def classify(self, fused: torch.Tensor):
        with torch.autocast("cuda", enabled=False):   # routing in float32
            return self.head(fused.float())

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        feats = self.encode(x)
        fused, attn_w = self.fuse(feats)
        out, caps = self.classify(fused)
        if return_aux:
            return out, dict(feats=feats, fused=fused, attn=attn_w, capsules=caps)
        return out

    # ------------------------------------------------------------ loss / scores
    def loss(self, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.loss_kind == "ce":
            return F.cross_entropy(out, y)
        t = F.one_hot(y, out.shape[1]).float()                    # margin loss (m+=0.9, m-=0.1, lambda=0.5)
        return (t * F.relu(0.9 - out) ** 2 + 0.5 * (1 - t) * F.relu(out - 0.1) ** 2).sum(1).mean()

    def scores(self, out: torch.Tensor) -> torch.Tensor:
        """Class scores in [0, 1] that sum to 1."""
        if self.loss_kind == "ce":
            return F.softmax(out, 1)
        return out / out.sum(1, keepdim=True).clamp_min(1e-8)
