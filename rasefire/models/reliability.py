import torch
import torch.nn as nn


class ReliabilityEstimator(nn.Module):
    """r_m(x,y) in [0,1] for every modality. Each head sees its own features and the mean of the *other* modalities'
    features (cross-modal consistency is what lets it detect a corrupted source without a validity mask).
    spatial=False collapses each map to one global scalar per sample (ablation: global vs pixel-level reliability)."""

    def __init__(self, dim, n_mod, spatial=True):
        super().__init__()
        self.spatial, self.n_mod = spatial, n_mod
        h = max(8, dim // 2)
        self.heads = nn.ModuleList([nn.Sequential(nn.Conv2d(2 * dim, h, 3, padding=1), nn.GELU()) for _ in range(n_mod)])
        self.outs = nn.ModuleList([nn.Conv2d(h, 1, 1) for _ in range(n_mod)])

    def forward(self, feats):
        s = torch.stack(feats, 1).sum(1)
        rs = []
        for m, f in enumerate(feats):
            ctx = (s - f) / (self.n_mod - 1)
            h = self.heads[m](torch.cat([f, ctx], 1))
            if not self.spatial:
                h = h.mean((2, 3), keepdim=True)
            r = torch.sigmoid(self.outs[m](h))
            rs.append(r.expand(-1, -1, f.shape[-2], f.shape[-1]))
        return torch.cat(rs, 1)  # (B,M,H,W)
