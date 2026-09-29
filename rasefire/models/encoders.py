import torch.nn as nn
from .common import _gn


class ModalityEncoder(nn.Module):
    """(B,T,C,H,W) or (B,C,H,W) -> (B,dim,H,W). Time is folded into channels (simple, resolution-preserving)."""

    def __init__(self, in_ch, T, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(T * in_ch, dim, 3, padding=1), _gn(dim), nn.GELU(),
                                 nn.Conv2d(dim, dim, 3, padding=1), _gn(dim), nn.GELU())

    def forward(self, x):
        if x.dim() == 5:
            x = x.flatten(1, 2)
        return self.net(x)
