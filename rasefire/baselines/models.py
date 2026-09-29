import torch
import torch.nn as nn
from .. import MODALITIES
from ..models.common import UNet, finalize
from ..models.encoders import ModalityEncoder


class ConcatUNet(nn.Module):
    """Baselines A/B (and E when trained with ModalityDropout): all modalities concatenated -> U-Net."""

    def __init__(self, chans, T, base=32, depth=3):
        super().__init__()
        cin = sum(chans[m] * (1 if m == "terrain" else T) for m in MODALITIES)
        self.net = UNet(cin, 1, base, depth)

    def forward(self, batch):
        x = torch.cat([batch[m].flatten(1, 2) if batch[m].dim() == 5 else batch[m] for m in MODALITIES], 1)
        return finalize(self.net(x))


class ConvLSTMCell(nn.Module):
    def __init__(self, i, h, k=3):
        super().__init__()
        self.conv = nn.Conv2d(i + h, 4 * h, k, padding=k // 2)

    def forward(self, x, state):
        h, c = state
        i, f, o, g = self.conv(torch.cat([x, h], 1)).chunk(4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        return torch.sigmoid(o) * torch.tanh(c), c


class ConvLSTMNet(nn.Module):
    """Baseline C: per-step [sat, weather, fire, terrain] -> stacked ConvLSTM -> head on last hidden state."""

    def __init__(self, chans, T, hidden=32, layers=2):
        super().__init__()
        cin = chans["sat"] + chans["weather"] + chans["fire"] + chans["terrain"]
        self.cells = nn.ModuleList([ConvLSTMCell(cin if i == 0 else hidden, hidden) for i in range(layers)])
        self.hidden = hidden
        self.head = nn.Conv2d(hidden, 1, 1)

    def forward(self, batch):
        B, T, _, H, W = batch["sat"].shape
        states = [(torch.zeros(B, self.hidden, H, W, device=batch["sat"].device),) * 2 for _ in self.cells]
        for t in range(T):
            x = torch.cat([batch["sat"][:, t], batch["weather"][:, t], batch["fire"][:, t], batch["terrain"]], 1)
            for i, cell in enumerate(self.cells):
                states[i] = cell(x, states[i])
                x = states[i][0]
        return finalize(self.head(x))


class AttentionFusion(nn.Module):
    """Baseline F: generic per-pixel cross-modal attention (learned scores, no reliability supervision/semantics)."""

    def __init__(self, chans, T, dim=32, base=32, depth=3):
        super().__init__()
        self.enc = nn.ModuleDict({m: ModalityEncoder(chans[m], 1 if m == "terrain" else T, dim) for m in MODALITIES})
        self.q, self.k = nn.Conv2d(dim, dim, 1), nn.Conv2d(dim, dim, 1)
        self.dec = UNet(dim, 1, base, depth)
        self.dim = dim

    def forward(self, batch):
        F_ = torch.stack([self.enc[m](batch[m]) for m in MODALITIES], 1)  # (B,M,D,H,W)
        q = self.q(F_.mean(1)).unsqueeze(1)
        k = torch.stack([self.k(F_[:, i]) for i in range(F_.shape[1])], 1)
        a = torch.softmax((q * k).sum(2) / self.dim ** 0.5, dim=1)          # (B,M,H,W)
        return finalize(self.dec((a.unsqueeze(2) * F_).sum(1)))


class UTAE(nn.Module):
    """Baseline D placeholder. Use the authors' UTAE implementation released with TS-SatFire so the published
    baseline is reproduced faithfully (see README, Phase 1/2); wire it up behind this interface."""

    def __init__(self, *a, **k):
        raise NotImplementedError("Plug in the official TS-SatFire UTAE implementation (returns finalize(logit)).")
