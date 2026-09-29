import torch
import torch.nn as nn
from .. import MODALITIES
from .common import UNet, finalize
from .encoders import ModalityEncoder
from .fusion import EvidenceArbitration
from .reliability import ReliabilityEstimator


class RASEFire(nn.Module):
    """Per-modality encoders -> reliability fields -> evidence arbitration -> U-Net decoder -> forecast (+ uncertainty).
    Ablation switches: use_reliability, spatial, gate, uncertainty (reliability supervision lives in the loss config)."""

    def __init__(self, chans, T, dim=32, base=32, depth=3, use_reliability=True, spatial=True, gate=True,
                 tau=0.25, uncertainty=True):
        super().__init__()
        self.use_reliability, self.uncertainty = use_reliability, uncertainty
        self.enc = nn.ModuleDict({m: ModalityEncoder(chans[m], 1 if m == "terrain" else T, dim) for m in MODALITIES})
        self.rel = ReliabilityEstimator(dim, len(MODALITIES), spatial) if use_reliability else None
        self.arb = EvidenceArbitration(tau, gate)
        self.dec = UNet(dim, 2 if uncertainty else 1, base, depth)

    def forward(self, batch):
        feats = [self.enc[m](batch[m]) for m in MODALITIES]
        if self.use_reliability:
            r = self.rel(feats)
            fused, alpha = self.arb(feats, r)
        else:
            B, _, H, W = feats[0].shape
            r = torch.ones(B, len(MODALITIES), H, W, device=feats[0].device)
            alpha = r / len(MODALITIES)
            fused = torch.stack(feats, 1).mean(1)
        y = self.dec(fused)
        out = finalize(y[:, :1], y[:, 1:2] if self.uncertainty else None)
        out.update(r=r, alpha=alpha)
        return out
