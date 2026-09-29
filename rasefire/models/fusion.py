import torch
import torch.nn as nn


class EvidenceArbitration(nn.Module):
    """alpha_m(x,y) = softmax_m(r_m / tau);  F_fused = sum_m alpha_m * g_m * F_m,  g_m = r_m if gate else 1.
    The gate lets total evidence shrink when *all* sources are unreliable (softmax alone always sums to 1)."""

    def __init__(self, tau=0.25, gate=True):
        super().__init__()
        self.tau, self.gate = tau, gate

    def forward(self, feats, r):
        F_ = torch.stack(feats, 1)                 # (B,M,D,H,W)
        alpha = torch.softmax(r / self.tau, dim=1)  # (B,M,H,W)
        w = alpha * (r if self.gate else 1.0)
        return (w.unsqueeze(2) * F_).sum(1), alpha
