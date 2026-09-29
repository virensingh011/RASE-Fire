import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(c):
    return nn.GroupNorm(min(8, c), c)


def conv_block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), _gn(o), nn.GELU(),
                         nn.Conv2d(o, o, 3, padding=1), _gn(o), nn.GELU())


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, base=32, depth=3):
        super().__init__()
        self.inc = conv_block(in_ch, base)
        self.downs = nn.ModuleList([conv_block(base * 2**i, base * 2**(i + 1)) for i in range(depth)])
        self.ups = nn.ModuleList([conv_block(base * 2**(i + 1) + base * 2**i, base * 2**i)
                                  for i in reversed(range(depth))])
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        skips = [self.inc(x)]
        for d in self.downs:
            skips.append(d(F.max_pool2d(skips[-1], 2)))
        h = skips.pop()
        for up in self.ups:
            s = skips.pop()
            h = F.interpolate(h, size=s.shape[-2:], mode="bilinear", align_corners=False)
            h = up(torch.cat([h, s], 1))
        return self.out(h)


def entropy(p, eps=1e-6):
    p = p.clamp(eps, 1 - eps)
    return -(p * p.log() + (1 - p) * (1 - p).log())


def finalize(logit, log_sigma=None):
    """Common output dict. With a heteroscedastic logit-noise head, the predictive probability uses the probit
    approximation E[sigmoid(z)] ~ sigmoid(mu / sqrt(1 + pi*s^2/8)) (deterministic, no MC at inference)."""
    out = {"logit": logit}
    if log_sigma is not None:
        ls = log_sigma.clamp(-6.0, 2.0)
        out["log_sigma"] = ls
        p = torch.sigmoid(logit / torch.sqrt(1 + math.pi * (2 * ls).exp() / 8))
    else:
        p = torch.sigmoid(logit)
    out["prob"] = p
    out["unc"] = entropy(p)  # predictive entropy, in [0, ln 2]
    return out
