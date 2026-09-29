"""Primitive corruption operators. Each returns (corrupted_x, c) with c in [0,1], shape (B,1,H,W):
per-pixel corruption severity (ground truth for reliability supervision / evaluation).
x is (B,T,C,H,W) (dynamic) or (B,C,H,W) (static). All randomness comes from an explicit seed."""
import math
import torch
import torch.nn.functional as F


def _lift(x):
    return (x.unsqueeze(1), True) if x.dim() == 4 else (x, False)


def _lower(x, sq):
    return x.squeeze(1) if sq else x


def _gen(seed):
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def _const(x5, s):
    B, _, _, H, W = x5.shape
    return torch.full((B, 1, H, W), float(s), device=x5.device, dtype=x5.dtype)


def missing(x, severity, seed, block=8):
    """Block-wise missing data (fill 0 = normalised mean). `severity` = fraction of blocks lost."""
    x5, sq = _lift(x)
    B, T, C, H, W = x5.shape
    gh, gw = math.ceil(H / block), math.ceil(W / block)
    drop = (torch.rand(B, 1, gh, gw, generator=_gen(seed)) < severity).float()
    m = F.interpolate(drop, size=(gh * block, gw * block), mode="nearest")[..., :H, :W].to(x5)
    return _lower(x5 * (1 - m.unsqueeze(1)), sq), m


def noise(x, severity, seed, radiometric=False, sigma_max=1.0):
    """Gaussian sensor noise, sigma = severity * sigma_max (in normalised units).
    radiometric=True adds per-frame/band gain error on top (satellite-style)."""
    x5, sq = _lift(x)
    B, T, C, H, W = x5.shape
    g = _gen(seed)
    s = severity * sigma_max
    out = x5 + s * torch.randn(x5.shape, generator=g).to(x5)
    if radiometric:
        gain = torch.randn(B, T, C, 1, 1, generator=g).to(x5)
        out = out + 0.5 * s * gain * x5
    return _lower(out, sq), _const(x5, min(1.0, severity))


def spatial(x, severity, seed):
    """One contiguous rectangular region (area fraction = severity) is lost."""
    x5, sq = _lift(x)
    B, T, C, H, W = x5.shape
    g = _gen(seed)
    frac = math.sqrt(max(0.0, min(1.0, severity)))
    h, w = int(round(H * frac)), int(round(W * frac))
    top = torch.randint(0, H - h + 1, (B,), generator=g)
    left = torch.randint(0, W - w + 1, (B,), generator=g)
    ys = torch.arange(H).view(1, H, 1)
    xs = torch.arange(W).view(1, 1, W)
    m = ((ys >= top.view(B, 1, 1)) & (ys < (top + h).view(B, 1, 1)) &
         (xs >= left.view(B, 1, 1)) & (xs < (left + w).view(B, 1, 1))).float().unsqueeze(1).to(x5)
    return _lower(x5 * (1 - m.unsqueeze(1)), sq), m


def temporal(x, severity, seed):
    """Staleness: sequence lagged by round(severity*(T-1)) steps (older frames repeated). No-op for static inputs."""
    if x.dim() == 4:
        return x, torch.zeros(x.shape[0], 1, x.shape[-2], x.shape[-1], device=x.device, dtype=x.dtype)
    T = x.shape[1]
    lag = int(round(severity * (T - 1)))
    if lag == 0:
        return x, _const(x, 0.0)
    idx = torch.clamp(torch.arange(T) - lag, min=0).to(x.device)
    return x[:, idx], _const(x, severity)


def bias(x, severity, seed, bias_max=1.5):
    """Systematic per-channel offset of +/- severity*bias_max (normalised units)."""
    x5, sq = _lift(x)
    B, T, C, H, W = x5.shape
    sign = torch.randint(0, 2, (B, 1, C, 1, 1), generator=_gen(seed)).float().to(x5) * 2 - 1
    return _lower(x5 + severity * bias_max * sign, sq), _const(x5, severity)
