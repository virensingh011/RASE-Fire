import math
import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-6


def iou_dice(pred, tgt):
    inter = (pred * tgt).sum((1, 2, 3))
    ps, ts = pred.sum((1, 2, 3)), tgt.sum((1, 2, 3))
    return ((inter + EPS) / (ps + ts - inter + EPS)), ((2 * inter + EPS) / (ps + ts + EPS))


def _edge(m):
    return m - (1 - F.max_pool2d(1 - m, 3, 1, 1))


def boundary_f1(pred, tgt, tol=1):
    pe, te = _edge(pred), _edge(tgt)
    k = 2 * tol + 1
    pd, td = F.max_pool2d(pe, k, 1, tol), F.max_pool2d(te, k, 1, tol)
    ps, ts = pe.sum((1, 2, 3)), te.sum((1, 2, 3))
    prec, rec = (pe * td).sum((1, 2, 3)) / (ps + EPS), (te * pd).sum((1, 2, 3)) / (ts + EPS)
    f1 = 2 * prec * rec / (prec + rec + EPS)
    return torch.where((ps == 0) & (ts == 0), torch.ones_like(f1), f1)


def hausdorff(pred, tgt):
    """Symmetric Hausdorff distance (pixels) between binary masks; empty-vs-nonempty = image diagonal."""
    from scipy.ndimage import distance_transform_edt as edt
    out = []
    for p, t in zip(pred[:, 0].cpu().numpy() > 0.5, tgt[:, 0].cpu().numpy() > 0.5):
        if not p.any() and not t.any():
            out.append(0.0)
        elif not p.any() or not t.any():
            out.append(math.hypot(*p.shape))
        else:
            out.append(max(edt(~t)[p].max(), edt(~p)[t].max()))
    return np.array(out)


def brier(prob, tgt):
    return ((prob - tgt) ** 2).mean((1, 2, 3))


class CalibrationAccumulator:
    """Streaming ECE of the fire probability (equal-width bins)."""

    def __init__(self, bins=15):
        self.n = bins
        self.cnt, self.conf, self.pos = (torch.zeros(bins, dtype=torch.float64) for _ in range(3))

    def update(self, prob, tgt):
        idx = (prob.flatten().clamp(0, 1 - 1e-7) * self.n).long().cpu()
        self.cnt += torch.bincount(idx, minlength=self.n).double()
        self.conf += torch.bincount(idx, weights=prob.flatten().double().cpu(), minlength=self.n)
        self.pos += torch.bincount(idx, weights=tgt.flatten().double().cpu(), minlength=self.n)

    def ece(self):
        N = self.cnt.sum().clamp(min=1)
        gap = (self.conf - self.pos).abs() / N
        return float(gap.sum())


class RiskCoverageAccumulator:
    """Streaming risk-coverage: pixels ranked by an uncertainty score in [0,1]; risk = pixel error rate of retained pixels."""

    def __init__(self, bins=200):
        self.n = bins
        self.cnt, self.err = torch.zeros(bins, dtype=torch.float64), torch.zeros(bins, dtype=torch.float64)

    def update(self, score, prob, tgt):
        idx = (score.flatten().clamp(0, 1 - 1e-7) * self.n).long().cpu()
        err = ((prob > 0.5).float() != tgt).flatten().double().cpu()
        self.cnt += torch.bincount(idx, minlength=self.n).double()
        self.err += torch.bincount(idx, weights=err, minlength=self.n)

    def curve(self):
        cc, ce = self.cnt.cumsum(0), self.err.cumsum(0)
        keep = cc > 0
        cov = (cc[keep] / cc[-1]).numpy()
        risk = (ce[keep] / cc[keep]).numpy()
        aurc = float(((risk[1:] + risk[:-1]) / 2 * np.diff(cov)).sum()) if len(cov) > 1 else float(risk[0])
        return cov, risk, aurc


class CorrAccumulator:
    """Streaming Pearson correlation (used for reliability vs. (1 - corruption))."""

    def __init__(self):
        self.s = torch.zeros(6, dtype=torch.float64)

    def update(self, x, y):
        x, y = x.flatten().double().cpu(), y.flatten().double().cpu()
        self.s += torch.tensor([len(x), x.sum(), y.sum(), (x * x).sum(), (y * y).sum(), (x * y).sum()])

    def value(self):
        n, sx, sy, sxx, syy, sxy = self.s.tolist()
        vx, vy = sxx / n - (sx / n) ** 2, syy / n - (sy / n) ** 2
        return float("nan") if vx < 1e-12 or vy < 1e-12 else (sxy / n - sx * sy / n ** 2) / math.sqrt(vx * vy)
