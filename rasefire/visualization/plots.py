import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from .. import MODALITIES
from ..corruption import apply_corruptions


def plot_degradation(results, scenario, path, key="iou_mean", ylabel="IoU"):
    """results: {model_name: robustness_sweep output}. Performance vs corruption severity, one curve per model."""
    plt.figure(figsize=(4.5, 3.4))
    for name, r in results.items():
        rec = r[scenario]
        plt.plot(np.array(rec["levels"]) * 100, rec[key], marker="o", label=f"{name} (R={rec['auc_iou']:.3f})")
    plt.xlabel("corruption severity (%)"); plt.ylabel(ylabel); plt.title(scenario); plt.legend(fontsize=7); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


def plot_risk_coverage(curves, path):
    """curves: {label: (coverage, risk, aurc)}."""
    plt.figure(figsize=(4.5, 3.4))
    for k, (cov, risk, aurc) in curves.items():
        plt.plot(cov, risk, label=f"{k} (AURC={aurc:.4f})")
    plt.xlabel("coverage"); plt.ylabel("pixel error rate (risk)"); plt.legend(fontsize=7); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


@torch.no_grad()
def plot_case_study(model, batch, specs, path, index=0):
    """Actual / predicted / uncertainty / per-modality reliability for one sample under a given corruption."""
    model.eval()
    cb, c = apply_corruptions({k: (v[index:index + 1] if torch.is_tensor(v) else v) for k, v in batch.items()}, specs)
    out = model(cb)
    panels = [("actual", cb["target"][0, 0]), ("predicted P(fire)", out["prob"][0, 0]),
              ("prediction uncertainty", out["unc"][0, 0])]
    if out.get("r") is not None:
        panels += [(f"{m} reliability", out["r"][0, i]) for i, m in enumerate(MODALITIES)]
        panels += [(f"{m} true corruption", c[0, i]) for i, m in enumerate(MODALITIES)]
    n = len(panels); cols = 4; rows = (n + cols - 1) // cols
    fig, ax = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    for a in np.ravel(ax):
        a.axis("off")
    for a, (t, im) in zip(np.ravel(ax), panels):
        a.imshow(im.cpu(), vmin=0, vmax=1, cmap="magma"); a.set_title(t, fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()
