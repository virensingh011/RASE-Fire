import math
from collections import defaultdict
from dataclasses import replace
import numpy as np
import torch
from .. import MODALITIES
from ..corruption import apply_corruptions
from . import metrics as M


def to_device(batch, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate_loader(model, loader, device, specs=None, sampler=None, hausdorff=False):
    """Clean or corrupted evaluation. `specs` = fixed CorruptionSpec list (seed offset per batch => reproducible);
    `sampler` = CorruptionSampler alternative. Returns per-sample metrics (with event ids) + streaming aggregates."""
    model.eval()
    per, cal, rc, rc_rel = defaultdict(list), M.CalibrationAccumulator(), M.RiskCoverageAccumulator(), M.RiskCoverageAccumulator()
    corr = {m: M.CorrAccumulator() for m in MODALITIES}
    has_rel = False
    for bi, batch in enumerate(loader):
        batch = to_device(batch, device)
        c = None
        if specs:
            batch, c = apply_corruptions(batch, [s.with_seed(s.seed + 1000 * bi) for s in specs])
        elif sampler is not None:
            batch, c = sampler(batch)
        out = model(batch)
        prob, tgt = out["prob"], batch["target"]
        pred = (prob > 0.5).float()
        iou, dice = M.iou_dice(pred, tgt)
        for k, v in (("iou", iou), ("dice", dice), ("boundary_f1", M.boundary_f1(pred, tgt)), ("brier", M.brier(prob, tgt))):
            per[k].append(v.cpu().numpy())
        if hausdorff:
            per["hausdorff"].append(M.hausdorff(pred, tgt))
        per["event_id"].append(np.asarray(batch["event_id"].cpu() if torch.is_tensor(batch["event_id"]) else batch["event_id"]))
        cal.update(prob, tgt)
        ent = out["unc"] / math.log(2)
        rc.update(ent, prob, tgt)
        if "r" in out and out["r"] is not None and model_has_reliability(model):
            has_rel = True
            rc_rel.update(0.5 * ent + 0.5 * (1 - out["r"].mean(1, keepdim=True)), prob, tgt)
            if c is not None:
                for i, m in enumerate(MODALITIES):
                    corr[m].update(out["r"][:, i], 1 - c[:, i])
    res = {"per_sample": {k: np.concatenate(v) for k, v in per.items()}, "ece": cal.ece()}
    res["risk_coverage"] = {"entropy": rc.curve()}
    if has_rel:
        res["risk_coverage"]["entropy+reliability"] = rc_rel.curve()
        if specs or sampler is not None:
            res["rel_corr"] = {m: corr[m].value() for m in MODALITIES}
    return res


def model_has_reliability(model):
    return bool(getattr(model, "use_reliability", False))
