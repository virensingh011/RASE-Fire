import numpy as np
from ..corruption import CorruptionSpec
from .evaluate import evaluate_loader
from .statistics import per_event_mean

LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
SINGLE = {  # scenario -> (modality, kind)
    f"{m}:{k}": (m, k)
    for m in ("sat", "weather", "terrain", "fire")
    for k in ("missing", "noise", "spatial", "temporal", "bias") if not (k == "temporal" and m == "terrain")
}


def combined_failure(s, seed=0):
    """Satellite partly missing, weather noisy AND stale, fire state partially observed, terrain clean."""
    return [CorruptionSpec("sat", "missing", s, seed), CorruptionSpec("weather", "noise", s, seed + 1),
            CorruptionSpec("weather", "temporal", s, seed + 2), CorruptionSpec("fire", "missing", 0.5 * s, seed + 3)]


def scenario_specs(name, severity, seed=0):
    if name == "combined":
        return combined_failure(severity, seed)
    m, k = SINGLE[name]
    return [CorruptionSpec(m, k, severity, seed)]


def robustness_auc(levels, values):
    """R = (1/(c_max-c_min)) * integral of performance over corruption severity (trapezoid)."""
    l, v = np.asarray(levels, float), np.asarray(values, float)
    return float(((v[1:] + v[:-1]) / 2 * np.diff(l)).sum() / (l[-1] - l[0]))


def robustness_sweep(model, loader, device, scenarios, levels=LEVELS, seed=0):
    out = {}
    for name in scenarios:
        rec = {"levels": list(levels), "iou_mean": [], "ece": [], "aurc": [], "rel_corr": [], "iou_per_event": []}
        for s in levels:
            r = evaluate_loader(model, loader, device, specs=scenario_specs(name, s, seed))
            ps = r["per_sample"]
            ev = per_event_mean(ps["iou"], ps["event_id"])
            rec["iou_per_event"].append(ev.tolist())
            rec["iou_mean"].append(float(ev.mean()))
            rec["ece"].append(r["ece"])
            rec["aurc"].append(r["risk_coverage"]["entropy"][2])
            rec["rel_corr"].append(r.get("rel_corr"))
        rec["auc_iou"] = robustness_auc(levels, rec["iou_mean"])
        out[name] = rec
    return out
