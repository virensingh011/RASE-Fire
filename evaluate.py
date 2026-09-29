"""Full evaluation: clean + OOD + robustness campaign + selective prediction + case study.
python evaluate.py --ckpt rase=runs/rase_full_s0/best.pt concat=runs/concat_unet_s0/best.pt --out results"""
import argparse, json, os
import numpy as np, torch
from rasefire.corruption import CorruptionSpec
from rasefire.data import build_loaders
from rasefire.evaluation import (SINGLE, bootstrap_ci, evaluate_loader, per_event_mean, robustness_sweep, to_device)
from rasefire.models import build_model
from rasefire.visualization import plot_case_study, plot_degradation, plot_risk_coverage

p = argparse.ArgumentParser()
p.add_argument("--ckpt", nargs="+", required=True, help="name=path ...")
p.add_argument("--out", default="results")
p.add_argument("--scenarios", nargs="*", default=["sat:missing", "sat:noise", "sat:spatial", "weather:noise", "weather:bias",
                                                  "weather:temporal", "fire:missing", "combined"])
p.add_argument("--levels", nargs="*", type=float, default=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
summary, sweeps, curves = {}, {}, {}
for spec in a.ckpt:
    name, path = spec.split("=")
    ck = torch.load(path, map_location=dev)
    cfg = ck["cfg"]
    loaders = build_loaders(cfg)
    model = build_model(cfg).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    row = {"params_M": sum(x.numel() for x in model.parameters()) / 1e6}
    for split in [s for s in ("test", "ood") if s in loaders]:
        r = evaluate_loader(model, loaders[split], dev, hausdorff=True)
        ps = r["per_sample"]
        for k in ("iou", "dice", "boundary_f1", "hausdorff", "brier"):
            row[f"{split}_{k}"] = bootstrap_ci(per_event_mean(ps[k], ps["event_id"]))
        row[f"{split}_ece"] = r["ece"]
        if split == "test":
            row["aurc"] = {k: v[2] for k, v in r["risk_coverage"].items()}
            curves[name] = r["risk_coverage"]
    sweeps[name] = robustness_sweep(model, loaders["test"], dev, a.scenarios, a.levels)
    row["robust_auc_iou"] = {s: v["auc_iou"] for s, v in sweeps[name].items()}
    if getattr(model, "use_reliability", False):
        row["rel_corr_at_50pct"] = {s: (v["rel_corr"][a.levels.index(0.5)] if 0.5 in a.levels else None) for s, v in sweeps[name].items()}
        b = next(iter(loaders["test"]))
        plot_case_study(model, to_device(b, dev), [CorruptionSpec("sat", "spatial", 0.4, 0)], f"{a.out}/case_{name}.png")
    summary[name] = row
    print(name, json.dumps(row, indent=1, default=float))
for s in a.scenarios:
    plot_degradation(sweeps and {n: sw for n, sw in sweeps.items()}, s, f"{a.out}/degradation_{s.replace(':', '_')}.png")
for n, c in curves.items():
    plot_risk_coverage(c, f"{a.out}/risk_coverage_{n}.png")
json.dump({"summary": summary, "sweeps": sweeps}, open(f"{a.out}/results.json", "w"), indent=1, default=float)
