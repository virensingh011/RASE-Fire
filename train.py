"""python train.py --config configs/rase_full.yaml [--set train.epochs=5 seed=1 ...]"""
import argparse, json, os
import torch
from rasefire.config import load_config
from rasefire.data import build_loaders
from rasefire.evaluation import evaluate_loader
from rasefire.models import build_model
from rasefire.training import Trainer, set_seed

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--set", nargs="*", default=[])
a = p.parse_args()
cfg = load_config(a.config, a.set)
set_seed(cfg.get("seed", 0))
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaders = build_loaders(cfg)
model = build_model(cfg)
print(f"params: {sum(x.numel() for x in model.parameters())/1e6:.3f}M  device: {dev}")
out_dir = os.path.join(cfg.get("out_dir", "runs"), cfg.get("name", os.path.basename(a.config)[:-5]) + f"_s{cfg.get('seed', 0)}")
Trainer(model, cfg, dev).fit(loaders["train"], loaders["val"], out_dir)
model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"), map_location=dev)["model"])
res = evaluate_loader(model, loaders["test"], dev, hausdorff=True)
summary = {k: float(v.mean()) for k, v in res["per_sample"].items() if k != "event_id"}
summary["ece"] = res["ece"]
print("TEST (clean):", json.dumps(summary))
json.dump(summary, open(os.path.join(out_dir, "test_clean.json"), "w"), indent=1)
