import json
import os
import random
import time
import numpy as np
import torch
from ..corruption import CorruptionSampler, ModalityDropout
from ..evaluation import evaluate_loader, to_device
from .losses import compute_loss


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def build_train_sampler(cfg):
    c = cfg["train"].get("corruption", {"mode": "none"})
    if c["mode"] == "aware":
        return CorruptionSampler(p=c.get("p", 0.5), kinds=c.get("kinds", ("missing", "noise", "spatial", "temporal", "bias")),
                                 seed=cfg.get("seed", 0) + 17)
    if c["mode"] == "dropout":
        return ModalityDropout(p=c.get("p", 0.3), seed=cfg.get("seed", 0) + 17)
    return None


class Trainer:
    def __init__(self, model, cfg, device):
        self.model, self.cfg, self.dev = model.to(device), cfg, device
        t = cfg["train"]
        self.opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t.get("weight_decay", 1e-4))
        self.sampler = build_train_sampler(cfg)
        self.lcfg = t.get("loss", {})
        self.log = []

    def _val_score(self, val_loader):
        clean = evaluate_loader(self.model, val_loader, self.dev)["per_sample"]["iou"].mean()
        if self.sampler is None:
            return float(clean), float(clean), None
        from ..corruption import CorruptionSampler
        fixed = CorruptionSampler(p=0.5, seed=1234)
        cor = evaluate_loader(self.model, val_loader, self.dev, sampler=fixed)["per_sample"]["iou"].mean()
        return float((clean + cor) / 2), float(clean), float(cor)

    def fit(self, train_loader, val_loader, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        best = -1.0
        for ep in range(self.cfg["train"]["epochs"]):
            self.model.train()
            agg, n, t0 = {}, 0, time.time()
            for batch in train_loader:
                batch = to_device(batch, self.dev)
                cmaps = None
                if self.sampler is not None:
                    batch, cmaps = self.sampler(batch)
                out = self.model(batch)
                out_p = None
                if self.lcfg.get("lambda_cons", 0) > 0:
                    pb = dict(batch)
                    for m in ("sat", "weather", "terrain"):
                        pb[m] = batch[m] + 0.02 * torch.randn_like(batch[m])
                    out_p = self.model(pb)
                loss, parts = compute_loss(out, batch, cmaps, self.lcfg, out_p)
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.opt.step()
                for k, v in parts.items():
                    agg[k] = agg.get(k, 0) + v
                n += 1
            score, clean, cor = self._val_score(val_loader)
            rec = {"epoch": ep, "sec": round(time.time() - t0, 1), "val_score": score, "val_iou_clean": clean,
                   "val_iou_corrupt": cor, **{f"loss_{k}": v / n for k, v in agg.items()}}
            self.log.append(rec)
            print(json.dumps(rec))
            if score > best:
                best = score
                torch.save({"model": self.model.state_dict(), "cfg": self.cfg, "epoch": ep, "val_score": score},
                           os.path.join(out_dir, "best.pt"))
        with open(os.path.join(out_dir, "train_log.json"), "w") as f:
            json.dump(self.log, f, indent=1)
        return best
