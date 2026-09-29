from torch.utils.data import DataLoader
from .synthetic import SyntheticFireDataset
from .npz_dataset import NpzEventDataset
from .splits import make_splits


def build_loaders(cfg):
    d, bs, seed = cfg["data"], cfg["train"]["batch_size"], cfg.get("seed", 0)
    if d["name"] == "synthetic":
        kw = dict(size=d["size"], T=d["T"], per_event=d.get("per_event", 8))
        tr = SyntheticFireDataset(n=d["n_train"], seed=seed + 1, event_offset=0, **kw)
        va = SyntheticFireDataset(n=d["n_val"], seed=seed + 2, event_offset=10_000, **kw)
        te = SyntheticFireDataset(n=d["n_test"], seed=seed + 3, event_offset=20_000, **kw)
        ood = SyntheticFireDataset(n=d["n_test"], seed=seed + 4, event_offset=30_000, shift=True, **kw)
        loaders = {"train": DataLoader(tr, bs, shuffle=True, drop_last=True)}
        loaders.update({k: DataLoader(v, bs) for k, v in (("val", va), ("test", te), ("ood", ood))})
        return loaders
    if d["name"] == "npz":
        ev = None
        tr = NpzEventDataset(d["root"], years=set(d["train_years"]))
        va = NpzEventDataset(d["root"], years=set(d["val_years"]))
        te = NpzEventDataset(d["root"], years=set(d["test_years"]))
        return {"train": DataLoader(tr, bs, shuffle=True, drop_last=True, num_workers=d.get("workers", 2)),
                "val": DataLoader(va, bs), "test": DataLoader(te, bs)}
    raise ValueError(d["name"])
