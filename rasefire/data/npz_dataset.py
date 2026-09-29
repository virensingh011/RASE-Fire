"""Adapter contract for real data (TS-SatFire after its official preprocessing, CanadaFireSat after conversion).
One .npz per sample with keys:  sat (T,Cs,H,W)  weather (T,Cw,H,W)  terrain (Ct,H,W)  fire (T,1,H,W)  target (1,H,W)
and scalars  event_id (int)  year (int)  region (str, optional).  All inputs pre-normalised (z-score per channel,
statistics from the TRAIN split only). Write the converter from the official TS-SatFire tensors to this layout
(Phase 1) - everything downstream (training, corruption, evaluation) only depends on this contract."""
import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class NpzEventDataset(Dataset):
    def __init__(self, root, event_ids=None, years=None):
        files = sorted(glob.glob(os.path.join(root, "*.npz")))
        self.files = []
        for f in files:
            z = np.load(f, allow_pickle=True)
            if event_ids is not None and int(z["event_id"]) not in event_ids:
                continue
            if years is not None and int(z["year"]) not in years:
                continue
            self.files.append(f)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        z = np.load(self.files[i], allow_pickle=True)
        d = {k: torch.from_numpy(z[k]).float() for k in ("sat", "weather", "terrain", "fire", "target")}
        d["event_id"] = int(z["event_id"])
        return d
