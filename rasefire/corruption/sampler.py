import random
import torch
from .. import MODALITIES
from .apply import KINDS, CorruptionSpec, apply_corruptions


class CorruptionSampler:
    """Training-time corruption: per sample, each modality is corrupted with prob p (at most M-1 modalities per sample,
    so some evidence always survives). Returns the corrupted batch and cmaps for reliability supervision."""

    def __init__(self, p=0.5, kinds=KINDS, severity_levels=None, seed=0):
        self.p, self.kinds, self.levels = p, tuple(kinds), severity_levels
        self.rng = random.Random(seed)

    def _sev(self):
        return self.rng.choice(self.levels) if self.levels else self.rng.uniform(0.0, 1.0)

    def __call__(self, batch):
        B = batch["target"].shape[0]
        outs = {m: [] for m in MODALITIES}
        cs = []
        for b in range(B):
            sample = {k: (v[b:b + 1] if torch.is_tensor(v) else v) for k, v in batch.items()}
            order = list(MODALITIES)
            self.rng.shuffle(order)
            specs = []
            for m in order[:len(MODALITIES) - 1]:
                if self.rng.random() < self.p:
                    kinds = [k for k in self.kinds if not (k == "temporal" and m == "terrain")]
                    if kinds:
                        specs.append(CorruptionSpec(m, self.rng.choice(kinds), self._sev(), self.rng.randint(0, 2**31 - 1)))
            cb, c = apply_corruptions(sample, specs)
            for m in MODALITIES:
                outs[m].append(cb[m])
            cs.append(c)
        out = dict(batch)
        for m in MODALITIES:
            out[m] = torch.cat(outs[m], 0)
        return out, torch.cat(cs, 0)


def ModalityDropout(p=0.3, seed=0):
    """Baseline E: whole modalities are zeroed at random (no reliability information is exposed)."""
    return CorruptionSampler(p=p, kinds=("missing",), severity_levels=(1.0,), seed=seed)
