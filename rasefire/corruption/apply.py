from dataclasses import dataclass, asdict, replace
import torch
from .. import MODALITIES
from . import ops

KINDS = ("missing", "noise", "spatial", "temporal", "bias")


@dataclass(frozen=True)
class CorruptionSpec:
    """Fully reproducible description of one corruption."""
    modality: str
    kind: str
    severity: float
    seed: int = 0

    def to_dict(self):
        return asdict(self)

    def with_seed(self, seed):
        return replace(self, seed=seed)


def corrupt_tensor(x, spec):
    s = float(spec.severity)
    if s <= 0 or spec.kind not in KINDS:
        B, H, W = x.shape[0], x.shape[-2], x.shape[-1]
        return x, torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype)
    if spec.kind == "noise":
        return ops.noise(x, s, spec.seed, radiometric=(spec.modality == "sat"))
    return getattr(ops, spec.kind)(x, s, spec.seed)


def apply_corruptions(batch, specs):
    """Returns (corrupted_batch, cmaps) with cmaps (B, len(MODALITIES), H, W). Several specs may hit one modality."""
    out = dict(batch)
    B, _, H, W = batch["target"].shape
    dev = batch["target"].device
    cm = {m: torch.zeros(B, 1, H, W, device=dev) for m in MODALITIES}
    for sp in specs:
        out[sp.modality], c = corrupt_tensor(out[sp.modality], sp)
        cm[sp.modality] = torch.maximum(cm[sp.modality], c)
    return out, torch.cat([cm[m] for m in MODALITIES], 1)
