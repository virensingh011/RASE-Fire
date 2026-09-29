import inspect
from .rase import RASEFire
from ..baselines import AttentionFusion, ConcatUNet, ConvLSTMNet, UTAE

_REGISTRY = {"rase": RASEFire, "concat_unet": ConcatUNet, "convlstm": ConvLSTMNet,
             "attention_fusion": AttentionFusion, "utae": UTAE}


def build_model(cfg):
    """Config inheritance merges dicts, so keys meant for other architectures are dropped by signature."""
    d, m = cfg["data"], dict(cfg["model"])
    cls = _REGISTRY[m.pop("name")]
    params = set(inspect.signature(cls.__init__).parameters) - {"self", "chans", "T"}
    if cls is not UTAE:
        m = {k: v for k, v in m.items() if k in params}
    chans = {"sat": d["sat_ch"], "weather": d["weather_ch"], "terrain": d["terrain_ch"], "fire": 1}
    return cls(chans, d["T"], **m)
