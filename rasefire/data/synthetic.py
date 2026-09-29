"""Synthetic wildfire-like data for smoke tests / method debugging ONLY (not a scientific benchmark).
Fire spreads stochastically as a function of fuel, wind direction, humidity; the satellite is a noisy observation of the
fire state; the fire-state modality is exact. Terrain and wind are fixed per event (so events can be split properly)."""
import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _smooth(g, H, W, k=7, iters=2):
    x = torch.randn(1, 1, H, W, generator=g)
    for _ in range(iters):
        x = F.avg_pool2d(F.pad(x, (k // 2,) * 4, mode="reflect"), k, 1)
    return ((x - x.mean()) / (x.std() + 1e-6))[0]


class SyntheticFireDataset(Dataset):
    def __init__(self, n=256, size=32, T=4, seed=0, per_event=8, event_offset=0, horizon=2, shift=False,
                 sat_ch=4, weather_ch=4, terrain_ch=3):
        assert sat_ch == 4 and weather_ch == 4 and terrain_ch == 3, "synthetic generator has fixed channel counts"
        self.n, self.size, self.T, self.seed = n, size, T, seed
        self.per_event, self.event_offset, self.horizon, self.shift = per_event, event_offset, horizon, shift

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        H = W = self.size
        T, ev = self.T, i // self.per_event + self.event_offset
        ge = torch.Generator().manual_seed(self.seed * 1000003 + ev)
        gs = torch.Generator().manual_seed(self.seed * 1000003 + 7919 * (i + 1) + 13)
        elev, fuel = _smooth(ge, H, W, 9), _smooth(ge, H, W, 5)
        if self.shift:
            fuel = 1.4 * fuel + 0.5
        gy, gx = torch.gradient(elev[0])
        slope = (gx ** 2 + gy ** 2).sqrt()
        slope = ((slope - slope.mean()) / (slope.std() + 1e-6)).unsqueeze(0)
        terrain = torch.cat([elev, slope, fuel], 0)
        ang = torch.rand(1, generator=ge).item() * 2 * math.pi
        dx, dy = int(round(math.cos(ang))), int(round(math.sin(ang)))
        w_amp = 1.0 + 0.5 * torch.rand(1, generator=ge).item()
        temp0, hum0 = _smooth(ge, H, W, 9), _smooth(ge, H, W, 9)
        weather = torch.stack([torch.cat([w_amp * math.cos(ang) * torch.ones(1, H, W) + 0.3 * _smooth(ge, H, W),
                                          w_amp * math.sin(ang) * torch.ones(1, H, W) + 0.3 * _smooth(ge, H, W),
                                          temp0 + 0.1 * t, -0.5 * temp0 + 0.5 * hum0 - 0.1 * t], 0) for t in range(T)])
        humid = weather[-1, 3:4]
        fire = torch.zeros(1, H, W)
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        for _ in range(1 + int(torch.randint(0, 3, (1,), generator=gs))):
            cy, cx = (int(torch.randint(4, H - 4, (1,), generator=gs)) for _ in range(2))
            fire = torch.maximum(fire, (((yy - cy) ** 2 + (xx - cx) ** 2) <= 4).float().unsqueeze(0))
        states = [fire]
        for _ in range(T - 1 + self.horizon):
            neigh = F.max_pool2d(fire[None], 3, 1, 1)[0]
            cand = (neigh - fire).clamp(0, 1)
            wind_c = (torch.roll(fire, (dy, dx), (-2, -1)) - fire).clamp(0, 1)
            p = torch.sigmoid(0.9 * fuel - 1.8 + 2.2 * wind_c - 0.6 * humid)
            fire = (fire + cand * (torch.rand(1, H, W, generator=gs) < p).float()).clamp(0, 1)
            states.append(fire)
        fs = torch.stack(states[:T])                                   # (T,1,H,W) observed fire history
        target = states[T - 1 + self.horizon]                          # (1,H,W)
        ns = 1.5 if self.shift else 1.0
        n = lambda: ns * 0.4 * torch.randn(T, 1, H, W, generator=gs)
        sat = torch.cat([2 * fs + n(), 1.5 * fs + 0.2 * fuel + n(), 0.5 * elev.expand(T, 1, H, W) + n(), 0.5 * n()], 1)
        return {"sat": sat, "weather": weather, "terrain": terrain, "fire": fs, "target": target,
                "event_id": ev}
