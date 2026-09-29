import math, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from rasefire import MODALITIES
from rasefire.config import load_config
from rasefire.corruption import CorruptionSpec, CorruptionSampler, apply_corruptions
from rasefire.data import SyntheticFireDataset
from rasefire.evaluation import bootstrap_ci, evaluate_loader, robustness_auc, per_event_mean
from rasefire.evaluation import metrics as M
from rasefire.models import build_model
from rasefire.training.losses import compute_loss
from torch.utils.data import DataLoader

CFG = os.path.join(os.path.dirname(__file__), "..", "configs")


def batch(n=4):
    return next(iter(DataLoader(SyntheticFireDataset(n=n, size=32), n)))


def test_dataset_shapes():
    b = batch()
    assert b["sat"].shape == (4, 4, 4, 32, 32) and b["terrain"].shape == (4, 3, 32, 32) and b["target"].shape == (4, 1, 32, 32)


def test_corruption_reproducible_and_bounded():
    b = batch()
    for kind in ("missing", "noise", "spatial", "temporal", "bias"):
        for mod in MODALITIES:
            sp = [CorruptionSpec(mod, kind, 0.5, 7)]
            o1, c1 = apply_corruptions(b, sp); o2, c2 = apply_corruptions(b, sp)
            assert torch.equal(o1[mod], o2[mod]) and torch.equal(c1, c2)
            assert c1.min() >= 0 and c1.max() <= 1 and o1[mod].shape == b[mod].shape
    o, c = apply_corruptions(b, [CorruptionSpec("sat", "missing", 1.0, 0)])
    assert o["sat"].abs().sum() == 0 and c[:, 0].min() == 1 and c[:, 1:].sum() == 0
    o, c = apply_corruptions(b, [CorruptionSpec("sat", "missing", 0.0, 0)])
    assert torch.equal(o["sat"], b["sat"]) and c.sum() == 0


def test_spatial_area_and_temporal_lag():
    b = batch()
    _, c = apply_corruptions(b, [CorruptionSpec("weather", "spatial", 0.25, 1)])
    assert abs(c[:, 1].mean().item() - 0.25) < 0.02
    o, _ = apply_corruptions(b, [CorruptionSpec("fire", "temporal", 1.0, 0)])
    assert torch.equal(o["fire"][:, -1], b["fire"][:, 0])


def test_sampler_leaves_evidence():
    s = CorruptionSampler(p=1.0, seed=0)
    for _ in range(3):
        _, c = s(batch(8))
        assert (c.amax((2, 3)) > 0).float().sum(1).max() <= len(MODALITIES) - 1


def test_all_models_forward_backward():
    for name in ("rase_full", "concat_unet", "convlstm", "attention_fusion", "modality_dropout"):
        cfg = load_config(os.path.join(CFG, f"{name}.yaml"))
        m = build_model(cfg)
        b = batch()
        out = m(b)
        assert out["prob"].shape == (4, 1, 32, 32) and torch.isfinite(out["prob"]).all()
        loss, parts = compute_loss(out, b, torch.zeros(4, 4, 32, 32), cfg["train"]["loss"])
        loss.backward()


def test_ablation_configs_build():
    d = os.path.join(CFG, "ablation")
    for f in sorted(os.listdir(d)):
        m = build_model(load_config(os.path.join(d, f)))
        out = m(batch())
        assert out["r"].shape[1] == 4


def test_rase_arbitration_weights_sum_to_one_and_reliability_supervision():
    cfg = load_config(os.path.join(CFG, "rase_full.yaml"))
    m = build_model(cfg); b = batch()
    out = m(b)
    assert torch.allclose(out["alpha"].sum(1), torch.ones(4, 32, 32), atol=1e-5)
    assert out["r"].min() >= 0 and out["r"].max() <= 1
    _, c = apply_corruptions(b, [CorruptionSpec("sat", "missing", 1.0, 0)])
    loss, parts = compute_loss(m(b), b, c, cfg["train"]["loss"])
    assert "rel" in parts


def test_metrics():
    t = torch.zeros(2, 1, 16, 16); t[:, :, 4:10, 4:10] = 1
    iou, dice = M.iou_dice(t, t)
    assert torch.allclose(iou, torch.ones(2), atol=1e-4) and M.hausdorff(t, t).max() == 0
    assert torch.allclose(M.boundary_f1(t, t), torch.ones(2), atol=1e-3)
    z = torch.zeros_like(t)
    assert M.iou_dice(z, z)[0].min() > 0.99 and M.iou_dice(z, t)[0].max() < 0.01
    cal = M.CalibrationAccumulator(); cal.update(t.clone().clamp(0.0, 0.999), t); assert cal.ece() < 0.01
    ca = M.CorrAccumulator(); x = torch.rand(1000); ca.update(x, 2 * x + 1); assert abs(ca.value() - 1) < 1e-6
    rc = M.RiskCoverageAccumulator(); rc.update(torch.rand(1000), torch.rand(1000), (torch.rand(1000) > .5).float())
    cov, risk, aurc = rc.curve(); assert cov[-1] == 1.0 and 0 <= aurc <= 1


def test_stats_and_auc():
    assert abs(robustness_auc([0, .5, 1], [1, 1, 1]) - 1) < 1e-9 and abs(robustness_auc([0, 1], [1, 0]) - 0.5) < 1e-9
    m, lo, hi = bootstrap_ci(np.random.RandomState(0).rand(30)); assert lo < m < hi
    assert list(per_event_mean([1, 3, 5], [0, 0, 1])) == [2.0, 5.0]


def test_evaluate_loader_runs():
    cfg = load_config(os.path.join(CFG, "rase_full.yaml"))
    m = build_model(cfg)
    ld = DataLoader(SyntheticFireDataset(n=16, size=32), 8)
    r = evaluate_loader(m, ld, torch.device("cpu"), specs=[CorruptionSpec("sat", "spatial", 0.5, 0)], hausdorff=True)
    assert len(r["per_sample"]["iou"]) == 16 and "rel_corr" in r and "entropy+reliability" in r["risk_coverage"]
