import torch
import torch.nn.functional as F


def dice_loss(prob, tgt, eps=1.0):
    inter = (prob * tgt).sum((1, 2, 3))
    return 1 - ((2 * inter + eps) / (prob.sum((1, 2, 3)) + tgt.sum((1, 2, 3)) + eps)).mean()


def forecast_loss(out, tgt, pos_weight=1.0, mc=4, dice_w=1.0):
    """BCE + Dice. With an uncertainty head, BCE is the MC expectation over logit noise ~ N(0, sigma^2) (Kendall & Gal)."""
    pw = torch.tensor(pos_weight, device=tgt.device)
    logit = out["logit"]
    if "log_sigma" in out:
        z = logit.unsqueeze(0) + out["log_sigma"].exp().unsqueeze(0) * torch.randn((mc,) + logit.shape, device=logit.device)
        bce = F.binary_cross_entropy_with_logits(z, tgt.unsqueeze(0).expand_as(z), pos_weight=pw)
    else:
        bce = F.binary_cross_entropy_with_logits(logit, tgt, pos_weight=pw)
    return bce + dice_w * dice_loss(torch.sigmoid(logit), tgt)


def reliability_loss(r, c):
    """L_rel = || r_m - (1 - c_m) ||^2 : reliability is grounded in the KNOWN corruption severity."""
    return F.mse_loss(r, 1 - c)


def calibration_loss(prob, tgt):
    """Brier score: a proper scoring rule that pushes probabilities towards observed frequencies."""
    return F.mse_loss(prob, tgt)


def consistency_loss(prob, prob_perturbed):
    return F.mse_loss(prob, prob_perturbed)


def compute_loss(out, batch, cmaps, lcfg, out_perturbed=None):
    parts = {"forecast": forecast_loss(out, batch["target"], lcfg.get("pos_weight", 1.0), lcfg.get("mc_samples", 4))}
    if lcfg.get("lambda_rel", 0) > 0 and cmaps is not None and out.get("r") is not None:
        parts["rel"] = lcfg["lambda_rel"] * reliability_loss(out["r"], cmaps)
    if lcfg.get("lambda_cal", 0) > 0:
        parts["cal"] = lcfg["lambda_cal"] * calibration_loss(out["prob"], batch["target"])
    if lcfg.get("lambda_cons", 0) > 0 and out_perturbed is not None:
        parts["cons"] = lcfg["lambda_cons"] * consistency_loss(out["prob"], out_perturbed["prob"])
    return sum(parts.values()), {k: float(v.detach()) for k, v in parts.items()}
