# RASE-Fire
Reliability-Aware Spatiotemporal Evidence Fusion for Wildfire Forecasting.
Hypothesis: a multimodal forecaster that explicitly estimates *modality reliability* and *predictive uncertainty* degrades
more gracefully than conventional fusion when observations are missing, noisy, stale, spatially corrupted, biased or shifted.

## Status (read this first)
| Piece | State |
|---|---|
| Model (encoders -> reliability fields -> spatial evidence arbitration -> decoder + uncertainty) | implemented, tested |
| Corruption engine (missing / noise / spatial / temporal / bias / combined; seeded, per-pixel severity map) | implemented, tested |
| Losses (forecast, reliability supervision, Brier calibration, consistency) | implemented, tested |
| Baselines: concat U-Net, ConvLSTM, attention fusion, modality dropout | implemented |
| Baseline UTAE | **stub** - plug in the authors' TS-SatFire implementation |
| Metrics (IoU, Dice, boundary F1, Hausdorff, ECE, Brier, robustness AUC, reliability-corruption corr., risk-coverage/AURC) | implemented, tested |
| Event-level bootstrap CIs + paired bootstrap | implemented |
| Synthetic dataset | smoke-test only, **not evidence for the paper** |
| **TS-SatFire / CanadaFireSat adapter** | **contract only** (`data/npz_dataset.py`); you write the converter (Phase 1) |

Nothing here has been run on real wildfire data. Numbers on the synthetic set only prove the code runs.

## Layout
`rasefire/{corruption,models,baselines,training,evaluation,visualization,data}` - `configs/` (base + baselines + `ablation/m1..m5`) -
`train.py`, `evaluate.py`, `tests/`.

## Quick start
    pip install -r requirements.txt
    python -m pytest tests -q
    python train.py --config configs/rase_full.yaml --set train.epochs=15 seed=0
    python train.py --config configs/concat_unet.yaml
    python evaluate.py --ckpt rase=runs/rase_full_s0/best.pt concat=runs/concat_unet_s0/best.pt --out results

## Batch contract
`sat (B,T,Cs,H,W)  weather (B,T,Cw,H,W)  terrain (B,Ct,H,W)  fire (B,T,1,H,W)  target (B,1,H,W)  event_id (B,)`.
Any model returning `{"prob": (B,1,H,W), "unc": ...}` (use `models.common.finalize`) plugs into training/evaluation.

## Design notes
* Reliability `r_m(x,y)` in [0,1] per modality; `alpha = softmax_m(r/tau)`; `F = sum_m alpha_m * r_m * F_m` (the `r` gate lets evidence
  shrink when *all* sources are bad). Reliability head sees own features + mean of other modalities (cross-modal consistency).
* Reliability supervision: `L_rel = ||r_m - (1 - c_m)||^2`, `c_m` = known per-pixel corruption from the corruption engine. The corruption
  is *not* given to the model as a mask.
* Uncertainty: heteroscedastic logit-noise head (MC-BCE in training, probit approximation at inference). Reliability (input trust) and
  predictive uncertainty (output doubt) are separate outputs. Selective prediction ranks pixels by entropy or entropy+(1-mean r).
* Corruption training modes: `none`, `dropout` (whole-modality zeroing = Baseline E), `aware` (all kinds, severity ~ U(0,1)).
* Ablation ladder (`configs/ablation`): m1 plain fusion -> m2 +global reliability -> m3 +supervision(+corruption-aware training)
  -> m4 +spatial -> m5 +uncertainty -> m6 selective prediction (evaluation-time, `risk_coverage`). `m3b`: corruption-aware without `L_rel`
  (isolates "just dropout-like training" vs. calibrated reliability).
* Statistics: bootstrap over **fires** (`evaluation/statistics.py`), never pixels. Run >=3-5 seeds (`--set seed=k`) and pool.

## Roadmap (your plan's phases)
1. **Data (blocking):** convert official TS-SatFire preprocessing output to the `.npz` contract; z-score stats from train only; split by
   year + held-out regions (`data/splits.py`); 2021 = final test, frozen. Reproduce one published baseline before touching RASE.
2. Plug in official UTAE; reproduce clean benchmark table.
3. Tune corruption *ranges* for realism (satellite radiometric model, weather bias magnitudes) - currently defensible defaults, not calibrated
   to real sensor failures. Document as a limitation.
4. Robustness campaign (`evaluate.py`), OOD (held-out region / 2021), CanadaFireSat external study (separate formulation, separate table).
5. Do NOT tune on test; do not tune until the curve looks good.

## Known limitations / honest caveats
* Time is folded into channels in the encoders (simple); a temporal attention/UTAE-style encoder is a natural upgrade.
* Weather is assumed on the pixel grid (broadcast scalars if your source is not gridded).
* Missing data is filled with 0 (normalised mean), which is indistinguishable from a real value of 0; this is deliberate (no mask leak) but
  makes "missing" partly a noise problem - report it.
* Temporal staleness for `fire` replaces history with older frames; for `terrain` it is a no-op.
