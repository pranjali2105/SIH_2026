"""Two-phase training for the displacement model.

Phase 1  ~5 epochs, plain MSE on the displacement head only. The Gaussian NLL
         can reduce its loss by inflating sigma before mu is anywhere near
         right, so the variance head is held out until the mean is roughly
         calibrated.
Phase 2  ~50 epochs on the full multi-task loss, cosine-decayed.

Early stopping is on **validation 60-second drift through the outage harness**,
not displacement MAE. A model can improve windowed MAE while getting worse at
the thing we actually need, because per-window errors that are individually
small but consistently signed accumulate over an outage. Drift is the quantity
the project exists to reduce, so it is the quantity we stop on.

Run:  python -m train --epochs 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.windows import CHANNELS, build_all, fit_normalisation, verify_no_leakage
from eval.buckets import BUCKET_NAMES, bucketed_mae, early_stopping_metric, pooled_mae
from model.losses import compute_pos_weight, multitask_loss
from model.resnet1d import ResNet1D

REPO_ROOT = Path(__file__).resolve().parents[1]
import os
OUT_DIR = REPO_ROOT / "results" / os.environ.get("TRAIN_OUT", "training")
CKPT_PATH = OUT_DIR / "best.pt"
CKPT_MA3_PATH = OUT_DIR / "best_ma3.pt"
LOG_PATH = OUT_DIR / "train_log.jsonl"
REPORT_PATH = OUT_DIR / "train_report.md"


@dataclass
class Config:
    warmup_epochs: int = 5
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 128
    seed: int = 0
    # Augmentation
    yaw_rotation: bool = True
    tilt_deg: float = 5.0
    noise_accel: float = 0.05        # m/s^2, per sample
    noise_gyro: float = 0.005        # rad/s
    bias_accel: float = 0.10         # m/s^2, constant across a window
    bias_gyro: float = 0.01          # rad/s
    drift_outage_s: float = 60.0
    drift_max_sessions: int = 7
    # Drift evaluation runs the model autoregressively, one forward pass per
    # simulated second, so a full validation sweep is ~36k single-sample
    # forwards per epoch. Subsampled to keep per-epoch cost bounded; the
    # sample is FIXED across epochs so the stopping signal is comparable.
    drift_max_outages: int = 90
    bucket_weighting: bool = False
    widths: tuple = (64, 128, 256, 512)
    window_samples: int = 100
    arch: str = "resnet"          # "resnet" | "tcn"
    stem_width: int = 64
    # Channel indices to suppress in the input, by position in
    # data.windows.CHANNELS = (acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z).
    # Index 2 is the vertical accelerometer -- the vibration-texture cue whose
    # relationship to speed INVERTS on validate (findings.md 7b).
    drop_channels: tuple = ()
    drop_scale: float = 0.0       # 0 = zero out; e.g. 0.25 = down-weight
    device: str = "cpu"


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------

class WindowDataset(Dataset):
    """Windows from one split, normalised, optionally augmented.

    Augmentation is applied in the LEVEL FRAME, which is what makes it valid:
    after levelling, the only unresolved mounting freedom is yaw about
    vertical, so a random yaw rotation generates a genuinely plausible
    alternative mounting rather than an impossible one.
    """

    def __init__(self, built, stats: dict, cfg: Config, augment: bool = False):
        self.X = np.concatenate([b.X for b in built]).astype(np.float32)
        self.y = np.concatenate([b.y for b in built]).astype(np.float32)
        self.stationary = np.concatenate(
            [b.is_stationary for b in built]).astype(np.float32)
        self.yaw = np.concatenate([b.yaw_rate_true for b in built]).astype(np.float32)
        self.t0 = np.concatenate([b.t0 for b in built]).astype(np.float32)
        sid = []
        for i, b in enumerate(built):
            sid.extend([i] * b.X.shape[0])
        self.session_id = np.asarray(sid, dtype=np.int64)
        self.session_names = [b.session for b in built]

        if cfg.bucket_weighting:
            from eval.buckets import inverse_frequency_weights
            self.weight, self.weight_detail = inverse_frequency_weights(self.y)
            self.weight = self.weight.astype(np.float32)
        else:
            self.weight = np.ones_like(self.y, dtype=np.float32)
            self.weight_detail = {}

        self.mean = np.asarray(stats["mean"], dtype=np.float32)
        self.std = np.asarray(stats["std"], dtype=np.float32)
        self.cfg = cfg
        self.augment = augment
        self.rng = np.random.default_rng(cfg.seed)

    def __len__(self) -> int:
        return self.X.shape[0]

    def _augment(self, w: np.ndarray) -> np.ndarray:
        """w is (T, 6): accel xyz then gyro xyz, in the level frame."""
        cfg = self.cfg
        acc, gyr = w[:, :3].copy(), w[:, 3:].copy()

        if cfg.yaw_rotation:
            th = self.rng.uniform(0, 2 * np.pi)
            c, s = np.cos(th), np.sin(th)
            R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
                         dtype=np.float32)
            acc, gyr = acc @ R.T, gyr @ R.T

        if cfg.tilt_deg > 0:
            # Small rotation about a random horizontal axis: the levelling is
            # only as good as the gravity estimate, so the model should not
            # assume it is exact.
            a = np.radians(self.rng.uniform(-cfg.tilt_deg, cfg.tilt_deg))
            b = np.radians(self.rng.uniform(-cfg.tilt_deg, cfg.tilt_deg))
            ca, sa, cb, sb = np.cos(a), np.sin(a), np.cos(b), np.sin(b)
            Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=np.float32)
            Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]], dtype=np.float32)
            R = Ry @ Rx
            acc, gyr = acc @ R.T, gyr @ R.T

        # Bias and noise matter most: test is a different phone with a
        # different noise floor and a different resting offset.
        if cfg.bias_accel > 0:
            acc = acc + self.rng.normal(0, cfg.bias_accel, size=(1, 3)).astype(np.float32)
        if cfg.bias_gyro > 0:
            gyr = gyr + self.rng.normal(0, cfg.bias_gyro, size=(1, 3)).astype(np.float32)
        if cfg.noise_accel > 0:
            acc = acc + self.rng.normal(0, cfg.noise_accel, acc.shape).astype(np.float32)
        if cfg.noise_gyro > 0:
            gyr = gyr + self.rng.normal(0, cfg.noise_gyro, gyr.shape).astype(np.float32)
        return np.concatenate([acc, gyr], axis=1)

    def __getitem__(self, i: int):
        w = self.X[i]
        if self.augment:
            w = self._augment(w)
        w = (w - self.mean) / self.std
        if self.cfg.drop_channels:
            w = w.copy()
            for c in self.cfg.drop_channels:
                w[:, c] *= self.cfg.drop_scale
        return {
            "x": torch.from_numpy(np.ascontiguousarray(w.T)),   # (6, T)
            "displacement": torch.tensor(self.y[i]),
            "is_stationary": torch.tensor(self.stationary[i]),
            "yaw_rate": torch.tensor(self.yaw[i]),
            "session_id": torch.tensor(self.session_id[i]),
            "t0": torch.tensor(self.t0[i]),
            "weight": torch.tensor(self.weight[i]),
        }


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device: str) -> dict:
    model.eval()
    ys, ps = [], []
    for batch in loader:
        out = model(batch["x"].to(device))
        ys.append(batch["displacement"].numpy())
        ps.append(out["mu"].cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    buckets = bucketed_mae(y, p)
    res = {"pooled_mae": pooled_mae(y, p),
           "bucket_mean_mae": early_stopping_metric(y, p)}
    for _, b in buckets.iterrows():
        res[f"mae_{b.bucket}"] = float(b.mae) if np.isfinite(b.mae) else None
    return res


class ModelPredictor:
    """Adapts the network to the harness `predictor` interface.

    Runs over an outage without ever seeing GPS after t0, exactly as in
    deployment.

    ZUPT gating. The stationary head is applied as a SOFT gate:

        mu_eff  = mu  * (1 - sigmoid(stationary_logit))
        yaw_eff = yaw * (1 - sigmoid(stationary_logit))

    Soft rather than thresholded so it degrades gracefully near the decision
    boundary instead of flipping discontinuously. The yaw gate matters as much
    as the displacement one: a stopped vehicle is not rotating, and gyro drift
    integrated while stationary corrupts heading for the REST of the outage,
    long after the vehicle moves off.

    Batching. The network reads only raw sensor windows -- no predicted value
    is ever fed back into its input -- so mu and yaw for every step of every
    outage are independent and can be evaluated in one batch. Only the heading
    accumulation is serial, and that is a cumulative sum applied afterwards.
    """

    name = "model"

    def __init__(self, model, session, stats, device="cpu", zupt: bool = True,
                 batch_size: int = 1024, calibration=None,
                 heading_source: str = "model",
                 conjunction_zupt: bool = False,
                 window_samples: int | None = None,
                 drop_channels: tuple = (), drop_scale: float = 0.0):
        """heading_source: "model" uses the yaw head; "gyro" integrates the raw
        levelled gyro z-axis instead, which ablates the head entirely."""
        from data.windows import GRID_HZ, WINDOW_SAMPLES, levelled_channels
        from data.sanity import gps_cumulative_distance
        import pandas as pd
        from data.loader import _find

        self.model, self.device = model, device
        self.zupt, self.batch_size = zupt, batch_size
        self.heading_source = heading_source
        # The stationary head alone fires on 11.2% of validate windows against
        # a 3.4% base rate (precision 0.288) -- pos_weight=16.9 overshot.
        # Requiring agreement with three independent physical signals brings
        # it to 4.1% at precision 0.661, and to 0.70% on motorway windows.
        self.conjunction_zupt = conjunction_zupt
        # Must match training exactly, or the model sees a channel at
        # inference that it never saw during training.
        self.drop_channels = tuple(drop_channels)
        self.drop_scale = float(drop_scale)
        self.calibration = calibration
        self.mean = np.asarray(stats["mean"], dtype=np.float32)
        self.std = np.asarray(stats["std"], dtype=np.float32)
        self.hz = GRID_HZ
        self.win = int(window_samples or WINDOW_SAMPLES)
        gps = gps_cumulative_distance(session)
        lo, hi = float(gps[0].min()), float(gps[0].max())
        self.grid = np.arange(lo, hi, 1.0 / GRID_HZ)
        self.chan = levelled_channels(session, self.grid)
        col = _find(session.df, r"^HEADING") or _find(session.df, r"GPS ORIENTATION")
        self.heading = (np.radians(np.interp(
            self.grid,
            pd.to_numeric(session.df["time_s"], errors="coerce").to_numpy(float),
            np.nan_to_num(pd.to_numeric(session.df[col], errors="coerce")
                          .to_numpy(float))))
            if col is not None else np.zeros_like(self.grid))

    def _window_starts(self, t0: float, n: int):
        """Window start indices for each second of one outage, and validity."""
        ends = np.searchsorted(self.grid, t0 + np.arange(n, dtype=float))
        starts = ends - self.win
        valid = (starts >= 0) & (ends <= (0 if self.chan is None
                                          else self.chan.shape[0]))
        return starts, ends, valid

    @torch.no_grad()
    def _run(self, starts, valid) -> tuple[np.ndarray, np.ndarray]:
        """Batched forward pass over arbitrary window starts."""
        shape = starts.shape
        flat_starts = starts.reshape(-1)
        flat_valid = valid.reshape(-1)
        mu = np.zeros(flat_starts.shape, dtype=np.float64)
        yaw = np.zeros(flat_starts.shape, dtype=np.float64)
        idx = np.flatnonzero(flat_valid)
        if idx.size == 0 or self.chan is None:
            return mu.reshape(shape), yaw.reshape(shape)
        for lo in range(0, idx.size, self.batch_size):
            sel = idx[lo:lo + self.batch_size]
            block = np.stack([self.chan[s:s + self.win] for s in flat_starts[sel]])
            block = (block - self.mean) / self.std
            if self.drop_channels:
                for c in self.drop_channels:
                    block[:, :, c] *= self.drop_scale
            x = torch.from_numpy(
                np.ascontiguousarray(block.transpose(0, 2, 1).astype(np.float32)))
            out = self.model(x.to(self.device))
            m = out["mu"].float().cpu().numpy()
            yw = out["yaw_rate"].float().cpu().numpy()
            if self.zupt:
                p_stat = torch.sigmoid(
                    out["stationary_logit"]).float().cpu().numpy()
                if self.conjunction_zupt:
                    blk = np.stack([self.chan[s:s + self.win]
                                    for s in flat_starts[sel]])
                    gyro_mag = np.abs(blk[:, :, 5]).mean(axis=1)
                    acc_var = blk[:, :, :3].std(axis=1).mean(axis=1)
                    stationary = ((p_stat > 0.5) & (m < 1.0)
                                  & (gyro_mag < 0.02) & (acc_var < 0.35))
                    moving = np.where(stationary, 0.0, 1.0)
                else:
                    moving = 1.0 - p_stat
                m = m * moving
                yw = yw * moving
            if self.calibration is not None:
                m = self.calibration(m)
            if self.heading_source == "gyro":
                # Raw levelled gyro z (rad/s) averaged over the window's last
                # second -- the vertical axis after levelling IS the yaw rate.
                step = int(self.hz)
                yw = np.array([
                    float(np.mean(self.chan[s + self.win - step:s + self.win, 5]))
                    for s in flat_starts[sel]])
            mu[sel] = m
            yaw[sel] = yw
        return mu.reshape(shape), yaw.reshape(shape)

    def predict(self, t0: float, duration: float) -> dict:
        return self.predict_many([t0], duration)[0]

    def predict_many(self, t0s, duration: float) -> list[dict]:
        """All steps of all given outages in one batched pass.

        Identical results to looping -- the only serial part, heading, is a
        cumulative sum performed after the forward passes.
        """
        n = int(round(duration))
        t0s = np.asarray(t0s, dtype=float)
        starts = np.empty((t0s.size, n), dtype=np.int64)
        valid = np.zeros((t0s.size, n), dtype=bool)
        for i, t0 in enumerate(t0s):
            st, _, va = self._window_starts(t0, n)
            starts[i], valid[i] = st, va

        mu, yaw = self._run(starts, valid)

        h0 = np.interp(t0s, self.grid, self.heading)
        # Heading integrates the (gated) yaw rate; a stationary step
        # contributes nothing, which is the point of gating it.
        headings = h0[:, None] + np.cumsum(yaw, axis=1)
        # Invalid steps hold the previous displacement rather than inventing one.
        for i in range(t0s.size):
            for k in range(n):
                if not valid[i, k]:
                    mu[i, k] = mu[i, k - 1] if k else 0.0
        return [{"displacements": mu[i], "headings": headings[i]}
                for i in range(t0s.size)]


_DRIFT_PLAN: list | None = None


def _drift_plan(sessions, cfg: Config) -> list:
    """Fixed (session, t0) sample, chosen once and reused every epoch.

    A resampled set would make the stopping signal noisy for reasons unrelated
    to the model, so the same outages are scored each time.
    """
    global _DRIFT_PLAN
    if _DRIFT_PLAN is not None:
        return _DRIFT_PLAN
    from data.loader import load_session
    from eval.harness import iter_outages

    plan = []
    for name in sessions[: cfg.drift_max_sessions]:
        try:
            s = load_session(name, check_rate=False)
            starts = list(iter_outages(s, cfg.drift_outage_s))
        except Exception:
            continue
        per = max(1, cfg.drift_max_outages // max(len(sessions), 1))
        if len(starts) > per:
            idx = np.linspace(0, len(starts) - 1, per).astype(int)
            starts = [starts[i] for i in idx]
        plan.append((name, starts))
    _DRIFT_PLAN = plan
    return plan


def validation_drift(model, sessions, stats, cfg: Config) -> tuple[float, float, int]:
    """Mean 60 s drift over a fixed sample of validation outages.

    This — not displacement MAE — is the early-stopping metric. Per-window
    errors that are small but consistently signed accumulate into large drift,
    so windowed MAE can improve while the deployable quantity gets worse.
    """
    from data.loader import load_session
    from eval.harness import OutageSkipped, run_outage

    model.eval()
    drifts = []
    for name, starts in _drift_plan(sessions, cfg):
        try:
            s = load_session(name, check_rate=False)
            pred = ModelPredictor(model, s, stats, cfg.device,
                                  window_samples=cfg.window_samples,
                                  drop_channels=cfg.drop_channels,
                                  drop_scale=cfg.drop_scale)
        except Exception as exc:
            print(f"    drift eval failed on {name}: {exc}", file=sys.stderr)
            continue
        for t0 in starts:
            try:
                drifts.append(run_outage(s, t0, cfg.drift_outage_s,
                                         pred).metrics()["crse"])
            except OutageSkipped:
                continue
    if not drifts:
        return float("nan"), float("nan"), 0
    d = np.asarray(drifts, dtype=float)
    # Standard error of the mean over the sampled outages. Logged so the
    # stopping signal can be checked against its own noise: if SE is comparable
    # to the epoch-to-epoch change in mean drift, "best epoch" is arbitrary.
    se = float(d.std(ddof=1) / np.sqrt(d.size)) if d.size > 1 else float("nan")
    return float(d.mean()), se, int(d.size)


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=Config.epochs)
    ap.add_argument("--warmup-epochs", type=int, default=Config.warmup_epochs)
    ap.add_argument("--batch-size", type=int, default=Config.batch_size)
    ap.add_argument("--lr", type=float, default=Config.lr)
    default_device = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--device", default=default_device)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--bucket-weighting", action="store_true",
                    help="inverse-frequency weighting of the displacement loss")
    ap.add_argument("--widths", default="64,128,256,512",
                    help="per-stage channel widths")
    ap.add_argument("--window-samples", type=int, default=100,
                    help="input window length in samples at 10 Hz")
    ap.add_argument("--arch", default="resnet", choices=("resnet", "tcn"))
    ap.add_argument("--stem-width", type=int, default=64)
    ap.add_argument("--drop-channels", default="",
                    help="comma-separated channel indices to suppress, "
                         "e.g. '2' for the vertical accelerometer")
    ap.add_argument("--drop-scale", type=float, default=0.0,
                    help="scale applied to dropped channels (0 = zero out)")
    ap.add_argument("--limit-train-batches", type=int, default=0,
                    help="debug: cap batches per epoch")
    args = ap.parse_args(argv)

    cfg = Config(epochs=args.epochs, warmup_epochs=args.warmup_epochs,
                 batch_size=args.batch_size, lr=args.lr, device=args.device,
                 bucket_weighting=args.bucket_weighting,
                 widths=tuple(int(w) for w in args.widths.split(",")),
                 window_samples=args.window_samples,
                 arch=args.arch, stem_width=args.stem_width,
                 drop_channels=tuple(int(c) for c in args.drop_channels.split(",")
                                     if c.strip()),
                 drop_scale=args.drop_scale)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print("building windows...", file=sys.stderr)
    built, failures = build_all(window_samples=cfg.window_samples)
    verify_no_leakage(built, expected_samples=cfg.window_samples)
    train_b = [b for b in built if b.role == "train"]
    val_b = [b for b in built if b.role == "validate"]
    stats = fit_normalisation(train_b)

    train_ds = WindowDataset(train_b, stats, cfg, augment=not args.no_augment)
    val_ds = WindowDataset(val_b, stats, cfg, augment=False)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    pos_weight = compute_pos_weight(train_ds.stationary)
    print(f"train {len(train_ds)} windows, validate {len(val_ds)}; "
          f"stationary pos_weight {pos_weight:.1f}", file=sys.stderr)
    if train_ds.weight_detail:
        print("inverse-frequency bucket weights:", file=sys.stderr)
        for b, d in train_ds.weight_detail.items():
            print(f"  {b:<7} n={d['n']:<7} weight {d['weight']:.3f}",
                  file=sys.stderr)

    if cfg.arch == "tcn":
        from model.tcn_model import TCNModel
        model = TCNModel(stem_width=cfg.stem_width,
                         channels=cfg.widths).to(cfg.device)
    else:
        model = ResNet1D(widths=cfg.widths).to(cfg.device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"widths {cfg.widths}, window {cfg.window_samples} samples "
          f"({cfg.window_samples / 10:g} s), {n_par:,} parameters "
          f"[{cfg.arch}]",
          file=sys.stderr)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    total_epochs = cfg.warmup_epochs + cfg.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    val_names = sorted({b.session for b in val_b})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_PATH.open("w", encoding="utf-8")
    best_drift, best_epoch = float("inf"), -1
    best_ma3, best_ma3_epoch = float("inf"), -1
    drift_history: list[float] = []

    for epoch in range(total_epochs):
        warmup = epoch < cfg.warmup_epochs
        model.train()
        agg = {}
        t_start = time.time()
        for i, batch in enumerate(train_dl):
            if args.limit_train_batches and i >= args.limit_train_batches:
                break
            x = batch["x"].to(cfg.device)
            out = model(x)
            targets = {k: batch[k].to(cfg.device)
                       for k in ("displacement", "is_stationary", "yaw_rate",
                                 "session_id", "t0", "weight")}
            if warmup:
                # Phase 1: MSE on the mean only. The NLL would otherwise buy
                # loss reduction by inflating sigma before mu is calibrated.
                # Warm-up MSE carries the same weighting, or phase 1 would
                # establish the shrinkage that phase 2 then has to undo.
                w = targets["weight"]
                se = (out["mu"] - targets["displacement"]) ** 2
                loss = (se * w).sum() / w.sum().clamp_min(1e-8)
                parts = {"total": loss, "displacement": loss}
            else:
                parts = multitask_loss(out, targets, pos_weight=pos_weight)
                loss = parts["total"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v.detach().item()
        n_batches = max(i + 1, 1)
        train_losses = {k: v / n_batches for k, v in agg.items()}
        if not warmup:
            sched.step()

        val_metrics = evaluate(model, val_dl, cfg.device)
        drift, drift_se, n_outages = validation_drift(model, val_names, stats, cfg)
        drift_history.append(drift)
        # A 3-epoch moving average is the noise-robust alternative to
        # best-single-epoch; both are tracked so the choice can be made on
        # evidence after the run rather than assumed now.
        ma3 = (float(np.mean(drift_history[-3:]))
               if len(drift_history) >= 3 else float("nan"))

        record = {"epoch": epoch, "phase": "warmup" if warmup else "full",
                  "lr": opt.param_groups[0]["lr"],
                  "seconds": round(time.time() - t_start, 1),
                  "train": {k: round(v, 5) for k, v in train_losses.items()},
                  "val": {k: (round(v, 5) if isinstance(v, float) else v)
                          for k, v in val_metrics.items()},
                  "val_drift_60s_m": None if not np.isfinite(drift) else round(drift, 3),
                  "val_drift_60s_se_m": None if not np.isfinite(drift_se) else round(drift_se, 3),
                  "val_drift_60s_ma3_m": None if not np.isfinite(ma3) else round(ma3, 3),
                  "n_drift_outages": n_outages}
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()

        buckets = "  ".join(
            f"{b}:{val_metrics.get(f'mae_{b}') or float('nan'):.2f}"
            for b in BUCKET_NAMES)
        print(f"[{epoch:3d}] {'warm' if warmup else 'full'} "
              f"train {train_losses['total']:8.4f}  "
              f"val bucket-mean {val_metrics['bucket_mean_mae']:.3f}  "
              f"pooled {val_metrics['pooled_mae']:.3f}  "
              f"drift60 {drift:8.2f}+-{drift_se:6.2f}  [{buckets}]", flush=True)

        payload = {"model": model.state_dict(), "config": asdict(cfg),
                   "norm_stats": stats, "epoch": epoch,
                   "val_drift_60s_m": drift, "val_drift_60s_se_m": drift_se}
        if np.isfinite(drift) and drift < best_drift:
            best_drift, best_epoch = drift, epoch
            torch.save(payload, CKPT_PATH)
        if np.isfinite(ma3) and ma3 < best_ma3:
            best_ma3, best_ma3_epoch = ma3, epoch
            torch.save({**payload, "val_drift_60s_ma3_m": ma3}, CKPT_MA3_PATH)

    log_file.close()

    # Is the stopping signal above its own noise? Compare the estimator's
    # standard error against the typical epoch-to-epoch movement it is being
    # asked to resolve.
    hist = np.asarray([d for d in drift_history if np.isfinite(d)], dtype=float)
    deltas = np.abs(np.diff(hist)) if hist.size > 1 else np.asarray([np.nan])
    median_delta = float(np.nanmedian(deltas))
    last_se = drift_se
    noise_dominates = bool(np.isfinite(last_se) and np.isfinite(median_delta)
                           and last_se >= median_delta)

    print(f"\nbest single-epoch drift {best_drift:.2f} m (epoch {best_epoch})")
    print(f"best 3-epoch MA drift   {best_ma3:.2f} m (epoch {best_ma3_epoch})")
    print(f"drift SE {last_se:.2f} m vs median epoch-to-epoch delta "
          f"{median_delta:.2f} m -> "
          f"{'NOISE DOMINATES, use the MA3 checkpoint' if noise_dominates else 'signal exceeds noise, single-epoch is fine'}")

    REPORT_PATH.write_text(
        "# Training run\n\n"
        f"- best single-epoch validation 60 s drift: **{best_drift:.2f} m** "
        f"(epoch {best_epoch}) -> `{CKPT_PATH.name}`\n"
        f"- best 3-epoch moving-average drift: **{best_ma3:.2f} m** "
        f"(epoch {best_ma3_epoch}) -> `{CKPT_MA3_PATH.name}`\n"
        f"- drift standard error (last epoch, {n_outages} outages): "
        f"{last_se:.2f} m\n"
        f"- median epoch-to-epoch change in mean drift: {median_delta:.2f} m\n"
        f"- **{'Noise dominates' if noise_dominates else 'Signal exceeds noise'}**: "
        + ("SE is comparable to or larger than the epoch-to-epoch movement, so "
           "best-single-epoch selection is arbitrary; the MA3 checkpoint is the "
           "one to use.\n" if noise_dominates else
           "single-epoch selection resolves real differences.\n")
        + f"\nLog: `{LOG_PATH.name}`\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
