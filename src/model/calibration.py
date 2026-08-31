"""Post-hoc calibration of the displacement head. No retraining.

Diagnosis: the model regresses toward the training mean. On test's slow
windows it predicts ~12.4 m against a 1.9 m truth; on validate's fast windows
it under-predicts by a comparable margin in the other direction. Both are the
same shrinkage, and both show as a consistently signed CAE.

A single global scale cannot fix this -- shrinkage is not a constant factor,
it is a squashing of the whole range toward the centre. So the map is
piecewise-linear in the PREDICTED value, fitted on train:

    for each quantile bin of mu, the mean true label in that bin

Fitted on the prediction (not the truth), because at inference the truth is
what we are trying to recover. Monotonicity is enforced so the correction can
never reorder two predictions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULT_BINS = 20


class PiecewiseCalibrator:
    """Monotone piecewise-linear map from predicted to calibrated displacement."""

    def __init__(self, knots_x=None, knots_y=None):
        self.knots_x = None if knots_x is None else np.asarray(knots_x, float)
        self.knots_y = None if knots_y is None else np.asarray(knots_y, float)

    def fit(self, mu: np.ndarray, y: np.ndarray, bins: int = DEFAULT_BINS):
        mu = np.asarray(mu, float)
        y = np.asarray(y, float)
        ok = np.isfinite(mu) & np.isfinite(y)
        mu, y = mu[ok], y[ok]
        # Quantile edges so every bin holds a comparable number of points;
        # uniform-width bins would leave the sparse high-speed tail unfitted.
        qs = np.linspace(0, 100, bins + 1)
        edges = np.unique(np.percentile(mu, qs))
        if edges.size < 3:
            self.knots_x = np.array([mu.min(), mu.max()])
            self.knots_y = np.array([y.mean(), y.mean()])
            return self
        idx = np.clip(np.digitize(mu, edges[1:-1], right=False), 0,
                      edges.size - 2)
        xs, ys = [], []
        for b in range(edges.size - 1):
            m = idx == b
            if m.sum() < 20:
                continue
            xs.append(float(np.mean(mu[m])))
            ys.append(float(np.mean(y[m])))
        if len(xs) < 2:
            self.knots_x = np.array([mu.min(), mu.max()])
            self.knots_y = np.array([y.mean(), y.mean()])
            return self
        xs = np.asarray(xs)
        ys = np.maximum.accumulate(np.asarray(ys))   # enforce monotonicity
        self.knots_x, self.knots_y = xs, ys
        return self

    def __call__(self, mu: np.ndarray) -> np.ndarray:
        if self.knots_x is None:
            return np.asarray(mu, float)
        # Linear inside the fitted range; clamped (not extrapolated) outside,
        # since extrapolating a fit made from bin means invents values the
        # training data never supported.
        return np.maximum(np.interp(np.asarray(mu, float),
                                    self.knots_x, self.knots_y), 0.0)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(
            {"knots_x": self.knots_x.tolist(),
             "knots_y": self.knots_y.tolist()}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "PiecewiseCalibrator":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["knots_x"], d["knots_y"])


def fit_on_train(model, stats, device: str = "mps",
                 bins: int = DEFAULT_BINS, zupt: bool = True):
    """Fit the calibrator on TRAIN predictions only."""
    import torch
    from data.windows import build_all

    built, _ = build_all(roles=("train",))
    X = np.concatenate([b.X for b in built])
    y = np.concatenate([b.y for b in built])
    mean = np.asarray(stats["mean"], np.float32)
    sd = np.asarray(stats["std"], np.float32)

    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), 1024):
            blk = (X[i:i + 1024] - mean) / sd
            t = torch.from_numpy(np.ascontiguousarray(
                blk.transpose(0, 2, 1).astype(np.float32))).to(device)
            out = model(t)
            mu = out["mu"].float().cpu().numpy()
            if zupt:
                mu = mu * (1.0 - torch.sigmoid(
                    out["stationary_logit"]).float().cpu().numpy())
            preds.append(mu)
    return PiecewiseCalibrator().fit(np.concatenate(preds), y, bins=bins), \
        np.concatenate(preds), y
