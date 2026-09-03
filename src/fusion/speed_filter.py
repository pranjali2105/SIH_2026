"""1-D Kalman filter fusing the displacement model's per-second speed
estimate with independently-integrated forward accelerometer signal, for the
along-road (map-matched) tracking path.

Distinct from `fusion.eskf.NonHolonomicESKF`, which fuses the same two kinds
of signal to estimate 2-D heading and gyro bias. This filter carries no
heading at all -- it exists for `mapmatch.predictor`'s along-road trackers,
whose whole point is to progress a SCALAR distance along a road polyline
without ever integrating a heading (see that module's docstring). Fusing
displacement there needs a scalar filter, not the ESKF's 5-state one.

Why this is not circular, and what it cannot do. Smoothing the model's own
mu with a filter whose only input IS mu would reduce variance and leave a
systematic bias exactly where it was -- and `results/mapmatch.md`'s own
finding is that `map_matched_model` carries a systematic +118 m over 60 s
bias (mean CAE 118.0, `mean_abs_cae` 261.9). The accelerometer is a
genuinely independent SECOND signal, but it only supplies a RATE (specific
force), not an absolute speed reference -- so it can correct bias that is
regime-dependent (which the real one is: CAE sign and size both vary sharply
by speed bucket in `results/final_scoring.md`, not a flat offset), because
the accelerometer's own error does not share that regime dependence. A
perfectly FLAT, constant model bias would not be meaningfully reduced by
this filter (verified below: `test_a_flat_constant_bias_is_not_fully_removed`
in `tests/test_speed_filter.py`), since both branches of the fusion would
carry it. How much of the REAL model's regime-dependent bias this actually
removes is an empirical question that needs the real model and a real
drive to answer; that is not available in this checkout (see the repo
README's "not verified" note).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SpeedFilterConfig:
    # Random-walk growth of the accelerometer-propagated speed's own
    # uncertainty, per second of NOT being corrected by the model. Mirrors
    # `eskf.py`'s `BIAS_RW` in spirit: the accelerometer path is trusted
    # more over the short term (it has no window-to-window jaggedness) and
    # less as its own uncorrected bias has longer to accumulate.
    accel_bias_rw: float = 0.05        # (m/s^2) per sqrt(s)
    init_var: float = 4.0              # (m/s)^2, initial speed uncertainty
    # The model's own predicted sigma is trusted only within this band --
    # a `logvar` head that has collapsed to near-zero variance (over-
    # confident) or exploded (declaring everything uncertain to dodge the
    # loss, guarded against in training but not guaranteed at inference on
    # an unseen device) should not be allowed to swing the filter to either
    # extreme.
    min_sigma_mu: float = 0.3          # m/s
    max_sigma_mu: float = 20.0         # m/s


def fuse_speed_sequence(mu: np.ndarray, sigma_mu: np.ndarray,
                        a_fwd: np.ndarray, dt: float = 1.0,
                        v0: float | None = None,
                        cfg: SpeedFilterConfig | None = None) -> np.ndarray:
    """Filter one outage's per-second (mu, sigma_mu, a_fwd) in one pass.

    `mu`: the displacement model's per-second output (metres over `dt` = 1 s,
    so numerically a speed in m/s).
    `sigma_mu`: the model's own predicted std, `exp(0.5 * logvar)`. Entries
    that are not finite fall back to `cfg.max_sigma_mu` (least trust) rather
    than raising, since a caller wrapping a model without an uncertainty
    head can legitimately pass an all-NaN array -- the filter then still
    runs, just leaning entirely on the accelerometer's short-term smoothing.
    `a_fwd`: forward specific force in the vehicle frame (m/s^2), e.g.
    `a_h1*cos(phi) + a_h2*sin(phi)` with a calibrated `phi` from
    `fusion.mount_calibration`, averaged over each second's IMU samples.
    Non-finite entries skip the predict step for that second (the filter
    holds its prior speed estimate rather than inventing an acceleration).

    Returns the filtered per-second speed/displacement sequence, the same
    shape as `mu` -- a drop-in replacement for `mu` in any of
    `mapmatch.predictor`'s trackers (`_displacements` / `_advance*`).
    """
    cfg = cfg or SpeedFilterConfig()
    mu = np.asarray(mu, dtype=float)
    sigma_mu = np.asarray(sigma_mu, dtype=float)
    a_fwd = np.asarray(a_fwd, dtype=float)
    n = mu.shape[0]
    if n == 0:
        return mu.copy()

    out = np.empty(n)
    v = float(mu[0]) if v0 is None else float(v0)
    var = float(cfg.init_var)

    for k in range(n):
        if k > 0 and np.isfinite(a_fwd[k]):
            v = v + float(a_fwd[k]) * dt
            var = var + (cfg.accel_bias_rw ** 2) * dt

        s = sigma_mu[k] if np.isfinite(sigma_mu[k]) else cfg.max_sigma_mu
        r = float(np.clip(s, cfg.min_sigma_mu, cfg.max_sigma_mu))
        rr = r * r

        innov = float(mu[k]) - v
        s_total = var + rr
        gain = var / s_total if s_total > 0 else 0.0
        v = v + gain * innov
        var = (1.0 - gain) * var
        out[k] = max(v, 0.0)           # a displacement/speed is non-negative

    return out
