"""Model + non-holonomic ESKF, exposed through the harness predictor interface.

Slots in beside INS DR, constant-velocity DR and the raw model with no harness
changes: it exposes `predict` and `predict_many` and returns per-second
displacements and headings exactly as they do.

Per simulated second:
  1. the CNN supplies displacement mu and its sigma (from the log-variance
     head), plus P(stationary)
  2. the filter propagates heading on the de-biased gyro and position on mu
  3. the non-holonomic constraint updates gyro bias and mount yaw
  4. where the stationary head fires, a ZUPT makes the bias directly observable

The filter runs at the sensor rate INSIDE each second, not once per second:
the constraint is a per-sample physical fact, and the gyro bias is what we are
trying to observe.
"""

from __future__ import annotations

import numpy as np
import torch

from .eskf import ESKFConfig, NonHolonomicESKF


class ESKFPredictor:
    """Wraps a ModelPredictor with the non-holonomic filter."""

    name = "model_eskf"

    def __init__(self, inner, session, cfg: ESKFConfig | None = None,
                 sub_hz: int = 10, conjunction_zupt: bool = True,
                 update_hz: float = 10.0, calibrate_mount_yaw: bool = True,
                 mount_yaw_lookback_s: float = 180.0):
        """`inner` is a ModelPredictor, reused for its channel grid and model.

        `conjunction_zupt`: require agreement between the classifier and three
        independent physical signals before declaring the vehicle stopped. The
        head alone fires on 11.2% of validate windows against a 3.4% base rate
        (precision 0.288) -- pos_weight=16.9 overshot. On a motorway it fires
        6.8% of the time, and each false fire sets the gyro bias to the
        vehicle's actual turn rate, which is how the filter diverged.

        `calibrate_mount_yaw`: fit phi offline from GPS-available driving
        before each outage (`fusion.mount_calibration`) instead of leaving
        it at 0 and letting the filter chase it during the outage, which is
        the divergence findings.md §8 documents. Falls back to phi=0 (prior
        behaviour) when calibration is unavailable or the fit is not
        `.observable` -- a bad phi is worse than no phi.
        """
        self.inner = inner
        self.session = session
        self.cfg = cfg or ESKFConfig()
        self.sub = int(sub_hz)
        self.conjunction_zupt = conjunction_zupt
        self.calibrate_mount_yaw = calibrate_mount_yaw
        self.mount_yaw_lookback_s = mount_yaw_lookback_s
        self._phi_cache: dict[float, float] = {}
        # How often the NHC pseudo-measurement is APPLIED, as opposed to how
        # often the state is propagated. Applying it every sub-sample treats
        # 600 readings of one physical fact as 600 independent measurements,
        # which collapses the covariance and makes the filter chase noise.
        self.update_every = max(1, int(round(sub_hz / max(update_hz, 1e-6))))
        self.grid = inner.grid
        self.chan = inner.chan

    # -- mount-yaw calibration ----------------------------------------------

    def _phi0(self, t0: float) -> float:
        """Cached, per-t0 offline phi estimate (0.0 if unavailable/untrusted).

        Cached rather than computed once per session: mirrors
        `mapmatch.predictor._gyro_sign`'s own per-t0 cache, and keeps the
        "nothing at or after t0 is read" guarantee exact for every outage
        scored from this session, not just the first.
        """
        if not self.calibrate_mount_yaw:
            return 0.0
        key = round(float(t0), 3)
        if key in self._phi_cache:
            return self._phi_cache[key]
        phi = 0.0
        try:
            from fusion.mount_calibration import estimate_mount_yaw_for_session
            est = estimate_mount_yaw_for_session(
                self.session, t0, lookback_s=self.mount_yaw_lookback_s)
            if est is not None and est.observable:
                phi = est.phi_rad
        except Exception:                      # noqa: BLE001
            # Calibration is a refinement, not a dependency: any failure to
            # fit (missing columns, too little history, an absent `data`
            # package) must fall back to the prior phi=0 behaviour rather
            # than crash the predictor.
            phi = 0.0
        self._phi_cache[key] = phi
        return phi

    # -- per-second model outputs -----------------------------------------

    @torch.no_grad()
    def _model_outputs(self, starts, valid):
        """mu, sigma, P(stationary) for every step of every outage."""
        inner = self.inner
        shape = starts.shape
        fs, fv = starts.reshape(-1), valid.reshape(-1)
        mu = np.zeros(fs.shape)
        sigma = np.full(fs.shape, 5.0)
        p_stat = np.zeros(fs.shape)
        idx = np.flatnonzero(fv)
        if idx.size == 0 or self.chan is None:
            return (mu.reshape(shape), sigma.reshape(shape), p_stat.reshape(shape))
        for lo in range(0, idx.size, inner.batch_size):
            sel = idx[lo:lo + inner.batch_size]
            block = np.stack([self.chan[s:s + inner.win] for s in fs[sel]])
            block = (block - inner.mean) / inner.std
            x = torch.from_numpy(np.ascontiguousarray(
                block.transpose(0, 2, 1).astype(np.float32))).to(inner.device)
            out = inner.model(x)
            mu[sel] = out["mu"].float().cpu().numpy()
            # sigma from the log-variance head: the displacement measurement
            # noise the filter should use.
            sigma[sel] = np.exp(0.5 * out["logvar"].float().cpu().numpy())
            p_stat[sel] = torch.sigmoid(
                out["stationary_logit"]).float().cpu().numpy()
        return mu.reshape(shape), sigma.reshape(shape), p_stat.reshape(shape)

    # -- interface ---------------------------------------------------------

    def predict(self, t0: float, duration: float) -> dict:
        return self.predict_many([t0], duration)[0]

    def predict_many(self, t0s, duration: float) -> list[dict]:
        n = int(round(duration))
        t0s = np.asarray(t0s, dtype=float)
        starts = np.empty((t0s.size, n), dtype=np.int64)
        valid = np.zeros((t0s.size, n), dtype=bool)
        for i, t0 in enumerate(t0s):
            st, _, va = self.inner._window_starts(t0, n)
            starts[i], valid[i] = st, va

        mu, sigma, p_stat = self._model_outputs(starts, valid)
        h0 = np.interp(t0s, self.grid, self.inner.heading)

        dt = 1.0 / self.sub
        results = []
        for i in range(t0s.size):
            f = NonHolonomicESKF(float(h0[i]), self.cfg, phi0=self._phi0(t0s[i]))
            disps, heads = np.zeros(n), np.zeros(n)
            for k in range(n):
                d = float(mu[i, k]) if valid[i, k] else (disps[k - 1] if k else 0.0)
                sd = float(sigma[i, k]) if valid[i, k] else 5.0
                if self.conjunction_zupt:
                    # A moving vehicle fails the physical tests regardless of
                    # what the classifier says.
                    st_i, en_i = int(starts[i, k]), int(starts[i, k]) + self.inner.win
                    w = (self.chan[max(st_i, 0):en_i]
                         if self.chan is not None and en_i > 0
                         else np.zeros((1, 6)))
                    if w.shape[0] == 0:
                        w = np.zeros((1, 6))
                    gyro_mag = float(np.abs(w[:, 5]).mean())
                    acc_var = float(w[:, :3].std(axis=0).mean())
                    stationary = (p_stat[i, k] > 0.5
                                  and float(mu[i, k]) < 1.0
                                  and gyro_mag < 0.02
                                  and acc_var < 0.35)
                else:
                    stationary = p_stat[i, k] > 0.5
                if stationary:
                    d = 0.0
                speed = d                       # metres in one second

                end = int(starts[i, k]) + self.inner.win
                lo = max(end - self.sub, 0)
                seg = (self.chan[lo:end] if (self.chan is not None and end > 0)
                       else np.zeros((self.sub, 6)))
                if seg.shape[0] == 0:
                    seg = np.zeros((1, 6))

                # Sub-steps within the second: the constraint is a per-sample
                # fact and the bias is what we are trying to observe.
                per = d / max(seg.shape[0], 1)
                # dv/dt across this second, from the model's own displacements.
                prev_d = float(disps[k - 1]) if k else d
                dv_dt = d - prev_d
                resid = np.nan
                for j, row in enumerate(seg):
                    a_h1, a_h2, omega = float(row[0]), float(row[1]), float(row[5])
                    f.propagate(omega, per, dt, sd / max(seg.shape[0], 1))
                    if j % self.update_every:
                        continue
                    if stationary:
                        f.update_zupt(omega)
                    else:
                        resid = f.update_nhc(a_h1, a_h2, omega, speed)
                        f.update_forward(a_h1, a_h2, dv_dt)
                f.record(resid, stationary)
                disps[k] = d
                heads[k] = f.heading
            results.append({"displacements": disps, "headings": heads,
                            "trace": f.trace})
        return results
