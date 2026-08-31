"""Classical strapdown INS dead reckoning, per Onyekpe et al. 2021 §2.1.

The reference baseline the learned model must beat, and the vehicle for the
Stage 0 known-answer test.

Pipeline
--------
1. Accelerometer bias: mean over the stationary recordings the dataset
   provides (speed == 0 stretches), removed from every sample.
2. Gravity: absent from the V- longitudinal/lateral channels, which are
   body-frame horizontal; for S- it is removed with the Gravity X/Y/Z columns.
3. Heading: integrated gyro yaw rate, initialised from GPS heading at the
   start of the outage.
4. Position: double integration in the navigation frame.

Units (see CLAUDE.md -- these are the classic failure points):
  * V- acceleration is in **g**            -> x 9.80665
  * V- yaw rate is in **deg/s**            -> x pi/180
  * V- speed is in **km/h**                -> the loader already divides by 3.6
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.loader import G_TO_MS2, Session, _find

DEG_TO_RAD = np.pi / 180.0

STATIONARY_SPEED_MS = 0.2     # "stationary" for bias estimation
MIN_STATIONARY_SAMPLES = 50


@dataclass
class INSState:
    """Navigation state carried through an outage."""
    x: float          # metres east of the outage start
    y: float          # metres north
    heading_rad: float
    speed_ms: float


class INSDeadReckoning:
    """Strapdown dead reckoning over a GNSS outage.

    Implements the `predictor` interface the harness expects:
    ``predict(session_data, t0, duration) -> array of per-second displacements``
    plus the per-second headings it integrated, so the harness can propagate
    position itself.
    """

    name = "ins_dr"

    def __init__(self, session: Session):
        self.session = session
        self.df = session.df
        self.t = pd.to_numeric(self.df["time_s"], errors="coerce").to_numpy(float)
        self.speed = pd.to_numeric(self.df.get("speed_ms"),
                                   errors="coerce").to_numpy(float)
        self.accel_long, self.yaw_rate = self._channels()
        self.accel_bias = self._estimate_bias()

    # -- channel extraction ------------------------------------------------

    def _channels(self) -> tuple[np.ndarray, np.ndarray]:
        df = self.df
        if self.session.family == "V":
            a_col = _find(df, r"LONGITUDINAL ACCEL")
            y_col = _find(df, r"YAW RATE")
            if a_col is None or y_col is None:
                raise ValueError("V- session lacks longitudinal accel or yaw rate")
            # g -> m/s^2, deg/s -> rad/s.
            a = pd.to_numeric(df[a_col], errors="coerce").to_numpy(float) * G_TO_MS2
            w = pd.to_numeric(df[y_col], errors="coerce").to_numpy(float) * DEG_TO_RAD
            return a, w

        # S-: gravity removed via the Gravity channels; the along-track axis is
        # unknown, so the forward direction is taken as the horizontal axis
        # best aligned with the vehicle's motion.
        axes = [_find(df, r"ACCELEROMETER", rf"\b{c}\b") for c in "XYZ"]
        grav = [_find(df, r"GRAVITY", rf"\b{c}\b") for c in "XYZ"]
        gyro = _find(df, r"GYROSCOPE", r"YAW")
        if any(c is None for c in axes) or gyro is None:
            raise ValueError("S- session lacks accelerometer or gyroscope")
        lin = []
        for acol, gcol in zip(axes, grav):
            v = pd.to_numeric(df[acol], errors="coerce")
            if gcol is not None:
                v = v - pd.to_numeric(df[gcol], errors="coerce")
            lin.append(v.to_numpy(float))
        A = np.column_stack(lin)
        # Project onto the direction most correlated with speed change.
        dv = np.gradient(np.nan_to_num(self.speed), self.t)
        best, best_r = 0, 0.0
        for i in range(3):
            col = np.nan_to_num(A[:, i])
            if col.std() < 1e-9:
                continue
            r = float(np.corrcoef(col, np.nan_to_num(dv))[0, 1])
            if np.isfinite(r) and abs(r) > abs(best_r):
                best, best_r = i, r
        a = A[:, best] * (-1.0 if best_r < 0 else 1.0)
        w = pd.to_numeric(df[gyro], errors="coerce").to_numpy(float)  # already rad/s
        return a, w

    # -- bias --------------------------------------------------------------

    def _estimate_bias(self) -> float:
        """Accelerometer bias as the mean over stationary stretches.

        A residual bias is double-integrated into a quadratically growing
        position error, so this is the single most consequential correction in
        the whole baseline.
        """
        still = np.isfinite(self.speed) & (self.speed < STATIONARY_SPEED_MS)
        vals = self.accel_long[still & np.isfinite(self.accel_long)]
        if vals.size < MIN_STATIONARY_SAMPLES:
            return 0.0
        return float(np.mean(vals))

    # -- prediction --------------------------------------------------------

    def predict(self, t0: float, duration: float) -> dict:
        """Dead-reckon `duration` seconds from `t0`.

        Returns per-second displacement magnitudes and the heading held during
        each second, so the harness can accumulate position identically for
        every predictor.
        """
        idx0 = int(np.searchsorted(self.t, t0))
        if idx0 >= self.t.size - 2:
            raise ValueError("t0 beyond the end of the session")

        speed = float(self.speed[idx0]) if np.isfinite(self.speed[idx0]) else 0.0
        heading = self._initial_heading(idx0)

        disps, headings = [], []
        t_cur = t0
        for _ in range(int(round(duration))):
            t_end = t_cur + 1.0
            sl = slice(int(np.searchsorted(self.t, t_cur)),
                       int(np.searchsorted(self.t, t_end)))
            ts = self.t[sl]
            if ts.size < 2:
                disps.append(speed)
                headings.append(heading)
                t_cur = t_end
                continue
            dt = np.diff(ts)
            a = np.nan_to_num(self.accel_long[sl]) - self.accel_bias
            w = np.nan_to_num(self.yaw_rate[sl])

            # Trapezoidal integration within the second: heading first, then
            # speed, then distance -- so the distance travelled uses the speed
            # profile actually integrated, not just its endpoints.
            head_k = heading + np.concatenate([[0.0], np.cumsum(
                0.5 * (w[:-1] + w[1:]) * dt)])
            spd_k = speed + np.concatenate([[0.0], np.cumsum(
                0.5 * (a[:-1] + a[1:]) * dt)])
            spd_k = np.maximum(spd_k, 0.0)          # a vehicle cannot reverse here
            dist = float(np.sum(0.5 * (spd_k[:-1] + spd_k[1:]) * dt))

            disps.append(dist)
            headings.append(float(np.mean(head_k)))
            speed = float(spd_k[-1])
            heading = float(head_k[-1])
            t_cur = t_end

        return {"displacements": np.asarray(disps),
                "headings": np.asarray(headings)}

    def _initial_heading(self, idx0: int) -> float:
        """GPS heading at the outage start, in radians, 0 = north.

        Meaningless at rest, which is why the harness refuses outages starting
        below 5 m/s.
        """
        col = _find(self.df, r"^HEADING") or _find(self.df, r"GPS ORIENTATION")
        if col is None:
            return 0.0
        v = pd.to_numeric(self.df[col], errors="coerce").to_numpy(float)
        h = v[idx0]
        if not np.isfinite(h):
            finite = v[np.isfinite(v)]
            h = finite[0] if finite.size else 0.0
        return float(h) * DEG_TO_RAD
