"""Constant-velocity dead reckoning — the deployable persistence baseline.

Windowed persistence (predict the previous second's displacement) scores 0.18 m
on validate, but it is NOT deployable: it consumes the previous second's TRUE
displacement, which is precisely what a GNSS outage removes. Fed through the
harness, where after the first step it can only see its own previous output, it
degenerates to this: hold the speed and heading observed at t0 and extrapolate.

That makes it a real baseline, and a diagnostic one. It knows the speed regime
at the moment the outage starts and nothing else — no IMU at all. Any model's
advantage over it is precisely the part that comes from reading the IMU rather
than from knowing how fast the vehicle was going.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.loader import Session, _find

DEG_TO_RAD = np.pi / 180.0


class ConstantVelocityDR:
    """Hold speed and heading from the outage start."""

    name = "constant_velocity_dr"

    def __init__(self, session: Session):
        self.session = session
        self.t = pd.to_numeric(session.df["time_s"], errors="coerce").to_numpy(float)
        self.speed = (pd.to_numeric(session.df["speed_ms"], errors="coerce")
                      .to_numpy(float) if "speed_ms" in session.df.columns
                      else np.full(self.t.shape, np.nan))
        col = _find(session.df, r"^HEADING") or _find(session.df, r"GPS ORIENTATION")
        self.heading = (pd.to_numeric(session.df[col], errors="coerce")
                        .to_numpy(float) if col is not None
                        else np.zeros_like(self.t))

    def predict(self, t0: float, duration: float) -> dict:
        n = int(round(duration))
        v0 = float(np.interp(t0, self.t, np.nan_to_num(self.speed)))
        h0 = float(np.interp(t0, self.t, np.nan_to_num(self.heading))) * DEG_TO_RAD
        # One second at the initial speed, repeated: the predictor never sees
        # another observation, so every step is identical.
        return {"displacements": np.full(n, v0),
                "headings": np.full(n, h0)}
