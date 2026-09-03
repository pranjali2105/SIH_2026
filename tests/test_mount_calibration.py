"""Tests for offline mount-yaw (phi) calibration.

`estimate_mount_yaw` has no dependency on the (currently missing, see
CLAUDE.md / repo `.gitignore`) `data` package -- everything here is built
from synthetic arrays, mirroring how `tests/test_mapmatch.py` drives
`RoadGraph.from_ways` directly instead of a real extract.
"""

from __future__ import annotations

import numpy as np
import pytest

from fusion.mount_calibration import estimate_mount_yaw


def _synthetic_drive(phi_true: float, duration_s: float = 200.0,
                     turning: bool = True, braking: bool = True):
    """A smooth speed/heading history and the raw accel it would produce.

    `chan[:, 0], chan[:, 1]` are built so that rotating them by `phi_true`
    (`eskf.py`'s `a_fwd = a_h1*cos+a_h2*sin`, `a_lat = -a_h1*sin+a_h2*cos`)
    exactly recovers the true forward/lateral specific force -- i.e. this is
    the forward model `estimate_mount_yaw` inverts, at 1 Hz so no grid
    interpolation error is introduced into the check.
    """
    t = np.arange(0.0, duration_s, 1.0)
    speed = 15.0 * np.ones_like(t)
    if braking:
        speed = speed + 5.0 * np.sin(2 * np.pi * t / 40.0)
    heading = (0.35 * np.sin(2 * np.pi * t / 23.0) if turning
              else np.zeros_like(t))

    a_fwd = np.gradient(speed, t)
    a_lat = speed * np.gradient(heading, t)

    c, s = np.cos(phi_true), np.sin(phi_true)
    chan = np.zeros((t.size, 6))
    chan[:, 0] = c * a_fwd - s * a_lat
    chan[:, 1] = s * a_fwd + c * a_lat
    return t, speed, heading, chan


def test_recovers_a_known_mount_yaw():
    phi_true = np.deg2rad(37.0)
    t, speed, heading, chan = _synthetic_drive(phi_true)
    est = estimate_mount_yaw(t, speed, heading, t, chan, t0=float(t[-1]) + 1.0,
                             lookback_s=250.0)
    assert est is not None
    assert est.observable
    err_deg = np.degrees(abs(np.arctan2(np.sin(est.phi_rad - phi_true),
                                        np.cos(est.phi_rad - phi_true))))
    assert err_deg < 5.0, f"recovered {np.degrees(est.phi_rad):.1f} deg, " \
        f"true {np.degrees(phi_true):.1f} deg"


@pytest.mark.parametrize("phi_deg", [-150.0, -20.0, 0.0, 65.0, 179.0])
def test_recovers_mount_yaw_across_the_full_circle(phi_deg):
    phi_true = np.deg2rad(phi_deg)
    t, speed, heading, chan = _synthetic_drive(phi_true)
    est = estimate_mount_yaw(t, speed, heading, t, chan, t0=float(t[-1]) + 1.0,
                             lookback_s=250.0)
    assert est is not None and est.observable
    err = abs(np.arctan2(np.sin(est.phi_rad - phi_true),
                         np.cos(est.phi_rad - phi_true)))
    assert np.degrees(err) < 5.0


def test_only_braking_no_turning_is_not_observable():
    """Straight-line driving gives the forward relation but not the lateral
    one -- exactly the ESKF's own diagnosed failure mode (eskf.py: "one
    equation for two unknowns"), reproduced offline as a diagnostic instead
    of a divergence.
    """
    phi_true = np.deg2rad(20.0)
    t, speed, heading, chan = _synthetic_drive(phi_true, turning=False)
    est = estimate_mount_yaw(t, speed, heading, t, chan, t0=float(t[-1]) + 1.0,
                             lookback_s=250.0)
    assert est is not None
    assert est.n_lateral_events == 0
    assert est.n_forward_events > 0
    assert not est.observable


def test_only_turning_no_braking_is_not_observable():
    phi_true = np.deg2rad(20.0)
    t, speed, heading, chan = _synthetic_drive(phi_true, braking=False)
    est = estimate_mount_yaw(t, speed, heading, t, chan, t0=float(t[-1]) + 1.0,
                             lookback_s=250.0)
    assert est is not None
    assert est.n_forward_events == 0
    assert not est.observable


def test_never_reads_at_or_after_t0():
    """Deployability guarantee: corrupting data at/after t0 must not change
    the fit, since a real system would not have it yet.
    """
    phi_true = np.deg2rad(37.0)
    t0 = 150.0
    t, speed, heading, chan = _synthetic_drive(phi_true, duration_s=300.0)
    est_before = estimate_mount_yaw(t, speed, heading, t, chan, t0=t0,
                                    lookback_s=140.0)

    t2, speed2, heading2, chan2 = t.copy(), speed.copy(), heading.copy(), chan.copy()
    after = t2 >= t0
    speed2[after] = 999.0
    heading2[after] = 999.0
    chan2[after, :] = 999.0
    est_after_corruption = estimate_mount_yaw(t2, speed2, heading2, t2, chan2,
                                              t0=t0, lookback_s=140.0)

    assert est_before is not None and est_after_corruption is not None
    assert est_before.phi_rad == pytest.approx(est_after_corruption.phi_rad)


def test_too_little_history_returns_none():
    t = np.array([0.0])
    speed = np.array([10.0])
    heading = np.array([0.0])
    chan = np.zeros((1, 6))
    assert estimate_mount_yaw(t, speed, heading, t, chan, t0=1.0,
                              lookback_s=180.0) is None


def test_gross_outliers_are_excluded_by_the_plausibility_guard():
    """A single GPS/IMU glitch implying >8 m/s^2 must not swing the fit --
    mirrors the 60 m/s displacement-label guard CLAUDE.md documents for the
    same class of defect.
    """
    phi_true = np.deg2rad(37.0)
    t, speed, heading, chan = _synthetic_drive(phi_true)
    speed_glitched = speed.copy()
    speed_glitched[100] += 500.0            # one insane GPS speed spike
    est_clean = estimate_mount_yaw(t, speed, heading, t, chan,
                                   t0=float(t[-1]) + 1.0, lookback_s=250.0)
    est_glitched = estimate_mount_yaw(t, speed_glitched, heading, t, chan,
                                      t0=float(t[-1]) + 1.0, lookback_s=250.0)
    assert est_clean is not None and est_glitched is not None
    assert abs(est_clean.phi_rad - est_glitched.phi_rad) < np.deg2rad(1.0)
