"""Error-state EKF applying non-holonomic constraints to correct heading.

Follows the approach of Brossard, Barrau & Bonnabel, "AI-IMU Dead-Reckoning"
(github.com/mbrossar/ai-imu-dr) in spirit -- pseudo-measurements from the
non-holonomic constraint of a road vehicle, with the measurement noise adapted
online -- but is an independent minimal implementation for this pipeline, not
a port.

The physical fact
-----------------
A road vehicle cannot translate sideways in its own body frame. Lateral
velocity is ~0 at all times, which is a measurement available every timestep
with no external sensor.

Why the constraint is applied in ACCELERATION, not velocity
-----------------------------------------------------------
The obvious form -- derive body-frame velocity from the position change and
drive its lateral component to zero -- is DEGENERATE in this pipeline. We
propagate position by stepping the model's displacement along the current
heading estimate, so the implied velocity is parallel to psi by construction
and its lateral component is identically zero. It carries no information and
would correct nothing.

The informative form is the derivative of the same constraint. If lateral
velocity is zero, then the lateral specific force the accelerometer sees is
purely centripetal:

    a_lateral = v * omega

with v the forward speed and omega the true yaw rate. The accelerometer
measures a_lateral directly, the model supplies v, and the gyro supplies a
biased omega. The residual

    r = a_lateral_measured - v * (omega_gyro - b_g)

is therefore non-zero exactly when the gyro bias is wrong, which makes b_g --
and hence heading drift -- observable. This is the same constraint, applied
where it has information.

Mount yaw is estimated, not assumed
-----------------------------------
The levelling in `data.windows` fixes roll and pitch but leaves an unknown
yaw offset phi between the phone's horizontal axes and the vehicle's forward
direction. phi is a state, observable from the pair of relations

    a_forward = dv/dt        a_lateral = v * omega

since the accelerometer's two horizontal components must decompose into
exactly those two quantities.

State
-----
    x = [px, py, psi, b_g, phi]

Position is carried for output; the filter's actual work is on the last three.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --- default noise parameters --------------------------------------------
SIGMA_NHC = 0.1           # m/s equivalent; the lateral pseudo-measurement
SIGMA_ZUPT_GYRO = 0.01    # rad/s, gyro output while genuinely stationary
GYRO_NOISE = 0.01         # rad/s, propagation
BIAS_RW = 1e-5            # rad/s per sqrt(s), bias random walk
PHI_RW = 1e-6             # rad per sqrt(s), mount rotates only if it slips

# The lateral constraint is violated during genuine cornering (tyre slip,
# body roll), so its noise is inflated with |omega| rather than trusted flat.
NHC_OMEGA_INFLATION = 8.0
STATIONARY_THRESHOLD = 0.5


@dataclass
class ESKFConfig:
    sigma_nhc: float = SIGMA_NHC
    sigma_zupt_gyro: float = SIGMA_ZUPT_GYRO
    gyro_noise: float = GYRO_NOISE
    bias_rw: float = BIAS_RW
    phi_rw: float = PHI_RW
    nhc_omega_inflation: float = NHC_OMEGA_INFLATION
    stationary_threshold: float = STATIONARY_THRESHOLD
    init_bias_std: float = 0.05
    init_heading_std: float = 0.05
    init_phi_std: float = np.pi          # mount yaw genuinely unknown
    apply_nhc: bool = True
    apply_zupt: bool = True
    # Mount yaw phi is handled UPSTREAM: windows are rotated into the
    # gravity-aligned level frame and the network is trained yaw-invariant by
    # augmentation, so its displacement output is an orientation-free scalar.
    # Carrying phi here adds a dimension nothing observes, which absorbs
    # residual and wanders. Off by default.
    estimate_phi: bool = False
    apply_forward: bool = False      # second measurement: a_fwd = dv/dt
    sigma_fwd: float = 0.3           # m/s^2


@dataclass
class ESKFTrace:
    """Per-step record, for the sanity checks."""
    heading: list = field(default_factory=list)
    bias: list = field(default_factory=list)
    phi: list = field(default_factory=list)
    nhc_residual: list = field(default_factory=list)
    zupt_applied: list = field(default_factory=list)


class NonHolonomicESKF:
    """Minimal error-state EKF over [px, py, psi, b_g, phi]."""

    N = 5
    I_PX, I_PY, I_PSI, I_BG, I_PHI = range(5)

    def __init__(self, psi0: float, cfg: ESKFConfig | None = None,
                 phi0: float = 0.0):
        """`phi0`: mount yaw to hold the filter at when `estimate_phi=False`.

        Findings.md §8 diagnosed why `estimate_phi=True` diverges: 60 s of
        outage data cannot identify phi and the gyro bias at once. `phi0`
        lets a caller supply phi from `fusion.mount_calibration` instead --
        fit offline, once per session, from GPS-available driving before the
        outage -- so the filter is never asked to identify it from a signal
        that cannot support it. Default 0.0 keeps prior behaviour unchanged.
        """
        self.cfg = cfg or ESKFConfig()
        self.x = np.zeros(self.N)
        self.x[self.I_PSI] = psi0
        self.x[self.I_PHI] = phi0
        phi_var = (self.cfg.init_phi_std ** 2) if self.cfg.estimate_phi else 0.0
        self.P = np.diag([1.0, 1.0,
                          self.cfg.init_heading_std ** 2,
                          self.cfg.init_bias_std ** 2,
                          phi_var])
        self.trace = ESKFTrace()

    # -- propagation -------------------------------------------------------

    def propagate(self, omega_z: float, displacement: float, dt: float,
                  sigma_d: float) -> None:
        """Advance heading by the de-biased gyro and position by the model."""
        psi, b_g = self.x[self.I_PSI], self.x[self.I_BG]
        d = float(displacement)

        self.x[self.I_PX] += d * np.cos(psi)
        self.x[self.I_PY] += d * np.sin(psi)
        self.x[self.I_PSI] = psi + (omega_z - b_g) * dt

        F = np.eye(self.N)
        # Position depends on heading through the direction it is stepped in.
        F[self.I_PX, self.I_PSI] = -d * np.sin(psi)
        F[self.I_PY, self.I_PSI] = d * np.cos(psi)
        # Heading integrates the bias with a negative sign.
        F[self.I_PSI, self.I_BG] = -dt

        Q = np.zeros((self.N, self.N))
        # Displacement noise is the model's OWN predicted sigma -- this is
        # what the uncertainty head was for.
        Q[self.I_PX, self.I_PX] = (sigma_d * np.cos(psi)) ** 2
        Q[self.I_PY, self.I_PY] = (sigma_d * np.sin(psi)) ** 2
        Q[self.I_PSI, self.I_PSI] = (self.cfg.gyro_noise ** 2) * dt * dt
        Q[self.I_BG, self.I_BG] = (self.cfg.bias_rw ** 2) * dt
        Q[self.I_PHI, self.I_PHI] = ((self.cfg.phi_rw ** 2) * dt
                                     if self.cfg.estimate_phi else 0.0)

        self.P = F @ self.P @ F.T + Q

    # -- updates -----------------------------------------------------------

    def _update(self, residual: float, H: np.ndarray, R: float) -> None:
        S = float(H @ self.P @ H.T + R)
        if not np.isfinite(S) or S <= 0:
            return
        K = (self.P @ H.T) / S
        self.x = self.x + K * residual
        A = np.eye(self.N) - np.outer(K, H)
        # Joseph form: stays symmetric positive-definite under the repeated
        # small updates this filter applies every timestep.
        self.P = A @ self.P @ A.T + np.outer(K, K) * R

    def update_nhc(self, a_h1: float, a_h2: float, omega_z: float,
                   speed: float) -> float:
        """Lateral specific force must be purely centripetal: a_lat = v*omega.

        Returns the residual, for diagnostics.
        """
        if not self.cfg.apply_nhc or speed < 0.5:
            return np.nan
        phi, b_g = self.x[self.I_PHI], self.x[self.I_BG]
        s, c = np.sin(phi), np.cos(phi)
        a_lat = -a_h1 * s + a_h2 * c
        predicted = speed * (omega_z - b_g)
        r = a_lat - predicted

        H = np.zeros(self.N)
        H[self.I_BG] = speed                      # d(-v*(w-b))/db = +v
        if self.cfg.estimate_phi:
            H[self.I_PHI] = -(a_h1 * c + a_h2 * s)   # d(a_lat)/dphi

        # Inflate during genuine cornering, where tyre slip and body roll
        # break the constraint the filter is asserting.
        R = (self.cfg.sigma_nhc *
             (1.0 + self.cfg.nhc_omega_inflation * abs(omega_z))) ** 2
        self._update(r, H, R)
        return float(r)

    def update_forward(self, a_h1: float, a_h2: float, dv_dt: float) -> float:
        """Longitudinal specific force must equal the rate of change of speed.

        The second half of the same non-holonomic decomposition. The
        accelerometer's two horizontal components must resolve into exactly
        a_fwd = dv/dt and a_lat = v*omega; with only the lateral relation there
        is one equation for two unknowns (mount yaw phi and gyro bias b_g) and
        they trade off, which is what made the filter wander. This closes it.
        """
        if not self.cfg.apply_forward:
            return np.nan
        phi = self.x[self.I_PHI]
        s_, c_ = np.sin(phi), np.cos(phi)
        a_fwd = a_h1 * c_ + a_h2 * s_
        r = a_fwd - dv_dt
        H = np.zeros(self.N)
        # d(a_fwd)/dphi = -a_h1 sin phi + a_h2 cos phi = a_lat
        H[self.I_PHI] = -a_h1 * s_ + a_h2 * c_
        self._update(r, H, self.cfg.sigma_fwd ** 2)
        return float(r)

    def update_zupt(self, omega_z: float) -> None:
        """Stationary: the gyro's output IS its bias.

        This is where ZUPT earns its place. Gating the displacement head at
        inference did essentially nothing (measured); making the bias
        observable is the useful form of the same information.
        """
        if not self.cfg.apply_zupt:
            return
        H = np.zeros(self.N)
        H[self.I_BG] = 1.0
        self._update(omega_z - self.x[self.I_BG], H,
                     self.cfg.sigma_zupt_gyro ** 2)

    # -- accessors ---------------------------------------------------------

    @property
    def heading(self) -> float:
        return float(self.x[self.I_PSI])

    @property
    def bias(self) -> float:
        return float(self.x[self.I_BG])

    def record(self, residual: float, zupt: bool) -> None:
        self.trace.heading.append(self.heading)
        self.trace.bias.append(self.bias)
        self.trace.phi.append(float(self.x[self.I_PHI]))
        self.trace.nhc_residual.append(residual)
        self.trace.zupt_applied.append(bool(zupt))
