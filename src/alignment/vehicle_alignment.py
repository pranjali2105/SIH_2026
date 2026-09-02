"""In-vehicle alignment and calibration engine.

Three stages:
  1. static pitch/roll from the gravity vector while stationary
  2. dynamic yaw from PCA on horizontal acceleration during a straight
     acceleration
  3. live propagation of the rotation matrix by the gyro, with SVD
     re-orthogonalisation

Provided as specified. Three syntax errors in the source (empty first rows in
the R_roll and R_pitch literals) are filled in with the standard rotation
matrices; two behavioural issues are flagged inline where they occur.
"""

from __future__ import annotations

import numpy as np

GRAVITY = 9.80665


class VehicleAlignmentEngine:
    def __init__(self, sample_rate_hz: float = 10.0):
        self.dt = 1.0 / sample_rate_hz
        self.is_calibrated = False

        self.static_buffer_acc: list = []
        self.calibration_threshold_samples = int(5.0 * sample_rate_hz)

        self.C_b_n = np.eye(3)
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw_offset = 0.0

    # -- step 1 ------------------------------------------------------------

    def update_static_calibration(self, acc_raw) -> bool:
        """Static pitch and roll from the gravity vector."""
        self.static_buffer_acc.append(np.asarray(acc_raw, dtype=float))
        if len(self.static_buffer_acc) < self.calibration_threshold_samples:
            return False

        ax, ay, az = np.mean(self.static_buffer_acc, axis=0)
        self.roll = np.arctan2(ay, az)
        self.pitch = np.arctan2(-ax, np.sqrt(ay ** 2 + az ** 2))

        R_roll = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(self.roll), -np.sin(self.roll)],
            [0.0, np.sin(self.roll), np.cos(self.roll)],
        ])
        R_pitch = np.array([
            [np.cos(self.pitch), 0.0, np.sin(self.pitch)],
            [0.0, 1.0, 0.0],
            [-np.sin(self.pitch), 0.0, np.cos(self.pitch)],
        ])
        self.C_b_n = R_pitch @ R_roll
        self.is_calibrated = True
        return True

    # -- step 2 ------------------------------------------------------------

    def align_dynamic_yaw(self, acceleration_window) -> float:
        """Yaw offset from the principal axis of horizontal acceleration.

        CAVEAT 1 -- sign ambiguity. A principal component is an AXIS, not a
        direction: `vh[0]` and `-vh[0]` are equally valid SVD outputs, so the
        recovered yaw is only determined modulo pi. Forward and reverse are
        indistinguishable. Resolved here by requiring the mean acceleration to
        project positively onto the chosen axis, which assumes the window is
        genuinely an ACCELERATION rather than a deceleration.

        CAVEAT 2 -- the window must be straight-line. PCA returns the axis of
        greatest variance. During cornering, lateral acceleration exceeds
        longitudinal, so the principal axis is the LATERAL one and the answer
        is 90 degrees wrong. The caller is responsible for supplying a
        straight, accelerating stretch; `align_dynamic_yaw_checked` does that
        filtering.
        """
        if not self.is_calibrated:
            raise ValueError("Run static calibration before dynamic yaw alignment.")

        window = np.asarray(acceleration_window, dtype=float)
        leveled_acc = window @ self.C_b_n.T
        horizontal_acc = leveled_acc[:, :2]
        centered = horizontal_acc - horizontal_acc.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        forward_vector = vh[0]

        # Resolve the pi ambiguity against the mean acceleration direction.
        if float(horizontal_acc.mean(axis=0) @ forward_vector) < 0:
            forward_vector = -forward_vector

        self.yaw_offset = float(np.arctan2(forward_vector[1], forward_vector[0]))
        R_yaw = np.array([
            [np.cos(self.yaw_offset), -np.sin(self.yaw_offset), 0.0],
            [np.sin(self.yaw_offset), np.cos(self.yaw_offset), 0.0],
            [0.0, 0.0, 1.0],
        ])
        self.C_b_n = R_yaw @ self.C_b_n
        return self.yaw_offset

    # -- step 3 ------------------------------------------------------------

    def process_live_imu(self, acc_raw, gyro_raw):
        """Propagate the rotation matrix and return Earth-stabilised vectors.

        CAVEAT 3 -- gravity is subtracted from index 2 of the navigation-frame
        acceleration, which is correct only if the level frame's z-axis is
        vertical. That holds immediately after static calibration, but this
        method propagates C_b_n by the FULL 3-axis gyro, so any pitch/roll
        error integrates and the z-axis drifts away from vertical. Over a long
        stream the gravity subtraction then leaks into the horizontal
        channels. This is the same unbounded-integration problem the INS
        baseline has.
        """
        acc_raw = np.asarray(acc_raw, dtype=float)
        gyro_raw = np.asarray(gyro_raw, dtype=float)
        if not self.is_calibrated:
            return acc_raw, gyro_raw

        wx, wy, wz = gyro_raw
        omega_skew = np.array([
            [0.0, -wz, wy],
            [wz, 0.0, -wx],
            [-wy, wx, 0.0],
        ])
        self.C_b_n = self.C_b_n @ (np.eye(3) + omega_skew * self.dt)

        # Re-orthogonalise: the first-order update is not a rotation matrix.
        U, _, Vt = np.linalg.svd(self.C_b_n)
        self.C_b_n = U @ Vt

        acc_nav = self.C_b_n @ acc_raw
        acc_nav[2] -= GRAVITY
        gyro_nav = self.C_b_n @ gyro_raw
        return acc_nav, gyro_nav

    # -- convenience -------------------------------------------------------

    def align_dynamic_yaw_checked(self, acc_window, gyro_window,
                                  max_yaw_rate: float = 0.02,
                                  min_accel: float = 0.3):
        """Only align on a window that is genuinely straight and accelerating.

        Returns (yaw_offset, accepted). See CAVEAT 2.
        """
        gy = np.abs(np.asarray(gyro_window, dtype=float)[:, 2]).mean()
        lev = np.asarray(acc_window, dtype=float) @ self.C_b_n.T
        along = np.linalg.norm(lev[:, :2].mean(axis=0))
        if gy > max_yaw_rate or along < min_accel:
            return self.yaw_offset, False
        return self.align_dynamic_yaw(acc_window), True
