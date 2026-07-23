# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Virtual trajectory compensation (LI + GPR) for bridging Kalman-filter state across lost frames.

When a track goes Lost and is later re-matched, the frames in between currently rely on pure
constant-velocity Kalman prediction (no correction), so state error accumulates and the recovered
trajectory can visibly "jump" at the re-match frame. This module builds a smooth sequence of
pseudo-observations for the lost frames (linear interpolation between the last-observed and
newly-matched states, refined with a Gaussian Process), so the tracker can replay a normal
predict()+update() cycle through the gap instead of coasting on prediction alone.
"""

from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from .kalman_filter import resolve_angle


class VirtualTrajectoryConfig:
    """Tunable parameters for virtual trajectory compensation.

    Attributes:
        MIN_LOST_FRAMES_FOR_VIRTUAL (int): Minimum number of consecutive lost frames required before
            building a virtual trajectory; below this, a single normal update is just as good and the
            LI+GPR machinery is skipped.
        GPR_LENGTH_SCALE (float): RBF kernel length scale (in frames); larger values smooth harder.
        GPR_NOISE_LEVEL (float): WhiteKernel noise level added to the GPR fit.
        ANGLE_PERIOD (float): Angle period in **radians** (this project's Kalman filter and detector
            angles are radian-valued throughout; a 180°-symmetric rotated box uses ``pi``). Matches
            ``KalmanFilterXYWHA``'s default so the two stay consistent.
        MOTION_STATIC_THRESHOLD (float): Speed (pixels/frame) below which a track is considered static
            for angle disambiguation purposes, falling back to nearest-branch continuity instead of a
            velocity heading. Matches ``KalmanFilterXYWHA``'s default motion threshold.
    """

    MIN_LOST_FRAMES_FOR_VIRTUAL = 2
    GPR_LENGTH_SCALE = 1.0
    GPR_NOISE_LEVEL = 0.1
    ANGLE_PERIOD = np.pi
    MOTION_STATIC_THRESHOLD = 2.0


def _li_gpr_1d(t1: int, t2: int, y1: float, y2: float, lost_frame_indices: list[int],
                length_scale: float, noise_level: float) -> np.ndarray:
    """Linearly interpolate a scalar between two anchors, then smooth it with a GP regressor.

    Args:
        t1 (int): Frame index of the first anchor.
        t2 (int): Frame index of the second anchor.
        y1 (float): Scalar value at `t1`.
        y2 (float): Scalar value at `t2`.
        lost_frame_indices (list[int]): Frame indices strictly between `t1` and `t2` to evaluate.
        length_scale (float): RBF kernel length scale.
        noise_level (float): WhiteKernel noise level.

    Returns:
        (np.ndarray): Smoothed scalar values at `lost_frame_indices`, shape `(len(lost_frame_indices),)`.
    """
    train_t = np.array([t1, *lost_frame_indices, t2], dtype=float)
    li_values = y1 + (y2 - y1) * (train_t - t1) / (t2 - t1)

    kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)
    gpr = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=True)
    gpr.fit(train_t.reshape(-1, 1), li_values)

    query_t = np.asarray(lost_frame_indices, dtype=float).reshape(-1, 1)
    return gpr.predict(query_t)


def build_virtual_trajectory(
    z_t1: np.ndarray,
    z_t2: np.ndarray,
    t1: int,
    t2: int,
    lost_frame_indices: list[int],
    angle_dim_index: int | None,
    angle_period: float = VirtualTrajectoryConfig.ANGLE_PERIOD,
    gpr_length_scale: float = VirtualTrajectoryConfig.GPR_LENGTH_SCALE,
    gpr_noise_level: float = VirtualTrajectoryConfig.GPR_NOISE_LEVEL,
    vx: float = 0.0,
    vy: float = 0.0,
    motion_static_threshold: float = VirtualTrajectoryConfig.MOTION_STATIC_THRESHOLD,
) -> np.ndarray:
    """Build smooth pseudo-observations for the frames between two known track states.

    Every non-angle dimension is interpolated independently with LI, then smoothed with a GP
    (`RBF(length_scale) + WhiteKernel(noise_level)`). The angle dimension (if present) is first
    disambiguated against the track's heading (or against `z_t1`'s angle when the track is near
    static), then carried through the same LI+GPR pipeline as `(cos, sin)` components to keep the
    interpolation periodic-safe, and finally recombined with `atan2`.

    Args:
        z_t1 (np.ndarray): Shape `(state_dim,)`, observed state at `t1` (last observation before loss).
        z_t2 (np.ndarray): Shape `(state_dim,)`, observed state at `t2` (newly re-matched detection).
        t1 (int): Frame index of `z_t1`.
        t2 (int): Frame index of `z_t2`.
        lost_frame_indices (list[int]): Frame indices strictly between `t1` and `t2`.
        angle_dim_index (int | None): Index of the angle component within the state vector, or `None`
            if the state has no angle dimension.
        angle_period (float): Angle physical period in radians (`pi` for a 180°-symmetric rotated box).
        gpr_length_scale (float): RBF kernel length scale (in frames).
        gpr_noise_level (float): WhiteKernel noise level.
        vx (float): Track's x velocity (pixels/frame) at the time of loss, used for heading-based angle
            disambiguation.
        vy (float): Track's y velocity (pixels/frame) at the time of loss.
        motion_static_threshold (float): Speed threshold below which heading is considered unreliable
            and angle disambiguation falls back to `z_t1`'s angle instead.

    Returns:
        (np.ndarray): Shape `(len(lost_frame_indices), state_dim)`, virtual observed states for each
            lost frame.
    """
    z_t1 = np.asarray(z_t1, dtype=float)
    z_t2 = np.asarray(z_t2, dtype=float)
    state_dim = z_t1.shape[0]
    n_lost = len(lost_frame_indices)
    out = np.zeros((n_lost, state_dim), dtype=float)

    if n_lost == 0:
        return out

    for dim in range(state_dim):
        if dim == angle_dim_index:
            continue
        out[:, dim] = _li_gpr_1d(t1, t2, z_t1[dim], z_t2[dim], lost_frame_indices, gpr_length_scale, gpr_noise_level)

    if angle_dim_index is not None:
        raw_a1 = float(z_t1[angle_dim_index])
        raw_a2 = float(z_t2[angle_dim_index])
        # Disambiguate both anchors against the track's heading (or against each other when static).
        a1 = resolve_angle(raw_a1, ref_angle=raw_a1, vx=vx, vy=vy, period=angle_period, motion_thresh=motion_static_threshold)
        a2 = resolve_angle(raw_a2, ref_angle=a1, vx=vx, vy=vy, period=angle_period, motion_thresh=motion_static_threshold)

        omega = 2.0 * np.pi / angle_period
        cos1, sin1 = np.cos(omega * a1), np.sin(omega * a1)
        cos2, sin2 = np.cos(omega * a2), np.sin(omega * a2)

        cos_pred = _li_gpr_1d(t1, t2, cos1, cos2, lost_frame_indices, gpr_length_scale, gpr_noise_level)
        sin_pred = _li_gpr_1d(t1, t2, sin1, sin2, lost_frame_indices, gpr_length_scale, gpr_noise_level)

        out[:, angle_dim_index] = np.arctan2(sin_pred, cos_pred) / omega

    return out
