from __future__ import annotations
import numpy as np
from dataclasses import dataclass

__all__ = [
    "normalize_quat",
    "quats_to_R_batch",
    "yaw_from_R",
    "apply_yaw_align",
    "world_vec",
    "estimate_fs",
    "moving_avg",
    "slerp",
    "resample_quat",
    "resample_vec",
    "gyro_from_quat",
    # Angle helpers
    "hip_angles_xyz",
    "knee_angles_xyz",
    "jcs_hip_angles",
    "jcs_knee_angles",
    # Resampling utility
    "SideResampled",
    "resample_side_to_femur_time",
]


def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12
    return q / n


def quats_to_R_batch(quats: np.ndarray) -> np.ndarray:
    q = normalize_quat(np.asarray(quats, dtype=float))
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    R = np.empty((q.shape[0], 3, 3), dtype=float)
    R[:, 0, 0] = 1 - 2 * (yy + zz)
    R[:, 0, 1] = 2 * (xy - wz)
    R[:, 0, 2] = 2 * (xz + wy)
    R[:, 1, 0] = 2 * (xy + wz)
    R[:, 1, 1] = 1 - 2 * (xx + zz)
    R[:, 1, 2] = 2 * (yz - wx)
    R[:, 2, 0] = 2 * (xz - wy)
    R[:, 2, 1] = 2 * (yz + wx)
    R[:, 2, 2] = 1 - 2 * (xx + yy)
    return R


def yaw_from_R(R_WS: np.ndarray) -> np.ndarray:
    ex = np.array([1.0, 0.0, 0.0])
    e = (R_WS @ ex[:, None]).squeeze(-1)
    return np.arctan2(e[:, 1], e[:, 0])


def apply_yaw_align(R_WS: np.ndarray, yaw_delta: float | np.ndarray) -> np.ndarray:
    y = np.asarray(yaw_delta, dtype=float)
    if y.ndim == 0:
        y = np.full((R_WS.shape[0],), float(y))
    cz, sz = np.cos(y), np.sin(y)
    zeros = np.zeros_like(cz)
    ones = np.ones_like(cz)
    Rz = np.stack(
        [
            np.stack([cz, -sz, zeros], axis=1),
            np.stack([sz, cz, zeros], axis=1),
            np.stack([zeros, zeros, ones], axis=1),
        ],
        axis=1,
    )
    return np.einsum("tij,tjk->tik", Rz, R_WS)


def world_vec(R_WS: np.ndarray, v_S: np.ndarray) -> np.ndarray:
    return (R_WS @ v_S[..., None]).squeeze(-1)


def estimate_fs(t: np.ndarray) -> float:
    t = np.asarray(t, dtype=float)
    if t.size < 2:
        return 100.0
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return 100.0
    return float(1.0 / np.median(dt))


def moving_avg(x: np.ndarray, win: int = 9) -> np.ndarray:
    if win <= 1:
        return x
    win = int(max(1, win))
    k = np.ones(win, dtype=float) / win
    if x.ndim == 1:
        return np.convolve(x, k, mode="same")
    return np.vstack(
        [np.convolve(x[:, i], k, mode="same") for i in range(x.shape[1])]
    ).T


def slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    dot = float(np.dot(q0, q1))
    q1c = q1.copy()
    if dot < 0.0:
        q1c = -q1c
        dot = -dot
    if dot > 0.9995:
        v = q0 + u * (q1c - q0)
        return v / (np.linalg.norm(v) + 1e-12)
    th0 = np.arccos(np.clip(dot, -1.0, 1.0))
    s0 = np.sin((1.0 - u) * th0) / np.sin(th0)
    s1 = np.sin(u * th0) / np.sin(th0)
    return s0 * q0 + s1 * q1c


def resample_quat(
    t_src: np.ndarray, Q_src: np.ndarray, t_dst: np.ndarray
) -> np.ndarray:
    Q_src = normalize_quat(Q_src)
    out = np.zeros((len(t_dst), 4), dtype=float)
    i = 0
    for k, tk in enumerate(t_dst):
        while i + 1 < len(t_src) and not (t_src[i] <= tk <= t_src[i + 1]):
            i += 1
        if i + 1 >= len(t_src):
            out[k] = Q_src[-1]
            continue
        t0, t1 = t_src[i], t_src[i + 1]
        u = 0.0 if t1 == t0 else float((tk - t0) / (t1 - t0))
        out[k] = slerp(Q_src[i], Q_src[i + 1], float(np.clip(u, 0.0, 1.0)))
    return normalize_quat(out)


def resample_vec(t_src: np.ndarray, X_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    X_src = np.asarray(X_src, dtype=float)
    out = np.zeros((len(t_dst), X_src.shape[1]), dtype=float)
    for j in range(X_src.shape[1]):
        out[:, j] = np.interp(t_dst, t_src, X_src[:, j])
    return out


def gyro_from_quat(t: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Approximate body-frame angular velocity from consecutive quaternions.
    Uses ΔR = R_k^T R_{k+1}, axis-angle over dt. Returns (T,3) in rad/s.
    """
    Q = np.asarray(quat, dtype=float)
    R = quats_to_R_batch(Q)
    Tn = len(t)
    w = np.zeros((Tn, 3), dtype=float)
    for i in range(Tn - 1):
        dt = float(max(1e-6, t[i + 1] - t[i]))
        dR = R[i].T @ R[i + 1]
        tr = float(np.trace(dR))
        c = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
        angle = float(np.arccos(c))
        if angle < 1e-6:
            axis = np.array([0.0, 0.0, 0.0])
        else:
            denom = 2.0 * np.sin(angle)
            axis = np.array(
                [
                    (dR[2, 1] - dR[1, 2]) / denom,
                    (dR[0, 2] - dR[2, 0]) / denom,
                    (dR[1, 0] - dR[0, 1]) / denom,
                ]
            )
        w[i] = axis * (angle / dt)
    if Tn > 1:
        w[-1] = w[-2]
    return w


@dataclass
class SideResampled:
    t: np.ndarray
    q_femur: np.ndarray
    gyro_femur: np.ndarray
    acc_femur: np.ndarray
    q_tibia: np.ndarray
    gyro_tibia: np.ndarray
    acc_tibia: np.ndarray
    q_pelvis: np.ndarray


def resample_side_to_femur_time(
    tF: np.ndarray,
    qF: np.ndarray,
    gF: np.ndarray,
    aF: np.ndarray,
    tT: np.ndarray,
    qT: np.ndarray,
    gT: np.ndarray,
    aT: np.ndarray,
    tP: np.ndarray,
    qP: np.ndarray,
) -> SideResampled:
    """Resample tibia and pelvis onto the femur timebase and return a structured bundle."""
    tF0 = tF - tF[0]
    tT0 = tT - tT[0]
    tP0 = tP - tP[0]
    t = tF0.copy()
    return SideResampled(
        t=t,
        q_femur=qF,
        gyro_femur=gF,
        acc_femur=aF,
        q_tibia=resample_quat(tT0, qT, t),
        gyro_tibia=resample_vec(tT0, gT, t),
        acc_tibia=resample_vec(tT0, aT, t),
        q_pelvis=resample_quat(tP0, qP, t),
    )


# -----------------------------
# Grood–Suntay-inspired JCS angles
# -----------------------------
def _normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if v.ndim == 1:
        n = float(np.linalg.norm(v))
        return v / (n + eps)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + eps)


def _proj_on_plane(v: np.ndarray, n_hat: np.ndarray) -> np.ndarray:
    # Project v onto plane with normal n_hat
    n = _normalize(n_hat)
    return v - (np.sum(v * n, axis=-1, keepdims=True) * n)


def _signed_angle(
    a: np.ndarray, b: np.ndarray, axis: np.ndarray, eps: float = 1e-9
) -> np.ndarray:
    # Returns signed angle from a to b around axis (right-hand rule)
    a_ = _proj_on_plane(a, axis)
    b_ = _proj_on_plane(b, axis)
    a_ = _normalize(a_)
    b_ = _normalize(b_)
    dot = np.clip(np.sum(a_ * b_, axis=-1), -1.0, 1.0)
    ang = np.arccos(dot)
    cross = np.cross(a_, b_)
    s = np.sign(np.sum(cross * _normalize(axis), axis=-1))
    return ang * s


def _euler_xyz_from_relative(R_rel: np.ndarray) -> np.ndarray:
    """Convert relative rotation matrices to intrinsic XYZ Euler angles.

    Contract:
    - Input: R_rel shape (T,3,3) or (3,3), right-handed rotations.
    - Output: angles (T,3) in radians [X, Y, Z] for intrinsic 'xyz'.
    - Behavior near gimbal lock (|Y| -> 90deg): relies on SciPy's
      handling; one DOF may be set to zero but reconstruction remains stable.
    """
    from scipy.spatial.transform import Rotation as _Rot

    Rr = np.asarray(R_rel, dtype=float)
    if Rr.ndim == 2:
        Rr = Rr[None, ...]

    # Guard tiny numerical drift by projecting to nearest rotation via SVD
    # only if a matrix is badly conditioned.
    def _project_rot(M: np.ndarray) -> np.ndarray:
        U, _, Vt = np.linalg.svd(M)
        Rm = U @ Vt
        if np.linalg.det(Rm) < 0:
            U[:, -1] *= -1
            Rm = U @ Vt
        return Rm

    bad = ~np.isfinite(Rr).all(axis=(1, 2))
    if np.any(bad):
        Rr[bad] = np.eye(3)
    # Convert
    rot = _Rot.from_matrix(Rr)
    return rot.as_euler("xyz", degrees=False)


def hip_angles_xyz(
    R_pelvis: np.ndarray, R_femur: np.ndarray, side: str = "L"
) -> np.ndarray:
    """Hip angles via intrinsic XYZ Euler sequence (ISB-style axes).

    Axes:
    - X (sagittal): + = flexion (forward raise)
    - Y (frontal):  + = adduction (toward midline) for both sides
    - Z (axial):    + = internal rotation (inward spin)

    Side handling:
    - Right limb uses a sign flip on the Y component only to keep adduction
      positive toward midline across limbs. Z (internal rot) remains positive
      for both sides without flipping.
    """
    Rp = np.asarray(R_pelvis, dtype=float)
    Rf = np.asarray(R_femur, dtype=float)
    if Rp.size == 0 or Rf.size == 0:
        return np.zeros((0, 3), dtype=float)
    # Relative pelvis->femur
    R_rel = np.einsum("tij,tjk->tik", np.transpose(Rp, (0, 2, 1)), Rf)
    e = _euler_xyz_from_relative(R_rel)
    # Unify adduction sign across limbs
    if str(side).upper() == "R":
        e[:, 1] *= -1.0
    return e


def knee_angles_xyz(
    R_femur: np.ndarray, R_tibia: np.ndarray, side: str = "L"
) -> np.ndarray:
    """Knee angles via intrinsic XYZ Euler sequence (ISB-style axes).

    Conventions as hip: X=flexion, Y=adduction (varus/valgus), Z=internal rotation.
    Returns angles (T,3) in radians. For side='R', Y-angle is sign-flipped so
    adduction is positive toward midline for both limbs. Z sign is unchanged
    so internal rotation remains positive on both sides.
    """
    Rf = np.asarray(R_femur, dtype=float)
    Rt = np.asarray(R_tibia, dtype=float)
    if Rf.size == 0 or Rt.size == 0:
        return np.zeros((0, 3), dtype=float)
    # Relative femur->tibia
    R_rel = np.einsum("tij,tjk->tik", np.transpose(Rf, (0, 2, 1)), Rt)
    e = _euler_xyz_from_relative(R_rel)
    if str(side).upper() == "R":
        e[:, 1] *= -1.0
    return e


# Back-compatible aliases
jcs_hip_angles = hip_angles_xyz
jcs_knee_angles = knee_angles_xyz
