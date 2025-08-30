from __future__ import annotations
import numpy as np

__all__ = [
    "normalize_quat","quats_to_R_batch","yaw_from_R","apply_yaw_align",
    "world_vec","estimate_fs","moving_avg","slerp","resample_quat","resample_vec",
    "gyro_from_quat",
    # JCS-inspired angle helpers
    "jcs_hip_angles","jcs_knee_angles",
]

def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12
    return q / n

def quats_to_R_batch(quats: np.ndarray) -> np.ndarray:
    q = normalize_quat(np.asarray(quats, dtype=float))
    w, x, y, z = q[:,0], q[:,1], q[:,2], q[:,3]
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    R = np.empty((q.shape[0],3,3), dtype=float)
    R[:,0,0] = 1 - 2*(yy+zz)
    R[:,0,1] = 2*(xy - wz)
    R[:,0,2] = 2*(xz + wy)
    R[:,1,0] = 2*(xy + wz)
    R[:,1,1] = 1 - 2*(xx + zz)
    R[:,1,2] = 2*(yz - wx)
    R[:,2,0] = 2*(xz - wy)
    R[:,2,1] = 2*(yz + wx)
    R[:,2,2] = 1 - 2*(xx + yy)
    return R

def yaw_from_R(R_WS: np.ndarray) -> np.ndarray:
    ex = np.array([1.0,0.0,0.0])
    e = (R_WS @ ex[:,None]).squeeze(-1)
    return np.arctan2(e[:,1], e[:,0])

def apply_yaw_align(R_WS: np.ndarray, yaw_delta: float | np.ndarray) -> np.ndarray:
    y = np.asarray(yaw_delta, dtype=float)
    if y.ndim == 0:
        y = np.full((R_WS.shape[0],), float(y))
    cz, sz = np.cos(y), np.sin(y)
    zeros = np.zeros_like(cz)
    ones = np.ones_like(cz)
    Rz = np.stack([
        np.stack([cz,-sz,zeros], axis=1),
        np.stack([sz, cz,zeros], axis=1),
        np.stack([zeros,zeros,ones], axis=1),
    ], axis=1)
    return np.einsum('tij,tjk->tik', Rz, R_WS)

def world_vec(R_WS: np.ndarray, v_S: np.ndarray) -> np.ndarray:
    return (R_WS @ v_S[...,None]).squeeze(-1)

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
        return np.convolve(x, k, mode='same')
    return np.vstack([np.convolve(x[:,i], k, mode='same') for i in range(x.shape[1])]).T

def slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    dot = float(np.dot(q0, q1))
    q1c = q1.copy()
    if dot < 0.0:
        q1c = -q1c; dot = -dot
    if dot > 0.9995:
        v = q0 + u*(q1c - q0)
        return v / (np.linalg.norm(v) + 1e-12)
    th0 = np.arccos(np.clip(dot, -1.0, 1.0))
    s0 = np.sin((1.0 - u)*th0) / np.sin(th0)
    s1 = np.sin(u*th0) / np.sin(th0)
    return s0*q0 + s1*q1c

def resample_quat(t_src: np.ndarray, Q_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    Q_src = normalize_quat(Q_src)
    out = np.zeros((len(t_dst), 4), dtype=float)
    i = 0
    for k, tk in enumerate(t_dst):
        while i+1 < len(t_src) and not (t_src[i] <= tk <= t_src[i+1]):
            i += 1
        if i+1 >= len(t_src):
            out[k] = Q_src[-1]
            continue
        t0, t1 = t_src[i], t_src[i+1]
        u = 0.0 if t1 == t0 else float((tk - t0) / (t1 - t0))
        out[k] = slerp(Q_src[i], Q_src[i+1], float(np.clip(u, 0.0, 1.0)))
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
    for i in range(Tn-1):
        dt = float(max(1e-6, t[i+1] - t[i]))
        dR = R[i].T @ R[i+1]
        tr = float(np.trace(dR))
        c = np.clip((tr - 1.0)/2.0, -1.0, 1.0)
        angle = float(np.arccos(c))
        if angle < 1e-6:
            axis = np.array([0.0,0.0,0.0])
        else:
            denom = 2.0*np.sin(angle)
            axis = np.array([
                (dR[2,1]-dR[1,2])/denom,
                (dR[0,2]-dR[2,0])/denom,
                (dR[1,0]-dR[0,1])/denom,
            ])
        w[i] = axis * (angle/dt)
    if Tn > 1:
        w[-1] = w[-2]
    return w

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

def _signed_angle(a: np.ndarray, b: np.ndarray, axis: np.ndarray, eps: float = 1e-9) -> np.ndarray:
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

def jcs_hip_angles(R_pelvis: np.ndarray, R_femur: np.ndarray, side: str = 'L') -> np.ndarray:
    """Compute hip angles (flexion, adduction, internal rotation) in radians using a
    Grood–Suntay-inspired Joint Coordinate System.

    Inputs are world-to-segment rotation matrices for pelvis and femur, shape (T,3,3).
    side: 'L' or 'R' adjusts the pelvis ML axis so adduction is positive toward midline.

    Returns array (T,3) in radians: [flex, add, rot].
    """
    R_p = np.asarray(R_pelvis, dtype=float)
    R_f = np.asarray(R_femur, dtype=float)
    Tn = R_p.shape[0]
    if Tn == 0:
        return np.zeros((0,3), dtype=float)
    # Pelvis anatomical axes in world
    x_p = R_p[:, :, 0]  # forward
    y_p = R_p[:, :, 1]  # left
    z_p = R_p[:, :, 2]  # up
    # Adjust ML axis for right so that adduction positive is toward midline
    i_ml = y_p if (str(side).upper() == 'L') else -y_p
    # Femur long axis (approx): segment z-axis
    k_f = R_f[:, :, 2]
    k_f = _normalize(k_f)
    i_ml = _normalize(i_ml)
    # Floating axis j = k_f x i_ml
    j_float = _normalize(np.cross(k_f, i_ml))
    # Flexion: rotate about floating axis; measure in sagittal plane of pelvis
    # Compute projection of femur long axis onto plane orthogonal to i_ml
    k_perp = _normalize(_proj_on_plane(k_f, i_ml))
    # Reference in that plane: pelvis z axis projected into same plane
    z_perp = _normalize(_proj_on_plane(z_p, i_ml))
    # Positive flexion when k_perp rotates toward pelvis x (forward)
    flex = _signed_angle(z_perp, k_perp, i_ml)
    # Adduction: angle between femur long axis and pelvis sagittal plane; sign via i_ml
    # Using asin of component along i_ml bounded to [-pi/2,pi/2]
    add = np.arcsin(np.clip(np.sum(k_f * i_ml, axis=-1), -1.0, 1.0))
    # Internal rotation about femur long axis k_f: compare projected pelvis x axes
    x_perp = _normalize(_proj_on_plane(x_p, k_f))
    # Femur transverse reference: femur x-axis projected on plane normal k_f
    x_f = R_f[:, :, 0]
    xf_perp = _normalize(_proj_on_plane(x_f, k_f))
    rot = _signed_angle(x_perp, xf_perp, k_f)
    return np.stack([flex, add, rot], axis=1)

def jcs_knee_angles(R_femur: np.ndarray, R_tibia: np.ndarray, side: str = 'L') -> np.ndarray:
    """Compute knee angles (flexion, adduction(varus negative), internal rotation) in radians
    using a Grood–Suntay-inspired JCS.

    Inputs are world-to-segment rotation matrices for femur and tibia, shape (T,3,3).
    side: 'L' or 'R' adjusts the femur ML axis so adduction is positive toward midline for each limb.
    Returns array (T,3) in radians: [flex, add, rot].
    """
    R_f = np.asarray(R_femur, dtype=float)
    R_t = np.asarray(R_tibia, dtype=float)
    Tn = R_f.shape[0]
    if Tn == 0:
        return np.zeros((0,3), dtype=float)
    # Femur axes
    x_f = R_f[:, :, 0]  # AP (approx flexion axis reference)
    y_f = R_f[:, :, 1]  # ML (left)
    z_f = R_f[:, :, 2]  # long axis (prox-dist)
    # Adjust ML axis as above so that adduction positive moves toward midline
    i_ml = y_f if (str(side).upper() == 'L') else -y_f
    # Tibia long axis
    k_t = _normalize(R_t[:, :, 2])
    i_ml = _normalize(i_ml)
    # Floating axis
    j_float = _normalize(np.cross(k_t, i_ml))
    # Flexion: projection of tibia long axis onto plane orthogonal to i_ml, vs femur z axis projected
    kt_perp = _normalize(_proj_on_plane(k_t, i_ml))
    zf_perp = _normalize(_proj_on_plane(z_f, i_ml))
    flex = _signed_angle(zf_perp, kt_perp, i_ml)
    # Adduction (varus/valgus): component along i_ml
    add = np.arcsin(np.clip(np.sum(k_t * i_ml, axis=-1), -1.0, 1.0))
    # Internal rotation about tibia long axis: compare projected femur x-axis and tibia x-axis
    x_t = R_t[:, :, 0]
    xf_perp = _normalize(_proj_on_plane(x_f, k_t))
    xt_perp = _normalize(_proj_on_plane(x_t, k_t))
    rot = _signed_angle(xf_perp, xt_perp, k_t)
    return np.stack([flex, add, rot], axis=1)
