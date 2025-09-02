from __future__ import annotations
import numpy as np
from .kinematics import world_vec
from ..config.constants import (
    FEMUR_LEN_FRACTION,
    FEMUR_MASS_FRACTION,
    FEMUR_LEN_MIN,
    FEMUR_LEN_MAX,
    DAMPING_COEFF,
    G,
)

__all__ = ["hip_inverse_dynamics", "hip_jcs_from_R", "resolve_in_jcs"]


def hip_inverse_dynamics(
    t: np.ndarray,
    R_femur: np.ndarray,
    omega_femur_S: np.ndarray,
    acc_femur_S: np.ndarray,
    height_m: float,
    mass_kg: float,
) -> np.ndarray:
    """Teaching-grade hip moment estimate in the WORLD frame.

    Returns
    - M_world: (T,3) moments in world axes [Nx, Ny, Nz] (Nm)

    Inputs
    - t: (T,) seconds
    - R_femur: (T,3,3) femur orientation world<-segment
    - omega_femur_S: (T,3) angular velocity in segment frame (rad/s)
    - acc_femur_S: (T,3) linear acceleration in segment frame (m/s^2)
    - height_m, mass_kg: for anthropometric scaling

    Model assumptions
    - Femur as a uniform rod: COM at 50% length, I = m*l^2/3 about transverse axes
    - No ground reaction forces (GRF), no hip joint translations
    - Linear term uses r x m*a_com with gravity explicitly added in world
    - Simple viscous damping term on angular velocity (DAMPING_COEFF)

    To express moments in the joint coordinate system (JCS), call resolve_in_jcs
    with R_WJ = hip_jcs_from_R(...).
    """
    t = np.asarray(t, dtype=float)
    if t.size < 2:
        return np.zeros((t.size, 3), dtype=float)

    l_th = float(max(FEMUR_LEN_MIN, min(FEMUR_LEN_MAX, FEMUR_LEN_FRACTION * height_m)))
    m_th = FEMUR_MASS_FRACTION * float(max(1e-6, mass_kg))
    r_com = 0.5 * l_th
    I_rod = (m_th * (l_th**2)) / 3.0

    omega_f_W = world_vec(R_femur, omega_femur_S)
    acc_f_W = world_vec(R_femur, acc_femur_S)

    # Angular acceleration
    if t.size >= 3:
        alpha_f_W = np.vstack(
            [np.gradient(omega_f_W[:, i], t, edge_order=2) for i in range(3)]
        ).T
    else:
        dt = np.gradient(t)
        alpha_f_W = np.gradient(omega_f_W, axis=0) / dt[:, None]

    M_inertial = I_rod * alpha_f_W

    e3_W = R_femur[:, :, 2]
    r_W = e3_W * r_com
    a_com_W = acc_f_W + G
    M_lin = np.cross(r_W, m_th * a_com_W)

    M = M_inertial + M_lin - DAMPING_COEFF * omega_f_W
    return M.astype(float)


def hip_jcs_from_R(R_pelvis: np.ndarray, R_femur: np.ndarray) -> np.ndarray:
    """Return world->joint rotation for resolving moments in JCS.

    Here we use the femur frame as a proxy JCS for teaching. If a dedicated
    joint frame is available, replace this mapping accordingly.
    """
    return R_femur


def resolve_in_jcs(M_world: np.ndarray, R_WJ: np.ndarray) -> np.ndarray:
    """Resolve world-frame moments into joint frame via R_WJ^T M_world.

    Shapes: M_world (T,3), R_WJ (T,3,3) where columns of R_WJ are joint axes in world.
    """
    Rt = np.transpose(R_WJ, (0, 2, 1))
    return (Rt @ M_world[..., None]).squeeze(-1)
