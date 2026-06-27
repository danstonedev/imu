
"""
hip_inverse_dynamics.py
-----------------------
Minimal 3D inverse-dynamics utilities to compute ankle/knee/hip net moments
from IMU-derived segment kinematics plus GRF/CoP (force plate, insole, or IMU-only estimate).

You provide per-frame segment kinematics (R, omega, alpha, r_COM, a_COM) and joint positions.
We provide:
- De Leva anthropometric helpers (masses, COM fractions, radii of gyration)
- World inertia transform
- GRF/CoP helpers for force plate & insole
- IMU-only GRF/CoP estimator (teaching/demo)
- Bottom-up Newton-Euler for foot→shank→thigh to get joint reaction forces & moments

Author: ChatGPT (GPT-5 Thinking) for Dr. Dan Stone
License: MIT
"""

from dataclasses import dataclass
import numpy as np

GRAVITY = np.array([0.0, 0.0, 9.81])  # +Z up (world)

# -------------------------------
# Data structures
# -------------------------------

@dataclass
class SegmentKinematics:
    """Kinematics for one segment at one frame, all expressed in world frame.
    R_WB: (3,3) rotation matrix world<-body
    omega_W: (3,) angular velocity [rad/s]
    alpha_W: (3,) angular acceleration [rad/s^2]
    r_COM_W: (3,) COM position [m]
    a_COM_W: (3,) COM linear acceleration [m/s^2] (gravity NOT included)
    r_prox_W: (3,) proximal joint position [m] (e.g., ankle for shank, knee for thigh, hip for pelvis)
    r_dist_W: (3,) distal joint position [m]
    """
    R_WB: np.ndarray
    omega_W: np.ndarray
    alpha_W: np.ndarray
    r_COM_W: np.ndarray
    a_COM_W: np.ndarray
    r_prox_W: np.ndarray
    r_dist_W: np.ndarray


@dataclass
class InertialProps:
    """Inertial properties about COM in the segment *body* frame."""
    mass: float               # [kg]
    I_body_COM: np.ndarray    # (3,3) inertia tensor about COM in body axes


@dataclass
class JointLoads:
    """Forces and moments at a joint acting on the proximal segment (Newton-Euler convention)."""
    F: np.ndarray   # (3,) reaction force [N]
    M: np.ndarray   # (3,) net internal moment [N·m]


# -------------------------------
# Anthropometrics (De Leva 1996)
# -------------------------------

def deleva_lower_limb_fractions():
    """Return mass fraction, COM location fraction (from proximal joint toward distal),
    and radii of gyration fractions (about segment length) for foot, shank, thigh.
    The numbers are typical approximations from de Leva (1996) adjustments to Dempster.
    """
    # Fractions by segment of whole-body mass (mB) and segment length (L)
    # COM from proximal joint along the segment axis toward distal (fraction of L)
    # Radii of gyration as fraction of segment length (about principal axes)
    return {
        "foot":  dict(mass=0.0145, com=0.5,   kx=0.475, ky=0.302, kz=0.475),
        "shank": dict(mass=0.0465, com=0.433, kx=0.302, ky=0.267, kz=0.302),
        "thigh": dict(mass=0.1416, com=0.433, kx=0.323, ky=0.323, kz=0.323),
    }


def inertia_from_radii(m, L, kx, ky, kz):
    """Approximate inertia tensor about COM in body axes assuming principal axes aligned
    with segment frame and radii of gyration (fractions of segment length)."""
    # Principal inertias: I = m * (k*L)^2 for each axis
    Ixx = m * (kx * L)**2
    Iy  = m * (ky * L)**2
    Izz = m * (kz * L)**2
    return np.diag([Ixx, Iy, Izz])


def make_inertial_props(height_m, mass_kg, seg_length_m, seg_name):
    params = deleva_lower_limb_fractions()[seg_name]
    m = params["mass"] * mass_kg
    I_body = inertia_from_radii(m, seg_length_m, params["kx"], params["ky"], params["kz"])
    return InertialProps(mass=m, I_body_COM=I_body)


# -------------------------------
# Frames & transforms
# -------------------------------

def world_inertia(R_WB, I_body_COM):
    """Rotate body inertia to world frame: I_W = R * I_body * R^T"""
    return R_WB @ I_body_COM @ R_WB.T


def cross(a, b):
    return np.cross(a, b)


# -------------------------------
# GRF / CoP helpers
# -------------------------------

def cop_from_forceplate(FP_F, FP_M, plate_height=0.0):
    """Compute CoP in plate frame given plate force/moment at plate origin.
    FP_F: (3,) [N], FP_M: (3,) [N·m]
    Using standard formulas:
      CoP_x = (-M_y - h * F_x) / F_z
      CoP_y = ( M_x - h * F_y) / F_z
    Returns CoP in plate frame (3,) with z=0.
    """
    Fx, Fy, Fz = FP_F
    Mx, My, Mz = FP_M
    if abs(Fz) < 1e-6:
        return np.array([0.0, 0.0, 0.0])
    cop_x = (-My - plate_height * Fx) / Fz
    cop_y = ( Mx - plate_height * Fy) / Fz
    return np.array([cop_x, cop_y, 0.0])


def transform_point(R_WP, p_WP, p_P):
    """Transform point from plate (P) to world (W): p_W = R_WP @ p_P + p_WP"""
    return R_WP @ p_P + p_WP


def transform_vec(R_WP, v_P):
    """Rotate vector from plate to world."""
    return R_WP @ v_P


def cop_from_insole(pressures, areas, coords_F):
    """Compute vertical GRF and CoP in the foot frame from insole pressure map.
    pressures: (n,) [Pa], areas: (n,) [m^2], coords_F: (n,3) foot-frame coordinates of sensel centers
    Returns: F_F (3,) with Fz only, CoP_F (3,) (z=0)
    """
    w = pressures * areas  # N per sensel (since Pa * m^2 = N)
    Fz = np.sum(w)
    if Fz < 1e-6:
        return np.zeros(3), np.zeros(3)
    r = coords_F  # (n,3)
    cop = np.sum(r * w[:,None], axis=0) / Fz
    F = np.array([0.0, 0.0, Fz])
    cop[2] = 0.0
    return F, cop


def estimate_grf_cop_from_imu(acc_pelvis_W, stance_L, stance_R, mass_kg, foot_len=0.24, heel_to_ankle=0.07):
    """Very rough GRF/CoP estimator for teaching/demo when no force hardware exists.
    acc_pelvis_W: (T,3) pelvis linear acceleration in world frame with gravity already removed
    stance_L, stance_R: (T,) booleans or weights in [0,1]
    Returns:
      F_W: (T,2,3) forces for feet [L=0,R=1]
      CoP_F: (T,2,3) CoP in each foot frame (x,0,0) rocker model (set y,z=0)
      u_progress: (T,2) normalized heel→toe progress
    """
    T = acc_pelvis_W.shape[0]
    F_W = np.zeros((T,2,3))
    CoP_F = np.zeros((T,2,3))
    u_prog = np.zeros((T,2))
    sz = stance_L + stance_R + 1e-9
    Ftot = np.c_[mass_kg*acc_pelvis_W[:,0], mass_kg*acc_pelvis_W[:,1], mass_kg*(acc_pelvis_W[:,2] + GRAVITY[2])]
    wL = (stance_L/sz)[:,None]; wR = (stance_R/sz)[:,None]
    F_W[:,0,:] = wL * Ftot
    F_W[:,1,:] = wR * Ftot

    def rocker(u):
        x_heel = -heel_to_ankle
        x_toe  = foot_len - heel_to_ankle
        return (1-u)*x_heel + u*x_toe

    for k, s in enumerate([stance_L, stance_R]):
        u = np.zeros(T); in_stance=False; t0=0
        for i in range(T):
            if (s[i] > 0.5) and (not in_stance):
                in_stance=True; t0=i
            if (s[i] <= 0.5) and in_stance:
                dur = i - t0
                if dur <= 1:
                    u[t0:i] = 0.0
                else:
                    u[t0:i] = np.linspace(0,1,dur)
                in_stance=False
        if in_stance:
            dur = T - t0
            if dur > 1:
                u[t0:T] = np.linspace(0,1,dur)
        u_prog[:,k] = u
        CoP_F[:,k,0] = rocker(u)
        CoP_F[:,k,1] = 0.0
        CoP_F[:,k,2] = 0.0
    return F_W, CoP_F, u_prog


# -------------------------------
# Newton–Euler inverse dynamics
# -------------------------------

def inverse_dynamics_lowerlimb_3D(kin_foot, kin_shank, kin_thigh,
                                  prop_foot, prop_shank, prop_thigh,
                                  F_GRF_W, r_CoP_W, M_free_W=None):
    """Compute ankle, knee, hip reaction forces & net moments using bottom-up Newton–Euler.
    Inputs per frame:
      kin_* : SegmentKinematics
      prop_*: InertialProps
      F_GRF_W: (3,) ground reaction force [N] applied to the foot (world)
      r_CoP_W: (3,) point of application in world
      M_free_W: (3,) optional free moment about CoP (e.g., plate torsional moment)
    Returns:
      dict with JointLoads for ankle, knee, hip acting on the proximal segment.
    Sign convention: JointLoads refer to the load acting ON the proximal segment by the distal one.
    For the foot segment, ankle loads are returned as the loads acting on the shank by the foot,
    which are opposite of the loads acting on the foot at the ankle.
    """
    # Helper to compute I*alpha + omega x (I*omega) in world
    def H(seg_kin, I_body, m):
        Iw = world_inertia(seg_kin.R_WB, I_body)
        return Iw @ seg_kin.alpha_W + np.cross(seg_kin.omega_W, Iw @ seg_kin.omega_W)

    g = GRAVITY

    # ----- FOOT free-body (unknown ankle load on foot) -----
    mF = prop_foot.mass
    Hf = H(kin_foot, prop_foot.I_body_COM, mF)

    # Forces: F_ankle_on_foot + F_grf + m*g = m*a_COM
    F_ankle_on_foot = mF*(kin_foot.a_COM_W + g) - F_GRF_W

    # Moments about COM: M_ankle_on_foot + r_a2c × F_ankle + M_grf + r_cop2c × F_grf = Hf
    r_a2c = kin_foot.r_prox_W - kin_foot.r_COM_W  # ankle is proximal for foot
    r_cop2c = r_CoP_W - kin_foot.r_COM_W
    M_grf = M_free_W if M_free_W is not None else np.zeros(3)
    M_ankle_on_foot = Hf - np.cross(r_a2c, F_ankle_on_foot) - M_grf - np.cross(r_cop2c, F_GRF_W)

    # Load on SHANK at ankle is equal and opposite
    F_ankle_on_shank = -F_ankle_on_foot
    M_ankle_on_shank = -M_ankle_on_foot

    # ----- SHANK free-body (unknown knee load on shank) -----
    mS = prop_shank.mass
    Hs = H(kin_shank, prop_shank.I_body_COM, mS)

    # Forces: F_knee + F_ankle + m*g = m*a_COM
    F_knee_on_shank = mS*(kin_shank.a_COM_W + g) - F_ankle_on_shank

    # Moments: M_knee + r_k2c × F_knee + M_ankle + r_a2c × F_ankle = Hs
    r_k2c = kin_shank.r_prox_W - kin_shank.r_COM_W    # knee is proximal for shank
    r_a2c_s = kin_shank.r_dist_W - kin_shank.r_COM_W  # ankle is distal for shank
    M_knee_on_shank = Hs - np.cross(r_a2c_s, F_ankle_on_shank) - M_ankle_on_shank - np.cross(r_k2c, F_knee_on_shank)

    # Load on THIGH at knee is equal and opposite
    F_knee_on_thigh = -F_knee_on_shank
    M_knee_on_thigh = -M_knee_on_shank

    # ----- THIGH free-body (unknown hip load on thigh) -----
    mT = prop_thigh.mass
    Ht = H(kin_thigh, prop_thigh.I_body_COM, mT)

    # Forces: F_hip + F_knee + m*g = m*a_COM
    F_hip_on_thigh = mT*(kin_thigh.a_COM_W + g) - F_knee_on_thigh

    # Moments: M_hip + r_h2c × F_hip + M_knee + r_k2c × F_knee = Ht
    r_h2c = kin_thigh.r_prox_W - kin_thigh.r_COM_W   # hip is proximal for thigh
    r_k2c_t = kin_thigh.r_dist_W - kin_thigh.r_COM_W # knee is distal for thigh
    M_hip_on_thigh = Ht - np.cross(r_k2c_t, F_knee_on_thigh) - M_knee_on_thigh - np.cross(r_h2c, F_hip_on_thigh)

    # Return loads acting on proximal segments
    ankle = JointLoads(F=F_ankle_on_shank, M=M_ankle_on_shank)  # load on shank at ankle
    knee  = JointLoads(F=F_knee_on_thigh, M=M_knee_on_thigh)    # load on thigh at knee
    hip   = JointLoads(F=F_hip_on_thigh,  M=M_hip_on_thigh)     # load on pelvis at hip

    return {"ankle": ankle, "knee": knee, "hip": hip}


# -------------------------------
# Batched Newton–Euler (vectorized across time)
# -------------------------------

def inverse_dynamics_lowerlimb_3D_batch(
    R_foot_WB, omega_foot_W, alpha_foot_W, rCOM_foot_W, aCOM_foot_W, r_ankle_W, r_toe_W,
    R_shank_WB, omega_shank_W, alpha_shank_W, rCOM_shank_W, aCOM_shank_W, r_knee_W,
    R_thigh_WB, omega_thigh_W, alpha_thigh_W, rCOM_thigh_W, aCOM_thigh_W, r_hip_W,
    prop_foot, prop_shank, prop_thigh,
    F_GRF_W, r_CoP_W, M_free_W=None
):
    """Vectorized inverse dynamics for foot→shank→thigh over all frames.
    Inputs are arrays over time with shapes:
      R_*: (T,3,3), omega_*, alpha_*: (T,3), rCOM_*, aCOM_*: (T,3)
      joint positions r_ankle_W, r_toe_W, r_knee_W, r_hip_W: (T,3)
      F_GRF_W, r_CoP_W: (T,3)
    Returns: dict with arrays of JointLoads components. Currently returns only hip moment array (T,3).
    """
    T = R_foot_WB.shape[0]
    g = GRAVITY

    # Helper: rotate body inertia to world for every frame
    def world_inertia_batch(Rs, I_body):
        Rt = np.transpose(Rs, (0, 2, 1))
        # Iw = R * I * R^T
        RI = np.einsum('tij,jk->tik', Rs, I_body)
        return np.einsum('tij,tjk->tik', RI, Rt)

    # FOOT
    IwF = world_inertia_batch(R_foot_WB, prop_foot.I_body_COM)
    IwF_omega = np.einsum('tij,tj->ti', IwF, omega_foot_W)
    Hf = np.einsum('tij,tj->ti', IwF, alpha_foot_W) + np.cross(omega_foot_W, IwF_omega)
    mF = prop_foot.mass
    F_ankle_on_foot = mF * (aCOM_foot_W + g) - F_GRF_W
    r_a2c_f = r_ankle_W - rCOM_foot_W
    r_cop2c = r_CoP_W - rCOM_foot_W
    if M_free_W is None:
        M_free = np.zeros((T,3))
    else:
        M_free = M_free_W
    M_ankle_on_foot = Hf - np.cross(r_a2c_f, F_ankle_on_foot) - M_free - np.cross(r_cop2c, F_GRF_W)
    F_ankle_on_shank = -F_ankle_on_foot
    M_ankle_on_shank = -M_ankle_on_foot

    # SHANK
    IwS = world_inertia_batch(R_shank_WB, prop_shank.I_body_COM)
    IwS_omega = np.einsum('tij,tj->ti', IwS, omega_shank_W)
    Hs = np.einsum('tij,tj->ti', IwS, alpha_shank_W) + np.cross(omega_shank_W, IwS_omega)
    mS = prop_shank.mass
    F_knee_on_shank = mS * (aCOM_shank_W + g) - F_ankle_on_shank
    r_k2c_s = r_knee_W - rCOM_shank_W
    r_a2c_s = r_ankle_W - rCOM_shank_W
    M_knee_on_shank = Hs - np.cross(r_a2c_s, F_ankle_on_shank) - M_ankle_on_shank - np.cross(r_k2c_s, F_knee_on_shank)
    F_knee_on_thigh = -F_knee_on_shank
    M_knee_on_thigh = -M_knee_on_shank

    # THIGH
    IwT = world_inertia_batch(R_thigh_WB, prop_thigh.I_body_COM)
    IwT_omega = np.einsum('tij,tj->ti', IwT, omega_thigh_W)
    Ht = np.einsum('tij,tj->ti', IwT, alpha_thigh_W) + np.cross(omega_thigh_W, IwT_omega)
    mT = prop_thigh.mass
    F_hip_on_thigh = mT * (aCOM_thigh_W + g) - F_knee_on_thigh
    r_h2c_t = r_hip_W - rCOM_thigh_W
    r_k2c_t = r_knee_W - rCOM_thigh_W
    M_hip_on_thigh = Ht - np.cross(r_k2c_t, F_knee_on_thigh) - M_knee_on_thigh - np.cross(r_h2c_t, F_hip_on_thigh)

    return {"hip_M_W": M_hip_on_thigh}


# -------------------------------
# Utility: build COM from joints (linear along segment) if needed
# -------------------------------

def com_from_joints_linear(r_prox_W, r_dist_W, frac_from_prox):
    """Return COM position along the line from proximal to distal joint."""
    return r_prox_W + frac_from_prox * (r_dist_W - r_prox_W)


# -------------------------------
# Resolve moments into Joint Coordinate System (optional)
# -------------------------------

def resolve_moment_in_jcs(M_world, R_WJ):
    """Rotate world moment vector into joint coordinate system (JCS)."""
    R_JW = R_WJ.T
    return R_JW @ M_world
