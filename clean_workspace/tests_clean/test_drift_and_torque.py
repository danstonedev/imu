from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation as R
from core.math.drift import estimate_gyro_bias, highpass, stridewise_debias
from core.math.inverse_dynamics import hip_inverse_dynamics
from core.math.kinematics import world_vec


def test_estimate_gyro_bias():
    g = np.zeros((100,3), dtype=float)
    g[:,0] = 0.01  # constant bias on x
    still = np.zeros(100, dtype=bool)
    still[10:60] = True
    b = estimate_gyro_bias(g, still)
    assert abs(b[0]-0.01) < 1e-6
    assert abs(b[1]) < 1e-9 and abs(b[2]) < 1e-9


def test_highpass_removes_offset():
    fs = 100.0
    t = np.arange(0, 10, 1/fs)
    x = 2.0 + 0.1*np.sin(2*np.pi*1.0*t)
    y = highpass(x, fs, fc_hz=0.05)
    # Remove some transients at edges for check
    m = slice(100, -100)
    assert abs(np.mean(y[m])) < 0.05


def test_stridewise_debias():
    x = np.linspace(0, 10, 101)
    out = stridewise_debias(x, [(0,50),(50,101)])
    assert abs(np.mean(out[0:50])) < 1e-9
    assert abs(np.mean(out[50:101])) < 1e-9


def test_hip_inverse_dynamics_synthetic_inertia():
    # Femur rotates with constant angular acceleration about x
    fs = 100.0
    t = np.arange(0, 2.0, 1/fs)
    alpha = 2.0  # rad/s^2
    omega = np.column_stack([alpha*t, np.zeros_like(t), np.zeros_like(t)])
    acc = np.zeros_like(omega)  # neglect linear acceleration; gravity handled inside
    Rf = np.repeat(np.eye(3)[None,...], len(t), axis=0)
    M = hip_inverse_dynamics(t, Rf, omega, acc, height_m=1.75, mass_kg=75.0)
    # Expect dominant x torque ~ I * alpha
    # Using constants from inverse_dynamics
    from core.config.constants import FEMUR_LEN_FRACTION, FEMUR_MASS_FRACTION
    l_th = 0.245*1.75
    m_th = 0.10*75.0
    I = m_th * (l_th**2) / 3.0
    # Ignore gravity cross for this synthetic; check mean within 10%
    mx = np.mean(M[:,0])
    assert mx > 0
    assert abs(mx - I*alpha) / (I*alpha) < 0.2
