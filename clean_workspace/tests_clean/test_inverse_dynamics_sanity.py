from __future__ import annotations
import numpy as np
from core.math.inverse_dynamics import hip_inverse_dynamics


def test_inverse_dynamics_constant_alpha_matches_Ia():
    fs = 200.0
    t = np.arange(0, 2, 1/fs)
    # Femur fixed in world: R=I, omega about X increases linearly -> constant alpha
    R = np.repeat(np.eye(3)[None,:,:], t.size, axis=0)
    alpha = 5.0  # rad/s^2 about X
    omega = np.zeros((t.size,3))
    omega[:,0] = alpha * (t - t[0])
    acc = np.zeros_like(omega)
    height = 1.75
    mass = 75.0
    M_W = hip_inverse_dynamics(t, R, omega, acc, height, mass)
    # Expect inertial term I*alpha on X; linear/gravity part small since acc=0 and r x m*g handled inside
    # Just check that mean Mx is within 10x to account for gravity cross term
    Mx = M_W[:,0]
    # Compute nominal I of femur rod
    l = 0.245*height
    m = 0.10*mass
    I = m*(l**2)/3.0
    expected = I*alpha
    rel_err = abs(np.mean(Mx) - expected)/max(1e-6, abs(expected))
    assert rel_err < 0.1
