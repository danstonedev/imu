from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation as R
from core.math.kinematics import hip_angles_xyz, knee_angles_xyz


def rot_x(th):
    return R.from_euler('x', th).as_matrix()

def rot_y(th):
    return R.from_euler('y', th).as_matrix()

def rot_z(th):
    return R.from_euler('z', th).as_matrix()


def test_hip_angles_xyz_synthetic():
    Rp = np.eye(3)[None, ...]
    # Neutral
    Rf_neutral = np.eye(3)[None, ...]
    a0 = hip_angles_xyz(Rp, Rf_neutral, side='L')
    np.testing.assert_allclose(a0, np.zeros((1,3)), atol=1e-7)

    # 30 deg flex only (about +X)
    th = np.deg2rad(30.0)
    Rf = (rot_x(th))[None, ...]
    a = hip_angles_xyz(Rp, Rf, side='L')
    np.testing.assert_allclose(a[0,0], th, atol=1e-7)
    np.testing.assert_allclose(a[0,1:], 0.0, atol=1e-7)

    # 15 deg adduction only (about +Y for L)
    th = np.deg2rad(15.0)
    Rf = (rot_y(th))[None, ...]
    a = hip_angles_xyz(Rp, Rf, side='L')
    np.testing.assert_allclose(a[0,1], th, atol=1e-7)
    np.testing.assert_allclose([a[0,0], a[0,2]], 0.0, atol=1e-7)

    # 20 deg internal rotation only (about +Z)
    th = np.deg2rad(20.0)
    Rf = (rot_z(th))[None, ...]
    a = hip_angles_xyz(Rp, Rf, side='L')
    np.testing.assert_allclose(a[0,2], th, atol=1e-7)
    np.testing.assert_allclose(a[0,:2], 0.0, atol=1e-7)


def test_knee_angles_xyz_synthetic():
    Rf = np.eye(3)[None, ...]
    # Neutral
    Rt0 = np.eye(3)[None, ...]
    a0 = knee_angles_xyz(Rf, Rt0, side='L')
    np.testing.assert_allclose(a0, np.zeros((1,3)), atol=1e-7)

    # 45 deg knee flexion
    th = np.deg2rad(45.0)
    Rt = (rot_x(th))[None, ...]
    a = knee_angles_xyz(Rf, Rt, side='L')
    np.testing.assert_allclose(a[0,0], th, atol=1e-7)
    np.testing.assert_allclose(a[0,1:], 0.0, atol=1e-7)

    # 5 deg knee adduction (valgus/varus), L side uses +
    th = np.deg2rad(5.0)
    Rt = (rot_y(th))[None, ...]
    a = knee_angles_xyz(Rf, Rt, side='L')
    np.testing.assert_allclose(a[0,1], th, atol=1e-7)
    np.testing.assert_allclose([a[0,0], a[0,2]], 0.0, atol=1e-7)

    # 15 deg internal rotation
    th = np.deg2rad(15.0)
    Rt = (rot_z(th))[None, ...]
    a = knee_angles_xyz(Rf, Rt, side='L')
    np.testing.assert_allclose(a[0,2], th, atol=1e-7)
    np.testing.assert_allclose(a[0,:2], 0.0, atol=1e-7)
