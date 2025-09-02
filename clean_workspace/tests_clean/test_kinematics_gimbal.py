from __future__ import annotations
import numpy as np
from core.math.kinematics import hip_angles_xyz


def test_euler_xyz_near_gimbal_lock():
    # Construct relative rotation with Y near 90 degrees
    # Start from identity and rotate about Y by 89.5 deg, small X,Z perturbations
    deg = np.deg2rad
    T = 100
    Y = deg(89.5) + deg(0.2)*np.sin(np.linspace(0, 2*np.pi, T))
    X = deg(1.0)*np.sin(np.linspace(0, 4*np.pi, T))
    Z = deg(0.5)*np.cos(np.linspace(0, 3*np.pi, T))
    # Build rotation matrices for pelvis and femur such that relative = Rxyz(X,Y,Z)
    def Rxyz(x,y,z):
        cx,sx = np.cos(x), np.sin(x)
        cy,sy = np.cos(y), np.sin(y)
        cz,sz = np.cos(z), np.sin(z)
        Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
        Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
        Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
        return Rz@Ry@Rx
    Rf = np.stack([Rxyz(X[i],Y[i],Z[i]) for i in range(T)], axis=0)
    Rp = np.repeat(np.eye(3)[None,:,:], T, axis=0)
    e = hip_angles_xyz(Rp, Rf, side='L')
    assert np.all(np.isfinite(e))
    # reconstruct and check small error
    def Rxyz_batch(a):
        out = []
        for xi, yi, zi in a:
            cx,sx = np.cos(xi), np.sin(xi)
            cy,sy = np.cos(yi), np.sin(yi)
            cz,sz = np.cos(zi), np.sin(zi)
            Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
            Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
            Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
            out.append(Rz@Ry@Rx)
        return np.stack(out, axis=0)
    Rrec = Rxyz_batch(e)
    err = np.linalg.norm(Rrec - Rf, axis=(1,2))
    assert np.percentile(err, 95) < 1e-2
