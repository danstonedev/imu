from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt

__all__ = [
    "estimate_gyro_bias",
    "highpass",
    "apply_yaw_drift_correction",
    "stridewise_debias",
]


def estimate_gyro_bias(gyro_S: np.ndarray, still_mask: np.ndarray | None) -> np.ndarray:
    """Estimate constant gyro bias per axis from still windows.

    gyro_S: (T,3) body-frame angular rate [rad/s]
    still_mask: (T,) boolean mask where motion is minimal. If None or no True,
    returns zeros.
    """
    g = np.asarray(gyro_S, dtype=float)
    if still_mask is None:
        return np.zeros(3, dtype=float)
    m = np.asarray(still_mask, dtype=bool)
    if g.shape[0] != m.shape[0] or not np.any(m):
        return np.zeros(3, dtype=float)
    return np.mean(g[m], axis=0)


def highpass(x: np.ndarray, fs_hz: float, fc_hz: float = 0.03, order: int = 2) -> np.ndarray:
    """Zero-phase high-pass Butterworth filter for drift removal.

    x can be (T,) or (T,D). fc_hz in ~[0.02,0.05] typically for gait.
    """
    x = np.asarray(x, dtype=float)
    if fs_hz <= 0 or x.size == 0:
        return x.copy()
    wc = max(1e-4, min(0.49, fc_hz / (fs_hz * 0.5)))
    b, a = butter(order, wc, btype='highpass')
    if x.ndim == 1:
        return filtfilt(b, a, x, axis=0)
    else:
        return np.vstack([filtfilt(b, a, x[:, i], axis=0) for i in range(x.shape[1])]).T


def apply_yaw_drift_correction(yaw_pelvis: np.ndarray, yaw_femur: np.ndarray, alpha: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Share low-frequency yaw between pelvis and femur to reduce relative drift.

    Returns corrected (yaw_pelvis_corr, yaw_femur_corr). alpha in [0,1] controls
    blend; 0.5 shares equally.
    """
    yp = np.asarray(yaw_pelvis, dtype=float)
    yf = np.asarray(yaw_femur, dtype=float)
    if yp.shape != yf.shape:
        n = min(yp.shape[0], yf.shape[0])
        yp = yp[:n]
        yf = yf[:n]
    # compute low-frequency components (mean here as simplest proxy)
    lp_p = np.mean(yp)
    lp_f = np.mean(yf)
    lp_shared = (1.0 - alpha) * lp_p + alpha * lp_f
    yp_corr = yp + (lp_shared - lp_p)
    yf_corr = yf + (lp_shared - lp_f)
    return yp_corr, yf_corr


def stridewise_debias(x: np.ndarray, stride_indices: list[tuple[int,int]]) -> np.ndarray:
    """Subtract mean per stride from signal to remove slow drift/bias.

    x: (T,D) or (T,). stride_indices: list of (i0,i1) inclusive or half-open windows
    per stride. Output shape matches x.
    """
    X = np.asarray(x, dtype=float)
    out = X.copy()
    if X.size == 0 or not stride_indices:
        return out
    if X.ndim == 1:
        for (i0, i1) in stride_indices:
            i0 = max(0, int(i0)); i1 = min(len(out), int(i1))
            if i1 > i0:
                out[i0:i1] -= np.mean(out[i0:i1])
        return out
    else:
        for (i0, i1) in stride_indices:
            i0 = max(0, int(i0)); i1 = min(out.shape[0], int(i1))
            if i1 > i0:
                out[i0:i1] -= np.mean(out[i0:i1], axis=0, keepdims=True)
        return out
