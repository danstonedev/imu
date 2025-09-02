from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfiltfilt

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

    - x can be (T,) or (T,D)
    - fc_hz in ~[0.02, 0.05] for gait
    - Uses sosfiltfilt for numerical robustness.
    """
    x = np.asarray(x, dtype=float)
    if fs_hz <= 0 or x.size == 0:
        return x.copy()
    wn = max(1e-4, min(0.99, fc_hz / (0.5 * fs_hz)))
    sos = butter(order, wn, btype='highpass', output='sos')
    if x.ndim == 1:
        return sosfiltfilt(sos, x, axis=0)
    return np.vstack([sosfiltfilt(sos, x[:, i], axis=0) for i in range(x.shape[1])]).T


def _unwrap_phase(a: np.ndarray) -> np.ndarray:
    """Unwrap angle-like series in radians along axis 0."""
    return np.unwrap(np.asarray(a, dtype=float), axis=0)

def apply_yaw_drift_correction(
    yaw_pelvis: np.ndarray,
    yaw_femur: np.ndarray,
    fs_hz: float | None = None,
    alpha: float = 0.5,
    lp_fc_hz: float = 0.03,
    order: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Share low-frequency yaw between pelvis and femur to reduce relative drift.

    - Unwrap both yaw series, low-pass each (Butterworth, sosfiltfilt) to get LF trend.
    - Blend LF trends: shared = (1-alpha)*pelvis_LF + alpha*femur_LF (alpha=0.5 equal).
    - Offset each series by (shared - its_LF) to align slow drift. Re-wrap at the end.
    """
    yp = _unwrap_phase(yaw_pelvis)
    yf = _unwrap_phase(yaw_femur)
    if yp.shape != yf.shape:
        n = min(yp.shape[0], yf.shape[0])
        yp = yp[:n]
        yf = yf[:n]
    if fs_hz is None or fs_hz <= 0:
        # Fallback to simple mean-based sharing
        lp_p = float(np.mean(yp))
        lp_f = float(np.mean(yf))
        shared = (1.0 - alpha) * lp_p + alpha * lp_f
        yp_corr = yp + (shared - lp_p)
        yf_corr = yf + (shared - lp_f)
        return yp_corr, yf_corr
    # Low-pass each
    wn = max(1e-4, min(0.99, lp_fc_hz / (0.5 * fs_hz)))
    sos = butter(order, wn, btype='lowpass', output='sos')
    yp_lf = sosfiltfilt(sos, yp, axis=0)
    yf_lf = sosfiltfilt(sos, yf, axis=0)
    shared = (1.0 - alpha) * yp_lf + alpha * yf_lf
    yp_corr = yp + (shared - yp_lf)
    yf_corr = yf + (shared - yf_lf)
    return yp_corr, yf_corr


def stridewise_debias(x: np.ndarray, stride_indices: list[tuple[int,int]], min_samples: int | None = 0) -> np.ndarray:
    """Subtract mean per stride from signal to remove slow drift/bias.

    - x: (T,D) or (T,)
    - stride_indices: list of (i0,i1) half-open windows per stride
    - min_samples: if provided, skip strides shorter than this; useful for noisy edges
    """
    X = np.asarray(x, dtype=float)
    out = X.copy()
    if X.size == 0 or not stride_indices:
        return out
    def _apply_window(i0: int, i1: int):
        if i1 <= i0:
            return
        if min_samples is not None and (i1 - i0) < int(min_samples):
            return
        if X.ndim == 1:
            out[i0:i1] -= np.mean(out[i0:i1])
        else:
            out[i0:i1] -= np.mean(out[i0:i1], axis=0, keepdims=True)
    for (s, e) in stride_indices:
        i0 = max(0, int(s)); i1 = min(out.shape[0], int(e))
        _apply_window(i0, i1)
    return out
