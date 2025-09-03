from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

__all__ = [
    "BaselineConfig",
    "unwrap_cols",
    "wrap_pi",
    "lowpass",
    "highpass",
    "yaw_share_timevarying",
    "stridewise_debias",
    "apply_baseline_correction",
]


class BaselineConfig:
    """Configuration for baseline/drift correction of Euler angles.

    Parameters
    - fs_hz: Sampling frequency in Hz
    - use_yaw_share: Whether to enable time-varying yaw sharing (requires yaws)
    - yaw_share_fc_hz: Low-pass cutoff for extracting LF trend to share (Hz)
    - stride_debias_axes: Axes to debias per stride (subset of {"X","Y","Z"})
    - highpass_fc_hz: Optional HP cutoff for final drift removal (Hz); None to disable
    - min_stride_samples: Minimum samples within a stride window to apply debias
    - allow_stack: If False, do not combine yaw-share with stride-debias (fail-safe)
    - rewrap_after: If True, wrap output angles to (-pi, pi] after correction
    - use_calibration_windows: Whether to apply calibration window-based corrections
    - calibration_axes: Axes to apply calibration window corrections (subset of {"X","Y","Z"})
    """

    def __init__(
        self,
        fs_hz: float,
        use_yaw_share: bool = True,
        yaw_share_fc_hz: float = 0.05,  # 0.03–0.05 Hz typical
        stride_debias_axes: tuple[str, ...] = ("Y", "Z"),
        highpass_fc_hz: float | None = None,  # OFF by default unless set
        min_stride_samples: int = 30,
        allow_stack: bool = False,
        rewrap_after: bool = True,
        use_calibration_windows: bool = True,
        calibration_axes: tuple[str, ...] = ("X", "Y", "Z"),
    ):
        self.fs_hz = float(fs_hz)
        self.use_yaw_share = bool(use_yaw_share)
        self.yaw_share_fc_hz = float(yaw_share_fc_hz)
        self.stride_debias_axes = tuple(str(a).upper() for a in stride_debias_axes)
        self.highpass_fc_hz = None if highpass_fc_hz is None else float(highpass_fc_hz)
        self.min_stride_samples = int(min_stride_samples)
        self.allow_stack = bool(allow_stack)
        self.rewrap_after = bool(rewrap_after)
        self.use_calibration_windows = bool(use_calibration_windows)
        self.calibration_axes = tuple(str(a).upper() for a in calibration_axes)


def unwrap_cols(A: np.ndarray) -> np.ndarray:
    """Unwrap each column of angle array (T, D) in radians along time axis.

    Returns a copy.
    """
    X = np.asarray(A, dtype=float)
    if X.ndim == 1:
        return np.unwrap(X)
    out = X.copy()
    for j in range(X.shape[1]):
        out[:, j] = np.unwrap(out[:, j])
    return out


def wrap_pi(A: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi] per element.

    For display/CSV. Processing should operate on unwrapped angles.
    """
    X = np.asarray(A, dtype=float)
    return (X + np.pi) % (2.0 * np.pi) - np.pi


def _mk_sos(fc_hz: float, fs_hz: float, btype: str, order: int = 2):
    wn = max(1e-5, min(0.999, float(fc_hz) / (0.5 * float(fs_hz))))
    return butter(int(order), wn, btype=btype, output="sos")


def lowpass(x: np.ndarray, fs_hz: float, fc_hz: float, order: int = 2) -> np.ndarray:
    """Zero-phase low-pass (Butterworth SOS) along time axis.

    - x: (T,) or (T,D)
    - fc_hz: cutoff in Hz
    """
    if x is None:
        return x
    X = np.asarray(x, dtype=float)
    if X.size == 0 or fs_hz <= 0 or fc_hz <= 0:
        return X.copy()
    sos = _mk_sos(fc_hz, fs_hz, btype="lowpass", order=order)
    if X.ndim == 1:
        return sosfiltfilt(sos, X, axis=0)
    return np.vstack([sosfiltfilt(sos, X[:, j], axis=0) for j in range(X.shape[1])]).T


def highpass(x: np.ndarray, fs_hz: float, fc_hz: float, order: int = 2) -> np.ndarray:
    """Zero-phase high-pass (Butterworth SOS) along time axis.

    - x: (T,) or (T,D)
    - fc_hz: cutoff in Hz (e.g., 0.02–0.05 for gait)
    """
    if x is None:
        return x
    X = np.asarray(x, dtype=float)
    if X.size == 0 or fs_hz <= 0 or fc_hz <= 0:
        return X.copy()
    # Complementary form to preserve unity gain at high frequencies:
    # HP(x) = x - LP(x). Using the same zero-phase Butterworth LP avoids
    # slight passband amplification seen in direct IIR HP designs with filtfilt.
    L = lowpass(X, fs_hz, fc_hz, order=order)
    return X - L


def yaw_share_timevarying(
    yaw_p: np.ndarray, yaw_f: np.ndarray, cfg: BaselineConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Unwrap → LP both traces → blend low-freq components → return corrected yaws.

    Keeps HF content by adding back (original - LF) per trace.
    """
    yp = np.unwrap(np.asarray(yaw_p, dtype=float))
    yf = np.unwrap(np.asarray(yaw_f, dtype=float))
    n = int(min(yp.shape[0], yf.shape[0]))
    yp = yp[:n]
    yf = yf[:n]
    if not np.isfinite(cfg.fs_hz) or cfg.fs_hz <= 0:
        # Fallback to mean share
        lp_p = float(np.mean(yp)) if yp.size else 0.0
        lp_f = float(np.mean(yf)) if yf.size else 0.0
        shared = 0.5 * (lp_p + lp_f)
        return yp + (shared - lp_p), yf + (shared - lp_f)
    yp_lf = lowpass(yp, cfg.fs_hz, cfg.yaw_share_fc_hz, order=2)
    yf_lf = lowpass(yf, cfg.fs_hz, cfg.yaw_share_fc_hz, order=2)
    shared = 0.5 * (yp_lf + yf_lf)
    yp_corr = yp + (shared - yp_lf)
    yf_corr = yf + (shared - yf_lf)
    return yp_corr, yf_corr


def stridewise_debias(
    X: np.ndarray,
    stride_indices: list[tuple[int, int]],
    axes=(1, 2),
    min_samples: int = 30,
    extend_across: bool = True,
) -> np.ndarray:
    """Subtract per-stride mean on selected columns with optional extension across time.

    - X: (T, D) angles in radians (unwrapped)
    - stride_indices: list of (i0, i1) half-open windows
    - axes: tuple of integer column indices to debias (default Y,Z)
    - min_samples: skip windows shorter than this
    - extend_across: if True (default), build a continuous offset(t) by linear
      interpolation of per-stride means at stride midpoints, apply across entire time.
      This eliminates start/stop steps. If False, subtract only within strides.
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A[:, None]
    out = A.copy()
    if A.size == 0 or not stride_indices:
        return out
    D = out.shape[1]
    axes_idx = tuple(i for i in axes if 0 <= int(i) < D)
    # Collect per-stride means and centers
    centers: list[int] = []
    means: list[np.ndarray] = []
    for s, e in stride_indices:
        i0 = max(0, int(s))
        i1 = min(out.shape[0], int(e))
        if i1 - i0 < int(min_samples):
            continue
        m = np.mean(out[i0:i1, axes_idx], axis=0)
        centers.append(int((i0 + i1) // 2))
        means.append(m.astype(float))
    if not centers:
        return out
    if extend_across:
        # Interpolate mean offsets at centers over entire time for each axis
        T = out.shape[0]
        centers_arr = np.asarray(centers, dtype=float)
        M = np.vstack(means)  # (K, len(axes_idx))
        grid = np.arange(T, dtype=float)
        off = np.zeros((T, len(axes_idx)), dtype=float)
        for j in range(len(axes_idx)):
            off[:, j] = np.interp(
                grid, centers_arr, M[:, j], left=M[0, j], right=M[-1, j]
            )
        # Subtract across whole series on selected axes
        for k, ax in enumerate(axes_idx):
            out[:, ax] -= off[:, k]
    else:
        # Original behavior: subtract inside each stride only
        for s, e in stride_indices:
            i0 = max(0, int(s))
            i1 = min(out.shape[0], int(e))
            if i1 - i0 < int(min_samples):
                continue
            m = np.mean(out[i0:i1, axes_idx], axis=0, keepdims=True)
            out[i0:i1, axes_idx] -= m
    return out


def _axes_to_idx(names: tuple[str, ...]) -> tuple[int, ...]:
    m = {"X": 0, "Y": 1, "Z": 2}
    return tuple(m.get(s.upper(), -1) for s in names if s.upper() in m)


def apply_baseline_correction(
    t: np.ndarray,
    euler_xyz: np.ndarray,  # (T,3) from hip/knee angles
    stride_indices: list[tuple[int, int]] | None,
    cfg: BaselineConfig,
    # optional context for yaw-share if caller provides segment yaw traces
    yaw_pelvis: np.ndarray | None = None,
    yaw_femur: np.ndarray | None = None,
    # calibration windows for window-based corrections
    calibration_windows: list[tuple[float, float]] | None = None,
    # Diagnostic isolation mode: None = normal behavior using cfg;
    # "none" | "yaw_share_only" | "stride_debias_only" | "highpass_only" | "calibration_only"
    mode: str | None = None,
) -> np.ndarray:
    """
    Fail-safe pipeline:
        unwrap → (optional) yaw-share OR stride-debias (based on cfg.allow_stack) → (optional) HP → (optional) rewrap

    - If cfg.use_yaw_share and yaws provided, prefer yaw-share. If allow_stack=False,
        skip stride-debias to avoid double DC adjustments. If True, both can run.
    - High-pass is OFF unless cfg.highpass_fc_hz is set to a positive value.
        - If cfg.rewrap_after=True, wrap to (-pi, pi] for safe plotting/CSV.
            Default is False to preserve continuity for downstream processing/tests.
    """
    A = np.asarray(euler_xyz, dtype=float)
    if A.ndim != 2 or A.shape[1] < 1:
        return A.copy()

    # Diagnostic isolation branch
    if isinstance(mode, str):
        mode_l = mode.lower().strip()
        valid = {"none", "yaw_share_only", "stride_debias_only", "highpass_only", "calibration_only"}
        if mode_l not in valid:
            raise ValueError(f"Invalid mode '{mode}'. Expected one of {sorted(valid)}")

        # Guardrails: don't accidentally stack and don't sneak HP unless requested
        assert not cfg.allow_stack, "allow_stack must be False for isolation tests"
        if mode_l not in ("highpass_only", "calibration_only"):
            assert cfg.highpass_fc_hz in (
                None,
                0,
                0.0,
            ), "cfg.highpass_fc_hz must be None/0 when not testing highpass_only"

        Au = unwrap_cols(A)

        if mode_l == "none":
            out = Au
        elif mode_l == "stride_debias_only":
            axes_idx = _axes_to_idx(cfg.stride_debias_axes)
            out = stridewise_debias(
                Au,
                stride_indices or [],
                axes=axes_idx,
                min_samples=cfg.min_stride_samples,
            )
        elif mode_l == "highpass_only":
            # Use provided cutoff if positive, else a conservative default
            fc = None if cfg.highpass_fc_hz is None else float(cfg.highpass_fc_hz)
            if not (isinstance(fc, (int, float)) and fc > 0):
                fc = 0.03  # default HP cutoff (Hz)
            out = highpass(Au, float(cfg.fs_hz), float(fc), order=2)
        elif mode_l == "yaw_share_only":
            # Simulate effect on axial (Z) via change in relative yaw trend
            out = Au.copy()
            if yaw_pelvis is not None and yaw_femur is not None:
                yp, yf = np.asarray(yaw_pelvis, dtype=float), np.asarray(
                    yaw_femur, dtype=float
                )
                yp_c, yf_c = yaw_share_timevarying(yp, yf, cfg)
                # delta of relative yaw (femur - pelvis)
                d_rel = (yf_c - yp_c) - (yf - yp)
                d_rel = d_rel[: out.shape[0]]
                if out.shape[1] >= 3 and d_rel.size == out.shape[0]:
                    out[:, 2] = out[:, 2] + d_rel
        elif mode_l == "calibration_only":
            # Apply only calibration window-based corrections
            out = Au.copy()
            if calibration_windows and cfg.use_calibration_windows:
                axes_idx = _axes_to_idx(cfg.calibration_axes)
                out = apply_calibration_window_correction(
                    t, out, calibration_windows, cfg.fs_hz, axes=axes_idx
                )
        else:
            out = Au

        # Always rewrap for diagnostics
        return wrap_pi(out)

    # Normal behavior (no explicit diagnostic mode): previous fail-safe pipeline
    Au = unwrap_cols(A)

    did_share = False
    # Yaw-share (time-varying LF alignment) if explicitly requested and data provided
    if cfg.use_yaw_share and yaw_pelvis is not None and yaw_femur is not None:
        _yp, _yf = yaw_share_timevarying(yaw_pelvis, yaw_femur, cfg)
        did_share = True

    # Calibration window-based corrections for start/end periods
    if calibration_windows and cfg.use_calibration_windows:
        axes_idx = _axes_to_idx(cfg.calibration_axes)
        Au = apply_calibration_window_correction(
            t, Au, calibration_windows, cfg.fs_hz, axes=axes_idx
        )

    # Stride-wise DC removal on selected axes (default Y,Z), unless stacking is disallowed
    if stride_indices and (cfg.allow_stack or not did_share):
        axes_idx = _axes_to_idx(cfg.stride_debias_axes)
        Au = stridewise_debias(
            Au, stride_indices, axes=axes_idx, min_samples=cfg.min_stride_samples
        )

    # Optional high-pass to clean residual drift (all axes)
    if (
        cfg.highpass_fc_hz is not None
        and float(cfg.highpass_fc_hz) > 0
        and cfg.fs_hz > 0
    ):
        Au = highpass(Au, cfg.fs_hz, float(cfg.highpass_fc_hz), order=2)

    if cfg.rewrap_after:
        Au = wrap_pi(Au)
    return Au


def apply_calibration_window_correction(
    t: np.ndarray,
    euler_xyz: np.ndarray,
    calibration_windows: list[dict] | list[tuple[float, float]],
    fs_hz: float,
    axes: tuple[int, ...] = (0, 1, 2),
) -> np.ndarray:
    """
    Apply calibration window-based corrections to Euler angles.
    
    Computes the mean offset across ALL calibration windows for each axis
    and removes it from the entire signal.
    
    Parameters:
    - t: Time array (seconds)
    - euler_xyz: Euler angles array (T,3) 
    - calibration_windows: List of calibration window dicts or (start_time, end_time) tuples
    - fs_hz: Sampling frequency
    - axes: Which axes to correct (0=X, 1=Y, 2=Z)
    
    Returns:
    - Corrected Euler angles
    """
    A = np.asarray(euler_xyz, dtype=float)
    if A.ndim != 2 or A.shape[1] < 1 or not calibration_windows:
        return A.copy()
    
    A_corr = A.copy()
    t_arr = np.asarray(t, dtype=float)
    
    # For each axis, collect data from all calibration windows and compute overall offset
    for axis_idx in axes:
        if axis_idx < A.shape[1]:
            cal_data_all = []
            
            # Collect data from all calibration windows
            for window in calibration_windows:
                # Handle both dict format (from pipeline) and tuple format (from tests)
                if isinstance(window, dict):
                    start_time = window['start_s']
                    end_time = window['end_s']
                else:
                    start_time, end_time = window
                    
                mask = (t_arr >= start_time) & (t_arr <= end_time)
                if np.any(mask):
                    cal_data = A_corr[mask, axis_idx]
                    if len(cal_data) > 0:
                        cal_data_all.extend(cal_data)
            
            # Compute overall offset from all calibration windows combined
            if cal_data_all:
                offset = np.nanmean(cal_data_all)
                # Remove offset from entire signal
                A_corr[:, axis_idx] -= offset
                    
    return A_corr
