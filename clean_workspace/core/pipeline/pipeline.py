from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any
from ..math.kinematics import (
    quats_to_R_batch, yaw_from_R, apply_yaw_align, resample_quat, resample_vec,
    moving_avg, world_vec, estimate_fs, gyro_from_quat,
    jcs_hip_angles, jcs_knee_angles, resample_side_to_femur_time,
)
from ..math.inverse_dynamics import hip_inverse_dynamics, hip_jcs_from_R, resolve_in_jcs
from ..math.conventions import enforce_lr_conventions
from .io_utils import read_xsens_bytes, extract_kinematics, extract_kinematics_ex
from .calibration import detect_still, calibration_windows_secs, calibrate_bias_trimmed, calibrate_all
from .unified_gait import detect_gait_cycles, gait_cycle_analysis

__all__ = ["run_pipeline_clean", "compute_yaw_reference"]

def compute_yaw_reference(tP: np.ndarray, RP_raw: np.ndarray, stillP: np.ndarray | None, cal_windows: list[dict] | None) -> tuple[float, str]:
    """
    Compute a stable yaw reference using only the start still window when available,
    else the end window, otherwise fall back to median over the provided still mask
    (or global median if mask empty).

    Returns (yaw_ref_rad, source_tag) where source_tag in {"start","end","global"}.
    """
    yawP = yaw_from_R(RP_raw)
    # Prefer explicit calibration windows (seconds) labeled 'start' or 'end'
    if cal_windows:
        # Build helper to median yaw over a time window
        def median_yaw_in_seconds(t0: float, t1: float) -> float | None:
            if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
                return None
            i0 = int(np.searchsorted(tP, float(t0), side='left'))
            i1 = int(np.searchsorted(tP, float(t1), side='right'))
            i0 = max(0, min(i0, len(yawP)))
            i1 = max(i0+1, min(i1, len(yawP)))
            return float(np.median(yawP[i0:i1])) if (i1 > i0) else None

        # Extract start and end windows, if present
        start_win = next((w for w in cal_windows if w.get('label') == 'start'), None)
        end_win   = next((w for w in cal_windows if w.get('label') == 'end'), None)

        if start_win is not None:
            m = median_yaw_in_seconds(float(start_win['start_s']), float(start_win['end_s']))
            if m is not None:
                return m, "start"
        if end_win is not None:
            m = median_yaw_in_seconds(float(end_win['start_s']), float(end_win['end_s']))
            if m is not None:
                return m, "end"

    # Fallbacks: median over detected pelvis-still mask, then global median
    sel = np.asarray(stillP, dtype=bool) if stillP is not None else None
    if sel is not None and sel.any():
        return float(np.median(yawP[sel[:len(yawP)]])), "global"
    return (float(np.median(yawP)) if len(yawP) else 0.0), "global"

def run_pipeline_clean(data: dict, height_m: float, mass_kg: float, options: dict) -> dict:
    do_cal = bool(options.get('do_cal', True)) if isinstance(options, dict) else True
    yaw_align_flag = bool(options.get('yaw_align', True)) if isinstance(options, dict) else True
    baseline_mode = (options.get('baseline_mode') if isinstance(options, dict) else None) or 'linear'

    # Check if data contains paths (str) or bytes
    is_paths = isinstance(next(iter(data.values())), str)

    if is_paths:
        pel_bytes = Path(data["pelvis"]).read_bytes()
        lf_bytes  = Path(data["lfemur"]).read_bytes()
        rf_bytes  = Path(data["rfemur"]).read_bytes()
        lt_bytes  = Path(data["ltibia"]).read_bytes()
        rt_bytes  = Path(data["rtibia"]).read_bytes()
    else:
        pel_bytes = data["pelvis"]
        lf_bytes  = data["lfemur"]
        rf_bytes  = data["rfemur"]
        lt_bytes  = data["ltibia"]
        rt_bytes  = data["rtibia"]

    pel = read_xsens_bytes(pel_bytes)
    lf  = read_xsens_bytes(lf_bytes)
    rf  = read_xsens_bytes(rf_bytes)
    lt  = read_xsens_bytes(lt_bytes)
    rt  = read_xsens_bytes(rt_bytes)

    tP, qP, gP, aP, metaP = extract_kinematics_ex(pel)
    tLf, qLf, gLf, aLf, _ = extract_kinematics_ex(lf)
    tRf, qRf, gRf, aRf, _ = extract_kinematics_ex(rf)
    tLt, qLt, gLt, aLt, _ = extract_kinematics_ex(lt)
    tRt, qRt, gRt, aRt, _ = extract_kinematics_ex(rt)

    # Fallback: derive gyro from quaternions where missing
    if gLf is None:
        gLf = gyro_from_quat(tLf, qLf)
    if gRf is None:
        gRf = gyro_from_quat(tRf, qRf)
    if gLt is None:
        gLt = gyro_from_quat(tLt, qLt)
    if gRt is None:
        gRt = gyro_from_quat(tRt, qRt)

    # Ensure pelvis gyro present for still detection; derive if missing
    if gP is None:
        gP = gyro_from_quat(tP, qP)
    stillP, calmeta = detect_still(tP, gP, aP)
    Fs = calmeta['Fs']

    # windows in seconds on pelvis time
    cal_windows = calibration_windows_secs(tP, stillP, Fs)

    # One-shot calibration: per-segment biases and yaw reference (pelvis driven)
    RP_raw = quats_to_R_batch(qP)
    if do_cal:
        seg_for_bias = {
            'pelvis':  (gP, aP),
            'L_femur': (gLf, aLf), 'R_femur': (gRf, aRf),
            'L_tibia': (gLt, aLt), 'R_tibia': (gRt, aRt),
        }
        cal = calibrate_all(tP, RP_raw, stillP, cal_windows, seg_for_bias)
        def _apply_bias(name, g, a):
            gb, ab = cal.biases.get(name, (np.zeros(3), np.zeros(3)))
            return (None if g is None else g - gb), (None if a is None else a - ab)
        gP,  aP  = _apply_bias('pelvis',  gP,  aP)
        gLf, aLf = _apply_bias('L_femur', gLf, aLf)
        gRf, aRf = _apply_bias('R_femur', gRf, aRf)
        gLt, aLt = _apply_bias('L_tibia', gLt, aLt)
        gRt, aRt = _apply_bias('R_tibia', gRt, aRt)
        yaw_ref, yaw_src = cal.yaw_ref, cal.yaw_source
    else:
        yaw_ref, yaw_src = 0.0, "global"

    RLf_raw = quats_to_R_batch(qLf)
    RRf_raw = quats_to_R_batch(qRf)
    RLt_raw = quats_to_R_batch(qLt)
    RRt_raw = quats_to_R_batch(qRt)

    # If calibration didn't run or yaw_align disabled, compute or null yaw as needed
    if yaw_align_flag and not do_cal:
        yaw_ref, yaw_src = compute_yaw_reference(tP, RP_raw, stillP, cal_windows)
    if not yaw_align_flag:
        yaw_ref, yaw_src = 0.0, "global"

    # ---------------------------------------------
    # Global overlap trimming (apply before any resampling)
    # Ensures all segments operate on a common time window to avoid edge extrapolation
    # ---------------------------------------------
    def _overlap_window(ts: list[np.ndarray]) -> tuple[float, float]:
        t0s = [float(x[0]) for x in ts if x is not None and len(x) > 0]
        t1s = [float(x[-1]) for x in ts if x is not None and len(x) > 0]
        if not t0s or not t1s:
            return 0.0, 0.0
        return max(t0s), min(t1s)

    def _trim_time_series(t: np.ndarray, *arrs: np.ndarray | None, start: float, end: float) -> tuple:
        if t is None or len(t) == 0:
            return (t, *arrs)
        if end <= start:
            return (t, *arrs)
        # inclusive bounds
        m = (t >= start) & (t <= end)
        if not np.any(m):
            return (t, *arrs)
        idx = np.where(m)[0]
        t2 = t[idx]
        trimmed = []
        for a in arrs:
            if a is None:
                trimmed.append(None)
            else:
                trimmed.append(a[idx])
        return tuple([t2, *trimmed])

    ov_start, ov_end = _overlap_window([tP, tLf, tRf, tLt, tRt])
    overlap_meta = {
        'start_s': float(ov_start),
        'end_s': float(ov_end),
        'duration_s': float(max(0.0, ov_end - ov_start)),
        'applied': bool(ov_end > ov_start)
    }

    if overlap_meta['applied']:
        # Trim pelvis (and still mask to stay aligned)
        # First, trim time and arrays, then trim stillP by the same mask
        tP_old = tP
        tP, qP, gP, aP = _trim_time_series(tP, qP, gP, aP, start=ov_start, end=ov_end)
        # Map stillP from old indexing to new by slicing with mask
        if stillP is not None and len(tP_old) == len(stillP):
            mP = (tP_old >= ov_start) & (tP_old <= ov_end)
            stillP = stillP[mP]

        # Left femur/tibia
        tLf, qLf, gLf, aLf = _trim_time_series(tLf, qLf, gLf, aLf, start=ov_start, end=ov_end)
        tLt, qLt, gLt, aLt = _trim_time_series(tLt, qLt, gLt, aLt, start=ov_start, end=ov_end)
        # Right femur/tibia
        tRf, qRf, gRf, aRf = _trim_time_series(tRf, qRf, gRf, aRf, start=ov_start, end=ov_end)
        tRt, qRt, gRt, aRt = _trim_time_series(tRt, qRt, gRt, aRt, start=ov_start, end=ov_end)

    # Shared resampling per side (to femur time)
    # Type guards to satisfy static checkers (these should be true at runtime)
    assert gLf is not None and aLf is not None and gLt is not None and aLt is not None
    assert gRf is not None and aRf is not None and gRt is not None and aRt is not None
    L_res = resample_side_to_femur_time(tLf, qLf, gLf, aLf, tLt, qLt, gLt, aLt, tP, qP)
    R_res = resample_side_to_femur_time(tRf, qRf, gRf, aRf, tRt, qRt, gRt, aRt, tP, qP)
    tL = L_res.t; qLf_res = L_res.q_femur; gLf_res = L_res.gyro_femur; aLf_res = L_res.acc_femur
    qLt_res = L_res.q_tibia; gLt_res = L_res.gyro_tibia; aLt_res = L_res.acc_tibia; qP_L = L_res.q_pelvis
    tR = R_res.t; qRf_res = R_res.q_femur; gRf_res = R_res.gyro_femur; aRf_res = R_res.acc_femur
    qRt_res = R_res.q_tibia; gRt_res = R_res.gyro_tibia; aRt_res = R_res.acc_tibia; qP_R = R_res.q_pelvis

    RLf = quats_to_R_batch(qLf_res)
    RLt = quats_to_R_batch(qLt_res)
    RRf = quats_to_R_batch(qRf_res)
    RRt = quats_to_R_batch(qRt_res)
    RP_L = quats_to_R_batch(qP_L)
    RP_R = quats_to_R_batch(qP_R)

    # Apply the same yaw alignment to pelvis and limb segments to preserve relative axial rotations
    if yaw_align_flag:
        # Rotate by -yaw_ref to bring pelvis heading toward x-axis
        RP_L = apply_yaw_align(RP_L, -yaw_ref)
        RP_R = apply_yaw_align(RP_R, -yaw_ref)
        RLf = apply_yaw_align(RLf, -yaw_ref)
        RLt = apply_yaw_align(RLt, -yaw_ref)
        RRf = apply_yaw_align(RRf, -yaw_ref)
        RRt = apply_yaw_align(RRt, -yaw_ref)

    # -----------------------------
    # Joint angles (Grood–Suntay-inspired JCS), radians -> degrees
    # -----------------------------
    hipL_rad = jcs_hip_angles(RP_L, RLf, side='L')
    hipR_rad = jcs_hip_angles(RP_R, RRf, side='R')
    kneeL_rad = jcs_knee_angles(RLf, RLt, side='L')
    kneeR_rad = jcs_knee_angles(RRf, RRt, side='R')
    rad2deg = 180.0 / np.pi
    hipL_deg = (hipL_rad * rad2deg).astype(np.float32)
    hipR_deg = (hipR_rad * rad2deg).astype(np.float32)
    kneeL_deg = (kneeL_rad * rad2deg).astype(np.float32)
    kneeR_deg = (kneeR_rad * rad2deg).astype(np.float32)

    gLf_res = moving_avg(gLf_res, win=7)
    gLt_res = moving_avg(gLt_res, win=7)
    gRf_res = moving_avg(gRf_res, win=7)
    gRt_res = moving_avg(gRt_res, win=7)

    aLf_res = moving_avg(aLf_res, win=7)
    aLt_res = moving_avg(aLt_res, win=7)
    aRf_res = moving_avg(aRf_res, win=7)
    aRt_res = moving_avg(aRt_res, win=7)

    omegaL_shank_W = world_vec(RLt, gLt_res)
    aL_W = world_vec(RLt, aLt_res)
    omegaR_shank_W = world_vec(RRt, gRt_res)
    aR_W = world_vec(RRt, aRt_res)

    # Make gravity handling explicit from headers, with optional safety net
    from ..config.constants import G as G_W
    acc_is_free = bool(metaP.get('acc_is_free', False))
    def gravity_handle(aW: np.ndarray) -> tuple[np.ndarray, dict]:
        if aW is None or len(aW) == 0:
            return aW, {'applied': False, 'mode': 'none'}
        if acc_is_free:
            return aW.astype(np.float32), {'applied': False, 'mode': 'freeacc_headers'}
        # Assume raw accelerometer: subtract gravity in world frame
        return (aW - G_W).astype(np.float32), {'applied': True, 'mode': 'subtract_g_world'}
    aL_free_W, acc_corr_L = gravity_handle(aL_W)
    aR_free_W, acc_corr_R = gravity_handle(aR_W)

    # Unified biomechanical gait cycle detection
    gait_results = detect_gait_cycles(
        t_left=tL,
        accel_left=aL_free_W,
        gyro_left=omegaL_shank_W,
        t_right=tR,
        accel_right=aR_free_W,
        gyro_right=omegaR_shank_W,
        fs=Fs
    )
    
    # Extract heel strikes for cycle analysis (indices)
    contacts_L = np.asarray(gait_results['heel_strikes_left'], dtype=int)
    contacts_R = np.asarray(gait_results['heel_strikes_right'], dtype=int)

    MhipL_W = hip_inverse_dynamics(tL, RLf, gLf_res, aLf_res, height_m, mass_kg)
    MhipR_W = hip_inverse_dynamics(tR, RRf, gRf_res, aRf_res, height_m, mass_kg)

    RJL = hip_jcs_from_R(RP_L, RLf)
    RJR = hip_jcs_from_R(RP_R, RRf)
    MhipL_J = resolve_in_jcs(MhipL_W, RJL)
    MhipR_J = resolve_in_jcs(MhipR_W, RJR)

    # Centralize sign conventions for moments and angles
    anglesL_dict = {'hip': hipL_deg, 'knee': kneeL_deg}
    anglesR_dict = {'hip': hipR_deg, 'knee': kneeR_deg}
    MhipL_J, MhipR_J, outL_angles, outR_angles = enforce_lr_conventions(MhipL_J, MhipR_J, anglesL_dict, anglesR_dict)
    hipL_deg, kneeL_deg = outL_angles['hip'], outL_angles['knee']
    hipR_deg, kneeR_deg = outR_angles['hip'], outR_angles['knee']

    # Baseline subtraction using still windows (median per component).
    # IMPORTANT: Align pelvis-based still windows to each limb's time origin.
    # tL and tR were rebased to their own starts (tLf[0], tRf[0]).
    # Compute offsets so we can map pelvis windows (relative to tP[0]) onto limb timelines.
    tP0 = tP - tP[0]
    stillP_f = stillP.astype(float)
    deltaL = float(tLf[0] - tP[0])
    deltaR = float(tRf[0] - tP[0])

    def resample_still(t_side: np.ndarray, delta: float) -> np.ndarray:
        if len(tP0) < 2:
            return np.zeros_like(t_side, dtype=bool)
        # Map limb-local time (since its own start) to pelvis-local time by adding delta
        s = np.interp(t_side + delta, tP0, stillP_f, left=stillP_f[0], right=stillP_f[-1])
        # Conservative threshold and morphological closing (1-2 samples) to avoid pinholes
        thr = 0.75
        m = s >= thr
        if m.size >= 3:
            # close 1-sample holes: True False True -> True True True
            holes = (~m[1:-1]) & (m[:-2] & m[2:])
            if np.any(holes):
                m[1:-1][holes] = True
        return m

    maskL = resample_still(tL, deltaL)
    maskR = resample_still(tR, deltaR)

    # Fallback: build masks from explicit calibration windows if still masks are sparse
    def mask_from_windows(t_side: np.ndarray, delta: float):
        if not cal_windows:
            return np.zeros_like(t_side, dtype=bool)
        m = np.zeros_like(t_side, dtype=bool)
        for w in cal_windows:
            # Windows are in absolute seconds; convert to pelvis-local (minus tP[0]) then to limb-local by subtracting delta
            s = float(w['start_s'] - tP[0]) - delta
            e = float(w['end_s'] - tP[0]) - delta
            m |= (t_side >= s) & (t_side <= e)
        return m

    if not maskL.any():
        maskL = mask_from_windows(tL, deltaL)
    if not maskR.any():
        maskR = mask_from_windows(tR, deltaR)

    def subtract_baseline(MJ: np.ndarray, t_side: np.ndarray, mask: np.ndarray):
        if baseline_mode == 'none':
            z = np.zeros(3, dtype=MJ.dtype)
            return MJ, z, z
        if not mask.any():
            z = np.zeros(3, dtype=MJ.dtype)
            return MJ, z, z
        idx = np.where(mask)[0]
        k = max(10, int(0.2 * len(idx)))
        k = min(k, len(idx))
        b0 = np.median(MJ[idx[:k]], axis=0)
        b1 = np.median(MJ[idx[-k:]], axis=0)
        if baseline_mode == 'constant':
            base_t = b0
            return MJ - base_t, b0, b0
        # Linear baseline across time (default)
        T = float(max(1e-6, t_side[-1] - t_side[0]))
        s = ((t_side - t_side[0]) / T).astype(MJ.dtype)
        base_t = b0 + s[:, None] * (b1 - b0)
        return MJ - base_t, b0, b1

    MhipL_J, baseL0, baseL1 = subtract_baseline(MhipL_J, tL, maskL)
    MhipR_J, baseR0, baseR1 = subtract_baseline(MhipR_J, tR, maskR)

    Lmx, Lmy, Lmz = MhipL_J[:,0], MhipL_J[:,1], MhipL_J[:,2]
    Rmx, Rmy, Rmz = MhipR_J[:,0], MhipR_J[:,1], MhipR_J[:,2]

    # BESR: Baseline-Estimate on Scalar Resultant
    # Magnitude is non-linear: ||x - b|| != ||x|| - ||b||. To avoid residual gravity offset in |M|,
    # estimate and subtract the baseline on the scalar magnitude itself using still windows.
    def subtract_scalar_baseline(y: np.ndarray, t_side: np.ndarray, mask: np.ndarray):
        if baseline_mode == 'none' or not mask.any() or len(y) < 2:
            return y.astype(y.dtype, copy=True)
        idx = np.where(mask)[0]
        k = max(10, int(0.2 * len(idx)))
        k = min(k, len(idx))
        b0 = float(np.median(y[idx[:k]]))
        b1 = float(np.median(y[idx[-k:]]))
        if baseline_mode == 'constant':
            return (y - b0).astype(np.float32)
        T = float(max(1e-6, t_side[-1] - t_side[0]))
        s = ((t_side - t_side[0]) / T).astype(np.float32)
        base_t = b0 + s * (b1 - b0)
        return (y - base_t).astype(np.float32)

    Lmag_raw = np.linalg.norm(MhipL_J, axis=1).astype(np.float32)
    Rmag_raw = np.linalg.norm(MhipR_J, axis=1).astype(np.float32)
    Lmag = subtract_scalar_baseline(Lmag_raw, tL, maskL)
    Rmag = subtract_scalar_baseline(Rmag_raw, tR, maskR)

    # -----------------------------
    # Angle normalization and sign conventions
    # -----------------------------
    # Normalize joint angles so that the calibration/stance pose averages to ~0 deg
    # using the same still masks as torque baselining. Then enforce flexion-positive
    # convention for both hips and knees by flipping the flexion component.
    def _unwrap_angles_in_deg(A_deg: np.ndarray) -> np.ndarray:
        A = np.asarray(A_deg, dtype=np.float32)
        if A.ndim != 2 or A.shape[1] == 0:
            return A
        out = A.copy()
        for j in range(A.shape[1]):
            rad = np.deg2rad(out[:, j].astype(np.float64))
            rad_u = np.unwrap(rad)
            out[:, j] = np.rad2deg(rad_u).astype(np.float32)
        return out

    def subtract_angle_baseline(A_deg: np.ndarray, t_side: np.ndarray, mask: np.ndarray):
        # Works for shape (T,3). Mirrors subtract_baseline but for degrees.
        if A_deg is None or len(A_deg) == 0:
            z = np.zeros(3, dtype=np.float32)
            return A_deg, z, z
        A = _unwrap_angles_in_deg(A_deg)
        if baseline_mode == 'none' or not mask.any():
            z = np.zeros(3, dtype=np.float32)
            return A, z, z
        idx = np.where(mask)[0]
        k = max(10, int(0.2 * len(idx)))
        k = min(k, len(idx))
        b0 = np.median(A[idx[:k]], axis=0).astype(np.float32)
        b1 = np.median(A[idx[-k:]], axis=0).astype(np.float32)
        if baseline_mode == 'constant':
            base_t = b0
            return (A - base_t).astype(np.float32), b0, b0
        T = float(max(1e-6, t_side[-1] - t_side[0]))
        s = ((t_side - t_side[0]) / T).astype(np.float32)
        base_t = b0 + s[:, None] * (b1 - b0)
        return (A - base_t).astype(np.float32), b0, b1

    hipL_deg, hipL_b0, hipL_b1 = subtract_angle_baseline(hipL_deg, tL, maskL)
    hipR_deg, hipR_b0, hipR_b1 = subtract_angle_baseline(hipR_deg, tR, maskR)
    kneeL_deg, kneeL_b0, kneeL_b1 = subtract_angle_baseline(kneeL_deg, tL, maskL)
    kneeR_deg, kneeR_b0, kneeR_b1 = subtract_angle_baseline(kneeR_deg, tR, maskR)

    # Angle flexion sign already enforced by enforce_lr_conventions

    def resample_mean_sd(arr, n=101):
        if len(arr) < 2:
            return np.zeros(n, np.float32), np.zeros(n, np.float32)
        x = np.linspace(0.0, 1.0, len(arr))
        xi = np.linspace(0.0, 1.0, n)
        mean = np.interp(xi, x, arr)
        w = max(3, len(arr)//20)
        s = pd.Series(arr).rolling(window=w, center=True, min_periods=1).std().to_numpy()
        sd = np.interp(xi, x, s)
        return mean.astype(np.float32), sd.astype(np.float32)

    def mid80_idx(n):
        if n<=2: return 0, n
        m = int(0.1*n); return m, n-m

    i0L, i1L = mid80_idx(len(tL))
    i0R, i1R = mid80_idx(len(tR))

    Lmean, Lsd = resample_mean_sd(Lmx[i0L:i1L])
    Rmean, Rsd = resample_mean_sd(Rmx[i0R:i1R])

    left_csv  = "time_s,L_Mx(Nm),L_My(Nm),L_Mz(Nm),L_Mmag(Nm)\n"  + "\n".join(
        f"{t:.6f},{mx:.6f},{my:.6f},{mz:.6f},{mm:.6f}" for t,mx,my,mz,mm in zip(tL, Lmx, Lmy, Lmz, Lmag)
    )
    right_csv = "time_s,R_Mx(Nm),R_My(Nm),R_Mz(Nm),R_Mmag(Nm)\n" + "\n".join(
        f"{t:.6f},{mx:.6f},{my:.6f},{mz:.6f},{mm:.6f}" for t,mx,my,mz,mm in zip(tR, Rmx, Rmy, Rmz, Rmag)
    )

    # Remove duplicate heel strike extraction since already done above
    # contacts_L and contacts_R already set from gait_results

    # Extract gait events for cycle analysis (indices)
    toe_offs_L = np.asarray(gait_results['toe_offs_left'], dtype=int)
    toe_offs_R = np.asarray(gait_results['toe_offs_right'], dtype=int)
    # Fallback: if toe-offs are missing or clearly insufficient compared to heel strikes, estimate at ~60% of stride
    def estimate_toe_offs_from_hs(hs_idx: np.ndarray) -> np.ndarray:
        hs_idx = np.asarray(hs_idx, dtype=int)
        if hs_idx.size < 2:
            return np.array([], dtype=int)
        est = []
        for a, b in zip(hs_idx[:-1], hs_idx[1:]):
            # Ensure strictly inside window
            k = int(round(a + 0.6 * (b - a)))
            if k > a and k < b:
                est.append(k)
        return np.asarray(est, dtype=int)
    if contacts_L.size >= 2 and (toe_offs_L.size == 0 or toe_offs_L.size < max(1, contacts_L.size - 1)):
        toe_offs_L = estimate_toe_offs_from_hs(contacts_L)
    if contacts_R.size >= 2 and (toe_offs_R.size == 0 or toe_offs_R.size < max(1, contacts_R.size - 1)):
        toe_offs_R = estimate_toe_offs_from_hs(contacts_R)
    
    # Convert event indices to times for meta calculations
    def safe_take(t_series: np.ndarray, idx: np.ndarray) -> np.ndarray:
        if idx is None or len(idx) == 0:
            return np.array([], dtype=float)
        idx = np.asarray(idx, dtype=int)
        idx = idx[(idx >= 0) & (idx < len(t_series))]
        if idx.size == 0:
            return np.array([], dtype=float)
        return t_series[idx]
    hsL_times = safe_take(tL, contacts_L)
    hsR_times = safe_take(tR, contacts_R)
    toL_times = safe_take(tL, toe_offs_L)
    toR_times = safe_take(tR, toe_offs_R)
    
    # Create stance arrays using event TIMES for robust shading
    def create_stance_array_from_times(t: np.ndarray, hs_times: np.ndarray, to_times: np.ndarray) -> np.ndarray:
        stance = np.zeros(len(t), dtype=bool)
        if hs_times is None or len(hs_times) == 0:
            return stance
        for hs_t in hs_times:
            hs_idx = int(np.searchsorted(t, hs_t, side='left'))
            # Find first toe-off time after this heel strike
            later_to = to_times[to_times > hs_t] if to_times is not None and len(to_times) else np.array([], dtype=float)
            if len(later_to) > 0:
                to_t = float(later_to[0])
                to_idx = int(np.searchsorted(t, to_t, side='left'))
                s = max(0, min(hs_idx, len(t)))
                e = max(s, min(to_idx, len(t)))
                stance[s:e] = True
        return stance

    # (times already computed above)

    stance_L = create_stance_array_from_times(tL, hsL_times, toL_times)
    stance_R = create_stance_array_from_times(tR, hsR_times, toR_times)

    # Estimate sampling frequencies for meta
    def _fs_est(t: np.ndarray) -> float:
        if t is None or len(t) < 2:
            return float('nan')
        dt = np.diff(t)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        return float(1.0 / np.median(dt)) if dt.size else float('nan')
    fs_meta = {
        'pelvis': _fs_est(tP),
        'L_femur': _fs_est(tL), 'R_femur': _fs_est(tR),
        'L_tibia': _fs_est(tL), 'R_tibia': _fs_est(tR),
    }
    
    def cycle_mean_sd(t, sig, contacts, toe_offs):
        """Mean and standard deviation for each gait cycle with adaptive edge-drop."""
        try:
            # Adaptive trimming like anchored comparisons: 0 for <6, 1 for 6-11, 2 for 12-17, 3 for >=18
            n_cycles = max(0, int(len(contacts)) - 1)
            drop_each = 0
            if n_cycles > 0:
                drop_each = max(0, min(3, n_cycles // 6))
            mean_cycle, sd_cycle, used, total = gait_cycle_analysis(
                t, sig, contacts, toe_offs,
                drop_first=drop_each, drop_last=drop_each
            )
            # Return the cycle arrays directly (not the mean of the means)
            return mean_cycle, sd_cycle, used, total
        except Exception as e:
            print(f"Cycle analysis failed: {e}")
            # Return default arrays with 101 points for compatibility
            n_points = 101
            shape = (n_points, sig.shape[1]) if sig.ndim > 1 else (n_points,)
            return (np.zeros(shape, dtype=np.float32), 
                    np.zeros(shape, dtype=np.float32), 0, 0)

    def cycles(sig, t, contacts, toe_offs):
        mean, sd, used, total = cycle_mean_sd(t, sig, contacts, toe_offs)
        return {"mean": mean, "sd": sd}, used, total

    LmxC, nL, tLtot = cycles(Lmx, tL, contacts_L, toe_offs_L)
    LmyC, _, _ = cycles(Lmy, tL, contacts_L, toe_offs_L)
    LmzC, _, _ = cycles(Lmz, tL, contacts_L, toe_offs_L)
    LmgC, _, _ = cycles(Lmag, tL, contacts_L, toe_offs_L)
    RmxC, nR, tRtot = cycles(Rmx, tR, contacts_R, toe_offs_R)
    RmyC, _, _ = cycles(Rmy, tR, contacts_R, toe_offs_R)
    RmzC, _, _ = cycles(Rmz, tR, contacts_R, toe_offs_R)
    RmgC, _, _ = cycles(Rmag, tR, contacts_R, toe_offs_R)

    # Angle cycles (temporarily removed here; will be defined after anchored helpers below)
    hip_cycles = None
    knee_cycles = None

    cycles_out = {
        'L': {
            'count_used': nL, 'count_total': tLtot,
            'Mx': LmxC, 'My': LmyC, 'Mz': LmzC, 'Mmag': LmgC,
        },
        'R': {
            'count_used': nR, 'count_total': tRtot,
            'Mx': RmxC, 'My': RmyC, 'Mz': RmzC, 'Mmag': RmgC,
        }
    }

    # Add derived magnitude variants at cycle level: |E[M]| and RMS(|M|)
    def derive_mag_variants_from_cycles(side_dict: dict):
        try:
            mx_m = np.asarray(side_dict['Mx']['mean'], dtype=np.float32)
            my_m = np.asarray(side_dict['My']['mean'], dtype=np.float32)
            mz_m = np.asarray(side_dict['Mz']['mean'], dtype=np.float32)
            # Magnitude of the average vector: |E[M]|
            mag_of_mean = np.sqrt(mx_m*mx_m + my_m*my_m + mz_m*mz_m).astype(np.float32)
            # RMS magnitude: sqrt(E[Mx^2]+E[My^2]+E[Mz^2]) with E[X^2]=Var+Mean^2 ≈ sd^2 + mean^2
            mx_sd = np.asarray(side_dict['Mx']['sd'], dtype=np.float32)
            my_sd = np.asarray(side_dict['My']['sd'], dtype=np.float32)
            mz_sd = np.asarray(side_dict['Mz']['sd'], dtype=np.float32)
            rms_mag = np.sqrt((mx_sd*mx_sd + my_sd*my_sd + mz_sd*mz_sd) + (mx_m*mx_m + my_m*my_m + mz_m*mz_m)).astype(np.float32)
            n = int(max(len(mag_of_mean), len(rms_mag)))
            z = np.zeros(n, dtype=np.float32)
            side_dict['Mmag_of_mean'] = { 'mean': mag_of_mean, 'sd': z }
            side_dict['Mmag_rms'] = { 'mean': rms_mag, 'sd': z }
        except Exception:
            # If anything missing, provide zero arrays to keep schema stable
            n = 101
            z = np.zeros(n, dtype=np.float32)
            side_dict['Mmag_of_mean'] = { 'mean': z, 'sd': z }
            side_dict['Mmag_rms'] = { 'mean': z, 'sd': z }

    derive_mag_variants_from_cycles(cycles_out['L'])
    derive_mag_variants_from_cycles(cycles_out['R'])

    # Build cross-foot comparison anchored to a reference foot's strides
    def phase_resample_by_time(t_series: np.ndarray, sig: np.ndarray, t0: float, t1: float, n: int = 101) -> np.ndarray | None:
        if t1 <= t0:
            return None
        # Find overlap samples and interpolate
        xi = np.linspace(t0, t1, n)
        if sig.ndim == 1:
            return np.interp(xi, t_series, sig)
        # For vector components
        return np.vstack([np.interp(xi, t_series, sig[:,j]) for j in range(sig.shape[1])]).T

    def _build_anchor_segments(t_ref: np.ndarray, hs_times: np.ndarray, min_dur=0.4, max_dur=2.0, drop_first: int | None = None, drop_last: int | None = None):
        # Use heel strike times directly and adaptively drop edge strides
        c = [float(x) for x in hs_times if np.isfinite(x)]
        if len(c) < 2:
            return []
        c.sort()
        segs = []
        for t0, t1 in zip(c[:-1], c[1:]):
            dur = t1 - t0
            if dur < min_dur or dur > max_dur:
                continue
            # ensure segment lies within recorded time
            if t0 < float(t_ref[0]) or t1 > float(t_ref[-1]):
                continue
            segs.append((float(t0), float(t1)))
        n = len(segs)
        if n == 0:
            return segs
        # Adaptive trimming: keep more when few strides available
        if drop_first is None or drop_last is None:
            drop_each = max(0, min(3, n // 6))  # 0 for <6, 1 for 6-11, 2 for 12-17, 3 for >=18
            drop_first = drop_each if drop_first is None else drop_first
            drop_last = drop_each if drop_last is None else drop_last
        if n <= (drop_first + drop_last):
            return segs
        return segs[drop_first:n-drop_last]

    def _compute_to_percent_from_events(hs_times: np.ndarray, to_times: np.ndarray, segs_use: list[tuple[float,float]]) -> float | None:
        # Compute toe-off percent within each HS->HS window using TO timestamps
        if not segs_use:
            return None
        to_times = np.asarray(to_times, dtype=float)
        if to_times.size == 0:
            return None
        percents = []
        for (t0, t1) in segs_use:
            # first toe-off strictly inside the stride window
            mask = (to_times > t0) & (to_times < t1)
            if not np.any(mask):
                continue
            to = float(np.min(to_times[mask]))
            frac = (to - t0) / max(1e-9, (t1 - t0))
            percents.append(100.0 * frac)
        if not percents:
            return None
        return float(np.nanmean(percents))

    def compare_over_anchor(t_ref: np.ndarray, segs_use: list[tuple[float,float]], sigL: np.ndarray, sigR: np.ndarray,
                             tL_: np.ndarray, tR_: np.ndarray):
        # Windows defined by successive reference contacts
        if not segs_use:
            return None
        # For each segment, resample L and R onto same phase grid
        def stack_segments(sig, t_src):
            S = []
            for (t0, t1) in segs_use:
                arr = phase_resample_by_time(t_src, sig, t0, t1, n=101)
                if arr is None:
                    continue
                if arr.ndim == 2:  # vector (shouldn't happen here)
                    arr = arr[:,0]
                S.append(arr)
            if not S:
                return None
            A = np.vstack(S)
            return A
        A_L = stack_segments(sigL, tL_)
        A_R = stack_segments(sigR, tR_)
        if A_L is None or A_R is None:
            return None
        meanL = np.nanmean(A_L, axis=0).astype(np.float32)
        sdL = np.nanstd(A_L, axis=0, ddof=0).astype(np.float32)
        meanR = np.nanmean(A_R, axis=0).astype(np.float32)
        sdR = np.nanstd(A_R, axis=0, ddof=0).astype(np.float32)
        return {
            'L': {'mean': meanL, 'sd': sdL},
            'R': {'mean': meanR, 'sd': sdR},
            'count_used': int(min(A_L.shape[0], A_R.shape[0])),
            'count_total': int(len(segs_use)),
        }

    def compare_all_components(anchor: str):
        if anchor == 'L':
            hs_times = hsL_times; t_ref = tL; stance_ref = stance_L; to_times = toL_times
        else:
            hs_times = hsR_times; t_ref = tR; stance_ref = stance_R; to_times = toR_times
        segs_use = _build_anchor_segments(t_ref, hs_times)
        to_pct = _compute_to_percent_from_events(hs_times, to_times, segs_use)
        out: dict[str, Any] = { 'meta': { 'to_percent': (float(to_pct) if to_pct is not None else None) } }
        for name, sL, sR in (
            ('Mx', Lmx, Rmx), ('My', Lmy, Rmy), ('Mz', Lmz, Rmz), ('Mmag', Lmag, Rmag)
        ):
            res = compare_over_anchor(t_ref, segs_use, sL, sR, tL, tR)
            if res is None:
                res = {'L': {'mean': np.zeros(101, np.float32), 'sd': np.zeros(101, np.float32)},
                       'R': {'mean': np.zeros(101, np.float32), 'sd': np.zeros(101, np.float32)},
                       'count_used': 0, 'count_total': 0}
            out[name] = res
        return out

    cycles_compare = {
        'anchor_L': compare_all_components('L'),
        'anchor_R': compare_all_components('R'),
    }

    # For anchored comparisons, add derived magnitude variants using component anchored means/sd
    def add_mag_variants_to_compare(cmp: dict):
        try:
            # Left
            Lmx_m = np.asarray(cmp['Mx']['L']['mean'], dtype=np.float32)
            Lmy_m = np.asarray(cmp['My']['L']['mean'], dtype=np.float32)
            Lmz_m = np.asarray(cmp['Mz']['L']['mean'], dtype=np.float32)
            Lmx_sd = np.asarray(cmp['Mx']['L']['sd'], dtype=np.float32)
            Lmy_sd = np.asarray(cmp['My']['L']['sd'], dtype=np.float32)
            Lmz_sd = np.asarray(cmp['Mz']['L']['sd'], dtype=np.float32)
            L_mag_of_mean = np.sqrt(Lmx_m*Lmx_m + Lmy_m*Lmy_m + Lmz_m*Lmz_m).astype(np.float32)
            L_rms_mag = np.sqrt((Lmx_sd*Lmx_sd + Lmy_sd*Lmy_sd + Lmz_sd*Lmz_sd) + (Lmx_m*Lmx_m + Lmy_m*Lmy_m + Lmz_m*Lmz_m)).astype(np.float32)
            nLmm = int(len(L_mag_of_mean)); zL = np.zeros(nLmm, dtype=np.float32)
            # Right
            Rmx_m = np.asarray(cmp['Mx']['R']['mean'], dtype=np.float32)
            Rmy_m = np.asarray(cmp['My']['R']['mean'], dtype=np.float32)
            Rmz_m = np.asarray(cmp['Mz']['R']['mean'], dtype=np.float32)
            Rmx_sd = np.asarray(cmp['Mx']['R']['sd'], dtype=np.float32)
            Rmy_sd = np.asarray(cmp['My']['R']['sd'], dtype=np.float32)
            Rmz_sd = np.asarray(cmp['Mz']['R']['sd'], dtype=np.float32)
            R_mag_of_mean = np.sqrt(Rmx_m*Rmx_m + Rmy_m*Rmy_m + Rmz_m*Rmz_m).astype(np.float32)
            R_rms_mag = np.sqrt((Rmx_sd*Rmx_sd + Rmy_sd*Rmy_sd + Rmz_sd*Rmz_sd) + (Rmx_m*Rmx_m + Rmy_m*Rmy_m + Rmz_m*Rmz_m)).astype(np.float32)
            nRmm = int(len(R_mag_of_mean)); zR = np.zeros(nRmm, dtype=np.float32)
            cmp['Mmag_of_mean'] = { 'L': { 'mean': L_mag_of_mean, 'sd': zL }, 'R': { 'mean': R_mag_of_mean, 'sd': zR }, 'count_used': cmp.get('Mx',{}).get('count_used',0), 'count_total': cmp.get('Mx',{}).get('count_total',0) }
            cmp['Mmag_rms'] = { 'L': { 'mean': L_rms_mag, 'sd': zL }, 'R': { 'mean': R_rms_mag, 'sd': zR }, 'count_used': cmp.get('Mx',{}).get('count_used',0), 'count_total': cmp.get('Mx',{}).get('count_total',0) }
        except Exception:
            n = 101; z = np.zeros(n, dtype=np.float32)
            cmp['Mmag_of_mean'] = { 'L': { 'mean': z, 'sd': z }, 'R': { 'mean': z, 'sd': z }, 'count_used': 0, 'count_total': 0 }
            cmp['Mmag_rms'] = { 'L': { 'mean': z, 'sd': z }, 'R': { 'mean': z, 'sd': z }, 'count_used': 0, 'count_total': 0 }

    if isinstance(cycles_compare.get('anchor_L'), dict):
        add_mag_variants_to_compare(cycles_compare['anchor_L'])
    if isinstance(cycles_compare.get('anchor_R'), dict):
        add_mag_variants_to_compare(cycles_compare['anchor_R'])

    # ---------------------------------------------
    # Angle anchored comparison (HS->HS) per joint/component
    # ---------------------------------------------
    def compare_angles_all_joints(anchor: str):
        if anchor == 'L':
            hs_times = hsL_times; t_ref = tL; to_times = toL_times
        else:
            hs_times = hsR_times; t_ref = tR; to_times = toR_times
        segs_use = _build_anchor_segments(t_ref, hs_times)
        to_pct = _compute_to_percent_from_events(hs_times, to_times, segs_use)

        def safe_cmp(sigL: np.ndarray, sigR: np.ndarray):
            res = compare_over_anchor(t_ref, segs_use, sigL, sigR, tL, tR)
            if res is None:
                z = np.zeros(101, np.float32)
                return {
                    'L': {'mean': z, 'sd': z},
                    'R': {'mean': z, 'sd': z},
                    'count_used': 0,
                    'count_total': 0,
                }
            return res

        out: dict[str, Any] = {
            'meta': { 'to_percent': (float(to_pct) if to_pct is not None else None) }
        }
        # Hip
        out['hip'] = {
            'flex': safe_cmp(hipL_deg[:, 0], hipR_deg[:, 0]),
            'add':  safe_cmp(hipL_deg[:, 1], hipR_deg[:, 1]),
            'rot':  safe_cmp(hipL_deg[:, 2], hipR_deg[:, 2]),
        }
        # Knee
        out['knee'] = {
            'flex': safe_cmp(kneeL_deg[:, 0], kneeR_deg[:, 0]),
            'add':  safe_cmp(kneeL_deg[:, 1], kneeR_deg[:, 1]),
            'rot':  safe_cmp(kneeL_deg[:, 2], kneeR_deg[:, 2]),
        }
        return out

    angle_compare = {
        'anchor_L': compare_angles_all_joints('L'),
        'anchor_R': compare_angles_all_joints('R'),
    }

    # Angle cycles anchored purely by heel strikes (HS->HS), same contacts/toe-offs as torque cycles
    def angle_cycles_triplet(angles_deg: np.ndarray, t_side: np.ndarray, contacts: np.ndarray, toe_offs: np.ndarray):
        flexC, n_used, n_tot = cycles(angles_deg[:, 0], t_side, contacts, toe_offs)
        addC,  _, _         = cycles(angles_deg[:, 1], t_side, contacts, toe_offs)
        rotC,  _, _         = cycles(angles_deg[:, 2], t_side, contacts, toe_offs)
        return { 'flex': flexC, 'add': addC, 'rot': rotC, 'count_used': n_used, 'count_total': n_tot }

    hip_cycles = {
        'L': angle_cycles_triplet(hipL_deg, tL, contacts_L, toe_offs_L),
        'R': angle_cycles_triplet(hipR_deg, tR, contacts_R, toe_offs_R),
    }
    knee_cycles = {
        'L': angle_cycles_triplet(kneeL_deg, tL, contacts_L, toe_offs_L),
        'R': angle_cycles_triplet(kneeR_deg, tR, contacts_R, toe_offs_R),
    }

    # Build cycle-level CSVs (101-point means) including magnitude variants for download
    def build_cycle_csv(side_key: str) -> str:
        side = cycles_out[side_key]
        # Required arrays (shape: 101)
        def arr(name: str, kind: str) -> np.ndarray:
            return np.asarray(side.get(name, {}).get(kind, np.zeros(101, np.float32)), dtype=np.float32)
        phase = np.linspace(0, 100, int(arr('Mx', 'mean').shape[0]), dtype=np.float32)
        mx_m, mx_s = arr('Mx','mean'), arr('Mx','sd')
        my_m, my_s = arr('My','mean'), arr('My','sd')
        mz_m, mz_s = arr('Mz','mean'), arr('Mz','sd')
        mm_m, mm_s = arr('Mmag','mean'), arr('Mmag','sd')  # E|M|
        mom_m = arr('Mmag_of_mean','mean')                  # |E[M]|
        rms_m = arr('Mmag_rms','mean')                      # RMS(|M|)
        header = (
            "phase_pct,"
            "Mx_mean(Nm),Mx_sd(Nm),"
            "My_mean(Nm),My_sd(Nm),"
            "Mz_mean(Nm),Mz_sd(Nm),"
            "Mmag_mean_E|M|(Nm),Mmag_sd(Nm),"
            "Mmag_of_mean_|E[M]|(Nm),"
            "Mmag_rms(Nm)"
        )
        lines = [header]
        for i in range(len(phase)):
            lines.append(
                f"{phase[i]:.2f},"
                f"{mx_m[i]:.6f},{mx_s[i]:.6f},"
                f"{my_m[i]:.6f},{my_s[i]:.6f},"
                f"{mz_m[i]:.6f},{mz_s[i]:.6f},"
                f"{mm_m[i]:.6f},{mm_s[i]:.6f},"
                f"{mom_m[i]:.6f},"
                f"{rms_m[i]:.6f}"
            )
        return "\n".join(lines)

    left_cycle_csv = build_cycle_csv('L')
    right_cycle_csv = build_cycle_csv('R')

    # Angle CSVs (time-based)
    left_angles_csv = (
        "time_s,L_hip_flex_deg,L_hip_add_deg,L_hip_rot_deg,L_knee_flex_deg,L_knee_add_deg,L_knee_rot_deg\n"
        + "\n".join(
            f"{t:.6f},{hf:.6f},{ha:.6f},{hr:.6f},{kf:.6f},{ka:.6f},{kr:.6f}"
            for t, (hf,ha,hr), (kf,ka,kr) in zip(tL, hipL_deg, kneeL_deg)
        )
    )
    right_angles_csv = (
        "time_s,R_hip_flex_deg,R_hip_add_deg,R_hip_rot_deg,R_knee_flex_deg,R_knee_add_deg,R_knee_rot_deg\n"
        + "\n".join(
            f"{t:.6f},{hf:.6f},{ha:.6f},{hr:.6f},{kf:.6f},{ka:.6f},{kr:.6f}"
            for t, (hf,ha,hr), (kf,ka,kr) in zip(tR, hipR_deg, kneeR_deg)
        )
    )

    # Angle cycle CSVs (101-point means)
    def build_angle_cycle_csv(side_key: str, joint: str) -> str:
        jd = hip_cycles[side_key] if joint == 'hip' else knee_cycles[side_key]
        # Choose length from flex mean
        flex_m = np.asarray(jd['flex']['mean'], dtype=np.float32)
        add_m  = np.asarray(jd['add']['mean'], dtype=np.float32)
        rot_m  = np.asarray(jd['rot']['mean'], dtype=np.float32)
        flex_s = np.asarray(jd['flex']['sd'], dtype=np.float32)
        add_s  = np.asarray(jd['add']['sd'], dtype=np.float32)
        rot_s  = np.asarray(jd['rot']['sd'], dtype=np.float32)
        n = int(max(flex_m.shape[0], add_m.shape[0], rot_m.shape[0]))
        phase = np.linspace(0, 100, n, dtype=np.float32)
        hdr = f"phase_pct,{joint}_flex_mean(deg),{joint}_flex_sd(deg),{joint}_add_mean(deg),{joint}_add_sd(deg),{joint}_rot_mean(deg),{joint}_rot_sd(deg)"
        lines = [hdr]
        for i in range(n):
            fm = float(flex_m[i]) if i < len(flex_m) else 0.0
            fs = float(flex_s[i]) if i < len(flex_s) else 0.0
            am = float(add_m[i]) if i < len(add_m) else 0.0
            asd = float(add_s[i]) if i < len(add_s) else 0.0
            rm = float(rot_m[i]) if i < len(rot_m) else 0.0
            rs = float(rot_s[i]) if i < len(rot_s) else 0.0
            lines.append(f"{phase[i]:.2f},{fm:.6f},{fs:.6f},{am:.6f},{asd:.6f},{rm:.6f},{rs:.6f}")
        return "\n".join(lines)

    left_hip_cycle_csv = build_angle_cycle_csv('L', 'hip')
    right_hip_cycle_csv = build_angle_cycle_csv('R', 'hip')
    left_knee_cycle_csv = build_angle_cycle_csv('L', 'knee')
    right_knee_cycle_csv = build_angle_cycle_csv('R', 'knee')

    # CSVs for anchored comparison (what the matrix view displays)
    def build_compare_csv(cmp: dict) -> str:
        def arr(path: list[str]):
            d = cmp
            for p in path:
                d = d.get(p, {})
            a = np.asarray(d if isinstance(d, (list, np.ndarray)) else d, dtype=np.float32)
            return a
        # Determine length from Mx L mean
        Lmx_m = np.asarray(cmp.get('Mx', {}).get('L', {}).get('mean', np.zeros(101, np.float32)), dtype=np.float32)
        n = int(Lmx_m.shape[0]) if Lmx_m.ndim == 1 else int(Lmx_m.shape[0])
        phase = np.linspace(0, 100, n, dtype=np.float32)
        header = (
            "phase_pct," \
            "L_Mx_mean(Nm),L_Mx_sd(Nm),L_My_mean(Nm),L_My_sd(Nm),L_Mz_mean(Nm),L_Mz_sd(Nm)," \
            "R_Mx_mean(Nm),R_Mx_sd(Nm),R_My_mean(Nm),R_My_sd(Nm),R_Mz_mean(Nm),R_Mz_sd(Nm)"
        )
        lines = [header]
        # Fetch arrays, defaulting to zeros of length n
        def safe(name: str, side: str, kind: str):
            a = np.asarray(cmp.get(name, {}).get(side, {}).get(kind, np.zeros(n, np.float32)), dtype=np.float32)
            return a if a.shape[0] == n else np.resize(a, (n,))
        Lmx_m = safe('Mx','L','mean'); Lmx_s = safe('Mx','L','sd')
        Lmy_m = safe('My','L','mean'); Lmy_s = safe('My','L','sd')
        Lmz_m = safe('Mz','L','mean'); Lmz_s = safe('Mz','L','sd')
        Rmx_m = safe('Mx','R','mean'); Rmx_s = safe('Mx','R','sd')
        Rmy_m = safe('My','R','mean'); Rmy_s = safe('My','R','sd')
        Rmz_m = safe('Mz','R','mean'); Rmz_s = safe('Mz','R','sd')
        for i in range(n):
            lines.append(
                f"{phase[i]:.2f},"
                f"{Lmx_m[i]:.6f},{Lmx_s[i]:.6f},"
                f"{Lmy_m[i]:.6f},{Lmy_s[i]:.6f},"
                f"{Lmz_m[i]:.6f},{Lmz_s[i]:.6f},"
                f"{Rmx_m[i]:.6f},{Rmx_s[i]:.6f},"
                f"{Rmy_m[i]:.6f},{Rmy_s[i]:.6f},"
                f"{Rmz_m[i]:.6f},{Rmz_s[i]:.6f}"
            )
        return "\n".join(lines)

    anchor_L_compare_csv = build_compare_csv(cycles_compare.get('anchor_L', {}))
    anchor_R_compare_csv = build_compare_csv(cycles_compare.get('anchor_R', {}))

    # Event diagnostics (if available)
    event_quality = gait_results.get('event_quality', {}) if isinstance(gait_results, dict) else {}
    detector_info = {
        'detector': 'unified_gait',
        'sampling_frequency': float(gait_results.get('sampling_frequency', Fs)),
        'params': gait_results.get('detector_params', {}),
        'filter_adjustments': gait_results.get('filter_adjustments', {}),
    }

    # Compute adaptive stance thresholds used (report only; stance shading uses events)
    def _stance_thresholds_report(omega_W: np.ndarray, a_free_W: np.ndarray):
        wmag = np.linalg.norm(omega_W, axis=1)
        amag = np.linalg.norm(a_free_W, axis=1)
        mad_w = float(1.4826 * np.median(np.abs(wmag - np.median(wmag)))) if wmag.size else 0.0
        mad_a = float(1.4826 * np.median(np.abs(amag - np.median(amag)))) if amag.size else 0.0
        th_w = float(max(2.0, 3.5 * mad_w))
        th_a = float(max(0.8, 3.0 * mad_a))
        return {'w_thr': th_w, 'a_thr': th_a, 'mad_w': mad_w, 'mad_a': mad_a}
    stance_thresholds = {
        'L': _stance_thresholds_report(omegaL_shank_W, aL_free_W),
        'R': _stance_thresholds_report(omegaR_shank_W, aR_free_W),
    }
    diag = {
        'left': {
            'hs_count': int(event_quality.get('left', {}).get('hs_count', int(contacts_L.size))),
            'to_count': int(event_quality.get('left', {}).get('to_count', int(toe_offs_L.size))),
            'cadence_spm': float(event_quality.get('left', {}).get('cadence_spm', 0.0)),
        },
        'right': {
            'hs_count': int(event_quality.get('right', {}).get('hs_count', int(contacts_R.size))),
            'to_count': int(event_quality.get('right', {}).get('to_count', int(toe_offs_R.size))),
            'cadence_spm': float(event_quality.get('right', {}).get('cadence_spm', 0.0)),
        },
    }

    return {
        'time_L': tL.astype(np.float32), 'time_R': tR.astype(np.float32),
        'L_mx': Lmx.astype(np.float32), 'L_my': Lmy.astype(np.float32), 'L_mz': Lmz.astype(np.float32),
        'R_mx': Rmx.astype(np.float32), 'R_my': Rmy.astype(np.float32), 'R_mz': Rmz.astype(np.float32),
        'L_Mmag': Lmag.astype(np.float32), 'R_Mmag': Rmag.astype(np.float32),
        'stance_L': stance_L.astype(np.uint8), 'stance_R': stance_R.astype(np.uint8),
    'left_csv': left_csv, 'right_csv': right_csv,
    'left_cycle_csv': left_cycle_csv, 'right_cycle_csv': right_cycle_csv,
        # New: angles time-series (deg) and CSVs
        'L_hip_angles_deg': hipL_deg, 'R_hip_angles_deg': hipR_deg,
        'L_knee_angles_deg': kneeL_deg, 'R_knee_angles_deg': kneeR_deg,
        'left_angles_csv': left_angles_csv, 'right_angles_csv': right_angles_csv,
        'left_hip_cycle_csv': left_hip_cycle_csv, 'right_hip_cycle_csv': right_hip_cycle_csv,
        'left_knee_cycle_csv': left_knee_cycle_csv, 'right_knee_cycle_csv': right_knee_cycle_csv,
    'anchor_L_compare_csv': anchor_L_compare_csv, 'anchor_R_compare_csv': anchor_R_compare_csv,
        'Lmean': Lmean, 'Lsd': Lsd, 'Rmean': Rmean, 'Rsd': Rsd,
        'cycles': cycles_out,
        'angle_cycles': { 'hip': hip_cycles, 'knee': knee_cycles },
    'angle_compare': angle_compare,
    'cycles_compare': cycles_compare,
        'cal_windows': cal_windows,
        'meta': {
            'baseline_mode': baseline_mode,
            'cycle_phase_points': 101,
            'sign_convention': 'flexion_positive_Mx; right_xz_mirrored',
            'magnitude_baseline': 'BESR_on_|M|',
            'magnitude_modes': ['mean_mag','mag_of_mean','rms_mag'],
            'angles_jcs': 'Grood–Suntay-inspired (hip,knee)',
            'events': diag,
            'yaw_ref_source': yaw_src,
            'yaw_ref_rad': float(yaw_ref),
            'still_resample': {'method': 'interp+thr+close', 'threshold': 0.75, 'close_radius_samples': 1},
            'moment_model': 'inertial+gravity (no GRF), teaching-grade',
            'timebase': {'L': 'femur', 'R': 'femur'},
            'fs_est': fs_meta,
            'stance_thresholds_used': stance_thresholds,
            'detector': detector_info,
            'acc_gravity_correction': { 'L': acc_corr_L, 'R': acc_corr_R },
            'overlap_window_s': overlap_meta,
        },
        'baseline_JCS': {
            'L': {'start': baseL0.astype(np.float32), 'end': baseL1.astype(np.float32)},
            'R': {'start': baseR0.astype(np.float32), 'end': baseR1.astype(np.float32)},
        },
        'angles_baseline': {
            'hip': {
                'L': {'start': hipL_b0.astype(np.float32), 'end': hipL_b1.astype(np.float32)},
                'R': {'start': hipR_b0.astype(np.float32), 'end': hipR_b1.astype(np.float32)},
            },
            'knee': {
                'L': {'start': kneeL_b0.astype(np.float32), 'end': kneeL_b1.astype(np.float32)},
                'R': {'start': kneeR_b0.astype(np.float32), 'end': kneeR_b1.astype(np.float32)},
            },
        },
        'angle_sign_convention': 'flexion_positive; adduction positive toward midline; internal rotation positive',
    }
