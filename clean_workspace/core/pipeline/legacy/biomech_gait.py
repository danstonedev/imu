"""
Advanced biomechanical gait event detection for tibia IMU data.
Based on heel strike impact signatures, toe-off gyroscopic patterns, and bilateral coordination.

LEGACY: This module predates the unified detector in
core.pipeline.unified_gait and is kept only for reference. Prefer using
core.pipeline.unified_gait.detect_gait_cycles and related helpers.
"""
from __future__ import annotations
import numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt, find_peaks
from typing import Tuple
from ..config.constants import CYCLE_N, CYCLE_DROP_FIRST, CYCLE_DROP_LAST, CYCLE_MIN_DUR_S, CYCLE_MAX_DUR_S

__all__ = [
    "detect_heel_strikes_biomech",
    "detect_hs_to_biomech",
    "bilateral_gait_cycles",
    "bilateral_gait_analysis",
    "cycle_mean_sd_biomech",
]

def _sos_bp(lo: float, hi: float, fs: float, order: int = 4):
    """Bandpass filter design."""
    return butter(order, [lo/(fs/2), hi/(fs/2)], btype='bandpass', output='sos')

def _sos_lp(cut: float, fs: float, order: int = 4):
    """Lowpass filter design."""
    return butter(order, cut/(fs/2), btype='low', output='sos')

def detect_hs_to_biomech(ax: np.ndarray, az: np.ndarray, gx: np.ndarray, fs: float,
                        vib_band: Tuple[float, float] = (8, 25),  # Adjusted for 60Hz data
                        gx_lp: float = 10,  # Adjusted for 60Hz data
                        hs_jerk_k: float = 2.0,  # More sensitive
                        to_win_ms: int = 120) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect heel strikes (HS) and toe-offs (TO) using biomechanical signatures.
    """
    N = len(az)
    if N < 100:
        return np.array([], dtype=int), np.array([], dtype=int)
    vib = sosfiltfilt(_sos_bp(*vib_band, fs), az)
    gx_f = sosfiltfilt(_sos_lp(gx_lp, fs), gx)
    dgx = np.gradient(gx_f) * fs
    jerk = np.gradient(vib) * fs
    mad = 1.4826 * np.median(np.abs(vib - np.median(vib)))
    if mad < 1e-6:
        mad = np.std(vib)
    hs_peaks, _ = find_peaks(np.abs(vib), height=hs_jerk_k*mad, distance=int(0.35*fs))
    HS = []
    for p in hs_peaks:
        w0, w1 = max(0, p-int(0.04*fs)), min(N-1, p+int(0.04*fs))
        if w1 <= w0:
            continue
        sign_change = np.any(np.signbit(gx_f[w0:p]) != np.signbit(gx_f[p:w1]))
        sharp_slope = np.max(np.abs(dgx[w0:w1])) > np.percentile(np.abs(dgx), 80)
        if sign_change or sharp_slope:
            HS.append(p)
    TO = []
    min_gap = int(0.25*fs)
    for i, hs in enumerate(HS):
        start = hs + int(0.15*fs)
        stop = hs + int(0.8*fs)
        if i+1 < len(HS):
            stop = min(stop, HS[i+1]-int(0.1*fs))
        start = min(max(start, hs + int(0.1*fs)), N-1)
        stop = min(max(stop, start+1), N-1)
        if stop <= start:
            continue
        w = max(1, int(0.06*fs))
        if w < len(vib):
            vib_rms = np.sqrt(np.convolve(vib**2, np.ones(w)/w, mode='same'))
        else:
            vib_rms = np.abs(vib)
        search_gx = gx_f[start:stop]
        if len(search_gx) > 3:
            mins, _ = find_peaks(-search_gx, distance=max(1, int(0.08*fs)))
        else:
            mins = []
        cand = None
        best_score = np.inf
        for m in mins:
            idx = start + m
            if idx >= N:
                continue
            slope_end = min(idx + int(0.04*fs), N)
            slope = np.mean(dgx[idx:slope_end]) if slope_end > idx else 0
            pre_idx = max(0, idx - w)
            post_idx = min(N, idx + w)
            if post_idx > idx and pre_idx < idx:
                drop = np.mean(vib_rms[pre_idx:idx]) - np.mean(vib_rms[idx:post_idx])
            else:
                drop = 0
            score = -(1.5*slope + 0.8*drop)
            if score < best_score:
                best_score, cand = score, idx
        if len(mins) == 0 and stop > start:
            min_idx = start + np.argmin(gx_f[start:stop])
            if min_idx - hs > min_gap:
                cand = min_idx
        if cand is not None and cand - hs > min_gap:
            TO.append(cand)
    return np.array(HS, dtype=int), np.array(TO, dtype=int)

def bilateral_gait_analysis(t_left: np.ndarray, a_left: np.ndarray, g_left: np.ndarray,
                           t_right: np.ndarray, a_right: np.ndarray, g_right: np.ndarray,
                           fs: float) -> dict:
    ax_l, az_l = a_left[:, 0], a_left[:, 2]
    ax_r, az_r = a_right[:, 0], a_right[:, 2]
    gx_l, gx_r = g_left[:, 0], g_right[:, 0]
    az_l_filt = az_l - np.mean(az_l)
    az_r_filt = az_r - np.mean(az_r)
    hs_l, to_l = detect_hs_to_biomech(ax_l, az_l_filt, gx_l, fs)
    hs_r, to_r = detect_hs_to_biomech(ax_r, az_r_filt, gx_r, fs)
    hs_l_time = t_left[hs_l] if len(hs_l) > 0 else np.array([])
    to_l_time = t_left[to_l] if len(to_l) > 0 else np.array([])
    hs_r_time = t_right[hs_r] if len(hs_r) > 0 else np.array([])
    to_r_time = t_right[to_r] if len(to_r) > 0 else np.array([])
    all_events = []
    if len(hs_l_time) > 0:
        all_events.extend([(t, 'L_HS') for t in hs_l_time])
    if len(to_l_time) > 0:
        all_events.extend([(t, 'L_TO') for t in to_l_time])
    if len(hs_r_time) > 0:
        all_events.extend([(t, 'R_HS') for t in hs_r_time])
    if len(to_r_time) > 0:
        all_events.extend([(t, 'R_TO') for t in to_r_time])
    all_events.sort(key=lambda x: x[0])
    results = {
        'hs_left_indices': hs_l,
        'to_left_indices': to_l,
        'hs_right_indices': hs_r,
        'to_right_indices': to_r,
        'hs_left_times': hs_l_time,
        'to_left_times': to_l_time,
        'hs_right_times': hs_r_time,
        'to_right_times': to_r_time,
        'all_events': all_events,
        'total_events': len(all_events),
        'duration': max(t_left[-1], t_right[-1]) - min(t_left[0], t_right[0]),
    }
    if len(hs_l_time) > 1:
        results['left_stride_times'] = np.diff(hs_l_time)
        results['left_avg_stride'] = np.mean(results['left_stride_times'])
    if len(hs_r_time) > 1:
        results['right_stride_times'] = np.diff(hs_r_time)
        results['right_avg_stride'] = np.mean(results['right_stride_times'])
    stance_percentages = []
    for i, hs_time in enumerate(hs_l_time[:-1]):
        next_hs_time = hs_l_time[i+1]
        to_in_stride = to_l_time[(to_l_time > hs_time) & (to_l_time < next_hs_time)]
        if len(to_in_stride) > 0:
            stance_duration = to_in_stride[0] - hs_time
            stride_duration = next_hs_time - hs_time
            stance_pct = (stance_duration / stride_duration) * 100
            stance_percentages.append(stance_pct)
    if stance_percentages:
        results['left_stance_percentages'] = stance_percentages
        results['left_avg_stance_pct'] = np.mean(stance_percentages)
    return results

def detect_heel_strikes_biomech(t: np.ndarray, omega_W: np.ndarray, a_free_W: np.ndarray, 
                               fs: float = 100.0) -> np.ndarray:
    sos_hp = signal.butter(2, 0.5, btype='high', fs=fs, output='sos')
    a_hp = signal.sosfiltfilt(sos_hp, a_free_W, axis=0)
    sos_impact = signal.butter(2, [10, 35], btype='band', fs=fs, output='sos')
    a_impact = signal.sosfiltfilt(sos_impact, a_free_W, axis=0)
    sos_lp = signal.butter(2, 10, btype='low', fs=fs, output='sos')
    omega_smooth = signal.sosfiltfilt(sos_lp, omega_W, axis=0)
    a_vert = a_impact[:, 2]
    a_resultant = np.linalg.norm(a_impact, axis=1)
    jerk_vert = np.gradient(a_vert, 1/fs)
    jerk_resultant = np.gradient(a_resultant, 1/fs)
    jerk_std = np.std(jerk_resultant)
    jerk_threshold = 3.0 * jerk_std
    from scipy.signal import find_peaks
    min_height = np.mean(np.abs(a_vert)) + 2 * np.std(np.abs(a_vert))
    min_distance = int(0.4 * fs)
    accel_peaks, accel_props = find_peaks(
        np.abs(a_vert), 
        height=min_height,
        distance=min_distance,
        prominence=np.std(np.abs(a_vert))
    )
    high_jerk_peaks = []
    for peak in accel_peaks:
        if np.abs(jerk_resultant[peak]) > jerk_threshold:
            high_jerk_peaks.append(peak)
    omega_sagittal = omega_smooth[:, 1]
    zero_crossings = []
    for i in range(1, len(omega_sagittal)):
        if omega_sagittal[i-1] * omega_sagittal[i] < 0:
            zero_crossings.append(i)
    gyro_peaks, _ = find_peaks(np.abs(omega_sagittal), distance=int(0.3 * fs))
    gyro_events = sorted(zero_crossings + gyro_peaks.tolist())
    confirmed_heel_strikes = []
    tolerance_samples = int(0.05 * fs)
    for accel_peak in high_jerk_peaks:
        closest_gyro = None
        min_distance = float('inf')
        for gyro_event in gyro_events:
            distance = abs(accel_peak - gyro_event)
            if distance < tolerance_samples and distance < min_distance:
                closest_gyro = gyro_event
                min_distance = distance
        if closest_gyro is not None:
            confirmed_heel_strikes.append(accel_peak)
    if len(confirmed_heel_strikes) < 2:
        confirmed_heel_strikes = high_jerk_peaks
    confirmed_heel_strikes = sorted(list(set(confirmed_heel_strikes)))
    final_heel_strikes = []
    min_interval = int(0.5 * fs)
    for i, hs in enumerate(confirmed_heel_strikes):
        if i == 0 or (hs - final_heel_strikes[-1]) >= min_interval:
            final_heel_strikes.append(hs)
    return np.array(final_heel_strikes, dtype=int)

def bilateral_gait_cycles(t_L: np.ndarray, hs_L: np.ndarray, 
                         t_R: np.ndarray, hs_R: np.ndarray) -> dict:
    if len(hs_L) == 0 or len(hs_R) == 0:
        return {"L": hs_L, "R": hs_R, "alternating_score": 0.0}
    hs_R_time = t_R[hs_R]
    hs_R_interp = np.interp(hs_R_time, t_L, np.arange(len(t_L)))
    hs_R_interp = hs_R_interp.astype(int)
    all_events = []
    for hs in hs_L:
        all_events.append((hs, 'L'))
    for hs in hs_R_interp:
        all_events.append((hs, 'R'))
    all_events.sort(key=lambda x: x[0])
    alternating_score = 0.0
    if len(all_events) > 1:
        correct_alternations = 0
        total_transitions = len(all_events) - 1
        for i in range(total_transitions):
            if all_events[i][1] != all_events[i+1][1]:
                correct_alternations += 1
        alternating_score = correct_alternations / total_transitions if total_transitions > 0 else 0.0
    return {
        "L": hs_L,
        "R": hs_R_interp,
        "all_events": all_events,
        "alternating_score": alternating_score
    }

def cycle_mean_sd_biomech(t: np.ndarray, sig: np.ndarray, heel_strikes: np.ndarray, 
                         n: int = CYCLE_N, drop_first: int = CYCLE_DROP_FIRST, 
                         drop_last: int = CYCLE_DROP_LAST, min_dur_s: float = CYCLE_MIN_DUR_S, 
                         max_dur_s: float = CYCLE_MAX_DUR_S):
    if len(heel_strikes) < 2:
        return np.zeros(n, np.float32), np.zeros(n, np.float32), 0, 0
    used = []
    total = 0
    for i in range(len(heel_strikes) - 1):
        total += 1
        if i < drop_first or i >= (len(heel_strikes) - 1 - drop_last):
            continue
        s = int(heel_strikes[i])
        e = int(heel_strikes[i+1])
        if e - s < 3:
            continue
        dur = float(t[min(e, len(t)-1)] - t[min(s, len(t)-1)])
        if not (min_dur_s <= dur <= max_dur_s):
            continue
        x = np.linspace(0.0, 1.0, e - s, dtype=float)
        xi = np.linspace(0.0, 1.0, n, dtype=float)
        seg = np.asarray(sig[s:e], dtype=float)
        if seg.ndim == 1:
            yi = np.interp(xi, x, seg)
        else:
            yi = np.vstack([np.interp(xi, x, seg[:, j]) for j in range(seg.shape[1])]).T
        used.append(yi)
    if not used:
        return np.zeros(n, np.float32), np.zeros(n, np.float32), 0, total
    arr = np.stack(used, axis=0)
    mean = np.nanmean(arr, axis=0)
    sd = np.nanstd(arr, axis=0)
    return mean.astype(np.float32), sd.astype(np.float32), int(len(used)), int(total)
