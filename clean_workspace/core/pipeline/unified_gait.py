"""
Unified gait analysis combining biomechanical heel strike detection 
with bilateral coordination for robust gait cycle identification.

This module replaces the fragmented approach across multiple files.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks, correlate
from typing import Tuple, Dict, Any, Optional
from ..config.constants import CYCLE_N, CYCLE_DROP_FIRST, CYCLE_DROP_LAST, CYCLE_MIN_DUR_S, CYCLE_MAX_DUR_S

__all__ = ["detect_gait_cycles", "gait_cycle_analysis"]

def _butter_bandpass(low_hz: float, high_hz: float, fs: float, order: int = 4):
    """Create bandpass filter coefficients."""
    nyquist = fs / 2
    # Sanitize cutoff frequencies against Nyquist
    margin = 0.05 * nyquist
    low_hz = max(0.01, min(low_hz, max(0.01, nyquist - margin)))
    high_hz = max(low_hz + 0.01, min(high_hz, max(low_hz + 0.01, nyquist - margin)))
    low = low_hz / nyquist
    high = high_hz / nyquist
    return butter(order, [low, high], btype='band', output='sos')

def _butter_lowpass(cutoff_hz: float, fs: float, order: int = 4):
    """Create lowpass filter coefficients."""
    nyquist = fs / 2
    margin = 0.05 * nyquist
    cutoff_hz = max(0.05, min(cutoff_hz, max(0.05, nyquist - margin)))
    return butter(order, cutoff_hz / nyquist, btype='low', output='sos')

def _estimate_stride_time_seconds(signal: np.ndarray, fs: float, min_s: float = 0.6, max_s: float = 2.5) -> Optional[float]:
    """
    Estimate dominant stride duration using autocorrelation of a 1D signal.
    Returns None if unreliable.
    """
    x = np.asarray(signal, dtype=float)
    if x.size < int(2.0 * fs):
        return None
    x = x - float(np.mean(x))
    r = correlate(x, x, mode='full')
    mid = len(r) // 2
    r = r[mid:]
    k0 = max(1, int(min_s * fs))
    k1 = min(len(r) - 1, int(max_s * fs))
    if k1 <= k0 + 2:
        return None
    seg = r[k0:k1]
    if not np.isfinite(seg).any() or seg.size == 0:
        return None
    idx_local, _ = find_peaks(seg)
    if idx_local.size == 0:
        k = int(np.argmax(seg)) + k0
        return float(k) / fs
    k = int(idx_local[np.argmax(seg[idx_local])]) + k0
    return float(k) / fs

def detect_heel_strikes_biomech(
    t: np.ndarray,
    accel: np.ndarray, 
    gyro: np.ndarray,
    fs: float,
    vib_band: Tuple[float, float] = (8, 25),
    gyro_lpf: float = 10,
    impact_threshold: float = 2.5,
    min_step_time: float = 0.4
) -> np.ndarray:
    """
    Detect heel strikes using biomechanical signatures from tibia IMU.
    
    Based on research showing heel strike creates:
    1. Sharp vertical acceleration spike (impact)
    2. Angular velocity zero-crossing or extremum (rotation change)
    3. High-frequency vibrations (10-35 Hz)
    
    Args:
        t: Time array (seconds)
        accel: 3D acceleration [N, 3] in world frame (m/s²)
        gyro: 3D angular velocity [N, 3] in world frame (rad/s)
        fs: Sampling frequency (Hz)
        vib_band: Frequency band for impact detection (Hz)
        gyro_lpf: Lowpass filter for gyroscope (Hz)
        impact_threshold: Multiple of MAD for impact detection
        min_step_time: Minimum time between heel strikes (s)
        
    Returns:
        Array of heel strike indices
    """
    N = len(t)
    if N < int(fs):  # Need at least 1 second of data
        return np.array([], dtype=int)
    
    # Extract relevant signals
    # Vertical acceleration (assume Z is vertical after world alignment)
    a_vert = accel[:, 2]  # Z component
    # Sagittal angular velocity (assume Y is mediolateral axis)
    g_sagittal = gyro[:, 1]  # Y component (pitch)
    
    # Filter for impact detection (vibrations from heel strike)
    try:
        vib_filter = _butter_bandpass(*vib_band, fs)
        impact_signal = sosfiltfilt(vib_filter, a_vert)
    except:
        # Fallback if filtering fails
        impact_signal = a_vert
    
    # Filter gyro for smooth analysis
    try:
        gyro_filter = _butter_lowpass(gyro_lpf, fs)
        g_smooth = sosfiltfilt(gyro_filter, g_sagittal)
    except:
        g_smooth = g_sagittal
    
    # Adaptive threshold using Median Absolute Deviation
    mad = 1.4826 * np.median(np.abs(impact_signal - np.median(impact_signal)))
    if mad < 1e-6:  # Avoid division by zero
        mad = np.std(impact_signal)
    # Dynamic stride estimation and min distance based on impact envelope
    est_stride_s = _estimate_stride_time_seconds(np.abs(impact_signal), fs)
    dyn_min_s = max(min_step_time, 0.5 * est_stride_s) if est_stride_s else min_step_time
    min_distance = int(max(1, dyn_min_s * fs))
    # Dynamic threshold combining MAD and high percentile
    p95 = float(np.percentile(np.abs(impact_signal), 95)) if impact_signal.size else 0.0
    thr_mad = float(impact_threshold) * float(mad)
    threshold = float(max(float(thr_mad), float(0.2 * p95)))
    
    # Find impact peaks (potential heel strikes) with minimum prominence
    prominence = float(max(0.0, float(0.5 * float(mad))))
    impact_peaks, _ = find_peaks(
        np.abs(impact_signal), 
        height=threshold,
        distance=min_distance,
        prominence=prominence,
    )
    
    # Confirm with gyroscope (look for rotation changes)
    confirmed_hs = []
    gyro_deriv = np.gradient(g_smooth) * fs
    
    for peak in impact_peaks:
        # Look for gyro zero-crossing or sharp change near the impact
        window_start = max(0, peak - int(0.05 * fs))  # ±50ms
        window_end = min(N, peak + int(0.05 * fs))
        
        gyro_window = g_smooth[window_start:window_end]
        deriv_window = gyro_deriv[window_start:window_end]
        
        # Check for zero crossing or high angular acceleration
        has_zero_cross = len(gyro_window) > 1 and np.any(
            np.diff(np.sign(gyro_window)) != 0
        )
        
        high_angular_accel = np.max(np.abs(deriv_window)) > np.percentile(
            np.abs(gyro_deriv), 75
        )
        
        if has_zero_cross or high_angular_accel:
            confirmed_hs.append(peak)
    
    return np.array(confirmed_hs, dtype=int)

def detect_gait_cycles(*args, **kwargs) -> Dict[str, Any]:
    """
    Unified gait cycle detection using biomechanical heel strike detection
    with bilateral coordination assessment.

    Backward-compatible signature:
    - Old: detect_gait_cycles(t, accel_left, gyro_left, accel_right, gyro_right, fs)
    - New: detect_gait_cycles(t_left, accel_left, gyro_left, t_right, accel_right, gyro_right, fs)
    - Keyword form also supported: t_left=..., t_right=..., accel_left=..., gyro_left=..., accel_right=..., gyro_right=..., fs=...
    """
    # Parse inputs
    if 't_left' in kwargs or 't_right' in kwargs:
        t_left = kwargs['t_left']
        t_right = kwargs.get('t_right', t_left)
        accel_left = kwargs['accel_left']
        gyro_left = kwargs['gyro_left']
        accel_right = kwargs['accel_right']
        gyro_right = kwargs['gyro_right']
        fs = kwargs['fs']
    else:
        if len(args) == 6:
            # Old signature: (t, accel_left, gyro_left, accel_right, gyro_right, fs)
            t = args[0]; accel_left = args[1]; gyro_left = args[2]; accel_right = args[3]; gyro_right = args[4]; fs = args[5]
            t_left = t
            t_right = t
        elif len(args) == 7:
            # New positional signature
            t_left = args[0]; accel_left = args[1]; gyro_left = args[2]; t_right = args[3]; accel_right = args[4]; gyro_right = args[5]; fs = args[6]
        else:
            raise TypeError("detect_gait_cycles called with unsupported arguments")

    # Low-fs safeguards for filter settings
    # Default bands
    vib_band_default = (8.0, 25.0)
    gyro_lpf_default = 10.0
    vib_band_used = list(vib_band_default)
    gyro_lpf_used = float(gyro_lpf_default)
    filter_adjustments: Dict[str, Any] = {}
    if fs < 25.0:
        # Narrow band for impact and reduce gyro LPF at low sampling rates
        vib_band_used = [4.0, min(10.0, max(5.0, fs/2.0 - max(0.5, 0.05*fs)))]
        gyro_lpf_used = min(6.0, max(2.0, fs/4.0))
        filter_adjustments = {
            'reason': 'low_fs',
            'fs': float(fs),
            'vib_band': tuple(vib_band_used),
            'gyro_lpf': float(gyro_lpf_used),
        }

    # Detect heel strikes for both legs
    vib_band_tuple: Tuple[float, float] = (float(vib_band_used[0]), float(vib_band_used[1]))
    hs_left = detect_heel_strikes_biomech(t_left, accel_left, gyro_left, fs, vib_band=vib_band_tuple, gyro_lpf=float(gyro_lpf_used))
    hs_right = detect_heel_strikes_biomech(t_right, accel_right, gyro_right, fs, vib_band=vib_band_tuple, gyro_lpf=float(gyro_lpf_used))

    # Detect toe-offs based on heel strikes
    to_left = detect_toe_offs_biomech(t_left, accel_left, gyro_left, hs_left, fs)
    to_right = detect_toe_offs_biomech(t_right, accel_right, gyro_right, hs_right, fs)

    # Assess bilateral coordination
    coordination = assess_bilateral_coordination(hs_left, t_left, hs_right, t_right)

    return {
        'heel_strikes_left': hs_left,
        'heel_strikes_right': hs_right,
        'toe_offs_left': to_left,
        'toe_offs_right': to_right,
        'coordination_metrics': coordination,
        'sampling_frequency': fs,
        'detector_params': {
            'vib_band_used': tuple(vib_band_used),
            'gyro_lpf_used': float(gyro_lpf_used)
        },
        'filter_adjustments': filter_adjustments
    }

def detect_toe_offs_biomech(
    t: np.ndarray,
    accel: np.ndarray,
    gyro: np.ndarray,
    heel_strikes: np.ndarray,
    fs: float,
) -> np.ndarray:
    """
    Detect toe-offs using features around late stance:
    - Local minimum of sagittal-plane angular velocity (Y-axis)
    - Rising slope after the minimum
    - Drop in vibration energy after candidate (leaving ground)
    - Optional AP acceleration peak near push-off

    Args:
        t: Time array (seconds)
        accel: 3D acceleration [N,3] in world frame (m/s^2)
        gyro: 3D angular velocity [N,3] in world frame (rad/s)
        heel_strikes: Indices of heel strikes for the same leg
        fs: Sampling frequency (Hz)

    Returns:
        Array of toe-off indices aligned to provided heel strikes
    """
    N = len(t)
    if N == 0 or len(heel_strikes) == 0:
        return np.array([], dtype=int)

    # Components: assume Z vertical, Y sagittal rotation, X AP
    a_vert = accel[:, 2]
    a_ap = accel[:, 0]
    g_sagittal = gyro[:, 1]

    # Smooth gyro and compute simple vibration proxy
    try:
        gyro_filter = _butter_lowpass(10.0, fs)
        g_smooth = sosfiltfilt(gyro_filter, g_sagittal)
    except Exception:
        g_smooth = g_sagittal

    vib_energy = a_vert.astype(float) ** 2

    # Rolling RMS of vibration energy (~100 ms)
    window_size = max(1, int(0.1 * fs))
    kernel = np.ones(window_size, dtype=float) / float(window_size)
    vib_rms = np.convolve(vib_energy, kernel, mode='same')

    toe_offs: list[int] = []

    # Pair TOs to stride windows between consecutive HS
    for i, hs in enumerate(heel_strikes[:-1] if len(heel_strikes) > 1 else []):
        hs_next = heel_strikes[i + 1]
        stride = max(1, hs_next - hs)
        search_start = int(hs + 0.30 * stride)
        search_end = max(search_start + 1, int(hs_next - 0.10 * stride))

        if search_end <= search_start:
            toe_offs.append(-1)
            continue

        # Candidate minima in smoothed sagittal gyro
        search_gyro = g_smooth[search_start:search_end]
        if len(search_gyro) < 3:
            toe_offs.append(-1)
            continue

        local_minima, _ = find_peaks(-search_gyro, distance=max(1, int(0.10 * fs)))

        best_to = None
        best_score = -np.inf

        for min_idx in local_minima:
            candidate = search_start + int(min_idx)

            # Rising slope after minimum (expect plantar flexion increase)
            slope_window = min(int(0.05 * fs), max(0, N - candidate - 1))
            if slope_window > 1:
                slope = float(np.mean(np.gradient(g_smooth[candidate:candidate + slope_window])))
            else:
                slope = 0.0

            # Vibration energy drop after leaving ground
            if candidate >= window_size and candidate + window_size < N:
                vib_before = float(np.mean(vib_rms[candidate - window_size:candidate]))
                vib_after = float(np.mean(vib_rms[candidate:candidate + window_size]))
                vib_drop = vib_before - vib_after
            else:
                vib_drop = 0.0

            # AP push-off peak near candidate
            ap_ws = int(0.05 * fs)
            ap_window_start = max(0, candidate - ap_ws)
            ap_window_end = min(N, candidate + ap_ws)
            ap_peak = float(np.max(a_ap[ap_window_start:ap_window_end])) if ap_window_end > ap_window_start else 0.0

            # Heuristic score
            score = 2.0 * slope + 1.0 * vib_drop + 0.5 * ap_peak

            if score > best_score:
                best_score = score
                best_to = candidate

        # Enforce minimum stance time of ~200 ms
        if best_to is not None and best_to > hs + int(0.20 * fs):
            toe_offs.append(int(best_to))
        else:
            toe_offs.append(-1)

    # Replace missing (-1) with NaN-safe removal
    return np.array([idx for idx in toe_offs if idx >= 0], dtype=int)

def assess_bilateral_coordination(
    hs_left: np.ndarray,
    t_left: np.ndarray,
    hs_right: np.ndarray,
    t_right: np.ndarray,
) -> Dict[str, float]:
    """
    Assess bilateral gait coordination quality.
    
    Returns metrics about left-right alternation and timing.
    """
    if len(hs_left) == 0 or len(hs_right) == 0:
        return {
            'alternation_score': 0.0,
            'step_time_cv': 1.0,
            'bilateral_symmetry': 0.0
        }
    
    # Combine and sort all heel strikes
    all_events = []
    for hs in hs_left:
        if 0 <= hs < len(t_left):
            all_events.append((t_left[hs], 'L', hs))
    for hs in hs_right:
        if 0 <= hs < len(t_right):
            all_events.append((t_right[hs], 'R', hs))
    
    all_events.sort()
    
    if len(all_events) < 4:
        return {
            'alternation_score': 0.0,
            'step_time_cv': 1.0,
            'bilateral_symmetry': 0.0
        }
    
    # Check alternation pattern
    sequence = [event[1] for event in all_events]
    alternation_errors = 0
    for i in range(1, len(sequence)):
        if sequence[i] == sequence[i-1]:
            alternation_errors += 1
    
    alternation_score = 1.0 - (alternation_errors / max(1, len(sequence) - 1))
    
    # Step time variability
    step_times = []
    for i in range(1, len(all_events)):
        step_times.append(all_events[i][0] - all_events[i-1][0])
    
    if len(step_times) > 0:
        mean_st = float(np.mean(step_times))
        std_st = float(np.std(step_times))
        step_time_cv = float(std_st / max(mean_st, 1e-6))
    else:
        step_time_cv = 1.0
    
    # Bilateral symmetry (left vs right cycle durations)
    left_cycles = np.diff([t_left[hs] for hs in hs_left if 0 <= hs < len(t_left)])
    right_cycles = np.diff([t_right[hs] for hs in hs_right if 0 <= hs < len(t_right)])
    
    if len(left_cycles) > 0 and len(right_cycles) > 0:
        ml = float(np.mean(left_cycles))
        mr = float(np.mean(right_cycles))
        symmetry = float(1.0 - abs(ml - mr) / max(ml, mr, 1e-6))
    else:
        symmetry = 0.0
    
    return {
    'alternation_score': float(alternation_score),
    'step_time_cv': float(step_time_cv),
    'bilateral_symmetry': float(max(0.0, symmetry))
    }

def gait_cycle_analysis(
    t: np.ndarray,
    signal: np.ndarray,
    heel_strikes: np.ndarray,
    toe_offs: np.ndarray,
    n_points: int = CYCLE_N,
    drop_first: int = CYCLE_DROP_FIRST,
    drop_last: int = CYCLE_DROP_LAST,
    min_duration: float = CYCLE_MIN_DUR_S,
    max_duration: float = CYCLE_MAX_DUR_S
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Compute cycle-averaged mean and standard deviation with toe-off timing.
    
    Args:
        t: Time array
        signal: Signal to analyze (can be 1D or 2D)
        heel_strikes: Heel strike indices
        toe_offs: Toe-off indices
        n_points: Number of points in normalized cycle
        drop_first/last: Number of cycles to drop
        min/max_duration: Valid cycle duration range
        
    Returns:
        (mean, std, used_cycles, total_cycles)
    """
    if len(heel_strikes) < 2:
        shape = (n_points, signal.shape[1]) if signal.ndim > 1 else (n_points,)
        return (np.zeros(shape, dtype=np.float32), 
                np.zeros(shape, dtype=np.float32), 0, 0)
    
    used_cycles = []
    total_cycles = 0
    
    for i in range(len(heel_strikes) - 1):
        total_cycles += 1
        
        # Skip cycles according to drop settings
        if i < drop_first or i >= (len(heel_strikes) - 1 - drop_last):
            continue
            
        start_idx = heel_strikes[i]
        end_idx = heel_strikes[i + 1]
        
        if end_idx - start_idx < 3:
            continue
            
        # Check cycle duration
        duration = t[min(end_idx, len(t) - 1)] - t[min(start_idx, len(t) - 1)]
        if not (min_duration <= duration <= max_duration):
            continue
        
        # Extract and normalize cycle
        cycle_signal = signal[start_idx:end_idx]
        
        # Normalize to percentage of gait cycle
        x_original = np.linspace(0.0, 1.0, len(cycle_signal))
        x_normalized = np.linspace(0.0, 1.0, n_points)
        
        if signal.ndim == 1:
            normalized_cycle = np.interp(x_normalized, x_original, cycle_signal)
        else:
            normalized_cycle = np.column_stack([
                np.interp(x_normalized, x_original, cycle_signal[:, j]) 
                for j in range(signal.shape[1])
            ])
        
        used_cycles.append(normalized_cycle)
    
    if not used_cycles:
        shape = (n_points, signal.shape[1]) if signal.ndim > 1 else (n_points,)
        return (np.zeros(shape, dtype=np.float32),
                np.zeros(shape, dtype=np.float32), 0, total_cycles)
    
    # Compute statistics across cycles
    cycles_array = np.stack(used_cycles, axis=0)
    mean_cycle = np.nanmean(cycles_array, axis=0).astype(np.float32)
    std_cycle = np.nanstd(cycles_array, axis=0).astype(np.float32)
    
    return mean_cycle, std_cycle, len(used_cycles), total_cycles
