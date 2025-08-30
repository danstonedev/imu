"""
Biomechanical heel strike detection for tibia-mounted IMUs.
Based on the fusion of acceleration spikes and gyroscope zero-crossings.
"""
from __future__ import annotations
import numpy as np
from scipy import signal
from ..config.constants import CYCLE_MIN_DUR_S, CYCLE_MAX_DUR_S

__all__ = ["detect_heel_strikes_biomech", "bilateral_gait_cycles"]

def detect_heel_strikes_biomech(t: np.ndarray, gyro: np.ndarray, accel: np.ndarray, 
                               fs: float, debug: bool = False) -> np.ndarray | dict:
    """
    Detect heel strikes using biomechanical signatures from tibia IMU.
    
    Algorithm based on:
    1. High-frequency acceleration spikes (impact signature)
    2. Sagittal gyroscope zero-crossings/extrema (rotation change)
    3. Fusion of both signals for robust detection
    
    Args:
        t: Time array [s]
        gyro: Angular velocity [rad/s] shape (N, 3) - xyz
        accel: Free acceleration [m/s²] shape (N, 3) - xyz  
        fs: Sampling frequency [Hz]
        debug: Return intermediate signals for analysis
        
    Returns:
        heel_strike_indices: Array of sample indices where heel strikes occur
    """
    # 1. PREPROCESSING
    # Remove gravity and low-frequency components from acceleration
    # High-pass filter at 0.5 Hz to isolate impact energy
    sos_hp = signal.butter(4, 0.5, btype='high', fs=fs, output='sos')
    accel_hp = signal.sosfiltfilt(sos_hp, accel, axis=0)
    
    # Band-pass acceleration for impact detection (5-25 Hz for 60Hz sampling)
    # Adjusted for sampling rate limitations
    max_freq = min(25, fs/2 - 1)  # Ensure we stay below Nyquist
    sos_bp = signal.butter(4, [5, max_freq], btype='band', fs=fs, output='sos')
    accel_impact = signal.sosfiltfilt(sos_bp, accel_hp, axis=0)
    
    # Low-pass gyroscope at 15 Hz to remove noise
    gyro_cutoff = min(15, fs/2 - 1)
    sos_lp = signal.butter(4, gyro_cutoff, btype='low', fs=fs, output='sos')
    gyro_smooth = signal.sosfiltfilt(sos_lp, gyro, axis=0)
    
    # 2. ACCELERATION SPIKE DETECTION
    # Calculate resultant acceleration magnitude
    accel_mag = np.linalg.norm(accel_impact, axis=1)
    
    # Calculate jerk (derivative of acceleration)
    jerk = np.gradient(accel_mag, t)
    jerk_mag = np.abs(jerk)
    
    # More sensitive adaptive threshold for jerk peaks
    jerk_threshold = np.mean(jerk_mag) + 1.5 * np.std(jerk_mag)  # Reduced from 2.5
    
    # Find acceleration peaks with prominence
    min_distance = int(0.3 * fs)  # Minimum 300ms between peaks
    accel_peaks, _ = signal.find_peaks(
        accel_mag, 
        distance=min_distance,
        prominence=np.std(accel_mag) * 0.8  # Reduced from 1.5
    )
    
    # Filter peaks by jerk threshold
    accel_candidates = accel_peaks[jerk_mag[accel_peaks] > jerk_threshold]
    
    # 3. GYROSCOPE ZERO-CROSSING DETECTION
    # Focus on sagittal plane (assuming Y-axis is mediolateral)
    # X and Z are anterior-posterior and superior-inferior
    gyro_sagittal = gyro_smooth[:, 0]  # Pitch rotation around X-axis
    
    # Find zero crossings and local extrema
    zero_crossings = []
    for i in range(1, len(gyro_sagittal)):
        if gyro_sagittal[i-1] * gyro_sagittal[i] < 0:  # Sign change
            zero_crossings.append(i)
    
    # Find local extrema (peaks and troughs)
    gyro_peaks, _ = signal.find_peaks(np.abs(gyro_sagittal), distance=min_distance)
    
    # Combine zero crossings and extrema
    gyro_events = np.unique(np.concatenate([zero_crossings, gyro_peaks]))
    
    # 4. FUSION: MATCH ACCELERATION SPIKES WITH GYRO EVENTS
    heel_strikes = []
    match_window = int(0.1 * fs)  # ±100ms matching window (increased from 50ms)
    
    for accel_idx in accel_candidates:
        # Find closest gyro event within window
        distances = np.abs(gyro_events - accel_idx)
        min_dist_idx = np.argmin(distances)
        
        if distances[min_dist_idx] <= match_window:
            heel_strikes.append(accel_idx)
    
    # If fusion doesn't work well, fall back to acceleration peaks only
    if len(heel_strikes) < len(accel_candidates) * 0.3:  # Less than 30% matched
        print("Warning: Poor accel-gyro fusion, using acceleration peaks only")
        heel_strikes = accel_candidates.tolist()
    
    heel_strikes = np.array(heel_strikes, dtype=int)
    
    # 5. POST-PROCESSING
    # Remove duplicates and ensure minimum interval
    if len(heel_strikes) > 1:
        # Sort by time
        heel_strikes = heel_strikes[np.argsort(heel_strikes)]
        
        # Remove heel strikes that are too close together
        filtered_hs = [heel_strikes[0]]
        for hs in heel_strikes[1:]:
            if (hs - filtered_hs[-1]) >= min_distance:
                filtered_hs.append(hs)
        
        heel_strikes = np.array(filtered_hs)
    
    # 6. CADENCE SANITY CHECK
    if len(heel_strikes) > 2:
        intervals = np.diff(heel_strikes) / fs
        
        # Remove intervals outside physiological range (0.5-2.0 seconds)
        valid_mask = (intervals >= CYCLE_MIN_DUR_S) & (intervals <= CYCLE_MAX_DUR_S)
        
        # Keep heel strikes that create valid intervals
        keep_indices = [0]  # Always keep first
        for i, valid in enumerate(valid_mask):
            if valid:
                keep_indices.append(i + 1)
        
        if len(keep_indices) > 1:
            heel_strikes = heel_strikes[keep_indices]
    
    if debug:
        return {
            'heel_strikes': heel_strikes,
            'accel_impact': accel_impact,
            'accel_mag': accel_mag,
            'jerk': jerk,
            'jerk_threshold': jerk_threshold,
            'accel_candidates': accel_candidates,
            'gyro_smooth': gyro_smooth,
            'gyro_events': gyro_events,
            'gyro_sagittal': gyro_sagittal
        }
    
    return heel_strikes

def bilateral_gait_cycles(t_L: np.ndarray, hs_L: np.ndarray, 
                         t_R: np.ndarray, hs_R: np.ndarray) -> dict:
    """
    Analyze bilateral gait coordination and compute proper gait cycles.
    
    Args:
        t_L, t_R: Time arrays for left and right legs
        hs_L, hs_R: Heel strike indices for left and right legs
        
    Returns:
        Dictionary with bilateral gait analysis results
    """
    # Convert indices to timestamps
    hs_L_times = t_L[hs_L] if len(hs_L) > 0 else np.array([])
    hs_R_times = t_R[hs_R] if len(hs_R) > 0 else np.array([])
    
    # Combine and sort all heel strikes
    all_events = []
    for t in hs_L_times:
        all_events.append((t, 'L'))
    for t in hs_R_times:
        all_events.append((t, 'R'))
    
    all_events.sort(key=lambda x: x[0])
    
    if len(all_events) < 4:
        return {
            'alternation_score': 0.0,
            'step_times': [],
            'stride_times_L': [],
            'stride_times_R': [],
            'double_support_times': []
        }
    
    # Calculate alternation score (perfect alternation = 1.0)
    pattern = [event[1] for event in all_events]
    alternations = 0
    for i in range(len(pattern) - 1):
        if pattern[i] != pattern[i + 1]:
            alternations += 1
    
    alternation_score = alternations / (len(pattern) - 1) if len(pattern) > 1 else 0.0
    
    # Calculate step times (time between opposite heel strikes)
    step_times = []
    for i in range(len(all_events) - 1):
        if all_events[i][1] != all_events[i + 1][1]:  # Different legs
            step_time = all_events[i + 1][0] - all_events[i][0]
            step_times.append(step_time)
    
    # Calculate stride times (time between same leg heel strikes)
    stride_times_L = []
    stride_times_R = []
    
    if len(hs_L_times) > 1:
        stride_times_L = np.diff(hs_L_times).tolist()
    
    if len(hs_R_times) > 1:
        stride_times_R = np.diff(hs_R_times).tolist()
    
    # Estimate double support times (simplified)
    # In normal walking, double support is ~10-20% of stride time
    double_support_times = []
    all_stride_times = stride_times_L + stride_times_R
    if all_stride_times:
        avg_stride = np.mean(all_stride_times)
        double_support_times = [avg_stride * 0.15] * len(step_times)  # Estimate 15%
    
    return {
        'alternation_score': alternation_score,
        'step_times': step_times,
        'stride_times_L': stride_times_L,
        'stride_times_R': stride_times_R,
        'double_support_times': double_support_times,
        'all_events': all_events
    }
