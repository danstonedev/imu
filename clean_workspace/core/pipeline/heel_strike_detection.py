"""
Legacy biomechanical heel strike detection; now delegates to unified_gait.
This file remains for backward compatibility with dev scripts.

DEPRECATED: Prefer core.pipeline.unified_gait for detection and
core.pipeline.bilateral_gait for bilateral cycle analysis.
"""
from __future__ import annotations
import numpy as np
import warnings
from .unified_gait import detect_heel_strikes_biomech as _ug_detect_heel_strikes

__all__ = ["detect_heel_strikes_biomech", "bilateral_gait_cycles"]

def detect_heel_strikes_biomech(t: np.ndarray, gyro: np.ndarray, accel: np.ndarray, 
                               fs: float, debug: bool = False) -> np.ndarray | dict:
    """
    Deprecated: use core.pipeline.unified_gait.detect_heel_strikes_biomech.
    This wrapper preserves the legacy signature (gyro, accel order).
    If debug=True, returns only indices (debug payload not supported in unified path).
    """
    warnings.warn(
        "core.pipeline.heel_strike_detection.detect_heel_strikes_biomech is deprecated; "
        "use core.pipeline.unified_gait.detect_heel_strikes_biomech",
        DeprecationWarning,
        stacklevel=2,
    )
    hs = _ug_detect_heel_strikes(t, accel, gyro, fs)
    if debug:
        return {'heel_strikes': hs}
    return hs

def bilateral_gait_cycles(t_L: np.ndarray, hs_L: np.ndarray, 
                         t_R: np.ndarray, hs_R: np.ndarray) -> dict:
    """
    Analyze bilateral gait coordination and compute proper gait cycles.
    
    Args:
        t_L, t_R: Time arrays for left and right legs
        hs_L, hs_R: Heel strike indices for left and right legs
        
    Returns:
        Dictionary with bilateral gait analysis results (legacy schema).
        Prefer core.pipeline.bilateral_gait.detect_bilateral_gait_cycles for new code.
    """
    warnings.warn(
        "core.pipeline.heel_strike_detection.bilateral_gait_cycles is deprecated; "
        "prefer core.pipeline.bilateral_gait.detect_bilateral_gait_cycles",
        DeprecationWarning,
        stacklevel=2,
    )
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
