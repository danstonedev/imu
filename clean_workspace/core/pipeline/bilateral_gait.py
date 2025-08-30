"""
Bilateral gait cycle detection with proper left-right coordination
"""
import numpy as np
from typing import Tuple, List, Dict

def detect_bilateral_gait_cycles(
    tL: np.ndarray, stance_L: np.ndarray,
    tR: np.ndarray, stance_R: np.ndarray,
    common_time: np.ndarray
) -> Dict:
    """
    Detect gait cycles with proper left-right coordination.
    
    Args:
        tL, tR: Time arrays for left and right legs
        stance_L, stance_R: Stance phase boolean arrays
        common_time: Common time base for both legs
    
    Returns:
        Dict with bilateral gait cycle information
    """
    from .stance_cycles import contacts_from_stance
    
    # Interpolate stance phases to common time base
    stance_L_interp = np.interp(common_time, tL, stance_L.astype(float)) > 0.5
    stance_R_interp = np.interp(common_time, tR, stance_R.astype(float)) > 0.5
    
    # Detect heel strikes on common timeline
    contacts_L_common = contacts_from_stance(stance_L_interp)
    contacts_R_common = contacts_from_stance(stance_R_interp)
    
    # Create combined events list
    events = []
    for contact in contacts_L_common:
        events.append((common_time[contact], contact, 'L', 'heel_strike'))
    for contact in contacts_R_common:
        events.append((common_time[contact], contact, 'R', 'heel_strike'))
    
    # Sort by time
    events.sort(key=lambda x: x[0])
    
    # Analyze alternating pattern
    pattern_quality = analyze_gait_pattern(events)
    
    # Define gait cycles (heel strike to heel strike of same leg)
    left_cycles = []
    right_cycles = []
    
    # Extract left heel strikes
    left_hs = [(t, idx) for t, idx, leg, event in events if leg == 'L']
    right_hs = [(t, idx) for t, idx, leg, event in events if leg == 'R']
    
    # Define left gait cycles
    for i in range(len(left_hs) - 1):
        cycle_start_time = left_hs[i][0]
        cycle_end_time = left_hs[i + 1][0]
        cycle_start_idx = left_hs[i][1]
        cycle_end_idx = left_hs[i + 1][1]
        
        # Find opposite (right) heel strike within this cycle
        right_hs_in_cycle = [hs for hs in right_hs 
                           if cycle_start_time < hs[0] < cycle_end_time]
        
        toe_off_percent = None
        if right_hs_in_cycle:
            # Use first right heel strike as toe-off reference
            right_hs_time = right_hs_in_cycle[0][0]
            toe_off_percent = ((right_hs_time - cycle_start_time) / 
                             (cycle_end_time - cycle_start_time)) * 100
        
        left_cycles.append({
            'start_time': cycle_start_time,
            'end_time': cycle_end_time,
            'start_idx': cycle_start_idx,
            'end_idx': cycle_end_idx,
            'duration': cycle_end_time - cycle_start_time,
            'toe_off_percent': toe_off_percent
        })
    
    # Define right gait cycles (similar process)
    for i in range(len(right_hs) - 1):
        cycle_start_time = right_hs[i][0]
        cycle_end_time = right_hs[i + 1][0]
        cycle_start_idx = right_hs[i][1]
        cycle_end_idx = right_hs[i + 1][1]
        
        # Find opposite (left) heel strike within this cycle
        left_hs_in_cycle = [hs for hs in left_hs 
                          if cycle_start_time < hs[0] < cycle_end_time]
        
        toe_off_percent = None
        if left_hs_in_cycle:
            left_hs_time = left_hs_in_cycle[0][0]
            toe_off_percent = ((left_hs_time - cycle_start_time) / 
                             (cycle_end_time - cycle_start_time)) * 100
        
        right_cycles.append({
            'start_time': cycle_start_time,
            'end_time': cycle_end_time,
            'start_idx': cycle_start_idx,
            'end_idx': cycle_end_idx,
            'duration': cycle_end_time - cycle_start_time,
            'toe_off_percent': toe_off_percent
        })
    
    return {
        'events': events,
        'pattern_quality': pattern_quality,
        'left_cycles': left_cycles,
        'right_cycles': right_cycles,
        'common_time': common_time
    }


def analyze_gait_pattern(events: List) -> Dict:
    """
    Analyze the quality of the detected gait pattern.
    
    Args:
        events: List of (time, idx, leg, event_type) tuples
    
    Returns:
        Dict with pattern quality metrics
    """
    if len(events) < 4:
        return {'quality': 'insufficient_data', 'alternation_score': 0.0}
    
    # Check alternation pattern
    legs = [event[2] for event in events]
    alternations = 0
    total_transitions = len(legs) - 1
    
    for i in range(total_transitions):
        if legs[i] != legs[i + 1]:
            alternations += 1
    
    alternation_score = alternations / total_transitions if total_transitions > 0 else 0
    
    # Check cycle duration consistency
    cycle_durations = []
    for leg in ['L', 'R']:
        leg_events = [e for e in events if e[2] == leg]
        if len(leg_events) >= 2:
            for i in range(len(leg_events) - 1):
                duration = leg_events[i + 1][0] - leg_events[i][0]
                cycle_durations.append(duration)
    
    duration_cv = 0.0
    if cycle_durations:
        duration_cv = np.std(cycle_durations) / np.mean(cycle_durations)
    
    # Overall quality assessment
    if alternation_score > 0.8 and duration_cv < 0.3:
        quality = 'excellent'
    elif alternation_score > 0.6 and duration_cv < 0.5:
        quality = 'good'
    elif alternation_score > 0.4:
        quality = 'fair'
    else:
        quality = 'poor'
    
    return {
        'quality': quality,
        'alternation_score': alternation_score,
        'duration_cv': duration_cv,
        'total_events': len(events),
        'cycle_durations': cycle_durations
    }
