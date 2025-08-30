#!/usr/bin/env python3
"""
Test bilateral gait cycle detection
"""

import sys
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from core.pipeline.io_utils import read_xsens_bytes, extract_kinematics
from core.pipeline.stance_cycles import composite_stance
from core.pipeline.bilateral_gait import detect_bilateral_gait_cycles
from core.math.kinematics import quats_to_R_batch, world_vec, gyro_from_quat
from core.config.constants import STANCE_THR_W, STANCE_THR_A, STANCE_HYST_SAMPLES

def test_bilateral_gait():
    # Load sample data
    root = Path(__file__).resolve().parent
    data_dir = root / "sample data"
    files = {
        "ltibia": data_dir / "DEMO6_2_20250209_221452_174.csv",
        "rtibia": data_dir / "DEMO6_3_20250209_221452_179.csv",
    }
    
    print("Testing Bilateral Gait Cycle Detection")
    print("="*50)
    
    # Load left tibia data
    with open(files["ltibia"], "rb") as f:
        lt_bytes = f.read()
    lt_df = read_xsens_bytes(lt_bytes)
    tL, qL, gL, aL = extract_kinematics(lt_df)
    
    if gL is None:
        gL = gyro_from_quat(tL, qL)
    
    RL = quats_to_R_batch(qL)
    omegaL_W = world_vec(RL, gL)
    aL_W = world_vec(RL, aL)
    stance_L = composite_stance(omegaL_W, aL_W, STANCE_THR_W, STANCE_THR_A, STANCE_HYST_SAMPLES)
    
    # Load right tibia data
    with open(files["rtibia"], "rb") as f:
        rt_bytes = f.read()
    rt_df = read_xsens_bytes(rt_bytes)
    tR, qR, gR, aR = extract_kinematics(rt_df)
    
    if gR is None:
        gR = gyro_from_quat(tR, qR)
    
    RR = quats_to_R_batch(qR)
    omegaR_W = world_vec(RR, gR)
    aR_W = world_vec(RR, aR)
    stance_R = composite_stance(omegaR_W, aR_W, STANCE_THR_W, STANCE_THR_A, STANCE_HYST_SAMPLES)
    
    # Create common time base
    t_start = max(tL[0], tR[0])
    t_end = min(tL[-1], tR[-1])
    common_time = np.linspace(t_start, t_end, int((t_end - t_start) * 100))  # 100 Hz
    
    print(f"Left leg: {len(tL)} samples, {tL[-1] - tL[0]:.1f}s")
    print(f"Right leg: {len(tR)} samples, {tR[-1] - tR[0]:.1f}s")
    print(f"Common time: {len(common_time)} samples, {t_end - t_start:.1f}s")
    print()
    
    # Detect bilateral gait cycles
    bilateral_results = detect_bilateral_gait_cycles(tL, stance_L, tR, stance_R, common_time)
    
    print("BILATERAL GAIT ANALYSIS")
    print("-"*30)
    print(f"Pattern quality: {bilateral_results['pattern_quality']['quality']}")
    print(f"Alternation score: {bilateral_results['pattern_quality']['alternation_score']:.2f}")
    print(f"Duration CV: {bilateral_results['pattern_quality']['duration_cv']:.2f}")
    print(f"Total events detected: {bilateral_results['pattern_quality']['total_events']}")
    print()
    
    # Show event sequence
    print("EVENT SEQUENCE:")
    for i, (time, idx, leg, event) in enumerate(bilateral_results['events'][:20]):  # First 20 events
        print(f"  {i+1:2d}. {time:6.2f}s - {leg} {event}")
    if len(bilateral_results['events']) > 20:
        print(f"  ... and {len(bilateral_results['events']) - 20} more events")
    print()
    
    # Analyze left cycles
    left_cycles = bilateral_results['left_cycles']
    if left_cycles:
        print("LEFT LEG CYCLES:")
        toe_offs = [c['toe_off_percent'] for c in left_cycles if c['toe_off_percent'] is not None]
        durations = [c['duration'] for c in left_cycles]
        
        print(f"  Total cycles: {len(left_cycles)}")
        print(f"  Average duration: {np.mean(durations):.2f}s ± {np.std(durations):.2f}s")
        if toe_offs:
            print(f"  Average toe-off: {np.mean(toe_offs):.1f}% ± {np.std(toe_offs):.1f}%")
            print(f"  Toe-off range: {min(toe_offs):.1f}% - {max(toe_offs):.1f}%")
        
        print("  First 5 cycles:")
        for i, cycle in enumerate(left_cycles[:5]):
            duration = cycle['duration']
            toe_off = cycle['toe_off_percent']
            toe_off_str = f"{toe_off:.1f}%" if toe_off else "N/A"
            print(f"    {i+1}. {duration:.2f}s, toe-off at {toe_off_str}")
    print()
    
    # Analyze right cycles  
    right_cycles = bilateral_results['right_cycles']
    if right_cycles:
        print("RIGHT LEG CYCLES:")
        toe_offs = [c['toe_off_percent'] for c in right_cycles if c['toe_off_percent'] is not None]
        durations = [c['duration'] for c in right_cycles]
        
        print(f"  Total cycles: {len(right_cycles)}")
        print(f"  Average duration: {np.mean(durations):.2f}s ± {np.std(durations):.2f}s")
        if toe_offs:
            print(f"  Average toe-off: {np.mean(toe_offs):.1f}% ± {np.std(toe_offs):.1f}%")
            print(f"  Toe-off range: {min(toe_offs):.1f}% - {max(toe_offs):.1f}%")
        
        print("  First 5 cycles:")
        for i, cycle in enumerate(right_cycles[:5]):
            duration = cycle['duration']
            toe_off = cycle['toe_off_percent']
            toe_off_str = f"{toe_off:.1f}%" if toe_off else "N/A"
            print(f"    {i+1}. {duration:.2f}s, toe-off at {toe_off_str}")
    
    print()
    print("EXPECTED PATTERN:")
    print("- Toe-off should occur around 50-60% of gait cycle")
    print("- Cycle durations should be consistent (~1.0-1.2s for normal walking)")
    print("- Alternation score should be close to 1.0 for good gait pattern")

if __name__ == "__main__":
    test_bilateral_gait()
