#!/usr/bin/env python3
"""
Test the new biomechanical heel strike detection algorithm.
"""

import sys
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from core.pipeline.io_utils import read_xsens_bytes, extract_kinematics
from core.pipeline.stance_cycles import composite_stance, contacts_from_stance
from core.pipeline.heel_strike_detection import detect_heel_strikes_biomech, bilateral_gait_cycles
from core.math.kinematics import quats_to_R_batch, world_vec, gyro_from_quat, estimate_fs

def compare_detection_methods():
    # Load sample data
    data_dir = Path("sample data")
    files = {
        "ltibia": data_dir / "DEMO6_2_20250209_221452_174.csv",
        "rtibia": data_dir / "DEMO6_3_20250209_221452_179.csv",
    }
    
    print("Comparing Heel Strike Detection Methods")
    print("="*60)
    
    results = {}
    
    for side, filepath in files.items():
        print(f"\n=== {side.upper()} ===")
        
        # Load and extract data
        with open(filepath, "rb") as f:
            data_bytes = f.read()
        df = read_xsens_bytes(data_bytes)
        t, q, g, a = extract_kinematics(df)
        
        if g is None:
            g = gyro_from_quat(t, q)
        
        # Convert to world coordinates
        R = quats_to_R_batch(q)
        omega_W = world_vec(R, g)
        a_free_W = world_vec(R, a)
        
        # Estimate sampling frequency
        fs = estimate_fs(t)
        duration = t[-1] - t[0]
        
        print(f"Duration: {duration:.1f}s, Sampling rate: {fs:.1f} Hz")
        
        # Method 1: Current stance-based approach
        print("\n--- Current Stance-Based Method ---")
        stance = composite_stance(omega_W, a_free_W, 3.0, 4.0, 8)
        contacts_old = contacts_from_stance(stance)
        
        stance_pct = stance.mean() * 100
        print(f"Stance percentage: {stance_pct:.1f}%")
        print(f"Heel strikes detected: {len(contacts_old)}")
        
        if len(contacts_old) > 1:
            old_intervals = np.diff(contacts_old) / fs
            avg_cycle_old = old_intervals.mean()
            cv_old = old_intervals.std() / avg_cycle_old
            print(f"Average cycle: {avg_cycle_old:.2f}s (CV: {cv_old:.2f})")
        
        # Method 2: New biomechanical approach
        print("\n--- New Biomechanical Method ---")
        try:
            contacts_new = detect_heel_strikes_biomech(t, omega_W, a_free_W, fs)
            
            print(f"Heel strikes detected: {len(contacts_new)}")
            
            if len(contacts_new) > 1:
                new_intervals = np.diff(contacts_new) / fs
                avg_cycle_new = new_intervals.mean()
                cv_new = new_intervals.std() / avg_cycle_new
                print(f"Average cycle: {avg_cycle_new:.2f}s (CV: {cv_new:.2f})")
                
                # Expected number of heel strikes for this duration
                expected_contacts = duration / 1.1  # Assume 1.1s average stride
                print(f"Expected heel strikes: ~{expected_contacts:.0f}")
                
                # Quality metrics
                print(f"Detection efficiency: {len(contacts_new)/expected_contacts:.2f}")
                print(f"Temporal consistency (CV): {cv_new:.2f} (lower is better)")
            
        except Exception as e:
            print(f"Error in biomechanical detection: {e}")
            contacts_new = np.array([])
        
        results[side] = {
            't': t,
            'contacts_old': contacts_old,
            'contacts_new': contacts_new,
            'duration': duration,
            'fs': fs
        }
    
    # Bilateral analysis
    print(f"\n{'='*60}")
    print("BILATERAL GAIT ANALYSIS")
    print("="*60)
    
    if 'ltibia' in results and 'rtibia' in results:
        # Old method bilateral analysis
        print("\n--- Current Method Bilateral Analysis ---")
        try:
            bilateral_old = bilateral_gait_cycles(
                results['ltibia']['t'], results['ltibia']['contacts_old'],
                results['rtibia']['t'], results['rtibia']['contacts_old']
            )
            print(f"Alternation score: {bilateral_old['alternation_score']:.2f} (1.0 = perfect)")
            print(f"Step times: {len(bilateral_old['step_times'])} steps")
            if bilateral_old['step_times']:
                step_mean = np.mean(bilateral_old['step_times'])
                step_cv = np.std(bilateral_old['step_times']) / step_mean
                print(f"Average step time: {step_mean:.2f}s (CV: {step_cv:.2f})")
        except Exception as e:
            print(f"Error in old bilateral analysis: {e}")
        
        # New method bilateral analysis  
        print("\n--- New Method Bilateral Analysis ---")
        try:
            bilateral_new = bilateral_gait_cycles(
                results['ltibia']['t'], results['ltibia']['contacts_new'],
                results['rtibia']['t'], results['rtibia']['contacts_new']
            )
            print(f"Alternation score: {bilateral_new['alternation_score']:.2f} (1.0 = perfect)")
            print(f"Step times: {len(bilateral_new['step_times'])} steps")
            if bilateral_new['step_times']:
                step_mean = np.mean(bilateral_new['step_times'])
                step_cv = np.std(bilateral_new['step_times']) / step_mean
                print(f"Average step time: {step_mean:.2f}s (CV: {step_cv:.2f})")
        except Exception as e:
            print(f"Error in new bilateral analysis: {e}")
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("="*60)
    print("Good heel strike detection should have:")
    print("- ~25-30 heel strikes in 50 seconds")
    print("- Average cycle time ~1.0-1.2 seconds")
    print("- Low coefficient of variation (CV < 0.2)")
    print("- High alternation score (> 0.8)")
    print("- Stance percentage ~60% (but depends on walking speed)")

if __name__ == "__main__":
    compare_detection_methods()
