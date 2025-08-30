#!/usr/bin/env python3
"""
Test optimal stance detection thresholds for this dataset
"""

import sys
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from core.pipeline.io_utils import read_xsens_bytes, extract_kinematics
from core.pipeline.stance_cycles import composite_stance, contacts_from_stance
from core.math.kinematics import quats_to_R_batch, world_vec, gyro_from_quat

def find_optimal_thresholds():
    # Test a range of thresholds
    data_dir = Path("sample data")
    filepath = data_dir / "DEMO6_2_20250209_221452_174.csv"  # Left tibia
    
    print("Finding optimal stance detection thresholds...")
    print("="*60)
    
    # Load data
    with open(filepath, "rb") as f:
        data_bytes = f.read()
    df = read_xsens_bytes(data_bytes)
    t, q, g, a = extract_kinematics(df)
    
    if g is None:
        g = gyro_from_quat(t, q)
    
    R = quats_to_R_batch(q)
    omega_W = world_vec(R, g)
    a_free_W = world_vec(R, a)
    
    # Calculate magnitudes for reference
    omega_mag = np.linalg.norm(omega_W, axis=1)
    a_mag = np.linalg.norm(a_free_W, axis=1)
    
    print(f"Data characteristics:")
    print(f"  Angular velocity: {omega_mag.min():.2f} - {omega_mag.max():.2f} rad/s (mean: {omega_mag.mean():.2f})")
    print(f"  Free acceleration: {a_mag.min():.2f} - {a_mag.max():.2f} m/s^2 (mean: {a_mag.mean():.2f})")
    print(f"  Duration: {t[-1] - t[0]:.1f} seconds")
    print()
    
    # Test various threshold combinations
    test_cases = [
        # (w_thresh, a_thresh, label)
        (1.0, 0.5, "Very Strict"),
        (2.0, 0.8, "Original"),
        (3.0, 1.0, "Loose"),
        (4.0, 1.5, "Looser"),
        (5.0, 2.0, "Very Loose"),
        (6.0, 2.5, "Extremely Loose"),
        (8.0, 3.0, "Maximum"),
        # Try asymmetric thresholds
        (10.0, 1.0, "High W, Low A"),
        (3.0, 4.0, "Low W, High A"),
    ]
    
    print("Threshold Testing Results:")
    print("="*60)
    print(f"{'Label':<15} {'W_thr':<6} {'A_thr':<6} {'Stance%':<8} {'Contacts':<9} {'Avg_Cycle':<10} {'TO_1st%':<8}")
    print("-"*60)
    
    for w_thr, a_thr, label in test_cases:
        stance = composite_stance(omega_W, a_free_W, w_thr, a_thr, 8)
        contacts = contacts_from_stance(stance)
        stance_pct = stance.mean() * 100
        
        avg_cycle_len = "N/A"
        toe_off_pct = "N/A"
        
        if len(contacts) >= 2:
            cycle_lengths = np.diff(contacts) / len(t) * (t[-1] - t[0])
            # Filter out very short cycles (likely noise)
            reasonable_cycles = cycle_lengths[(cycle_lengths >= 0.5) & (cycle_lengths <= 2.0)]
            if len(reasonable_cycles) > 0:
                avg_cycle_len = f"{reasonable_cycles.mean():.2f}s"
            
            # Check first reasonable cycle for toe-off
            for i in range(len(contacts)-1):
                start_idx = contacts[i]
                end_idx = contacts[i+1]
                cycle_len = (end_idx - start_idx) / len(t) * (t[-1] - t[0])
                
                if 0.5 <= cycle_len <= 2.0:  # Reasonable cycle length
                    cycle_stance = stance[start_idx:end_idx]
                    stance_to_swing = np.where(~cycle_stance)[0]
                    if len(stance_to_swing) > 0:
                        toe_off_pct = f"{(stance_to_swing[0] / len(cycle_stance)) * 100:.1f}%"
                    break
        
        print(f"{label:<15} {w_thr:<6.1f} {a_thr:<6.1f} {stance_pct:<8.1f} {len(contacts):<9} {avg_cycle_len:<10} {toe_off_pct:<8}")
    
    print()
    print("Recommendations:")
    print("- Target: ~60% stance, ~25-30 contacts in 50s, ~1.0-1.2s cycles, ~60% toe-off")
    print("- Best candidates appear to be combinations with high W threshold + moderate A threshold")

if __name__ == "__main__":
    find_optimal_thresholds()
