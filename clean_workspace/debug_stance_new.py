#!/usr/bin/env python3
"""
Debug stance detection to understand why toe-off is happening so early
"""

import sys
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from core.pipeline.io_utils import read_xsens_bytes, extract_kinematics
from core.pipeline.stance_cycles import composite_stance, contacts_from_stance
from core.math.kinematics import quats_to_R_batch, world_vec, gyro_from_quat
from core.config.constants import STANCE_THR_W, STANCE_THR_A, STANCE_HYST_SAMPLES

def debug_stance_detection():
    # Load sample data
    data_dir = Path("sample data")
    files = {
        "ltibia": data_dir / "DEMO6_2_20250209_221452_174.csv",
        "rtibia": data_dir / "DEMO6_3_20250209_221452_179.csv",
    }
    
    print(f"Loading tibia sensor data...")
    print(f"Current thresholds: w={STANCE_THR_W} rad/s, a={STANCE_THR_A} m/s^2, hyst={STANCE_HYST_SAMPLES}")
    print()
    
    for side, filepath in files.items():
        print(f"=== {side.upper()} ===")
        
        # Load and extract data
        with open(filepath, "rb") as f:
            data_bytes = f.read()
        df = read_xsens_bytes(data_bytes)
        t, q, g, a = extract_kinematics(df)
        
        # Check if gyroscope data is available
        if g is None:
            print(f"No gyroscope data for {side}, deriving from quaternions...")
            g = gyro_from_quat(t, q)
        
        # Convert to world coordinates
        R = quats_to_R_batch(q)
        omega_W = world_vec(R, g)
        a_free_W = world_vec(R, a)
        
        # Calculate stance with current thresholds
        stance = composite_stance(omega_W, a_free_W, STANCE_THR_W, STANCE_THR_A, STANCE_HYST_SAMPLES)
        contacts = contacts_from_stance(stance)
        
        # Calculate magnitudes for analysis
        omega_mag = np.linalg.norm(omega_W, axis=1)
        a_mag = np.linalg.norm(a_free_W, axis=1)
        
        print(f"Data length: {len(t)} samples")
        print(f"Duration: {t[-1] - t[0]:.1f} seconds")
        print(f"Angular velocity range: {omega_mag.min():.2f} - {omega_mag.max():.2f} rad/s")
        print(f"Free acceleration range: {a_mag.min():.2f} - {a_mag.max():.2f} m/s^2")
        print(f"Stance percentage: {stance.mean()*100:.1f}%")
        print(f"Number of heel strikes detected: {len(contacts)}")
        
        if len(contacts) >= 2:
            cycle_lengths_samples = np.diff(contacts)
            cycle_lengths_seconds = cycle_lengths_samples / len(t) * (t[-1] - t[0])
            print(f"Cycle lengths: {cycle_lengths_seconds} seconds")
            
            # Look at first few cycles
            for i in range(min(3, len(contacts)-1)):
                start_idx = contacts[i]
                end_idx = contacts[i+1]
                cycle_stance = stance[start_idx:end_idx]
                
                # Find first False (toe-off)
                stance_to_swing = np.where(~cycle_stance)[0]
                if len(stance_to_swing) > 0:
                    toe_off_idx = stance_to_swing[0]
                    toe_off_pct = (toe_off_idx / len(cycle_stance)) * 100
                    print(f"  Cycle {i+1}: toe-off at {toe_off_pct:.1f}% of cycle")
                else:
                    print(f"  Cycle {i+1}: no toe-off detected (all stance)")
        
        print()
        
        # Test with relaxed thresholds
        print("Testing with relaxed thresholds:")
        test_thresholds = [
            (4.0, 1.5, "Relaxed"),
            (6.0, 2.0, "Very relaxed"),
            (1.0, 0.5, "Strict"),
        ]
        
        for w_thr, a_thr, label in test_thresholds:
            test_stance = composite_stance(omega_W, a_free_W, w_thr, a_thr, STANCE_HYST_SAMPLES)
            test_contacts = contacts_from_stance(test_stance)
            print(f"  {label} (w={w_thr}, a={a_thr}): {test_stance.mean()*100:.1f}% stance, {len(test_contacts)} contacts")
            
            if len(test_contacts) >= 2:
                start_idx = test_contacts[0]
                end_idx = test_contacts[1]
                cycle_stance = test_stance[start_idx:end_idx]
                stance_to_swing = np.where(~cycle_stance)[0]
                if len(stance_to_swing) > 0:
                    toe_off_pct = (stance_to_swing[0] / len(cycle_stance)) * 100
                    print(f"    First cycle toe-off: {toe_off_pct:.1f}%")
        
        print("="*50)

if __name__ == "__main__":
    debug_stance_detection()
