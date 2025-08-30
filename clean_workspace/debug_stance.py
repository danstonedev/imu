#!/usr/bin/env python3
"""Debug script to analyze stance detection and toe-off calculation."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core.pipeline.pipeline import run_pipeline_clean
from core.pipeline.stance_cycles import composite_stance
from core.pipeline.io_utils import load_imu_files
from core.config.constants import STANCE_THR_W, STANCE_THR_A, STANCE_HYST_SAMPLES

def debug_stance_detection():
    """Run pipeline and debug stance detection."""
    
    # Run the pipeline on sample data
    files = [
        "sample data/DEMO6_0_20250209_221452_176.csv",
        "sample data/DEMO6_1_20250209_221452_178.csv", 
        "sample data/DEMO6_2_20250209_221452_174.csv",
        "sample data/DEMO6_3_20250209_221452_179.csv",
        "sample data/DEMO6_4_20250209_221452_177.csv"
    ]
    
    print("Running clean pipeline...")
    results = run_clean_pipeline(files, height=170, mass=70)
    
    print("\n=== STANCE DETECTION ANALYSIS ===")
    
    # Check toe-off percentages
    if 'cycles_compare' in results:
        for key, data in results['cycles_compare'].items():
            if 'meta' in data and 'to_percent' in data['meta']:
                to_pct = data['meta']['to_percent']
                print(f"{key}: toe-off at {to_pct:.1f}%")
    
    # Let's examine the raw stance detection
    print(f"\nStance detection parameters:")
    print(f"  Angular velocity threshold: {STANCE_THR_W} rad/s")
    print(f"  Acceleration threshold: {STANCE_THR_A} m/s^2") 
    print(f"  Hysteresis samples: {STANCE_HYST_SAMPLES}")
    
    # Load one of the sample files to examine raw data
    print(f"\n=== RAW DATA ANALYSIS ===")
    try:
        df = pd.read_csv(files[1])  # Left tibia
        print(f"Loaded {files[1]}")
        print(f"Columns: {list(df.columns)}")
        
        # Check if we can find gyro and accel data
        gyro_cols = []
        accel_cols = []
        
        for col in df.columns:
            col_lower = col.lower()
            if any(x in col_lower for x in ['gyr', 'gyro', 'omega', 'rate']):
                gyro_cols.append(col)
            elif any(x in col_lower for x in ['acc', 'freeacc']):
                accel_cols.append(col)
        
        print(f"Gyro columns found: {gyro_cols}")
        print(f"Accel columns found: {accel_cols}")
        
        if len(gyro_cols) >= 3 and len(accel_cols) >= 3:
            # Extract gyro and accel data
            omega_data = df[gyro_cols[:3]].values
            accel_data = df[accel_cols[:3]].values
            
            # Calculate magnitudes
            omega_mag = np.linalg.norm(omega_data, axis=1)
            accel_mag = np.linalg.norm(accel_data, axis=1)
            
            print(f"\nGyro magnitude stats:")
            print(f"  Mean: {np.mean(omega_mag):.3f} rad/s")
            print(f"  Std: {np.std(omega_mag):.3f} rad/s")
            print(f"  Min: {np.min(omega_mag):.3f} rad/s")
            print(f"  Max: {np.max(omega_mag):.3f} rad/s")
            print(f"  % below threshold ({STANCE_THR_W}): {100*np.mean(omega_mag < STANCE_THR_W):.1f}%")
            
            print(f"\nAccel magnitude stats:")
            print(f"  Mean: {np.mean(accel_mag):.3f} m/s²")
            print(f"  Std: {np.std(accel_mag):.3f} m/s²")
            print(f"  Min: {np.min(accel_mag):.3f} m/s²")
            print(f"  Max: {np.max(accel_mag):.3f} m/s²")
            print(f"  % below threshold ({STANCE_THR_A}): {100*np.mean(accel_mag < STANCE_THR_A):.1f}%")
            
            # Apply stance detection
            stance_mask = composite_stance(omega_data, accel_data)
            print(f"\nStance detection results:")
            print(f"  % time in stance: {100*np.mean(stance_mask):.1f}%")
            
            # Look at stance transitions
            stance_changes = np.diff(stance_mask.astype(int))
            heel_strikes = np.where(stance_changes == 1)[0] + 1  # False to True
            toe_offs = np.where(stance_changes == -1)[0] + 1     # True to False
            
            print(f"  Number of heel strikes: {len(heel_strikes)}")
            print(f"  Number of toe-offs: {len(toe_offs)}")
            
            if len(heel_strikes) > 1 and len(toe_offs) > 0:
                # Calculate toe-off percentages for first few cycles
                print(f"\nFirst few cycle toe-off percentages:")
                for i in range(min(3, len(heel_strikes)-1)):
                    cycle_start = heel_strikes[i]
                    cycle_end = heel_strikes[i+1]
                    cycle_length = cycle_end - cycle_start
                    
                    # Find toe-offs in this cycle
                    cycle_toe_offs = toe_offs[(toe_offs > cycle_start) & (toe_offs < cycle_end)]
                    if len(cycle_toe_offs) > 0:
                        first_to = cycle_toe_offs[0]
                        to_percent = 100 * (first_to - cycle_start) / cycle_length
                        print(f"  Cycle {i+1}: {to_percent:.1f}% (samples {cycle_start}-{cycle_end}, TO at {first_to})")
                    else:
                        print(f"  Cycle {i+1}: No toe-off detected")
            
            # Check if thresholds are too strict/loose
            both_below = (omega_mag < STANCE_THR_W) & (accel_mag < STANCE_THR_A)
            print(f"\n% time both thresholds met: {100*np.mean(both_below):.1f}%")
            
            # Suggest alternative thresholds
            print(f"\nAlternative threshold suggestions:")
            for w_mult in [0.5, 0.75, 1.5, 2.0]:
                new_w_thr = w_mult * STANCE_THR_W
                new_stance = (omega_mag < new_w_thr) & (accel_mag < STANCE_THR_A)
                print(f"  Gyro threshold {new_w_thr:.1f} rad/s: {100*np.mean(new_stance):.1f}% stance")
            
        else:
            print("Could not find sufficient gyro/accel columns for analysis")
            
    except Exception as e:
        print(f"Error loading sample data: {e}")

if __name__ == "__main__":
    debug_stance_detection()
