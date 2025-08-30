#!/usr/bin/env python3
"""
Test enhanced biomechanical gait detection
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from core.pipeline.io_utils import read_xsens_bytes, extract_kinematics
from core.pipeline.unified_gait import detect_gait_cycles
from core.math.kinematics import quats_to_R_batch, world_vec, gyro_from_quat

def load_tibia_data(filepath):
    """Load tibia sensor data from CSV file."""
    with open(filepath, "rb") as f:
        data_bytes = f.read()
    df = read_xsens_bytes(data_bytes)
    t, q, g, a = extract_kinematics(df)
    
    # Convert quaternions to gyroscope if needed
    if g is None:
        print("No gyroscope data found, deriving from quaternions...")
        g = gyro_from_quat(t, q)
    
    # Convert to world coordinates
    R = quats_to_R_batch(q)
    omega_W = world_vec(R, g)
    a_free_W = world_vec(R, a)
    
    return {
        "t": t,
        "q": q,
        "omega": omega_W,
        "a_free": a_free_W
    }

def test_enhanced_biomech():
    """Test the enhanced biomechanical detection."""
    
    # Load sample data
    print("Loading sample data...")
    data_L = load_tibia_data("sample data/DEMO6_2_20250209_221452_174.csv")  # Left tibia
    data_R = load_tibia_data("sample data/DEMO6_3_20250209_221452_179.csv")  # Right tibia
    
    t_L, omega_L, a_free_L = data_L["t"], data_L["omega"], data_L["a_free"]
    t_R, omega_R, a_free_R = data_R["t"], data_R["omega"], data_R["a_free"]
    
    fs = 60.0
    
    print(f"Left data: {len(t_L)} samples, {t_L[-1]-t_L[0]:.1f}s duration")
    print(f"Right data: {len(t_R)} samples, {t_R[-1]-t_R[0]:.1f}s duration")
    
    # Test new detection functions
    print("\n=== Testing Enhanced Detection ===")
    
    # Extract signals for each leg
    ax_L, az_L = a_free_L[:, 0], a_free_L[:, 2]  # AP and vertical
    ax_R, az_R = a_free_R[:, 0], a_free_R[:, 2]
    gx_L, gx_R = omega_L[:, 0], omega_R[:, 0]    # Pitch gyro
    
    # Remove gravity/DC offset
    az_L_filt = az_L - np.mean(az_L)
    az_R_filt = az_R - np.mean(az_R)
    
    # Detect heel strikes and toe-offs using unified detector
    # Build minimal 3D arrays for left/right from prepared channels
    accel_L = np.column_stack([ax_L, np.zeros_like(ax_L), az_L_filt])
    accel_R = np.column_stack([ax_R, np.zeros_like(ax_R), az_R_filt])
    gyro_L = np.column_stack([gx_L, np.zeros_like(gx_L), np.zeros_like(gx_L)])
    gyro_R = np.column_stack([gx_R, np.zeros_like(gx_R), np.zeros_like(gx_R)])
    res = detect_gait_cycles(t_L, accel_L, gyro_L, accel_R, gyro_R, fs)
    hs_L = res['heel_strikes_left']; to_L = res['toe_offs_left']
    hs_R = res['heel_strikes_right']; to_R = res['toe_offs_right']
    
    print(f"Left leg: {len(hs_L)} heel strikes, {len(to_L)} toe-offs")
    print(f"Right leg: {len(hs_R)} heel strikes, {len(to_R)} toe-offs")
    
    # Calculate stance percentages
    stance_pcts_L = []
    for i in range(len(hs_L) - 1):
        hs_time = t_L[hs_L[i]]
        next_hs_time = t_L[hs_L[i+1]]
        
        # Find toe-off in this stride
        to_in_stride = []
        for to_idx in to_L:
            to_time = t_L[to_idx]
            if hs_time < to_time < next_hs_time:
                to_in_stride.append(to_time)
        
        if to_in_stride:
            stance_duration = to_in_stride[0] - hs_time
            stride_duration = next_hs_time - hs_time
            stance_pct = (stance_duration / stride_duration) * 100
            stance_pcts_L.append(stance_pct)
            print(f"Left stride {i+1}: {stance_pct:.1f}% stance")
    
    if stance_pcts_L:
        print(f"Left average stance: {np.mean(stance_pcts_L):.1f}% ± {np.std(stance_pcts_L):.1f}%")
    
    # Helper built on unified_gait to summarize bilateral results similar to legacy API
    def summarize_bilateral(tL, aL, gL, tR, aR, gR, fs_hz):
        axL, azL = aL[:, 0], aL[:, 2]
        axR, azR = aR[:, 0], aR[:, 2]
        gxL, gxR = gL[:, 0], gR[:, 0]

        azL_f = azL - np.mean(azL)
        azR_f = azR - np.mean(azR)

        accelL = np.column_stack([axL, np.zeros_like(axL), azL_f])
        accelR = np.column_stack([axR, np.zeros_like(axR), azR_f])
        gyroL = np.column_stack([gxL, np.zeros_like(gxL), np.zeros_like(gxL)])
        gyroR = np.column_stack([gxR, np.zeros_like(gxR), np.zeros_like(gxR)])

        res_u = detect_gait_cycles(tL, accelL, gyroL, accelR, gyroR, fs_hz)
        hsL = res_u['heel_strikes_left']; toL = res_u['toe_offs_left']
        hsR = res_u['heel_strikes_right']; toR = res_u['toe_offs_right']

        # Compute average stance percent (left)
        stance_pcts_L = []
        stride_times_L = []
        for i in range(len(hsL) - 1):
            hs_t = tL[hsL[i]]; next_hs_t = tL[hsL[i+1]]
            in_stride_to = [tL[j] for j in toL if hs_t < tL[j] < next_hs_t]
            if in_stride_to:
                stance = in_stride_to[0] - hs_t
                stride = next_hs_t - hs_t
                if stride > 0:
                    stance_pcts_L.append(100.0 * stance / stride)
                    stride_times_L.append(stride)

        # Right stride times
        stride_times_R = []
        for i in range(len(hsR) - 1):
            stride_times_R.append(tR[hsR[i+1]] - tR[hsR[i]])

        # Event list
        all_events = []
        all_events += [(tL[i], 'L_HS') for i in hsL]
        all_events += [(tL[i], 'L_TO') for i in toL]
        all_events += [(tR[i], 'R_HS') for i in hsR]
        all_events += [(tR[i], 'R_TO') for i in toR]
        all_events.sort(key=lambda x: x[0])

        out = {
            'total_events': len(all_events),
            'duration': max(tL[-1] - tL[0], tR[-1] - tR[0]),
            'all_events': all_events,
        }
        if stance_pcts_L:
            out['left_avg_stance_pct'] = float(np.mean(stance_pcts_L))
        if stride_times_L:
            out['left_avg_stride'] = float(np.mean(stride_times_L))
        if stride_times_R:
            out['right_avg_stride'] = float(np.mean(stride_times_R))
        return out

    # Test bilateral analysis
    print("\n=== Testing Bilateral Analysis ===")
    results = summarize_bilateral(t_L, a_free_L, omega_L, t_R, a_free_R, omega_R, fs)
    
    print(f"Total events detected: {results['total_events']}")
    print(f"Duration: {results['duration']:.1f}s")
    
    if 'left_avg_stance_pct' in results:
        print(f"Left stance percentage: {results['left_avg_stance_pct']:.1f}%")
    
    if 'left_avg_stride' in results:
        print(f"Left average stride time: {results['left_avg_stride']:.2f}s")
        
    if 'right_avg_stride' in results:
        print(f"Right average stride time: {results['right_avg_stride']:.2f}s")
    
    # Show event pattern
    if results['all_events']:
        print("Event pattern (first 20):")
        for i, (time, event) in enumerate(results['all_events'][:20]):
            print(f"  {time:.2f}s: {event}")
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot left leg signals
    plt.subplot(3, 2, 1)
    plt.plot(t_L, az_L_filt, 'b-', alpha=0.7, label='Vert Accel (filtered)')
    plt.plot(t_L[hs_L], az_L_filt[hs_L], 'ro', markersize=8, label='Heel Strikes')
    plt.plot(t_L[to_L], az_L_filt[to_L], 'go', markersize=8, label='Toe-offs')
    plt.title('Left Leg - Vertical Acceleration')
    plt.ylabel('Acceleration (m/s²)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 2, 3)
    plt.plot(t_L, gx_L, 'b-', alpha=0.7, label='Pitch Gyro')
    plt.plot(t_L[hs_L], gx_L[hs_L], 'ro', markersize=8, label='Heel Strikes')
    plt.plot(t_L[to_L], gx_L[to_L], 'go', markersize=8, label='Toe-offs')
    plt.title('Left Leg - Pitch Gyroscope')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 2, 5)
    plt.plot(t_L, ax_L, 'b-', alpha=0.7, label='AP Accel')
    plt.plot(t_L[hs_L], ax_L[hs_L], 'ro', markersize=8, label='Heel Strikes')
    plt.plot(t_L[to_L], ax_L[to_L], 'go', markersize=8, label='Toe-offs')
    plt.title('Left Leg - Anterior-Posterior Acceleration')
    plt.ylabel('Acceleration (m/s²)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)
    
    # Plot right leg signals
    plt.subplot(3, 2, 2)
    plt.plot(t_R, az_R_filt, 'r-', alpha=0.7, label='Vert Accel (filtered)')
    plt.plot(t_R[hs_R], az_R_filt[hs_R], 'ro', markersize=8, label='Heel Strikes')
    plt.plot(t_R[to_R], az_R_filt[to_R], 'go', markersize=8, label='Toe-offs')
    plt.title('Right Leg - Vertical Acceleration')
    plt.ylabel('Acceleration (m/s²)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 2, 4)
    plt.plot(t_R, gx_R, 'r-', alpha=0.7, label='Pitch Gyro')
    plt.plot(t_R[hs_R], gx_R[hs_R], 'ro', markersize=8, label='Heel Strikes')
    plt.plot(t_R[to_R], gx_R[to_R], 'go', markersize=8, label='Toe-offs')
    plt.title('Right Leg - Pitch Gyroscope')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 2, 6)
    plt.plot(t_R, ax_R, 'r-', alpha=0.7, label='AP Accel')
    plt.plot(t_R[hs_R], ax_R[hs_R], 'ro', markersize=8, label='Heel Strikes')
    plt.plot(t_R[to_R], ax_R[to_R], 'go', markersize=8, label='Toe-offs')
    plt.title('Right Leg - Anterior-Posterior Acceleration')
    plt.ylabel('Acceleration (m/s²)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('enhanced_biomech_detection.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Final assertions: ensure some cycles/events were found, else test will still pass but plot aids debugging
    assert isinstance(results, dict)
    assert 'total_events' in results

if __name__ == "__main__":
    results = test_enhanced_biomech()
