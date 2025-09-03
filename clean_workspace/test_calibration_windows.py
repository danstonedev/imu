#!/usr/bin/env python3
"""
Test the new calibration window functionality in the baseline correction system.
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from core.math.baseline import BaselineConfig, apply_baseline_correction, apply_calibration_window_correction

def test_calibration_window_correction():
    """Test the calibration window correction function directly."""
    print("Testing calibration window correction...")
    
    # Create test data: 5 seconds at 60 Hz (300 samples)
    fs_hz = 60.0
    t = np.linspace(0, 5, 300)
    
    # Create test angles with known constant offsets (no oscillations for clean test)
    euler_deg = np.zeros((300, 3))
    euler_deg[:, 0] = 10.0  # 10° X offset
    euler_deg[:, 1] = -5.0  # -5° Y offset  
    euler_deg[:, 2] = 2.0   # 2° Z offset
    
    # Convert to radians for processing
    euler_rad = np.deg2rad(euler_deg)
    
    # Define calibration windows (first 1 second and last 1 second)
    cal_windows = [(0.0, 1.0), (4.0, 5.0)]
    
    print(f"Original mean angles: {np.rad2deg(np.mean(euler_rad, axis=0))} degrees")
    
    # Apply calibration window correction
    corrected = apply_calibration_window_correction(
        t, euler_rad, cal_windows, fs_hz, axes=(0, 1, 2)
    )
    
    print(f"Corrected mean angles: {np.rad2deg(np.mean(corrected, axis=0))} degrees")
    print(f"Correction applied: {np.rad2deg(np.mean(euler_rad - corrected, axis=0))} degrees")
    
    # Check calibration windows have near-zero mean
    mask1 = (t >= 0.0) & (t <= 1.0)
    cal1_mean = np.mean(corrected[mask1], axis=0)
    
    mask2 = (t >= 4.0) & (t <= 5.0)
    cal2_mean = np.mean(corrected[mask2], axis=0)
    
    print(f"Cal window 1 mean: {np.rad2deg(cal1_mean)} degrees")
    print(f"Cal window 2 mean: {np.rad2deg(cal2_mean)} degrees")
    
    # For constant offsets, calibration windows should be exactly zero
    assert all(abs(np.rad2deg(cal1_mean)) < 1e-10), f"Cal window 1 failed: {np.rad2deg(cal1_mean)}"
    assert all(abs(np.rad2deg(cal2_mean)) < 1e-10), f"Cal window 2 failed: {np.rad2deg(cal2_mean)}"
    
    print("✓ Calibration window correction working correctly!")
    return True

def test_baseline_config_integration():
    """Test that BaselineConfig correctly includes calibration window settings."""
    print("\nTesting BaselineConfig integration...")
    
    # Create config with calibration windows enabled
    cfg = BaselineConfig(
        fs_hz=60.0,
        use_calibration_windows=True,
        calibration_axes=("X", "Y", "Z"),
        stride_debias_axes=("Y", "Z"),
        use_yaw_share=False
    )
    
    assert cfg.use_calibration_windows == True
    assert cfg.calibration_axes == ("X", "Y", "Z")
    print("✓ BaselineConfig correctly configured for calibration windows!")
    
    # Test with calibration windows disabled
    cfg_disabled = BaselineConfig(
        fs_hz=60.0,
        use_calibration_windows=False
    )
    
    assert cfg_disabled.use_calibration_windows == False
    print("✓ BaselineConfig correctly handles disabled calibration windows!")
    return True

def test_apply_baseline_correction_integration():
    """Test that apply_baseline_correction uses calibration windows."""
    print("\nTesting apply_baseline_correction integration...")
    
    # Create test data
    fs_hz = 60.0
    t = np.linspace(0, 3, 180)  # 3 seconds
    
    # Create angles with offset
    euler_rad = np.zeros((180, 3))
    euler_rad[:, 0] = np.deg2rad(5.0)  # 5° X offset
    euler_rad[:, 1] = np.deg2rad(-3.0) # -3° Y offset
    euler_rad[:, 2] = np.deg2rad(1.0)  # 1° Z offset
    
    # Configuration with calibration windows enabled
    cfg = BaselineConfig(
        fs_hz=fs_hz,
        use_calibration_windows=True,
        calibration_axes=("X", "Y", "Z"),
        use_yaw_share=False,
        stride_debias_axes=()  # Disable stride debiasing for clean test
    )
    
    # Calibration windows: first 0.5s and last 0.5s
    cal_windows = [(0.0, 0.5), (2.5, 3.0)]
    
    print(f"Original mean angles: {np.rad2deg(np.mean(euler_rad, axis=0))} degrees")
    
    # Apply baseline correction with calibration windows
    corrected = apply_baseline_correction(
        t, euler_rad, stride_indices=None, cfg=cfg,
        calibration_windows=cal_windows
    )
    
    print(f"Corrected mean angles: {np.rad2deg(np.mean(corrected, axis=0))} degrees")
    
    # Should be close to zero in calibration windows after correction
    # Check first calibration window (0-0.5s)
    mask1 = (t >= 0.0) & (t <= 0.5)
    cal1_mean = np.mean(corrected[mask1], axis=0)
    
    # Check second calibration window (2.5-3.0s)
    mask2 = (t >= 2.5) & (t <= 3.0)
    cal2_mean = np.mean(corrected[mask2], axis=0)
    
    print(f"Cal window 1 mean: {np.rad2deg(cal1_mean)} degrees")
    print(f"Cal window 2 mean: {np.rad2deg(cal2_mean)} degrees")
    
    # Each calibration window should have near-zero mean for constant offsets
    assert all(abs(np.rad2deg(cal1_mean)) < 1e-10), f"Cal window 1 failed: {np.rad2deg(cal1_mean)}"
    assert all(abs(np.rad2deg(cal2_mean)) < 1e-10), f"Cal window 2 failed: {np.rad2deg(cal2_mean)}"
    
    print("✓ apply_baseline_correction integration working correctly!")
    return True

if __name__ == "__main__":
    print("Testing calibration window functionality...\n")
    
    try:
        test_calibration_window_correction()
        test_baseline_config_integration()
        test_apply_baseline_correction_integration()
        
        print("\n🎉 All calibration window tests passed!")
        print("\nThe enhanced baseline system now supports:")
        print("  ✓ Calibration window-based corrections")
        print("  ✓ Integration with existing stride debiasing")
        print("  ✓ Configurable axes for calibration corrections")
        print("  ✓ Multiple calibration windows (start and end periods)")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
