"""Centralized constants, thresholds, and column aliases for clean pipeline."""
from __future__ import annotations

import numpy as np

# Physics
G = np.array([0.0, 0.0, -9.80665], dtype=float)

# Anthropometrics (teaching-grade)
FEMUR_LEN_FRACTION = 0.245  # l_th ≈ 0.245 * height
FEMUR_MASS_FRACTION = 0.10  # femur mass ≈ 10% body mass
FEMUR_LEN_MIN = 0.20
FEMUR_LEN_MAX = 0.50
DAMPING_COEFF = 0.02        # multiplier on omega

# Still detection (calibration)
STILL_THR_W = 0.08          # rad/s
STILL_THR_A = 0.30          # m/s^2
STILL_SMOOTH_S = 0.7        # seconds
STILL_MIN_S_PRIMARY = 1.0
STILL_MIN_S_SECONDARY = 0.5
STILL_EDGE_FRAC = 0.25
STILL_FALLBACK_WINDOW_S = 0.75

# Stance detection (tibia based)
STANCE_THR_W = 8.0          # rad/s (very relaxed for quaternion-derived gyro)
STANCE_THR_A = 5.0          # m/s^2 (very relaxed for robust detection)
STANCE_HYST_SAMPLES = 16    # increased hysteresis to reduce false transitions

# Cycle averaging
CYCLE_N = 101
CYCLE_DROP_FIRST = 3
CYCLE_DROP_LAST = 3
CYCLE_MIN_DUR_S = 0.4
CYCLE_MAX_DUR_S = 2.0

# CSV column aliases
TIME_CANDS = [
    "sampletimefine","time_s","time","timestamp_s","timestamps_s","seconds","sec"
]
QW = ["quat_w","w","qw","q_w","quaternion_w"]
QX = ["quat_x","x","qx","q_x","quaternion_x"]
QY = ["quat_y","y","qy","q_y","quaternion_y"]
QZ = ["quat_z","z","qz","q_z","quaternion_z"]
GYR = {
    "x": ["gyr_x","gyro_x","wx","omega_x","gyr__x_","rate_x"],
    "y": ["gyr_y","gyro_y","wy","omega_y","gyr__y_","rate_y"],
    "z": ["gyr_z","gyro_z","wz","omega_z","gyr__z_","rate_z"],
}
ACC = {
    "x": ["freeacc_x","acc_free_x","ax_free","free_acc_x","acc_x_free","freeaccx"],
    "y": ["freeacc_y","acc_free_y","ay_free","free_acc_y","acc_y_free","freeaccy"],
    "z": ["freeacc_z","acc_free_z","az_free","free_acc_z","acc_z_free","freeaccz"],
}
