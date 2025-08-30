import sys
from pathlib import Path
import numpy as np

# Ensure 'clean_workspace' root is on sys.path so 'core' package resolves in CI
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from core.pipeline.pipeline import compute_yaw_reference
from core.math.kinematics import quats_to_R_batch


def _yaw_to_quat(yaw_rad: np.ndarray):
    # ZYX with only yaw non-zero -> quaternion [w,x,y,z]
    c = np.cos(yaw_rad/2.0); s = np.sin(yaw_rad/2.0)
    # yaw about z only
    return np.stack([c, np.zeros_like(s), np.zeros_like(s), s], axis=1).astype(np.float64)


def test_yaw_reference_prefers_start_window():
    # Build a timeline with two still windows: start yaw ~0 rad, end yaw ~+pi/6
    t = np.linspace(0.0, 10.0, 1001)
    yaw = np.zeros_like(t)
    # From 0..2s yaw ~0; from 5..7s yaw ramps to 30deg
    yaw[(t >= 5.0) & (t <= 7.0)] = np.deg2rad(30.0)
    q = _yaw_to_quat(yaw)
    R = quats_to_R_batch(q)
    # still mask includes both windows, but we want to ensure start is chosen
    still = (t <= 2.0) | ((t >= 5.0) & (t <= 7.0))
    cal_windows = [
        {'label': 'start', 'start_s': float(0.0), 'end_s': float(2.0), 'duration_s': 2.0, 'samples': int(np.sum(t<=2.0))},
        {'label': 'end', 'start_s': float(5.0), 'end_s': float(7.0), 'duration_s': 2.0, 'samples': int(np.sum((t>=5.0)&(t<=7.0)))}
    ]
    yaw_ref, src = compute_yaw_reference(t, R, still, cal_windows)
    assert src == 'start'
    # near 0 rad
    assert abs(yaw_ref) < np.deg2rad(3.0)


def test_yaw_reference_falls_back_to_end_when_no_start():
    t = np.linspace(0.0, 10.0, 1001)
    yaw = np.zeros_like(t)
    yaw[(t >= 5.0) & (t <= 7.0)] = np.deg2rad(30.0)
    q = _yaw_to_quat(yaw)
    R = quats_to_R_batch(q)
    still = (t >= 5.0) & (t <= 7.0)
    cal_windows = [
        {'label': 'end', 'start_s': float(5.0), 'end_s': float(7.0), 'duration_s': 2.0, 'samples': int(np.sum(still))}
    ]
    yaw_ref, src = compute_yaw_reference(t, R, still, cal_windows)
    assert src == 'end'
    assert abs(yaw_ref - np.deg2rad(30.0)) < np.deg2rad(3.0)
