from __future__ import annotations
from pathlib import Path
import numpy as np
from core.pipeline.pipeline import run_pipeline_clean

root = Path(__file__).resolve().parents[1]


def pick(pattern: str) -> str:
    matches = sorted(root.joinpath('sample data').glob(pattern))
    if not matches:
        raise FileNotFoundError(pattern)
    return str(matches[0])


def max_abs_step_deg(arr2d: np.ndarray) -> float:
    if arr2d is None or len(arr2d) < 2:
        return 0.0
    A = np.asarray(arr2d, dtype=float)
    if A.ndim == 1:
        A = A[:, None]
    d = np.diff(A, axis=0)
    return float(np.nanmax(np.abs(d)))


def test_angles_unwrap_continuity_sample():
    # Use bundled sample data and run the clean pipeline
    paths = {
        'pelvis': pick('DEMO6_0_*.csv'),
        'lfemur': pick('DEMO6_1_*.csv'),
        'rfemur': pick('DEMO6_2_*.csv'),
        'ltibia': pick('DEMO6_3_*.csv'),
        'rtibia': pick('DEMO6_4_*.csv'),
    }
    out = run_pipeline_clean(paths, height_m=1.75, mass_kg=75.0, options={'do_cal': True, 'yaw_align': True})

    # Angle arrays in degrees, shape (T,3)
    L_hip = np.asarray(out.get('L_hip_angles_deg'))
    R_hip = np.asarray(out.get('R_hip_angles_deg'))
    L_knee = np.asarray(out.get('L_knee_angles_deg'))
    R_knee = np.asarray(out.get('R_knee_angles_deg'))

    # Ensure present
    for A in (L_hip, R_hip, L_knee, R_knee):
        assert isinstance(A, np.ndarray) and A.ndim == 2 and A.shape[1] == 3 and A.shape[0] > 2

    # After unwrap, consecutive steps should not exhibit wrap jumps (~360 deg)
    # Use a generous bound that real motion between samples will not exceed
    bound = 150.0
    assert max_abs_step_deg(L_hip) < bound
    assert max_abs_step_deg(R_hip) < bound
    assert max_abs_step_deg(L_knee) < bound
    assert max_abs_step_deg(R_knee) < bound
