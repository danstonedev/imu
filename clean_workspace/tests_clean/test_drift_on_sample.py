from __future__ import annotations
from pathlib import Path
import sys, os
import numpy as np
import pytest

from core.pipeline.io_utils import read_xsens_bytes, extract_kinematics_ex
from core.math.kinematics import (
    quats_to_R_batch,
    yaw_from_R,
    hip_angles_xyz,
    knee_angles_xyz,
    resample_quat,
)


root = Path(__file__).resolve().parents[1]
# Make sure 'core' package is importable even if pytest runs from workspace root
if str(root) not in sys.path:
    sys.path.insert(0, str(root))


def _pick(pattern: str) -> str:
    matches = sorted(root.joinpath("sample data").glob(pattern))
    if not matches:
        raise FileNotFoundError(pattern)
    return str(matches[0])


def _slope_deg_per_min(t: np.ndarray, y_rad: np.ndarray) -> float:
    t = np.asarray(t, float)
    y = np.unwrap(np.asarray(y_rad, float))
    if t.size < 5:
        return 0.0
    # robust center by subtracting median to avoid intercept extremes
    t0 = t - t[0]
    p = np.polyfit(t0, y, 1)
    slope_rad_s = float(p[0])
    return slope_rad_s * (180.0 / np.pi) * 60.0  # deg/min


@pytest.mark.slow
def test_sample_drift_diagnostics(capfd):
    # Load sample pelvis and left femur
    pel = Path(_pick("DEMO6_0_*.csv")).read_bytes()
    lf = Path(_pick("DEMO6_1_*.csv")).read_bytes()
    lt = Path(_pick("DEMO6_2_*.csv")).read_bytes()
    rf = Path(_pick("DEMO6_3_*.csv")).read_bytes()
    rt = Path(_pick("DEMO6_4_*.csv")).read_bytes()

    tP, qP, gP, aP, metaP = extract_kinematics_ex(read_xsens_bytes(pel))
    tLF, qLF, gLF, aLF, metaLF = extract_kinematics_ex(read_xsens_bytes(lf))
    tLT, qLT, gLT, aLT, metaLT = extract_kinematics_ex(read_xsens_bytes(lt))
    tRF, qRF, gRF, aRF, metaRF = extract_kinematics_ex(read_xsens_bytes(rf))
    tRT, qRT, gRT, aRT, metaRT = extract_kinematics_ex(read_xsens_bytes(rt))

    # Pelvis resampled to each femur timeline
    tP0 = tP - tP[0]
    tLF0 = tLF - tLF[0]
    tRF0 = tRF - tRF[0]
    qP_on_L = resample_quat(tP0, qP, tLF0)
    qP_on_R = resample_quat(tP0, qP, tRF0)

    # Tibias resampled to their femur timelines
    tLT0 = tLT - tLT[0]
    tRT0 = tRT - tRT[0]
    qLT_on_L = resample_quat(tLT0, qLT, tLF0)
    qRT_on_R = resample_quat(tRT0, qRT, tRF0)

    # Rotation matrices
    RP_L = quats_to_R_batch(qP_on_L)
    RF_L = quats_to_R_batch(qLF)
    RT_L = quats_to_R_batch(qLT_on_L)

    RP_R = quats_to_R_batch(qP_on_R)
    RF_R = quats_to_R_batch(qRF)
    RT_R = quats_to_R_batch(qRT_on_R)

    # Yaw drift metrics (relative and common-mode) for each side
    yawP_L = yaw_from_R(RP_L)
    yawF_L = yaw_from_R(RF_L)
    yawP_R = yaw_from_R(RP_R)
    yawF_R = yaw_from_R(RF_R)
    yawRel_L = np.unwrap(yawF_L - yawP_L)
    yawRel_R = np.unwrap(yawF_R - yawP_R)

    s_rel_L = _slope_deg_per_min(tLF, yawRel_L)
    s_pel_L = _slope_deg_per_min(tLF, yawP_L)
    s_fem_L = _slope_deg_per_min(tLF, yawF_L)

    s_rel_R = _slope_deg_per_min(tRF, yawRel_R)
    s_pel_R = _slope_deg_per_min(tRF, yawP_R)
    s_fem_R = _slope_deg_per_min(tRF, yawF_R)

    # Hip angles (deg) for both sides
    hip_L = hip_angles_xyz(RP_L, RF_L, side="L") * (180.0 / np.pi)
    hip_R = hip_angles_xyz(RP_R, RF_R, side="R") * (180.0 / np.pi)
    # Knee angles (deg) for both sides
    knee_L = knee_angles_xyz(RF_L, RT_L, side="L") * (180.0 / np.pi)
    knee_R = knee_angles_xyz(RF_R, RT_R, side="R") * (180.0 / np.pi)

    # Helper to compute slopes for angle components on a given timeline
    def comp_slopes_deg_per_min(tline, angles_deg):
        x, y, z = angles_deg[:, 0], angles_deg[:, 1], angles_deg[:, 2]
        return {
            "flex_x": round(_slope_deg_per_min(tline, np.deg2rad(x)), 3),
            "add_y": round(_slope_deg_per_min(tline, np.deg2rad(y)), 3),
            "rot_z": round(_slope_deg_per_min(tline, np.deg2rad(z)), 3),
        }

    hip_slopes = {
        "L": comp_slopes_deg_per_min(tLF, hip_L),
        "R": comp_slopes_deg_per_min(tRF, hip_R),
    }
    knee_slopes = {
        "L": comp_slopes_deg_per_min(tLF, knee_L),
        "R": comp_slopes_deg_per_min(tRF, knee_R),
    }

    # Pelvis angles vs world (Euler xyz) and slopes (use original pelvis time)
    from scipy.spatial.transform import Rotation as _Rot

    RP_world = quats_to_R_batch(qP)
    pel_eul = _Rot.from_matrix(RP_world).as_euler("xyz", degrees=True)
    pel_slopes = comp_slopes_deg_per_min(tP, pel_eul)

    # Simple gyro bias check on yaw (gz) in first 5 seconds
    def _gyro_bias(g, time):
        if g is None:
            return None
        g = np.asarray(g, float)
        if g.ndim != 2 or g.shape[1] < 3:
            return None
        t0 = float(time[0])
        idx = (time - t0) <= 5.0
        if not np.any(idx):
            idx = slice(0, min(100, len(time)))
        return float(np.nanmedian(g[idx, 2]))

    bias_pel_gz = _gyro_bias(gP, tP)
    bias_l_fem_gz = _gyro_bias(gLF, tLF)
    bias_r_fem_gz = _gyro_bias(gRF, tRF)

    # Classify probable cause per side using yaw metrics
    TH_REL = 0.5  # deg/min
    TH_CM = 0.5

    def classify(s_rel, s_pel, s_fem):
        if abs(s_rel) > TH_REL:
            return "relative_yaw_drift (sensor mismatch/gyro bias)"
        if (
            abs(s_pel) > TH_CM
            and abs(s_fem) > TH_CM
            and np.sign(s_pel) == np.sign(s_fem)
            and abs(s_rel) < TH_REL / 2
        ):
            return "common_mode_heading_drift"
        return "no_significant_drift"

    cause_L = classify(s_rel_L, s_pel_L, s_fem_L)
    cause_R = classify(s_rel_R, s_pel_R, s_fem_R)

    # Print diagnostics for CI logs
    print(
        {
            "yaw_slopes_deg_per_min": {
                "L": {
                    "relative": round(s_rel_L, 3),
                    "pelvis": round(s_pel_L, 3),
                    "femur": round(s_fem_L, 3),
                },
                "R": {
                    "relative": round(s_rel_R, 3),
                    "pelvis": round(s_pel_R, 3),
                    "femur": round(s_fem_R, 3),
                },
            },
            "joint_angle_drift_deg_per_min": {
                "hip": hip_slopes,
                "knee": knee_slopes,
                "pelvis_world": pel_slopes,
            },
            "acc_is_free": {
                "pelvis": bool(metaP.get("acc_is_free", False)),
                "L_femur": bool(metaLF.get("acc_is_free", False)),
                "L_tibia": bool(metaLT.get("acc_is_free", False)),
                "R_femur": bool(metaRF.get("acc_is_free", False)),
                "R_tibia": bool(metaRT.get("acc_is_free", False)),
            },
            "cause": {"L": cause_L, "R": cause_R},
            "gyro_bias_gz_rad_s": {
                "pelvis": None if bias_pel_gz is None else round(bias_pel_gz, 4),
                "L_femur": None if bias_l_fem_gz is None else round(bias_l_fem_gz, 4),
                "R_femur": None if bias_r_fem_gz is None else round(bias_r_fem_gz, 4),
            },
        }
    )
    out, _ = capfd.readouterr()
    # Re-emit captured output so it is visible in CI/logs
    if out:
        print(out)
    assert "joint_angle_drift_deg_per_min" in out  # printed summary available
    assert "cause" in out  # sanity
    # Accept any of the classification labels per side
    assert any(
        lbl in out
        for lbl in [
            "no_significant_drift",
            "relative_yaw_drift (sensor mismatch/gyro bias)",
            "common_mode_heading_drift",
        ]
    )
